#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_file(src: Path, dst: Path, copied: list[dict[str, str]]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"source": str(src), "destination": str(dst)})


def _copy_dir(src: Path, dst: Path, copied: list[dict[str, str]]) -> None:
    if not src.exists() or not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    copied.append({"source": str(src), "destination": str(dst)})


def _task_dir(task: str) -> str:
    return "segmentation" if task == "segment" else "detection"


def build_bundle(
    *,
    config_path: Path,
    runs_root: Path,
    output_dir: Path,
    zip_output: bool = True,
) -> tuple[Path, Path | None, dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for task_dir in ("detection", "segmentation"):
        (output_dir / task_dir).mkdir(parents=True, exist_ok=True)

    def run_dir(item: dict[str, Any], key: str) -> Path:
        return runs_root / str(item.get("name", key))

    for key, item in (cfg.get("general_models", {}) or {}).items():
        task_dir = _task_dir(str(item.get("task", "detect")))
        src_run = run_dir(item, key)
        src = src_run / "weights" / "best.pt"
        dst = output_dir / task_dir / "general" / "best.pt"
        if src.exists():
            _copy_file(src, dst, copied)
            _copy_file(src_run / "tcc_metadata" / "experiment.json", dst.parent / "experiment.json", copied)
        else:
            missing.append({"id": key, "expected": str(src)})

    for key, item in (cfg.get("lora_specialists", {}) or {}).items():
        task_dir = _task_dir(str(item.get("task", "detect")))
        src_run = run_dir(item, key)
        src = src_run / "lora_adapter"
        dst = output_dir / task_dir / key.replace("det_", "").replace("seg_", "") / "adapter"
        if src.exists():
            _copy_dir(src, dst, copied)
            _copy_file(src_run / "tcc_metadata" / "experiment.json", dst.parent / "experiment.json", copied)
        else:
            missing.append({"id": key, "expected": str(src)})

    for key, item in (cfg.get("full_finetune_controls", {}) or {}).items():
        task_dir = _task_dir(str(item.get("task", "detect")))
        src_run = run_dir(item, key)
        src = src_run / "weights" / "best.pt"
        dst = output_dir / task_dir / key.replace("det_", "").replace("seg_", "") / "best.pt"
        if src.exists():
            _copy_file(src, dst, copied)
            _copy_file(src_run / "tcc_metadata" / "experiment.json", dst.parent / "experiment.json", copied)
        else:
            missing.append({"id": key, "expected": str(src)})

    metadata_dir = output_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("experiment_summary.csv", "experiment_summary.json"):
        _copy_file(runs_root / filename, metadata_dir / filename, copied)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Conteudo pronto para ser extraido/copiado diretamente para TCC_Analise/models/",
        "config": str(config_path),
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "copied": copied,
        "missing": missing,
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    zip_path: Path | None = None
    if zip_output:
        archive_base = output_dir.parent / output_dir.name
        archive = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
        zip_path = Path(archive)

    return output_dir, zip_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monta a pasta/ZIP de artefatos pronta para TCC_Analise/models/."
    )
    parser.add_argument("--config", default="configs/training_experiments.yaml")
    parser.add_argument("--runs-root", default="workspace/training_runs")
    parser.add_argument("--output", default="exports/TCC_Analise_models")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    output_dir, zip_path, manifest = build_bundle(
        config_path=resolve(args.config),
        runs_root=resolve(args.runs_root),
        output_dir=resolve(args.output),
        zip_output=not args.no_zip,
    )

    print(f"Pasta pronta: {output_dir}")
    if zip_path:
        print(f"ZIP pronto: {zip_path}")
    print(f"Arquivos/diretorios copiados: {len(manifest['copied'])}")
    print(f"Etapas ainda ausentes: {len(manifest['missing'])}")
    for item in manifest["missing"]:
        print(f"  [AUSENTE] {item['id']}: {item['expected']}")


if __name__ == "__main__":
    main()
