from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, name))
        else:
            flat[name] = _jsonable(value)
    return flat


def _parameter_stats(module) -> dict[str, int | float]:
    total = sum(int(p.numel()) for p in module.parameters())
    trainable = sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
    lora = sum(
        int(p.numel())
        for name, p in module.named_parameters()
        if "lora" in name.lower() and p.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round((100.0 * trainable / total), 6) if total else 0.0,
        "lora_trainable_parameters": lora,
    }


def _model_info(yolo_model) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        result = yolo_model.info(verbose=False)
        if isinstance(result, tuple) and len(result) >= 4:
            info.update(
                {
                    "layers": int(result[0]),
                    "info_parameters": int(result[1]),
                    "info_gradients": int(result[2]),
                    "gflops": float(result[3]),
                }
            )
    except Exception:
        pass
    return info


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "git_commit": _git_commit(),
    }
    try:
        import ultralytics

        env["ultralytics_version"] = getattr(ultralytics, "__version__", None)
        env["ultralytics_path"] = getattr(ultralytics, "__file__", None)
    except Exception:
        env["ultralytics_version"] = None
        env["ultralytics_path"] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu": props.name,
                "gpu_total_vram_gb": round(props.total_memory / 1024**3, 3),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return env


def _has_yolo_master_lora() -> bool:
    try:
        return importlib.util.find_spec("ultralytics.utils.lora") is not None
    except Exception:
        return False


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuração de experimentos inválida.")
    return cfg


def list_experiments(config: dict[str, Any], group: str = "experiments") -> dict[str, dict[str, Any]]:
    if group == "all":
        merged: dict[str, dict[str, Any]] = {}
        merged.update(config.get("smoke_tests", {}) or {})
        merged.update(config.get("experiments", {}) or {})
        return merged
    return dict(config.get(group, {}) or {})


def _resolve_data_path(data: str | Path) -> Path:
    return Path(data).expanduser().resolve()


def _metrics_dict(metrics) -> dict[str, Any]:
    if metrics is None:
        return {}
    raw = getattr(metrics, "results_dict", None)
    if isinstance(raw, dict):
        return {str(k): _jsonable(v) for k, v in raw.items()}
    return {}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(summary_path: Path, row: dict[str, Any]) -> None:
    json_path = summary_path.with_suffix(".json")
    existing: dict[str, dict[str, Any]] = {}
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = {str(x.get("experiment")): x for x in loaded if isinstance(x, dict)}
        except Exception:
            pass
    existing[str(row["experiment"])] = _jsonable(row)
    rows = list(existing.values())
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = [_flatten_dict(r) for r in rows]
    fields = sorted({k for r in flat_rows for k in r})
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def run_experiment(
    experiment_name: str,
    experiment: dict[str, Any],
    global_cfg: dict[str, Any],
    *,
    resume: bool = False,
    evaluate: bool = True,
    fail_on_missing_data: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO

    merged_train = dict(global_cfg)
    project = Path(str(merged_train.pop("project", "workspace/training_runs")))
    eval_split = str(merged_train.pop("eval_split", "test"))

    merged_train.update(experiment.get("train_args", {}) or {})
    model_name = str(experiment["model"])
    data_path = _resolve_data_path(experiment["data"])
    run_name = str(experiment.get("name", experiment_name))
    regime = str(experiment.get("regime", "unknown"))
    task = str(experiment.get("task", "detect"))

    project = project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    run_dir = project / run_name
    metadata_dir = run_dir / "tcc_metadata"
    summary_csv = project / "experiment_summary.csv"

    base_row: dict[str, Any] = {
        "experiment": experiment_name,
        "run_name": run_name,
        "task": task,
        "regime": regime,
        "model": model_name,
        "data": str(data_path),
        "status": "planned",
        "environment": _environment(),
        "train_args": merged_train,
    }

    if not data_path.exists():
        base_row["status"] = "missing_data"
        base_row["error"] = f"Dataset não encontrado: {data_path}"
        _save_json(metadata_dir / "experiment.json", base_row)
        _write_summary(summary_csv, base_row)
        if fail_on_missing_data:
            raise FileNotFoundError(data_path)
        print(f"[SKIP] {experiment_name}: dataset não encontrado: {data_path}")
        return base_row

    if regime == "lora" and not _has_yolo_master_lora():
        message = (
            "Este experimento LoRA requer o fork YOLO-Master, que adiciona LoRA ao YOLO26. "
            "Instale-o antes de executar os experimentos LoRA."
        )
        base_row["status"] = "missing_lora_backend"
        base_row["error"] = message
        _save_json(metadata_dir / "experiment.json", base_row)
        _write_summary(summary_csv, base_row)
        raise RuntimeError(message)

    train_kwargs = dict(merged_train)
    train_kwargs.update(
        {
            "data": str(data_path),
            "project": str(project),
            "name": run_name,
            "exist_ok": True,
        }
    )

    if dry_run:
        base_row["status"] = "dry_run"
        base_row["resolved_train_args"] = train_kwargs
        _save_json(metadata_dir / "experiment.json", base_row)
        _write_summary(summary_csv, base_row)
        print(json.dumps(base_row, ensure_ascii=False, indent=2))
        return base_row

    last_pt = run_dir / "weights" / "last.pt"
    if resume and last_pt.exists():
        yolo = YOLO(str(last_pt))
        train_kwargs = {"resume": True}
    else:
        yolo = YOLO(model_name)

    initial_stats = _parameter_stats(yolo.model)
    base_row["initial_parameters"] = initial_stats
    base_row["model_info"] = _model_info(yolo)
    base_row["started_at_utc"] = _utc_now()
    base_row["status"] = "running"
    _save_json(metadata_dir / "experiment.json", base_row)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    try:
        train_result = yolo.train(**train_kwargs)
        elapsed = time.perf_counter() - started
        trainer_model = getattr(getattr(yolo, "trainer", None), "model", None) or yolo.model
        final_stats = _parameter_stats(trainer_model)

        save_dir = Path(str(getattr(train_result, "save_dir", run_dir)))
        if not save_dir.exists() and getattr(yolo, "trainer", None) is not None:
            save_dir = Path(str(yolo.trainer.save_dir))
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        row = dict(base_row)
        row.update(
            {
                "status": "completed",
                "ended_at_utc": _utc_now(),
                "train_seconds": round(elapsed, 3),
                "save_dir": str(save_dir),
                "best_checkpoint": str(best_pt) if best_pt.exists() else None,
                "last_checkpoint": str(last_pt) if last_pt.exists() else None,
                "best_checkpoint_mb": round(best_pt.stat().st_size / 1024**2, 3) if best_pt.exists() else None,
                "final_parameters": final_stats,
                "train_metrics": _metrics_dict(train_result),
            }
        )
        if torch.cuda.is_available():
            row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            row["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

        if regime == "lora":
            adapter_dir = save_dir / "lora_adapter"
            try:
                if hasattr(yolo, "save_lora_only"):
                    yolo.save_lora_only(str(adapter_dir))
                    row["lora_adapter"] = str(adapter_dir)
            except Exception as exc:
                row["lora_adapter_save_error"] = str(exc)

        if evaluate and best_pt.exists():
            eval_started = time.perf_counter()
            evaluator = YOLO(str(best_pt))
            try:
                metrics = evaluator.val(
                    data=str(data_path),
                    split=eval_split,
                    imgsz=int(merged_train.get("imgsz", 640)),
                    batch=merged_train.get("batch", 16),
                    device=merged_train.get("device", 0),
                    workers=int(merged_train.get("workers", 2)),
                    plots=True,
                )
                row["eval_split"] = eval_split
            except Exception as exc:
                if eval_split != "val":
                    metrics = evaluator.val(
                        data=str(data_path),
                        split="val",
                        imgsz=int(merged_train.get("imgsz", 640)),
                        batch=merged_train.get("batch", 16),
                        device=merged_train.get("device", 0),
                        workers=int(merged_train.get("workers", 2)),
                        plots=True,
                    )
                    row["eval_split"] = "val"
                    row["eval_fallback_reason"] = str(exc)
                else:
                    raise
            row["eval_seconds"] = round(time.perf_counter() - eval_started, 3)
            row["eval_metrics"] = _metrics_dict(metrics)
            speed = getattr(metrics, "speed", None)
            if isinstance(speed, dict):
                row["eval_speed_ms"] = {str(k): _jsonable(v) for k, v in speed.items()}

        _save_json(metadata_dir / "experiment.json", row)
        _write_summary(summary_csv, row)
        return row
    except Exception as exc:
        row = dict(base_row)
        row.update(
            {
                "status": "failed",
                "ended_at_utc": _utc_now(),
                "train_seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        if torch.cuda.is_available():
            row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        _save_json(metadata_dir / "experiment.json", row)
        _write_summary(summary_csv, row)
        raise


def run_matrix(
    config_path: str | Path = "configs/training_experiments.yaml",
    *,
    group: str = "experiments",
    only: list[str] | None = None,
    resume: bool = False,
    evaluate: bool = True,
    fail_on_missing_data: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    config = load_experiment_config(config_path)
    global_cfg = dict(config.get("global", {}) or {})
    experiments = list_experiments(config, group)
    if only:
        wanted = set(only)
        experiments = {k: v for k, v in experiments.items() if k in wanted}
        missing = wanted.difference(experiments)
        if missing:
            raise KeyError(f"Experimentos não encontrados: {sorted(missing)}")

    results: list[dict[str, Any]] = []
    for name, experiment in experiments.items():
        if not bool(experiment.get("enabled", True)):
            continue
        print(f"\n{'=' * 80}\nEXPERIMENTO: {name}\n{'=' * 80}")
        result = run_experiment(
            name,
            experiment,
            global_cfg,
            resume=resume,
            evaluate=evaluate,
            fail_on_missing_data=fail_on_missing_data,
            dry_run=dry_run,
        )
        results.append(result)
    return results
