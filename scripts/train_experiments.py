#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from tcc_discovery.runtime_config import load_runtime_config, materialize_training_plan
from tcc_discovery.training_experiments import list_experiments, run_matrix

GROUP_CHOICES = [
    "general_models",
    "lora_specialists",
    "full_finetune_controls",
    "smoke_tests",
    "experiments",
    "all",
]


def _refresh_analysis_bundle(config_path: str, output_dir: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_analysis_bundle.py"),
        "--config",
        config_path,
        "--runs-root",
        str(REPO_ROOT / "workspace" / "training_runs"),
        "--output",
        str((REPO_ROOT / output_dir).resolve()) if not Path(output_dir).is_absolute() else output_dir,
    ]
    print("\nAtualizando pacote para TCC_Analise...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("[AVISO] Treino concluído, mas a exportação para análise falhou.")
        return
    print(f"Pacote atualizado em: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina Modelo Geral, LoRAs especialistas e controles Full Fine-Tune usando a configuração compartilhada."
    )
    parser.add_argument("--runtime-config", default="configs/runtime.yaml")
    parser.add_argument("--profile", default=None, help="Perfil de hardware: local ou colab.")
    parser.add_argument(
        "--config",
        default=None,
        help="Plano base de experimentos. Normalmente vem de paths.training_plan em configs/runtime.yaml.",
    )
    parser.add_argument("--group", choices=GROUP_CHOICES, default="general_models")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--fail-on-missing-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    runtime_plan_path = REPO_ROOT / "workspace" / "training_plan_runtime.yaml"

    # Primeira materialização sem copiar datasets: resolve modelos, caminhos e parâmetros.
    runtime_plan_path, plan_cfg = materialize_training_plan(
        runtime,
        profile=args.profile,
        base_plan_path=args.config,
        output_path=runtime_plan_path,
        selected_stages=args.only,
        stage_datasets=False,
    )

    if args.list:
        for name, exp in list_experiments(plan_cfg, args.group).items():
            source = exp.get("base_from") or exp.get("model")
            print(
                f"{name:20s} enabled={str(exp.get('enabled', True)):5s} "
                f"task={str(exp.get('task')):7s} regime={str(exp.get('regime')):28s} "
                f"source={source} | {exp.get('label', '')}"
            )
        return

    if args.only:
        selected = list(args.only)
    else:
        selected = list(list_experiments(plan_cfg, args.group).keys())

    # Na execução real, o perfil pode copiar apenas os datasets selecionados para SSD local.
    runtime_plan_path, _ = materialize_training_plan(
        runtime,
        profile=args.profile,
        base_plan_path=args.config,
        output_path=runtime_plan_path,
        selected_stages=selected,
        stage_datasets=not args.dry_run,
    )

    run_matrix(
        runtime_plan_path,
        group=args.group,
        only=args.only,
        resume=args.resume,
        evaluate=not args.no_eval,
        fail_on_missing_data=args.fail_on_missing_data,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        output_dir = str((runtime.get("paths", {}) or {}).get("analysis_export", "exports/TCC_Analise_models"))
        _refresh_analysis_bundle(str(runtime_plan_path), output_dir)


if __name__ == "__main__":
    main()
