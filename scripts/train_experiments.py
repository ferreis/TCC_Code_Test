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

from tcc_discovery.training_experiments import load_experiment_config, list_experiments, run_matrix

GROUP_CHOICES = [
    "general_models",
    "lora_specialists",
    "full_finetune_controls",
    "smoke_tests",
    "experiments",  # alias: todas as etapas de treino, sem smoke tests
    "all",
]


def _refresh_analysis_bundle(config_path: str) -> None:
    """Atualiza pasta e ZIP prontos para TCC_Analise/models/."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_analysis_bundle.py"),
        "--config",
        config_path,
        "--runs-root",
        str(REPO_ROOT / "workspace" / "training_runs"),
        "--output",
        str(REPO_ROOT / "exports" / "TCC_Analise_models"),
    ]
    print("\nAtualizando pacote para TCC_Analise...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "[AVISO] O treinamento terminou, mas o pacote de exportação não pôde ser atualizado. "
            "Execute scripts/export_analysis_bundle.py manualmente para diagnosticar."
        )
        return
    print("Pacote atualizado:")
    print("  pasta: exports/TCC_Analise_models/")
    print("  zip:   exports/TCC_Analise_models.zip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Executa etapas separadas do TCC: modelo geral, adapters LoRA especialistas "
            "e controles Full Fine-Tune."
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "training_experiments.yaml"),
        help="Arquivo YAML do plano de treinamento.",
    )
    parser.add_argument(
        "--group",
        choices=GROUP_CHOICES,
        default="general_models",
        help="Etapa a executar. LoRAs não são executados automaticamente junto com o modelo geral.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Executa apenas os IDs informados, mesmo que pertençam a grupos diferentes.",
    )
    parser.add_argument("--resume", action="store_true", help="Retoma de last.pt quando disponível.")
    parser.add_argument("--no-eval", action="store_true", help="Não roda validação final após o treino.")
    parser.add_argument(
        "--fail-on-missing-data",
        action="store_true",
        help="Falha se um dataset estiver ausente; por padrão a etapa é pulada.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Mostra configuração resolvida sem treinar.")
    parser.add_argument("--list", action="store_true", help="Lista as etapas e encerra.")
    args = parser.parse_args()

    if args.list:
        cfg = load_experiment_config(args.config)
        for name, exp in list_experiments(cfg, args.group).items():
            enabled = exp.get("enabled", True)
            source = exp.get("base_from") or exp.get("model")
            print(
                f"{name:20s} enabled={str(enabled):5s} "
                f"task={exp.get('task'):7s} regime={exp.get('regime'):28s} "
                f"source={source} | {exp.get('label', '')}"
            )
        return

    run_matrix(
        args.config,
        group=args.group,
        only=args.only,
        resume=args.resume,
        evaluate=not args.no_eval,
        fail_on_missing_data=args.fail_on_missing_data,
        dry_run=args.dry_run,
    )

    # Qualquer etapa real executada pelo notebook atualiza automaticamente o
    # bundle persistido em exports/, que no Colab está ligado ao Google Drive.
    if not args.dry_run:
        _refresh_analysis_bundle(args.config)


if __name__ == "__main__":
    main()
