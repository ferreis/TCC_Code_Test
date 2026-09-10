#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quando este arquivo é executado como `python scripts/train_experiments.py`,
# o Python coloca `scripts/` (e não a raiz do repositório) em sys.path[0].
# Como `tcc_discovery/` é um diretório irmão de `scripts/`, o import abaixo
# falhava no Colab. Inserimos explicitamente a raiz e também executamos a CLI
# a partir dela para que caminhos relativos de configs/dados sejam estáveis.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from tcc_discovery.training_experiments import load_experiment_config, list_experiments, run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa a matriz experimental YOLO26 do TCC e consolida métricas/parâmetros."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "training_experiments.yaml"),
        help="Arquivo YAML da matriz experimental.",
    )
    parser.add_argument(
        "--group",
        choices=["experiments", "smoke_tests", "all"],
        default="experiments",
        help="Conjunto de experimentos a executar.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Executa apenas os nomes informados.",
    )
    parser.add_argument("--resume", action="store_true", help="Retoma de last.pt quando disponível.")
    parser.add_argument("--no-eval", action="store_true", help="Não roda validação final após o treino.")
    parser.add_argument(
        "--fail-on-missing-data",
        action="store_true",
        help="Falha se um dataset estiver ausente; por padrão o experimento é apenas pulado.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Mostra configuração resolvida sem treinar.")
    parser.add_argument("--list", action="store_true", help="Lista os experimentos e encerra.")
    args = parser.parse_args()

    if args.list:
        cfg = load_experiment_config(args.config)
        for name, exp in list_experiments(cfg, args.group).items():
            enabled = exp.get("enabled", True)
            print(
                f"{name:28s} enabled={str(enabled):5s} "
                f"task={exp.get('task')} regime={exp.get('regime')} model={exp.get('model')}"
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


if __name__ == "__main__":
    main()
