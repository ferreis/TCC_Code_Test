#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clona/atualiza o fork YOLO-Master e instala em modo editável. "
            "Ele mantém a API ultralytics e adiciona suporte experimental a LoRA no YOLO26."
        )
    )
    parser.add_argument("--dir", default="/content/YOLO-Master", help="Diretório do clone.")
    parser.add_argument(
        "--repo",
        default="https://github.com/Tencent/YOLO-Master.git",
        help="Repositório YOLO-Master.",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="Branch/tag/commit. O commit efetivamente usado é impresso e salvo nos metadados dos experimentos.",
    )
    args = parser.parse_args()

    target = Path(args.dir).expanduser().resolve()
    if (target / ".git").exists():
        run(["git", "-C", str(target), "fetch", "--all", "--tags"])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", args.repo, str(target)])

    run(["git", "-C", str(target), "checkout", args.ref])
    if args.ref in {"main", "master"}:
        run(["git", "-C", str(target), "pull", "--ff-only"])

    run([sys.executable, "-m", "pip", "install", "-e", str(target)])
    commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    print("YOLO-Master commit:", commit)
    print("Reinicie o runtime/kernel se o Python já tinha importado outro pacote 'ultralytics'.")


if __name__ == "__main__":
    main()
