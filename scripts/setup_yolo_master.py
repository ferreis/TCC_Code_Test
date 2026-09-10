#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, check: bool = True) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clona/atualiza o Tencent/YOLO-Master e instala o pacote ultralytics "
            "de forma não-editável por padrão. Isso evita depender de arquivos .pth "
            "que podem exigir reinício do kernel do Colab."
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
        help="Branch/tag/commit. O commit efetivamente usado é impresso.",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Instala com -e. Não recomendado no Colab; pode exigir reinício do kernel.",
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

    # Remove qualquer distribuição oficial/antiga antes de instalar o fork.
    # O módulo ainda não deve ter sido importado no notebook nesta altura.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "ultralytics"], check=False)

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--force-reinstall",
        "--no-deps",
    ]
    if args.editable:
        install_cmd.append("-e")
    install_cmd.append(str(target))
    run(install_cmd)

    commit = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    print("YOLO-Master commit:", commit, flush=True)

    # Verificação em um processo Python novo usando exatamente o mesmo
    # interpretador do notebook. Se isto falhar, a célula para imediatamente.
    verify_code = r'''
import importlib.util
import pathlib
import ultralytics

spec = importlib.util.find_spec("ultralytics.utils.lora")
print("Ultralytics/YOLO-Master:", ultralytics.__version__)
print("Caminho:", pathlib.Path(ultralytics.__file__).resolve())
print("LoRA disponível:", spec is not None)
if spec is None:
    raise SystemExit("ERRO: ultralytics.utils.lora não encontrado no backend instalado")
'''
    run([sys.executable, "-c", verify_code])
    print("YOLO-Master instalado e validado sem necessidade de reiniciar o kernel.", flush=True)


if __name__ == "__main__":
    main()
