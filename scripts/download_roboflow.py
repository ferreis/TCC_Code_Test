from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Baixa um dataset Roboflow para teste local.")
    parser.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE"))
    parser.add_argument("--project", default=os.getenv("ROBOFLOW_PROJECT"))
    parser.add_argument("--version", type=int, default=int(os.getenv("ROBOFLOW_VERSION", "1")))
    parser.add_argument(
        "--format",
        default="yolo26",
        help="Formato de exportação do Roboflow. Padrão: yolo26",
    )
    parser.add_argument("--output", default="data/roboflow")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Apaga o diretório de saída existente e baixa novamente.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Defina ROBOFLOW_API_KEY no arquivo .env")
    if not args.workspace or not args.project:
        raise SystemExit("Informe --workspace e --project ou configure o .env")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # O SDK do Roboflow retorna imediatamente quando `location` já existe e
    # overwrite=False. Portanto, não devemos criar a pasta final antes do download.
    if output.exists():
        image_count = count_images(output)

        if args.overwrite:
            shutil.rmtree(output)
        elif image_count > 0:
            print(f"Dataset já existe em: {output}")
            print(f"Imagens encontradas: {image_count}")
            print("Use --overwrite para baixar novamente.")
            return
        else:
            print(f"A pasta {output} existe, mas não contém imagens. Removendo para refazer o download...")
            shutil.rmtree(output)

    print("loading Roboflow workspace...")
    rf = Roboflow(api_key=api_key)
    print("loading Roboflow project...")
    project = rf.workspace(args.workspace).project(args.project)

    dataset = project.version(args.version).download(
        args.format,
        location=str(output),
        overwrite=args.overwrite,
    )

    image_count = count_images(output)
    if image_count == 0:
        raise SystemExit(
            f"O Roboflow retornou o dataset em {dataset.location}, mas nenhuma imagem foi encontrada. "
            "Tente novamente com --overwrite."
        )

    print(f"Dataset disponível em: {dataset.location}")
    print(f"Imagens encontradas: {image_count}")

    data_yaml = output / "data.yaml"
    if data_yaml.exists():
        print(f"Configuração YOLO: {data_yaml}")


if __name__ == "__main__":
    main()
