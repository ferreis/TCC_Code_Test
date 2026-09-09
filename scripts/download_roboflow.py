from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Baixa um dataset Roboflow para teste local.")
    parser.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE"))
    parser.add_argument("--project", default=os.getenv("ROBOFLOW_PROJECT"))
    parser.add_argument("--version", type=int, default=int(os.getenv("ROBOFLOW_VERSION", "1")))
    parser.add_argument("--format", default="yolov8", help="Ex.: yolov8, coco, coco-segmentation")
    parser.add_argument("--output", default="data/roboflow")
    args = parser.parse_args()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Defina ROBOFLOW_API_KEY no arquivo .env")
    if not args.workspace or not args.project:
        raise SystemExit("Informe --workspace e --project ou configure o .env")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=str(output), overwrite=False)
    print(f"Dataset disponível em: {dataset.location}")


if __name__ == "__main__":
    main()
