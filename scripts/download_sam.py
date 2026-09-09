from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
NAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def main():
    parser = argparse.ArgumentParser(description="Baixa checkpoint oficial do Segment Anything (SAM).")
    parser.add_argument("--model", choices=URLS, default="vit_b")
    parser.add_argument("--output-dir", default="weights")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / NAMES[args.model]
    if target.exists():
        print(f"Já existe: {target}")
        return

    def hook(block, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block * block_size * 100.0 / total_size)
            print(f"\rBaixando {pct:5.1f}%", end="", flush=True)

    print(f"Origem: {URLS[args.model]}")
    urllib.request.urlretrieve(URLS[args.model], target, reporthook=hook)
    print(f"\nSalvo em: {target}")


if __name__ == "__main__":
    main()
