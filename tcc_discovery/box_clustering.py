from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from .core import (
    DiscoveryConfig,
    _cluster_embeddings,
    _embed_crops,
    load_rgb,
    save_project,
    save_rgb,
)

ProgressCallback = Callable[[int, int, str], None]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class BoxClusteringConfig:
    dino_model: str = "facebook/dinov2-small"
    embedding_batch_size: int = 4
    min_cluster_size: int = 6
    min_samples: int = 3
    pca_components: int = 50
    context_scale: float = 1.8
    min_crop_size_px: int = 64
    dino_cpu_fallback_on_oom: bool = True


def _resolve_split_dirs(root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(root).expanduser().resolve()
    if root.name == "images":
        split_dir = root.parent
        images_dir = root
        labels_dir = split_dir / "labels"
    else:
        split_dir = root
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Pasta de labels não encontrada: {labels_dir}")
    return split_dir, images_dir, labels_dir


def _load_class_names(split_dir: Path) -> dict[int, str]:
    candidates = [split_dir / "data.yaml", split_dir.parent / "data.yaml"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = data.get("names", {})
            if isinstance(names, list):
                return {i: str(name) for i, name in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
        except Exception:
            pass
    return {}


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        p = images_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    # Roboflow normally preserves matching stems, but handle unusual case/extension.
    matches = [p for p in images_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return matches[0] if matches else None


def _parse_yolo_detection(line: str, width: int, height: int) -> tuple[int, list[int]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
    except ValueError:
        return None
    if bw <= 0 or bh <= 0:
        return None
    x1 = int(round((cx - bw / 2.0) * width))
    y1 = int(round((cy - bh / 2.0) * height))
    x2 = int(round((cx + bw / 2.0) * width))
    y2 = int(round((cy + bh / 2.0) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return class_id, [x1, y1, x2, y2]


def _context_box(box: list[int], width: int, height: int, scale: float, min_size: int) -> list[int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    crop_w = max(float(min_size), bw * float(scale))
    crop_h = max(float(min_size), bh * float(scale))
    nx1 = int(round(cx - crop_w / 2.0))
    ny1 = int(round(cy - crop_h / 2.0))
    nx2 = int(round(cx + crop_w / 2.0))
    ny2 = int(round(cy + crop_h / 2.0))
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > width:
        nx1 -= nx2 - width
        nx2 = width
    if ny2 > height:
        ny1 -= ny2 - height
        ny2 = height
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    return [nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2)]


def run_box_clustering(
    dataset_split_dir: str | Path,
    project_dir: str | Path,
    config: BoxClusteringConfig,
    overwrite: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    """Cria clusters usando exclusivamente as bounding boxes YOLO já anotadas.

    Nenhum tile é usado para formar os objetos: cada crop é centrado na bounding
    box original e inclui uma margem de contexto. Isso evita cortar um cetáceo em
    partes só por ele atravessar a borda de um tile.
    """
    split_dir, images_dir, labels_dir = _resolve_split_dirs(dataset_split_dir)
    project_dir = Path(project_dir).expanduser().resolve()
    if (project_dir / "catalog.json").exists() and not overwrite:
        raise FileExistsError("O projeto já existe. Marque 'Recriar projeto' para substituí-lo.")
    if overwrite and project_dir.exists():
        shutil.rmtree(project_dir)
    (project_dir / "crops").mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(split_dir)
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise ValueError(f"Nenhum arquivo YOLO encontrado em {labels_dir}")

    catalog = {
        "version": 3,
        "project_type": "bbox_clustering",
        "input_dir": str(split_dir),
        "config": config.__dict__.copy(),
        "images": [],
        "objects": [],
    }
    crop_paths: list[Path] = []
    total = len(label_files)

    for idx, label_path in enumerate(label_files, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f"Lendo boxes de {label_path.name}")
        image_path = _find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        rgb = load_rgb(image_path)
        h, w = rgb.shape[:2]
        image_id = f"img_{idx:06d}"
        catalog["images"].append(
            {
                "id": image_id,
                "path": str(image_path),
                "width": w,
                "height": h,
                "name": image_path.name,
                "label_path": str(label_path),
            }
        )
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for ann_idx, line in enumerate(lines, start=1):
            parsed = _parse_yolo_detection(line, w, h)
            if parsed is None:
                continue
            class_id, box = parsed
            crop_box = _context_box(box, w, h, config.context_scale, config.min_crop_size_px)
            x1, y1, x2, y2 = crop_box
            crop = rgb[y1:y2, x1:x2]
            obj_id = f"{image_id}_box_{ann_idx:04d}_{uuid.uuid4().hex[:6]}"
            crop_rel = Path("crops") / f"{obj_id}.jpg"
            save_rgb(project_dir / crop_rel, crop)
            crop_paths.append(project_dir / crop_rel)
            catalog["objects"].append(
                {
                    "id": obj_id,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "image_width": w,
                    "image_height": h,
                    "bbox_xyxy": box,
                    "crop_bbox_xyxy": crop_box,
                    "mask_bbox_xyxy": None,
                    "mask_path": None,
                    "crop_path": crop_rel.as_posix(),
                    "cluster": -1,
                    "label": None,
                    "ignored": False,
                    "source": "roboflow_yolo_box",
                    "source_class_id": class_id,
                    "source_class_name": class_names.get(class_id, str(class_id)),
                    "source_label_path": str(label_path),
                    "source_annotation_index": ann_idx,
                }
            )
        if progress_cb:
            progress_cb(idx, total, f"Boxes carregadas de {image_path.name}")

    save_project(project_dir, catalog)
    if not crop_paths:
        return catalog

    dcfg = DiscoveryConfig(
        dino_model=config.dino_model,
        embedding_batch_size=config.embedding_batch_size,
        dino_cpu_fallback_on_oom=config.dino_cpu_fallback_on_oom,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        pca_components=config.pca_components,
    )
    embeddings = _embed_crops(
        crop_paths,
        config.dino_model,
        config.embedding_batch_size,
        progress_cb,
        cpu_fallback=config.dino_cpu_fallback_on_oom,
    )
    np.save(project_dir / "embeddings.npy", embeddings)
    labels = _cluster_embeddings(embeddings, dcfg)
    for obj, cluster in zip(catalog["objects"], labels.tolist()):
        obj["cluster"] = int(cluster)
    save_project(project_dir, catalog)
    return catalog


def render_box_overlay(obj: dict) -> np.ndarray:
    image = load_rgb(obj["image_path"]).copy()
    x1, y1, x2, y2 = [int(v) for v in obj["bbox_xyxy"]]
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return image


def export_bbox_manifest(project_dir: str | Path, output_path: str | Path) -> dict:
    """Exporta uma tabela JSON simples para auditoria/reprodutibilidade."""
    project_dir = Path(project_dir)
    catalog = json.loads((project_dir / "catalog.json").read_text(encoding="utf-8"))
    rows = []
    for obj in catalog.get("objects", []):
        rows.append(
            {
                "id": obj.get("id"),
                "image": obj.get("image_path"),
                "bbox_xyxy": obj.get("bbox_xyxy"),
                "cluster": obj.get("cluster"),
                "label": obj.get("label"),
                "ignored": obj.get("ignored", False),
                "source_class_id": obj.get("source_class_id"),
                "source_class_name": obj.get("source_class_name"),
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": len(rows), "output": str(output_path.resolve())}
