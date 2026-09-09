from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ProgressCallback = Callable[[int, int, str], None]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DiscoveryConfig:
    sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    dino_model: str = "facebook/dinov2-small"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 1
    min_mask_region_area: int = 80
    min_area_px: int = 120
    max_area_fraction: float = 0.70
    max_objects_per_image: int = 40
    nms_iou: float = 0.82
    crop_padding: float = 0.08
    tile_size: int = 0
    tile_overlap: float = 0.15
    embedding_batch_size: int = 16
    min_cluster_size: int = 6
    min_samples: int = 3
    pca_components: int = 50


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def save_rgb(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path, quality=quality)


def _sanitize_stem(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return value[:80] or "image"


def _box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


def _nms(candidates: list[dict], iou_threshold: float, limit: int) -> list[dict]:
    candidates = sorted(
        candidates,
        key=lambda x: (float(x.get("stability_score", 0.0)) + float(x.get("predicted_iou", 0.0))),
        reverse=True,
    )
    kept: list[dict] = []
    for cand in candidates:
        if all(_box_iou(cand["bbox_xyxy"], x["bbox_xyxy"]) < iou_threshold for x in kept):
            kept.append(cand)
            if len(kept) >= limit:
                break
    return kept


def _iter_tiles(width: int, height: int, tile_size: int, overlap: float):
    if tile_size <= 0 or (width <= tile_size and height <= tile_size):
        yield 0, 0, width, height
        return
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            yield x, y, min(width, x + tile_size), min(height, y + tile_size)


def _crop_box(box: list[int], width: int, height: int, padding: float) -> list[int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    return [max(0, x1 - px), max(0, y1 - py), min(width, x2 + px), min(height, y2 + py)]


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _save_mask_crop(mask_crop: np.ndarray, path: Path) -> None:
    crop = (mask_crop > 0).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def load_full_mask(project_dir: str | Path, obj: dict) -> np.ndarray:
    project_dir = Path(project_dir)
    h, w = int(obj["image_height"]), int(obj["image_width"])
    canvas = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in obj["mask_bbox_xyxy"]]
    mask_crop = cv2.imread(str(project_dir / obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask_crop is None:
        raise FileNotFoundError(project_dir / obj["mask_path"])
    target_h, target_w = y2 - y1, x2 - x1
    if mask_crop.shape != (target_h, target_w):
        mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[y1:y2, x1:x2] = (mask_crop > 127).astype(np.uint8)
    return canvas


def save_project(project_dir: str | Path, catalog: dict) -> None:
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    tmp = project_dir / "catalog.json.tmp"
    tmp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(project_dir / "catalog.json")


def load_project(project_dir: str | Path) -> dict:
    path = Path(project_dir) / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_sam(config: DiscoveryConfig):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    checkpoint = Path(config.sam_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint SAM não encontrado em {checkpoint}. Execute: python scripts/download_sam.py"
        )
    sam = sam_model_registry[config.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=_device())
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        crop_n_layers=config.crop_n_layers,
        min_mask_region_area=config.min_mask_region_area,
    )


def _discover_image(image: np.ndarray, mask_generator, config: DiscoveryConfig) -> list[dict]:
    h, w = image.shape[:2]
    image_area = h * w
    candidates: list[dict] = []
    for tx1, ty1, tx2, ty2 in _iter_tiles(w, h, config.tile_size, config.tile_overlap):
        tile = image[ty1:ty2, tx1:tx2]
        masks = mask_generator.generate(tile)
        for proposal in masks:
            local_mask = proposal["segmentation"].astype(np.uint8)
            local_box = _mask_bbox(local_mask)
            if local_box is None:
                continue
            lx1, ly1, lx2, ly2 = local_box
            box = [lx1 + tx1, ly1 + ty1, lx2 + tx1, ly2 + ty1]
            area = int(proposal.get("area", int(local_mask.sum())))
            if area < config.min_area_px or area > image_area * config.max_area_fraction:
                continue
            candidates.append(
                {
                    "bbox_xyxy": box,
                    "mask_bbox_xyxy": box,
                    "mask_crop": local_mask[ly1:ly2, lx1:lx2].copy(),
                    "area": area,
                    "predicted_iou": float(proposal.get("predicted_iou", 0.0)),
                    "stability_score": float(proposal.get("stability_score", 0.0)),
                }
            )
    return _nms(candidates, config.nms_iou, config.max_objects_per_image)


def _embed_crops(crop_paths: list[Path], model_name: str, batch_size: int, progress_cb=None) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    device = _device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    total = len(crop_paths)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_paths = crop_paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs).last_hidden_state[:, 0, :]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            vectors.append(out.detach().cpu().numpy().astype(np.float32))
            for im in images:
                im.close()
            if progress_cb:
                progress_cb(min(start + len(batch_paths), total), total, "Extraindo embeddings DINOv2")
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(vectors, axis=0)


def _cluster_embeddings(embeddings: np.ndarray, config: DiscoveryConfig) -> np.ndarray:
    if len(embeddings) < max(2, config.min_cluster_size):
        return np.full(len(embeddings), -1, dtype=int)
    data = embeddings
    max_components = min(config.pca_components, data.shape[1], max(2, len(data) - 1))
    if max_components >= 2 and data.shape[1] > max_components:
        from sklearn.decomposition import PCA

        data = PCA(n_components=max_components, random_state=42).fit_transform(data)
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=max(2, config.min_cluster_size),
        min_samples=max(1, config.min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    return model.fit_predict(data).astype(int)


def run_discovery(
    input_dir: str | Path,
    project_dir: str | Path,
    config: DiscoveryConfig,
    overwrite: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    input_dir = Path(input_dir).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()
    images = list_images(input_dir)
    if not images:
        raise ValueError(f"Nenhuma imagem encontrada em {input_dir}")
    if (project_dir / "catalog.json").exists() and not overwrite:
        raise FileExistsError("O projeto já existe. Marque overwrite para recriá-lo.")
    if overwrite and project_dir.exists():
        shutil.rmtree(project_dir)
    (project_dir / "masks").mkdir(parents=True, exist_ok=True)
    (project_dir / "crops").mkdir(parents=True, exist_ok=True)

    mask_generator = _build_sam(config)
    catalog = {
        "version": 1,
        "input_dir": str(input_dir),
        "config": config.__dict__.copy(),
        "images": [],
        "objects": [],
    }
    crop_paths: list[Path] = []
    total = len(images)

    for idx, image_path in enumerate(images, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f"Segmentando {image_path.name}")
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
            }
        )
        proposals = _discover_image(rgb, mask_generator, config)
        for p_idx, proposal in enumerate(proposals, start=1):
            obj_id = f"{image_id}_obj_{p_idx:04d}_{uuid.uuid4().hex[:6]}"
            box = [int(v) for v in proposal["bbox_xyxy"]]
            mask_bbox = [int(v) for v in proposal["mask_bbox_xyxy"]]
            mask_rel = Path("masks") / f"{obj_id}.png"
            _save_mask_crop(proposal["mask_crop"], project_dir / mask_rel)
            crop_box = _crop_box(box, w, h, config.crop_padding)
            x1, y1, x2, y2 = crop_box
            crop = rgb[y1:y2, x1:x2]
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
                    "mask_bbox_xyxy": mask_bbox,
                    "mask_path": mask_rel.as_posix(),
                    "crop_path": crop_rel.as_posix(),
                    "area": int(proposal["area"]),
                    "predicted_iou": float(proposal["predicted_iou"]),
                    "stability_score": float(proposal["stability_score"]),
                    "cluster": -1,
                    "label": None,
                    "ignored": False,
                    "source": "sam_auto",
                }
            )
        if progress_cb:
            progress_cb(idx, total, f"Segmentado {image_path.name}")

    save_project(project_dir, catalog)
    if not crop_paths:
        return catalog

    embeddings = _embed_crops(crop_paths, config.dino_model, config.embedding_batch_size, progress_cb)
    np.save(project_dir / "embeddings.npy", embeddings)
    labels = _cluster_embeddings(embeddings, config)
    for obj, cluster in zip(catalog["objects"], labels.tolist()):
        obj["cluster"] = int(cluster)
    save_project(project_dir, catalog)
    return catalog


def set_cluster_label(project_dir: str | Path, cluster: int, label: str | None, ignored: bool = False) -> int:
    catalog = load_project(project_dir)
    count = 0
    for obj in catalog["objects"]:
        if int(obj.get("cluster", -1)) == int(cluster):
            obj["label"] = label.strip() if label and label.strip() else None
            obj["ignored"] = bool(ignored)
            count += 1
    save_project(project_dir, catalog)
    return count


def set_object_label(project_dir: str | Path, object_id: str, label: str | None, ignored: bool = False) -> None:
    catalog = load_project(project_dir)
    for obj in catalog["objects"]:
        if obj["id"] == object_id:
            obj["label"] = label.strip() if label and label.strip() else None
            obj["ignored"] = bool(ignored)
            save_project(project_dir, catalog)
            return
    raise KeyError(object_id)


def render_object_overlay(project_dir: str | Path, obj: dict, alpha: float = 0.35) -> np.ndarray:
    image = load_rgb(obj["image_path"]).copy()
    mask = load_full_mask(project_dir, obj).astype(bool)
    overlay = image.copy()
    overlay[mask] = np.array([255, 0, 255], dtype=np.uint8)
    image = (image * (1.0 - alpha) + overlay * alpha).astype(np.uint8)
    x1, y1, x2, y2 = [int(v) for v in obj["bbox_xyxy"]]
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return image


def mask_to_polygon(mask: np.ndarray, min_points: int = 3) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < min_points:
        return None
    epsilon = max(1.0, 0.002 * cv2.arcLength(contour.astype(np.float32), True))
    contour = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True).reshape(-1, 2)
    if len(contour) < min_points:
        return None
    return contour


def _yolo_box(box: list[int], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def export_yolo(project_dir: str | Path, output_dir: str | Path, include_unlabeled: bool = False) -> dict:
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    catalog = load_project(project_dir)
    labeled = [o for o in catalog["objects"] if o.get("label") and not o.get("ignored")]
    classes = sorted({o["label"] for o in labeled})
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    for sub in ["images", "labels_detection", "labels_segmentation"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    objects_by_image: dict[str, list[dict]] = {}
    for obj in catalog["objects"]:
        if obj.get("ignored"):
            continue
        if obj.get("label") or include_unlabeled:
            objects_by_image.setdefault(obj["image_id"], []).append(obj)

    exported_images = 0
    exported_objects = 0
    image_index = {x["id"]: x for x in catalog["images"]}
    for image_id, objs in objects_by_image.items():
        valid = [o for o in objs if o.get("label")]
        if not valid:
            continue
        info = image_index[image_id]
        src = Path(info["path"])
        stem = f"{image_id}_{_sanitize_stem(src.stem)}"
        dst_image = output_dir / "images" / f"{stem}.jpg"
        save_rgb(dst_image, load_rgb(src))
        det_lines: list[str] = []
        seg_lines: list[str] = []
        for obj in valid:
            class_id = class_to_id[obj["label"]]
            w, h = int(obj["image_width"]), int(obj["image_height"])
            cx, cy, bw, bh = _yolo_box(obj["bbox_xyxy"], w, h)
            det_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            mask = load_full_mask(project_dir, obj)
            polygon = mask_to_polygon(mask)
            if polygon is not None:
                coords: list[str] = []
                for x, y in polygon:
                    coords.extend([f"{float(x) / w:.6f}", f"{float(y) / h:.6f}"])
                seg_lines.append(f"{class_id} " + " ".join(coords))
            exported_objects += 1
        (output_dir / "labels_detection" / f"{stem}.txt").write_text("\n".join(det_lines), encoding="utf-8")
        (output_dir / "labels_segmentation" / f"{stem}.txt").write_text("\n".join(seg_lines), encoding="utf-8")
        exported_images += 1

    (output_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    yaml_text = "path: .\ntrain: images\nval: images\nnames:\n" + "".join(
        f"  {idx}: {json.dumps(name, ensure_ascii=False)}\n" for idx, name in enumerate(classes)
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return {
        "images": exported_images,
        "objects": exported_objects,
        "classes": classes,
        "output_dir": str(output_dir.resolve()),
    }
