from __future__ import annotations

import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from .core import load_full_mask, load_project, load_rgb, mask_to_polygon


def build_transform(size: int = 640, aggressive: bool = False) -> A.Compose:
    rotate = 25 if aggressive else 12
    shear = 10 if aggressive else 6
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size, p=1.0),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.10),
            A.RandomRotate90(p=0.20),
            A.Rotate(limit=rotate, border_mode=cv2.BORDER_REFLECT_101, p=0.30),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.04, 0.04),
                shear=(-shear, shear),
                rotate=0,
                mode=cv2.BORDER_REFLECT_101,
                p=0.25,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.35),
            A.RandomGamma(gamma_limit=(80, 120), p=0.20),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=12, p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.12),
            A.MotionBlur(blur_limit=7, p=0.10),
            A.GaussNoise(var_limit=(5.0, 35.0), p=0.18),
            A.ToGray(p=0.06),
            A.CoarseDropout(
                max_holes=6,
                max_height=max(8, size // 16),
                max_width=max(8, size // 16),
                min_holes=1,
                fill_value=0,
                p=0.12,
            ),
        ]
    )


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _apply_object_photometric(image: np.ndarray, masks: list[np.ndarray], rng: random.Random) -> np.ndarray:
    """Efeitos do estilo 'BB Brightness/Exposure/Blur/Noise/Motion Blur/Camera Gain'."""
    out = image.copy()
    for mask in masks:
        if rng.random() > 0.22:
            continue
        box = _bbox_from_mask(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        roi = out[y1:y2, x1:x2].copy()
        local_mask = (mask[y1:y2, x1:x2] > 0)
        if roi.size == 0 or not local_mask.any():
            continue
        effect = rng.choice(["brightness", "gain", "blur", "noise", "motion"])
        changed = roi.copy()
        if effect == "brightness":
            beta = rng.randint(-28, 28)
            changed = cv2.convertScaleAbs(roi, alpha=1.0, beta=beta)
        elif effect == "gain":
            alpha = rng.uniform(0.80, 1.25)
            changed = cv2.convertScaleAbs(roi, alpha=alpha, beta=0)
        elif effect == "blur":
            changed = cv2.GaussianBlur(roi, (5, 5), 0)
        elif effect == "motion":
            k = 7
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            changed = cv2.filter2D(roi, -1, kernel)
        else:
            noise = np.random.normal(0, rng.uniform(3.0, 12.0), roi.shape).astype(np.float32)
            changed = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        roi[local_mask] = changed[local_mask]
        out[y1:y2, x1:x2] = roi
    return out


def _write_yolo_labels(
    label_path_det: Path,
    label_path_seg: Path,
    masks: list[np.ndarray],
    labels: list[str],
    class_to_id: dict[str, int],
):
    det_lines: list[str] = []
    seg_lines: list[str] = []
    for mask, label in zip(masks, labels):
        box = _bbox_from_mask(mask)
        if box is None:
            continue
        h, w = mask.shape[:2]
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < 10:
            continue
        cid = class_to_id[label]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        det_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        poly = mask_to_polygon(mask)
        if poly is not None:
            coords: list[str] = []
            for x, y in poly:
                coords.extend([f"{float(x) / w:.6f}", f"{float(y) / h:.6f}"])
            seg_lines.append(f"{cid} " + " ".join(coords))
    label_path_det.write_text("\n".join(det_lines), encoding="utf-8")
    label_path_seg.write_text("\n".join(seg_lines), encoding="utf-8")


def _resize_pair(image: np.ndarray, masks: list[np.ndarray], width: int, height: int):
    image_r = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    masks_r = [cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) for m in masks]
    return image_r, masks_r


def _mosaic(samples: list[tuple[np.ndarray, list[np.ndarray], list[str]]], size: int):
    half = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    out_masks: list[np.ndarray] = []
    out_labels: list[str] = []
    positions = [(0, 0), (half, 0), (0, half), (half, half)]
    for (image, masks, labels), (x0, y0) in zip(samples, positions):
        image_r, masks_r = _resize_pair(image, masks, half, half)
        canvas[y0 : y0 + half, x0 : x0 + half] = image_r
        for mask_r, label in zip(masks_r, labels):
            full = np.zeros((size, size), dtype=np.uint8)
            full[y0 : y0 + half, x0 : x0 + half] = (mask_r > 0).astype(np.uint8)
            if full.sum() > 0:
                out_masks.append(full)
                out_labels.append(label)
    return canvas, out_masks, out_labels


def augment_project(
    project_dir: str | Path,
    output_dir: str | Path,
    copies_per_image: int = 3,
    size: int = 640,
    aggressive: bool = False,
    include_mosaic: bool = True,
    seed: int = 42,
) -> dict:
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    np.random.seed(seed)
    catalog = load_project(project_dir)
    labeled = [o for o in catalog["objects"] if o.get("label") and not o.get("ignored")]
    classes = sorted({o["label"] for o in labeled})
    class_to_id = {name: i for i, name in enumerate(classes)}
    for sub in ["images", "labels_detection", "labels_segmentation"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    by_image: dict[str, list[dict]] = {}
    for obj in labeled:
        by_image.setdefault(obj["image_id"], []).append(obj)
    image_map = {x["id"]: x for x in catalog["images"]}
    transform = build_transform(size=size, aggressive=aggressive)
    prepared: list[tuple[np.ndarray, list[np.ndarray], list[str], str]] = []
    generated = 0

    for image_id, objs in by_image.items():
        image = load_rgb(image_map[image_id]["path"])
        masks = [load_full_mask(project_dir, o) for o in objs]
        labels = [o["label"] for o in objs]
        prepared.append((image, masks, labels, image_id))
        for copy_idx in range(copies_per_image):
            result = transform(image=image, masks=masks)
            aug_image = result["image"]
            aug_masks = [(m > 0).astype(np.uint8) for m in result["masks"]]
            aug_image = _apply_object_photometric(aug_image, aug_masks, rng)
            stem = f"{image_id}_aug_{copy_idx + 1:02d}"
            Image.fromarray(aug_image).save(output_dir / "images" / f"{stem}.jpg", quality=95)
            _write_yolo_labels(
                output_dir / "labels_detection" / f"{stem}.txt",
                output_dir / "labels_segmentation" / f"{stem}.txt",
                aug_masks,
                labels,
                class_to_id,
            )
            generated += 1

    mosaics = 0
    if include_mosaic and len(prepared) >= 4:
        order = prepared[:]
        rng.shuffle(order)
        for start in range(0, len(order) - 3, 4):
            batch = [(x[0], x[1], x[2]) for x in order[start : start + 4]]
            image, masks, labels = _mosaic(batch, size)
            stem = f"mosaic_{mosaics + 1:04d}"
            Image.fromarray(image).save(output_dir / "images" / f"{stem}.jpg", quality=95)
            _write_yolo_labels(
                output_dir / "labels_detection" / f"{stem}.txt",
                output_dir / "labels_segmentation" / f"{stem}.txt",
                masks,
                labels,
                class_to_id,
            )
            mosaics += 1
            generated += 1

    (output_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    data_yaml = "path: .\ntrain: images\nval: images\nnames:\n" + "".join(
        f"  {idx}: {json.dumps(name, ensure_ascii=False)}\n" for idx, name in enumerate(classes)
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return {"generated_images": generated, "mosaics": mosaics, "classes": classes, "output_dir": str(output_dir.resolve())}
