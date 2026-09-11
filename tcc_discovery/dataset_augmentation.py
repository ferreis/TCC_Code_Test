from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml
from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

DEFAULT_AUGMENTATIONS: dict[str, bool] = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotate90': True,
    'rotate': True,
    'affine': True,
    'brightness_contrast': True,
    'gamma': True,
    'hsv': True,
    'gaussian_blur': True,
    'motion_blur': True,
    'gaussian_noise': True,
    'grayscale': True,
    'coarse_dropout': True,
}


def _resolve_dataset_split(data_yaml: Path, cfg: dict[str, Any], split: str) -> Path | None:
    value = cfg.get(split)
    if not value:
        return None
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f'{split}: listas com múltiplos caminhos ainda não são suportadas.')
        value = value[0]
    root = Path(str(cfg.get('path', '.'))).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _labels_dir_from_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == 'images':
            parts[index] = 'labels'
            return Path(*parts)
    return images_dir.parent / 'labels'


def _read_yolo_boxes(label_path: Path) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.exists():
        return boxes, classes
    for line_no, raw in enumerate(label_path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f'{label_path}:{line_no}: esperado label YOLO detection com 5 colunas; '
                f'encontradas {len(parts)}. Para segmentação use o pipeline de máscaras.'
            )
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        boxes.append([cx, cy, bw, bh])
        classes.append(cid)
    return boxes, classes


def _write_yolo_boxes(path: Path, boxes: list[list[float]], classes: list[int]) -> None:
    lines = []
    for cid, (cx, cy, bw, bh) in zip(classes, boxes):
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f'{int(cid)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def _safe_stem(relative_path: Path) -> str:
    text = '__'.join(relative_path.with_suffix('').parts)
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text)


def _build_detection_transform(
    size: int,
    aggressive: bool,
    enabled: dict[str, bool],
) -> A.Compose:
    rotate = 25 if aggressive else 12
    shear = 10 if aggressive else 6
    transforms: list[Any] = [
        A.LongestMaxSize(max_size=size, p=1.0),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    ]
    if enabled.get('horizontal_flip', True):
        transforms.append(A.HorizontalFlip(p=0.50))
    if enabled.get('vertical_flip', True):
        transforms.append(A.VerticalFlip(p=0.15))
    if enabled.get('rotate90', True):
        transforms.append(A.RandomRotate90(p=0.25))
    if enabled.get('rotate', True):
        transforms.append(A.Rotate(limit=rotate, border_mode=cv2.BORDER_REFLECT_101, p=0.35))
    if enabled.get('affine', True):
        transforms.append(
            A.Affine(
                scale=(0.88, 1.12) if aggressive else (0.92, 1.08),
                translate_percent=(-0.06, 0.06) if aggressive else (-0.04, 0.04),
                shear=(-shear, shear),
                rotate=0,
                mode=cv2.BORDER_REFLECT_101,
                p=0.30,
            )
        )
    if enabled.get('brightness_contrast', True):
        transforms.append(A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.40))
    if enabled.get('gamma', True):
        transforms.append(A.RandomGamma(gamma_limit=(78, 122), p=0.25))
    if enabled.get('hsv', True):
        transforms.append(A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=14, p=0.30))
    if enabled.get('gaussian_blur', True):
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.12))
    if enabled.get('motion_blur', True):
        transforms.append(A.MotionBlur(blur_limit=7, p=0.12))
    if enabled.get('gaussian_noise', True):
        transforms.append(A.GaussNoise(var_limit=(5.0, 35.0), p=0.18))
    if enabled.get('grayscale', True):
        transforms.append(A.ToGray(p=0.06))
    if enabled.get('coarse_dropout', True):
        transforms.append(
            A.CoarseDropout(
                max_holes=6,
                max_height=max(8, size // 16),
                max_width=max(8, size // 16),
                min_holes=1,
                fill_value=0,
                p=0.12,
            )
        )
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.15,
        ),
    )


def _iter_pairs(images_dir: Path) -> list[tuple[Path, Path, Path]]:
    labels_dir = _labels_dir_from_images(images_dir)
    pairs: list[tuple[Path, Path, Path]] = []
    for image_path in sorted(p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix('.txt')
        pairs.append((image_path, label_path, relative))
    return pairs


def _copy_split(images_dir: Path, output_root: Path, split_name: str) -> int:
    pairs = _iter_pairs(images_dir)
    for image_path, label_path, relative in pairs:
        out_image = output_root / split_name / 'images' / relative
        out_label = output_root / split_name / 'labels' / relative.with_suffix('.txt')
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_image)
        if label_path.exists():
            shutil.copy2(label_path, out_label)
        else:
            out_label.write_text('', encoding='utf-8')
    return len(pairs)


def _mosaic_detection(
    samples: list[tuple[np.ndarray, list[list[float]], list[int]]],
    size: int,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    half = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    out_boxes: list[list[float]] = []
    out_classes: list[int] = []
    positions = [(0, 0), (half, 0), (0, half), (half, half)]
    for (image, boxes, classes), (x0, y0) in zip(samples, positions):
        resized = cv2.resize(image, (half, half), interpolation=cv2.INTER_LINEAR)
        canvas[y0:y0 + half, x0:x0 + half] = resized
        for box, cid in zip(boxes, classes):
            cx, cy, bw, bh = box
            out_boxes.append([
                (x0 + cx * half) / size,
                (y0 + cy * half) / size,
                bw * half / size,
                bh * half / size,
            ])
            out_classes.append(cid)
    return canvas, out_boxes, out_classes


def augment_yolo_detection_dataset(
    data_yaml: str | Path,
    output_dir: str | Path,
    *,
    copies_per_image: int = 3,
    size: int = 640,
    aggressive: bool = False,
    include_mosaic: bool = True,
    include_original_train: bool = True,
    preserve_validation: bool = True,
    preserve_test: bool = True,
    enabled_augmentations: dict[str, bool] | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_yaml = Path(data_yaml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f'Saída já existe: {output_dir}. Use overwrite=True para recriar.')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    train_images = _resolve_dataset_split(data_yaml, cfg, 'train')
    val_images = _resolve_dataset_split(data_yaml, cfg, 'val')
    test_images = _resolve_dataset_split(data_yaml, cfg, 'test')
    if train_images is None or not train_images.exists():
        raise FileNotFoundError(f'Split train não encontrado: {train_images}')

    enabled = dict(DEFAULT_AUGMENTATIONS)
    if enabled_augmentations:
        enabled.update({str(k): bool(v) for k, v in enabled_augmentations.items()})
    transform = _build_detection_transform(size, aggressive, enabled)
    rng = random.Random(seed)
    np.random.seed(seed)

    train_pairs = _iter_pairs(train_images)
    if not train_pairs:
        raise ValueError(f'Nenhuma imagem encontrada em {train_images}')

    copied_train = _copy_split(train_images, output_dir, 'train') if include_original_train else 0
    copied_val = 0
    copied_test = 0
    if preserve_validation and val_images is not None and val_images.exists():
        copied_val = _copy_split(val_images, output_dir, 'valid')
    if preserve_test and test_images is not None and test_images.exists():
        copied_test = _copy_split(test_images, output_dir, 'test')

    generated = 0
    mosaic_pool: list[tuple[np.ndarray, list[list[float]], list[int]]] = []
    for image_path, label_path, relative in train_pairs:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f'Falha lendo imagem: {image_path}')
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, classes = _read_yolo_boxes(label_path)
        mosaic_pool.append((image, boxes, classes))
        stem = _safe_stem(relative)
        for copy_idx in range(copies_per_image):
            result = transform(image=image, bboxes=boxes, class_labels=classes)
            aug_image = result['image']
            aug_boxes = [list(map(float, box)) for box in result['bboxes']]
            aug_classes = [int(x) for x in result['class_labels']]
            out_stem = f'{stem}_aug_{copy_idx + 1:02d}'
            Image.fromarray(aug_image).save(output_dir / 'train' / 'images' / f'{out_stem}.jpg', quality=95)
            _write_yolo_boxes(output_dir / 'train' / 'labels' / f'{out_stem}.txt', aug_boxes, aug_classes)
            generated += 1

    mosaics = 0
    if include_mosaic and len(mosaic_pool) >= 4:
        order = mosaic_pool[:]
        rng.shuffle(order)
        for start in range(0, len(order) - 3, 4):
            image, boxes, classes = _mosaic_detection(order[start:start + 4], size)
            out_stem = f'mosaic_{mosaics + 1:05d}'
            Image.fromarray(image).save(output_dir / 'train' / 'images' / f'{out_stem}.jpg', quality=95)
            _write_yolo_boxes(output_dir / 'train' / 'labels' / f'{out_stem}.txt', boxes, classes)
            mosaics += 1

    names = cfg.get('names', {})
    output_cfg: dict[str, Any] = {
        'path': '.',
        'train': 'train/images',
        'names': names,
    }
    if copied_val:
        output_cfg['val'] = 'valid/images'
    if copied_test:
        output_cfg['test'] = 'test/images'
    if 'nc' in cfg:
        output_cfg['nc'] = cfg['nc']
    (output_dir / 'data.yaml').write_text(
        yaml.safe_dump(output_cfg, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )

    manifest = {
        'source_data_yaml': str(data_yaml),
        'output_data_yaml': str(output_dir / 'data.yaml'),
        'source_train_images': len(train_pairs),
        'copied_original_train': copied_train,
        'generated_augmented_train': generated,
        'generated_mosaics_train': mosaics,
        'final_train_images': copied_train + generated + mosaics,
        'validation_images_unchanged': copied_val,
        'test_images_unchanged': copied_test,
        'copies_per_image': copies_per_image,
        'size': size,
        'aggressive': aggressive,
        'enabled_augmentations': enabled,
        'seed': seed,
    }
    (output_dir / 'augmentation_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return manifest


def discover_yolo_datasets(*roots: str | Path) -> list[Path]:
    found: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        if root.is_file() and root.name in {'data.yaml', 'dataset.yaml'}:
            found.add(root.resolve())
            continue
        for name in ('data.yaml', 'dataset.yaml'):
            found.update(path.resolve() for path in root.rglob(name))
    return sorted(found)
