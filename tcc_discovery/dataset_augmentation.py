from __future__ import annotations

import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import albumentations as A
import cv2
import numpy as np
import yaml
from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]

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

_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    'train': ('train',),
    'val': ('valid', 'val'),
    'test': ('test',),
}


def _configured_split_value(cfg: dict[str, Any], split: str) -> str | None:
    value = cfg.get(split)
    if not value:
        return None
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f'{split}: listas com múltiplos caminhos ainda não são suportadas.')
        value = value[0]
    return str(value)


def _resolve_dataset_split(data_yaml: Path, cfg: dict[str, Any], split: str) -> Path | None:
    """Resolve um split YOLO e tolera exports Roboflow com ../train/images."""
    raw_value = _configured_split_value(cfg, split)
    if raw_value is None:
        return None

    root = Path(str(cfg.get('path', '.'))).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    configured = Path(raw_value).expanduser()
    primary = configured if configured.is_absolute() else (root / configured).resolve()
    if primary.exists():
        return primary

    clean_parts = [part for part in configured.parts if part not in {'.', '..', '/'}]
    candidates: list[Path] = []
    if clean_parts:
        candidates.append((data_yaml.parent / Path(*clean_parts)).resolve())

    for alias in _SPLIT_ALIASES.get(split, (split,)):
        candidates.append((data_yaml.parent / alias / 'images').resolve())
        candidates.append((data_yaml.parent / alias).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return primary


def resolve_yolo_split(data_yaml: str | Path, split: str) -> Path | None:
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'data.yaml inválido: {data_yaml}')
    return _resolve_dataset_split(data_yaml, cfg, split)


def _count_images(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def inspect_yolo_detection_dataset(data_yaml: str | Path) -> dict[str, Any]:
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'data.yaml inválido: {data_yaml}')
    resolved = {
        'train': _resolve_dataset_split(data_yaml, cfg, 'train'),
        'val': _resolve_dataset_split(data_yaml, cfg, 'val'),
        'test': _resolve_dataset_split(data_yaml, cfg, 'test'),
    }
    return {
        'data_yaml': str(data_yaml),
        'configured': {key: _configured_split_value(cfg, key) for key in ('train', 'val', 'test')},
        'resolved': {key: str(value) if value else None for key, value in resolved.items()},
        'exists': {key: bool(value and value.exists()) for key, value in resolved.items()},
        'images': {key: _count_images(value) for key, value in resolved.items()},
        'names': cfg.get('names', {}),
    }


def _labels_dir_from_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == 'images':
            parts[index] = 'labels'
            return Path(*parts)
    return images_dir.parent / 'labels'


def _sanitize_yolo_box(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    *,
    max_overflow: float = 0.02,
) -> tuple[list[float] | None, str]:
    """Normaliza uma box YOLO antes de passá-la ao Albumentations."""
    values = np.asarray([cx, cy, bw, bh], dtype=np.float64)
    if not np.isfinite(values).all() or bw <= 0.0 or bh <= 0.0:
        return None, 'invalid'

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    overflow = max(0.0, -x1, -y1, x2 - 1.0, y2 - 1.0)
    if overflow > max_overflow:
        return None, 'invalid'

    x1c = min(1.0, max(0.0, x1))
    y1c = min(1.0, max(0.0, y1))
    x2c = min(1.0, max(0.0, x2))
    y2c = min(1.0, max(0.0, y2))
    if x2c <= x1c or y2c <= y1c:
        return None, 'invalid'

    sanitized = [
        (x1c + x2c) / 2.0,
        (y1c + y2c) / 2.0,
        x2c - x1c,
        y2c - y1c,
    ]
    corrected = any(abs(a - b) > 1e-9 for a, b in zip(sanitized, [cx, cy, bw, bh]))
    return sanitized, 'corrected' if corrected else 'ok'


def _read_yolo_boxes(label_path: Path) -> tuple[list[list[float]], list[int], dict[str, int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    stats = {'total': 0, 'corrected': 0, 'skipped_invalid': 0}
    if not label_path.exists():
        return boxes, classes, stats

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
        stats['total'] += 1
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        sanitized, status = _sanitize_yolo_box(cx, cy, bw, bh)
        if sanitized is None:
            stats['skipped_invalid'] += 1
            continue
        if status == 'corrected':
            stats['corrected'] += 1
        boxes.append(sanitized)
        classes.append(cid)
    return boxes, classes, stats


def _write_yolo_boxes(path: Path, boxes: list[list[float]], classes: list[int]) -> None:
    lines = []
    for cid, box in zip(classes, boxes):
        if len(box) < 4:
            continue
        sanitized, _ = _sanitize_yolo_box(*map(float, box[:4]), max_overflow=0.05)
        if sanitized is None:
            continue
        cx, cy, bw, bh = sanitized
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


def _copy_pairs(
    pairs: list[tuple[Path, Path, Path]],
    output_root: Path,
    split_name: str,
    advance: Callable[[str], None],
) -> int:
    for index, (image_path, label_path, relative) in enumerate(pairs, start=1):
        out_image = output_root / split_name / 'images' / relative
        out_label = output_root / split_name / 'labels' / relative.with_suffix('.txt')
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_image)
        if label_path.exists():
            shutil.copy2(label_path, out_label)
        else:
            out_label.write_text('', encoding='utf-8')
        advance(f'Copiando {split_name}: {index}/{len(pairs)}')
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
    progress_cb: ProgressCallback | None = None,
    log_cb: LogCallback | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    data_yaml = Path(data_yaml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    train_images = _resolve_dataset_split(data_yaml, cfg, 'train')
    val_images = _resolve_dataset_split(data_yaml, cfg, 'val')
    test_images = _resolve_dataset_split(data_yaml, cfg, 'test')
    if train_images is None or not train_images.exists():
        raise FileNotFoundError(f'Split train não encontrado: {train_images}')

    train_pairs = _iter_pairs(train_images)
    if not train_pairs:
        raise ValueError(f'Nenhuma imagem encontrada em {train_images}')
    val_pairs = _iter_pairs(val_images) if preserve_validation and val_images is not None and val_images.exists() else []
    test_pairs = _iter_pairs(test_images) if preserve_test and test_images is not None and test_images.exists() else []
    planned_mosaics = len(train_pairs) // 4 if include_mosaic and len(train_pairs) >= 4 else 0
    total_work = (
        (len(train_pairs) if include_original_train else 0)
        + len(val_pairs)
        + len(test_pairs)
        + len(train_pairs) * int(copies_per_image)
        + planned_mosaics
    )
    total_work = max(1, total_work)
    done = 0
    report_every = max(1, total_work // 400)

    def log(message: str) -> None:
        if log_cb is not None:
            log_cb(message)

    def report(message: str, *, force: bool = False) -> None:
        if progress_cb is not None and (force or done <= 1 or done >= total_work or done % report_every == 0):
            progress_cb(done, total_work, message)

    def advance(message: str) -> None:
        nonlocal done
        done += 1
        report(message)

    log(f'Dataset validado: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test.')
    report('Preparando saída...', force=True)

    if output_dir.exists():
        if not overwrite:
            if any(output_dir.iterdir()):
                raise FileExistsError(f'Saída já existe: {output_dir}. Use overwrite=True para recriar.')
            shutil.rmtree(output_dir)
        else:
            log(f'Recriando pasta de saída: {output_dir}')
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled = dict(DEFAULT_AUGMENTATIONS)
    if enabled_augmentations:
        enabled.update({str(k): bool(v) for k, v in enabled_augmentations.items()})
    transform = _build_detection_transform(size, aggressive, enabled)
    rng = random.Random(seed)
    np.random.seed(seed)

    copied_train = 0
    if include_original_train:
        log('Copiando imagens originais de TRAIN...')
        copied_train = _copy_pairs(train_pairs, output_dir, 'train', advance)

    copied_val = 0
    if val_pairs:
        log('Copiando VALID sem augmentation...')
        copied_val = _copy_pairs(val_pairs, output_dir, 'valid', advance)

    copied_test = 0
    if test_pairs:
        log('Copiando TEST sem augmentation...')
        copied_test = _copy_pairs(test_pairs, output_dir, 'test', advance)

    generated = 0
    bbox_stats = {'total': 0, 'corrected': 0, 'skipped_invalid': 0}
    mosaic_pool: list[tuple[np.ndarray, list[list[float]], list[int]]] = []
    log(f'Gerando {len(train_pairs) * int(copies_per_image)} imagens aumentadas...')
    for image_number, (image_path, label_path, relative) in enumerate(train_pairs, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f'Falha lendo imagem: {image_path}')
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, classes, current_stats = _read_yolo_boxes(label_path)
        for key in bbox_stats:
            bbox_stats[key] += current_stats[key]
        mosaic_pool.append((image, boxes, classes))
        stem = _safe_stem(relative)
        for copy_idx in range(copies_per_image):
            result = transform(image=image, bboxes=boxes, class_labels=classes)
            aug_image = result['image']
            aug_boxes = [list(map(float, box[:4])) for box in result['bboxes']]
            aug_classes = [int(x) for x in result['class_labels']]
            out_stem = f'{stem}_aug_{copy_idx + 1:02d}'
            Image.fromarray(aug_image).save(output_dir / 'train' / 'images' / f'{out_stem}.jpg', quality=95)
            _write_yolo_boxes(output_dir / 'train' / 'labels' / f'{out_stem}.txt', aug_boxes, aug_classes)
            generated += 1
            advance(
                f'Augmentation: imagem {image_number}/{len(train_pairs)} · '
                f'cópia {copy_idx + 1}/{copies_per_image}'
            )

    mosaics = 0
    if planned_mosaics:
        log(f'Gerando {planned_mosaics} mosaicos 2x2...')
        order = mosaic_pool[:]
        rng.shuffle(order)
        for start in range(0, len(order) - 3, 4):
            image, boxes, classes = _mosaic_detection(order[start:start + 4], size)
            out_stem = f'mosaic_{mosaics + 1:05d}'
            Image.fromarray(image).save(output_dir / 'train' / 'images' / f'{out_stem}.jpg', quality=95)
            _write_yolo_boxes(output_dir / 'train' / 'labels' / f'{out_stem}.txt', boxes, classes)
            mosaics += 1
            advance(f'Mosaicos: {mosaics}/{planned_mosaics}')

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

    duration_seconds = round(time.monotonic() - started, 3)
    manifest = {
        'source_data_yaml': str(data_yaml),
        'output_data_yaml': str(output_dir / 'data.yaml'),
        'resolved_train': str(train_images),
        'resolved_validation': str(val_images) if val_images else None,
        'resolved_test': str(test_images) if test_images else None,
        'source_train_images': len(train_pairs),
        'copied_original_train': copied_train,
        'generated_augmented_train': generated,
        'generated_mosaics_train': mosaics,
        'final_train_images': copied_train + generated + mosaics,
        'validation_images_unchanged': copied_val,
        'test_images_unchanged': copied_test,
        'bbox_labels_total': bbox_stats['total'],
        'bbox_labels_corrected_at_border': bbox_stats['corrected'],
        'bbox_labels_skipped_invalid': bbox_stats['skipped_invalid'],
        'copies_per_image': copies_per_image,
        'size': size,
        'aggressive': aggressive,
        'enabled_augmentations': enabled,
        'seed': seed,
        'duration_seconds': duration_seconds,
    }
    (output_dir / 'augmentation_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    done = total_work
    report('Concluído.', force=True)
    log(f'Concluído em {duration_seconds:.1f}s. Train final: {manifest["final_train_images"]} imagens.')
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
