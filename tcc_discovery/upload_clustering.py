from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

ProgressCallback = Callable[[int, int, str], None]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:80] or "upload"


def _media_kind(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"Formato não suportado: {suffix or '(sem extensão)'}")


def _write_rgb_jpeg(path: Path, rgb: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path, format="JPEG", quality=quality)


def prepare_uploaded_media(
    data: bytes,
    filename: str,
    cache_root: str | Path = "workspace/_uploads",
    sample_every_seconds: float = 1.0,
    max_frames: int = 120,
    extract_all_frames: bool = False,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    overwrite: bool = True,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    """Materializa um upload de imagem/vídeo como uma pasta de frames.

    Para vídeo há dois modos:
    - ``extract_all_frames=True``: mantém todos os frames do intervalo escolhido;
    - ``extract_all_frames=False``: amostra segundo ``sample_every_seconds``.

    ``max_frames=0`` significa sem limite. Um valor positivo limita a quantidade
    final de frames extraídos. A pasta de frames fica fora do diretório do
    projeto de clustering para que o projeto possa ser recriado sem apagar a
    mídia enviada.
    """
    if not data:
        raise ValueError("Upload vazio.")
    kind = _media_kind(filename)
    digest = hashlib.sha256(data).hexdigest()[:16]
    root = Path(cache_root).expanduser().resolve() / f"{_safe_stem(filename)}_{digest}"
    frames_dir = root / "frames"
    original_dir = root / "original"

    if overwrite and root.exists():
        shutil.rmtree(root)
    frames_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower()
    original_path = original_dir / f"{_safe_stem(filename)}{suffix}"
    original_path.write_bytes(data)

    if kind == "image":
        try:
            with Image.open(original_path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"Não foi possível abrir a imagem {filename}.") from exc
        frame_path = frames_dir / "frame_000001.jpg"
        _write_rgb_jpeg(frame_path, rgb)
        metadata = {
            "media_type": "image",
            "filename": filename,
            "sha256_16": digest,
            "original_path": str(original_path),
            "frames_dir": str(frames_dir),
            "source_width": int(rgb.shape[1]),
            "source_height": int(rgb.shape[0]),
            "source_frames": 1,
            "sampled_frames": 1,
            "fps": None,
            "duration_seconds": None,
            "sample_every_seconds": None,
            "extract_all_frames": False,
            "max_frames": 1,
            "start_seconds": 0.0,
            "end_seconds": None,
        }
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress_cb:
            progress_cb(1, 1, "Imagem preparada para clustering")
        return metadata

    cap = cv2.VideoCapture(str(original_path))
    if not cap.isOpened():
        raise ValueError(
            "Não foi possível abrir o vídeo. O contêiner/codec pode não ser suportado pelo OpenCV desta instalação."
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0:
            fps = 30.0
        duration = (total_frames / fps) if total_frames > 0 else None

        start_seconds = max(0.0, float(start_seconds))
        if duration is not None:
            start_seconds = min(start_seconds, max(0.0, duration))
        resolved_end = duration if end_seconds is None else max(start_seconds, float(end_seconds))
        if duration is not None and resolved_end is not None:
            resolved_end = min(resolved_end, duration)

        start_frame = max(0, int(round(start_seconds * fps)))
        if resolved_end is None:
            end_frame = max(start_frame + 1, total_frames) if total_frames > 0 else start_frame + 1
        else:
            end_frame = int(round(resolved_end * fps))
        if total_frames > 0:
            end_frame = min(max(start_frame + 1, end_frame), total_frames)

        if extract_all_frames:
            step = 1
            resolved_sample_every = 1.0 / fps
        else:
            resolved_sample_every = max(0.01, float(sample_every_seconds))
            step = max(1, int(round(resolved_sample_every * fps)))

        frame_limit = int(max_frames)
        unlimited = frame_limit <= 0

        if total_frames > 0:
            indices = list(range(start_frame, end_frame, step))
        else:
            indices = []

        if indices and not unlimited and len(indices) > frame_limit:
            # Mantém cobertura de toda a duração escolhida quando existe limite.
            positions = np.linspace(0, len(indices) - 1, frame_limit, dtype=int)
            indices = [indices[int(i)] for i in positions]

        saved = 0
        source_indices: list[int] = []

        if indices:
            total_targets = len(indices)
            for i, frame_idx in enumerate(indices, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                saved += 1
                source_indices.append(int(frame_idx))
                frame_path = frames_dir / f"frame_{saved:06d}_src_{int(frame_idx):09d}.jpg"
                _write_rgb_jpeg(frame_path, rgb)
                if progress_cb:
                    progress_cb(i, total_targets, f"Extraindo frame {i}/{total_targets}")
        else:
            # Leitura sequencial quando frame_count/duração não está disponível.
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            next_keep = start_frame
            while unlimited or saved < frame_limit:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if resolved_end is not None and (frame_idx / fps) >= resolved_end:
                    break
                if frame_idx >= next_keep:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    saved += 1
                    source_indices.append(int(frame_idx))
                    frame_path = frames_dir / f"frame_{saved:06d}_src_{int(frame_idx):09d}.jpg"
                    _write_rgb_jpeg(frame_path, rgb)
                    next_keep = frame_idx + step
                    if progress_cb:
                        progress_cb(saved, max(saved, 1), f"Extraindo frame {saved}")
                frame_idx += 1

        if saved == 0:
            raise ValueError("Nenhum frame pôde ser extraído do vídeo.")

        metadata = {
            "media_type": "video",
            "filename": filename,
            "sha256_16": digest,
            "original_path": str(original_path),
            "frames_dir": str(frames_dir),
            "source_width": width,
            "source_height": height,
            "source_frames": total_frames if total_frames > 0 else None,
            "sampled_frames": saved,
            "fps": fps,
            "duration_seconds": duration,
            "sample_every_seconds": resolved_sample_every,
            "extract_all_frames": bool(extract_all_frames),
            "max_frames": None if unlimited else frame_limit,
            "start_seconds": start_seconds,
            "end_seconds": resolved_end,
            "source_frame_indices": source_indices,
        }
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata
    finally:
        cap.release()
