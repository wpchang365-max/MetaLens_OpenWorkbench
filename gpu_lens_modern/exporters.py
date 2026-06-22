from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TEXT_FORMATS = {".txt", ".csv"}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def export_array(path: Path, data: np.ndarray, headers: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data)
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            if headers:
                writer.writerow(headers)
            writer.writerows(arr.reshape((-1, arr.shape[-1] if arr.ndim > 1 else 1)))
    else:
        np.savetxt(path, arr, fmt="%.10g", delimiter="\t", header="\t".join(headers or []))
    return path


def export_records(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_image(path: Path, matrix: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=np.float64)
    arr -= float(arr.min())
    arr /= max(float(arr.max()), 1e-15)
    r = np.clip(255 * arr * 1.6, 0, 255)
    g = np.clip(255 * np.maximum(arr - 0.2, 0) * 1.4, 0, 255)
    b = np.clip(255 * np.maximum(arr - 0.65, 0) * 3.0, 0, 255)
    image = Image.fromarray(np.dstack([r, g, b]).astype(np.uint8), "RGB")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0)
    else:
        image.save(path)
    return path


def export_manifest(path: Path, payload: Any) -> Path:
    if is_dataclass(payload):
        payload = asdict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
