"""Deterministic test-time augmentation for sketch images."""
from __future__ import annotations

import cv2
import numpy as np


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    img = np.asarray(image)
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def deterministic_tta_batch(x: np.ndarray, pixels: int = 1) -> np.ndarray:
    """Return a batch of slight translations: center, left, right, up, down."""
    arr = np.asarray(x)
    if arr.ndim == 4:
        base = arr[0]
    else:
        base = arr
    if base.ndim == 3 and base.shape[-1] == 1:
        gray = base[:, :, 0]
    else:
        gray = base
    variants = [gray]
    for dx, dy in [(-pixels, 0), (pixels, 0), (0, -pixels), (0, pixels)]:
        variants.append(shift_image(gray, dx, dy))
    out = np.stack(variants).astype("float32")
    if out.ndim == 3:
        out = out[..., None]
    return out


def average_predictions(model, x: np.ndarray, use_tta: bool = True) -> np.ndarray:
    if not use_tta:
        return np.asarray(model.predict(x, verbose=0))[0]
    batch = deterministic_tta_batch(x)
    probs = np.asarray(model.predict(batch, verbose=0))
    return probs.mean(axis=0)
