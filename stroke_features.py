"""
stroke_features.py — Trích xuất đặc trưng chuỗi nét vẽ (stroke) DÙNG CHUNG cho cả
train (src/training/train_stroke_model.py) và inference (backend/app.py).
(Phase 2, Task 10)

Mục tiêu: chỉ có MỘT nguồn duy nhất định nghĩa cách biến strokes -> tensor, để
train và inference không bao giờ bị lệch (đây là lỗi rất hay gặp với model chuỗi).

Định dạng đầu vào `strokes`:
    list[stroke], mỗi stroke = list[point], point = {"x": float, "y": float, "t": float}
    (hoặc một chuỗi JSON của cấu trúc trên)

Đầu ra: mảng float32 shape (MAX_LEN, NUM_FEATURES).

Bộ đặc trưng (NUM_FEATURES = 9) cho mỗi điểm sau khi chuẩn hóa & resample:
    0: x          - tọa độ x chuẩn hóa [0,1]
    1: y          - tọa độ y chuẩn hóa [0,1]
    2: dx         - chênh lệch x so với điểm trước
    3: dy         - chênh lệch y so với điểm trước
    4: speed      - tốc độ di chuyển = hypot(dx, dy)
    5: dir_cos    - cos hướng di chuyển
    6: dir_sin    - sin hướng di chuyển
    7: pen_up     - 1.0 tại điểm cuối mỗi nét (nhấc bút), 0.0 nếu không
    8: t          - thời gian chuẩn hóa [0,1]
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

# Kích thước canvas vẽ (web/game) — phải khớp với phía thu thập dữ liệu.
CANVAS_W = 960
CANVAS_H = 540

MAX_LEN = 96
NUM_FEATURES = 9

_EPS = 1e-6


def _parse(strokes: Any) -> list:
    if isinstance(strokes, str):
        try:
            strokes = json.loads(strokes)
        except Exception:
            strokes = []
    if not isinstance(strokes, list):
        return []
    return strokes


def strokes_to_sequence(
    strokes: Any,
    max_len: int = MAX_LEN,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> np.ndarray:
    """Biến strokes -> tensor (max_len, NUM_FEATURES) float32."""
    strokes = _parse(strokes)

    xs: list[float] = []
    ys: list[float] = []
    ts: list[float] = []
    pen_up: list[float] = []

    for stroke in strokes:
        if not isinstance(stroke, list) or not stroke:
            continue
        pts = [p for p in stroke if isinstance(p, dict)]
        n = len(pts)
        for i, p in enumerate(pts):
            xs.append(float(p.get("x", 0.0)))
            ys.append(float(p.get("y", 0.0)))
            ts.append(float(p.get("t", 0.0)))
            pen_up.append(1.0 if i == n - 1 else 0.0)

    if not xs:
        return np.zeros((max_len, NUM_FEATURES), dtype="float32")

    x = np.asarray(xs, dtype="float32") / max(canvas_w, 1)
    y = np.asarray(ys, dtype="float32") / max(canvas_h, 1)
    t = np.asarray(ts, dtype="float32")
    if t.max() > t.min():
        t = (t - t.min()) / (t.max() - t.min())
    else:
        t = np.zeros_like(t)

    dx = np.concatenate([[0.0], np.diff(x)]).astype("float32")
    dy = np.concatenate([[0.0], np.diff(y)]).astype("float32")
    speed = np.hypot(dx, dy).astype("float32")
    safe = np.where(speed > _EPS, speed, 1.0)
    dir_cos = (dx / safe).astype("float32")
    dir_sin = (dy / safe).astype("float32")
    pen = np.asarray(pen_up, dtype="float32")

    seq = np.stack([x, y, dx, dy, speed, dir_cos, dir_sin, pen, t], axis=1).astype("float32")

    # Resample/pad về độ dài cố định.
    if len(seq) >= max_len:
        idx = np.linspace(0, len(seq) - 1, max_len).astype(int)
        seq = seq[idx]
    else:
        pad = np.zeros((max_len - len(seq), NUM_FEATURES), dtype="float32")
        seq = np.vstack([seq, pad])
    return seq.astype("float32")


def strokes_to_batch(
    strokes: Any,
    max_len: int = MAX_LEN,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> np.ndarray:
    """Như strokes_to_sequence nhưng thêm batch dim -> (1, max_len, NUM_FEATURES)."""
    seq = strokes_to_sequence(strokes, max_len=max_len, canvas_w=canvas_w, canvas_h=canvas_h)
    return seq.reshape(1, max_len, NUM_FEATURES)


def count_active_points(seq: np.ndarray) -> int:
    """Đếm số điểm có chuyển động (không phải padding) — dùng để lọc mẫu rác."""
    return int(np.count_nonzero(np.abs(seq[:, :2]).sum(axis=1)))
