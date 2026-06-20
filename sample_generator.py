"""
sample_generator.py — Tự sinh ảnh minh hoạ & dự đoán (dùng chung cho WEB và notebook).

Trả về PNG dạng bytes cho 3 loại ảnh giống bộ mẫu của dự án:
  - sample_grid_png(...)            -> 1 hình mẫu mỗi lớp
  - prediction_sample_png(...)      -> 1 hình + biểu đồ xác suất tất cả lớp
  - multiple_predictions_png(...)   -> lưới nhiều dự đoán (xanh=đúng, đỏ=sai)

Dữ liệu lấy từ DATA_DIR (file .npy 28x28). Nếu thiếu, tự tải phần nhỏ từ QuickDraw.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

QD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def _load_class_samples(label: str, n: int, data_dir: Path) -> np.ndarray:
    """Lấy n mẫu của 1 lớp; ưu tiên file .npy local, thiếu thì tải phần nhỏ."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{label}.npy"
    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if len(arr) >= n:
            return np.array(arr[:n])
    nbytes = 256 + n * 784
    r = requests.get(QD_URL.format(quote(label)), headers={"Range": f"bytes=0-{nbytes}"}, timeout=60)
    r.raise_for_status()
    raw = r.content
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{label}: dữ liệu không hợp lệ")
    major = raw[6]
    off = (10 + int.from_bytes(raw[8:10], "little")) if major == 1 else (12 + int.from_bytes(raw[8:12], "little"))
    avail = (len(raw) - off) // 784
    arr = np.frombuffer(raw[off:off + avail * 784], dtype=np.uint8).reshape(avail, 784)
    return arr[:n]


def _fig_to_png() -> bytes:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def _norm(a: np.ndarray) -> np.ndarray:
    return (a.astype("float32") / 255.0).reshape(-1, 28, 28, 1)


def sample_grid_png(categories, data_dir: Path, vi_map=None) -> bytes:
    """1 hình mẫu mỗi lớp."""
    vi_map = vi_map or {}
    n = len(categories)
    cols = 8
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 1.6, rows * 1.8))
    for c, label in enumerate(categories):
        try:
            img = _load_class_samples(label, 1, data_dir)[0].reshape(28, 28)
        except Exception:
            img = np.zeros((28, 28), dtype="uint8")
        plt.subplot(rows, cols, c + 1)
        plt.imshow(img, cmap="gray_r")
        plt.axis("off")
        vi = vi_map.get(label, "")
        plt.title(f"{label}\n({vi})" if vi else label, fontsize=8)
    plt.suptitle(f"Sample Drawings - QuickDraw ({n} lớp)", fontsize=14, fontweight="bold")
    plt.tight_layout(h_pad=1.6)
    return _fig_to_png()


def _gather_test(categories, data_dir: Path, per_class: int):
    X, y = [], []
    for c, label in enumerate(categories):
        try:
            arr = _load_class_samples(label, per_class, data_dir)
        except Exception:
            continue
        X.append(_norm(arr))
        y.append(np.full(len(arr), c))
    if not X:
        raise RuntimeError("Không nạp được dữ liệu mẫu (thiếu .npy và không tải được QuickDraw).")
    return np.concatenate(X), np.concatenate(y)


def prediction_sample_png(model, categories, data_dir: Path, seed: int = 0) -> bytes:
    """1 hình + biểu đồ xác suất tất cả lớp."""
    n = len(categories)
    X, y = _gather_test(categories, data_dir, per_class=20)
    i = int(np.random.default_rng(seed).integers(len(X)))
    prob = model.predict(X[i:i + 1], verbose=0)[0]
    pred = int(prob.argmax())
    fig, ax = plt.subplots(1, 2, figsize=(13, max(6, n * 0.2)))
    ax[0].imshow(X[i].reshape(28, 28), cmap="gray_r"); ax[0].axis("off")
    ax[0].set_title(f"True: {categories[int(y[i])]}", fontweight="bold")
    colors = ["#ef4444" if k == pred else "#94a3b8" for k in range(n)]
    ax[1].barh(range(n), prob, color=colors)
    ax[1].set_yticks(range(n)); ax[1].set_yticklabels(categories, fontsize=8); ax[1].set_xlim(0, 1)
    ax[1].set_title(f"Predicted: {categories[pred]} ({prob[pred]*100:.1f}%)", fontweight="bold")
    plt.suptitle(f"True: {categories[int(y[i])]} | Predicted: {categories[pred]}", fontweight="bold")
    plt.tight_layout()
    return _fig_to_png()


def multiple_predictions_png(model, categories, data_dir: Path, count: int = 15, seed: int = 1) -> bytes:
    """Lưới nhiều dự đoán (xanh=đúng, đỏ=sai)."""
    X, y = _gather_test(categories, data_dir, per_class=20)
    sel = np.random.default_rng(seed).choice(len(X), size=min(count, len(X)), replace=False)
    prob = model.predict(X[sel], verbose=0)
    pred = prob.argmax(1); conf = prob.max(1)
    cols = 5
    rows = int(np.ceil(len(sel) / cols))
    plt.figure(figsize=(cols * 2.4, rows * 2.7))
    for k, i in enumerate(sel):
        ok = pred[k] == y[i]
        plt.subplot(rows, cols, k + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray_r"); plt.axis("off")
        plt.title(f"T:{categories[int(y[i])]}\nP:{categories[int(pred[k])]} {conf[k]*100:.0f}%",
                  fontsize=8, color=("green" if ok else "crimson"))
    plt.suptitle("Multiple Predictions (xanh=đúng, đỏ=sai)", fontweight="bold")
    plt.tight_layout(h_pad=2.0)
    return _fig_to_png()
