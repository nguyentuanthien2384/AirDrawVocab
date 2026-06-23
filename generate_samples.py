"""
generate_samples.py — Tự sinh các ảnh minh hoạ & dự đoán cho báo cáo.

Tạo ra (giống bộ ảnh mẫu của dự án):
  - sample_drawings.png       : 1 hình mẫu mỗi lớp
  - prediction_sample.png     : 1 hình + biểu đồ xác suất tất cả lớp
  - multiple_predictions.png  : lưới nhiều dự đoán (xanh=đúng, đỏ=sai)

Ví dụ:
    python generate_samples.py --model models/airdrawvocab_best_advanced.keras
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from config import CATEGORIES, NUM_CLASSES, ROOT, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset

try:
    from vocab_pairs import VI_MEANINGS as VI
except Exception:
    VI = {}

OUT_DIR = ROOT / "assets" / "reports" / "samples"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/airdrawvocab_best_advanced.keras")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--multi", type=int, default=15)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    run_dir = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    _, _, x_te, _, _, y_te = split_dataset(X, y, seed=args.seed)
    model = keras.models.load_model(model_path, compile=False)
    prob = model.predict(x_te, verbose=0)
    pred = prob.argmax(1)
    conf = prob.max(1)

    # 1) sample_drawings.png
    cols = 8
    rows = int(np.ceil(NUM_CLASSES / cols))
    plt.figure(figsize=(cols * 1.6, rows * 1.8))
    for c in range(NUM_CLASSES):
        idx = np.where(y_te == c)[0]
        if len(idx) == 0:
            continue
        plt.subplot(rows, cols, c + 1)
        plt.imshow(x_te[idx[0]].reshape(28, 28), cmap="gray_r")
        plt.axis("off")
        label = CATEGORIES[c]
        plt.title(f"{label}\n({VI.get(label, '')})", fontsize=8)
    plt.suptitle(f"Sample Drawings - QuickDraw ({NUM_CLASSES} lớp)", fontsize=15, fontweight="bold")
    plt.tight_layout(h_pad=1.6)
    plt.savefig(run_dir / "sample_drawings.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) prediction_sample.png
    i = int(np.random.default_rng(0).integers(len(x_te)))
    fig, ax = plt.subplots(1, 2, figsize=(13, max(6, NUM_CLASSES * 0.2)))
    ax[0].imshow(x_te[i].reshape(28, 28), cmap="gray_r")
    ax[0].axis("off")
    ax[0].set_title(f"True: {CATEGORIES[int(y_te[i])]}", fontweight="bold")
    colors = ["#ef4444" if k == pred[i] else "#94a3b8" for k in range(NUM_CLASSES)]
    ax[1].barh(range(NUM_CLASSES), prob[i], color=colors)
    ax[1].set_yticks(range(NUM_CLASSES))
    ax[1].set_yticklabels(CATEGORIES, fontsize=8)
    ax[1].set_xlim(0, 1)
    ax[1].set_title(f"Predicted: {CATEGORIES[int(pred[i])]} ({conf[i]*100:.1f}%)", fontweight="bold")
    plt.suptitle(f"True: {CATEGORIES[int(y_te[i])]} | Predicted: {CATEGORIES[int(pred[i])]}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(run_dir / "prediction_sample.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) multiple_predictions.png
    sel = np.random.default_rng(1).choice(len(x_te), size=min(args.multi, len(x_te)), replace=False)
    cols = 5
    rows = int(np.ceil(len(sel) / cols))
    plt.figure(figsize=(cols * 2.4, rows * 2.7))
    for k, i in enumerate(sel):
        ok = pred[i] == y_te[i]
        plt.subplot(rows, cols, k + 1)
        plt.imshow(x_te[i].reshape(28, 28), cmap="gray_r")
        plt.axis("off")
        plt.title(f"T:{CATEGORIES[int(y_te[i])]}\nP:{CATEGORIES[int(pred[i])]} {conf[i]*100:.0f}%",
                  fontsize=8, color=("green" if ok else "crimson"))
    plt.suptitle("Multiple Predictions (xanh=đúng, đỏ=sai)", fontweight="bold")
    plt.tight_layout(h_pad=2.0)
    plt.savefig(run_dir / "multiple_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Đã tự sinh 3 ảnh vào: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
