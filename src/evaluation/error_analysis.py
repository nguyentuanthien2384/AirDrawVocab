"""
error_analysis.py — Phân tích lỗi (định tính) cho mô hình AirDrawVocab.

Tạo ra các sản phẩm phục vụ mục "Đánh giá giải pháp / Phân tích lỗi" trong báo cáo:
  - Bảng các LỚP KHÓ NHẤT (F1 thấp nhất)
  - Các CẶP HAY NHẦM NHẤT (true -> pred) từ confusion matrix
  - Lưới ảnh các mẫu model ĐOÁN SAI (lỗi "tự tin" nhất)
  - File CSV liệt kê toàn bộ mẫu sai

Ví dụ:
    python error_analysis.py --model models/airdrawvocab_best_advanced.keras
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from config import CATEGORIES, NUM_CLASSES, ROOT, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset

OUT_DIR = ROOT / "assets" / "reports" / "error_analysis"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phân tích lỗi cho model AirDrawVocab.")
    ap.add_argument("--model", default="models/airdrawvocab_best_advanced.keras")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--hardest", type=int, default=10)
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--examples", type=int, default=16)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Không thấy model: {model_path}")

    run_dir = OUT_DIR / f"{model_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Nạp dữ liệu test (cùng split cố định với train/evaluate)...")
    X, y = load_dataset()
    _, _, x_te, _, _, y_te = split_dataset(X, y, seed=args.seed)

    model = keras.models.load_model(model_path, compile=False)
    prob = model.predict(x_te, verbose=0)
    pred = prob.argmax(1)
    conf = prob.max(1)

    acc = float((pred == y_te).mean())
    print(f"\nAccuracy: {acc*100:.2f}%  trên {len(y_te)} mẫu test")

    # 1) Lớp khó nhất theo F1
    rep = classification_report(y_te, pred, target_names=CATEGORIES,
                                output_dict=True, zero_division=0)
    hard = sorted([(c, rep[c]["precision"], rep[c]["recall"], rep[c]["f1-score"])
                   for c in CATEGORIES], key=lambda r: r[3])
    print(f"\n--- {args.hardest} LỚP KHÓ NHẤT (F1 thấp) ---")
    lines = ["class,precision,recall,f1"]
    for c, p, r, f in hard[:args.hardest]:
        print(f"  {c:12s} P={p:.3f} R={r:.3f} F1={f:.3f}")
    for c in CATEGORIES:
        lines.append(f"{c},{rep[c]['precision']:.4f},{rep[c]['recall']:.4f},{rep[c]['f1-score']:.4f}")
    (run_dir / "per_class_metrics.csv").write_text("\n".join(lines), encoding="utf-8")

    # 2) Cặp hay nhầm nhất
    cm = confusion_matrix(y_te, pred, labels=range(NUM_CLASSES))
    pairs = sorted([(cm[i, j], CATEGORIES[i], CATEGORIES[j])
                    for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
                    if i != j and cm[i, j] > 0], reverse=True)
    print(f"\n--- {args.pairs} CẶP HAY NHẦM NHẤT (thực -> đoán) ---")
    for n, a, b in pairs[:args.pairs]:
        print(f"  {a:12s} -> {b:12s} : {n} lần")

    # 3) Lưới ảnh đoán sai (tự tin nhất)
    wrong = np.where(pred != y_te)[0]
    print(f"\nTổng mẫu sai: {len(wrong)}/{len(y_te)} ({len(wrong)/len(y_te)*100:.2f}%)")
    sel = wrong[np.argsort(-conf[wrong])][:args.examples]
    cols = 4
    rows = int(np.ceil(len(sel) / cols)) or 1
    plt.figure(figsize=(3 * cols, 3 * rows))
    for k, i in enumerate(sel):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(x_te[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"thực:{CATEGORIES[y_te[i]]}\nđoán:{CATEGORIES[pred[i]]} {conf[i]*100:.0f}%",
                  fontsize=8, color="crimson")
    plt.suptitle("Các mẫu model ĐOÁN SAI (tự tin nhất)")
    plt.tight_layout()
    plt.savefig(run_dir / "misclassified_examples.png", dpi=120)
    plt.close()

    # 4) CSV toàn bộ mẫu sai
    with open(run_dir / "misclassified.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "true", "pred", "confidence"])
        for i in wrong:
            w.writerow([int(i), CATEGORIES[int(y_te[i])], CATEGORIES[int(pred[i])], round(float(conf[i]), 4)])

    print(f"\nĐã lưu kết quả phân tích lỗi vào: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
