"""
hard_negative_mining.py — Khai thác "hard negatives" cho AirDrawVocab.
(Phase 2, Task 13)

Chạy một model trên tập test (split cố định hoặc benchmark) và xuất ra:
  - Các mẫu model ĐOÁN SAI với độ tự tin cao nhất (hard negatives) -> CSV
  - Các CẶP LỚP HAY NHẦM nhất (vd leaf <-> diamond) từ confusion matrix
  - Các LỚP KÉM nhất (recall thấp) -> gợi ý nên bổ sung dữ liệu

Kết quả giúp định hướng cải thiện model: thu thập thêm dữ liệu cho lớp/cặp khó,
hoặc oversample khi train lại.

Ví dụ:
    python src/evaluation/hard_negative_mining.py
    python src/evaluation/hard_negative_mining.py --model models/airdrawvocab_best_advanced.keras --top 150
    python src/evaluation/hard_negative_mining.py --benchmark data/benchmark/benchmark_test.npz --tta 6
"""
from __future__ import annotations

# --- bootstrap ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

from config import ROOT, CATEGORIES, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset

OUT_DIR = ROOT / "assets" / "reports" / "hard_negative"


def load_test_set(benchmark: str | None, seed: int):
    if benchmark:
        p = Path(benchmark)
        if not p.is_absolute():
            p = ROOT / p
        data = np.load(p, allow_pickle=True)
        return data["x"].reshape(-1, 28, 28, 1).astype("float32"), data["y"].astype("int64")
    X, y = load_dataset()
    _, _, x_test, _, _, y_test = split_dataset(X, y, seed=seed)
    return x_test.reshape(-1, 28, 28, 1).astype("float32"), y_test.astype("int64")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hard negative mining cho AirDrawVocab.")
    ap.add_argument("--model", default="models/airdrawvocab_best_advanced.keras")
    ap.add_argument("--benchmark", default=None, help="Đường dẫn .npz benchmark (tùy chọn)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--top", type=int, default=100, help="Số hard negative xuất ra")
    ap.add_argument("--pairs", type=int, default=20, help="Số cặp lớp hay nhầm xuất ra")
    ap.add_argument("--tta", type=int, default=0, help="Số view TTA (0 = tắt)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        print(f"Không tìm thấy model: {model_path}")
        return 1

    print("Nạp tập test...")
    x_test, y_test = load_test_set(args.benchmark, args.seed)
    print(f"Test samples: {len(y_test)}")

    model = keras.models.load_model(model_path, compile=False)

    print("Dự đoán...")
    if args.tta:
        from airdraw_models import predict_tta
        probs = predict_tta(model, x_test, n=args.tta)
    else:
        probs = model.predict(x_test, batch_size=256, verbose=0)
    pred = probs.argmax(1)
    conf = probs.max(1)
    acc = float((pred == y_test).mean())
    print(f"Accuracy: {acc*100:.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # 1) Hard negatives: sai + confidence cao nhất
    wrong = np.where(pred != y_test)[0]
    wrong_sorted = wrong[np.argsort(-conf[wrong])]
    hard = wrong_sorted[:args.top]
    hn_csv = OUT_DIR / f"hard_negatives_{stamp}.csv"
    with hn_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "true_label", "predicted_label", "confidence", "true_prob"])
        for i in hard:
            w.writerow([int(i), CATEGORIES[int(y_test[i])], CATEGORIES[int(pred[i])],
                        f"{conf[i]:.4f}", f"{probs[i][y_test[i]]:.4f}"])

    # 2) Cặp lớp hay nhầm (off-diagonal confusion)
    n = len(CATEGORIES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_test, pred):
        cm[t, p] += 1
    pairs = []
    for t in range(n):
        for p in range(n):
            if t != p and cm[t, p] > 0:
                pairs.append((CATEGORIES[t], CATEGORIES[p], int(cm[t, p])))
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:args.pairs]

    # 3) Lớp kém nhất (recall thấp)
    per_class = []
    for c in range(n):
        mask = y_test == c
        total = int(mask.sum())
        correct = int((pred[mask] == c).sum()) if total else 0
        recall = (correct / total) if total else 0.0
        per_class.append({"label": CATEGORIES[c], "recall": round(recall, 4),
                          "support": total, "errors": total - correct})
    worst = sorted(per_class, key=lambda d: d["recall"])[:10]

    summary = {
        "generated_at": stamp,
        "model": model_path.name,
        "tta": args.tta,
        "accuracy": acc,
        "num_wrong": int(len(wrong)),
        "most_confused_pairs": [
            {"true": a, "predicted": b, "count": c} for a, b, c in top_pairs
        ],
        "worst_classes": worst,
        "hard_negatives_csv": str(hn_csv),
    }
    summary_path = OUT_DIR / f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== CẶP LỚP HAY NHẦM NHẤT ===")
    for a, b, c in top_pairs[:10]:
        print(f"  {a:>12} -> {b:<12} : {c}")
    print("\n=== LỚP KÉM NHẤT (recall thấp) ===")
    for d in worst[:10]:
        print(f"  {d['label']:>12} : recall={d['recall']*100:5.1f}%  (lỗi {d['errors']}/{d['support']})")
    print(f"\nĐã lưu: {hn_csv}")
    print(f"Đã lưu: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
