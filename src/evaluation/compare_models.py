"""
compare_models.py — So sánh tự động nhiều model AirDrawVocab trên cùng một tập
test cố định, rồi xuất bảng so sánh (console + CSV) và (tùy chọn) log vào MLflow.
(Phase 1, Task 7)

Mặc định script sẽ quét các file *.keras trong thư mục models/ (gồm cả model
deploy và các member_*.keras), đánh giá từng cái và xếp hạng theo accuracy.

Ví dụ:
    # So sánh tất cả model trong models/
    python compare_models.py

    # So sánh một danh sách model cụ thể
    python compare_models.py --models models/airdrawvocab_best_advanced.keras models/member_resnet_gn.keras

    # Gồm cả các bản version trong models/versions/
    python compare_models.py --include-versions
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from tensorflow import keras

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import ROOT, MODELS_DIR, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset
from src.utils.mlflow_utils import start_mlflow_run, log_metrics, end_mlflow_run

REPORTS_DIR = ROOT / "assets" / "reports" / "comparisons"


def discover_models(include_versions: bool) -> list[Path]:
    paths = sorted(MODELS_DIR.glob("*.keras"))
    if include_versions:
        paths += sorted((MODELS_DIR / "versions").glob("*.keras"))
    # Bỏ trùng lặp, giữ thứ tự
    seen, unique = set(), []
    for p in paths:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def evaluate_model(model_path: Path, x_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"  [BỎ QUA] {model_path.name}: không load được ({exc})")
        return None

    try:
        t0 = time.time()
        probs = model.predict(x_test, batch_size=256, verbose=0)
        infer_s = time.time() - t0
        y_pred = np.argmax(probs, axis=1)
        top3 = float(np.mean([t in row for t, row in zip(y_test, np.argsort(probs, axis=1)[:, -3:])]))
        return {
            "model": model_path.name,
            "path": str(model_path),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "top3_accuracy": top3,
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "params": int(model.count_params()),
            "infer_seconds": round(infer_s, 2),
        }
    except Exception as exc:
        print(f"  [BỎ QUA] {model_path.name}: lỗi khi đánh giá ({exc})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="So sánh nhiều model AirDrawVocab.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Danh sách file model cụ thể. Mặc định: quét models/*.keras")
    ap.add_argument("--include-versions", action="store_true",
                    help="Gồm cả các bản trong models/versions/")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--benchmark", default=None,
                    help="Đường dẫn .npz benchmark (tạo bằng make_benchmark.py). "
                         "Nếu có sẽ dùng thay cho split mặc định.")
    ap.add_argument("--no-mlflow", action="store_true", help="Không log vào MLflow")
    args = ap.parse_args()

    if args.models:
        model_paths = []
        for m in args.models:
            p = Path(m)
            if not p.is_absolute():
                p = ROOT / p
            model_paths.append(p)
    else:
        model_paths = discover_models(args.include_versions)

    model_paths = [p for p in model_paths if p.exists()]
    if not model_paths:
        print("Không tìm thấy model nào để so sánh.")
        return 1

    print(f"Sẽ so sánh {len(model_paths)} model.")
    if args.benchmark:
        bench_path = Path(args.benchmark)
        if not bench_path.is_absolute():
            bench_path = ROOT / bench_path
        print(f"Nạp benchmark cố định: {bench_path}")
        data = np.load(bench_path, allow_pickle=True)
        x_test = data["x"].reshape(-1, 28, 28, 1).astype("float32")
        y_test = data["y"].astype("int64")
    else:
        print(f"Nạp dữ liệu (split cố định, seed={args.seed})...")
        X, y = load_dataset()
        _, _, x_test, _, _, y_test = split_dataset(X, y, seed=args.seed)
        x_test = x_test.reshape(-1, 28, 28, 1)
    print(f"Test samples: {len(y_test)}\n")

    results = []
    for p in model_paths:
        print(f"==> Đánh giá {p.name} ...")
        r = evaluate_model(p, x_test, y_test)
        if r:
            results.append(r)
            print(f"    acc={r['accuracy']*100:.2f}%  top3={r['top3_accuracy']*100:.2f}%  "
                  f"f1={r['f1_weighted']*100:.2f}%  params={r['params']:,}")

    if not results:
        print("Không có model nào đánh giá thành công.")
        return 1

    results.sort(key=lambda d: d["accuracy"], reverse=True)

    # In bảng xếp hạng
    print("\n" + "=" * 78)
    print(f"{'#':<3}{'Model':<42}{'Acc':>8}{'Top3':>8}{'F1':>8}")
    print("-" * 78)
    for i, r in enumerate(results, 1):
        print(f"{i:<3}{r['model']:<42}{r['accuracy']*100:>7.2f}%"
              f"{r['top3_accuracy']*100:>7.2f}%{r['f1_weighted']*100:>7.2f}%")
    print("=" * 78)
    print(f"Model tốt nhất: {results[0]['model']} ({results[0]['accuracy']*100:.2f}%)")

    # Lưu CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / f"model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nĐã lưu bảng so sánh: {out_csv}")

    # Log MLflow (tùy chọn)
    if not args.no_mlflow:
        start_mlflow_run(experiment_name="AirDrawVocab_Comparison",
                         run_name=f"compare_{len(results)}models",
                         tags={"task": "model_comparison"})
        for r in results:
            stem = Path(r["model"]).stem.replace(".", "_")
            log_metrics({f"{stem}__accuracy": r["accuracy"],
                         f"{stem}__top3": r["top3_accuracy"],
                         f"{stem}__f1": r["f1_weighted"]})
        end_mlflow_run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
