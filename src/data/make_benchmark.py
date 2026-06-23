"""
make_benchmark.py — Tạo BỘ TEST CHUẨN (benchmark) cố định cho AirDrawVocab.
(Phase 2, Task 14)

Mục tiêu: đóng băng một tập test chất lượng, có version, để MỌI model được so
sánh trên CÙNG dữ liệu (không bị lệch do random split khác nhau).

Tập test được lấy từ split cố định (theo seed trong config) rồi lưu thành một
file .npz duy nhất kèm manifest (counts + SHA1) để tái lập và truy vết.

Ví dụ:
    python src/data/make_benchmark.py
    python src/data/make_benchmark.py --out data/benchmark/benchmark_test.npz
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
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ROOT, CATEGORIES, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Tạo benchmark test set cố định.")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmark" / "benchmark_test.npz"))
    args = ap.parse_args()

    print("Nạp dữ liệu QuickDraw...")
    X, y = load_dataset()
    _, _, x_test, _, _, y_test = split_dataset(X, y, seed=args.seed)
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")
    y_test = y_test.astype("int64")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x_test,
        y=y_test,
        categories=np.array(CATEGORIES),
        seed=np.array(args.seed),
    )

    sha1 = hashlib.sha1(out_path.read_bytes()).hexdigest()
    per_class = {CATEGORIES[i]: int((y_test == i).sum()) for i in range(len(CATEGORIES))}
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file": out_path.name,
        "seed": args.seed,
        "num_samples": int(len(y_test)),
        "num_classes": len(CATEGORIES),
        "image_shape": [28, 28, 1],
        "sha1": sha1,
        "per_class": per_class,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== BENCHMARK TEST SET =====")
    print(f"Mẫu: {len(y_test):,} | Lớp: {len(CATEGORIES)} | seed={args.seed}")
    print(f"SHA1: {sha1}")
    print(f"Đã lưu: {out_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
