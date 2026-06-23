"""
data_versioning.py — Ghi "manifest" mô tả phiên bản dữ liệu cho AirDrawVocab.
(Phase 2, Task 9 — bản nhẹ, không cần DVC)

Quét dataset QuickDraw (.npy) và database stroke (SQLite), tính:
- số mẫu / lớp, shape, dtype, kích thước file, mã băm SHA1 (để phát hiện thay đổi)
- tổng quan dataset (tổng mẫu, số lớp, tổng dung lượng)
- thống kê stroke_samples trong SQLite

Kết quả ghi ra data/dataset_manifest.json (mặc định). So sánh 2 manifest theo
thời gian để biết dữ liệu đã thay đổi ra sao.

Ví dụ:
    python src/data/data_versioning.py
    python src/data/data_versioning.py --no-hash   # bỏ qua SHA1 cho nhanh
"""
from __future__ import annotations

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

# Tránh lỗi UnicodeEncodeError khi in tiếng Việt trên console Windows (cp1252).
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ROOT, DATA_DIR, CATEGORIES


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan_npy(data_dir: Path, do_hash: bool) -> dict:
    classes = {}
    total_samples = 0
    total_bytes = 0
    for cat in CATEGORIES:
        path = data_dir / f"{cat}.npy"
        if not path.exists():
            classes[cat] = {"present": False}
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            n = int(arr.shape[0])
            shape = list(arr.shape)
            dtype = str(arr.dtype)
        except Exception as exc:
            classes[cat] = {"present": True, "error": str(exc)}
            continue
        size = path.stat().st_size
        total_samples += n
        total_bytes += size
        entry = {
            "present": True,
            "samples": n,
            "shape": shape,
            "dtype": dtype,
            "size_bytes": size,
        }
        if do_hash:
            entry["sha1"] = sha1_of_file(path)
        classes[cat] = entry
    present = [c for c, v in classes.items() if v.get("present") and "samples" in v]
    return {
        "num_classes_expected": len(CATEGORIES),
        "num_classes_present": len(present),
        "total_samples": total_samples,
        "total_size_mb": round(total_bytes / 1024 / 1024, 2),
        "classes": classes,
    }


def scan_stroke_db(db_path: Path) -> dict:
    if not db_path.exists():
        return {"present": False}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) AS n FROM stroke_samples").fetchone()["n"]
            per_target = {}
            for r in conn.execute(
                "SELECT target, COUNT(*) AS n FROM stroke_samples GROUP BY target ORDER BY n DESC"
            ):
                key = str(r["target"] or "").strip().lower() or "(empty)"
                per_target[key] = int(r["n"])
        return {
            "present": True,
            "total_stroke_samples": int(total),
            "distinct_targets": len(per_target),
            "per_target": per_target,
        }
    except Exception as exc:
        return {"present": True, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tạo manifest phiên bản dữ liệu AirDrawVocab.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--db", default=str(ROOT / "data" / "airdrawvocab_app.sqlite3"))
    ap.add_argument("--out", default=str(ROOT / "data" / "dataset_manifest.json"))
    ap.add_argument("--no-hash", action="store_true", help="Bỏ qua SHA1 (nhanh hơn)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Quét dataset .npy tại: {data_dir}")
    npy = scan_npy(data_dir, do_hash=not args.no_hash)
    print(f"Quét stroke DB tại: {args.db}")
    strokes = scan_stroke_db(Path(args.db))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "with_hash": not args.no_hash,
        "quickdraw_npy": npy,
        "stroke_db": strokes,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== TÓM TẮT DATASET =====")
    print(f"QuickDraw: {npy['num_classes_present']}/{npy['num_classes_expected']} lớp, "
          f"{npy['total_samples']:,} mẫu, {npy['total_size_mb']} MB")
    if strokes.get("present"):
        print(f"Stroke DB: {strokes.get('total_stroke_samples', 0):,} mẫu, "
              f"{strokes.get('distinct_targets', 0)} target")
    else:
        print("Stroke DB: (không tìm thấy)")
    print(f"Đã ghi manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
