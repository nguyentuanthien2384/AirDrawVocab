"""
data_drift.py — Phát hiện trôi dữ liệu (data drift) cho AirDrawVocab.
(Phase 3, Task 21)

Hai chế độ:
  1) So sánh PHÂN PHỐI LỚP giữa hai mốc dữ liệu (DB stroke chia 2 nửa theo thời
     gian) -> phát hiện người dùng đang vẽ lệch về một số lớp.
  2) So sánh hai manifest dataset (tạo bởi data_versioning.py).

Chỉ số: PSI (Population Stability Index) trên phân phối lớp:
    PSI < 0.1  : ổn định
    0.1–0.25   : thay đổi vừa
    > 0.25     : trôi mạnh (cần chú ý)

Ví dụ:
    python src/evaluation/data_drift.py --db
    python src/evaluation/data_drift.py --baseline a.json --current b.json
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
import json
import math
import sqlite3
from pathlib import Path

try:
    from config import ROOT
except Exception:
    ROOT = Path(_PROJECT_ROOT)

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
_EPS = 1e-6


def _normalize(counts: dict, keys: list) -> list[float]:
    total = sum(counts.get(k, 0) for k in keys)
    if total <= 0:
        return [0.0 for _ in keys]
    return [counts.get(k, 0) / total for k in keys]


def population_stability_index(expected: dict, actual: dict) -> tuple[float, dict]:
    """PSI giữa hai phân phối đếm theo lớp. Pure function -> dễ test.

    Trả về (psi_tổng, psi_theo_lớp).
    """
    keys = sorted(set(expected) | set(actual))
    e = _normalize(expected, keys)
    a = _normalize(actual, keys)
    per_key = {}
    psi = 0.0
    for k, ei, ai in zip(keys, e, a):
        ei = max(ei, _EPS)
        ai = max(ai, _EPS)
        term = (ai - ei) * math.log(ai / ei)
        per_key[k] = round(term, 6)
        psi += term
    return round(psi, 6), per_key


def interpret(psi: float) -> str:
    if psi < 0.1:
        return "ổn định"
    if psi < 0.25:
        return "thay đổi vừa"
    return "TRÔI MẠNH"


def db_target_halves(db_path: Path) -> tuple[dict, dict]:
    """Chia mẫu stroke thành nửa cũ / nửa mới (theo created_at) -> 2 phân phối lớp."""
    if not Path(db_path).exists():
        return {}, {}
    with sqlite3.connect(str(db_path)) as conn:
        rows = [
            str(r[0] or "").strip().lower()
            for r in conn.execute(
                "SELECT target FROM stroke_samples ORDER BY created_at ASC, id ASC"
            )
        ]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return {}, {}
    mid = len(rows) // 2
    old, new = rows[:mid], rows[mid:]

    def to_counts(items):
        c: dict = {}
        for it in items:
            c[it] = c.get(it, 0) + 1
        return c

    return to_counts(old), to_counts(new)


def counts_from_manifest(path: Path) -> dict:
    """Lấy phân phối lớp từ manifest data_versioning (quickdraw hoặc stroke)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    qd = data.get("quickdraw_npy", {}).get("classes")
    if qd:
        return {k: int(v.get("samples", 0)) for k, v in qd.items() if isinstance(v, dict)}
    strokes = data.get("stroke_db", {}).get("per_target")
    if strokes:
        return {k: int(v) for k, v in strokes.items()}
    # manifest dạng phẳng {label: count}
    return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phát hiện data drift cho AirDrawVocab.")
    ap.add_argument("--db", action="store_true", help="So sánh 2 nửa thời gian của stroke DB")
    ap.add_argument("--db-path", default=str(DB_PATH))
    ap.add_argument("--baseline", default=None, help="Manifest baseline (.json)")
    ap.add_argument("--current", default=None, help="Manifest hiện tại (.json)")
    args = ap.parse_args()

    if args.baseline and args.current:
        expected = counts_from_manifest(Path(args.baseline))
        actual = counts_from_manifest(Path(args.current))
        title = "manifest baseline vs current"
    else:
        expected, actual = db_target_halves(Path(args.db_path))
        title = "stroke DB: nửa cũ vs nửa mới"

    if not expected or not actual:
        print("Không đủ dữ liệu để tính drift (cần >=2 mẫu hoặc 2 manifest hợp lệ).")
        return 1

    psi, per_key = population_stability_index(expected, actual)
    print(f"===== DATA DRIFT ({title}) =====")
    print(f"PSI tổng: {psi}  -> {interpret(psi)}")
    top = sorted(per_key.items(), key=lambda kv: -kv[1])[:10]
    print("Đóng góp PSI theo lớp (cao nhất):")
    for k, v in top:
        print(f"  {k:>12}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
