"""
monitor.py — Theo dõi hiệu năng model trong thực tế (production monitoring).
(Phase 3, Task 18)

Đọc bảng stroke_samples trong SQLite (mỗi lượt chơi backend lưu target/predicted/
confidence/correct/mode/created_at) và tổng hợp:
  - tổng lượt dự đoán, accuracy thực tế (theo cờ correct)
  - confidence trung bình, tỉ lệ low-confidence
  - accuracy theo từng lớp (target) và theo mode
  - tỉ lệ "đoán sai nhưng tự tin cao" (cảnh báo)

Hỗ trợ lọc theo số ngày gần đây (--days).

Ví dụ:
    python src/evaluation/monitor.py
    python src/evaluation/monitor.py --days 7 --out assets/reports/monitor.json
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
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    from config import ROOT
except Exception:
    ROOT = Path(_PROJECT_ROOT)

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"


def fetch_rows(db_path: Path, since_iso: str | None) -> list[dict]:
    if not Path(db_path).exists():
        return []
    q = "SELECT target, predicted, confidence, correct, mode, created_at FROM stroke_samples"
    params: tuple = ()
    if since_iso:
        q += " WHERE created_at >= ?"
        params = (since_iso,)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(q, params)]


def summarize(rows: list[dict], low_conf: float = 0.5) -> dict:
    """Tổng hợp số liệu monitoring. Pure function -> dễ test."""
    n = len(rows)
    if n == 0:
        return {"total": 0}

    correct = sum(1 for r in rows if int(r.get("correct") or 0) == 1)
    confs = [float(r.get("confidence") or 0.0) for r in rows]
    low = sum(1 for c in confs if c < low_conf)
    confident_wrong = sum(
        1 for r in rows
        if int(r.get("correct") or 0) == 0 and float(r.get("confidence") or 0.0) >= 0.8
    )

    per_class: dict[str, dict] = {}
    per_mode: dict[str, dict] = {}
    for r in rows:
        tgt = str(r.get("target") or "").strip().lower() or "(empty)"
        mode = str(r.get("mode") or "").strip().lower() or "(empty)"
        ok = int(r.get("correct") or 0) == 1
        for bucket, key in ((per_class, tgt), (per_mode, mode)):
            d = bucket.setdefault(key, {"total": 0, "correct": 0})
            d["total"] += 1
            d["correct"] += int(ok)

    def with_acc(bucket: dict) -> dict:
        return {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0}
            for k, v in sorted(bucket.items(), key=lambda kv: -kv[1]["total"])
        }

    return {
        "total": n,
        "accuracy": round(correct / n, 4),
        "mean_confidence": round(sum(confs) / n, 4),
        "low_confidence_rate": round(low / n, 4),
        "confident_wrong_rate": round(confident_wrong / n, 4),
        "per_class": with_acc(per_class),
        "per_mode": with_acc(per_mode),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitoring model AirDrawVocab.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--days", type=int, default=0, help="Chỉ lấy N ngày gần nhất (0 = tất cả)")
    ap.add_argument("--low-conf", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="Ghi JSON ra file (tùy chọn)")
    args = ap.parse_args()

    since = None
    if args.days > 0:
        since = (datetime.now() - timedelta(days=args.days)).isoformat()

    rows = fetch_rows(Path(args.db), since)
    report = summarize(rows, low_conf=args.low_conf)
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    report["window_days"] = args.days

    print("===== MODEL MONITORING =====")
    if report["total"] == 0:
        print("Chưa có dữ liệu dự đoán nào.")
    else:
        print(f"Tổng lượt: {report['total']}")
        print(f"Accuracy thực tế: {report['accuracy']*100:.1f}%")
        print(f"Confidence TB: {report['mean_confidence']*100:.1f}%")
        print(f"Low-confidence (<{args.low_conf}): {report['low_confidence_rate']*100:.1f}%")
        print(f"Sai nhưng tự tin cao: {report['confident_wrong_rate']*100:.1f}%")
        print("\nTop lớp (theo số lượt):")
        for k, v in list(report["per_class"].items())[:10]:
            print(f"  {k:>12}: {v['accuracy']*100:5.1f}%  ({v['correct']}/{v['total']})")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi báo cáo: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
