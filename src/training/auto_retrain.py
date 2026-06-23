"""
auto_retrain.py — Pipeline tự động train lại khi có đủ dữ liệu mới.
(Phase 3, Task 16)

Kiểm tra số mẫu stroke người dùng đã thu thập trong SQLite. Nếu số mẫu MỚI (so
với lần train trước, lưu ở data/auto_retrain_state.json) >= ngưỡng, sẽ kích hoạt
train lại (gọi train_stroke_model.py hoặc self_improve_retrain.py), rồi cập nhật
registry.

Quyết định retrain (decide_retrain) tách riêng để dễ kiểm thử.

Ví dụ:
    python src/training/auto_retrain.py --mode stroke --min-new-samples 20 --epochs 30
    python src/training/auto_retrain.py --dry-run        # chỉ kiểm tra, không train
"""
from __future__ import annotations

from pathlib import Path

# parents[2] -> gốc dự án (file nằm trong src/training/)
ROOT = Path(__file__).resolve().parents[2]

import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
STATE_PATH = ROOT / "data" / "auto_retrain_state.json"
SCRIPTS = {
    "stroke": ROOT / "src" / "training" / "train_stroke_model.py",
    "image": ROOT / "src" / "training" / "self_improve_retrain.py",
}


def read_stroke_count(db_path: Path) -> int:
    """Đếm tổng số mẫu stroke trong DB (0 nếu không có DB/bảng)."""
    if not Path(db_path).exists():
        return 0
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM stroke_samples").fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def read_state(state_path: Path) -> dict:
    try:
        return json.loads(Path(state_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(state_path: Path, state: dict) -> None:
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def decide_retrain(current_count: int, last_count: int, min_new: int) -> tuple[bool, int]:
    """Trả về (có_nên_train, số_mẫu_mới). Pure function -> dễ test."""
    new_samples = max(0, current_count - last_count)
    return (new_samples >= min_new and min_new > 0), new_samples


def main() -> int:
    ap = argparse.ArgumentParser(description="Tự động train lại khi đủ dữ liệu mới.")
    ap.add_argument("--mode", choices=["stroke", "image"], default="stroke")
    ap.add_argument("--min-new-samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra, không train")
    args = ap.parse_args()

    db_path = Path(args.db)
    state = read_state(STATE_PATH)
    last_count = int(state.get(f"last_count_{args.mode}", 0))
    current = read_stroke_count(db_path)
    should, new_samples = decide_retrain(current, last_count, args.min_new_samples)

    print(f"[auto_retrain] mode={args.mode} tổng={current} lần trước={last_count} "
          f"mới={new_samples} ngưỡng={args.min_new_samples}")

    if not should:
        print("[auto_retrain] Chưa đủ dữ liệu mới -> bỏ qua.")
        return 0

    if args.dry_run:
        print("[auto_retrain] DRY-RUN: đủ điều kiện train (không thực thi).")
        return 0

    script = SCRIPTS[args.mode]
    cmd = [_sys.executable, str(script), "--epochs", str(args.epochs)]
    print(f"[auto_retrain] Bắt đầu train: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[auto_retrain] Train thất bại (exit {result.returncode}).")
        return result.returncode

    # Cập nhật state + rebuild registry
    state[f"last_count_{args.mode}"] = current
    state[f"last_trained_at_{args.mode}"] = datetime.now().isoformat(timespec="seconds")
    write_state(STATE_PATH, state)
    try:
        from src.utils.model_registry import rebuild
        reg = rebuild()
        print(f"[auto_retrain] Registry cập nhật: {reg['count']} version.")
    except Exception as exc:
        print(f"[auto_retrain] Bỏ qua cập nhật registry: {exc}")

    print("[auto_retrain] Hoàn tất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
