"""Build a fixed real-user benchmark from AirDrawVocab SQLite strokes.

Outputs a release folder containing train/calibration/test JSONL and a manifest.
This script intentionally stores strokes, metadata and labels, not raw webcam frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stroke_features import strokes_to_sequence, count_active_points

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
DEFAULT_OUT = ROOT / "data" / "benchmark" / "release_v1"


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _row_to_item(row: sqlite3.Row, min_points: int) -> Dict[str, Any] | None:
    target = str(row["target"] or "").strip().lower()
    if not target:
        return None
    strokes = _safe_json_loads(row["strokes_json"], [])
    seq = strokes_to_sequence(strokes)
    active_points = count_active_points(seq)
    if active_points < min_points:
        return None
    judge = _safe_json_loads(row["judge_json"] if "judge_json" in row.keys() else "{}", {})
    return {
        "id": int(row["id"]),
        "target": target,
        "predicted": str(row["predicted"] or ""),
        "confidence": float(row["confidence"] or 0.0),
        "correct": bool(row["correct"]),
        "mode": str(row["mode"] or "mouse"),
        "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
        "created_at": str(row["created_at"] or ""),
        "point_count": int(row["point_count"] or active_points),
        "active_points": active_points,
        "judge": judge,
        "strokes": strokes,
    }


def load_items(db_path: Path, min_points: int) -> List[Dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    cols: set[str]
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(stroke_samples)")}
        select_cols = [
            "id", "user_id", "target", "predicted", "confidence", "correct",
            "mode", "strokes_json", "created_at",
        ]
        if "judge_json" in cols:
            select_cols.append("judge_json")
        if "point_count" in cols:
            select_cols.append("point_count")
        else:
            select_cols.append("0 AS point_count")
        rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM stroke_samples ORDER BY id ASC").fetchall()
    items = []
    for row in rows:
        item = _row_to_item(row, min_points)
        if item is not None:
            items.append(item)
    return items


def stratified_split(items: List[Dict[str, Any]], seed: int, train_ratio: float, calibration_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_label[item["target"]].append(item)
    splits = {"train": [], "calibration": [], "test": []}
    for label, rows in sorted(by_label.items()):
        # Prefer grouping by user_id when possible: deterministic sort by user then shuffle user groups.
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            group = f"u:{row.get('user_id')}" if row.get("user_id") is not None else f"sample:{row['id']}"
            groups[group].append(row)
        group_values = list(groups.values())
        rng.shuffle(group_values)
        flat: list[dict] = []
        for group_rows in group_values:
            rng.shuffle(group_rows)
            flat.extend(group_rows)
        n = len(flat)
        n_train = int(round(n * train_ratio))
        n_cal = int(round(n * calibration_ratio))
        # Guarantee at least one test item per class when possible.
        if n >= 3 and n_train + n_cal >= n:
            n_train = max(1, n_train - 1)
        if n >= 5 and n_cal == 0:
            n_cal = 1
            n_train = max(1, n_train - 1)
        splits["train"].extend(flat[:n_train])
        splits["calibration"].extend(flat[n_train:n_train + n_cal])
        splits["test"].extend(flat[n_train + n_cal:])
    for split_rows in splits.values():
        split_rows.sort(key=lambda x: x["id"])
    return splits


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(out_dir: Path, splits: Dict[str, List[Dict[str, Any]]], seed: int, min_points: int) -> dict:
    all_items = [item for split in splits.values() for item in split]
    per_label = Counter(item["target"] for item in all_items)
    per_mode = Counter(item.get("mode", "mouse") for item in all_items)
    split_counts = {name: len(rows) for name, rows in splits.items()}
    files = {}
    for name in splits:
        path = out_dir / f"{name}.jsonl"
        if path.exists():
            files[name] = {"file": path.name, "sha1": sha1_file(path)}
    min_class = min(per_label.values(), default=0)
    data_warning = None
    if min_class < 20:
        data_warning = "Benchmark còn ít mẫu mỗi lớp; chỉ dùng để smoke test, chưa nên coi là chuẩn production."
    return {
        "benchmark_name": out_dir.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "min_points": min_points,
        "num_samples": len(all_items),
        "num_classes": len(per_label),
        "min_samples_per_class": min_class,
        "split_counts": split_counts,
        "per_label": dict(sorted(per_label.items())),
        "per_mode": dict(sorted(per_mode.items())),
        "files": files,
        "privacy": {
            "raw_camera_frames_saved": False,
            "contains_derived_strokes": True,
        },
        "data_warning": data_warning,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    args = parser.parse_args()

    items = load_items(args.db, min_points=args.min_points)
    if not items:
        raise RuntimeError("No usable stroke samples. Save training samples first.")
    splits = stratified_split(items, args.seed, args.train_ratio, args.calibration_ratio)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)
    manifest = build_manifest(args.out, splits, args.seed, args.min_points)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
