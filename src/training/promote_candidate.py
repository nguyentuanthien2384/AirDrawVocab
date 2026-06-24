"""Promotion gate for AirDrawVocab candidate models."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import MODELS_DIR, MODEL_PATH

PROMOTION_LOG = MODELS_DIR / "promotion_log.jsonl"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gate_report(report: dict, allow_if_weak_data: bool = False) -> tuple[bool, list[str]]:
    reasons = []
    samples = int(report.get("samples") or report.get("metrics", {}).get("val_samples") or 0)
    top1 = float(report.get("top1_accuracy") or report.get("metrics", {}).get("val_accuracy") or 0.0)
    macro_f1 = report.get("macro_f1") or report.get("metrics", {}).get("val_macro_f1")
    macro_f1 = float(macro_f1) if macro_f1 is not None else None
    if samples < 30 and not allow_if_weak_data:
        reasons.append("Benchmark/eval samples < 30. Thêm --allow-if-weak-data nếu chỉ prototype.")
    if top1 < 0.50 and not allow_if_weak_data:
        reasons.append(f"Top-1 accuracy quá thấp: {top1:.3f}")
    if macro_f1 is not None and macro_f1 < 0.45 and not allow_if_weak_data:
        reasons.append(f"Macro F1 quá thấp: {macro_f1:.3f}")
    return len(reasons) == 0, reasons


def copy_if_exists(src: Path | None, dst: Path | None) -> bool:
    if not src or not dst or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup = dst.with_suffix(dst.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if dst.exists():
        shutil.copy2(dst, backup)
    shutil.copy2(src, dst)
    return True


def append_log(payload: dict) -> None:
    PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROMOTION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "assets" / "reports" / "releases" / "current" / "summary.json")
    parser.add_argument("--candidate-image", type=Path, default=MODELS_DIR / "image_cnn_candidate.keras")
    parser.add_argument("--candidate-categories", type=Path, default=MODELS_DIR / "categories_candidate.json")
    parser.add_argument("--deploy-image", type=Path, default=MODELS_DIR / "airdrawvocab_best_advanced.keras")
    parser.add_argument("--deploy-categories", type=Path, default=MODELS_DIR / "categories.json")
    parser.add_argument("--allow-if-weak-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = load_json(args.report)
    ok, reasons = gate_report(report, allow_if_weak_data=args.allow_if_weak_data)
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report": str(args.report),
        "candidate_image": str(args.candidate_image),
        "candidate_categories": str(args.candidate_categories),
        "deploy_image": str(args.deploy_image),
        "deploy_categories": str(args.deploy_categories),
        "approved": ok,
        "dry_run": args.dry_run,
        "reasons": reasons,
        "metrics": {k: report.get(k) for k in ["top1_accuracy", "top3_accuracy", "macro_f1", "ece_10_bins", "samples"]},
    }
    if ok and not args.dry_run:
        payload["model_copied"] = copy_if_exists(args.candidate_image, args.deploy_image)
        payload["categories_copied"] = copy_if_exists(args.candidate_categories, args.deploy_categories)
    append_log(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
