"""
model_registry.py — Model Registry nhẹ cho AirDrawVocab.
(Phase 3, Task 17)

Quản lý các bản model đã lưu version (bởi model_versioning.save_versioned_model)
trong models/versions/, tổng hợp thành models/registry.json, và hỗ trợ "promote"
(đưa một version lên làm model deploy chính).

CLI:
    python src/utils/model_registry.py rebuild
    python src/utils/model_registry.py list [--base stroke_bigru]
    python src/utils/model_registry.py best --metric val_accuracy
    python src/utils/model_registry.py promote --version v20260622_141530
    python src/utils/model_registry.py promote --path models/versions/xxx.keras
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
import shutil
from datetime import datetime
from pathlib import Path

try:
    from config import MODELS_DIR, MODEL_PATH
except Exception:
    MODELS_DIR = Path(_PROJECT_ROOT) / "models"
    MODEL_PATH = MODELS_DIR / "airdrawvocab_best_advanced.keras"

from src.utils.model_versioning import list_versions, VERSIONS_DIR

REGISTRY_PATH = MODELS_DIR / "registry.json"


def build_registry(versions_dir: Path | None = None) -> dict:
    """Quét metadata các version -> dict registry (chưa ghi file)."""
    entries = list_versions(versions_dir=versions_dir)
    registry = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(entries),
        "models": entries,
    }
    return registry


def save_registry(registry: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def rebuild(versions_dir: Path | None = None, path: Path | None = None) -> dict:
    reg = build_registry(versions_dir)
    save_registry(reg, path)
    return reg


def find_best(metric: str, base_name: str | None = None,
              versions_dir: Path | None = None) -> dict | None:
    """Trả về entry có `metric` cao nhất (đọc trong metadata['metrics'])."""
    best, best_val = None, float("-inf")
    for entry in list_versions(base_name=base_name, versions_dir=versions_dir):
        val = (entry.get("metrics") or {}).get(metric)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if val > best_val:
            best, best_val = entry, val
    return best


def resolve_version_path(version: str, versions_dir: Path | None = None) -> Path | None:
    """Tìm file .keras theo tag version (vd v20260622_141530)."""
    vdir = Path(versions_dir) if versions_dir else VERSIONS_DIR
    for entry in list_versions(versions_dir=versions_dir):
        if entry.get("version") == version:
            return vdir / entry.get("model_file", "")
    # fallback: khớp theo tên file chứa version
    matches = sorted(vdir.glob(f"*{version}*.keras"))
    return matches[0] if matches else None


def promote(model_file: Path, deploy_path: Path | None = None) -> Path:
    """Copy một file model version -> vị trí deploy chính."""
    model_file = Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_file}")
    deploy_path = Path(deploy_path) if deploy_path else Path(MODEL_PATH)
    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_file, deploy_path)
    return deploy_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Model Registry cho AirDrawVocab.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("rebuild", help="Quét versions -> ghi registry.json")

    p_list = sub.add_parser("list", help="Liệt kê các version")
    p_list.add_argument("--base", default=None)

    p_best = sub.add_parser("best", help="Tìm version tốt nhất theo metric")
    p_best.add_argument("--metric", default="val_accuracy")
    p_best.add_argument("--base", default=None)

    p_prom = sub.add_parser("promote", help="Đưa một version lên làm model deploy")
    p_prom.add_argument("--version", default=None)
    p_prom.add_argument("--path", default=None)
    p_prom.add_argument("--deploy-path", default=None)

    args = ap.parse_args()

    if args.cmd == "rebuild":
        reg = rebuild()
        print(f"Đã ghi registry: {REGISTRY_PATH} ({reg['count']} version)")
        return 0

    if args.cmd == "list":
        entries = list_versions(base_name=args.base)
        if not entries:
            print("Chưa có version nào trong models/versions/.")
            return 0
        for e in entries:
            metrics = e.get("metrics") or {}
            ms = ", ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"- {e.get('version')} [{e.get('base_name')}] {ms}")
        return 0

    if args.cmd == "best":
        best = find_best(args.metric, base_name=args.base)
        if not best:
            print(f"Không tìm thấy version nào có metric '{args.metric}'.")
            return 1
        print(json.dumps(best, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "promote":
        if args.path:
            src = Path(args.path)
            if not src.is_absolute():
                src = Path(_PROJECT_ROOT) / src
        elif args.version:
            src = resolve_version_path(args.version)
            if not src:
                print(f"Không tìm thấy version: {args.version}")
                return 1
        else:
            print("Cần --version hoặc --path.")
            return 1
        deploy = promote(src, args.deploy_path)
        print(f"Đã promote: {src} -> {deploy}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
