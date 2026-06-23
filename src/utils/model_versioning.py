"""
model_versioning.py — Quản lý version model đơn giản cho AirDrawVocab.
(Phase 1, Task 3)

Mỗi lần train xong, ngoài việc ghi đè model deploy (airdrawvocab_best_advanced.keras),
ta lưu thêm một bản có version + metadata vào models/versions/ để dễ truy vết:

    models/versions/
        resnet_sketch_v20260622_141530.keras
        resnet_sketch_v20260622_141530.json     # metadata: metrics, params, env

Dùng:

    from src.utils.model_versioning import save_versioned_model
    info = save_versioned_model(
        model,
        base_name="resnet_sketch",
        metrics={"test_accuracy": 0.95},
        params={"epochs": 45},
        extra={"env": env_dict},
    )
    print(info["version"], info["model_path"])
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# bootstrap: cho phép import config ở thư mục gốc dự án
import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import MODELS_DIR
except Exception:  # fallback nếu chạy ngoài project
    MODELS_DIR = _PROJECT_ROOT / "models"

VERSIONS_DIR = MODELS_DIR / "versions"


def make_version_tag() -> str:
    """Tạo tag version theo thời gian: vYYYYMMDD_HHMMSS."""
    return datetime.now().strftime("v%Y%m%d_%H%M%S")


def save_versioned_model(
    model,
    base_name: str,
    metrics: dict | None = None,
    params: dict | None = None,
    extra: dict | None = None,
    versions_dir: Path | None = None,
) -> dict:
    """Lưu một bản model có version + file metadata JSON. Trả về dict thông tin.

    Không ném lỗi làm hỏng quá trình train: nếu gặp sự cố sẽ in cảnh báo và
    trả về dict rỗng.
    """
    versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR
    try:
        versions_dir.mkdir(parents=True, exist_ok=True)
        tag = make_version_tag()
        stem = f"{base_name}_{tag}"
        model_path = versions_dir / f"{stem}.keras"
        meta_path = versions_dir / f"{stem}.json"

        model.save(model_path)

        metadata = {
            "base_name": base_name,
            "version": tag,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_file": model_path.name,
            "metrics": metrics or {},
            "params": params or {},
        }
        if extra:
            metadata.update(extra)
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"[Versioning] Đã lưu bản version: {model_path}")
        return {
            "version": tag,
            "model_path": str(model_path),
            "metadata_path": str(meta_path),
            "metadata": metadata,
        }
    except Exception as exc:  # pragma: no cover
        print(f"[Versioning] Warning: không lưu được bản version - {exc}")
        return {}


def list_versions(base_name: str | None = None, versions_dir: Path | None = None) -> list[dict]:
    """Liệt kê metadata của các version đã lưu (mới nhất trước)."""
    versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR
    if not versions_dir.exists():
        return []
    out: list[dict] = []
    for meta in sorted(versions_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        if base_name and data.get("base_name") != base_name:
            continue
        out.append(data)
    return out
