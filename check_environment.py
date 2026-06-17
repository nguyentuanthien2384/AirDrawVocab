"""
AirDrawVocab environment checker.
    python check_environment.py
"""
from __future__ import annotations

import platform
import sys

from config import MODEL_PATH, MODELS_DIR, DATA_DIR, CATEGORIES_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BEST_MODEL = MODELS_DIR / "airdrawvocab_best_advanced.keras"


def ok(msg): print(f"[OK] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def fail(msg): print(f"[FAIL] {msg}")


def check_python() -> bool:
    v = sys.version_info
    print(f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    if platform.architecture()[0] != "64bit":
        fail("Bạn đang dùng Python 32-bit. Hãy cài Python 64-bit.")
        return False
    if not (v.major == 3 and 10 <= v.minor <= 12):
        fail("Khuyến nghị dùng Python 3.11 hoặc 3.12.")
        return False
    ok("Phiên bản Python phù hợp")
    return True


def import_version(module_name: str) -> bool:
    try:
        mod = __import__(module_name)
        ok(f"{module_name}: {getattr(mod, '__version__', 'unknown')}")
        return True
    except Exception as e:
        fail(f"Không import được {module_name}: {e}")
        return False


def check_model() -> bool:
    if not MODEL_PATH.exists():
        fail(f"Thiếu model: {MODEL_PATH}")
        return False
    ok(f"Có model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    if BEST_MODEL.exists():
        ok(f"Có advanced checkpoint: {BEST_MODEL.name}")
    if not CATEGORIES_PATH.exists():
        fail("Thiếu models/categories.json.")
        return False
    ok(f"Có categories: {CATEGORIES_PATH.name}")
    try:
        import tensorflow as tf
        tf.keras.models.load_model(MODEL_PATH, compile=False)
        ok("Load model thành công")
        return True
    except Exception as e:
        fail(f"Không load được model: {e}")
        return False


def check_dataset() -> bool:
    if not DATA_DIR.exists():
        warn("Không thấy dataset local data/npy_28.")
        return False
    files = list(DATA_DIR.glob("*.npy"))
    if len(files) < 19:
        warn(f"Dataset chỉ có {len(files)} file .npy, cần 19.")
        return False
    ok(f"Có dataset local: {len(files)} lớp")
    return True


def check_camera() -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        cap.release()
        if opened:
            ok("Webcam mở được")
            return True
        warn("Không mở được webcam.")
        return False
    except Exception:
        warn("Không kiểm tra được webcam.")
        return False


def main() -> int:
    print("=" * 64)
    print("AirDrawVocab - kiểm tra môi trường")
    print("=" * 64)

    py_ok = check_python()
    print("\nThư viện:")
    imports_ok = all(import_version(m) for m in [
        "numpy", "cv2", "pygame", "mediapipe", "pyttsx3",
        "tensorflow", "sklearn", "matplotlib", "seaborn",
    ])

    print("\nTài nguyên project:")
    model_ok = check_model()
    check_dataset()
    check_camera()

    print("\n" + "=" * 64)
    if py_ok and imports_ok and model_ok:
        ok("Môi trường đã sẵn sàng. Chạy: python game.py")
        return 0
    fail("Môi trường chưa sẵn sàng.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
