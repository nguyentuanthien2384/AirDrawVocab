"""
repro.py — Tiện ích đảm bảo khả năng tái lập (reproducibility) cho AirDrawVocab.
(Phase 1, Task 5)

Cung cấp:
- set_global_seed(seed): cố định seed cho random / numpy / tensorflow.
- collect_environment(): thu thập thông tin môi trường (Python, TF, OS, GPU...).

Dùng trong script train:

    from src.utils.repro import set_global_seed, collect_environment
    set_global_seed(42)
    env = collect_environment()   # dict, có thể log vào MLflow
"""
from __future__ import annotations

import os
import platform
import random
import sys


def set_global_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Cố định seed cho mọi nguồn ngẫu nhiên thường gặp.

    deterministic=True sẽ bật op-determinism của TensorFlow (chậm hơn nhưng
    kết quả lặp lại chính xác hơn).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
    except Exception:
        pass
    return seed


def _gpu_info() -> list[str]:
    try:
        import tensorflow as tf
        return [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        return []


def collect_environment() -> dict:
    """Trả về dict mô tả môi trường chạy (để log/reproduce)."""
    info: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
    except Exception:
        pass
    gpus = _gpu_info()
    info["gpu_count"] = len(gpus)
    info["gpus"] = ", ".join(gpus) if gpus else "none"
    return info
