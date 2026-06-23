"""
Hàm dùng chung cho load và chia dữ liệu QuickDraw.
Dùng bởi train_model.py, advanced_train_model.py, baseline_model.py, evaluate_model.py.
"""
from __future__ import annotations

# --- bootstrap: đảm bảo import được config ở thư mục gốc dự án ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from config import CATEGORIES, DATA_DIR, TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS


def load_dataset(samples_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tải và tiền xử lý dataset QuickDraw từ file .npy."""
    if samples_per_class is None:
        samples_per_class = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS

    X_parts, y_parts = [], []
    for class_id, category in enumerate(CATEGORIES):
        path = DATA_DIR / f"{category}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file dataset: {path}")
        data = np.load(path)
        data = data[data.sum(axis=1) > 0]  # Lọc mẫu rỗng
        if len(data) < samples_per_class:
            raise ValueError(f"{category} chỉ có {len(data)} mẫu, cần {samples_per_class}.")
        data = data[:samples_per_class].astype("float32") / 255.0
        X_parts.append(data)
        y_parts.append(np.full(len(data), class_id, dtype=np.int32))
        print(f"  {category}: {len(data)} mẫu")
    return np.concatenate(X_parts), np.concatenate(y_parts)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    train_per_class: int = TRAIN_PER_CLASS,
    val_per_class: int = VAL_PER_CLASS,
    test_per_class: int = TEST_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia dữ liệu train/val/test theo tỷ lệ cố định mỗi lớp."""
    rng = np.random.default_rng(seed)
    required = train_per_class + val_per_class + test_per_class
    train_idx, val_idx, test_idx = [], [], []

    for class_id in range(len(CATEGORIES)):
        indices = np.where(y == class_id)[0]
        if len(indices) < required:
            raise ValueError(f"Lớp {CATEGORIES[class_id]} có {len(indices)} mẫu, cần {required}.")
        rng.shuffle(indices)
        train_idx.extend(indices[:train_per_class])
        val_idx.extend(indices[train_per_class:train_per_class + val_per_class])
        test_idx.extend(indices[train_per_class + val_per_class:required])

    for idx in [train_idx, val_idx, test_idx]:
        rng.shuffle(idx)

    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]
