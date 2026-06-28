"""
kalman.py — Bộ lọc làm mượt vị trí đầu ngón tay (giống kiến trúc trong tài liệu của bạn).

Dùng cv2.KalmanFilter nếu có OpenCV; nếu không thì fallback sang lọc EMA đơn giản,
để module không "chết" trên máy thiếu OpenCV (vd CI/test offline).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


class FingertipSmoother:
    """4 trạng thái (x, y, dx, dy), 2 đo lường (x, y). Tự bù 1-2 frame mất tay."""

    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.10):
        self._initialized = False
        self._use_cv = _HAS_CV2
        if self._use_cv:
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
            )
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
            self.kf = kf
        else:  # EMA fallback
            self._alpha = 0.5
            self._last = None

    def reset(self, x: float, y: float) -> None:
        if self._use_cv:
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePre = self.kf.statePost.copy()
        else:
            self._last = np.array([x, y], dtype=np.float64)
        self._initialized = True

    def update(self, x: float, y: float) -> Tuple[int, int]:
        if not self._initialized:
            self.reset(x, y)
            return int(x), int(y)
        if self._use_cv:
            self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
            pred = self.kf.predict()
            return int(pred[0][0]), int(pred[1][0])
        self._last = self._alpha * np.array([x, y]) + (1 - self._alpha) * self._last
        return int(self._last[0]), int(self._last[1])

    def predict_only(self) -> Optional[Tuple[int, int]]:
        """Khi mất tay 1-2 frame: dự đoán vị trí kế tiếp để nét không bị đứt."""
        if not self._initialized or not self._use_cv:
            return None
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])
