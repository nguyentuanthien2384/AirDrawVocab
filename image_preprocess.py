"""
image_preprocess.py — Tiền xử lý ảnh vẽ DÙNG CHUNG cho AirDrawVocab.
(Phase 2, Task 15)

Trước đây logic này chỉ nằm trong backend/app.py. Tách ra module dùng chung để:
  - backend (/predict, camera) và các script đánh giá đều preprocess GIỐNG HỆT
    nhau (tránh lệch giữa production và evaluation).
  - dễ kiểm thử và cải tiến ở một nơi.

Hành vi mặc định (deskew=False) GIỮ NGUYÊN như backend cũ:
  ảnh -> grayscale nền đen nét trắng -> cắt sát nét -> scale vừa 20px ->
  đặt giữa canvas 28x28 -> căn theo trọng tâm (center of mass) -> chuẩn hóa [0,1].

Tùy chọn deskew=True: chỉnh nghiêng (shear) theo moment trước khi căn giữa —
giúp ổn định hơn với nét vẽ tay bị nghiêng, nhưng MẶC ĐỊNH TẮT để không đổi hành
vi model đang deploy.
"""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

TARGET_BOX = 20   # nét vẽ được scale vừa khung 20x20 (chuẩn MNIST/QuickDraw)
CANVAS = 28
CENTER = (CANVAS - 1) / 2.0  # 13.5


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Chỉnh nghiêng ảnh theo moment (shear ngang). Không đổi kích thước."""
    m = cv2.moments(gray)
    if abs(m["mu02"]) < 1e-2:
        return gray
    skew = m["mu11"] / m["mu02"]
    h, w = gray.shape
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def preprocess_drawing(image_bytes: bytes, deskew: bool = False) -> np.ndarray:
    """Ảnh bytes -> tensor (1, 28, 28, 1) float32. Raise ValueError nếu ảnh hỏng."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except Exception as exc:
        raise ValueError("File gửi lên không phải ảnh hợp lệ.") from exc

    rgba = np.array(image)
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.uint8)
    rgb = (rgb.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Frontend: nền trắng nét đen. Model: nền đen nét trắng -> đảo nếu ảnh sáng.
    if float(gray.mean()) > 127:
        gray = 255 - gray

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((1, CANVAS, CANVAS, 1), dtype="float32")

    x, y, w, h = cv2.boundingRect(coords)
    gray = gray[y:y + h, x:x + w]

    scale = TARGET_BOX / max(w, h, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    y_start = (CANVAS - new_h) // 2
    x_start = (CANVAS - new_w) // 2
    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized

    if deskew:
        canvas = _deskew(canvas)

    moments = cv2.moments(canvas, binaryImage=False)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        shift_x = int(round(CENTER - cx))
        shift_y = int(round(CENTER - cy))
        translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, translation, (CANVAS, CANVAS), borderValue=0)

    normalized = canvas.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
