from io import BytesIO

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="cần OpenCV")
PIL = pytest.importorskip("PIL", reason="cần Pillow")

from PIL import Image, ImageDraw  # noqa: E402

from image_preprocess import preprocess_drawing  # noqa: E402


def _make_png() -> bytes:
    img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.line([(40, 40), (160, 160)], fill=(0, 0, 0, 255), width=8)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_output_shape():
    out = preprocess_drawing(_make_png())
    assert out.shape == (1, 28, 28, 1)
    assert out.dtype == np.float32


def test_output_range():
    out = preprocess_drawing(_make_png())
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() > 0.0  # có nét vẽ


def test_invalid_bytes():
    with pytest.raises(ValueError):
        preprocess_drawing(b"not-an-image")


def test_blank_image():
    img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    out = preprocess_drawing(buf.getvalue())
    assert out.shape == (1, 28, 28, 1)
    assert float(out.sum()) == 0.0
