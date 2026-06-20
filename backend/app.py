import json
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import load_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_auth import FaceAuthManager

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

from config import MODEL_PATH, MODELS_DIR, ROOT
try:
    from config import DATA_DIR as _DATA_DIR
except Exception:
    _DATA_DIR = ROOT / "data" / "npy_28"
try:
    import sample_generator as _samplegen
except Exception:
    _samplegen = None

# Từ vựng mở rộng (40 lớp) + dịch nghĩa/câu ví dụ — nguồn từ vocab_pairs.py.
try:
    from vocab_pairs import (VI_MEANINGS as _VI, EXAMPLE_SENTENCES as _EX,
                             EXAMPLE_SENTENCES_VI as _EXVI, IPA as _IPA)
except Exception:
    _VI = _EX = _EXVI = _IPA = {}

LIVE_DRAWING_MODEL_CANDIDATES = [
    MODELS_DIR / "airdrawvocab_enhanced_model.h5",
    MODEL_PATH,
]
LIVE_DRAWING_MODEL_PATH = next((path for path in LIVE_DRAWING_MODEL_CANDIDATES if path.exists()), MODEL_PATH)
CATEGORIES_PATH = MODELS_DIR / "categories.json"
FRONTEND_DIR = ROOT / "frontend"
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")
OPENAI_IMAGE_ENABLED = os.getenv("OPENAI_IMAGE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}

# Test-Time Augmentation: trung bình dự đoán trên ảnh gốc + vài bản dịch nhẹ.
# Tăng độ chính xác/ổn định ~0.3-1% mà không cần train lại. Tắt: USE_TTA=0
USE_TTA = os.getenv("USE_TTA", "1").strip().lower() not in {"0", "false", "no"}
TTA_SHIFTS = int(os.getenv("TTA_SHIFTS", "4"))

VI_MEANINGS: Dict[str, str] = {
    "apple": "quả táo",
    "baseball": "bóng chày",
    "book": "quyển sách",
    "bowtie": "nơ bướm",
    "diamond": "kim cương",
    "dog": "con chó",
    "door": "cánh cửa",
    "envelope": "phong bì",
    "eye": "con mắt",
    "fish": "con cá",
    "hat": "cái mũ",
    "leaf": "chiếc lá",
    "lightning": "tia sét",
    "moon": "mặt trăng",
    "pants": "quần dài",
    "scissors": "cái kéo",
    "square": "hình vuông",
    "star": "ngôi sao",
    "t-shirt": "áo thun",
}

EXAMPLE_SENTENCES: Dict[str, str] = {
    "apple": "I eat an apple every morning.",
    "baseball": "He plays baseball after school.",
    "book": "This book is very interesting.",
    "bowtie": "He wears a bowtie at the party.",
    "diamond": "The diamond is very shiny.",
    "dog": "The dog is friendly.",
    "door": "Please close the door.",
    "envelope": "She puts the letter in an envelope.",
    "eye": "My eye is blue.",
    "fish": "The fish swims in the water.",
    "hat": "I wear a hat on sunny days.",
    "leaf": "A leaf falls from the tree.",
    "lightning": "Lightning appears during the storm.",
    "moon": "The moon is bright tonight.",
    "pants": "These pants are black.",
    "scissors": "I cut paper with scissors.",
    "square": "This is a red square.",
    "star": "A star shines in the sky.",
    "t-shirt": "I like this t-shirt.",
}

# Gộp từ vựng mở rộng (40 lớp) vào dict gốc — ưu tiên dữ liệu vocab_pairs.
VI_MEANINGS.update(_VI)
EXAMPLE_SENTENCES.update(_EX)

REFERENCE_PROMPTS: Dict[str, str] = {
    "apple": "a fresh red apple with a small stem and leaf",
    "baseball": "a clean white baseball with red stitching",
    "book": "a realistic closed school book with visible pages",
    "bowtie": "a neat black bow tie made of satin fabric",
    "diamond": "a clear cut diamond gemstone with natural reflections",
    "dog": "a friendly small dog sitting in soft studio light",
    "door": "a realistic wooden front door with a simple handle",
    "envelope": "a white paper envelope on a clean tabletop",
    "eye": "a realistic human eye close-up, educational and neutral",
    "fish": "a small colorful fish in a clean aquarium-like scene",
    "hat": "a casual baseball cap made of fabric",
    "leaf": "a fresh green leaf with visible veins",
    "lightning": "a realistic lightning bolt in a cloudy night sky",
    "moon": "a detailed full moon in a dark sky",
    "pants": "a pair of blue jeans laid flat",
    "scissors": "a pair of metal scissors on a clean surface",
    "square": "a realistic square wooden tile viewed from above",
    "star": "a five-point golden star ornament",
    "t-shirt": "a plain white t-shirt on a neutral background",
}

app = FastAPI(
    title="AirDrawVocab Recognition Chatbot API",
    description="Nhận diện hình vẽ AirDrawVocab bằng Keras model và trả lời bằng chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise RuntimeError(f"Không tìm thấy model: {MODEL_PATH}")

if not CATEGORIES_PATH.exists():
    raise RuntimeError(f"Không tìm thấy categories.json: {CATEGORIES_PATH}")

with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    categories: List[str] = json.load(f)

# compile=False giúp tránh lỗi khi môi trường TensorFlow/Keras khác phiên bản lúc train.
model = load_model(MODEL_PATH, compile=False)
live_drawing_model = model
if LIVE_DRAWING_MODEL_PATH != MODEL_PATH:
    live_drawing_model = load_model(LIVE_DRAWING_MODEL_PATH, compile=False)
face_auth = FaceAuthManager(PROJECT_ROOT / "face_data")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def home():
    """Mở giao diện vẽ nếu có frontend, nếu không trả về trạng thái API."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AirDrawVocab API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_categories": len(categories),
        "categories": categories,
        "model_input_shape": str(getattr(model, "input_shape", "unknown")),
        "model_path": str(MODEL_PATH),
        "live_drawing_model_path": str(LIVE_DRAWING_MODEL_PATH),
        "face_model_exists": (PROJECT_ROOT / "face_data" / "lbph_face_model.yml").exists(),
        "image_generation": {
            "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "openai_enabled": OPENAI_IMAGE_ENABLED,
            "model": OPENAI_IMAGE_MODEL,
            "size": OPENAI_IMAGE_SIZE,
            "quality": OPENAI_IMAGE_QUALITY,
            "offline_fallback": True,
        },
    }


@app.get("/vocab")
def vocab():
    """Trả về toàn bộ cặp từ vựng + dịch nghĩa (EN, VI, IPA, ví dụ EN/VI)."""
    items = []
    for label in categories:
        items.append({
            "label": label,
            "meaning_vi": VI_MEANINGS.get(label, label),
            "ipa": _IPA.get(label, ""),
            "example_en": EXAMPLE_SENTENCES.get(label, f"This is a {label}."),
            "example_vi": _EXVI.get(label, ""),
        })
    return {"count": len(items), "vocab": items}


def _samples_response(maker):
    """Bọc các hàm sinh ảnh: trả PNG hoặc JSON lỗi (thiếu dữ liệu/thư viện)."""
    if _samplegen is None:
        return JSONResponse(status_code=503, content={"error": "Chưa có sample_generator.py."})
    try:
        png = maker()
        return Response(content=png, media_type="image/png")
    except Exception as exc:  # thiếu dữ liệu npy hoặc không tải được QuickDraw
        return JSONResponse(status_code=503, content={"error": str(exc)})


@app.get("/samples/grid")
def samples_grid():
    """Tự sinh ảnh: 1 hình mẫu mỗi lớp (sample_drawings)."""
    return _samples_response(lambda: _samplegen.sample_grid_png(categories, _DATA_DIR, VI_MEANINGS))


@app.get("/samples/prediction")
def samples_prediction(seed: int = 0):
    """Tự sinh ảnh: 1 hình + biểu đồ xác suất tất cả lớp."""
    return _samples_response(lambda: _samplegen.prediction_sample_png(model, categories, _DATA_DIR, seed=seed))


@app.get("/samples/multiple")
def samples_multiple(count: int = 15, seed: int = 1):
    """Tự sinh ảnh: lưới nhiều dự đoán (xanh=đúng, đỏ=sai)."""
    return _samples_response(lambda: _samplegen.multiple_predictions_png(model, categories, _DATA_DIR, count=count, seed=seed))


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Chuyển ảnh canvas/user upload về đúng input model: 28x28 grayscale, nền đen nét trắng."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="File gửi lên không phải ảnh hợp lệ.") from exc

    rgba = np.array(image)

    # Nếu ảnh có alpha, ghép lên nền trắng để xử lý ổn định.
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.uint8)
    rgb = (rgb.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Frontend thường là nền trắng, nét đen. Dataset/model dùng nền đen, nét trắng.
    # Nếu ảnh trung bình sáng thì đảo màu.
    if float(gray.mean()) > 127:
        gray = 255 - gray

    # Cắt vùng có nét vẽ để object nằm giữa ảnh, tránh canvas quá nhiều khoảng trắng.
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(8, int(max(w, h) * 0.20))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        gray = gray[y1:y2, x1:x2]

    # Đưa ảnh về canvas vuông trước khi resize, tránh méo hình.
    h, w = gray.shape[:2]
    side = max(h, w, 1)
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = gray

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=(0, -1))
    return normalized


def _image_to_rgb_array(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="File gửi lên không phải ảnh hợp lệ.") from exc

    rgba = np.array(image)
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.uint8)
    return (rgb.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def encode_png_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def enhance_drawing_image(image_bytes: bytes, size: int = 768) -> str:
    """Tạo bản line-art rõ nét hơn từ ảnh vẽ tay/canvas."""
    rgb = _image_to_rgb_array(image_bytes)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if float(gray.mean()) > 127:
        ink = 255 - gray
    else:
        ink = gray.copy()

    ink = cv2.GaussianBlur(ink, (3, 3), 0)
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(24, int(max(w, h) * 0.18))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(mask.shape[1], x + w + pad)
        y2 = min(mask.shape[0], y + h + pad)
        mask = mask[y1:y2, x1:x2]

    h, w = mask.shape[:2]
    side = max(h, w, 1)
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = mask

    # Làm nét nhưng vẫn giữ cảm giác vẽ tay.
    kernel = np.ones((3, 3), np.uint8)
    square = cv2.morphologyEx(square, cv2.MORPH_CLOSE, kernel, iterations=1)
    upscaled = cv2.resize(square, (size, size), interpolation=cv2.INTER_LANCZOS4)
    upscaled = cv2.GaussianBlur(upscaled, (3, 3), 0)
    _, upscaled = cv2.threshold(upscaled, 88, 255, cv2.THRESH_BINARY)

    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    ink_pixels = upscaled > 0
    canvas[ink_pixels] = (22, 28, 38)

    # Thêm nền giấy rất nhẹ để ảnh preview bớt thô, không ảnh hưởng inference.
    paper = Image.fromarray(canvas, "RGB")
    return encode_png_data_uri(paper)


def build_realistic_prompt(label: str) -> str:
    subject = REFERENCE_PROMPTS.get(label, label.replace("-", " "))
    vi = VI_MEANINGS.get(label, label)
    return (
        "Create a photorealistic educational reference image for an English vocabulary learning app. "
        f"Main subject: {subject}. Vocabulary label: {label}, Vietnamese meaning: {vi}. "
        "Use a single clear object, centered composition, natural lighting, realistic material texture, "
        "clean neutral background, no text, no watermark, no logo, no extra unrelated objects. "
        "The image should help a student understand what the drawn sketch represents in real life."
    )


def _draw_soft_shadow(draw, box: Tuple[int, int, int, int], alpha: int = 45) -> None:
    x1, y1, x2, y2 = box
    for offset, opacity in [(20, alpha // 4), (12, alpha // 3), (6, alpha // 2)]:
        draw.ellipse((x1 - offset, y1 - offset, x2 + offset, y2 + offset), fill=(15, 23, 42, opacity))


def create_offline_reference_image(label: str, size: int = 768) -> str:
    """Fallback ảnh tham khảo rõ nét khi chưa cấu hình API tạo ảnh."""
    image = Image.new("RGB", (size, size), (246, 248, 252))
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = size // 2, size // 2

    # Nền sáng với vignette nhẹ.
    for r in range(size // 2, 0, -12):
        tone = int(250 - (size // 2 - r) * 0.025)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(tone, tone + 1, 255, 6))

    if label == "apple":
        _draw_soft_shadow(draw, (225, 235, 545, 585))
        draw.ellipse((230, 220, 540, 595), fill=(178, 34, 42, 255), outline=(120, 20, 28, 255), width=4)
        draw.ellipse((310, 180, 440, 500), fill=(224, 45, 55, 210))
        draw.rectangle((370, 150, 395, 240), fill=(88, 52, 29, 255))
        draw.ellipse((395, 145, 510, 215), fill=(47, 129, 78, 255))
    elif label == "baseball":
        _draw_soft_shadow(draw, (190, 190, 578, 578))
        draw.ellipse((190, 170, 578, 558), fill=(250, 250, 246, 255), outline=(210, 210, 204, 255), width=5)
        for x in [300, 470]:
            draw.arc((x - 90, 190, x + 90, 545), 250, 110, fill=(180, 28, 42, 255), width=7)
    elif label == "book":
        draw.rounded_rectangle((175, 210, 585, 545), radius=24, fill=(37, 99, 235, 255), outline=(20, 62, 150, 255), width=5)
        draw.rectangle((205, 235, 550, 520), fill=(239, 246, 255, 255))
        draw.line((382, 225, 382, 530), fill=(148, 163, 184, 255), width=5)
    elif label == "bowtie":
        draw.polygon([(110, 250), (340, 360), (110, 520)], fill=(20, 20, 28, 255))
        draw.polygon([(658, 250), (428, 360), (658, 520)], fill=(20, 20, 28, 255))
        draw.rounded_rectangle((335, 300, 435, 455), radius=24, fill=(45, 45, 58, 255))
    elif label == "diamond":
        draw.polygon([(384, 125), (600, 310), (384, 650), (168, 310)], fill=(210, 244, 255, 255), outline=(14, 116, 144, 255))
        draw.line((168, 310, 600, 310), fill=(14, 116, 144, 255), width=4)
        draw.line((384, 125, 300, 310, 384, 650, 468, 310, 384, 125), fill=(14, 116, 144, 255), width=4)
    elif label == "dog":
        draw.ellipse((210, 190, 558, 580), fill=(180, 120, 72, 255), outline=(120, 74, 44, 255), width=5)
        draw.ellipse((155, 220, 290, 430), fill=(130, 78, 46, 255))
        draw.ellipse((478, 220, 613, 430), fill=(130, 78, 46, 255))
        draw.ellipse((285, 330, 315, 360), fill=(20, 20, 20, 255))
        draw.ellipse((453, 330, 483, 360), fill=(20, 20, 20, 255))
        draw.ellipse((355, 405, 410, 450), fill=(35, 20, 18, 255))
    elif label == "door":
        draw.rounded_rectangle((235, 105, 545, 665), radius=18, fill=(128, 78, 40, 255), outline=(78, 48, 28, 255), width=6)
        draw.rectangle((270, 150, 510, 325), outline=(82, 48, 28, 255), width=5)
        draw.rectangle((270, 360, 510, 625), outline=(82, 48, 28, 255), width=5)
        draw.ellipse((465, 380, 500, 415), fill=(229, 180, 70, 255))
    elif label == "envelope":
        draw.rounded_rectangle((145, 245, 623, 530), radius=10, fill=(255, 255, 255, 255), outline=(160, 160, 160, 255), width=4)
        draw.line((145, 245, 384, 395, 623, 245), fill=(160, 160, 160, 255), width=4)
        draw.line((145, 530, 330, 380), fill=(160, 160, 160, 255), width=4)
        draw.line((623, 530, 438, 380), fill=(160, 160, 160, 255), width=4)
    elif label == "eye":
        draw.ellipse((115, 265, 653, 505), fill=(255, 255, 255, 255), outline=(38, 38, 48, 255), width=7)
        draw.ellipse((300, 245, 468, 525), fill=(59, 130, 246, 255))
        draw.ellipse((347, 315, 421, 450), fill=(10, 10, 15, 255))
        draw.ellipse((376, 300, 405, 330), fill=(255, 255, 255, 220))
    elif label == "fish":
        draw.ellipse((145, 250, 555, 520), fill=(247, 156, 74, 255), outline=(170, 83, 30, 255), width=5)
        draw.polygon([(555, 385), (690, 250), (690, 520)], fill=(239, 120, 55, 255), outline=(170, 83, 30, 255))
        draw.ellipse((245, 330, 280, 365), fill=(20, 20, 20, 255))
    elif label == "hat":
        draw.pieslice((155, 210, 555, 590), 180, 360, fill=(30, 100, 210, 255), outline=(20, 64, 140, 255), width=5)
        draw.rounded_rectangle((135, 420, 660, 500), radius=38, fill=(37, 99, 235, 255))
    elif label == "leaf":
        draw.ellipse((220, 130, 555, 630), fill=(47, 150, 82, 255), outline=(20, 100, 55, 255), width=5)
        draw.line((385, 170, 385, 640), fill=(220, 255, 220, 210), width=5)
        for y in range(240, 570, 70):
            draw.line((385, y, 500, y - 55), fill=(220, 255, 220, 170), width=3)
            draw.line((385, y, 270, y - 55), fill=(220, 255, 220, 170), width=3)
    elif label == "lightning":
        draw.polygon([(430, 80), (220, 410), (360, 410), (280, 690), (570, 330), (425, 330)], fill=(250, 204, 21, 255), outline=(180, 120, 0, 255))
    elif label == "moon":
        draw.ellipse((170, 120, 610, 560), fill=(225, 229, 235, 255), outline=(180, 185, 195, 255), width=5)
        draw.ellipse((260, 190, 330, 260), fill=(195, 200, 210, 170))
        draw.ellipse((420, 320, 500, 400), fill=(195, 200, 210, 160))
    elif label == "pants":
        draw.polygon([(245, 115), (535, 115), (610, 665), (445, 665), (390, 305), (335, 665), (170, 665)], fill=(37, 99, 235, 255), outline=(20, 64, 140, 255))
        draw.line((390, 305, 390, 665), fill=(20, 64, 140, 255), width=6)
    elif label == "scissors":
        draw.line((230, 210, 560, 590), fill=(110, 116, 128, 255), width=16)
        draw.line((560, 210, 230, 590), fill=(110, 116, 128, 255), width=16)
        draw.ellipse((145, 150, 285, 290), outline=(37, 99, 235, 255), width=16)
        draw.ellipse((145, 520, 285, 660), outline=(37, 99, 235, 255), width=16)
    elif label == "square":
        draw.rectangle((200, 200, 568, 568), fill=(202, 138, 74, 255), outline=(120, 72, 36, 255), width=8)
        for x in range(230, 560, 55):
            draw.line((x, 205, x + 70, 565), fill=(150, 90, 45, 80), width=3)
    elif label == "star":
        pts = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            r = 265 if i % 2 == 0 else 110
            pts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
        draw.polygon(pts, fill=(245, 184, 35, 255), outline=(170, 110, 20, 255))
    elif label == "t-shirt":
        draw.polygon([(250, 150), (330, 110), (384, 170), (438, 110), (518, 150), (610, 295), (520, 355), (520, 650), (248, 650), (248, 355), (158, 295)], fill=(255, 255, 255, 255), outline=(90, 96, 110, 255))
        draw.arc((332, 125, 436, 225), 0, 180, fill=(90, 96, 110, 255), width=5)
    else:
        draw.ellipse((180, 180, 588, 588), fill=(226, 232, 240, 255), outline=(100, 116, 139, 255), width=6)

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return encode_png_data_uri(image)


def generate_openai_reference_image(label: str) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not OPENAI_IMAGE_ENABLED or not api_key:
        return None, None

    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": build_realistic_prompt(label),
        "size": OPENAI_IMAGE_SIZE,
        "quality": OPENAI_IMAGE_QUALITY,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        image_b64 = data["data"][0]["b64_json"]
        return f"data:image/png;base64,{image_b64}", None
    except Exception as exc:
        return None, str(exc)


def fallback_chatbot_reply(label: str, confidence: float, top3: List[dict]) -> str:
    vi = VI_MEANINGS.get(label, label)
    example = EXAMPLE_SENTENCES.get(label, f"This is a {label}.")
    ipa = _IPA.get(label, "")
    example_vi = _EXVI.get(label, "")
    percent = round(confidence * 100, 2)
    top3_text = ", ".join([item["label"] for item in top3])
    ipa_text = f" {ipa}" if ipa else ""
    ex_vi_text = f" ({example_vi})" if example_vi else ""

    if confidence >= 0.80:
        return (
            f"Mình nhận diện hình bạn vẽ là **{label}**{ipa_text} — nghĩa tiếng Việt là **{vi}**. "
            f"Độ tin cậy khoảng **{percent}%**. Ví dụ: *{example}*{ex_vi_text}"
        )

    if confidence >= 0.50:
        return (
            f"Mình đoán hình này là **{label}**{ipa_text} — nghĩa là **{vi}**, với độ tin cậy khoảng **{percent}%**. "
            f"Ví dụ: *{example}*{ex_vi_text} Các khả năng gần giống: {top3_text}. "
            "Bạn có thể vẽ nét to, rõ và ở giữa khung để model nhận diện chắc hơn."
        )

    return (
        f"Mình chưa chắc lắm nha. Dự đoán gần nhất là **{label}** — **{vi}**, "
        f"nhưng độ tin cậy chỉ khoảng **{percent}%**. Top dự đoán: {top3_text}. "
        "Bạn thử xóa và vẽ lại to hơn, ít nét thừa hơn để kết quả tốt hơn nhé."
    )


def foza_chatbot_reply(label: str, confidence: float, top3: List[dict]) -> str:
    """Gọi Foza/Anthropic-compatible API nếu có API key, lỗi thì tự fallback."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.foza.ai/v1").rstrip("/")
    model_name = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4.6")

    if not api_key or api_key == "sk-foza-xxxxx":
        return fallback_chatbot_reply(label, confidence, top3)

    vi = VI_MEANINGS.get(label, label)
    payload = {
        "model": model_name,
        "max_tokens": 350,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Bạn là chatbot học từ vựng tiếng Anh cho ứng dụng AirDrawVocab. "
                    "Hãy trả lời ngắn gọn, thân thiện bằng tiếng Việt.\n\n"
                    f"Model nhận diện hình vẽ là: {label}\n"
                    f"Nghĩa tiếng Việt dự kiến: {vi}\n"
                    f"Độ tin cậy: {round(confidence * 100, 2)}%\n"
                    f"Top 3 dự đoán: {top3}\n\n"
                    "Yêu cầu trả lời gồm: tên tiếng Anh, nghĩa tiếng Việt, 1 câu ví dụ tiếng Anh, "
                    "và nếu độ tin cậy thấp thì gợi ý cách vẽ rõ hơn."
                ),
            }
        ],
    }

    try:
        response = requests.post(
            f"{base_url}/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    except Exception:
        return fallback_chatbot_reply(label, confidence, top3)


def predict_proba(active_model, x: np.ndarray) -> np.ndarray:
    """Dự đoán xác suất; nếu bật TTA thì trung bình trên ảnh gốc + vài bản dịch nhẹ."""
    base = active_model.predict(x, verbose=0)[0]
    if not USE_TTA or TTA_SHIFTS <= 0:
        return base
    pad = 3
    probs = [base]
    for _ in range(TTA_SHIFTS):
        xs = tf.image.random_crop(
            tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]), tf.shape(x))
        probs.append(active_model.predict(xs.numpy(), verbose=0)[0])
    return np.mean(probs, axis=0)


def _top3_from_predictions(preds: np.ndarray) -> List[dict]:
    top_indices = preds.argsort()[-3:][::-1]
    return [
        {
            "label": categories[int(i)],
            "meaning_vi": VI_MEANINGS.get(categories[int(i)], categories[int(i)]),
            "confidence": float(preds[int(i)]),
        }
        for i in top_indices
    ]


@app.post("/predict")
async def predict(file: UploadFile = File(...), source: str = Form("canvas")):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi ảnh lên.")

    x = preprocess_image(image_bytes)
    enhanced_drawing = enhance_drawing_image(image_bytes)
    normalized_source = source.strip().lower()
    active_model = live_drawing_model if normalized_source in {"camera", "hand", "airdraw", "live"} else model
    active_model_path = LIVE_DRAWING_MODEL_PATH if active_model is live_drawing_model else MODEL_PATH
    preds = predict_proba(active_model, x)

    best_index = int(np.argmax(preds))
    label = categories[best_index]
    confidence = float(preds[best_index])

    top3 = _top3_from_predictions(preds)

    reply = foza_chatbot_reply(label, confidence, top3)

    return {
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "top3": top3,
        "chatbot_reply": reply,
        "enhanced_drawing": enhanced_drawing,
        "source": normalized_source,
        "model_used": str(active_model_path),
    }


@app.post("/image/enhance")
async def image_enhance(file: UploadFile = File(...)):
    """Làm rõ nét ảnh vẽ tay/canvas thành line-art độ phân giải cao."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi ảnh lên.")
    return {
        "ok": True,
        "enhanced_drawing": enhance_drawing_image(image_bytes),
    }


@app.post("/image/reference")
async def image_reference(label: str = Form(...)):
    """Trả ảnh tham khảo offline theo nhãn, dùng cho preview realtime không tốn API tạo ảnh."""
    label = label.strip().lower()
    if label not in categories:
        raise HTTPException(status_code=400, detail=f"Nhãn không hợp lệ: {label}")
    return {
        "ok": True,
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "provider": "offline-pil-reference",
        "image": create_offline_reference_image(label),
    }


@app.post("/image/generate")
async def image_generate(label: str = Form(...)):
    """Tạo ảnh tham khảo chân thực theo nhãn đã nhận diện, có fallback offline."""
    label = label.strip().lower()
    if label not in categories:
        raise HTTPException(status_code=400, detail=f"Nhãn không hợp lệ: {label}")

    ai_image, error = generate_openai_reference_image(label)
    if ai_image:
        return {
            "ok": True,
            "label": label,
            "meaning_vi": VI_MEANINGS.get(label, label),
            "provider": f"openai:{OPENAI_IMAGE_MODEL}",
            "prompt": build_realistic_prompt(label),
            "image": ai_image,
        }

    return {
        "ok": True,
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "provider": "offline-pil-reference",
        "prompt": build_realistic_prompt(label),
        "image": create_offline_reference_image(label),
        "note": (
            "Chưa có OPENAI_API_KEY hoặc API tạo ảnh chưa gọi được, "
            "nên hệ thống dùng ảnh tham khảo offline. Cấu hình OPENAI_API_KEY để tạo ảnh photorealistic."
        ),
        "error": error,
    }


@app.post("/face/enroll")
async def face_enroll(username: str = Form(...), file: UploadFile = File(...)):
    """Đăng ký một mẫu khuôn mặt từ ảnh webcam/frontend gửi lên."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi ảnh khuôn mặt.")
    try:
        result = face_auth.enroll_image_bytes(username, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": result.ok,
        "message": result.message,
        "username": result.username,
        "sample_count": result.sample_count,
        "threshold": result.threshold,
    }


@app.post("/face/verify")
async def face_verify(username: str = Form(""), file: UploadFile = File(...)):
    """Xác thực khuôn mặt từ ảnh webcam/frontend gửi lên."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi ảnh khuôn mặt.")
    try:
        result = face_auth.verify_image_bytes(image_bytes, username=username.strip() or None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": result.ok,
        "message": result.message,
        "username": result.username,
        "score": result.score,
        "threshold": result.threshold,
    }
