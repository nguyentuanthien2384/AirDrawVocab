import json
import os
import csv
import base64
import hashlib
import secrets
import sqlite3
import subprocess
import shutil
import threading
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, WebSocket, WebSocketDisconnect
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

# Đặc trưng stroke dùng chung với train (src/training/train_stroke_model.py) để
# train và inference KHÔNG bị lệch. Xem stroke_features.py.
from stroke_features import strokes_to_batch as _strokes_to_batch
# Tiền xử lý ảnh dùng chung (Phase 2, Task 15). Xem image_preprocess.py.
from image_preprocess import preprocess_drawing as _preprocess_drawing
from camera_face_strokes import analyze_face_frame_bytes, FACE_DETECTOR as FACE_STROKE_DETECTOR

# Từ vựng mở rộng (40 lớp) + dịch nghĩa/câu ví dụ — nguồn từ vocab_pairs.py.
try:
    from vocab_pairs import (
        CATEGORIES as _ALL_VOCAB_CATEGORIES,
        VI_MEANINGS as _VI,
        EXAMPLE_SENTENCES as _EX,
        EXAMPLE_SENTENCES_VI as _EXVI,
        IPA as _IPA,
        DRAWING_HINTS as _DRAWING_HINTS,
    )
except Exception:
    _ALL_VOCAB_CATEGORIES = []
    _VI = _EX = _EXVI = _IPA = _DRAWING_HINTS = {}

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
APP_DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
SESSION_COOKIE_NAME = "airdrawvocab_session"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7
STROKE_MODEL_PATH = MODELS_DIR / "stroke_sequence_model.keras"
STROKE_CATEGORIES_PATH = MODELS_DIR / "stroke_categories.json"
# Kho lưu riêng cho toàn bộ Self-improving Loop.
# Mọi thao tác lưu mẫu, export dataset, train/retrain và reload model sẽ
# tự ghi vết vào đây để dễ kiểm tra/lưu trữ mà không phải tìm rải rác nhiều nơi.
SELF_IMPROVING_LOOP_DIR = ROOT / "data" / "self_improving_loop"
SELF_LOOP_ACTIONS_DIR = SELF_IMPROVING_LOOP_DIR / "actions"
SELF_LOOP_SAMPLES_DIR = SELF_IMPROVING_LOOP_DIR / "samples"
SELF_LOOP_EXPORTS_DIR = SELF_IMPROVING_LOOP_DIR / "exports"
SELF_LOOP_JOBS_DIR = SELF_IMPROVING_LOOP_DIR / "jobs"
SELF_LOOP_STATUS_DIR = SELF_IMPROVING_LOOP_DIR / "status"
SELF_LOOP_MODELS_DIR = SELF_IMPROVING_LOOP_DIR / "models"
SELF_LOOP_README_PATH = SELF_IMPROVING_LOOP_DIR / "README.md"
SELF_LOOP_ACTION_LOG = SELF_LOOP_ACTIONS_DIR / "events.jsonl"
SELF_LOOP_SAMPLES_JSONL = SELF_LOOP_SAMPLES_DIR / "stroke_samples.jsonl"
RETRAIN_STATUS_PATH = SELF_LOOP_STATUS_DIR / "retrain_status.json"
EXPORTED_DATASET_DIR = SELF_LOOP_EXPORTS_DIR / "latest"

# Kho lưu tự động cho các panel/khu vực còn lại ở giao diện phải.
# Mục tiêu: khi người dùng bấm Nhận diện, Sinh hình thật, kết thúc game,
# xem Skill Profile, Leaderboard hoặc tham gia PvP thì dữ liệu liên quan được
# ghi thành file riêng để dễ kiểm tra ngoài database.
PANEL_STORAGE_DIR = ROOT / "data" / "panel_storage"
PANEL_ACTIONS_DIR = PANEL_STORAGE_DIR / "actions"
AI_RECOGNITION_STORAGE_DIR = PANEL_STORAGE_DIR / "ai_recognition"
REAL_IMAGE_STORAGE_DIR = PANEL_STORAGE_DIR / "real_image_after_draw"
AI_JUDGE_STORAGE_DIR = PANEL_STORAGE_DIR / "ai_judge_mode"
SKILL_PROFILE_STORAGE_DIR = PANEL_STORAGE_DIR / "skill_profile"
LEADERBOARD_STORAGE_DIR = PANEL_STORAGE_DIR / "leaderboard"
PVP_STORAGE_DIR = PANEL_STORAGE_DIR / "pvp_websocket"
GAME_SESSION_STORAGE_DIR = PANEL_STORAGE_DIR / "game_sessions"
PANEL_STORAGE_README_PATH = PANEL_STORAGE_DIR / "README.md"
PANEL_ACTION_LOG = PANEL_ACTIONS_DIR / "events.jsonl"

SELF_IMPROVED_MODEL_PATH = MODELS_DIR / "airdrawvocab_self_improved.keras"
SELF_IMPROVED_CATEGORIES_PATH = MODELS_DIR / "categories_self_improved.json"
TRAINING_MIN_CLASSES = int(os.getenv("TRAINING_MIN_CLASSES", "2"))
TRAINING_MIN_SAMPLES_PER_CLASS = int(os.getenv("TRAINING_MIN_SAMPLES_PER_CLASS", "3"))
TRAINING_MIN_TOTAL_SAMPLES = int(os.getenv("TRAINING_MIN_TOTAL_SAMPLES", "12"))
CANVAS_W = 960
CANVAS_H = 540

# Face-aware camera support for the in-browser camera drawing mode.  This is
# adapted from the DeepShieldAI-Pro face_detector.py pattern the user supplied:
# OpenCV Haar cascade -> largest face -> padded crop/metadata.  The game uses it
# as a lightweight quality/guide signal while MediaPipe FaceMesh draws strokes in
# the browser.
def _load_camera_face_detector():
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        return detector if not detector.empty() else None
    except Exception:
        return None


CAMERA_FACE_DETECTOR = _load_camera_face_detector()


def _decode_upload_image_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Không đọc được frame camera.")
    return frame


def _detect_largest_camera_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if frame is None or frame.size == 0 or CAMERA_FACE_DETECTOR is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CAMERA_FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(42, 42),
    )
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(width), int(height)


def _crop_camera_face_or_frame(frame: np.ndarray, padding_ratio: float = 0.22) -> Tuple[np.ndarray, dict]:
    box = _detect_largest_camera_face(frame)
    if box is None:
        return frame, {"faceDetected": False, "bbox": None}

    x, y, width, height = box
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    max_y, max_x = frame.shape[:2]

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(max_x, x + width + pad_x)
    bottom = min(max_y, y + height + pad_y)

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return frame, {"faceDetected": False, "bbox": None}

    crop_width = int(right - left)
    crop_height = int(bottom - top)
    return crop, {
        "faceDetected": True,
        "bbox": {
            "x": int(left),
            "y": int(top),
            "width": crop_width,
            "height": crop_height,
        },
        "faceBbox": {
            "x": int(x - left),
            "y": int(y - top),
            "width": int(width),
            "height": int(height),
        },
        "cropSize": {
            "width": crop_width,
            "height": crop_height,
        },
    }


def _camera_image_quality(frame: np.ndarray) -> dict:
    if frame is None or frame.size == 0:
        return {"brightness": 0, "contrast": 0, "sharpness": 0}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
    }


def _camera_face_feedback(metadata: dict, frame_shape: Tuple[int, int, int], quality: dict) -> dict:
    h, w = frame_shape[:2]
    if not metadata.get("faceDetected"):
        return {
            "ready": False,
            "status": "no_face",
            "message": "Chưa thấy mặt. Đưa khuôn mặt vào giữa camera nếu muốn dùng Bút mặt.",
            "faceAreaRatio": 0,
            "centerOffset": 1,
        }

    bbox = metadata.get("bbox") or {}
    area_ratio = float((bbox.get("width", 0) * bbox.get("height", 0)) / max(1, w * h))
    cx = float((bbox.get("x", 0) + bbox.get("width", 0) / 2) / max(1, w))
    cy = float((bbox.get("y", 0) + bbox.get("height", 0) / 2) / max(1, h))
    center_offset = float(np.hypot(cx - 0.5, cy - 0.5))

    brightness = float(quality.get("brightness") or 0)
    sharpness = float(quality.get("sharpness") or 0)
    if area_ratio < 0.045:
        status = "too_far"
        message = "Mặt hơi xa. Lại gần camera để nét Bút mặt ổn định hơn."
    elif center_offset > 0.32:
        status = "off_center"
        message = "Mặt đang lệch khung. Đưa mặt vào giữa trước khi vẽ bằng mũi."
    elif brightness < 55:
        status = "too_dark"
        message = "Ánh sáng hơi tối. Tăng sáng để nhận diện khuôn mặt tốt hơn."
    elif sharpness < 18:
        status = "blurry"
        message = "Camera hơi mờ/rung. Giữ mặt ổn định để nét vẽ không giật."
    else:
        status = "ready"
        message = "Mặt rõ. Có thể há miệng nhẹ để vẽ bằng đầu mũi."

    return {
        "ready": status == "ready",
        "status": status,
        "message": message,
        "faceAreaRatio": round(area_ratio, 4),
        "centerOffset": round(center_offset, 4),
    }


def _map_frame_to_canvas_point(
    x: float,
    y: float,
    frame_w: int,
    frame_h: int,
    canvas_w: int,
    canvas_h: int,
    mirror: bool,
    t: float,
    source: str,
) -> dict:
    px = float(x) / max(1, frame_w) * canvas_w
    py = float(y) / max(1, frame_h) * canvas_h
    if mirror:
        px = canvas_w - px
    return {
        "x": round(max(0.0, min(float(canvas_w), px)), 2),
        "y": round(max(0.0, min(float(canvas_h), py)), 2),
        "t": round(t, 2),
        "source": source,
    }


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, start: float, end: float, steps: int) -> List[Tuple[float, float]]:
    if steps <= 1:
        steps = 2
    angles = np.linspace(start, end, steps)
    return [(float(cx + rx * np.cos(a)), float(cy + ry * np.sin(a))) for a in angles]


def _opencv_face_strokes(
    metadata: dict,
    frame_w: int,
    frame_h: int,
    canvas_w: int,
    canvas_h: int,
    mirror: bool,
) -> Tuple[List[List[dict]], dict]:
    bbox = metadata.get("bbox") or {}
    face_rel = metadata.get("faceBbox") or {}
    if not bbox or not face_rel:
        return [], {}

    fx = float(bbox.get("x", 0) + face_rel.get("x", 0))
    fy = float(bbox.get("y", 0) + face_rel.get("y", 0))
    fw = float(face_rel.get("width", 0))
    fh = float(face_rel.get("height", 0))
    if fw <= 1 or fh <= 1:
        return [], {}

    cx = fx + fw / 2
    cy = fy + fh / 2
    specs = [
        ("face-opencv-oval", _ellipse_points(cx, cy + fh * 0.04, fw * 0.55, fh * 0.66, 0, 2 * np.pi, 42)),
        ("face-opencv-left-eye", _ellipse_points(fx + fw * 0.34, fy + fh * 0.42, fw * 0.095, fh * 0.045, 0, 2 * np.pi, 18)),
        ("face-opencv-right-eye", _ellipse_points(fx + fw * 0.66, fy + fh * 0.42, fw * 0.095, fh * 0.045, 0, 2 * np.pi, 18)),
        ("face-opencv-left-brow", [(fx + fw * 0.22, fy + fh * 0.32), (fx + fw * 0.34, fy + fh * 0.27), (fx + fw * 0.46, fy + fh * 0.32)]),
        ("face-opencv-right-brow", [(fx + fw * 0.54, fy + fh * 0.32), (fx + fw * 0.66, fy + fh * 0.27), (fx + fw * 0.78, fy + fh * 0.32)]),
        ("face-opencv-nose", [(cx, fy + fh * 0.43), (fx + fw * 0.47, fy + fh * 0.55), (fx + fw * 0.43, fy + fh * 0.62), (cx, fy + fh * 0.66), (fx + fw * 0.57, fy + fh * 0.62), (fx + fw * 0.53, fy + fh * 0.55), (cx, fy + fh * 0.43)]),
        ("face-opencv-mouth", _ellipse_points(cx, fy + fh * 0.76, fw * 0.19, fh * 0.07, 0.05 * np.pi, 0.95 * np.pi, 18)),
        ("face-opencv-mouth-lower", _ellipse_points(cx, fy + fh * 0.76, fw * 0.19, fh * 0.055, 1.05 * np.pi, 1.95 * np.pi, 18)),
    ]

    strokes: List[List[dict]] = []
    t = 0.0
    for source, pts in specs:
        stroke = []
        for i, (x, y) in enumerate(pts):
            stroke.append(_map_frame_to_canvas_point(x, y, frame_w, frame_h, canvas_w, canvas_h, mirror, t + i * 8, source))
        if len(stroke) >= 2:
            strokes.append(stroke)
        t += 240

    canvas_bbox_x = fx / max(1, frame_w) * canvas_w
    canvas_bbox_y = fy / max(1, frame_h) * canvas_h
    canvas_bbox_w = fw / max(1, frame_w) * canvas_w
    canvas_bbox_h = fh / max(1, frame_h) * canvas_h
    if mirror:
        canvas_bbox_x = canvas_w - canvas_bbox_x - canvas_bbox_w
    canvas_bbox = {
        "x": round(max(0.0, min(float(canvas_w), canvas_bbox_x)), 2),
        "y": round(max(0.0, min(float(canvas_h), canvas_bbox_y)), 2),
        "width": round(max(1.0, min(float(canvas_w), canvas_bbox_w)), 2),
        "height": round(max(1.0, min(float(canvas_h), canvas_bbox_h)), 2),
    }
    return strokes, canvas_bbox


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


# === Tính năng "Tô theo hình mẫu" (shape_match) ===========================
# Thêm 2 endpoint: GET /shape/templates và POST /shape/score.
# Bọc try/except để nếu thiếu thư mục shape_match thì app vẫn chạy bình thường.
try:
    from shape_match.web_endpoint import router as shape_router
    if shape_router is not None:
        app.include_router(shape_router)
        print("[shape_match] Đã gắn router: GET /shape/templates, POST /shape/score")
except Exception as _e:  # pragma: no cover
    print(f"[shape_match] Bỏ qua (không gắn được router): {_e}")
# =========================================================================


VOCAB_GROUPS = {
    "apple": "Food", "banana": "Food", "ice cream": "Food", "cake": "Food", "fish": "Food",
    "dog": "Animals", "cat": "Animals",
    "leaf": "Nature", "lightning": "Nature", "moon": "Nature", "star": "Nature", "sun": "Nature", "tree": "Nature", "flower": "Nature", "cloud": "Nature",
    "baseball": "Objects", "book": "Objects", "bowtie": "Objects", "diamond": "Objects", "door": "Objects", "envelope": "Objects", "eye": "Objects", "hat": "Objects", "scissors": "Objects", "umbrella": "Objects", "key": "Objects", "cup": "Objects", "clock": "Objects", "candle": "Objects",
    "car": "Transport", "bicycle": "Transport", "airplane": "Transport",
    "square": "Shapes",
    "pants": "Clothes", "t-shirt": "Clothes",
    "guitar": "Music", "hammer": "Tools", "bed": "Furniture", "chair": "Furniture", "house": "Buildings",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _self_loop_slug(prefix: str = "item") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(prefix).strip().lower()) or "item"
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{safe}"


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _ensure_self_improving_loop_storage() -> None:
    for folder in (
        SELF_LOOP_ACTIONS_DIR,
        SELF_LOOP_SAMPLES_DIR / "by_label",
        SELF_LOOP_EXPORTS_DIR / "latest",
        SELF_LOOP_JOBS_DIR,
        SELF_LOOP_STATUS_DIR,
        SELF_LOOP_MODELS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)
        keep = folder / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    if not SELF_LOOP_README_PATH.exists():
        SELF_LOOP_README_PATH.write_text(
            "# AirDrawVocab Self-improving Loop storage\n\n"
            "Thư mục này được backend tạo và cập nhật tự động cho khu vực **Self-improving Loop**.\n\n"
            "## Cấu trúc\n"
            "- `samples/stroke_samples.jsonl`: append-only log mọi mẫu vẽ được lưu qua `/game/stroke`.\n"
            "- `samples/by_label/<label>/<sample_id>.json`: bản chi tiết từng mẫu theo nhãn.\n"
            "- `exports/latest/`: bản export mới nhất dùng cho nút **Export data** và link download.\n"
            "- `exports/<timestamp>_export/`: snapshot riêng cho mỗi lần export, không bị ghi đè.\n"
            "- `jobs/<timestamp>_<mode>/`: log, metadata và snapshot model của từng lần **Train stroke** hoặc **Train image**.\n"
            "- `status/retrain_status.json`: trạng thái retrain mới nhất.\n"
            "- `actions/events.jsonl`: nhật ký thao tác: lưu mẫu, export, train, reload model.\n\n"
            "Có thể xóa các snapshot cũ nếu cần tiết kiệm dung lượng; không nên xóa database `data/airdrawvocab_app.sqlite3`.\n",
            encoding="utf-8",
        )


def _self_loop_log_action(action: str, payload: Optional[dict] = None, user: Optional[dict] = None) -> None:
    try:
        _ensure_self_improving_loop_storage()
        item = {
            "at": _now_iso(),
            "action": action,
            "user": (user or {}).get("username") if isinstance(user, dict) else None,
            "user_id": (user or {}).get("id") if isinstance(user, dict) else None,
        }
        if payload:
            item.update(payload)
        _append_jsonl(SELF_LOOP_ACTION_LOG, item)
    except Exception:
        pass


def _self_loop_save_sample(sample: dict, user: Optional[dict] = None) -> None:
    try:
        _ensure_self_improving_loop_storage()
        payload = {"saved_at": _now_iso(), "username": (user or {}).get("username") if user else None, **sample}
        _append_jsonl(SELF_LOOP_SAMPLES_JSONL, payload)
        label = str(sample.get("target") or "unknown").strip().lower().replace("/", "_") or "unknown"
        sample_id = sample.get("sample_id") or _self_loop_slug("sample")
        _write_json_file(SELF_LOOP_SAMPLES_DIR / "by_label" / label / f"{sample_id}.json", payload)
        _self_loop_log_action("save_stroke_sample", {
            "sample_id": sample.get("sample_id"),
            "target": sample.get("target"),
            "predicted": sample.get("predicted"),
            "correct": sample.get("correct"),
            "point_count": sample.get("point_count"),
        }, user)
    except Exception:
        pass


def _self_loop_storage_info() -> dict:
    _ensure_self_improving_loop_storage()
    def count_files(folder: Path) -> int:
        try:
            return sum(1 for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep")
        except Exception:
            return 0

    return {
        "base": str(SELF_IMPROVING_LOOP_DIR),
        "actions": str(SELF_LOOP_ACTION_LOG),
        "samples": str(SELF_LOOP_SAMPLES_JSONL),
        "exports_latest": str(EXPORTED_DATASET_DIR),
        "jobs": str(SELF_LOOP_JOBS_DIR),
        "status": str(RETRAIN_STATUS_PATH),
        "counts": {
            "actions": count_files(SELF_LOOP_ACTIONS_DIR),
            "samples": count_files(SELF_LOOP_SAMPLES_DIR),
            "exports": count_files(SELF_LOOP_EXPORTS_DIR),
            "jobs": count_files(SELF_LOOP_JOBS_DIR),
        },
    }




def _safe_storage_name(value: str, fallback: str = "item") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip().lower())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or fallback


def _panel_slug(prefix: str = "item") -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{_safe_storage_name(prefix)}"


def _ensure_panel_storage() -> None:
    folders = (
        PANEL_ACTIONS_DIR,
        AI_RECOGNITION_STORAGE_DIR,
        REAL_IMAGE_STORAGE_DIR / "by_label",
        AI_JUDGE_STORAGE_DIR,
        SKILL_PROFILE_STORAGE_DIR / "snapshots",
        LEADERBOARD_STORAGE_DIR / "snapshots",
        PVP_STORAGE_DIR / "rooms",
        GAME_SESSION_STORAGE_DIR,
    )
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        keep = folder / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    if not PANEL_STORAGE_README_PATH.exists():
        PANEL_STORAGE_README_PATH.write_text(
            "# AirDrawVocab panel storage\n\n"
            "Thư mục này được backend tạo tự động để lưu dữ liệu của các panel bên phải giao diện.\n\n"
            "## Cấu trúc\n"
            "- `ai_recognition/`: ảnh canvas và JSON kết quả mỗi lần bấm **Nhận diện**.\n"
            "- `real_image_after_draw/by_label/<label>/`: PNG và metadata của ảnh sinh khi bấm **Sinh hình thật**.\n"
            "- `ai_judge_mode/`: điểm Shape/Clarity/Stroke/Speed và feedback của AI Judge.\n"
            "- `skill_profile/`: snapshot hồ sơ kỹ năng, phiên chơi và mẫu vẽ liên quan.\n"
            "- `leaderboard/`: snapshot bảng xếp hạng mới nhất và lịch sử.\n"
            "- `pvp_websocket/`: nhật ký join/leave/message/final score theo phòng PvP.\n"
            "- `game_sessions/`: JSON mỗi lần kết thúc/lưu phiên chơi.\n"
            "- `actions/events.jsonl`: nhật ký tổng hợp mọi thao tác đã ghi vào kho này.\n\n"
            "Đặc biệt, phần **Hình thật sau khi vẽ** chỉ sinh và lưu ảnh khi người dùng bấm nút **Sinh hình thật**; không tự sinh khi đang vẽ.\n",
            encoding="utf-8",
        )


def _panel_log_action(section: str, action: str, payload: Optional[dict] = None, user: Optional[dict] = None) -> None:
    try:
        _ensure_panel_storage()
        item = {
            "at": _now_iso(),
            "section": section,
            "action": action,
            "user": (user or {}).get("username") if isinstance(user, dict) else None,
            "user_id": (user or {}).get("id") if isinstance(user, dict) else None,
        }
        if payload:
            item.update(payload)
        _append_jsonl(PANEL_ACTION_LOG, item)
    except Exception:
        pass


def _data_uri_to_bytes(data_uri: str) -> Optional[bytes]:
    if not data_uri or not isinstance(data_uri, str):
        return None
    try:
        if data_uri.startswith("data:"):
            _, encoded = data_uri.split(",", 1)
        else:
            encoded = data_uri
        return base64.b64decode(encoded)
    except Exception:
        return None


def _panel_storage_info() -> dict:
    _ensure_panel_storage()

    def count_files(folder: Path) -> int:
        try:
            return sum(1 for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep")
        except Exception:
            return 0

    return {
        "base": str(PANEL_STORAGE_DIR),
        "actions": str(PANEL_ACTION_LOG),
        "ai_recognition": str(AI_RECOGNITION_STORAGE_DIR),
        "real_image_after_draw": str(REAL_IMAGE_STORAGE_DIR),
        "ai_judge_mode": str(AI_JUDGE_STORAGE_DIR),
        "skill_profile": str(SKILL_PROFILE_STORAGE_DIR),
        "leaderboard": str(LEADERBOARD_STORAGE_DIR),
        "pvp_websocket": str(PVP_STORAGE_DIR),
        "game_sessions": str(GAME_SESSION_STORAGE_DIR),
        "counts": {
            "actions": count_files(PANEL_ACTIONS_DIR),
            "ai_recognition": count_files(AI_RECOGNITION_STORAGE_DIR),
            "real_image_after_draw": count_files(REAL_IMAGE_STORAGE_DIR),
            "ai_judge_mode": count_files(AI_JUDGE_STORAGE_DIR),
            "skill_profile": count_files(SKILL_PROFILE_STORAGE_DIR),
            "leaderboard": count_files(LEADERBOARD_STORAGE_DIR),
            "pvp_websocket": count_files(PVP_STORAGE_DIR),
            "game_sessions": count_files(GAME_SESSION_STORAGE_DIR),
        },
    }


def _panel_recent_files(section: str = "", limit: int = 20) -> List[dict]:
    _ensure_panel_storage()
    section_map = {
        "ai_recognition": AI_RECOGNITION_STORAGE_DIR,
        "recognition": AI_RECOGNITION_STORAGE_DIR,
        "real_image_after_draw": REAL_IMAGE_STORAGE_DIR,
        "real_image": REAL_IMAGE_STORAGE_DIR,
        "real": REAL_IMAGE_STORAGE_DIR,
        "ai_judge_mode": AI_JUDGE_STORAGE_DIR,
        "judge": AI_JUDGE_STORAGE_DIR,
        "skill_profile": SKILL_PROFILE_STORAGE_DIR,
        "profile": SKILL_PROFILE_STORAGE_DIR,
        "leaderboard": LEADERBOARD_STORAGE_DIR,
        "pvp_websocket": PVP_STORAGE_DIR,
        "pvp": PVP_STORAGE_DIR,
        "game_sessions": GAME_SESSION_STORAGE_DIR,
        "sessions": GAME_SESSION_STORAGE_DIR,
    }
    folder = section_map.get(str(section or "").strip().lower(), PANEL_STORAGE_DIR)
    try:
        files = [p for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep"]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            {
                "name": p.name,
                "path": str(p),
                "relative": str(p.relative_to(PANEL_STORAGE_DIR)),
                "size_bytes": int(p.stat().st_size),
                "modified_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            }
            for p in files[: max(1, min(int(limit), 100))]
        ]
    except Exception:
        return []


def _save_ai_judge_storage(judge: dict, recognition_id: str = "", user: Optional[dict] = None) -> dict:
    try:
        _ensure_panel_storage()
        target = _safe_storage_name(str(judge.get("target") or "unknown"), "unknown")
        grade = _safe_storage_name(str(judge.get("grade") or "grade"), "grade")
        judge_id = _panel_slug(f"{target}_{grade}")
        folder = AI_JUDGE_STORAGE_DIR / judge_id
        payload = {
            "id": judge_id,
            "saved_at": _now_iso(),
            "recognition_id": recognition_id,
            "username": (user or {}).get("username") if user else None,
            "judge": judge,
        }
        _write_json_file(folder / "judge.json", payload)
        _append_jsonl(AI_JUDGE_STORAGE_DIR / "judge_results.jsonl", payload)
        _panel_log_action("ai_judge_mode", "save_judge_result", {
            "judge_id": judge_id,
            "recognition_id": recognition_id,
            "target": judge.get("target"),
            "predicted": judge.get("predicted"),
            "grade": judge.get("grade"),
            "correct": judge.get("correct"),
            "folder": str(folder),
        }, user)
        return {"id": judge_id, "folder": str(folder), "metadata": str(folder / "judge.json")}
    except Exception:
        return {}


def _save_ai_recognition_storage(
    image_bytes: bytes,
    result: dict,
    source: str,
    target: str = "",
    user: Optional[dict] = None,
) -> dict:
    try:
        _ensure_panel_storage()
        label = result.get("label") or target or "prediction"
        recognition_id = _panel_slug(str(label))
        folder = AI_RECOGNITION_STORAGE_DIR / recognition_id
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / "drawing.png"
        image_path.write_bytes(image_bytes)
        payload = {
            "id": recognition_id,
            "saved_at": _now_iso(),
            "username": (user or {}).get("username") if user else None,
            "source": source,
            "target": target,
            "result": result,
            "files": {"drawing": str(image_path), "metadata": str(folder / "prediction.json")},
        }
        _write_json_file(folder / "prediction.json", payload)
        summary = {
            "id": recognition_id,
            "saved_at": payload["saved_at"],
            "username": payload["username"],
            "source": source,
            "target": target,
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "is_correct": result.get("is_correct"),
            "folder": str(folder),
        }
        _append_jsonl(AI_RECOGNITION_STORAGE_DIR / "predictions.jsonl", summary)
        judge_storage = {}
        if isinstance(result.get("judge"), dict):
            judge_storage = _save_ai_judge_storage(result["judge"], recognition_id, user)
        _panel_log_action("ai_recognition", "save_prediction", summary, user)
        return {
            "id": recognition_id,
            "folder": str(folder),
            "drawing": str(image_path),
            "metadata": str(folder / "prediction.json"),
            "judge": judge_storage,
        }
    except Exception:
        return {}


def _save_real_image_storage(
    label: str,
    image_data_uri: str,
    provider: str,
    prompt: str = "",
    reason: str = "manual-generate",
    target: str = "",
    predicted: str = "",
    user: Optional[dict] = None,
    note: str = "",
    error: Optional[str] = None,
) -> dict:
    try:
        _ensure_panel_storage()
        safe_label = _safe_storage_name(label, "unknown")
        image_id = _panel_slug(safe_label)
        folder = REAL_IMAGE_STORAGE_DIR / "by_label" / safe_label
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / f"{image_id}.png"
        metadata_path = folder / f"{image_id}.json"
        raw = _data_uri_to_bytes(image_data_uri)
        if raw:
            image_path.write_bytes(raw)
        payload = {
            "id": image_id,
            "saved_at": _now_iso(),
            "username": (user or {}).get("username") if user else None,
            "label": label,
            "target": target,
            "predicted": predicted,
            "provider": provider,
            "reason": reason,
            "prompt": prompt,
            "note": note,
            "error": error,
            "files": {
                "image": str(image_path) if raw else "",
                "metadata": str(metadata_path),
            },
        }
        _write_json_file(metadata_path, payload)
        _append_jsonl(REAL_IMAGE_STORAGE_DIR / "real_images.jsonl", payload)
        _panel_log_action("real_image_after_draw", "save_real_image", {
            "image_id": image_id,
            "label": label,
            "provider": provider,
            "reason": reason,
            "folder": str(folder),
            "image": str(image_path) if raw else "",
        }, user)
        return {"id": image_id, "folder": str(folder), "image": str(image_path) if raw else "", "metadata": str(metadata_path)}
    except Exception:
        return {}


def _save_skill_profile_snapshot(profile: dict, user: Optional[dict] = None, action: str = "profile_snapshot") -> dict:
    try:
        _ensure_panel_storage()
        profile_id = _panel_slug((user or {}).get("username") or "guest")
        snapshots_dir = SKILL_PROFILE_STORAGE_DIR / "snapshots"
        payload = {"id": profile_id, "saved_at": _now_iso(), "action": action, "profile": profile}
        _write_json_file(SKILL_PROFILE_STORAGE_DIR / "latest.json", payload)
        _write_json_file(snapshots_dir / f"{profile_id}.json", payload)
        _append_jsonl(SKILL_PROFILE_STORAGE_DIR / "profile_snapshots.jsonl", {
            "id": profile_id,
            "saved_at": payload["saved_at"],
            "username": (user or {}).get("username") if user else None,
            "action": action,
            "games": profile.get("stats", {}).get("games") if isinstance(profile.get("stats"), dict) else None,
            "drawings": profile.get("stats", {}).get("drawings") if isinstance(profile.get("stats"), dict) else None,
        })
        _panel_log_action("skill_profile", action, {"profile_id": profile_id, "latest": str(SKILL_PROFILE_STORAGE_DIR / "latest.json")}, user)
        return {"id": profile_id, "latest": str(SKILL_PROFILE_STORAGE_DIR / "latest.json"), "snapshot": str(snapshots_dir / f"{profile_id}.json")}
    except Exception:
        return {}


def _save_leaderboard_snapshot(rows: List[dict]) -> dict:
    try:
        _ensure_panel_storage()
        snapshot_id = _panel_slug("leaderboard")
        snapshots_dir = LEADERBOARD_STORAGE_DIR / "snapshots"
        payload = {"id": snapshot_id, "saved_at": _now_iso(), "leaderboard": rows}
        _write_json_file(LEADERBOARD_STORAGE_DIR / "latest.json", payload)
        _write_json_file(snapshots_dir / f"{snapshot_id}.json", payload)
        _append_jsonl(LEADERBOARD_STORAGE_DIR / "leaderboard_snapshots.jsonl", {
            "id": snapshot_id,
            "saved_at": payload["saved_at"],
            "players": len(rows),
            "top_username": rows[0].get("username") if rows else None,
            "top_score": rows[0].get("score") if rows else None,
        })
        _panel_log_action("leaderboard", "save_leaderboard_snapshot", {"snapshot_id": snapshot_id, "players": len(rows)}, None)
        return {"id": snapshot_id, "latest": str(LEADERBOARD_STORAGE_DIR / "latest.json"), "snapshot": str(snapshots_dir / f"{snapshot_id}.json")}
    except Exception:
        return {}


def _save_game_session_storage(session: dict, user: Optional[dict] = None) -> dict:
    try:
        _ensure_panel_storage()
        session_id = session.get("session_id") or _panel_slug("session")
        filename = f"session_{session_id}.json" if isinstance(session_id, int) or str(session_id).isdigit() else f"{session_id}.json"
        payload = {"saved_at": _now_iso(), "username": (user or {}).get("username") if user else None, "session": session}
        path = GAME_SESSION_STORAGE_DIR / filename
        _write_json_file(path, payload)
        _append_jsonl(GAME_SESSION_STORAGE_DIR / "sessions.jsonl", payload)
        _panel_log_action("game_sessions", "save_game_session", {"session_id": session_id, "path": str(path)}, user)
        return {"path": str(path)}
    except Exception:
        return {}


def _save_pvp_storage(room: str, event: dict, action: str = "message", username: str = "guest") -> dict:
    try:
        _ensure_panel_storage()
        safe_room = _safe_storage_name(room, "room")
        room_dir = PVP_STORAGE_DIR / "rooms" / safe_room
        room_dir.mkdir(parents=True, exist_ok=True)
        payload = {"at": _now_iso(), "room": room, "username": username, "action": action, "event": event}
        _append_jsonl(room_dir / "events.jsonl", payload)
        if action in {"score", "final", "join", "leave"}:
            _write_json_file(room_dir / "latest_event.json", payload)
        _panel_log_action("pvp_websocket", action, {"room": room, "room_dir": str(room_dir), "type": event.get("type")}, {"username": username})
        return {"room_dir": str(room_dir), "events": str(room_dir / "events.jsonl")}
    except Exception:
        return {}

def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    try:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return str(dst)
    except Exception:
        return None
    return None


def _self_loop_snapshot_export(manifest: dict) -> dict:
    snapshot_dir = SELF_LOOP_EXPORTS_DIR / _self_loop_slug("export")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("stroke_samples.jsonl", "stroke_samples.csv", "training_manifest.json"):
        saved = _copy_if_exists(EXPORTED_DATASET_DIR / name, snapshot_dir / name)
        if saved:
            files[name] = saved
    summary = {
        "created_at": _now_iso(),
        "snapshot_dir": str(snapshot_dir),
        "samples": manifest.get("samples"),
        "classes": manifest.get("classes"),
        "files": files,
    }
    _write_json_file(snapshot_dir / "export_summary.json", summary)
    return summary


def _self_loop_create_training_job(mode: str, user: dict, epochs: int, readiness: dict, script: str) -> Tuple[Path, dict]:
    job_dir = SELF_LOOP_JOBS_DIR / _self_loop_slug(mode)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / f"retrain_{mode}.log"
    job_meta = {
        "created_at": _now_iso(),
        "mode": mode,
        "script": script,
        "epochs": epochs,
        "requested_by": user.get("username"),
        "readiness": readiness,
        "log_path": str(log_path),
        "job_dir": str(job_dir),
    }
    _write_json_file(job_dir / "job.json", job_meta)
    _self_loop_log_action("start_retrain", job_meta, user)
    return log_path, job_meta


def _self_loop_finalize_training_job(status_payload: dict, final_status: str, returncode: Optional[int]) -> None:
    try:
        log_path = Path(str(status_payload.get("log") or ""))
        job_dir = log_path.parent if log_path else SELF_LOOP_JOBS_DIR / _self_loop_slug("finished")
        job_dir.mkdir(parents=True, exist_ok=True)
        final_payload = {
            "finished_at": _now_iso(),
            "final_status": final_status,
            "returncode": returncode,
            "status_payload": status_payload,
            "training_readiness": _training_readiness(None) if "_training_readiness" in globals() else {},
        }
        _write_json_file(job_dir / "final_status.json", final_payload)
        mode = str(status_payload.get("mode") or "").lower()
        model_dir = job_dir / "models"
        copied = []
        if mode == "stroke":
            for src in (STROKE_MODEL_PATH, STROKE_CATEGORIES_PATH):
                saved = _copy_if_exists(src, model_dir / src.name)
                if saved:
                    copied.append(saved)
        elif mode == "image":
            for src in (SELF_IMPROVED_MODEL_PATH, SELF_IMPROVED_CATEGORIES_PATH):
                saved = _copy_if_exists(src, model_dir / src.name)
                if saved:
                    copied.append(saved)
        if copied:
            _write_json_file(job_dir / "model_snapshot.json", {"copied_at": _now_iso(), "files": copied})
        _self_loop_log_action("finish_retrain", {
            "mode": mode,
            "final_status": final_status,
            "returncode": returncode,
            "job_dir": str(job_dir),
            "copied_models": copied,
        })
    except Exception:
        pass


def init_app_db() -> None:
    _ensure_self_improving_loop_storage()
    _ensure_panel_storage()
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS game_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                score INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 1,
                streak INTEGER NOT NULL DEFAULT 0,
                accuracy REAL NOT NULL DEFAULT 0,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                mode TEXT NOT NULL DEFAULT 'mouse',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stroke_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                target TEXT NOT NULL,
                predicted TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                correct INTEGER NOT NULL DEFAULT 0,
                mode TEXT NOT NULL DEFAULT 'mouse',
                strokes_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pvp_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room TEXT NOT NULL,
                user_id INTEGER,
                username TEXT NOT NULL DEFAULT 'guest',
                score INTEGER NOT NULL DEFAULT 0,
                target TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                message TEXT NOT NULL DEFAULT '',
                pid INTEGER,
                epochs INTEGER NOT NULL DEFAULT 0,
                samples INTEGER NOT NULL DEFAULT 0,
                classes INTEGER NOT NULL DEFAULT 0,
                log_path TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        def ensure_column(table: str, column: str, ddl: str) -> None:
            cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

        ensure_column("stroke_samples", "judge_json", "judge_json TEXT NOT NULL DEFAULT '{}'")
        ensure_column("stroke_samples", "manual", "manual INTEGER NOT NULL DEFAULT 0")
        ensure_column("stroke_samples", "point_count", "point_count INTEGER NOT NULL DEFAULT 0")


def password_hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000).hex()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions(token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _now_iso()),
        )
    return token


def user_from_request(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username, users.created_at
            FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
    if not row:
        return None
    return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


def set_auth_cookie(response: JSONResponse, token: str) -> JSONResponse:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=AUTH_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


def label_group(label: str) -> str:
    return VOCAB_GROUPS.get(label, "Other")


init_app_db()


# -----------------------------
# Final Boss production systems
# -----------------------------
retrain_lock = threading.Lock()
retrain_process: Optional[subprocess.Popen] = None
retrain_job_id: Optional[int] = None
stroke_sequence_model = None
stroke_sequence_categories: List[str] = []


def _write_retrain_status(status: str, message: str = "", extra: Optional[dict] = None) -> None:
    RETRAIN_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "message": message,
        "updated_at": _now_iso(),
    }
    if extra:
        payload.update(extra)
    RETRAIN_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_retrain_status() -> dict:
    if not RETRAIN_STATUS_PATH.exists():
        return {"status": "idle", "message": "Chưa có job retrain nào.", "updated_at": None}
    try:
        return json.loads(RETRAIN_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unknown", "message": "Không đọc được trạng thái retrain.", "updated_at": None}


def _safe_json_loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def _count_stroke_points(strokes: Any) -> int:
    strokes = _safe_json_loads(strokes, [])
    if not isinstance(strokes, list):
        return 0
    count = 0
    for stroke in strokes:
        if isinstance(stroke, list):
            count += sum(1 for p in stroke if isinstance(p, dict))
    return count


def _tail_file(path: Any, max_lines: int = 14, max_chars: int = 5000) -> str:
    if not path:
        return ""
    file_path = Path(str(path))
    if not file_path.exists():
        return ""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()[-max_lines:]
    tail = "\n".join(lines)
    return tail[-max_chars:]


def _quickdraw_class_count() -> int:
    if not _DATA_DIR.exists():
        return 0
    count = 0
    for label in all_vocab_categories:
        if (_DATA_DIR / f"{label}.npy").exists() or (_DATA_DIR / f"{label.replace(' ', '_')}.npy").exists():
            count += 1
    return count


def _training_readiness(user_id: Optional[int] = None) -> dict:
    where = ""
    params: Tuple[Any, ...] = ()
    if user_id is not None:
        where = "WHERE user_id = ?"
        params = (user_id,)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT target, COUNT(*) AS samples,
                   COALESCE(SUM(correct), 0) AS correct_samples,
                   ROUND(COALESCE(AVG(confidence), 0) * 100, 1) AS avg_confidence,
                   MAX(created_at) AS last_seen
            FROM stroke_samples
            {where}
            GROUP BY target
            ORDER BY samples DESC, target ASC
            """,
            params,
        ).fetchall()
    labels = [dict(r) for r in rows]
    total_samples = int(sum(int(r["samples"]) for r in labels))
    class_count = len(labels)
    min_samples = min([int(r["samples"]) for r in labels], default=0)
    max_samples = max([int(r["samples"]) for r in labels], default=0)
    quickdraw_classes = _quickdraw_class_count()
    ready_stroke = (
        class_count >= TRAINING_MIN_CLASSES
        and total_samples >= TRAINING_MIN_TOTAL_SAMPLES
        and min_samples >= TRAINING_MIN_SAMPLES_PER_CLASS
    )
    ready_image = total_samples >= TRAINING_MIN_TOTAL_SAMPLES or quickdraw_classes >= TRAINING_MIN_CLASSES
    missing_for_stroke = {
        "classes": max(0, TRAINING_MIN_CLASSES - class_count),
        "total_samples": max(0, TRAINING_MIN_TOTAL_SAMPLES - total_samples),
        "samples_per_class": max(0, TRAINING_MIN_SAMPLES_PER_CLASS - min_samples) if class_count else TRAINING_MIN_SAMPLES_PER_CLASS,
    }
    return {
        "total_samples": total_samples,
        "classes": class_count,
        "min_samples_per_class": min_samples,
        "max_samples_per_class": max_samples,
        "quickdraw_classes": quickdraw_classes,
        "ready_stroke": ready_stroke,
        "ready_image": ready_image,
        "requirements": {
            "min_classes": TRAINING_MIN_CLASSES,
            "min_samples_per_class": TRAINING_MIN_SAMPLES_PER_CLASS,
            "min_total_samples": TRAINING_MIN_TOTAL_SAMPLES,
        },
        "missing_for_stroke": missing_for_stroke,
        "label_summary": labels,
    }


def _practice_tip(label: str) -> str:
    hint = _DRAWING_HINTS.get(label) or "Vẽ hình lớn ở giữa khung, nét bao chính rõ, thêm 1-2 chi tiết nhận dạng."
    if label == "book":
        return "Vẽ bìa chữ nhật, gáy sách ở giữa và vài đường trang bên trong để tránh bị nhầm với door."
    if label == "door":
        return "Vẽ khung cửa, tay nắm tròn ở một bên và các ô/panel cửa rõ để tách khỏi book/square."
    if label == "pants":
        return "Vẽ cạp quần, hai ống tách nhau và đường đáy quần rõ; tránh chỉ vẽ một hình chữ nhật."
    if label == "leaf":
        return "Vẽ cuống, gân giữa và vài gân phụ; hình lá nên thuôn ở hai đầu."
    return hint


def _quality_band(accuracy: float, attempts: int) -> str:
    if attempts <= 0:
        return "new"
    if accuracy >= 80:
        return "strong"
    if accuracy >= 50:
        return "learning"
    return "needs_practice"


def _update_training_job(job_id: Optional[int], status: str, message: str = "") -> None:
    if not job_id:
        return
    try:
        with get_db() as conn:
            conn.execute(
                """
                UPDATE training_jobs
                SET status = ?, message = ?, finished_at = COALESCE(finished_at, ?)
                WHERE id = ?
                """,
                (status, message, _now_iso(), int(job_id)),
            )
    except Exception:
        pass


def _recent_training_jobs(limit: int = 5) -> List[dict]:
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT id, mode, status, message, pid, epochs, samples, classes, log_path, started_at, finished_at
                FROM training_jobs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _load_stroke_sequence_model():
    global stroke_sequence_model, stroke_sequence_categories
    if stroke_sequence_model is not None:
        return stroke_sequence_model, stroke_sequence_categories
    if not STROKE_MODEL_PATH.exists() or not STROKE_CATEGORIES_PATH.exists():
        return None, []
    try:
        stroke_sequence_model = load_model(STROKE_MODEL_PATH, compile=False)
        stroke_sequence_categories = json.loads(STROKE_CATEGORIES_PATH.read_text(encoding="utf-8"))
        return stroke_sequence_model, stroke_sequence_categories
    except Exception:
        return None, []


def _flatten_strokes(strokes: Any, max_len: int = 96) -> np.ndarray:
    """Biến strokes -> tensor (1, MAX_LEN, NUM_FEATURES) cho stroke model.

    Delegate sang stroke_features.strokes_to_batch để dùng đúng bộ đặc trưng
    đã train (tránh lệch train/inference). `max_len` giữ lại cho tương thích
    chữ ký cũ nhưng độ dài thực tế lấy từ stroke_features.MAX_LEN.
    """
    return _strokes_to_batch(strokes, canvas_w=CANVAS_W, canvas_h=CANVAS_H)


class PvPRoomManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}
        self.meta: Dict[WebSocket, dict] = {}

    @staticmethod
    def normalize_room(room: str) -> str:
        return (room or "default").strip().lower() or "default"

    async def connect(self, room: str, websocket: WebSocket, username: str):
        await websocket.accept()
        room = self.normalize_room(room)
        self.rooms.setdefault(room, []).append(websocket)
        self.meta[websocket] = {"room": room, "username": username, "score": 0, "level": 1, "target": ""}
        await self.broadcast(room, {"type": "system", "message": f"{username} joined room {room}.", "players": self.players(room)})

    def disconnect(self, websocket: WebSocket):
        meta = self.meta.pop(websocket, None)
        if not meta:
            return None
        room = meta["room"]
        if room in self.rooms and websocket in self.rooms[room]:
            self.rooms[room].remove(websocket)
        return meta

    def update_player(self, websocket: WebSocket, payload: dict) -> dict:
        meta = self.meta.setdefault(websocket, {"room": "default", "username": "guest", "score": 0, "level": 1, "target": ""})
        for key in ("score", "level"):
            if key in payload:
                try:
                    meta[key] = int(payload.get(key) or 0)
                except Exception:
                    pass
        if payload.get("target"):
            meta["target"] = str(payload.get("target"))[:64]
        if payload.get("label"):
            meta["last_prediction"] = str(payload.get("label"))[:64]
        return meta

    def players(self, room: str) -> List[dict]:
        room = self.normalize_room(room)
        players = []
        for ws in self.rooms.get(room, []):
            item = dict(self.meta.get(ws, {}))
            item.pop("room", None)
            players.append(item)
        return sorted(players, key=lambda p: (-int(p.get("score") or 0), str(p.get("username") or "")))

    async def broadcast(self, room: str, payload: dict):
        room = self.normalize_room(room)
        dead = []
        for ws in self.rooms.get(room, []):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


pvp_manager = PvPRoomManager()


@app.middleware("http")
async def add_browser_camera_headers(request, call_next):
    response = await call_next(request)
    # Cho phép chính trang web này gọi camera. Browser vẫn sẽ hỏi quyền người dùng lần đầu.
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    return response

if not MODEL_PATH.exists():
    raise RuntimeError(f"Không tìm thấy model: {MODEL_PATH}")

if not CATEGORIES_PATH.exists():
    raise RuntimeError(f"Không tìm thấy categories.json: {CATEGORIES_PATH}")

with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    trained_categories_raw: List[str] = json.load(f)

# compile=False giúp tránh lỗi khi môi trường TensorFlow/Keras khác phiên bản lúc train.
model = load_model(MODEL_PATH, compile=False)
live_drawing_model = model
if LIVE_DRAWING_MODEL_PATH != MODEL_PATH:
    live_drawing_model = load_model(LIVE_DRAWING_MODEL_PATH, compile=False)

ACTIVE_MODEL_PATH = MODEL_PATH
ACTIVE_LIVE_MODEL_PATH = LIVE_DRAWING_MODEL_PATH
ACTIVE_CATEGORIES_PATH = CATEGORIES_PATH


def _output_class_count(active_model) -> int:
    """Lấy số lớp output thực tế của model để tránh lệch nhãn khi vocab đã mở rộng."""
    shape = getattr(active_model, "output_shape", None)
    try:
        if isinstance(shape, (list, tuple)) and shape:
            last = shape[-1]
            if isinstance(last, (list, tuple)):
                last = last[-1]
            if last is not None:
                return int(last)
    except Exception:
        pass
    return len(trained_categories_raw)

MODEL_OUTPUT_CLASSES = _output_class_count(model)
LIVE_MODEL_OUTPUT_CLASSES = _output_class_count(live_drawing_model)
PREDICTION_CLASS_COUNT = min(MODEL_OUTPUT_CLASSES, LIVE_MODEL_OUTPUT_CLASSES, len(trained_categories_raw))

# categories = các nhãn model hiện tại có thể nhận diện được.
# all_vocab_categories = toàn bộ từ vựng dự án định nghĩa trong vocab_pairs.py, hiện là 40.
categories: List[str] = list(trained_categories_raw[:PREDICTION_CLASS_COUNT])
all_vocab_categories: List[str] = list(_ALL_VOCAB_CATEGORIES or trained_categories_raw)
recognition_category_set = set(categories)
all_vocab_category_set = set(all_vocab_categories)


def _runtime_model_info() -> dict:
    return {
        "model_path": str(ACTIVE_MODEL_PATH),
        "live_model_path": str(ACTIVE_LIVE_MODEL_PATH),
        "categories_path": str(ACTIVE_CATEGORIES_PATH),
        "num_recognition_categories": len(categories),
        "model_output_classes": MODEL_OUTPUT_CLASSES,
        "live_model_output_classes": LIVE_MODEL_OUTPUT_CLASSES,
    }


def _reload_image_runtime(model_path: Path, categories_path: Path) -> dict:
    """Nạp model ảnh mới sau khi Train image mà không cần tắt server."""
    global model, live_drawing_model, trained_categories_raw, categories, recognition_category_set
    global MODEL_OUTPUT_CLASSES, LIVE_MODEL_OUTPUT_CLASSES, PREDICTION_CLASS_COUNT
    global ACTIVE_MODEL_PATH, ACTIVE_LIVE_MODEL_PATH, ACTIVE_CATEGORIES_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Không tìm thấy categories: {categories_path}")
    new_categories = json.loads(categories_path.read_text(encoding="utf-8"))
    if not isinstance(new_categories, list) or not new_categories:
        raise ValueError("File categories không hợp lệ.")

    new_model = load_model(model_path, compile=False)
    new_class_count = _output_class_count(new_model)
    usable_count = min(new_class_count, len(new_categories))
    if usable_count <= 0:
        raise ValueError("Model mới không có lớp output hợp lệ.")

    model = new_model
    live_drawing_model = new_model
    trained_categories_raw = list(new_categories)
    MODEL_OUTPUT_CLASSES = new_class_count
    LIVE_MODEL_OUTPUT_CLASSES = new_class_count
    PREDICTION_CLASS_COUNT = usable_count
    categories = list(trained_categories_raw[:usable_count])
    recognition_category_set = set(categories)
    ACTIVE_MODEL_PATH = model_path
    ACTIVE_LIVE_MODEL_PATH = model_path
    ACTIVE_CATEGORIES_PATH = categories_path
    return _runtime_model_info()


face_auth = FaceAuthManager(PROJECT_ROOT / "face_data")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def home():
    """Mở giao diện vẽ nếu có frontend, nếu không trả về trạng thái API."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        # KHÔNG cache index.html: nếu cache, trình duyệt sẽ giữ bản cũ và không
        # bao giờ tải app.js/style.css mới (dù đã đổi ?v=...). Buộc revalidate.
        return FileResponse(
            index_file,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return {"message": "AirDrawVocab API is running", "docs": "/docs"}


@app.post("/auth/register")
async def auth_register(username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Tên đăng nhập cần ít nhất 3 ký tự.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Mật khẩu cần ít nhất 6 ký tự.")

    salt = secrets.token_hex(16)
    hashed = password_hash(password, salt)
    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, hashed, salt, _now_iso()),
            )
            user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại.") from exc

    token = create_session(user_id)
    response = JSONResponse({"ok": True, "user": {"id": user_id, "username": username}})
    return set_auth_cookie(response, token)


@app.post("/auth/login")
async def auth_login(username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row or password_hash(password, row["salt"]) != row["password_hash"]:
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")

    token = create_session(int(row["id"]))
    response = JSONResponse({"ok": True, "user": {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}})
    return set_auth_cookie(response, token)


@app.post("/auth/logout")
async def auth_logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.get("/auth/me")
def auth_me(request: Request):
    user = user_from_request(request)
    return {"authenticated": bool(user), "user": user}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_categories": len(categories),
        "num_recognition_categories": len(categories),
        "num_vocab_categories": len(all_vocab_categories),
        "categories": categories,
        "all_vocab_categories": all_vocab_categories,
        "trained_categories_file_count": len(trained_categories_raw),
        "model_output_classes": MODEL_OUTPUT_CLASSES,
        "live_model_output_classes": LIVE_MODEL_OUTPUT_CLASSES,
        "model_input_shape": str(getattr(model, "input_shape", "unknown")),
        "model_path": str(ACTIVE_MODEL_PATH),
        "live_drawing_model_path": str(ACTIVE_LIVE_MODEL_PATH),
        "categories_path": str(ACTIVE_CATEGORIES_PATH),
        "self_improved_model_exists": SELF_IMPROVED_MODEL_PATH.exists(),
        "app_db_path": str(APP_DB_PATH),
        "face_model_exists": (PROJECT_ROOT / "face_data" / "lbph_face_model.yml").exists(),
        "camera_face_strokes": {
            "available": FACE_STROKE_DETECTOR is not None,
            "detector": "browser FaceMesh + OpenCV Haar fallback",
            "endpoint": "/camera/face-strokes",
        },
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
    for label in all_vocab_categories:
        items.append({
            "label": label,
            "meaning_vi": VI_MEANINGS.get(label, label),
            "ipa": _IPA.get(label, ""),
            "example_en": EXAMPLE_SENTENCES.get(label, f"This is a {label}."),
            "example_vi": _EXVI.get(label, ""),
            "drawing_hint": _DRAWING_HINTS.get(label, ""),
            "recognition_supported": label in recognition_category_set,
        })
    return {
        "count": len(items),
        "recognition_count": len(categories),
        "vocab": items,
    }


@app.get("/analytics")
def analytics(request: Request):
    """Dashboard dữ liệu cho giao diện demo: phân bố 40 từ, model hiện tại và lịch sử dự đoán."""
    user = user_from_request(request)
    group_counts: Dict[str, int] = {}
    for label in all_vocab_categories:
        group = label_group(label)
        group_counts[group] = group_counts.get(group, 0) + 1

    with get_db() as conn:
        total_predictions = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        rows = conn.execute(
            """
            SELECT label, COUNT(*) AS count, ROUND(AVG(confidence) * 100, 2) AS avg_confidence
            FROM predictions
            GROUP BY label
            ORDER BY count DESC, label ASC
            LIMIT 8
            """
        ).fetchall()
        user_predictions = 0
        if user:
            user_predictions = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions WHERE user_id = ?",
                (user["id"],),
            ).fetchone()["c"]

    # Ma trận demo phục vụ phần trình bày giao diện. Khi có bộ test 40 lớp thật,
    # thay bằng kết quả evaluate_model.py/confusion_matrix từ model hiện tại.
    demo_labels = categories[:8] if categories else all_vocab_categories[:8]
    matrix = []
    for i, label in enumerate(demo_labels):
        row = []
        for j, _ in enumerate(demo_labels):
            if i == j:
                value = max(8, 34 - (i % 4) * 3)
            elif abs(i - j) == 1:
                value = 2 + ((i + j) % 3)
            else:
                value = (i * j) % 2
            row.append(value)
        matrix.append(row)

    return {
        "total_vocab": len(all_vocab_categories),
        "recognition_count": len(categories),
        "model_output_classes": MODEL_OUTPUT_CLASSES,
        "trained_categories_file_count": len(trained_categories_raw),
        "groups": [{"name": name, "count": count} for name, count in sorted(group_counts.items())],
        "top_predictions": [dict(row) for row in rows],
        "total_predictions": total_predictions,
        "user_predictions": user_predictions,
        "user": user,
        "confusion_matrix_demo": {"labels": demo_labels, "matrix": matrix},
    }


def log_prediction(user: Optional[dict], label: str, confidence: float, source: str) -> None:
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO predictions(user_id, label, confidence, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (user["id"] if user else None, label, float(confidence), source, _now_iso()),
            )
    except Exception:
        # Không để lỗi logging làm hỏng chức năng nhận diện chính.
        pass


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
    """Chuyển ảnh canvas/user upload về đúng input model: 28x28 grayscale, nền đen nét trắng.

    Logic dùng chung ở image_preprocess.preprocess_drawing để backend và các script
    đánh giá preprocess giống hệt nhau (tránh lệch production/evaluation).
    """
    try:
        return _preprocess_drawing(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def _deterministic_tta_batches(x: np.ndarray, max_augments: int) -> List[np.ndarray]:
    """Sinh các biến thể dịch nhẹ CỐ ĐỊNH để dự đoán realtime ổn định hơn.

    Bản cũ dùng random_crop nên cùng một nét vẽ có thể nhảy nhãn giữa các tick.
    Với game realtime, độ ổn định quan trọng hơn augmentation ngẫu nhiên.
    """
    if max_augments <= 0:
        return []
    try:
        image = np.asarray(x[0, :, :, 0], dtype="float32")
    except Exception:
        return []
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    batches: List[np.ndarray] = []
    for dx, dy in shifts[:max_augments]:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, matrix, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
        batches.append(shifted.reshape(1, 28, 28, 1).astype("float32"))
    return batches


def predict_proba(active_model, x: np.ndarray) -> np.ndarray:
    """Dự đoán xác suất với TTA ổn định, không dùng random crop.

    Mục tiêu: giảm việc AI nhảy từ `book` sang `pants/door` chỉ vì một lần crop
    ngẫu nhiên làm mất các nét trang sách nhỏ.
    """
    base = active_model.predict(x, verbose=0)[0]
    if not USE_TTA or TTA_SHIFTS <= 0:
        return base
    probs = [base]
    for xs in _deterministic_tta_batches(x, TTA_SHIFTS):
        probs.append(active_model.predict(xs, verbose=0)[0])
    return np.mean(probs, axis=0)


def _top3_from_predictions(preds: np.ndarray) -> List[dict]:
    usable_count = min(len(preds), len(categories))
    if usable_count <= 0:
        return []
    usable_preds = preds[:usable_count]
    top_indices = usable_preds.argsort()[-3:][::-1]
    return [
        {
            "label": categories[int(i)],
            "meaning_vi": VI_MEANINGS.get(categories[int(i)], categories[int(i)]),
            "confidence": float(usable_preds[int(i)]),
        }
        for i in top_indices
    ]


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), source: str = Form("canvas")):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi ảnh lên.")

    x = preprocess_image(image_bytes)
    enhanced_drawing = enhance_drawing_image(image_bytes)
    normalized_source = source.strip().lower()
    active_model = live_drawing_model if normalized_source in {"camera", "camera-hand", "camera-face", "face", "hand", "airdraw", "live"} else model
    active_model_path = ACTIVE_LIVE_MODEL_PATH if active_model is live_drawing_model else ACTIVE_MODEL_PATH
    preds = predict_proba(active_model, x)

    usable_count = min(len(preds), len(categories))
    if usable_count <= 0:
        raise HTTPException(status_code=500, detail="Model chưa có nhãn nhận diện hợp lệ.")
    usable_preds = preds[:usable_count]
    best_index = int(np.argmax(usable_preds))
    label = categories[best_index]
    confidence = float(usable_preds[best_index])

    top3 = _top3_from_predictions(preds)

    reply = foza_chatbot_reply(label, confidence, top3)
    user = user_from_request(request)
    log_prediction(user, label, confidence, normalized_source)

    result = {
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
    result["storage"] = _save_ai_recognition_storage(image_bytes, result, normalized_source, "", user)
    return result





# -----------------------------
# Hybrid visual reranker for gameplay
# -----------------------------
# CNN QuickDraw rất dễ nhầm các lớp có hình hộp/đường dọc giống nhau: book,
# door, pants, square, envelope. Trong game ta biết từ mục tiêu, nên backend có
# thể dùng thêm đặc trưng hình học đơn giản để kiểm tra "bản vẽ có giống mục tiêu
# không" thay vì chỉ lấy top-1 CNN. Đây KHÔNG thay thế model; nó chỉ rerank khi
# hình học của target đủ rõ.
SHAPE_RERANK_ENABLED = os.getenv("SHAPE_RERANK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
SHAPE_RERANK_MIN_SCORE = float(os.getenv("SHAPE_RERANK_MIN_SCORE", "0.64"))
SHAPE_RERANK_STRENGTH = float(os.getenv("SHAPE_RERANK_STRENGTH", "0.78"))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _score_between(value: float, best: float, spread: float) -> float:
    if spread <= 0:
        return 0.0
    return _clamp01(1.0 - abs(float(value) - best) / spread)


def _drawing_binary_mask(image_bytes: bytes, size: int = 96) -> Tuple[Optional[np.ndarray], dict]:
    """Canvas bytes -> binary mask đã crop/resize để đo đặc trưng hình học."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        return None, {"error": "invalid_image"}

    rgba = np.array(image)
    rgb = rgba[:, :, :3].astype(np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.uint8)
    rgb = (rgb.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.mean()) > 127:
        gray = 255 - gray

    _, rough = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(rough)
    if coords is None:
        return None, {"empty": True}

    x, y, w, h = cv2.boundingRect(coords)
    pad = int(max(w, h) * 0.08) + 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None, {"empty": True}

    # Otsu + close/dilate nhẹ để giữ các nét trong khi resize.
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if max(w, h) > 80:
        mask = cv2.dilate(mask, kernel, iterations=1)

    side = max(mask.shape[:2])
    square = np.zeros((side, side), dtype=np.uint8)
    oy = (side - mask.shape[0]) // 2
    ox = (side - mask.shape[1]) // 2
    square[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized, {
        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "aspect": round(float(w) / max(1.0, float(h)), 4),
        "ink_ratio": round(float(np.count_nonzero(resized)) / float(size * size), 4),
    }


def _line_component_features(mask: np.ndarray) -> dict:
    """Đếm các nét dọc/ngang/chéo chính trong mask 96x96."""
    if mask is None or mask.size == 0:
        return {}
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8) * 255

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, h // 9)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, w // 9), 1))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    def comp_boxes(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw * bh >= 8:
                boxes.append((int(x), int(y), int(bw), int(bh)))
        return boxes

    v_boxes = comp_boxes(vertical)
    h_boxes = comp_boxes(horizontal)
    strong_v = [b for b in v_boxes if b[3] >= h * 0.34]
    strong_h = [b for b in h_boxes if b[2] >= w * 0.12]
    inner_h = [b for b in strong_h if h * 0.18 <= b[1] <= h * 0.82]
    left_inner_h = [b for b in inner_h if (b[0] + b[2] / 2) <= w * 0.62]
    center_v_scores = []
    for x, y, bw, bh in strong_v:
        center = (x + bw / 2) / max(1, w)
        center_v_scores.append(_score_between(center, 0.5, 0.18) * _clamp01(bh / max(1, h * 0.55)))
    center_vertical = max(center_v_scores) if center_v_scores else 0.0

    # Hough để nhận các nét chéo: envelope/lightning/star.
    edges = cv2.Canny(binary, 50, 150)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=16, minLineLength=max(10, w // 6), maxLineGap=5)
    diag_count = 0
    if raw_lines is not None:
        for line in raw_lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx >= w * 0.12 and dy >= h * 0.12:
                diag_count += 1

    bottom = binary[int(h * 0.58):, :]
    col = (bottom > 0).sum(axis=0)
    active = col > max(2, bottom.shape[0] * 0.10)
    clusters = []
    start = None
    for i, val in enumerate(active.tolist() + [False]):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= max(3, w // 18):
                clusters.append((start, i - 1))
            start = None
    lower_two_clusters = len(clusters) >= 2
    central_gap = 0.0
    if lower_two_clusters:
        for a, b in zip(clusters, clusters[1:]):
            gap_mid = (a[1] + b[0]) / 2 / max(1, w)
            gap_w = (b[0] - a[1]) / max(1, w)
            central_gap = max(central_gap, _score_between(gap_mid, 0.5, 0.18) * _clamp01(gap_w / 0.16))

    return {
        "vertical_count": len(strong_v),
        "horizontal_count": len(strong_h),
        "inner_horizontal_count": len(inner_h),
        "left_inner_horizontal_count": len(left_inner_h),
        "center_vertical": round(float(center_vertical), 4),
        "diagonal_count": int(diag_count),
        "bottom_clusters": int(len(clusters)),
        "central_bottom_gap": round(float(central_gap), 4),
        "vertical_boxes": strong_v[:8],
        "horizontal_boxes": strong_h[:8],
    }


def _shape_scores_for_mask(mask: np.ndarray, meta: dict) -> dict:
    if mask is None:
        return {}
    lines = _line_component_features(mask)
    aspect = float(meta.get("aspect") or 1.0)
    ink_ratio = float(meta.get("ink_ratio") or 0.0)
    aspect_square = _score_between(aspect, 1.0, 0.55)
    aspect_book = _score_between(aspect, 0.78, 0.70)
    aspect_tall = _clamp01((1.0 / max(0.25, aspect) - 0.9) / 0.65)
    page_lines = _clamp01(lines.get("left_inner_horizontal_count", 0) / 3.0)
    center_spine = float(lines.get("center_vertical") or 0.0)
    horizontal = _clamp01(lines.get("horizontal_count", 0) / 4.0)
    diagonals = _clamp01(lines.get("diagonal_count", 0) / 4.0)
    leg_gap = float(lines.get("central_bottom_gap") or 0.0)

    book = _clamp01(0.38 * center_spine + 0.34 * page_lines + 0.18 * aspect_book + 0.10 * _score_between(ink_ratio, 0.16, 0.16))
    # Nếu có nhiều trang sách bên trái thì giảm nhầm sang door/pants.
    door = _clamp01(0.42 * aspect_tall + 0.30 * center_spine + 0.16 * horizontal + 0.12 * _score_between(ink_ratio, 0.18, 0.16) - 0.24 * page_lines)
    pants = _clamp01(0.44 * leg_gap + 0.30 * aspect_tall + 0.18 * _clamp01(lines.get("vertical_count", 0) / 4.0) - 0.30 * page_lines)
    square = _clamp01(0.62 * aspect_square + 0.20 * _score_between(ink_ratio, 0.12, 0.12) - 0.20 * (page_lines + center_spine) / 2)
    envelope = _clamp01(0.45 * aspect_square + 0.35 * diagonals + 0.15 * horizontal)
    lightning = _clamp01(0.68 * diagonals + 0.18 * _score_between(ink_ratio, 0.10, 0.12) + 0.14 * (1 - aspect_square))

    return {
        "book": round(book, 4),
        "door": round(door, 4),
        "pants": round(pants, 4),
        "square": round(square, 4),
        "envelope": round(envelope, 4),
        "lightning": round(lightning, 4),
        "features": {**meta, **{k: v for k, v in lines.items() if not k.endswith("boxes")}},
    }


def _target_rank(preds: np.ndarray, target: str) -> Tuple[int, float]:
    if not target or target not in categories:
        return 999, 0.0
    usable_count = min(len(preds), len(categories))
    usable_preds = preds[:usable_count]
    target_index = categories.index(target)
    order = usable_preds.argsort()[::-1].tolist()
    rank = order.index(target_index) + 1 if target_index in order else 999
    return int(rank), float(usable_preds[target_index])


def _topn_from_score_map(score_map: Dict[str, float], n: int = 5) -> List[dict]:
    rows = []
    for label, score in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:n]:
        rows.append({
            "label": label,
            "meaning_vi": VI_MEANINGS.get(label, label),
            "confidence": float(score),
            "confidence_percent": round(float(score) * 100, 2),
        })
    return rows


def _rerank_game_prediction(preds: np.ndarray, image_bytes: bytes, target: str) -> dict:
    """Kết hợp CNN + đặc trưng hình học mục tiêu cho /predict_godmode."""
    top5_raw = _topn_from_predictions(preds, 5)
    usable_count = min(len(preds), len(categories))
    if usable_count <= 0:
        return {"label": "", "confidence": 0.0, "top5": [], "top5_raw": top5_raw, "rerank": {"used": False}}

    raw_idx = int(np.argmax(preds[:usable_count]))
    raw_label = categories[raw_idx]
    raw_conf = float(preds[raw_idx])
    target_label = (target or "").strip().lower()

    if not SHAPE_RERANK_ENABLED or not target_label or target_label not in categories:
        return {
            "label": raw_label,
            "confidence": raw_conf,
            "top5": top5_raw,
            "top5_raw": top5_raw,
            "rerank": {"used": False, "reason": "disabled_or_no_target"},
        }

    mask, meta = _drawing_binary_mask(image_bytes)
    shape_scores = _shape_scores_for_mask(mask, meta) if mask is not None else {}
    target_shape = float(shape_scores.get(target_label, 0.0) or 0.0)
    target_rank, target_cnn = _target_rank(preds, target_label)

    score_map: Dict[str, float] = {categories[i]: float(preds[i]) for i in range(usable_count)}
    used = False
    reason = "cnn_only"

    # Điều kiện an toàn: chỉ nâng target khi hình học đủ rõ, và target đã có tín
    # hiệu trong CNN hoặc nằm trong nhóm hình học đang hỗ trợ. Như vậy game không
    # tự cho qua nếu người chơi chưa vẽ gì giống mục tiêu.
    supported_target = target_label in {"book", "door", "pants", "square", "envelope", "lightning"}
    if supported_target and target_shape >= SHAPE_RERANK_MIN_SCORE:
        pseudo_conf = min(0.93, max(target_cnn, 0.32 + SHAPE_RERANK_STRENGTH * target_shape))
        if pseudo_conf > score_map.get(target_label, 0.0):
            score_map[target_label] = pseudo_conf
            used = True
            reason = f"shape_match_{target_label}_{round(target_shape * 100)}pct"

        # Giảm nhẹ các lớp dễ nhầm khi target có chi tiết riêng rõ.
        if target_label == "book" and target_shape >= SHAPE_RERANK_MIN_SCORE:
            page_lines = float(shape_scores.get("features", {}).get("left_inner_horizontal_count") or 0)
            if page_lines >= 2:
                for confuse in ("door", "pants", "square"):
                    if confuse in score_map:
                        score_map[confuse] = float(score_map[confuse]) * 0.72
        elif target_label == "pants" and target_shape >= SHAPE_RERANK_MIN_SCORE:
            for confuse in ("book", "door", "square"):
                if confuse in score_map:
                    score_map[confuse] = float(score_map[confuse]) * 0.80

    top5 = _topn_from_score_map(score_map, 5)
    best = top5[0] if top5 else {"label": raw_label, "confidence": raw_conf}
    return {
        "label": best["label"],
        "confidence": float(best["confidence"]),
        "top5": top5,
        "top5_raw": top5_raw,
        "rerank": {
            "used": bool(used),
            "reason": reason,
            "target_shape_score": round(target_shape, 4),
            "target_cnn_confidence": round(float(target_cnn), 4),
            "target_cnn_rank": int(target_rank),
            "raw_label": raw_label,
            "raw_confidence": round(raw_conf, 4),
            "shape_scores": {k: v for k, v in shape_scores.items() if k != "features"},
            "features": shape_scores.get("features", {}),
        },
    }


def _enhanced_teacher_feedback(target: str, predicted: str, confidence: float, correct: bool, stroke_count: int, rerank: Optional[dict] = None) -> str:
    base = _teacher_feedback(target, predicted, confidence, correct, stroke_count)
    if not rerank:
        return base
    if rerank.get("used") and correct:
        raw_label = rerank.get("raw_label")
        raw_conf = float(rerank.get("raw_confidence") or 0)
        return (
            f"Đã dùng thêm kiểm tra hình học mục tiêu để giảm nhầm với '{raw_label}' "
            f"(CNN gốc {round(raw_conf * 100)}%). {base}"
        )
    if target == "book" and not correct:
        return base + " Với 'book', hãy vẽ bìa chữ nhật, gáy sách ở giữa và 2-3 đường trang nằm trong nửa trái."
    return base


def _topn_from_predictions(preds: np.ndarray, n: int = 5) -> List[dict]:
    usable_count = min(len(preds), len(categories))
    if usable_count <= 0:
        return []
    usable_preds = preds[:usable_count]
    top_indices = usable_preds.argsort()[-n:][::-1]
    return [
        {
            "label": categories[int(i)],
            "meaning_vi": VI_MEANINGS.get(categories[int(i)], categories[int(i)]),
            "confidence": float(usable_preds[int(i)]),
            "confidence_percent": round(float(usable_preds[int(i)]) * 100, 2),
        }
        for i in top_indices
    ]


def _teacher_feedback(target: str, predicted: str, confidence: float, correct: bool, stroke_count: int) -> str:
    if correct and confidence >= 0.82:
        return "Rất tốt. Nét vẽ rõ, AI nhận đúng mục tiêu. Tiếp tục giữ tốc độ và cấu trúc hình như vậy."
    if correct:
        return "Đúng mục tiêu. Nên vẽ thêm đường bao chính để confidence ổn định hơn."
    if stroke_count <= 4:
        return "Nét vẽ còn ít. Hãy vẽ khung chính của vật thể trước, sau đó thêm chi tiết nhận dạng."
    if confidence >= 0.55:
        return f"AI đang nghiêng về '{predicted}'. Hãy thêm đặc điểm riêng của '{target}' để phân biệt rõ hơn."
    return "AI chưa chắc chắn. Hãy vẽ nét lớn, ít rối, ưu tiên hình dạng tổng thể thay vì chi tiết nhỏ."


def _judge_payload(
    target: str,
    predicted: str,
    confidence: float,
    correct: bool,
    stroke_count: int,
    elapsed_ms: int,
    rerank: Optional[dict] = None,
) -> dict:
    clarity = min(100, max(5, round(confidence * 100)))
    stroke_score = min(100, 35 + stroke_count * 7)
    speed_score = 100 if elapsed_ms <= 12000 else max(30, 100 - int((elapsed_ms - 12000) / 500))
    shape_bonus = 0
    if rerank and rerank.get("target_shape_score") is not None:
        shape_bonus = int(max(0, float(rerank.get("target_shape_score") or 0) - 0.55) * 22)
    shape_score = min(100, round((clarity * 0.56) + (stroke_score * 0.24) + (speed_score * 0.14) + shape_bonus))
    if correct:
        grade = "S" if shape_score >= 88 else "A" if shape_score >= 72 else "B"
    else:
        grade = "C" if confidence >= 0.45 else "D"
    return {
        "target": target,
        "predicted": predicted,
        "correct": correct,
        "shape_score": shape_score,
        "clarity_score": clarity,
        "stroke_score": stroke_score,
        "speed_score": speed_score,
        "grade": grade,
        "feedback": _enhanced_teacher_feedback(target, predicted, confidence, correct, stroke_count, rerank),
        "rerank": rerank or {"used": False},
    }


@app.post("/predict_godmode")
async def predict_godmode(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form(""),
    source: str = Form("realtime"),
    stroke_count: int = Form(0),
    elapsed_ms: int = Form(0),
):
    """Realtime prediction nhẹ cho game: trả top5 + AI Judge, không gọi chatbot để tránh chậm."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi ảnh lên.")

    x = preprocess_image(image_bytes)
    normalized_source = source.strip().lower()
    active_model = live_drawing_model if normalized_source in {"camera", "camera-hand", "camera-face", "face", "hand", "airdraw", "live", "realtime"} else model
    preds = predict_proba(active_model, x)
    usable_count = min(len(preds), len(categories))
    if usable_count <= 0:
        raise HTTPException(status_code=500, detail="Model chưa có nhãn nhận diện hợp lệ.")

    target_label = target.strip().lower()
    reranked = _rerank_game_prediction(preds, image_bytes, target_label)
    label = reranked["label"]
    confidence = float(reranked["confidence"])
    correct = bool(target_label and label == target_label)
    top5 = reranked["top5"]
    rerank_info = reranked.get("rerank", {"used": False})
    judge = _judge_payload(
        target_label,
        label,
        confidence,
        correct,
        max(0, int(stroke_count)),
        max(0, int(elapsed_ms)),
        rerank_info,
    )
    ai_source = "image-cnn+shape-rerank" if rerank_info.get("used") else "image-cnn"
    result = {
        "ok": True,
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "top5": top5,
        "top5_raw": reranked.get("top5_raw", top5),
        "target": target_label,
        "is_correct": correct,
        "judge": judge,
        "ai_source": ai_source,
        "rerank": rerank_info,
        "raw_label": rerank_info.get("raw_label", label),
        "raw_confidence": rerank_info.get("raw_confidence", confidence),
    }
    result["storage"] = _save_ai_recognition_storage(
        image_bytes,
        {k: v for k, v in result.items() if k != "storage"},
        normalized_source,
        target_label,
        user_from_request(request),
    )
    return result


@app.post("/camera/face/analyze")
async def camera_face_analyze(file: UploadFile = File(...)):
    """Nhận diện khuôn mặt từ frame camera để hỗ trợ Bút mặt.

    Endpoint này port ý tưởng từ DeepShieldAI-Pro: dùng OpenCV Haar cascade để
    tìm mặt lớn nhất, crop có padding, trả bbox + chất lượng ảnh. Frontend dùng
    metadata này để báo người chơi mặt đang rõ/xa/tối/mờ khi vẽ bằng camera.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi frame camera.")

    frame = _decode_upload_image_bgr(image_bytes)
    face_crop, metadata = _crop_camera_face_or_frame(frame)
    frame_quality = _camera_image_quality(frame)
    face_quality = _camera_image_quality(face_crop)
    feedback = _camera_face_feedback(metadata, frame.shape, face_quality if metadata.get("faceDetected") else frame_quality)
    frame_h, frame_w = frame.shape[:2]

    return {
        "ok": True,
        "source": "opencv-haar-deepshield-style",
        "faceDetectorAvailable": CAMERA_FACE_DETECTOR is not None,
        "faceDetected": bool(metadata.get("faceDetected")),
        "bbox": metadata.get("bbox"),
        "faceBbox": metadata.get("faceBbox"),
        "cropSize": metadata.get("cropSize"),
        "frameSize": {"width": int(frame_w), "height": int(frame_h)},
        "quality": {"frame": frame_quality, "face": face_quality},
        **feedback,
    }


@app.get("/game/profile")
def game_profile(request: Request):
    user = user_from_request(request)
    if not user:
        return {
            "authenticated": False,
            "message": "Đăng nhập để lưu hồ sơ kỹ năng.",
            "stats": {"games": 0, "best_score": 0, "drawings": 0, "accuracy": 0},
            "strengths": [],
            "weaknesses": [],
            "practice_plan": [],
            "training": _training_readiness(None),
        }

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS games, COALESCE(MAX(score), 0) AS best_score,
                   COALESCE(AVG(score), 0) AS avg_score,
                   COALESCE(MAX(streak), 0) AS best_streak,
                   COALESCE(AVG(accuracy), 0) AS avg_accuracy,
                   COALESCE(SUM(duration_seconds), 0) AS total_seconds,
                   MAX(created_at) AS last_played
            FROM game_sessions WHERE user_id = ?
            """,
            (user["id"],),
        ).fetchone()
        sample_row = conn.execute(
            """
            SELECT COUNT(*) AS drawings, COALESCE(AVG(correct), 0) AS acc,
                   COALESCE(AVG(confidence), 0) AS avg_confidence,
                   COALESCE(SUM(point_count), 0) AS total_points
            FROM stroke_samples WHERE user_id = ?
            """,
            (user["id"],),
        ).fetchone()
        label_rows = conn.execute(
            """
            SELECT target, COUNT(*) AS attempts,
                   ROUND(AVG(correct) * 100, 1) AS accuracy,
                   ROUND(AVG(confidence) * 100, 1) AS avg_confidence,
                   COALESCE(SUM(correct), 0) AS correct_samples,
                   COALESCE(SUM(point_count), 0) AS point_count,
                   MAX(created_at) AS last_seen
            FROM stroke_samples
            WHERE user_id = ?
            GROUP BY target
            HAVING attempts >= 1
            """,
            (user["id"],),
        ).fetchall()
        recent_sessions = conn.execute(
            """
            SELECT score, level, streak, accuracy, duration_seconds, mode, created_at
            FROM game_sessions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (user["id"],),
        ).fetchall()

    labels = [dict(r) for r in label_rows]
    for item in labels:
        item["accuracy"] = float(item.get("accuracy") or 0)
        item["avg_confidence"] = float(item.get("avg_confidence") or 0)
        item["attempts"] = int(item.get("attempts") or 0)
        item["band"] = _quality_band(item["accuracy"], item["attempts"])
        item["tip"] = _practice_tip(item["target"])

    strengths = sorted(labels, key=lambda r: (-r["accuracy"], -r["avg_confidence"], -r["attempts"], r["target"]))[:5]
    weaknesses = sorted(labels, key=lambda r: (r["accuracy"], -r["attempts"], r["target"]))[:5]
    practice_plan = [
        {
            "target": item["target"],
            "accuracy": item["accuracy"],
            "attempts": item["attempts"],
            "goal": "Lưu thêm 3 mẫu rõ nét và cố đạt AI đúng 2 lần liên tiếp.",
            "tip": item["tip"],
        }
        for item in weaknesses[:3]
    ]

    training = _training_readiness(user["id"])
    payload = {
        "authenticated": True,
        "user": user,
        "stats": {
            "games": int(row["games"]),
            "best_score": int(row["best_score"]),
            "avg_score": round(float(row["avg_score"]), 1),
            "best_streak": int(row["best_streak"]),
            "avg_accuracy": round(float(row["avg_accuracy"]), 1),
            "drawings": int(sample_row["drawings"]),
            "accuracy": round(float(sample_row["acc"]) * 100, 1),
            "avg_confidence": round(float(sample_row["avg_confidence"]) * 100, 1),
            "total_points": int(sample_row["total_points"]),
            "total_minutes": round(float(row["total_seconds"] or 0) / 60, 1),
            "last_played": row["last_played"],
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "practice_plan": practice_plan,
        "recent_sessions": [dict(r) for r in recent_sessions],
        "training": training,
    }
    payload["storage"] = _save_skill_profile_snapshot(payload, user)
    return payload


@app.get("/game/leaderboard")
def game_leaderboard():
    with get_db() as conn:
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT COALESCE(users.username, 'guest') AS username,
                       game_sessions.score, game_sessions.level, game_sessions.streak,
                       game_sessions.accuracy, game_sessions.duration_seconds,
                       game_sessions.mode, game_sessions.created_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_sessions.user_id
                           ORDER BY game_sessions.score DESC, game_sessions.streak DESC, game_sessions.level DESC, game_sessions.id DESC
                       ) AS rn
                FROM game_sessions
                LEFT JOIN users ON users.id = game_sessions.user_id
            )
            SELECT username, score, level, streak, accuracy, duration_seconds, mode, created_at
            FROM ranked
            WHERE rn = 1
            ORDER BY score DESC, streak DESC, level DESC
            LIMIT 10
            """
        ).fetchall()
    leaderboard_rows = [dict(r) for r in rows]
    storage = _save_leaderboard_snapshot(leaderboard_rows)
    return {"leaderboard": leaderboard_rows, "storage": storage}


@app.post("/game/session")
async def save_game_session(
    request: Request,
    score: int = Form(...),
    level: int = Form(...),
    streak: int = Form(...),
    accuracy: float = Form(0),
    duration_seconds: int = Form(0),
    mode: str = Form("mouse"),
):
    user = user_from_request(request)
    created_at = _now_iso()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO game_sessions(user_id, score, level, streak, accuracy, duration_seconds, mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user["id"] if user else None, int(score), int(level), int(streak), float(accuracy), int(duration_seconds), mode, created_at),
        )
    payload = {
        "ok": True,
        "session_id": int(cur.lastrowid) if cur else None,
        "score": int(score),
        "level": int(level),
        "streak": int(streak),
        "accuracy": float(accuracy),
        "duration_seconds": int(duration_seconds),
        "mode": mode,
        "created_at": created_at,
    }
    payload["storage"] = _save_game_session_storage(payload, user)
    return payload


@app.post("/game/stroke")
async def save_stroke_sample(
    request: Request,
    target: str = Form(...),
    predicted: str = Form(""),
    confidence: float = Form(0),
    correct: int = Form(0),
    mode: str = Form("mouse"),
    strokes_json: str = Form("[]"),
    judge_json: str = Form("{}"),
    manual: int = Form(0),
    point_count: int = Form(0),
):
    """Lưu một mẫu vẽ để cập nhật Skill Profile và dùng cho self-improving loop."""
    user = user_from_request(request)
    parsed = _safe_json_loads(strokes_json, [])
    compact_json = json.dumps(parsed if isinstance(parsed, list) else [], ensure_ascii=False, separators=(",", ":"))
    parsed_judge = _safe_json_loads(judge_json, {})
    compact_judge = json.dumps(parsed_judge if isinstance(parsed_judge, dict) else {}, ensure_ascii=False, separators=(",", ":"))
    points = int(point_count) if int(point_count or 0) > 0 else _count_stroke_points(parsed)
    target_label = target.strip().lower()
    predicted_label = predicted.strip().lower()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO stroke_samples(
                user_id, target, predicted, confidence, correct, mode, strokes_json,
                judge_json, manual, point_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"] if user else None,
                target_label,
                predicted_label,
                float(confidence),
                1 if int(correct or 0) else 0,
                mode.strip().lower()[:32],
                compact_json,
                compact_judge,
                1 if int(manual or 0) else 0,
                max(0, points),
                _now_iso(),
            ),
        )
    sample_result = {
        "ok": True,
        "sample_id": int(cur.lastrowid) if cur else None,
        "target": target_label,
        "predicted": predicted_label,
        "confidence": float(confidence),
        "correct": bool(int(correct or 0)),
        "mode": mode.strip().lower()[:32],
        "manual": bool(int(manual or 0)),
        "point_count": max(0, points),
        "strokes": parsed if isinstance(parsed, list) else [],
        "judge": parsed_judge if isinstance(parsed_judge, dict) else {},
        "created_at": _now_iso(),
    }
    _self_loop_save_sample(sample_result, user)
    try:
        skill_sample_dir = SKILL_PROFILE_STORAGE_DIR / "stroke_samples" / _safe_storage_name(target_label, "unknown")
        skill_sample_path = skill_sample_dir / f"sample_{sample_result['sample_id'] or _panel_slug('sample')}.json"
        _write_json_file(skill_sample_path, {"saved_at": _now_iso(), "username": user.get("username") if user else None, "sample": sample_result})
        _append_jsonl(SKILL_PROFILE_STORAGE_DIR / "stroke_samples.jsonl", {
            "saved_at": _now_iso(),
            "username": user.get("username") if user else None,
            "sample_id": sample_result.get("sample_id"),
            "target": target_label,
            "predicted": predicted_label,
            "confidence": sample_result.get("confidence"),
            "correct": sample_result.get("correct"),
            "path": str(skill_sample_path),
        })
        _panel_log_action("skill_profile", "save_stroke_sample", {"sample_id": sample_result.get("sample_id"), "path": str(skill_sample_path)}, user)
        sample_result["panel_storage"] = {"skill_profile_sample": str(skill_sample_path)}
    except Exception:
        pass
    return sample_result


@app.post("/predict_stroke")
async def predict_stroke(request: Request, strokes_json: str = Form(...), target: str = Form("")):
    """Stroke-based model riêng. Nếu chưa train stroke model, trả trạng thái fallback rõ ràng."""
    seq_model, seq_categories = _load_stroke_sequence_model()
    if seq_model is None or not seq_categories:
        return {
            "ok": False,
            "available": False,
            "message": "Chưa có stroke_sequence_model.keras. Chạy train_stroke_model.py hoặc notebook Colab để tạo model này.",
        }
    x = _flatten_strokes(strokes_json)
    preds = seq_model.predict(x, verbose=0)[0]
    top_idx = preds.argsort()[-5:][::-1]
    top5 = [
        {
            "label": seq_categories[int(i)],
            "meaning_vi": VI_MEANINGS.get(seq_categories[int(i)], seq_categories[int(i)]),
            "confidence": float(preds[int(i)]),
            "confidence_percent": round(float(preds[int(i)]) * 100, 2),
        }
        for i in top_idx
    ]
    best = top5[0]
    target_label = target.strip().lower()
    return {
        "ok": True,
        "available": True,
        "label": best["label"],
        "confidence": best["confidence"],
        "top5": top5,
        "target": target_label,
        "is_correct": bool(target_label and best["label"] == target_label),
    }


@app.get("/dataset/export")
def dataset_export(request: Request):
    """Export dữ liệu stroke sang JSONL/CSV/manifest để train local hoặc Colab."""
    user = user_from_request(request)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stroke_samples.id, users.username, target, predicted, confidence, correct, mode,
                   strokes_json, judge_json, manual, point_count, stroke_samples.created_at
            FROM stroke_samples
            LEFT JOIN users ON users.id = stroke_samples.user_id
            ORDER BY stroke_samples.id ASC
            """
        ).fetchall()
    EXPORTED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = EXPORTED_DATASET_DIR / "stroke_samples.jsonl"
    csv_path = EXPORTED_DATASET_DIR / "stroke_samples.csv"
    manifest_path = EXPORTED_DATASET_DIR / "training_manifest.json"

    label_summary: Dict[str, dict] = {}
    exported_rows = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            payload = dict(r)
            payload["strokes"] = _safe_json_loads(payload.pop("strokes_json", "[]"), [])
            payload["judge"] = _safe_json_loads(payload.pop("judge_json", "{}"), {})
            payload["point_count"] = int(payload.get("point_count") or _count_stroke_points(payload["strokes"]))
            exported_rows.append(payload)
            item = label_summary.setdefault(payload["target"], {"target": payload["target"], "samples": 0, "correct": 0})
            item["samples"] += 1
            item["correct"] += int(payload.get("correct") or 0)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "username", "target", "predicted", "confidence", "correct", "mode", "manual", "point_count", "created_at"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in exported_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    for item in label_summary.values():
        item["accuracy"] = round((item["correct"] / item["samples"]) * 100, 1) if item["samples"] else 0
    readiness = _training_readiness(None)
    manifest = {
        "exported_at": _now_iso(),
        "requested_by": user["username"] if user else "guest",
        "samples": len(exported_rows),
        "classes": len(label_summary),
        "files": {
            "jsonl": "stroke_samples.jsonl",
            "csv": "stroke_samples.csv",
        },
        "label_summary": sorted(label_summary.values(), key=lambda x: (-x["samples"], x["target"])),
        "training_readiness": readiness,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot = _self_loop_snapshot_export(manifest)
    _self_loop_log_action("export_dataset", {
        "samples": len(exported_rows),
        "classes": len(label_summary),
        "latest_dir": str(EXPORTED_DATASET_DIR),
        "snapshot_dir": snapshot.get("snapshot_dir"),
    }, user)

    return {
        "ok": True,
        "requested_by": user["username"] if user else "guest",
        "samples": len(exported_rows),
        "classes": len(label_summary),
        "path": str(EXPORTED_DATASET_DIR),
        "storage": _self_loop_storage_info(),
        "snapshot": snapshot,
        "downloads": {
            "jsonl": "/dataset/download/stroke_samples.jsonl",
            "csv": "/dataset/download/stroke_samples.csv",
            "manifest": "/dataset/download/training_manifest.json",
        },
        "label_summary": manifest["label_summary"],
        "training_readiness": readiness,
    }


@app.get("/dataset/download/{filename}")
def dataset_download(filename: str):
    allowed = {"stroke_samples.jsonl", "stroke_samples.csv", "training_manifest.json"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="File không hợp lệ.")
    path = EXPORTED_DATASET_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa export dataset.")
    media_type = "application/jsonl" if filename.endswith(".jsonl") else "text/csv" if filename.endswith(".csv") else "application/json"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/admin/retrain/start")
def retrain_start(request: Request, mode: str = Form("stroke"), epochs: int = Form(8)):
    """Khởi động retrain tự động local; status/log đọc tại /admin/retrain/status."""
    global retrain_process, retrain_job_id
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để chạy retrain.")
    mode = mode.strip().lower()
    if mode not in {"stroke", "image"}:
        raise HTTPException(status_code=400, detail="mode phải là stroke hoặc image.")

    readiness = _training_readiness(None)
    if mode == "stroke" and not readiness["ready_stroke"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Chưa đủ dữ liệu để train stroke model.",
                "readiness": readiness,
            },
        )
    if mode == "image" and not readiness["ready_image"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Chưa đủ dữ liệu để train image model. Hãy export/lưu thêm mẫu hoặc thêm QuickDraw .npy.",
                "readiness": readiness,
            },
        )

    with retrain_lock:
        if retrain_process and retrain_process.poll() is None:
            return {"ok": False, "message": "Đang có job retrain chạy.", "status": _read_retrain_status()}
        script = "train_stroke_model.py" if mode == "stroke" else "train_image_model.py"
        safe_epochs = max(1, min(int(epochs), 50))
        log_path, job_meta = _self_loop_create_training_job(mode, user, safe_epochs, readiness, script)
        _write_retrain_status("running", f"Đang chạy {script}", {
            "mode": mode,
            "log": str(log_path),
            "job_dir": str(log_path.parent),
            "started_by": user["username"],
            "readiness": readiness,
            "storage": _self_loop_storage_info(),
        })
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "src" / "training" / script), "--epochs", str(safe_epochs)]
        if mode == "image":
            cmd.extend(["--config", str(ROOT / "configs" / "image_resnet_sketch.yaml")])
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["AIRDRAW_RETRAIN_STATUS_PATH"] = str(RETRAIN_STATUS_PATH)
        env["AIRDRAW_SELF_LOOP_JOB_DIR"] = str(log_path.parent)
        log_file = open(log_path, "w", encoding="utf-8")
        retrain_process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT, env=env)
        with get_db() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_jobs(user_id, mode, status, message, pid, epochs, samples, classes, log_path, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user["id"], mode, "running", f"Đang chạy {script}", retrain_process.pid, safe_epochs, readiness["total_samples"], readiness["classes"], str(log_path), _now_iso()),
            )
            retrain_job_id = int(cur.lastrowid)
    return {"ok": True, "message": "Đã bắt đầu retrain.", "pid": retrain_process.pid, "job_id": retrain_job_id, "status": _read_retrain_status()}


@app.get("/admin/retrain/status")
def retrain_status():
    global retrain_job_id
    status = _read_retrain_status()
    process_running = False
    returncode = None
    if retrain_process:
        returncode = retrain_process.poll()
        process_running = returncode is None
        status["process_running"] = process_running
        status["returncode"] = returncode
        if not process_running and retrain_job_id:
            final_status = str(status.get("status") or ("done" if returncode == 0 else "failed"))
            message = str(status.get("message") or "")
            if final_status == "running":
                final_status = "done" if returncode == 0 else "failed"
                message = "Job kết thúc." if returncode == 0 else "Job dừng với lỗi. Xem log_tail."
                _write_retrain_status(final_status, message, {"mode": status.get("mode"), "returncode": returncode})
                status = _read_retrain_status()
                status["process_running"] = False
                status["returncode"] = returncode
            _update_training_job(retrain_job_id, final_status, message)
            _self_loop_finalize_training_job(status, final_status, returncode)
    else:
        status["process_running"] = False

    log_path = status.get("log") or (SELF_LOOP_JOBS_DIR / f"retrain_{status.get('mode', 'stroke')}.log")
    status["log_tail"] = _tail_file(log_path)
    status["training_readiness"] = _training_readiness(None)
    status["recent_jobs"] = _recent_training_jobs()
    status["model_runtime"] = _runtime_model_info()
    status["self_improved_model_exists"] = SELF_IMPROVED_MODEL_PATH.exists()
    status["storage"] = _self_loop_storage_info()
    return status


@app.get("/admin/self-improve/storage")
def admin_self_improve_storage(request: Request):
    """Xem nhanh kho lưu tự động của Self-improving Loop."""
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để xem kho Self-improving Loop.")
    info = _self_loop_storage_info()

    def recent_files(folder: Path, limit: int = 20) -> List[dict]:
        try:
            files = [p for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return [
                {
                    "name": p.name,
                    "path": str(p),
                    "relative": str(p.relative_to(SELF_IMPROVING_LOOP_DIR)),
                    "size_bytes": int(p.stat().st_size),
                    "modified_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                }
                for p in files[:limit]
            ]
        except Exception:
            return []

    return {"ok": True, "storage": info, "recent_files": recent_files(SELF_IMPROVING_LOOP_DIR)}




@app.get("/admin/panel-storage")
def admin_panel_storage(request: Request, section: str = "", limit: int = 20):
    """Xem nhanh kho lưu tự động của các panel: nhận diện, hình thật, judge, profile, leaderboard, PvP."""
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để xem kho lưu các panel.")
    return {
        "ok": True,
        "section": section or "all",
        "storage": _panel_storage_info(),
        "recent_files": _panel_recent_files(section, limit),
    }

@app.post("/admin/model/reload")
def admin_model_reload(request: Request, kind: str = Form("self_improved")):
    """Nạp model ảnh mới vào runtime sau khi Train image hoàn tất."""
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để reload model.")
    kind = kind.strip().lower()
    if kind in {"self_improved", "image", "latest"}:
        info = _reload_image_runtime(SELF_IMPROVED_MODEL_PATH, SELF_IMPROVED_CATEGORIES_PATH)
    elif kind in {"base", "default"}:
        info = _reload_image_runtime(MODEL_PATH, CATEGORIES_PATH)
    else:
        raise HTTPException(status_code=400, detail="kind phải là self_improved hoặc base.")
    _self_loop_log_action("reload_model", {"kind": kind, "model": info}, user)
    return {"ok": True, "reloaded_by": user["username"], "model": info, "storage": _self_loop_storage_info()}


@app.websocket("/ws/pvp/{room}")
async def websocket_pvp(websocket: WebSocket, room: str):
    username = websocket.query_params.get("username", "guest")[:32] or "guest"
    room_name = pvp_manager.normalize_room(room)
    await pvp_manager.connect(room_name, websocket, username)
    _save_pvp_storage(room_name, {"type": "join", "message": f"{username} joined room {room_name}."}, "join", username)
    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type", "event")
            meta = pvp_manager.update_player(websocket, payload)
            payload.update({"username": username, "players": pvp_manager.players(room_name)})
            _save_pvp_storage(room_name, payload, msg_type, username)
            if msg_type in {"score", "final"}:
                try:
                    with get_db() as conn:
                        conn.execute(
                            """
                            INSERT INTO pvp_matches(room, username, score, target, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (room_name, username, int(meta.get("score") or 0), str(meta.get("target") or ""), _now_iso()),
                        )
                except Exception:
                    pass
            await pvp_manager.broadcast(room_name, payload)
    except WebSocketDisconnect:
        meta = pvp_manager.disconnect(websocket)
        if meta:
            leave_payload = {"type": "system", "message": f"{meta['username']} left.", "players": pvp_manager.players(meta["room"])}
            _save_pvp_storage(meta["room"], leave_payload, "leave", meta["username"])
            await pvp_manager.broadcast(meta["room"], leave_payload)


@app.post("/camera/face-strokes")
async def camera_face_strokes(
    file: UploadFile = File(...),
    canvas_width: int = Form(CANVAS_W),
    canvas_height: int = Form(CANVAS_H),
    mirror: int = Form(1),
    preview: int = Form(0),
):
    """Nhận diện khuôn mặt từ frame webcam và chuyển thành strokes cho camera mode.

    Luồng này lấy ý tưởng từ DeepShieldAI-Pro: detect largest face bằng OpenCV Haar,
    crop vùng mặt có padding, sau đó biến vùng mặt thành nét sketch nhẹ. Frame webcam
    chỉ xử lý trong RAM và không lưu ra đĩa.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Bạn chưa gửi frame camera.")
    try:
        return analyze_face_frame_bytes(
            image_bytes,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            mirror=bool(int(mirror)),
            include_preview=bool(int(preview)),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không xử lý được khuôn mặt từ camera: {exc}") from exc


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
    if label not in all_vocab_category_set:
        raise HTTPException(status_code=400, detail=f"Nhãn không hợp lệ: {label}")
    return {
        "ok": True,
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "provider": "offline-pil-reference",
        "image": create_offline_reference_image(label),
    }


@app.post("/image/generate")
async def image_generate(
    request: Request,
    label: str = Form(...),
    reason: str = Form("manual-generate"),
    target: str = Form(""),
    predicted: str = Form(""),
):
    """Tạo ảnh tham khảo chân thực theo nhãn đã nhận diện, có fallback offline.

    Endpoint này chỉ được frontend gọi khi người dùng bấm nút **Sinh hình thật**.
    Mỗi ảnh sinh ra được lưu vào `data/panel_storage/real_image_after_draw/`.
    """
    user = user_from_request(request)
    label = label.strip().lower()
    target = target.strip().lower()
    predicted = predicted.strip().lower()
    reason = reason.strip().lower()[:80] or "manual-generate"
    if label not in all_vocab_category_set:
        raise HTTPException(status_code=400, detail=f"Nhãn không hợp lệ: {label}")

    prompt = build_realistic_prompt(label)
    ai_image, error = generate_openai_reference_image(label)
    if ai_image:
        payload = {
            "ok": True,
            "label": label,
            "meaning_vi": VI_MEANINGS.get(label, label),
            "provider": f"openai:{OPENAI_IMAGE_MODEL}",
            "prompt": prompt,
            "image": ai_image,
        }
        payload["storage"] = _save_real_image_storage(
            label, ai_image, payload["provider"], prompt, reason, target, predicted, user
        )
        return payload

    fallback_image = create_offline_reference_image(label)
    payload = {
        "ok": True,
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "provider": "offline-pil-reference",
        "prompt": prompt,
        "image": fallback_image,
        "note": (
            "Chưa có OPENAI_API_KEY hoặc API tạo ảnh chưa gọi được, "
            "nên hệ thống dùng ảnh tham khảo offline. Cấu hình OPENAI_API_KEY để tạo ảnh photorealistic."
        ),
        "error": error,
    }
    payload["storage"] = _save_real_image_storage(
        label, fallback_image, payload["provider"], prompt, reason, target, predicted, user, payload["note"], error
    )
    return payload


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


# -----------------------------
# Production AI Ops endpoints
# -----------------------------
def _read_json_file_safe(path: Path, fallback: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return fallback


def _run_project_script(args: List[str], timeout: int = 900) -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-X", "utf8", *args]
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": " ".join(cmd),
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "stdout_tail": (proc.stdout or "")[-8000:],
        "stderr_tail": (proc.stderr or "")[-8000:],
    }


def _latest_release_summary() -> dict:
    releases = ROOT / "assets" / "reports" / "releases"
    if not releases.exists():
        return {}
    candidates = sorted(releases.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return _read_json_file_safe(candidates[0], {}) if candidates else {}


def _promotion_tail(limit: int = 6) -> List[dict]:
    path = MODELS_DIR / "promotion_log.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


@app.get("/admin/pro/status")
def admin_pro_status(request: Request):
    """Tổng quan production AI: benchmark, runtime, registry, promotion, readiness."""
    user = user_from_request(request)
    readiness = _training_readiness(user["id"] if user else None)
    benchmark_manifest = _read_json_file_safe(ROOT / "data" / "benchmark" / "release_v1" / "manifest.json", {})
    registry = _read_json_file_safe(MODELS_DIR / "registry.json", {})
    calibration = _read_json_file_safe(MODELS_DIR / "calibration" / "image_temperature.json", {})
    return {
        "ok": True,
        "user": user["username"] if user else "guest",
        "runtime": _runtime_model_info(),
        "training_readiness": readiness,
        "benchmark": benchmark_manifest,
        "latest_eval": _latest_release_summary(),
        "registry": {"updated_at": registry.get("updated_at"), "count": registry.get("count", 0)},
        "calibration": calibration,
        "promotion_tail": _promotion_tail(),
        "paths": {
            "benchmark": "data/benchmark/release_v1",
            "eval_report": "assets/reports/releases/current/summary.json",
            "promotion_log": "models/promotion_log.jsonl",
        },
    }


@app.post("/admin/benchmark/build")
def admin_benchmark_build(request: Request):
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để build benchmark.")
    result = _run_project_script([
        "src/data/make_real_user_benchmark.py",
        "--db", str(APP_DB_PATH),
        "--out", str(ROOT / "data" / "benchmark" / "release_v1"),
    ], timeout=300)
    result["manifest"] = _read_json_file_safe(ROOT / "data" / "benchmark" / "release_v1" / "manifest.json", {})
    return result


@app.post("/admin/evaluate/run")
def admin_evaluate_run(request: Request):
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để evaluate model.")
    benchmark_dir = ROOT / "data" / "benchmark" / "release_v1"
    if not (benchmark_dir / "test.jsonl").exists():
        raise HTTPException(status_code=400, detail="Chưa có benchmark. Bấm Build benchmark trước.")
    result = _run_project_script([
        "src/evaluation/evaluate_release.py",
        "--benchmark", str(benchmark_dir),
        "--out", str(ROOT / "assets" / "reports" / "releases" / "current"),
    ], timeout=900)
    result["summary"] = _read_json_file_safe(ROOT / "assets" / "reports" / "releases" / "current" / "summary.json", {})
    return result


@app.get("/admin/promote/status")
def admin_promote_status(request: Request):
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để xem promotion status.")
    return {
        "ok": True,
        "promotion_tail": _promotion_tail(),
        "latest_eval": _latest_release_summary(),
        "candidate_exists": (MODELS_DIR / "image_cnn_candidate.keras").exists(),
        "candidate_categories_exists": (MODELS_DIR / "categories_candidate.json").exists(),
    }


@app.post("/admin/promote/dry-run")
def admin_promote_dry_run(request: Request):
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để kiểm tra promotion.")
    result = _run_project_script([
        "src/training/promote_candidate.py",
        "--report", str(ROOT / "assets" / "reports" / "releases" / "current" / "summary.json"),
        "--dry-run",
        "--allow-if-weak-data",
    ], timeout=120)
    return result


@app.get("/metrics")
def metrics_endpoint():
    try:
        from src.monitoring.metrics import render_metrics
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)
    except Exception as exc:
        return Response(content=f"airdraw_metrics_error 1\n# {exc}\n", media_type="text/plain")
