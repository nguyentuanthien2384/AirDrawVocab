import json
import os
import base64
import hashlib
import secrets
import sqlite3
import subprocess
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
RETRAIN_STATUS_PATH = ROOT / "data" / "retrain_status.json"
EXPORTED_DATASET_DIR = ROOT / "data" / "self_improve_export"
CANVAS_W = 960
CANVAS_H = 540

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


def init_app_db() -> None:
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
    points = []
    if isinstance(strokes, str):
        try:
            strokes = json.loads(strokes)
        except Exception:
            strokes = []
    if not isinstance(strokes, list):
        strokes = []
    for stroke in strokes:
        if not isinstance(stroke, list):
            continue
        for p in stroke:
            if isinstance(p, dict):
                points.append([float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("t", 0))])
    if not points:
        return np.zeros((1, max_len, 5), dtype="float32")
    arr = np.array(points, dtype="float32")
    x = arr[:, 0] / max(CANVAS_W, 1)
    y = arr[:, 1] / max(CANVAS_H, 1)
    t = arr[:, 2]
    if t.max() > t.min():
        t = (t - t.min()) / (t.max() - t.min())
    else:
        t = np.zeros_like(t)
    dx = np.concatenate([[0], np.diff(x)])
    dy = np.concatenate([[0], np.diff(y)])
    seq = np.stack([x, y, dx, dy, t], axis=1)
    if len(seq) >= max_len:
        idx = np.linspace(0, len(seq) - 1, max_len).astype(int)
        seq = seq[idx]
    else:
        pad = np.zeros((max_len - len(seq), 5), dtype="float32")
        seq = np.vstack([seq, pad])
    return seq.reshape(1, max_len, 5).astype("float32")


class PvPRoomManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}
        self.meta: Dict[WebSocket, dict] = {}

    async def connect(self, room: str, websocket: WebSocket, username: str):
        await websocket.accept()
        room = room.strip().lower() or "default"
        self.rooms.setdefault(room, []).append(websocket)
        self.meta[websocket] = {"room": room, "username": username, "score": 0}
        await self.broadcast(room, {"type": "system", "message": f"{username} joined room {room}.", "players": self.players(room)})

    def disconnect(self, websocket: WebSocket):
        meta = self.meta.pop(websocket, None)
        if not meta:
            return None
        room = meta["room"]
        if room in self.rooms and websocket in self.rooms[room]:
            self.rooms[room].remove(websocket)
        return meta

    def players(self, room: str) -> List[dict]:
        players = []
        for ws in self.rooms.get(room, []):
            item = dict(self.meta.get(ws, {}))
            item.pop("room", None)
            players.append(item)
        return players

    async def broadcast(self, room: str, payload: dict):
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
        "model_path": str(MODEL_PATH),
        "live_drawing_model_path": str(LIVE_DRAWING_MODEL_PATH),
        "app_db_path": str(APP_DB_PATH),
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

    # Cắt sát vùng có nét vẽ.
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        # Không có nét vẽ -> trả ảnh đen để model không đoán bừa.
        return np.zeros((1, 28, 28, 1), dtype="float32")

    x, y, w, h = cv2.boundingRect(coords)
    gray = gray[y:y + h, x:x + w]

    # Scale nét vẽ vừa khung 20x20 (giữ tỉ lệ), giống chuẩn MNIST/QuickDraw bitmap.
    target = 20
    scale = target / max(w, h, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Đặt vào canvas 28x28 rồi căn theo trọng tâm (center of mass) ra chính giữa.
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_start = (28 - new_h) // 2
    x_start = (28 - new_w) // 2
    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized

    moments = cv2.moments(canvas, binaryImage=False)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))
        translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, translation, (28, 28), borderValue=0)

    normalized = canvas.astype("float32") / 255.0
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
    active_model = live_drawing_model if normalized_source in {"camera", "hand", "airdraw", "live"} else model
    active_model_path = LIVE_DRAWING_MODEL_PATH if active_model is live_drawing_model else MODEL_PATH
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
    log_prediction(user_from_request(request), label, confidence, normalized_source)

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


def _judge_payload(target: str, predicted: str, confidence: float, correct: bool, stroke_count: int, elapsed_ms: int) -> dict:
    clarity = min(100, max(5, round(confidence * 100)))
    stroke_score = min(100, 35 + stroke_count * 7)
    speed_score = 100 if elapsed_ms <= 12000 else max(30, 100 - int((elapsed_ms - 12000) / 500))
    shape_score = round((clarity * 0.6) + (stroke_score * 0.25) + (speed_score * 0.15))
    if correct:
        grade = "S" if shape_score >= 85 else "A" if shape_score >= 70 else "B"
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
        "feedback": _teacher_feedback(target, predicted, confidence, correct, stroke_count),
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
    active_model = live_drawing_model if normalized_source in {"camera", "hand", "airdraw", "live", "realtime"} else model
    preds = predict_proba(active_model, x)
    usable_count = min(len(preds), len(categories))
    if usable_count <= 0:
        raise HTTPException(status_code=500, detail="Model chưa có nhãn nhận diện hợp lệ.")
    usable_preds = preds[:usable_count]
    best_index = int(np.argmax(usable_preds))
    label = categories[best_index]
    confidence = float(usable_preds[best_index])
    target_label = target.strip().lower()
    correct = bool(target_label and label == target_label)
    top5 = _topn_from_predictions(preds, 5)
    judge = _judge_payload(target_label, label, confidence, correct, max(0, int(stroke_count)), max(0, int(elapsed_ms)))
    return {
        "ok": True,
        "label": label,
        "meaning_vi": VI_MEANINGS.get(label, label),
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "top5": top5,
        "target": target_label,
        "is_correct": correct,
        "judge": judge,
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
        }
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS games, COALESCE(MAX(score), 0) AS best_score,
                   COALESCE(AVG(accuracy), 0) AS avg_accuracy
            FROM game_sessions WHERE user_id = ?
            """,
            (user["id"],),
        ).fetchone()
        sample_row = conn.execute(
            """
            SELECT COUNT(*) AS drawings, COALESCE(AVG(correct), 0) AS acc
            FROM stroke_samples WHERE user_id = ?
            """,
            (user["id"],),
        ).fetchone()
        label_rows = conn.execute(
            """
            SELECT target, COUNT(*) AS attempts, ROUND(AVG(correct) * 100, 1) AS accuracy
            FROM stroke_samples
            WHERE user_id = ?
            GROUP BY target
            HAVING attempts >= 1
            ORDER BY accuracy DESC, attempts DESC
            """,
            (user["id"],),
        ).fetchall()
    strengths = [dict(r) for r in label_rows[:5]]
    weaknesses = [dict(r) for r in sorted(label_rows, key=lambda r: (r["accuracy"], -r["attempts"]))[:5]]
    return {
        "authenticated": True,
        "user": user,
        "stats": {
            "games": int(row["games"]),
            "best_score": int(row["best_score"]),
            "avg_accuracy": round(float(row["avg_accuracy"]), 1),
            "drawings": int(sample_row["drawings"]),
            "accuracy": round(float(sample_row["acc"]) * 100, 1),
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


@app.get("/game/leaderboard")
def game_leaderboard():
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(users.username, 'guest') AS username, MAX(game_sessions.score) AS score,
                   MAX(game_sessions.level) AS level,
                   MAX(game_sessions.streak) AS streak
            FROM game_sessions
            LEFT JOIN users ON users.id = game_sessions.user_id
            GROUP BY game_sessions.user_id
            ORDER BY score DESC, streak DESC, level DESC
            LIMIT 10
            """
        ).fetchall()
    return {"leaderboard": [dict(r) for r in rows]}


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
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO game_sessions(user_id, score, level, streak, accuracy, duration_seconds, mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user["id"] if user else None, int(score), int(level), int(streak), float(accuracy), int(duration_seconds), mode, _now_iso()),
        )
    return {"ok": True, "session_id": int(cur.lastrowid) if cur else None}


@app.post("/game/stroke")
async def save_stroke_sample(
    request: Request,
    target: str = Form(...),
    predicted: str = Form(""),
    confidence: float = Form(0),
    correct: int = Form(0),
    mode: str = Form("mouse"),
    strokes_json: str = Form("[]"),
):
    user = user_from_request(request)
    try:
        parsed = json.loads(strokes_json)
        compact_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        compact_json = "[]"
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO stroke_samples(user_id, target, predicted, confidence, correct, mode, strokes_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user["id"] if user else None, target.strip().lower(), predicted.strip().lower(), float(confidence), int(correct), mode, compact_json, _now_iso()),
        )
    return {"ok": True, "sample_id": int(cur.lastrowid) if cur else None}



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
    """Export dữ liệu stroke đã thu thập sang JSONL để dùng train/retrain trên Colab."""
    user = user_from_request(request)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT stroke_samples.id, users.username, target, predicted, confidence, correct, mode, strokes_json, stroke_samples.created_at
            FROM stroke_samples
            LEFT JOIN users ON users.id = stroke_samples.user_id
            ORDER BY stroke_samples.id ASC
            """
        ).fetchall()
    EXPORTED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORTED_DATASET_DIR / "stroke_samples.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            payload = dict(r)
            try:
                payload["strokes"] = json.loads(payload.pop("strokes_json") or "[]")
            except Exception:
                payload["strokes"] = []
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {
        "ok": True,
        "requested_by": user["username"] if user else "guest",
        "samples": len(rows),
        "path": str(out_path),
        "download": "/dataset/download/stroke_samples.jsonl",
    }


@app.get("/dataset/download/{filename}")
def dataset_download(filename: str):
    allowed = {"stroke_samples.jsonl"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="File không hợp lệ.")
    path = EXPORTED_DATASET_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa export dataset.")
    return FileResponse(path, media_type="application/jsonl", filename=filename)


@app.post("/admin/retrain/start")
def retrain_start(request: Request, mode: str = Form("stroke"), epochs: int = Form(8)):
    """Khởi động retrain tự động local. Colab vẫn là hướng khuyến nghị nếu train nặng."""
    global retrain_process
    user = user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập để chạy retrain.")
    mode = mode.strip().lower()
    if mode not in {"stroke", "image"}:
        raise HTTPException(status_code=400, detail="mode phải là stroke hoặc image.")
    with retrain_lock:
        if retrain_process and retrain_process.poll() is None:
            return {"ok": False, "message": "Đang có job retrain chạy.", "status": _read_retrain_status()}
        script = "train_stroke_model.py" if mode == "stroke" else "self_improve_retrain.py"
        log_path = ROOT / "data" / f"retrain_{mode}.log"
        _write_retrain_status("running", f"Đang chạy {script}", {"mode": mode, "log": str(log_path), "started_by": user["username"]})
        cmd = [sys.executable, str(ROOT / script), "--epochs", str(max(1, min(int(epochs), 50)))]
        log_file = open(log_path, "w", encoding="utf-8")
        retrain_process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)
    return {"ok": True, "message": "Đã bắt đầu retrain.", "pid": retrain_process.pid, "status": _read_retrain_status()}


@app.get("/admin/retrain/status")
def retrain_status():
    status = _read_retrain_status()
    if retrain_process:
        status["process_running"] = retrain_process.poll() is None
        status["returncode"] = retrain_process.poll()
    else:
        status["process_running"] = False
    return status


@app.websocket("/ws/pvp/{room}")
async def websocket_pvp(websocket: WebSocket, room: str):
    username = websocket.query_params.get("username", "guest")[:32] or "guest"
    await pvp_manager.connect(room, websocket, username)
    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type", "event")
            meta = pvp_manager.meta.get(websocket, {})
            if msg_type == "score":
                try:
                    meta["score"] = int(payload.get("score", 0))
                except Exception:
                    pass
            payload.update({"username": username, "players": pvp_manager.players(room)})
            await pvp_manager.broadcast(room, payload)
    except WebSocketDisconnect:
        meta = pvp_manager.disconnect(websocket)
        if meta:
            await pvp_manager.broadcast(meta["room"], {"type": "system", "message": f"{meta['username']} left.", "players": pvp_manager.players(meta["room"])})


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
async def image_generate(label: str = Form(...)):
    """Tạo ảnh tham khảo chân thực theo nhãn đã nhận diện, có fallback offline."""
    label = label.strip().lower()
    if label not in all_vocab_category_set:
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
