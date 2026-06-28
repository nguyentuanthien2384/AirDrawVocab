

import os
import sys

# Buộc stdout/stderr dùng UTF-8: trên Windows console mặc định (cp1252) các câu
# print tiếng Việt sẽ gây UnicodeEncodeError và làm văng app. Phải đặt sớm nhất.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
from pathlib import Path
import time
import random
import threading
from collections import deque
import numpy as np
import cv2
import pygame

# Các thư viện nặng: cho phép fallback để game vẫn mở được bằng chuột nếu thiếu webcam/MediaPipe/TTS.
HAS_MEDIAPIPE = False
HAS_MP_SOLUTIONS = False
HAS_MP_TASKS = False
mp_tasks_python = None
mp_vision = None
try:
    import mediapipe as mp
    # API cũ (legacy solutions) — có trên Python 3.9–3.12.
    HAS_MP_SOLUTIONS = hasattr(mp, "solutions")
    # API mới (Tasks) — có trên cả Python 3.13, dùng file hand_landmarker.task.
    try:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_vision
        HAS_MP_TASKS = hasattr(mp_vision, "HandLandmarker") and hasattr(mp, "Image")
    except Exception:
        HAS_MP_TASKS = False
    HAS_MEDIAPIPE = HAS_MP_SOLUTIONS or HAS_MP_TASKS
    if not HAS_MEDIAPIPE:
        raise AttributeError("mediapipe thiếu cả 'solutions' lẫn 'tasks'. Hãy cài lại mediapipe.")
    print(f"[SETUP] MediaPipe sẵn sàng (solutions={HAS_MP_SOLUTIONS}, tasks={HAS_MP_TASKS}).")
except Exception as e:
    mp = None
    HAS_MEDIAPIPE = False
    print(f"[SETUP] MediaPipe chưa sẵn sàng: {e}. Game vẫn có thể vẽ bằng chuột.")

try:
    import pyttsx3
    HAS_TTS = True
except Exception as e:
    pyttsx3 = None
    HAS_TTS = False
    print(f"[SETUP] pyttsx3 chưa sẵn sàng: {e}. Tính năng phát âm sẽ tắt.")

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except Exception as e:
    tf = None
    HAS_TENSORFLOW = False
    print(f"[SETUP] TensorFlow chưa sẵn sàng: {e}. AI nhận diện sẽ tắt.")

from vocab_data import VOCAB_DATA, CATEGORIES as ALL_CATEGORIES

# AI Assistant (100% Offline)
try:
    from ai_assistant import AIManager
    HAS_AI = True
except ImportError:
    HAS_AI = False
    print("[GAME] ai_assistant.py not found. AI features disabled.")


# ========================
# CẤU HÌNH
# ========================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 620
CAMERA_WIDTH = 760
CAMERA_HEIGHT = 465
CANVAS_SIZE = 28  # Kích thước canvas cho model (28x28)
# Tiền xử lý khớp định dạng QuickDraw (giống image_preprocess.py của backend):
#   nét vẽ được scale vừa hộp TARGET_BOX rồi căn theo trọng tâm trên canvas 28x28.
PREPROC_TARGET_BOX = 22          # nét vẽ chiếm ~22/28 px, chừa lề như dữ liệu QuickDraw
PREPROC_CENTER = (CANVAS_SIZE - 1) / 2.0  # 13.5 — tâm canvas để dịch trọng tâm về giữa
FPS = 30

# Thời gian vẽ mỗi từ (giây)
TIME_PER_WORD = 60
# Số mạng sống
MAX_LIVES = 3
# Số level trong game
TOTAL_LEVELS = 40
# Ngưỡng confidence để chấp nhận dự đoán
CONFIDENCE_THRESHOLD = 0.5

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 100, 255)
YELLOW = (255, 220, 50)
CYAN = (0, 220, 220)
HUD_BG = (22, 50, 86)
BTN_SKY = (120, 210, 245)
BTN_SHADOW = (12, 18, 28)
MAGENTA = (255, 75, 220)
DARK_BG = (30, 30, 45)
PANEL_BG = (45, 45, 65)
ORANGE = (255, 165, 0)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (80, 80, 80)
PURPLE = (150, 100, 255)

# ========================
# BẢNG MÀU HIỆN ĐẠI (theme tối, phẳng, chuyên nghiệp)
# ========================
COL_BG_TOP = (15, 23, 42)       # slate-900 (nền trên cho gradient)
COL_BG_BOTTOM = (2, 6, 23)      # gần đen (nền dưới cho gradient)
COL_CARD = (30, 41, 59)         # nền thẻ
COL_CARD_SOFT = (38, 50, 71)    # nền thẻ sáng hơn
COL_LINE = (51, 65, 85)         # viền mảnh
COL_ACCENT = (56, 189, 248)     # sky-400 (nhấn chính)
COL_ACCENT_2 = (45, 212, 191)   # teal-400 (nhấn phụ)
COL_SUCCESS = (74, 222, 128)    # green-400
COL_DANGER = (248, 113, 113)    # red-400
COL_WARNING = (250, 204, 21)    # yellow-400
COL_TEXT = (226, 232, 240)      # slate-200
COL_MUTED = (148, 163, 184)     # slate-400


def load_vn_font(size, bold=False):
    """Tải font HỖ TRỢ TIẾNG VIỆT một cách chắc chắn.

    pygame.font.SysFont("Arial") trên một số máy fallback sang font mặc định
    không có ký tự có dấu -> chữ bị vỡ (ô vuông). Ở đây ta nạp trực tiếp file
    .ttf của Windows (Segoe UI/Arial/Tahoma đều đủ glyph tiếng Việt) để đảm bảo
    hiển thị đúng "Học Tiếng Anh Qua Vẽ Hình Trực Tuyến".
    """
    win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        ("segoeuib.ttf" if bold else "segoeui.ttf"),
        "segoeui.ttf",
        ("arialbd.ttf" if bold else "arial.ttf"),
        "arial.ttf",
        "tahoma.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
    ]
    for name in candidates:
        path = win_fonts / name
        if path.exists():
            try:
                font = pygame.font.Font(str(path), size)
                if bold and not name.lower().endswith(("b.ttf", "bd.ttf")):
                    font.set_bold(True)
                return font
            except Exception:
                continue
    # Fallback cuối: match_font theo tên, rồi tới SysFont.
    matched = pygame.font.match_font("segoeui,arial,tahoma,dejavusans", bold=bold)
    if matched:
        try:
            return pygame.font.Font(matched, size)
        except Exception:
            pass
    return pygame.font.SysFont("arial", size, bold=bold)


def draw_vertical_gradient(surface, top_color, bottom_color, rect=None):
    """Vẽ nền gradient dọc cho cảm giác hiện đại."""
    if rect is None:
        rect = surface.get_rect()
    x, y, w, h = rect
    h = max(1, h)
    for i in range(h):
        ratio = i / h
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w, y + i))


def draw_panel(surface, rect, color=COL_CARD, radius=16, border=COL_LINE, border_w=1, shadow=True):
    """Vẽ thẻ bo góc, có viền mảnh và bóng đổ nhẹ."""
    rect = pygame.Rect(rect)
    if shadow:
        shadow_rect = rect.move(0, 4)
        shadow_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 90), shadow_surf.get_rect(), border_radius=radius)
        surface.blit(shadow_surf, shadow_rect.topleft)
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_w:
        pygame.draw.rect(surface, border, rect, border_w, border_radius=radius)


def draw_button(surface, rect, label, font, base_color, text_color=COL_TEXT, mouse_pos=None, radius=12):
    """Nút phẳng bo góc + hiệu ứng hover (sáng lên khi rê chuột). Trả về rect."""
    rect = pygame.Rect(rect)
    hovered = bool(mouse_pos and rect.collidepoint(mouse_pos))
    color = tuple(min(255, c + 28) for c in base_color) if hovered else base_color
    shadow_rect = rect.move(0, 4)
    shadow_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 110), shadow_surf.get_rect(), border_radius=radius)
    surface.blit(shadow_surf, shadow_rect.topleft)
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if hovered:
        pygame.draw.rect(surface, COL_TEXT, rect, 2, border_radius=radius)
    text = font.render(label, True, text_color)
    surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
    return rect

# Đường dẫn model: import từ config dùng chung
from config import MODEL_PATH

CATEGORIES_JSON_PATH = Path(__file__).resolve().parent / "models" / "categories.json"
HAND_LANDMARKER_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# Khung xương bàn tay (21 điểm) để tự vẽ landmark khi dùng Tasks API.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def load_model_categories():
    """Nhãn mà model hiện tại nhận diện được. vocab_pairs.py có thể là 40 từ,
    nhưng file model cũ có thể chỉ output 19 lớp; khi đó chỉ dùng 19 từ để chơi
    để tránh round không thể thắng. Khi train/gắn model 40 lớp, game tự dùng 40.
    """
    try:
        with open(CATEGORIES_JSON_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and labels:
            return list(labels)
    except Exception:
        pass
    return list(ALL_CATEGORIES)


MODEL_CATEGORIES = load_model_categories()
CATEGORIES = MODEL_CATEGORIES



# ========================
# LỚP TTS (Text-to-Speech)
# ========================
class TTSEngine:
    """Chuyển văn bản thành giọng nói bằng pyttsx3"""

    def __init__(self):
        if not HAS_TTS:
            self.available = False
            return
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            # Chọn giọng tiếng Anh nếu có
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            self.available = True
        except Exception as e:
            print(f"[TTS] Không thể khởi tạo pyttsx3: {e}")
            self.available = False

    def speak(self, text):
        """Phát âm từ trong thread riêng để không block game"""
        if self.available:
            try:
                thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
                thread.start()
            except Exception:
                pass

    def _speak_thread(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass


# ========================
# LỚP HAND TRACKER (MediaPipe)
# ========================
class HandTracker:
    """Theo dõi và nhận diện cử chỉ tay bằng MediaPipe.

    Hỗ trợ 2 backend để chạy được trên nhiều phiên bản Python:
      - "solutions": API cũ mp.solutions.hands (Python 3.9–3.12).
      - "tasks": API mới HandLandmarker + file hand_landmarker.task (cả 3.13).
    Dù backend nào, sau find_hands() ta luôn lưu self.landmarks là list 21 điểm
    (x, y) đã chuẩn hóa [0,1] của bàn tay đầu tiên (hoặc None).
    """

    def __init__(self):
        self.available = HAS_MEDIAPIPE
        self.backend = None
        self.landmarks = None
        self.hands = None
        self.detector = None
        self.mp_hands = None
        self._frame_idx = 0
        if not self.available:
            return

        # Ưu tiên solutions (nhẹ, mượt) nếu có.
        if HAS_MP_SOLUTIONS:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.4,
                )
                self.backend = "solutions"
                return
            except Exception as e:
                print(f"[HAND] Không khởi tạo được solutions API: {e}")

        # Fallback: Tasks API (Python 3.13).
        if HAS_MP_TASKS:
            if not HAND_LANDMARKER_PATH.exists():
                print(f"[HAND] Thiếu model {HAND_LANDMARKER_PATH.name}. Hãy tải hand_landmarker.task vào thư mục models/.")
                self.available = False
                return
            try:
                base = mp_tasks_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base,
                    num_hands=1,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.detector = mp_vision.HandLandmarker.create_from_options(options)
                self.backend = "tasks"
                return
            except Exception as e:
                print(f"[HAND] Không khởi tạo được Tasks API: {e}")

        self.available = False

    def find_hands(self, frame):
        """Phát hiện tay trong frame, lưu self.landmarks (list (x,y) chuẩn hóa)."""
        self.landmarks = None
        if not self.available:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            if self.backend == "solutions":
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    self.landmarks = [(lm.x, lm.y) for lm in hand.landmark]
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)
                if result.hand_landmarks:
                    self.landmarks = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
        except Exception as e:
            print(f"[HAND] Lỗi xử lý frame: {e}")
            self.landmarks = None
        return self.landmarks

    def get_index_finger_tip(self, frame):
        """Lấy tọa độ pixel đầu ngón trỏ (landmark 8)."""
        if self.landmarks:
            h, w = frame.shape[:2]
            x, y = self.landmarks[8]
            return int(x * w), int(y * h)
        return None

    def is_drawing_gesture(self):
        """Cử chỉ vẽ: ngón trỏ duỗi (8 cao hơn 6) và ngón giữa gập (12 thấp hơn 10)."""
        if not self.landmarks:
            return False
        index_up = self.landmarks[8][1] < self.landmarks[6][1]
        middle_down = self.landmarks[12][1] > self.landmarks[10][1]
        return index_up and middle_down

    def is_index_extended(self):
        """Ngón trỏ duỗi — dùng khi đang vẽ để không nhấc bút khi tay đi qua mặt."""
        if not self.landmarks:
            return False
        return self.landmarks[8][1] < self.landmarks[6][1]

    def is_erase_gesture(self):
        """Cử chỉ xóa: xòe cả bàn tay (>=4 ngón duỗi)."""
        if not self.landmarks:
            return False
        fingers_up = 0
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if self.landmarks[tip][1] < self.landmarks[pip][1]:
                fingers_up += 1
        return fingers_up >= 4

    def draw_landmarks(self, frame):
        """Tự vẽ khung xương bàn tay lên frame (dùng chung cho cả 2 backend)."""
        if not self.landmarks:
            return
        h, w = frame.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y in self.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 255, 255), -1)


# ========================
# LỚP DRAWING CANVAS
# ========================
class DrawingCanvas:
    """Canvas vẽ hình trong game"""

    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self.prev_point = None
        self.brush_size = 12

    def draw_line(self, x, y):
        """Vẽ nét từ điểm trước đến điểm hiện tại"""
        if self.prev_point is not None:
            cv2.line(self.canvas, self.prev_point, (x, y), 255, self.brush_size)
        else:
            cv2.circle(self.canvas, (x, y), self.brush_size // 2, 255, -1)
        self.prev_point = (x, y)

    def stop_drawing(self):
        """Dừng vẽ (ngón tay rời canvas)"""
        self.prev_point = None

    def clear(self):
        """Xóa canvas"""
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        self.prev_point = None

    def get_preprocessed_image(self):
        """
        Tiền xử lý canvas thành ảnh 28x28 cho model, KHỚP định dạng QuickDraw mà
        model được huấn luyện (giống image_preprocess.py của backend). Đây là yếu
        tố quyết định độ chính xác khi nhận diện nét vẽ từ camera:

          1) Tách nét bằng ngưỡng Otsu trên vùng có mực (ổn định với nét nhạt/camera).
          2) Cắt sát nét (bounding box).
          3) Làm dày nhẹ nét lớn để chi tiết không biến mất khi thu nhỏ.
          4) Scale nét vừa hộp cố định PREPROC_TARGET_BOX (22px) -> chuẩn hóa kích
             thước, bất kể người dùng vẽ to/nhỏ.
          5) Đặt vào canvas 28x28 rồi DỊCH theo TRỌNG TÂM (center of mass) về giữa
             — đúng cách QuickDraw căn ảnh, bất kể vẽ lệch góc.
          6) Chuẩn hóa pixel về [0, 1].

        Trả về mảng (28, 28) float32, hoặc None nếu canvas rỗng.
        """
        gray = self.canvas  # nền đen (0), nét trắng (255) — đúng polarity QuickDraw

        # 1) Mask nét: ngưỡng thô rồi Otsu trên vùng có mực
        _, rough = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
        ink = gray[rough > 0]
        if ink.size:
            local = np.zeros_like(gray)
            local[rough > 0] = ink
            _, thresh = cv2.threshold(local, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = rough

        # 2) Bounding box vùng có nét
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        crop = gray[y:y + h, x:x + w].copy()
        crop_mask = thresh[y:y + h, x:x + w]

        # 3) Làm dày nhẹ nét lớn (tránh mất chi tiết sau khi resize 28x28)
        if max(w, h) >= 80:
            crop_mask = cv2.dilate(crop_mask, np.ones((2, 2), np.uint8), iterations=1)
            crop = np.maximum(crop, crop_mask)

        # 4) Scale nét vừa hộp PREPROC_TARGET_BOX
        scale = PREPROC_TARGET_BOX / max(w, h, 1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 5) Đặt vào canvas 28x28 (tạm căn giữa hình học)
        out = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        y_start = (CANVAS_SIZE - new_h) // 2
        x_start = (CANVAS_SIZE - new_w) // 2
        out[y_start:y_start + new_h, x_start:x_start + new_w] = resized

        # 5b) Dịch theo trọng tâm về tâm canvas (center of mass) — như QuickDraw
        moments = cv2.moments(out, binaryImage=False)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            shift_x = int(round(PREPROC_CENTER - cx))
            shift_y = int(round(PREPROC_CENTER - cy))
            translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            out = cv2.warpAffine(out, translation, (CANVAS_SIZE, CANVAS_SIZE), borderValue=0)

        # 6) Chuẩn hóa [0, 1]
        return out.astype('float32') / 255.0

    def has_content(self):
        """Kiểm tra canvas có nội dung không"""
        return cv2.findNonZero(self.canvas) is not None

    def get_overlay(self):
        """Lấy canvas dạng BGRA để overlay lên camera"""
        colored = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        # Tô màu cyan cho nét vẽ
        mask = self.canvas > 0
        colored[mask] = [255, 255, 0]  # Cyan (BGR)
        return colored


# ========================
# LỚP AI PREDICTOR
# ========================
class AIPredictor:
    """Nhận diện hình vẽ bằng model CNN đã huấn luyện"""

    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.categories = list(MODEL_CATEGORIES)
        self.load_model(model_path)

    def load_model(self, model_path):
        """Tải model CNN"""
        if not HAS_TENSORFLOW:
            print("[AI] TensorFlow chưa được cài/khởi tạo. Hãy chạy setup_and_run.bat để cài thư viện.")
            self.model = None
            return

        if os.path.exists(model_path):
            try:
                # compile=False giúp load file .h5 ổn định hơn giữa các phiên bản Keras/TensorFlow.
                self.model = tf.keras.models.load_model(model_path, compile=False)
                try:
                    output_classes = int(self.model.output_shape[-1])
                    if output_classes != len(self.categories):
                        print(f"[AI] Model output {output_classes} lớp, file nhãn có {len(self.categories)} lớp. Sẽ dùng {min(output_classes, len(self.categories))} lớp nhận diện được.")
                        self.categories = self.categories[:output_classes]
                except Exception:
                    pass
                print(f"[AI] Đã tải model: {model_path}")
            except Exception as e:
                print(f"[AI] Lỗi tải model: {e}")
                self.model = None
        else:
            print(f"[AI] Không tìm thấy model: {model_path}")
            print(f"[AI] Hãy chạy train_model.py trước để tạo model!")
            self.model = None

    def predict(self, image_28x28):
        """
        Dự đoán từ vựng từ ảnh 28x28
        Returns: (predicted_label, confidence, all_probabilities)
        """
        if self.model is None or image_28x28 is None:
            return None, 0.0, None

        # Reshape: [1, 28, 28, 1]
        input_data = image_28x28.reshape(1, CANVAS_SIZE, CANVAS_SIZE, 1)

        # Dự đoán
        probs = self.model.predict(input_data, verbose=0)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        pred_label = self.categories[pred_idx] if pred_idx < len(self.categories) else ""

        return pred_label, float(confidence), probs

    def get_top_predictions(self, image_28x28, top_k=3):
        """Lấy top-k dự đoán"""
        if self.model is None or image_28x28 is None:
            return []

        input_data = image_28x28.reshape(1, CANVAS_SIZE, CANVAS_SIZE, 1)
        probs = self.model.predict(input_data, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.categories):
                results.append((self.categories[idx], float(probs[idx])))
        return results


# ========================
# LỚP GAME CHÍNH
# ========================
class AirDrawVocabGame:
    """
    Game học tiếng Anh qua vẽ hình trong không khí
    - Người chơi vẽ từ vựng bằng cử chỉ tay qua webcam
    - AI nhận diện hình vẽ và cho điểm
    - Hiển thị nghĩa tiếng Việt, IPA, ví dụ
    """

    def __init__(self):
        # Khởi tạo Pygame
        pygame.init()
        pygame.display.set_caption("Draw & Learn English - AirDrawVocab")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Font hỗ trợ tiếng Việt (nạp trực tiếp file .ttf để không bị vỡ dấu)
        self.font_large = load_vn_font(28, bold=True)
        self.font_medium = load_vn_font(22)
        self.font_small = load_vn_font(17)
        self.font_tiny = load_vn_font(14)
        self.font_title = load_vn_font(44, bold=True)
        self.font_huge = load_vn_font(60, bold=True)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.cap.isOpened():
            print("[CAMERA] Khong the mo webcam! Van co the ve bang CHUOT.")
            print("[CAMERA] Giu chuot trai va keo trong vung camera de ve.")

        # Các thành phần chính
        self.hand_tracker = HandTracker()
        self.drawing_canvas = DrawingCanvas()
        self.ai_predictor = AIPredictor()
        self.tts = TTSEngine()

        # Trạng thái game
        self.state = "MENU"  # MENU, PLAYING, RESULT, VOCAB_INFO, GAME_OVER
        self.score = 0
        self.lives = MAX_LIVES
        self.streak = 0
        self.current_level = 1
        self.total_levels = TOTAL_LEVELS

        # Từ vựng
        self.word_list = []
        self.current_word = ""
        self.time_left = TIME_PER_WORD
        self.timer_start = 0

        # Kết quả dự đoán
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.prediction_correct = False

        # AI real-time predictions
        self.realtime_prediction = ""
        self.realtime_confidence = 0.0
        self.prediction_timer = 0
        self.prediction_interval = 0.5  # Dự đoán mỗi 0.5 giây
        # Làm mượt dự đoán: trung bình xác suất qua vài lần gần nhất để giảm
        # nhấp nháy và bắt đáp án ổn định hơn khi vẽ bằng camera.
        self.prob_history = deque(maxlen=4)

        # Hiển thị vocab info
        self.show_vocab_timer = 0
        self.vocab_display_duration = 5  # Hiển thị thông tin từ 5 giây

        # Animation
        self.feedback_text = ""

        # === VE BANG CHUOT ===
        self.mouse_drawing = False
        self.cam_x = (WINDOW_WIDTH - CAMERA_WIDTH) // 2
        self.cam_y = 102
        self.no_camera = not self.cap.isOpened()

        # Trạng thái bút tay: giữ nét khi tay đi qua mặt / mất tracking tạm thời.
        self.pen_down = False
        self.pen_miss_frames = 0
        self.PEN_MISS_TOLERANCE = 12

        # === AI ASSISTANT ===
        self.ai_manager = None
        self.ai_hint_text = ""
        self.ai_encouragement = ""
        self.show_hint = False
        if HAS_AI:
            model = self.ai_predictor.model if self.ai_predictor.model else None
            self.ai_manager = AIManager(model, self.ai_predictor.categories)
        self.feedback_color = GREEN
        self.feedback_timer = 0

    def generate_word_list(self):
        """Tạo danh sách từ vựng ngẫu nhiên cho game"""
        words = list(self.ai_predictor.categories or MODEL_CATEGORIES or ALL_CATEGORIES)
        random.shuffle(words)
        self.total_levels = min(TOTAL_LEVELS, len(words))
        self.word_list = words[:self.total_levels]

    def start_game(self):
        """Bắt đầu game mới"""
        self.state = "PLAYING"
        self.score = 0
        self.lives = MAX_LIVES
        self.streak = 0
        self.current_level = 1
        self.generate_word_list()
        self.next_word()

    def next_word(self):
        """Chuyển sang từ tiếp theo"""
        if self.current_level > self.total_levels or self.current_level > len(self.word_list):
            self.state = "GAME_OVER"
            return

        self.current_word = self.word_list[self.current_level - 1]
        self.time_left = TIME_PER_WORD
        self.timer_start = time.time()
        self.drawing_canvas.clear()
        self.pen_down = False
        self.pen_miss_frames = 0
        self.realtime_prediction = ""
        self.realtime_confidence = 0.0
        self.prediction_timer = time.time()
        self.prob_history.clear()

    def check_prediction(self):
        """Kiểm tra dự đoán của AI với từ hiện tại"""
        if not self.drawing_canvas.has_content():
            return

        img = self.drawing_canvas.get_preprocessed_image()
        if img is None:
            return

        pred_label, confidence, _ = self.ai_predictor.predict(img)

        if pred_label == self.current_word and confidence >= CONFIDENCE_THRESHOLD:
            # Đúng!
            self.prediction_correct = True
            self.last_prediction = pred_label
            self.last_confidence = confidence
            bonus = max(1, int(self.time_left / 10))
            streak_bonus = self.streak
            self.score += 10 + bonus + streak_bonus
            self.streak += 1
            self.feedback_text = f"Correct! +{10 + bonus + streak_bonus} points"
            self.feedback_color = GREEN
            self.feedback_timer = time.time()

            # Phát âm từ
            self.tts.speak(self.current_word)

            # Hiển thị thông tin từ vựng
            self.state = "VOCAB_INFO"
            self.show_vocab_timer = time.time()
        else:
            self.last_prediction = pred_label if pred_label else ""
            self.last_confidence = confidence

    def handle_timeout(self):
        """Xử lý khi hết thời gian"""
        self.lives -= 1
        self.streak = 0
        self.feedback_text = f"Time's up! The word was: {self.current_word}"
        self.feedback_color = RED
        self.feedback_timer = time.time()

        if self.lives <= 0:
            self.state = "GAME_OVER"
        else:
            self.state = "VOCAB_INFO"
            self.show_vocab_timer = time.time()
            self.prediction_correct = False
            self.tts.speak(self.current_word)

    def update_realtime_prediction(self):
        """Cập nhật dự đoán real-time (có làm mượt qua nhiều khung)."""
        current_time = time.time()
        if current_time - self.prediction_timer >= self.prediction_interval:
            self.prediction_timer = current_time
            if self.drawing_canvas.has_content():
                img = self.drawing_canvas.get_preprocessed_image()
                if img is not None:
                    pred_label, confidence, probs = self.ai_predictor.predict(img)
                    # Làm mượt: trung bình xác suất qua vài lần gần nhất.
                    if probs is not None:
                        self.prob_history.append(np.asarray(probs, dtype="float32"))
                        avg = np.mean(self.prob_history, axis=0)
                        idx = int(np.argmax(avg))
                        cats = self.ai_predictor.categories
                        pred_label = cats[idx] if idx < len(cats) else (pred_label or "")
                        confidence = float(avg[idx])
                    self.realtime_prediction = pred_label if pred_label else ""
                    self.realtime_confidence = confidence

                    # Auto-check: nếu đúng từ với confidence cao
                    if pred_label == self.current_word and confidence >= CONFIDENCE_THRESHOLD:
                        self.check_prediction()
            else:
                self.prob_history.clear()

    # ========================
    # VẼ GIAO DIỆN
    # ========================

    def _menu_button_rect(self):
        return pygame.Rect(WINDOW_WIDTH // 2 - 130, WINDOW_HEIGHT - 92, 260, 56)

    def draw_menu(self):
        """Vẽ màn hình menu chính (thiết kế lại: gradient + thẻ + nút hover)."""
        mouse_pos = pygame.mouse.get_pos()
        draw_vertical_gradient(self.screen, COL_BG_TOP, COL_BG_BOTTOM)

        # Tiêu đề + thanh nhấn
        title = self.font_huge.render("AirDrawVocab", True, COL_ACCENT)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 64))
        pygame.draw.rect(
            self.screen, COL_ACCENT_2,
            (WINDOW_WIDTH // 2 - 60, 64 + title.get_height() + 6, 120, 4),
            border_radius=2,
        )
        subtitle = self.font_medium.render("Học Tiếng Anh Qua Vẽ Hình Trực Tuyến", True, COL_MUTED)
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 150))

        # Thẻ hướng dẫn ở giữa
        card = pygame.Rect(WINDOW_WIDTH // 2 - 270, 200, 540, 278)
        draw_panel(self.screen, card, color=COL_CARD, radius=20)
        head = self.font_large.render("Cách chơi", True, COL_TEXT)
        self.screen.blit(head, (card.x + 28, card.y + 22))

        instructions = [
            ("1", "Giơ ngón trỏ lên để vẽ trong không khí"),
            ("2", "Mở bàn tay (5 ngón) để dừng / nhấc bút"),
            ("3", "AI sẽ nhận diện hình vẽ của bạn theo thời gian thực"),
            ("4", "Vẽ đúng từ vựng trước khi hết giờ để ghi điểm"),
        ]
        row_gap = 38
        y = card.y + 72
        for num, line in instructions:
            badge = pygame.Rect(card.x + 28, y, 26, 26)
            pygame.draw.rect(self.screen, COL_ACCENT_2, badge, border_radius=8)
            num_s = self.font_small.render(num, True, COL_BG_BOTTOM)
            self.screen.blit(num_s, (badge.centerx - num_s.get_width() // 2, badge.centery - num_s.get_height() // 2))
            text = self.font_small.render(line, True, COL_TEXT)
            self.screen.blit(text, (badge.right + 14, y + 4))
            y += row_gap

        divider_y = y + 10
        pygame.draw.line(
            self.screen, COL_LINE,
            (card.x + 28, divider_y), (card.right - 28, divider_y), 1,
        )
        note = self.font_small.render(
            f"Bạn có {MAX_LIVES} mạng · {TIME_PER_WORD}s mỗi từ · nhấn H để xem gợi ý",
            True, COL_MUTED,
        )
        self.screen.blit(note, (card.centerx - note.get_width() // 2, divider_y + 16))

        # Nút START GAME
        btn_rect = draw_button(
            self.screen, self._menu_button_rect(), "BẮT ĐẦU CHƠI",
            self.font_large, COL_SUCCESS, text_color=COL_BG_BOTTOM, mouse_pos=mouse_pos,
        )

        # Credits
        credit = self.font_tiny.render("Nhóm 1 - ĐH Phenikaa  |  Mạng Nơron và Học Sâu", True, COL_MUTED)
        self.screen.blit(credit, (WINDOW_WIDTH // 2 - credit.get_width() // 2, WINDOW_HEIGHT - 26))

        return btn_rect

    def get_game_button_rects(self):
        """Vị trí nút overlay giống ảnh minh họa: Menu trái, Submit giữa, Clear phải."""
        button_y = self.cam_y + CAMERA_HEIGHT - 50
        menu_btn = pygame.Rect(self.cam_x + 12, button_y, 96, 36)
        submit_btn = pygame.Rect(WINDOW_WIDTH // 2 - 58, button_y, 116, 36)
        clear_btn = pygame.Rect(self.cam_x + CAMERA_WIDTH - 108, button_y, 96, 36)
        return menu_btn, clear_btn, submit_btn

    def _hud_stat(self, x, y, w, label, value, value_color=COL_TEXT):
        """Vẽ một ô thống kê nhỏ trong HUD."""
        cell = pygame.Rect(x, y, w, 56)
        draw_panel(self.screen, cell, color=COL_CARD_SOFT, radius=12, shadow=False)
        lab = self.font_tiny.render(label, True, COL_MUTED)
        self.screen.blit(lab, (cell.x + 12, cell.y + 8))
        val = self.font_large.render(str(value), True, value_color)
        self.screen.blit(val, (cell.x + 12, cell.y + 24))

    def draw_game_ui(self, camera_surface):
        """Vẽ giao diện game: HUD dạng thẻ thống kê, camera bo góc, nút hover."""
        mouse_pos = pygame.mouse.get_pos()
        draw_vertical_gradient(self.screen, COL_BG_TOP, COL_BG_BOTTOM)

        cam_x = self.cam_x
        cam_y = self.cam_y

        # Thanh tiêu đề trên: từ cần vẽ (trái) + các ô thống kê (phải).
        left_x = 20
        word_label = self.font_tiny.render("ĐANG VẼ", True, COL_MUTED)
        self.screen.blit(word_label, (left_x, 12))
        word_text = self.font_title.render(self.current_word, True, COL_ACCENT)
        word_y = 28
        self.screen.blit(word_text, (left_x, word_y))
        streak_y = word_y + word_text.get_height() + 10
        streak_s = self.font_small.render(f"Streak: {self.streak}", True, COL_WARNING)
        streak_bg = pygame.Rect(left_x - 4, streak_y - 2, streak_s.get_width() + 16, streak_s.get_height() + 8)
        draw_panel(self.screen, streak_bg, color=COL_CARD_SOFT, radius=10, shadow=False, border_w=0)
        self.screen.blit(streak_s, (left_x + 4, streak_y + 2))

        stat_w = 92
        gap = 8
        total = stat_w * 4 + gap * 3
        sx = WINDOW_WIDTH - total - 18
        time_color = COL_DANGER if self.time_left <= 12 else COL_TEXT
        lives_str = "♥" * self.lives + "·" * (MAX_LIVES - self.lives)
        self._hud_stat(sx, 16, stat_w, "TIME", f"{int(self.time_left)}s", time_color)
        self._hud_stat(sx + (stat_w + gap), 16, stat_w, "SCORE", self.score, COL_ACCENT_2)
        self._hud_stat(sx + (stat_w + gap) * 2, 16, stat_w, "LEVEL", f"{self.current_level}/{self.total_levels}")
        self._hud_stat(sx + (stat_w + gap) * 3, 16, stat_w, "LIVES", lives_str, COL_DANGER)

        # Khung camera bo góc.
        if camera_surface.get_width() != CAMERA_WIDTH or camera_surface.get_height() != CAMERA_HEIGHT:
            camera_surface = pygame.transform.smoothscale(camera_surface, (CAMERA_WIDTH, CAMERA_HEIGHT))
        frame_rect = pygame.Rect(cam_x - 3, cam_y - 3, CAMERA_WIDTH + 6, CAMERA_HEIGHT + 6)
        pygame.draw.rect(self.screen, COL_LINE, frame_rect, border_radius=14)
        self.screen.blit(camera_surface, (cam_x, cam_y))
        pygame.draw.rect(self.screen, COL_ACCENT, (cam_x, cam_y, CAMERA_WIDTH, CAMERA_HEIGHT), 2, border_radius=12)

        # Badge dự đoán real-time.
        if self.realtime_prediction:
            correct = self.realtime_prediction == self.current_word
            pred_color = COL_SUCCESS if correct else COL_TEXT
            pred_text = self.font_small.render(
                f"AI: {self.realtime_prediction}  {self.realtime_confidence:.0%}",
                True, pred_color,
            )
            badge = pygame.Rect(cam_x + 14, cam_y + 14, pred_text.get_width() + 24, 34)
            draw_panel(self.screen, badge, color=(8, 15, 30), radius=17,
                       border=COL_SUCCESS if correct else COL_LINE, shadow=False)
            self.screen.blit(pred_text, (badge.x + 12, badge.y + 8))

        # Gợi ý (nhấn H).
        if self.show_hint and self.ai_hint_text:
            hint = self.font_small.render(f"Gợi ý: {self.ai_hint_text}", True, COL_WARNING)
            hb = pygame.Rect(cam_x + 14, cam_y + 54, hint.get_width() + 24, 32)
            draw_panel(self.screen, hb, color=(8, 15, 30), radius=12, border=COL_WARNING, shadow=False)
            self.screen.blit(hint, (hb.x + 12, hb.y + 7))

        # Nút điều khiển (hover).
        menu_btn, clear_btn, submit_btn = self.get_game_button_rects()
        draw_button(self.screen, menu_btn, "Menu", self.font_small, COL_CARD_SOFT, mouse_pos=mouse_pos, radius=10)
        draw_button(self.screen, submit_btn, "Nhận diện", self.font_small, COL_ACCENT, text_color=COL_BG_BOTTOM, mouse_pos=mouse_pos, radius=10)
        draw_button(self.screen, clear_btn, "Xóa", self.font_small, COL_DANGER, mouse_pos=mouse_pos, radius=10)

        # Phản hồi đúng/sai.
        if self.feedback_text and time.time() - self.feedback_timer < 3:
            fb = self.font_medium.render(self.feedback_text, True, COL_BG_BOTTOM)
            bg = pygame.Rect(WINDOW_WIDTH // 2 - fb.get_width() // 2 - 18, WINDOW_HEIGHT - 40, fb.get_width() + 36, 32)
            pygame.draw.rect(self.screen, self.feedback_color, bg, border_radius=16)
            self.screen.blit(fb, (WINDOW_WIDTH // 2 - fb.get_width() // 2, WINDOW_HEIGHT - 35))

        return menu_btn, clear_btn, submit_btn

    def draw_vocab_info(self):
        """Hiển thị thông tin từ vựng sau khi dự đoán (thiết kế lại dạng thẻ)."""
        draw_vertical_gradient(self.screen, COL_BG_TOP, COL_BG_BOTTOM)

        word = self.current_word
        vocab = VOCAB_DATA.get(word, {})

        # Banner kết quả
        ok = self.prediction_correct
        banner_color = COL_SUCCESS if ok else COL_DANGER
        title = self.font_huge.render("CHÍNH XÁC!" if ok else "HẾT GIỜ!", True, banner_color)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 44))

        word_text = self.font_title.render(word.upper(), True, COL_ACCENT)
        self.screen.blit(word_text, (WINDOW_WIDTH // 2 - word_text.get_width() // 2, 120))

        # Thẻ thông tin từ vựng
        card = pygame.Rect(WINDOW_WIDTH // 2 - 290, 200, 580, 270)
        draw_panel(self.screen, card, color=COL_CARD, radius=20)

        info_items = [
            ("Nghĩa", vocab.get("vietnamese", "N/A")),
            ("Phiên âm", vocab.get("ipa", "N/A")),
            ("Ví dụ", vocab.get("example", "N/A")),
            ("Dịch", vocab.get("example_vi", "N/A")),
        ]
        y = card.y + 26
        for label, value in info_items:
            label_surf = self.font_small.render(label.upper(), True, COL_ACCENT_2)
            self.screen.blit(label_surf, (card.x + 28, y))
            value_surf = self.font_medium.render(str(value), True, COL_TEXT)
            self.screen.blit(value_surf, (card.x + 28, y + 22))
            y += 60

        # Dải điểm
        if ok:
            score_text = self.font_medium.render(f"Điểm: {self.score}   ·   Streak: {self.streak}", True, COL_SUCCESS)
        else:
            score_text = self.font_medium.render(f"Mạng còn lại: {self.lives}", True, COL_DANGER)
        self.screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, card.bottom + 18))

        continue_text = self.font_small.render("Nhấn SPACE hoặc đợi 5 giây để tiếp tục...", True, COL_MUTED)
        self.screen.blit(continue_text, (WINDOW_WIDTH // 2 - continue_text.get_width() // 2, WINDOW_HEIGHT - 44))

    def get_game_over_rects(self):
        replay_btn = pygame.Rect(WINDOW_WIDTH // 2 - 270, 470, 260, 56)
        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 + 10, 470, 260, 56)
        return replay_btn, menu_btn

    def draw_game_over(self):
        """Màn hình kết thúc game (thiết kế lại: thẻ tổng kết + 2 nút hover)."""
        mouse_pos = pygame.mouse.get_pos()
        draw_vertical_gradient(self.screen, COL_BG_TOP, COL_BG_BOTTOM)

        title = self.font_huge.render("KẾT THÚC", True, COL_DANGER)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 80))

        # Thẻ tổng kết
        card = pygame.Rect(WINDOW_WIDTH // 2 - 260, 190, 520, 240)
        draw_panel(self.screen, card, color=COL_CARD, radius=20)

        completed = min(self.current_level - 1, self.total_levels)
        stats = [
            ("Tổng điểm", str(self.score), COL_WARNING),
            ("Hoàn thành", f"{completed}/{self.total_levels} từ", COL_ACCENT),
            ("Streak tốt nhất", str(self.streak), COL_ACCENT_2),
        ]
        cell_w = (card.width - 28 * 2 - 20 * 2) // 3
        cx = card.x + 28
        for label, value, color in stats:
            cell = pygame.Rect(cx, card.y + 30, cell_w, 110)
            draw_panel(self.screen, cell, color=COL_CARD_SOFT, radius=14, shadow=False)
            lab = self.font_small.render(label, True, COL_MUTED)
            self.screen.blit(lab, (cell.centerx - lab.get_width() // 2, cell.y + 16))
            val = self.font_title.render(value, True, color)
            self.screen.blit(val, (cell.centerx - val.get_width() // 2, cell.y + 48))
            cx += cell_w + 20

        tip = self.font_small.render("Luyện vẽ đều mỗi ngày để AI nhận diện nét của bạn tốt hơn!", True, COL_MUTED)
        self.screen.blit(tip, (WINDOW_WIDTH // 2 - tip.get_width() // 2, card.bottom - 40))

        replay_btn, menu_btn = self.get_game_over_rects()
        draw_button(self.screen, replay_btn, "CHƠI LẠI", self.font_large, COL_SUCCESS, text_color=COL_BG_BOTTOM, mouse_pos=mouse_pos)
        draw_button(self.screen, menu_btn, "VỀ MENU", self.font_large, COL_CARD_SOFT, mouse_pos=mouse_pos)

        return replay_btn, menu_btn

    # ========================
    # VÒNG LẶP CHÍNH
    # ========================

    def run(self):
        """Vòng lặp chính của game"""
        running = True

        while running:
            # Doc camera (hoac tao frame trong neu khong co camera)
            if self.no_camera:
                ret = False
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "VE BANG CHUOT", (160, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(frame, "Giu chuot trai + keo de ve", (120, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                frame = cv2.flip(frame, 1)

            # QUAN TRỌNG: webcam thật có thể trả về độ phân giải khác (vd 640x480)
            # so với CANVAS overlay (760x465). Nếu không đồng bộ kích thước, phép
            # frame[mask] / addWeighted ở state PLAYING sẽ lệch shape -> CRASH/đóng
            # app ngay khi bấm START. Ép frame về đúng kích thước canvas.
            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            # --- XỬ LÝ EVENTS ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == "PLAYING":
                            self.state = "MENU"
                        else:
                            running = False

                    elif event.key == pygame.K_SPACE:
                        if self.state == "VOCAB_INFO":
                            self.current_level += 1
                            self.next_word()
                            self.state = "PLAYING"
                        elif self.state == "MENU":
                            self.start_game()

                    elif event.key == pygame.K_c and self.state == "PLAYING":
                        self.drawing_canvas.clear()

                    elif event.key == pygame.K_h and self.state == "PLAYING":
                        if self.ai_manager:
                            self.ai_hint_text = self.ai_manager.get_drawing_hint(self.current_word)
                            self.show_hint = True

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()

                    if self.state == "MENU":
                        if self._menu_button_rect().collidepoint(mouse_pos):
                            self.start_game()

                    elif self.state == "VOCAB_INFO":
                        self.current_level += 1
                        self.next_word()
                        self.state = "PLAYING"

                    elif self.state == "GAME_OVER":
                        replay_btn, menu_btn = self.get_game_over_rects()
                        if replay_btn.collidepoint(mouse_pos):
                            self.start_game()
                        elif menu_btn.collidepoint(mouse_pos):
                            self.state = "MENU"

                    elif self.state == "PLAYING":
                        mx, my = mouse_pos
                        button_clicked = False
                        menu_r, clear_r, submit_r = self.get_game_button_rects()
                        if clear_r.collidepoint(mx, my):
                            self.drawing_canvas.clear(); button_clicked = True
                        if menu_r.collidepoint(mx, my):
                            self.state = "MENU"; button_clicked = True
                        if submit_r.collidepoint(mx, my):
                            self.check_prediction(); button_clicked = True
                        if not button_clicked:
                            cx, cy = mx - self.cam_x, my - self.cam_y
                            if 0 <= cx < CAMERA_WIDTH and 0 <= cy < CAMERA_HEIGHT:
                                self.mouse_drawing = True
                                self.drawing_canvas.draw_line(cx, cy)

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.mouse_drawing:
                        self.mouse_drawing = False
                        self.drawing_canvas.stop_drawing()

                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_drawing and self.state == "PLAYING":
                        mx, my = event.pos
                        cx, cy = mx - self.cam_x, my - self.cam_y
                        if 0 <= cx < CAMERA_WIDTH and 0 <= cy < CAMERA_HEIGHT:
                            self.drawing_canvas.draw_line(cx, cy)
                        else:
                            self.mouse_drawing = False
                            self.drawing_canvas.stop_drawing()


            # --- CẬP NHẬT TRẠNG THÁI ---
            if self.state == "PLAYING":
                # Hand tracking — bút dính: sau khi bắt đầu vẽ, giữ nét khi tay đi qua mặt
                # hoặc mất tracking tạm thời; chỉ nhấc bút khi xòe tay hoặc mất tay lâu.
                self.hand_tracker.find_hands(frame)
                finger_pos = self.hand_tracker.get_index_finger_tip(frame)
                erase_gesture = self.hand_tracker.is_erase_gesture()
                start_gesture = self.hand_tracker.is_drawing_gesture()
                continue_gesture = self.pen_down and self.hand_tracker.is_index_extended()

                if erase_gesture:
                    self.pen_down = False
                    self.pen_miss_frames = 0
                    self.drawing_canvas.stop_drawing()
                elif finger_pos and (start_gesture or continue_gesture):
                    cx, cy = finger_pos
                    self.pen_down = True
                    self.pen_miss_frames = 0
                    self.drawing_canvas.draw_line(cx, cy)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                elif self.pen_down and self.pen_miss_frames < self.PEN_MISS_TOLERANCE:
                    self.pen_miss_frames += 1
                    if finger_pos:
                        cx, cy = finger_pos
                        self.drawing_canvas.draw_line(cx, cy)
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                else:
                    self.pen_down = False
                    self.pen_miss_frames = 0
                    self.drawing_canvas.stop_drawing()

                # Vẽ landmarks
                self.hand_tracker.draw_landmarks(frame)

                # Overlay canvas lên camera
                overlay = self.drawing_canvas.get_overlay()
                mask = self.drawing_canvas.canvas > 0
                frame[mask] = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)[mask]

                # Hien con tro chuot tren camera
                if self.mouse_drawing:
                    mx, my = pygame.mouse.get_pos()
                    cx, cy = mx - self.cam_x, my - self.cam_y
                    if 0 <= cx < CAMERA_WIDTH and 0 <= cy < CAMERA_HEIGHT:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                # Cập nhật timer
                elapsed = time.time() - self.timer_start
                self.time_left = max(0, TIME_PER_WORD - elapsed)

                if self.time_left <= 0:
                    self.handle_timeout()

                # Dự đoán real-time
                self.update_realtime_prediction()

                # Chuyển frame sang Pygame surface
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))

                # Vẽ UI
                menu_btn, clear_btn, submit_btn = self.draw_game_ui(frame_surface)

                # Xử lý click buttons
                mouse_pressed = pygame.mouse.get_pressed()
                if mouse_pressed[0]:
                    mpos = pygame.mouse.get_pos()
                    if clear_btn.collidepoint(mpos):
                        self.drawing_canvas.clear()
                    elif menu_btn.collidepoint(mpos):
                        self.state = "MENU"
                    elif submit_btn.collidepoint(mpos):
                        self.check_prediction()

            elif self.state == "MENU":
                self.draw_menu()

            elif self.state == "VOCAB_INFO":
                self.draw_vocab_info()
                # Auto-continue sau 5 giây
                if time.time() - self.show_vocab_timer >= self.vocab_display_duration:
                    self.current_level += 1
                    self.next_word()
                    self.state = "PLAYING"

            elif self.state == "GAME_OVER":
                self.draw_game_over()

            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)

        # Cleanup
        self.cap.release()
        pygame.quit()
        sys.exit(0)


# ========================
# MAIN
# ========================
def _run_face_enroll(username: str, camera_index: int = 0):
    """Đăng ký khuôn mặt từ webcam trước khi chơi."""
    try:
        from face_auth import FaceAuthManager
        manager = FaceAuthManager()
        result = manager.enroll_from_camera(username=username, camera_index=camera_index)
        print(result.message)
        return result.ok
    except Exception as exc:
        print(f"[FACE] Không thể đăng ký khuôn mặt: {exc}")
        return False


def _run_face_login(username: str = None, camera_index: int = 0):
    """Xác thực khuôn mặt từ webcam trước khi chơi."""
    try:
        from face_auth import FaceAuthManager
        manager = FaceAuthManager()
        result = manager.verify_from_camera(username=username, camera_index=camera_index)
        print(result.message)
        return result.ok
    except Exception as exc:
        print(f"[FACE] Không thể xác thực khuôn mặt: {exc}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirDrawVocab Unified: game + chatbot + face recognition")
    parser.add_argument("--enroll-face", metavar="NAME", help="Đăng ký khuôn mặt cho người dùng rồi thoát")
    parser.add_argument("--face-login", action="store_true", help="Xác thực khuôn mặt trước khi mở game")
    parser.add_argument("--face-user", default=None, help="Tên người dùng cần xác thực, bỏ trống để tự nhận diện người gần nhất")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, mặc định 0")
    args = parser.parse_args()

    print("=" * 50)
    print("  AirDrawVocab Unified AI")
    print("  Vẽ hình + chatbot + nhận diện khuôn mặt")
    print("=" * 50)
    print()

    if args.enroll_face:
        ok = _run_face_enroll(args.enroll_face, args.camera)
        sys.exit(0 if ok else 1)

    if args.face_login:
        ok = _run_face_login(args.face_user, args.camera)
        if not ok:
            print("[FACE] Xác thực thất bại. Game sẽ không mở để tránh đăng nhập sai người dùng.")
            sys.exit(1)

    # Kiểm tra model
    if not os.path.exists(MODEL_PATH):
        fallback_model = "airdrawvocab_model.h5"
        if os.path.exists(fallback_model):
            MODEL_PATH = fallback_model
        else:
            print(f"[!] Không tìm thấy model '{MODEL_PATH}'")
            print("[!] Hãy kiểm tra thư mục models/ hoặc chạy train_model.py để huấn luyện model.")
            print()
            ans = input("Bạn có muốn tiếp tục không có model? (y/n): ").strip().lower()
            if ans != 'y':
                sys.exit(0)

    try:
        game = AirDrawVocabGame()
        game.run()
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        log_path = Path(__file__).resolve().parent / "game_crash.log"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(err)
        except Exception:
            pass
        print("=" * 50)
        print("[LỖI] Game gặp sự cố và phải đóng. Chi tiết:")
        print(err)
        print(f"[LỖI] Đã lưu log vào: {log_path}")
        print("=" * 50)
        try:
            input("Nhấn Enter để thoát...")
        except Exception:
            pass
        sys.exit(1)
