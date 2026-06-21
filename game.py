"""
=================================================================
AirDrawVocab - Trò chơi học tiếng Anh qua vẽ hình trong không khí
=================================================================
Đại học Phenikaa - Trường Công nghệ Thông tin
Môn: Mạng Nơron và Học Sâu
Nhóm 1: Hoàng Quốc Mạnh (22010144), Nguyễn Hữu Vinh (22010283)
Giảng viên: TS. Đặng Thị Thúy An

Yêu cầu: Python 3.8+, webcam
Cài đặt: pip install pygame mediapipe opencv-python tensorflow pyttsx3 numpy
Chạy: python game.py
=================================================================
"""

import os
import sys
import json
from pathlib import Path
import time
import random
import threading
import numpy as np
import cv2
import pygame

# Các thư viện nặng: cho phép fallback để game vẫn mở được bằng chuột nếu thiếu webcam/MediaPipe/TTS.
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise AttributeError("mediapipe đã import được nhưng thiếu mp.solutions. Hãy cài lại mediapipe trong Python 3.11/3.12.")
    HAS_MEDIAPIPE = True
except Exception as e:
    mp = None
    HAS_MEDIAPIPE = False
    print(f"[SETUP] MediaPipe chưa sẵn sàng: {e}. Game vẫn có thể vẽ bằng chuột nếu TensorFlow hoạt động.")

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

# Đường dẫn model: import từ config dùng chung
from config import MODEL_PATH

CATEGORIES_JSON_PATH = Path(__file__).resolve().parent / "models" / "categories.json"


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
    """Theo dõi và nhận diện cử chỉ tay bằng MediaPipe"""

    def __init__(self):
        self.results = None
        self.available = HAS_MEDIAPIPE
        if not self.available:
            self.mp_hands = None
            self.mp_draw = None
            self.hands = None
            return
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def find_hands(self, frame):
        """Phát hiện tay trong frame"""
        if not self.available:
            self.results = None
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return self.results

    def get_index_finger_tip(self, frame):
        """Lấy tọa độ đầu ngón trỏ (landmark 8)"""
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            tip = hand.landmark[8]  # INDEX_FINGER_TIP
            cx, cy = int(tip.x * w), int(tip.y * h)
            return cx, cy
        return None

    def is_drawing_gesture(self):
        """
        Kiểm tra cử chỉ vẽ: ngón trỏ duỗi, các ngón khác gập
        - Ngón trỏ (tip 8) cao hơn PIP (6)
        - Ngón giữa (tip 12) thấp hơn PIP (10)
        """
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            # Ngón trỏ duỗi
            index_up = hand.landmark[8].y < hand.landmark[6].y
            # Ngón giữa gập
            middle_down = hand.landmark[12].y > hand.landmark[10].y
            return index_up and middle_down
        return False

    def is_erase_gesture(self):
        """
        Cử chỉ xóa: tất cả ngón tay duỗi (bàn tay mở)
        """
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            fingers_up = 0
            # Ngón trỏ
            if hand.landmark[8].y < hand.landmark[6].y:
                fingers_up += 1
            # Ngón giữa
            if hand.landmark[12].y < hand.landmark[10].y:
                fingers_up += 1
            # Ngón áp út
            if hand.landmark[16].y < hand.landmark[14].y:
                fingers_up += 1
            # Ngón út
            if hand.landmark[20].y < hand.landmark[18].y:
                fingers_up += 1
            return fingers_up >= 4
        return False

    def draw_landmarks(self, frame):
        """Vẽ các điểm landmark lên frame"""
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )


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
        Tiền xử lý canvas thành ảnh 28x28 cho model
        - Crop vùng có nội dung
        - Resize về 28x28
        - Chuẩn hóa [0, 1]
        """
        # Tìm bounding box
        coords = cv2.findNonZero(self.canvas)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        # Thêm padding
        pad = 30
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(self.width - x, w + 2 * pad)
        h = min(self.height - y, h + 2 * pad)

        # Crop
        roi = self.canvas[y:y + h, x:x + w]

        # Tạo ảnh vuông
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square[offset_y:offset_y + h, offset_x:offset_x + w] = roi

        # Resize về 28x28
        resized = cv2.resize(square, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_AREA)

        # Chuẩn hóa [0, 1]
        normalized = resized.astype('float32') / 255.0

        return normalized

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

        # Font
        self.font_large = pygame.font.SysFont("Arial", 30, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 23)
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_title = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_huge = pygame.font.SysFont("Arial", 64, bold=True)

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

        # Hiển thị vocab info
        self.show_vocab_timer = 0
        self.vocab_display_duration = 5  # Hiển thị thông tin từ 5 giây

        # Animation
        self.feedback_text = ""

        # === VE BANG CHUOT ===
        self.mouse_drawing = False
        self.cam_x = (WINDOW_WIDTH - CAMERA_WIDTH) // 2
        self.cam_y = 88
        self.no_camera = not self.cap.isOpened()

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
        self.realtime_prediction = ""
        self.realtime_confidence = 0.0
        self.prediction_timer = time.time()

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
        """Cập nhật dự đoán real-time"""
        current_time = time.time()
        if current_time - self.prediction_timer >= self.prediction_interval:
            self.prediction_timer = current_time
            if self.drawing_canvas.has_content():
                img = self.drawing_canvas.get_preprocessed_image()
                if img is not None:
                    pred_label, confidence, _ = self.ai_predictor.predict(img)
                    self.realtime_prediction = pred_label if pred_label else ""
                    self.realtime_confidence = confidence

                    # Auto-check: nếu đúng từ với confidence cao
                    if pred_label == self.current_word and confidence >= CONFIDENCE_THRESHOLD:
                        self.check_prediction()

    # ========================
    # VẼ GIAO DIỆN
    # ========================

    def draw_menu(self):
        """Vẽ màn hình menu chính"""
        self.screen.fill(DARK_BG)

        # Tiêu đề
        title = self.font_huge.render("AirDrawVocab", True, CYAN)
        subtitle = self.font_medium.render("Học Tiếng Anh Qua Vẽ Hình Trực Tuyến", True, LIGHT_GRAY)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 120))
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 200))

        # Hướng dẫn
        instructions = [
            "Cách chơi:",
            "1. Giơ ngón trỏ lên để VẼ trong không khí",
            "2. Mở bàn tay (5 ngón) để DỪNG vẽ",
            "3. AI sẽ nhận diện hình vẽ của bạn",
            "4. Vẽ đúng từ vựng để ghi điểm!",
            "",
            f"Bạn có {MAX_LIVES} mạng và {TIME_PER_WORD}s cho mỗi từ"
        ]
        y_offset = 300
        for line in instructions:
            text = self.font_small.render(line, True, LIGHT_GRAY)
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y_offset))
            y_offset += 30

        # Nút START
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 120, 550, 240, 60)
        pygame.draw.rect(self.screen, GREEN, btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, btn_rect, 2, border_radius=10)
        start_text = self.font_large.render("START GAME", True, WHITE)
        self.screen.blit(start_text, (btn_rect.centerx - start_text.get_width() // 2,
                                      btn_rect.centery - start_text.get_height() // 2))

        # Credits
        credit = self.font_small.render("Nhóm 1 - ĐH Phenikaa | MẠNG NƠRON VÀ HỌC SÂU", True, DARK_GRAY)
        self.screen.blit(credit, (WINDOW_WIDTH // 2 - credit.get_width() // 2, WINDOW_HEIGHT - 40))

        return btn_rect

    def get_game_button_rects(self):
        """Vị trí nút overlay giống ảnh minh họa: Menu trái, Submit giữa, Clear phải."""
        button_y = self.cam_y + CAMERA_HEIGHT - 50
        menu_btn = pygame.Rect(self.cam_x + 12, button_y, 96, 36)
        submit_btn = pygame.Rect(WINDOW_WIDTH // 2 - 58, button_y, 116, 36)
        clear_btn = pygame.Rect(self.cam_x + CAMERA_WIDTH - 108, button_y, 96, 36)
        return menu_btn, clear_btn, submit_btn

    def draw_game_ui(self, camera_surface):
        """Vẽ giao diện game camera theo style ảnh mẫu: HUD trên, camera toàn màn, nút overlay."""
        self.screen.fill((14, 20, 32))

        cam_x = self.cam_x
        cam_y = self.cam_y

        # Top HUD nền xanh đậm giống screenshot.
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, WINDOW_WIDTH, cam_y))
        pygame.draw.line(self.screen, (45, 72, 105), (0, cam_y - 1), (WINDOW_WIDTH, cam_y - 1), 2)

        word_text = self.font_large.render(f"Draw: {self.current_word}", True, MAGENTA)
        self.screen.blit(word_text, (18, 16))
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (18, 50))

        time_text = self.font_large.render(f"Time: {int(self.time_left)}", True, WHITE)
        self.screen.blit(time_text, (WINDOW_WIDTH // 2 - time_text.get_width() // 2, 14))
        level_text = self.font_medium.render(f"Level: {self.current_level}/{self.total_levels}", True, WHITE)
        self.screen.blit(level_text, (WINDOW_WIDTH // 2 - level_text.get_width() // 2, 50))

        lives_str = "♥" * self.lives + "♡" * (MAX_LIVES - self.lives)
        lives_text = self.font_medium.render(f"Lives: {lives_str}", True, WHITE)
        self.screen.blit(lives_text, (WINDOW_WIDTH - lives_text.get_width() - 22, 16))
        streak_text = self.font_medium.render(f"Streak: {self.streak}", True, WHITE)
        self.screen.blit(streak_text, (WINDOW_WIDTH - streak_text.get_width() - 22, 50))

        # Camera lớn ngay dưới HUD.
        if camera_surface.get_width() != CAMERA_WIDTH or camera_surface.get_height() != CAMERA_HEIGHT:
            camera_surface = pygame.transform.smoothscale(camera_surface, (CAMERA_WIDTH, CAMERA_HEIGHT))
        self.screen.blit(camera_surface, (cam_x, cam_y))
        pygame.draw.rect(self.screen, (5, 8, 13), (cam_x, cam_y, CAMERA_WIDTH, CAMERA_HEIGHT), 2)

        if self.realtime_prediction:
            pred_color = GREEN if self.realtime_prediction == self.current_word else WHITE
            pred_text = self.font_small.render(
                f"AI thinks: {self.realtime_prediction} ({self.realtime_confidence:.0%})",
                True, pred_color,
            )
            badge = pygame.Rect(cam_x + 14, cam_y + 12, pred_text.get_width() + 20, 30)
            pygame.draw.rect(self.screen, (12, 18, 28), badge, border_radius=14)
            self.screen.blit(pred_text, (badge.x + 10, badge.y + 6))

        menu_btn, clear_btn, submit_btn = self.get_game_button_rects()
        for rect, label in [(menu_btn, "Menu"), (submit_btn, "Submit"), (clear_btn, "Clear")]:
            shadow = rect.move(5, 5)
            pygame.draw.rect(self.screen, BTN_SHADOW, shadow, border_radius=10)
            pygame.draw.rect(self.screen, BTN_SKY if label != "Submit" else BLUE, rect, border_radius=10)
            text = self.font_small.render(label, True, WHITE)
            self.screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

        if self.feedback_text and time.time() - self.feedback_timer < 3:
            fb = self.font_medium.render(self.feedback_text, True, self.feedback_color)
            bg = pygame.Rect(WINDOW_WIDTH // 2 - fb.get_width() // 2 - 14, WINDOW_HEIGHT - 34, fb.get_width() + 28, 28)
            pygame.draw.rect(self.screen, (12, 18, 28), bg, border_radius=10)
            self.screen.blit(fb, (WINDOW_WIDTH // 2 - fb.get_width() // 2, WINDOW_HEIGHT - 30))

        return menu_btn, clear_btn, submit_btn

    def draw_vocab_info(self):
        """Hiển thị thông tin từ vựng sau khi dự đoán"""
        self.screen.fill(DARK_BG)

        word = self.current_word
        vocab = VOCAB_DATA.get(word, {})

        # Tiêu đề
        if self.prediction_correct:
            title = self.font_huge.render("CORRECT!", True, GREEN)
        else:
            title = self.font_huge.render("TIME'S UP!", True, RED)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))

        # Từ vựng
        word_text = self.font_title.render(word.upper(), True, CYAN)
        self.screen.blit(word_text, (WINDOW_WIDTH // 2 - word_text.get_width() // 2, 150))

        # Thông tin
        y = 240
        info_items = [
            ("Nghĩa:", vocab.get("vietnamese", "N/A")),
            ("IPA:", vocab.get("ipa", "N/A")),
            ("Ví dụ:", vocab.get("example", "N/A")),
            ("Dịch:", vocab.get("example_vi", "N/A")),
        ]

        for label, value in info_items:
            label_surf = self.font_medium.render(label, True, YELLOW)
            value_surf = self.font_medium.render(f" {value}", True, WHITE)
            total_w = label_surf.get_width() + value_surf.get_width()
            x = WINDOW_WIDTH // 2 - total_w // 2
            self.screen.blit(label_surf, (x, y))
            self.screen.blit(value_surf, (x + label_surf.get_width(), y))
            y += 45

        # Score info
        if self.prediction_correct:
            score_text = self.font_medium.render(f"Score: {self.score} | Streak: {self.streak}", True, GREEN)
        else:
            score_text = self.font_medium.render(f"Lives remaining: {self.lives}", True, RED)
        self.screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, y + 20))

        # Tiếp tục
        continue_text = self.font_small.render("Nhấn SPACE hoặc đợi 5 giây để tiếp tục...", True, LIGHT_GRAY)
        self.screen.blit(continue_text, (WINDOW_WIDTH // 2 - continue_text.get_width() // 2, WINDOW_HEIGHT - 60))

    def draw_game_over(self):
        """Màn hình kết thúc game"""
        self.screen.fill(DARK_BG)

        # Tiêu đề
        title = self.font_huge.render("GAME OVER", True, RED)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 100))

        # Điểm
        score_text = self.font_title.render(f"Final Score: {self.score}", True, YELLOW)
        self.screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, 220))

        level_text = self.font_large.render(
            f"Completed: {min(self.current_level - 1, self.total_levels)}/{self.total_levels} levels",
            True, LIGHT_GRAY
        )
        self.screen.blit(level_text, (WINDOW_WIDTH // 2 - level_text.get_width() // 2, 300))

        # Nút
        replay_btn = pygame.Rect(WINDOW_WIDTH // 2 - 130, 420, 260, 60)
        pygame.draw.rect(self.screen, GREEN, replay_btn, border_radius=10)
        replay_text = self.font_large.render("PLAY AGAIN", True, WHITE)
        self.screen.blit(replay_text, (replay_btn.centerx - replay_text.get_width() // 2,
                                       replay_btn.centery - replay_text.get_height() // 2))

        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 - 130, 500, 260, 60)
        pygame.draw.rect(self.screen, BLUE, menu_btn, border_radius=10)
        menu_text = self.font_large.render("MAIN MENU", True, WHITE)
        self.screen.blit(menu_text, (menu_btn.centerx - menu_text.get_width() // 2,
                                     menu_btn.centery - menu_text.get_height() // 2))

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
                if not ret:
                    frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                frame = cv2.flip(frame, 1)

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
                        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 120, 550, 240, 60)
                        if btn_rect.collidepoint(mouse_pos):
                            self.start_game()

                    elif self.state == "VOCAB_INFO":
                        self.current_level += 1
                        self.next_word()
                        self.state = "PLAYING"

                    elif self.state == "GAME_OVER":
                        replay_btn = pygame.Rect(WINDOW_WIDTH // 2 - 130, 420, 260, 60)
                        menu_btn = pygame.Rect(WINDOW_WIDTH // 2 - 130, 500, 260, 60)
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
                # Hand tracking
                self.hand_tracker.find_hands(frame)
                finger_pos = self.hand_tracker.get_index_finger_tip(frame)

                if self.hand_tracker.is_drawing_gesture() and finger_pos:
                    # Đang vẽ
                    cx, cy = finger_pos
                    self.drawing_canvas.draw_line(cx, cy)
                    # Vẽ dấu chấm tại vị trí ngón tay
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                elif self.hand_tracker.is_erase_gesture():
                    # Cử chỉ dừng vẽ
                    self.drawing_canvas.stop_drawing()
                else:
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

    game = AirDrawVocabGame()
    game.run()
