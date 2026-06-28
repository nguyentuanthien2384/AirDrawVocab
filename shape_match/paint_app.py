"""
paint_app.py — "Vẽ trên màn hình bằng tay" (AI Virtual Painter) — bám sát mô hình
cử chỉ kinh điển của các video dạng "Draw on Screen Using Hand Tracking" (OpenCV +
MediaPipe).

CỬ CHỈ (giống video):
  • CHỈ ngón trỏ giơ lên           -> chế độ VẼ (bút vẽ tại đầu ngón trỏ, landmark 8)
  • Ngón trỏ + ngón giữa cùng giơ  -> chế độ CHỌN (không vẽ; đưa lên thanh màu để chọn
                                      màu hoặc Tẩy/Xóa)
  • Thanh công cụ ở đỉnh màn hình   -> các ô màu + Tẩy + Xóa hết
  • Ảnh lật gương, nét làm mượt (One-Euro/Kalman), canvas giữ nguyên đến khi Xóa

Phím tắt: [C] xóa hết · [S] lưu PNG · [Q]/[Esc] thoát

Chạy:  python -m shape_match.paint_app        (đặt shape_match/ ở gốc project)
       python -m shape_match.paint_app --thickness 18 --eraser 60
Yêu cầu: opencv-python, mediapipe, numpy (đã có trong requirements của AirDrawVocab).
Nên dùng Python 3.11/3.12 để MediaPipe ổn định.
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise SystemExit("Cần OpenCV: pip install opencv-python  (" + str(e) + ")")

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False

try:
    from .kalman import FingertipSmoother
except ImportError:  # chạy trực tiếp
    from kalman import FingertipSmoother  # type: ignore


# Bảng màu thanh công cụ (BGR). Ô cuối là Tẩy, rồi nút Xóa.
PALETTE = [
    ("Tim",    (180, 60, 230)),
    ("Xanh",   (240, 160, 30)),
    ("La",     (60, 210, 90)),
    ("Vang",   (40, 210, 245)),
    ("Do",     (60, 60, 235)),
    ("Trang",  (240, 240, 240)),
]
ERASER_LABEL = "Tay"
CLEAR_LABEL = "Xoa"

HEADER_H = 86
TIP_IDS = [4, 8, 12, 16, 20]  # đầu các ngón: cái, trỏ, giữa, áp út, út


class HandModel:
    def __init__(self):
        self.available = _HAS_MP
        if not self.available:
            self.hands = None
            return
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.6,
        )
        self.results = None

    def process(self, frame_bgr):
        if not self.available:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)

    def landmarks_px(self, w: int, h: int) -> Optional[List[Tuple[int, int]]]:
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        lm = self.results.multi_hand_landmarks[0].landmark
        return [(int(p.x * w), int(p.y * h)) for p in lm]

    def handedness(self) -> str:
        try:
            return self.results.multi_handedness[0].classification[0].label  # 'Left'/'Right'
        except Exception:
            return "Right"

    def draw_skeleton(self, frame):
        if self.results and self.results.multi_hand_landmarks:
            for hl in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hl, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1),
                )


def fingers_up(lms: List[Tuple[int, int]], hand_label: str) -> List[int]:
    """Trả [thumb, index, middle, ring, pinky] với 1=giơ, 0=gập (chuẩn cvzone)."""
    f = [0, 0, 0, 0, 0]
    # Ngón cái: so theo trục x, có xét tay trái/phải (ảnh đã lật gương).
    if hand_label == "Right":
        f[0] = 1 if lms[TIP_IDS[0]][0] > lms[TIP_IDS[0] - 1][0] else 0
    else:
        f[0] = 1 if lms[TIP_IDS[0]][0] < lms[TIP_IDS[0] - 1][0] else 0
    # 4 ngón còn lại: đầu ngón cao hơn khớp PIP (y nhỏ hơn) = giơ.
    for i in range(1, 5):
        f[i] = 1 if lms[TIP_IDS[i]][1] < lms[TIP_IDS[i] - 2][1] else 0
    return f


class VirtualPainter:
    def __init__(self, cam=0, width=1280, height=720, thickness=15, eraser=55):
        self.cam, self.width, self.height = cam, width, height
        self.brush, self.eraser = thickness, eraser
        self.model = HandModel()
        self.smoother = FingertipSmoother()
        self.canvas = None
        self.xp = self.yp = 0
        self.color = PALETTE[0][1]
        self.color_name = PALETTE[0][0]
        self.is_eraser = False

    # ---- thanh công cụ ----
    def _slots(self):
        """Tính vùng (x0,x1) cho từng ô: các màu + Tẩy + Xóa."""
        items = [p[0] for p in PALETTE] + [ERASER_LABEL, CLEAR_LABEL]
        n = len(items)
        w = self.width // n
        slots = []
        for i, name in enumerate(items):
            slots.append((name, i * w, (i + 1) * w if i < n - 1 else self.width))
        return slots

    def _draw_header(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, HEADER_H), (28, 28, 34), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        for name, x0, x1 in self._slots():
            cx = (x0 + x1) // 2
            if name == ERASER_LABEL:
                cv2.rectangle(frame, (x0 + 14, 16), (x1 - 14, HEADER_H - 16), (60, 60, 60), 2)
                txt = "TAY"
            elif name == CLEAR_LABEL:
                cv2.rectangle(frame, (x0 + 14, 16), (x1 - 14, HEADER_H - 16), (40, 40, 200), 2)
                txt = "XOA"
            else:
                col = dict((p[0], p[1]) for p in PALETTE)[name]
                cv2.rectangle(frame, (x0 + 14, 16), (x1 - 14, HEADER_H - 16), col, -1)
                txt = ""
            # đánh dấu ô đang chọn
            active = (self.is_eraser and name == ERASER_LABEL) or \
                     (not self.is_eraser and name == self.color_name)
            if active:
                cv2.rectangle(frame, (x0 + 8, 10), (x1 - 8, HEADER_H - 10), (255, 255, 255), 3)
            if txt:
                cv2.putText(frame, txt, (cx - 26, HEADER_H // 2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def _pick(self, x):
        for name, x0, x1 in self._slots():
            if x0 <= x < x1:
                if name == ERASER_LABEL:
                    self.is_eraser = True
                elif name == CLEAR_LABEL:
                    self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                else:
                    self.is_eraser = False
                    self.color = dict((p[0], p[1]) for p in PALETTE)[name]
                    self.color_name = name
                return

    # ---- vòng lặp ----
    def run(self):
        if not self.model.available:
            raise SystemExit("Không import được mediapipe. Cài 'mediapipe' và dùng Python 3.11/3.12.")
        cap = cv2.VideoCapture(self.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise SystemExit("Không mở được camera.")
        win = "Virtual Painter - AirDrawVocab"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            self.height, self.width = frame.shape[:2]
            if self.canvas is None or self.canvas.shape[:2] != frame.shape[:2]:
                self.canvas = np.zeros_like(frame)

            self.model.process(frame)
            lms = self.model.landmarks_px(self.width, self.height)
            mode = "—"

            if lms:
                fu = fingers_up(lms, self.model.handedness())
                x_raw, y_raw = lms[8]                     # đầu ngón trỏ
                sx, sy = self.smoother.update(x_raw, y_raw)

                index_only = fu[1] == 1 and fu[2] == 0
                two_fingers = fu[1] == 1 and fu[2] == 1

                if two_fingers:                          # CHỌN
                    mode = "CHON"
                    self.xp = self.yp = 0
                    cv2.line(frame, (lms[8][0], lms[8][1]), (lms[12][0], lms[12][1]),
                             (200, 200, 200), 2)
                    cv2.circle(frame, (sx, sy), 14, self.color, cv2.FILLED)
                    if sy < HEADER_H:
                        self._pick(sx)
                elif index_only:                         # VẼ
                    mode = "TAY" if self.is_eraser else "VE"
                    cv2.circle(frame, (sx, sy), 12, (255, 255, 255), 2)
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = sx, sy
                    col = (0, 0, 0) if self.is_eraser else self.color
                    thick = self.eraser if self.is_eraser else self.brush
                    cv2.line(self.canvas, (self.xp, self.yp), (sx, sy), col, thick)
                    self.xp, self.yp = sx, sy
                else:
                    self.xp = self.yp = 0
            else:
                self.xp = self.yp = 0

            # Trộn canvas vào frame (kỹ thuật mặt nạ kinh điển)
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
            inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inv)
            frame = cv2.bitwise_or(frame, self.canvas)

            self._draw_header(frame)
            cv2.putText(frame, f"Che do: {mode}", (16, self.height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("c"):
                self.canvas = np.zeros_like(frame)
            elif k == ord("s"):
                cv2.imwrite(f"painting_{int(time.time())}.png", self.canvas)

        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="AI Virtual Painter — vẽ trên màn hình bằng tay.")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--thickness", type=int, default=15, help="độ dày nét vẽ")
    ap.add_argument("--eraser", type=int, default=55, help="độ dày tẩy")
    a = ap.parse_args()
    VirtualPainter(a.cam, a.width, a.height, a.thickness, a.eraser).run()


if __name__ == "__main__":
    main()
