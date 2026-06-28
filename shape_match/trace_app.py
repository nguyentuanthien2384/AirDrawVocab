"""
trace_app.py — Ứng dụng "VẼ TAY KHỚP HÌNH MẪU" chạy realtime bằng webcam.

Luồng (đúng kiến trúc trong tài liệu của bạn):
    MediaPipe Hands -> đầu ngón trỏ (landmark 8)
    -> Kalman Filter làm mượt nét
    -> Cử chỉ vẽ (ngón trỏ duỗi + ngón giữa gập) + Debounce chống rung lệnh
    -> Khi nhấc bút: ShapeMatcher chấm điểm độ khớp với HÌNH MẪU đang hiển thị

Phím tắt:
    [Space] hoặc nhấc tay : chấm điểm nét vừa vẽ
    [C]                    : xóa nét
    [N] / [P]             : đổi hình mẫu kế tiếp / trước
    [Q] hoặc [Esc]        : thoát

Chạy:  python -m shape_match.trace_app
(cần: opencv-python, mediapipe, numpy, scipy — đã có trong requirements của project)
"""
from __future__ import annotations

import sys
import time
from collections import deque, Counter
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("Cần OpenCV để chạy app camera:", e)
    sys.exit(1)

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False

# Hỗ trợ chạy cả khi import như package lẫn chạy trực tiếp trong thư mục.
try:
    from . import templates
    from .matcher import ShapeMatcher, MatchConfig
    from .kalman import FingertipSmoother
except ImportError:  # pragma: no cover
    import templates  # type: ignore
    from matcher import ShapeMatcher, MatchConfig  # type: ignore
    from kalman import FingertipSmoother  # type: ignore


# Màu (BGR)
CLR_TEMPLATE = (255, 196, 0)     # outline mẫu (xanh dương sáng)
CLR_DRAW = (0, 230, 80)          # nét người dùng
CLR_TIP = (0, 165, 255)          # đầu ngón
CLR_OK = (0, 220, 0)
CLR_BAD = (60, 60, 235)
CLR_TXT = (255, 255, 255)


class HandTracker:
    """Bọc MediaPipe Hands theo đúng cấu hình project (max_num_hands=1, landmark 8)."""

    def __init__(self):
        self.available = _HAS_MP
        if not self.available:
            self.hands = None
            return
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.results = None

    def process(self, frame_bgr) -> None:
        if not self.available:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)

    def index_tip(self, w: int, h: int) -> Optional[Tuple[int, int]]:
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        lm = self.results.multi_hand_landmarks[0].landmark[8]
        return int(lm.x * w), int(lm.y * h)

    def is_drawing(self) -> bool:
        """Cử chỉ vẽ: ngón trỏ duỗi (8 cao hơn 6), ngón giữa gập (12 thấp hơn 10)."""
        if not self.results or not self.results.multi_hand_landmarks:
            return False
        lm = self.results.multi_hand_landmarks[0].landmark
        index_up = lm[8].y < lm[6].y
        middle_down = lm[12].y > lm[10].y
        return bool(index_up and middle_down)


class GestureDebouncer:
    """Chống rung lệnh: chỉ đổi trạng thái khi đa số N frame gần đây đồng thuận."""

    def __init__(self, history: int = 5):
        self.buf: deque = deque(maxlen=history)
        self.stable = False

    def update(self, value: bool) -> bool:
        self.buf.append(bool(value))
        if len(self.buf) == self.buf.maxlen:
            self.stable = Counter(self.buf).most_common(1)[0][0]
        return self.stable


def _draw_polyline(img, pts, color, thickness=3, closed=False):
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _put(img, text, org, scale=0.7, color=CLR_TXT, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, val, label):
    cv2.rectangle(img, (x, y), (x + w, y + 14), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, val))), y + 14), (0, 200, 200), -1)
    _put(img, f"{label} {int(val * 100)}%", (x + w + 10, y + 13), 0.5, CLR_TXT, 1)


class TraceApp:
    def __init__(self, cam_index: int = 0, width: int = 1280, height: int = 720,
                 config: Optional[MatchConfig] = None):
        self.cam_index = cam_index
        self.width, self.height = width, height
        self.tracker = HandTracker()
        self.smoother = FingertipSmoother()
        self.draw_debounce = GestureDebouncer(history=5)
        self.matcher = ShapeMatcher(config or MatchConfig())

        self.template_names: List[str] = templates.names()
        self.idx = self.template_names.index("star") if "star" in self.template_names else 0

        self.stroke: List[Tuple[int, int]] = []
        self.was_drawing = False
        self.last_result = None
        self.result_until = 0.0

    # ----------------------------------------------------------------- #
    @property
    def current_name(self) -> str:
        return self.template_names[self.idx]

    def template_pixels(self) -> np.ndarray:
        return templates.get(self.current_name).pixels(self.width, self.height, margin=0.7)

    def next_template(self, step: int = 1):
        self.idx = (self.idx + step) % len(self.template_names)
        self.clear()

    def clear(self):
        self.stroke.clear()
        self.smoother = FingertipSmoother()
        self.last_result = None

    def score_now(self):
        if len(self.stroke) < 8:
            return
        self.last_result = self.matcher.score(np.asarray(self.stroke, dtype=np.float64),
                                              self.current_name)
        self.result_until = time.time() + 4.0

    # ----------------------------------------------------------------- #
    def _overlay(self, frame):
        tpl = templates.get(self.current_name)
        tpl_px = self.template_pixels()
        # outline mẫu (mờ) — vẽ dày để dễ bám theo
        _draw_polyline(frame, tpl_px, CLR_TEMPLATE, thickness=2, closed=tpl.closed)
        # điểm bắt đầu gợi ý
        cv2.circle(frame, tuple(tpl_px[0].astype(int)), 9, (0, 255, 255), -1)

        # nét người dùng
        _draw_polyline(frame, self.stroke, CLR_DRAW, thickness=4, closed=False)

        # tiêu đề
        _put(frame, f"Mau: {self.current_name.upper()} ({tpl.label_vi})", (20, 36), 0.8, CLR_TEMPLATE, 2)
        _put(frame, "Ngon tro de ve | Space/nhac tay: cham diem | C: xoa | N/P: doi hinh | Q: thoat",
             (20, self.height - 18), 0.5, CLR_TXT, 1)

        # kết quả
        r = self.last_result
        if r is not None and time.time() < self.result_until:
            col = CLR_OK if r.passed else CLR_BAD
            verdict = "DAT" if r.passed else "CHUA DAT"
            stars = "*" * r.stars + "-" * (3 - r.stars)
            _put(frame, f"{int(round(r.accuracy))}%  {verdict}  [{stars}]", (20, 78), 1.0, col, 3)
            _put(frame, r.message, (20, 112), 0.6, CLR_TXT, 2)
            x0, y0 = self.width - 320, 70
            _bar(frame, x0, y0, 160, r.shape_score, "Quy dao")
            _bar(frame, x0, y0 + 28, 160, r.coverage, "Day du ")
            _bar(frame, x0, y0 + 56, 160, r.precision, "Sach se")
            _bar(frame, x0, y0 + 84, 160, r.corner_score, "Goc/cong")
        return frame

    # ----------------------------------------------------------------- #
    def run(self):
        if not self.tracker.available:
            print("CẢNH BÁO: không import được mediapipe. Hãy cài 'mediapipe' "
                  "và dùng Python 3.11/3.12. Tạm thời chỉ hiển thị hình mẫu.")
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            print("Không mở được camera. Kiểm tra quyền camera / chỉ số cam_index.")
            return

        win = "Shape Trace Pro - AirDrawVocab"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            self.height, self.width = frame.shape[:2]

            drawing_stable = False
            if self.tracker.available:
                self.tracker.process(frame)
                tip = self.tracker.index_tip(self.width, self.height)
                drawing_stable = self.draw_debounce.update(self.tracker.is_drawing())

                if tip is not None:
                    sx, sy = self.smoother.update(*tip)
                    cv2.circle(frame, (sx, sy), 8, CLR_TIP, -1)
                    if drawing_stable:
                        self.stroke.append((sx, sy))
                else:
                    # mất tay: thử bù 1 frame để nét không đứt gãy
                    pred = self.smoother.predict_only()
                    if drawing_stable and pred is not None and self.stroke:
                        self.stroke.append(pred)

                # nhấc bút (đang vẽ -> ngừng) => tự chấm điểm
                if self.was_drawing and not drawing_stable and len(self.stroke) >= 8:
                    self.score_now()
                self.was_drawing = drawing_stable

            frame = self._overlay(frame)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                self.clear()
            elif key == ord(" "):
                self.score_now()
            elif key == ord("n"):
                self.next_template(+1)
            elif key == ord("p"):
                self.next_template(-1)

        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Vẽ tay khớp hình mẫu (shape tracing).")
    ap.add_argument("--cam", type=int, default=0, help="chỉ số camera")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--threshold", type=float, default=None, help="ngưỡng đạt (0-100)")
    args = ap.parse_args()

    cfg = MatchConfig()
    if args.threshold is not None:
        cfg.pass_threshold = args.threshold
    TraceApp(args.cam, args.width, args.height, cfg).run()


if __name__ == "__main__":
    main()
