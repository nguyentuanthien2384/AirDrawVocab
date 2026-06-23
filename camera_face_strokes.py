"""
camera_face_strokes.py - Face-to-stroke helper for AirDrawVocab camera mode.

Adapted from the supplied DeepShieldAI-Pro face detector pattern: OpenCV Haar
face detection, crop-with-padding, and in-memory preprocessing. It receives one
webcam frame, finds a face, and converts it into clean face-sketch strokes for
AirDrawVocab. Frames are not saved to disk.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

Point = Dict[str, float | str]
Stroke = List[Point]
Box = Tuple[int, int, int, int]


def _load_face_detector() -> Optional[cv2.CascadeClassifier]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    return detector if not detector.empty() else None


FACE_DETECTOR = _load_face_detector()


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Empty image bytes.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not decode camera frame.")
    return frame


def detect_largest_face(frame: np.ndarray) -> Optional[Box]:
    if frame is None or frame.size == 0 or FACE_DETECTOR is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(width), int(height)


def crop_face_or_frame(frame: np.ndarray, padding_ratio: float = 0.22) -> Tuple[np.ndarray, Dict[str, Any]]:
    box = detect_largest_face(frame)
    if box is None:
        return frame, {"faceDetected": False, "bbox": None, "faceBbox": None}

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
        return frame, {"faceDetected": False, "bbox": None, "faceBbox": None}

    return crop, {
        "faceDetected": True,
        "bbox": {"x": left, "y": top, "width": right - left, "height": bottom - top},
        "faceBbox": {"x": x, "y": y, "width": width, "height": height},
        "cropSize": {"width": right - left, "height": bottom - top},
        "frameSize": {"width": max_x, "height": max_y},
    }


def _map_frame_point_to_canvas(
    frame_x: float,
    frame_y: float,
    frame_w: int,
    frame_h: int,
    canvas_w: int,
    canvas_h: int,
    mirror: bool,
    t: float,
    source: str = "face-backend",
) -> Point:
    x = (frame_x / max(frame_w, 1)) * canvas_w
    if mirror:
        x = canvas_w - x
    y = (frame_y / max(frame_h, 1)) * canvas_h
    return {
        "x": float(max(0.0, min(float(canvas_w), x))),
        "y": float(max(0.0, min(float(canvas_h), y))),
        "t": float(t),
        "source": source,
    }


def _map_box_to_canvas(
    bbox: Dict[str, int],
    frame_w: int,
    frame_h: int,
    canvas_w: int,
    canvas_h: int,
    mirror: bool,
) -> Dict[str, float]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    width = float(bbox["width"])
    height = float(bbox["height"])
    if mirror:
        cx = canvas_w - ((x + width) / max(frame_w, 1)) * canvas_w
    else:
        cx = (x / max(frame_w, 1)) * canvas_w
    return {
        "x": cx,
        "y": (y / max(frame_h, 1)) * canvas_h,
        "width": (width / max(frame_w, 1)) * canvas_w,
        "height": (height / max(frame_h, 1)) * canvas_h,
    }


def _ellipse_stroke(cx: float, cy: float, rx: float, ry: float, start: float, end: float, steps: int, t0: float, source: str) -> Stroke:
    return [
        {"x": float(cx + np.cos(angle) * rx), "y": float(cy + np.sin(angle) * ry), "t": t0 + i * 8.0, "source": source}
        for i, angle in enumerate(np.linspace(start, end, steps))
    ]


def _curve_stroke(points: List[Tuple[float, float]], t0: float, source: str) -> Stroke:
    return [{"x": float(x), "y": float(y), "t": t0 + i * 8.0, "source": source} for i, (x, y) in enumerate(points)]


def _semantic_face_strokes(face_canvas_box: Dict[str, float], canvas_w: int, canvas_h: int, source: str = "face-template") -> List[Stroke]:
    """Generate stable face sketch strokes from the detected face bounding box."""
    x = max(0.0, min(float(canvas_w), float(face_canvas_box["x"])))
    y = max(0.0, min(float(canvas_h), float(face_canvas_box["y"])))
    w = max(24.0, min(float(canvas_w), float(face_canvas_box["width"])))
    h = max(24.0, min(float(canvas_h), float(face_canvas_box["height"])))
    cx = x + w * 0.5
    cy = y + h * 0.53
    t = 0.0
    strokes: List[Stroke] = []
    strokes.append(_ellipse_stroke(cx, cy, w * 0.36, h * 0.43, 0.0, np.pi * 2, 44, t, source)); t += 400
    strokes.append(_curve_stroke([(cx - w * 0.27, cy - h * 0.16), (cx - w * 0.19, cy - h * 0.21), (cx - w * 0.10, cy - h * 0.18)], t, source)); t += 100
    strokes.append(_curve_stroke([(cx + w * 0.10, cy - h * 0.18), (cx + w * 0.19, cy - h * 0.21), (cx + w * 0.27, cy - h * 0.16)], t, source)); t += 100
    strokes.append(_ellipse_stroke(cx - w * 0.18, cy - h * 0.08, w * 0.09, h * 0.045, 0.0, np.pi * 2, 20, t, source)); t += 180
    strokes.append(_ellipse_stroke(cx + w * 0.18, cy - h * 0.08, w * 0.09, h * 0.045, 0.0, np.pi * 2, 20, t, source)); t += 180
    strokes.append(_ellipse_stroke(cx - w * 0.18, cy - h * 0.08, w * 0.025, h * 0.025, 0.0, np.pi * 2, 12, t, source)); t += 120
    strokes.append(_ellipse_stroke(cx + w * 0.18, cy - h * 0.08, w * 0.025, h * 0.025, 0.0, np.pi * 2, 12, t, source)); t += 120
    strokes.append(_curve_stroke([(cx, cy - h * 0.11), (cx - w * 0.03, cy + h * 0.04), (cx, cy + h * 0.12), (cx + w * 0.05, cy + h * 0.10)], t, source)); t += 140
    strokes.append(_curve_stroke([(cx - w * 0.09, cy + h * 0.14), (cx, cy + h * 0.17), (cx + w * 0.09, cy + h * 0.14)], t, source)); t += 120
    strokes.append(_ellipse_stroke(cx, cy + h * 0.27, w * 0.18, h * 0.055, 0.0, np.pi, 22, t, source)); t += 180
    strokes.append(_ellipse_stroke(cx, cy + h * 0.255, w * 0.16, h * 0.05, 0.0, np.pi, 18, t, source))
    for stroke in strokes:
        for p in stroke:
            p["x"] = float(max(0.0, min(float(canvas_w), float(p["x"]))))
            p["y"] = float(max(0.0, min(float(canvas_h), float(p["y"]))))
    return strokes


def _edge_strokes_from_crop(frame: np.ndarray, bbox: Dict[str, int], canvas_w: int, canvas_h: int, mirror: bool, max_strokes: int = 44, max_total_points: int = 420) -> List[Stroke]:
    """Convert high-contrast face edges into optional sketch strokes."""
    frame_h, frame_w = frame.shape[:2]
    left = int(bbox["x"]); top = int(bbox["y"]); width = int(bbox["width"]); height = int(bbox["height"])
    crop = frame[top : top + height, left : left + width]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    median = float(np.median(gray))
    lower = int(max(30, 0.66 * median))
    upper = int(min(180, 1.33 * median + 35))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, closed=False), reverse=True)
    strokes: List[Stroke] = []
    total_points = 0
    t = 2000.0
    for contour in contours:
        arc = float(cv2.arcLength(contour, closed=False))
        if arc < 20:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=max(1.0, arc * 0.012), closed=False)
        if len(approx) < 3:
            continue
        raw_points = approx.reshape(-1, 2)
        if len(raw_points) > 36:
            idx = np.linspace(0, len(raw_points) - 1, 36).astype(int)
            raw_points = raw_points[idx]
        stroke: Stroke = []
        for i, (px, py) in enumerate(raw_points):
            stroke.append(_map_frame_point_to_canvas(left + float(px), top + float(py), frame_w, frame_h, canvas_w, canvas_h, mirror, t + i * 6.0))
        if len(stroke) >= 3:
            strokes.append(stroke)
            total_points += len(stroke)
        t += 260.0
        if len(strokes) >= max_strokes or total_points >= max_total_points:
            break
    return strokes


def _make_stroke_preview(strokes: List[Stroke], canvas_w: int, canvas_h: int) -> str:
    scale = min(1.0, 640.0 / max(canvas_w, 1))
    out_w = max(1, int(canvas_w * scale)); out_h = max(1, int(canvas_h * scale))
    image = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        pts = [(float(p.get("x", 0.0)) * scale, float(p.get("y", 0.0)) * scale) for p in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=(85, 230, 165, 220), width=max(2, int(3 * scale)), joint="curve")
    buffer = BytesIO(); image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def analyze_face_frame(frame: np.ndarray, canvas_width: int = 960, canvas_height: int = 540, mirror: bool = True, include_preview: bool = False) -> Dict[str, Any]:
    canvas_w = int(max(64, min(4096, canvas_width)))
    canvas_h = int(max(64, min(4096, canvas_height)))
    frame_h, frame_w = frame.shape[:2]
    _crop, meta = crop_face_or_frame(frame)
    if not meta.get("faceDetected"):
        return {
            "ok": True,
            "face_detected": False,
            "detector": "opencv-haar",
            "message": "Không thấy khuôn mặt rõ. Hãy nhìn thẳng camera, đủ sáng và tránh che mặt.",
            "frame_size": {"width": frame_w, "height": frame_h},
            "canvas_size": {"width": canvas_w, "height": canvas_h},
            "bbox": None,
            "face_bbox": None,
            "strokes": [],
            "semantic_strokes": [],
            "edge_strokes": [],
            "stroke_count": 0,
            "point_count": 0,
            "quality": 0,
        }
    crop_bbox = meta["bbox"]
    face_bbox = meta["faceBbox"]
    canvas_face_bbox = _map_box_to_canvas(face_bbox, frame_w, frame_h, canvas_w, canvas_h, mirror)
    canvas_crop_bbox = _map_box_to_canvas(crop_bbox, frame_w, frame_h, canvas_w, canvas_h, mirror)
    semantic = _semantic_face_strokes(canvas_face_bbox, canvas_w, canvas_h)
    edges = _edge_strokes_from_crop(frame, crop_bbox, canvas_w, canvas_h, mirror=mirror)
    strokes = semantic + edges
    point_count = sum(len(stroke) for stroke in strokes)
    quality = int(max(20, min(100, (face_bbox["width"] * face_bbox["height"] / max(frame_w * frame_h, 1)) * 950 + min(point_count, 450) * 0.08)))
    result: Dict[str, Any] = {
        "ok": True,
        "face_detected": True,
        "detector": "opencv-haar+template-strokes",
        "message": "Đã nhận diện khuôn mặt và chuyển thành nét vẽ.",
        "frame_size": {"width": frame_w, "height": frame_h},
        "canvas_size": {"width": canvas_w, "height": canvas_h},
        "bbox": canvas_crop_bbox,
        "face_bbox": canvas_face_bbox,
        "strokes": strokes,
        "semantic_strokes": semantic,
        "edge_strokes": edges,
        "stroke_count": len(strokes),
        "semantic_count": len(semantic),
        "edge_count": len(edges),
        "point_count": point_count,
        "quality": quality,
        "privacy": "frame processed in memory only; no image saved",
    }
    if include_preview:
        result["sketch_preview"] = _make_stroke_preview(strokes, canvas_w, canvas_h)
    return result


def analyze_face_frame_bytes(image_bytes: bytes, canvas_width: int = 960, canvas_height: int = 540, mirror: bool = True, include_preview: bool = False) -> Dict[str, Any]:
    frame = decode_image_bytes(image_bytes)
    return analyze_face_frame(frame, canvas_width=canvas_width, canvas_height=canvas_height, mirror=mirror, include_preview=include_preview)
