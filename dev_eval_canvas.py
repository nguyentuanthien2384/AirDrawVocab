"""
Đánh giá nhanh pipeline nhận diện trên các nét vẽ mô phỏng kiểu canvas
(nền trắng, nét đen) giống người dùng thật, KHÔNG cần dataset .npy.

Mỗi nhãn được vẽ bằng PIL theo gợi ý trong frontend (GUIDE_DRAWERS),
sau đó đẩy qua hàm tiền xử lý + model để xem dự đoán/độ tin cậy.

Chạy:
    .\.venv311\Scripts\python.exe dev_eval_canvas.py
"""
from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from backend.app import categories, model, preprocess_image

W = 280  # canvas vẽ giống frontend


def _canvas():
    img = Image.new("RGB", (W, W), (255, 255, 255))
    return img, ImageDraw.Draw(img)


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def draw_square(d):
    d.rectangle((78, 78, 202, 202), outline=(0, 0, 0), width=8)


def draw_diamond(d):
    d.polygon([(140, 40), (220, 140), (140, 240), (60, 140)], outline=(0, 0, 0), width=8)


def draw_star(d):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = 95 if i % 2 == 0 else 40
        pts.append((140 + r * math.cos(ang), 145 + r * math.sin(ang)))
    d.line(pts + [pts[0]], fill=(0, 0, 0), width=8, joint="curve")


def draw_circle(d):  # apple-ish
    d.ellipse((80, 80, 200, 210), outline=(0, 0, 0), width=8)
    d.line((140, 80, 140, 55), fill=(0, 0, 0), width=8)


def draw_book(d):
    d.rectangle((60, 80, 220, 200), outline=(0, 0, 0), width=8)
    d.line((140, 80, 140, 200), fill=(0, 0, 0), width=8)


def draw_envelope(d):
    d.rectangle((50, 95, 230, 190), outline=(0, 0, 0), width=8)
    d.line((50, 95, 140, 155), fill=(0, 0, 0), width=8)
    d.line((230, 95, 140, 155), fill=(0, 0, 0), width=8)


def draw_lightning(d):
    d.line([(168, 42), (95, 140), (140, 140), (112, 238), (200, 118), (154, 118)],
           fill=(0, 0, 0), width=8, joint="curve")


def draw_door(d):
    d.rectangle((90, 50, 190, 235), outline=(0, 0, 0), width=8)
    d.ellipse((165, 135, 178, 148), fill=(0, 0, 0))


def draw_eye(d):
    d.ellipse((45, 105, 235, 195), outline=(0, 0, 0), width=8)
    d.ellipse((112, 110, 168, 188), outline=(0, 0, 0), width=8)
    d.ellipse((130, 135, 150, 162), fill=(0, 0, 0))


def draw_moon(d):
    d.arc((70, 50, 210, 240), start=40, end=320, fill=(0, 0, 0), width=8)
    d.arc((110, 50, 250, 240), start=70, end=290, fill=(0, 0, 0), width=8)


SAMPLES = {
    "square": draw_square,
    "diamond": draw_diamond,
    "star": draw_star,
    "apple": draw_circle,
    "book": draw_book,
    "envelope": draw_envelope,
    "lightning": draw_lightning,
    "door": draw_door,
    "eye": draw_eye,
    "moon": draw_moon,
}


def predict_bytes(image_bytes: bytes):
    x = preprocess_image(image_bytes)
    preds = model.predict(x, verbose=0)[0]
    order = preds.argsort()[::-1]
    top3 = [(categories[i], float(preds[i])) for i in order[:3]]
    return top3


def main():
    print(f"Model output classes: {len(categories)}")
    correct = 0
    total = 0
    for label, drawer in SAMPLES.items():
        img, d = _canvas()
        drawer(d)
        top3 = predict_bytes(_png_bytes(img))
        pred = top3[0][0]
        ok = pred == label
        in_top3 = label in [t[0] for t in top3]
        correct += int(ok)
        total += 1
        flag = "OK " if ok else ("~T3" if in_top3 else "XX ")
        top3_str = ", ".join(f"{l}:{c*100:.0f}%" for l, c in top3)
        print(f"[{flag}] target={label:10s} -> {pred:10s} ({top3[0][1]*100:5.1f}%) | top3: {top3_str}")
    print(f"\nTop-1 đúng: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()
