"""Self-improving image-CNN retrain pipeline for AirDrawVocab.

This script reads user-collected stroke samples from SQLite, rasterizes them
into 28x28 QuickDraw-like images, mixes them with existing QuickDraw .npy data
when available, and exports a new Keras image classifier.

Usage:
  python self_improve_retrain.py --epochs 10
  python self_improve_retrain.py --only-user-data --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw

# parents[2] vì file đã chuyển vào src/training/ (giữ ROOT trỏ về gốc dự án)
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
NPY_DIR = ROOT / "data" / "npy_28"
MODELS_DIR = ROOT / "models"
STATUS_PATH = Path(os.getenv("AIRDRAW_RETRAIN_STATUS_PATH", str(ROOT / "data" / "self_improving_loop" / "status" / "retrain_status.json")))
CANVAS_W = 960
CANVAS_H = 540

# bootstrap: cho phép import vocab_pairs ở thư mục gốc
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

# Backend chạy script này dạng subprocess -> ép UTF-8 để không crash 'charmap'
# khi in tiếng Việt trên Windows.
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from vocab_pairs import CATEGORIES as VOCAB_CATEGORIES
except Exception:
    VOCAB_CATEGORIES = []


def write_status(status: str, message: str, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({"status": status, "message": message, **extra}, ensure_ascii=False, indent=2), encoding="utf-8")


def strokes_to_bitmap(strokes: Any) -> np.ndarray:
    if isinstance(strokes, str):
        try:
            strokes = json.loads(strokes)
        except Exception:
            strokes = []
    img = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    draw = ImageDraw.Draw(img)
    if isinstance(strokes, list):
        for stroke in strokes:
            if not isinstance(stroke, list) or len(stroke) < 2:
                continue
            pts = []
            for p in stroke:
                if isinstance(p, dict):
                    pts.append((float(p.get("x", 0)), float(p.get("y", 0))))
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=18, joint="curve")
    arr = np.asarray(img)
    coords = cv2.findNonZero((arr > 10).astype("uint8") * 255)
    if coords is None:
        return np.zeros((28, 28), dtype="float32")
    x, y, w, h = cv2.boundingRect(coords)
    crop = arr[y:y+h, x:x+w]
    scale = 20 / max(w, h, 1)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype="uint8")
    xs, ys = (28-nw)//2, (28-nh)//2
    canvas[ys:ys+nh, xs:xs+nw] = resized
    return canvas.astype("float32") / 255.0


def load_user_images(db_path: Path) -> Tuple[List[np.ndarray], List[str]]:
    xs, ys = [], []
    if not db_path.exists():
        return xs, ys
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute("SELECT target, strokes_json FROM stroke_samples ORDER BY id ASC"):
            target = str(r["target"] or "").strip().lower()
            if not target:
                continue
            bitmap = strokes_to_bitmap(r["strokes_json"] or "[]")
            if bitmap.max() > 0:
                xs.append(bitmap)
                ys.append(target)
    return xs, ys


def load_quickdraw_images(labels: List[str], limit_per_class: int) -> Tuple[List[np.ndarray], List[str]]:
    xs, ys = [], []
    for label in labels:
        path = NPY_DIR / f"{label}.npy"
        if not path.exists():
            path = NPY_DIR / f"{label.replace(' ', '_')}.npy"
        if not path.exists():
            continue
        arr = np.load(path)
        arr = arr[:limit_per_class].reshape((-1, 28, 28)).astype("float32") / 255.0
        xs.extend(list(arr))
        ys.extend([label] * len(arr))
    return xs, ys


def build_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(96, 3, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quickdraw-limit", type=int, default=800)
    parser.add_argument("--only-user-data", action="store_true")
    args = parser.parse_args()

    labels = [str(x).strip().lower() for x in VOCAB_CATEGORIES] or []
    write_status("running", "Loading user-collected strokes", mode="image")
    ux, uy = load_user_images(DB_PATH)
    qx, qy = ([], []) if args.only_user_data else load_quickdraw_images(labels, args.quickdraw_limit)
    all_x = qx + ux
    all_y = qy + uy
    if not all_x:
        raise RuntimeError("No training data found. Add QuickDraw .npy files or collect user strokes first.")
    categories = sorted(set(all_y))
    if len(categories) < 2:
        raise RuntimeError("Need at least 2 classes to train.")
    label_to_idx = {label: i for i, label in enumerate(categories)}
    x = np.stack(all_x).reshape((-1, 28, 28, 1)).astype("float32")
    y = keras.utils.to_categorical([label_to_idx[v] for v in all_y], num_classes=len(categories))

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    val_size = max(1, int(len(x) * 0.12)) if len(x) >= 20 else 0
    validation_data = None
    if val_size:
        x_train, y_train = x[:-val_size], y[:-val_size]
        validation_data = (x[-val_size:], y[-val_size:])
    else:
        x_train, y_train = x, y

    write_status("running", "Training self-improved image CNN", mode="image", samples=len(x), classes=len(categories), user_samples=len(ux), quickdraw_samples=len(qx))
    model = build_model(len(categories))
    callbacks = [keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy" if validation_data else "accuracy")]
    model.fit(x_train, y_train, validation_data=validation_data, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks, verbose=2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "airdrawvocab_self_improved.keras")
    (MODELS_DIR / "categories_self_improved.json").write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status("done", "Self-improved image model trained", mode="image", samples=len(x), classes=len(categories), model=str(MODELS_DIR / "airdrawvocab_self_improved.keras"))
    print("Saved:", MODELS_DIR / "airdrawvocab_self_improved.keras")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("failed", str(exc), mode="image")
        raise
