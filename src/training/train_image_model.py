"""Canonical image training pipeline for AirDrawVocab.

This replaces ad-hoc image retraining with a release-candidate pipeline.
It can train:
- resnet_sketch: scratch CNN for sketches.
- mobilenetv2: lightweight transfer-learning candidate at 96x96x3.

The script works with QuickDraw .npy and user-collected strokes from SQLite.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import CATEGORIES, DATA_DIR, MODELS_DIR
from src.utils.model_versioning import save_versioned_model

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
STATUS_PATH = Path(os.getenv("AIRDRAW_RETRAIN_STATUS_PATH", str(ROOT / "data" / "self_improving_loop" / "status" / "retrain_status.json")))
REPORTS_DIR = ROOT / "assets" / "reports" / "image_training"
CANVAS_W = 960
CANVAS_H = 540


def write_status(status: str, message: str, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tiny_yaml(path: Path | None) -> dict:
    """Tiny YAML reader for the simple configs in this project.

    Uses PyYAML if available; otherwise supports key/value indentation enough
    for configs/*.yaml shipped here.
    """
    if not path or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        pass
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value.strip('"\'')
            parent[key] = parsed
    return root


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def strokes_to_bitmap(strokes: Any, size: int = 64) -> np.ndarray:
    strokes = safe_json_loads(strokes, [])
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
        return np.zeros((size, size), dtype="float32")
    x, y, w, h = cv2.boundingRect(coords)
    crop = arr[y:y+h, x:x+w]
    target_box = int(size * 0.78)
    scale = target_box / max(w, h, 1)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype="uint8")
    xs, ys = (size - nw) // 2, (size - nh) // 2
    canvas[ys:ys+nh, xs:xs+nw] = resized
    return canvas.astype("float32") / 255.0


def resize_sketch(img: np.ndarray, size: int, channels: int) -> np.ndarray:
    arr = np.asarray(img, dtype="float32")
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    if channels == 3:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    else:
        arr = arr[..., None]
    return arr.astype("float32")


def load_user_images(db_path: Path, size: int, channels: int) -> Tuple[List[np.ndarray], List[str]]:
    xs, ys = [], []
    if not db_path.exists():
        return xs, ys
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT target, strokes_json FROM stroke_samples ORDER BY id ASC").fetchall()
    for row in rows:
        label = str(row["target"] or "").strip().lower()
        if not label:
            continue
        bitmap = strokes_to_bitmap(row["strokes_json"] or "[]", size=size)
        if bitmap.max() > 0:
            xs.append(resize_sketch(bitmap, size, channels))
            ys.append(label)
    return xs, ys


def load_quickdraw_images(labels: List[str], limit_per_class: int, size: int, channels: int) -> Tuple[List[np.ndarray], List[str]]:
    xs, ys = [], []
    for label in labels:
        path = DATA_DIR / f"{label}.npy"
        if not path.exists():
            path = DATA_DIR / f"{label.replace(' ', '_')}.npy"
        if not path.exists():
            continue
        arr = np.load(path)[:limit_per_class].reshape((-1, 28, 28)).astype("float32") / 255.0
        for img in arr:
            xs.append(resize_sketch(img, size, channels))
            ys.append(label)
    return xs, ys


def augment_layers(input_size: int) -> keras.Sequential:
    return keras.Sequential([
        layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
        layers.RandomRotation(0.06, fill_mode="constant"),
        layers.RandomZoom(0.10, fill_mode="constant"),
    ], name="sketch_augmentation")


def build_resnet_sketch(num_classes: int, input_size: int, channels: int, lr: float) -> keras.Model:
    inputs = keras.Input(shape=(input_size, input_size, channels))
    x = augment_layers(input_size)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="swish")(x)
    for filters in [32, 64, 96]:
        shortcut = layers.Conv2D(filters, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="swish")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="swish")(x)
        x = layers.Add()([x, shortcut])
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="AirDrawVocabResNetSketch")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.04), metrics=["accuracy"])
    return model


def build_mobilenetv2(num_classes: int, input_size: int, lr: float, alpha: float = 0.35) -> keras.Model:
    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = augment_layers(input_size)(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    base = keras.applications.MobileNetV2(input_shape=(input_size, input_size, 3), include_top=False, weights="imagenet", alpha=alpha)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="AirDrawVocabMobileNetV2")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.04), metrics=["accuracy"])
    return model


def make_class_weights(y_int: np.ndarray, num_classes: int) -> dict:
    counts = np.bincount(y_int, minlength=num_classes).astype("float64")
    counts[counts == 0] = 1.0
    total = counts.sum()
    return {i: float(total / (num_classes * counts[i])) for i in range(num_classes)}


def evaluate(model: keras.Model, x_val: np.ndarray, y_val_int: np.ndarray, categories: list[str], out_dir: Path) -> dict:
    probs = model.predict(x_val, verbose=0)
    pred = probs.argmax(axis=1)
    acc = float((pred == y_val_int).mean())
    top3 = float(np.mean([t in row for t, row in zip(y_val_int, np.argsort(probs, axis=1)[:, -min(3, probs.shape[1]):])]))
    summary = {"val_accuracy": acc, "val_top3_accuracy": top3, "val_samples": int(len(y_val_int))}
    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = list(range(len(categories)))
        report = classification_report(y_val_int, pred, labels=labels, target_names=categories, zero_division=0, output_dict=True)
        macro_f1 = float(f1_score(y_val_int, pred, labels=labels, average="macro", zero_division=0))
        summary["val_macro_f1"] = macro_f1
        (out_dir / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        cm = confusion_matrix(y_val_int, pred, labels=labels)
        np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    except Exception as exc:
        summary["report_warning"] = str(exc)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "image_resnet_sketch.yaml")
    parser.add_argument("--model", default=None, choices=[None, "resnet_sketch", "mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--quickdraw-limit", type=int, default=None)
    parser.add_argument("--only-user-data", action="store_true")
    args = parser.parse_args()

    cfg = read_tiny_yaml(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    model_name = args.model or str(model_cfg.get("name", "resnet_sketch"))
    input_size = int(model_cfg.get("input_size", 96 if model_name == "mobilenetv2" else 64))
    channels = 3 if model_name == "mobilenetv2" else 1
    epochs = int(args.epochs or train_cfg.get("epochs", 15))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("learning_rate", 0.001))
    quickdraw_limit = int(args.quickdraw_limit or train_cfg.get("quickdraw_limit_per_class", 800))
    use_quickdraw = bool(train_cfg.get("use_quickdraw", True)) and not args.only_user_data

    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    labels = [str(x).strip().lower() for x in CATEGORIES]
    write_status("running", "Loading canonical image dataset", mode="image", model=model_name)
    ux, uy = load_user_images(DB_PATH, input_size, channels)
    qx, qy = ([], []) if not use_quickdraw else load_quickdraw_images(labels, quickdraw_limit, input_size, channels)
    all_x, all_y = qx + ux, qy + uy
    if not all_x:
        raise RuntimeError("No image training data. Add QuickDraw .npy files or save user strokes.")
    categories = sorted(set(all_y))
    label_to_idx = {label: i for i, label in enumerate(categories)}
    x = np.stack(all_x).astype("float32")
    y_int = np.asarray([label_to_idx[v] for v in all_y], dtype="int64")
    y = keras.utils.to_categorical(y_int, num_classes=len(categories))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    x, y, y_int = x[idx], y[idx], y_int[idx]
    val_size = max(1, int(len(x) * float(train_cfg.get("validation_ratio", 0.15)))) if len(x) >= 20 else 0
    if val_size:
        x_train, y_train, y_train_int = x[:-val_size], y[:-val_size], y_int[:-val_size]
        x_val, y_val, y_val_int = x[-val_size:], y[-val_size:], y_int[-val_size:]
    else:
        x_train, y_train, y_train_int = x, y, y_int
        x_val = y_val = y_val_int = None

    write_status("running", "Training image candidate", mode="image", model=model_name, samples=len(x), classes=len(categories), user_samples=len(ux), quickdraw_samples=len(qx))
    if model_name == "mobilenetv2":
        model = build_mobilenetv2(len(categories), input_size, lr, alpha=float(model_cfg.get("alpha", 0.35)))
    else:
        model = build_resnet_sketch(len(categories), input_size, channels, lr)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy" if val_size else "accuracy", mode="max", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if val_size else "loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    model.fit(x_train, y_train, validation_data=(x_val, y_val) if val_size else None, epochs=epochs, batch_size=batch_size, class_weight=make_class_weights(y_train_int, len(categories)), callbacks=callbacks, verbose=2)

    metrics = {"train_samples": int(len(x_train)), "classes": int(len(categories)), "user_samples": int(len(ux)), "quickdraw_samples": int(len(qx))}
    report_dir = REPORTS_DIR / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if val_size:
        metrics.update(evaluate(model, x_val, y_val_int, categories, report_dir))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = MODELS_DIR / "image_cnn_candidate.keras"
    candidate_categories_path = MODELS_DIR / "categories_candidate.json"
    deploy_path = MODELS_DIR / "airdrawvocab_self_improved.keras"
    deploy_categories_path = MODELS_DIR / "categories_self_improved.json"
    model.save(candidate_path)
    model.save(deploy_path)
    candidate_categories_path.write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
    deploy_categories_path.write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        save_versioned_model(model, base_name=f"image_{model_name}", metrics=metrics, params={"epochs": epochs, "input_size": input_size, "channels": channels}, extra={"categories": categories})
    except Exception as exc:
        metrics["versioning_warning"] = str(exc)
    summary = {"model": model_name, "candidate_model": str(candidate_path), "deploy_model": str(deploy_path), "categories": categories, "metrics": metrics, "class_counts": dict(Counter(all_y))}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status("done", "Image candidate trained and saved", mode="image", model=model_name, samples=len(x), classes=len(categories), candidate=str(candidate_path), report=str(report_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_status("failed", str(exc), mode="image")
        raise
