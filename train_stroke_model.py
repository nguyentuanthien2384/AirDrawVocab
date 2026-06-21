"""Train stroke-sequence model from AirDrawVocab SQLite samples.

This is the stroke-based deep learning branch for Final Boss Mode.
It reads data/airdrawvocab_app.sqlite3 -> stroke_samples, converts each
stroke sequence to a fixed-length temporal tensor, and trains a small GRU model.

Usage:
  python train_stroke_model.py --epochs 12
  python train_stroke_model.py --db data/airdrawvocab_app.sqlite3 --min-samples 2
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
MODELS_DIR = ROOT / "models"
STATUS_PATH = ROOT / "data" / "retrain_status.json"
CANVAS_W = 960
CANVAS_H = 540
MAX_LEN = 96


def write_status(status: str, message: str, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "message": message, **extra}
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_strokes(strokes: Any, max_len: int = MAX_LEN) -> np.ndarray:
    points: List[List[float]] = []
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
        return np.zeros((max_len, 5), dtype="float32")
    arr = np.asarray(points, dtype="float32")
    x = arr[:, 0] / CANVAS_W
    y = arr[:, 1] / CANVAS_H
    t = arr[:, 2]
    if t.max() > t.min():
        t = (t - t.min()) / (t.max() - t.min())
    else:
        t = np.zeros_like(t)
    dx = np.concatenate([[0.0], np.diff(x)])
    dy = np.concatenate([[0.0], np.diff(y)])
    seq = np.stack([x, y, dx, dy, t], axis=1).astype("float32")
    if len(seq) >= max_len:
        idx = np.linspace(0, len(seq) - 1, max_len).astype(int)
        seq = seq[idx]
    else:
        seq = np.vstack([seq, np.zeros((max_len - len(seq), 5), dtype="float32")])
    return seq.astype("float32")


def load_samples(db_path: Path, min_points: int = 4) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    rows = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT target, strokes_json FROM stroke_samples ORDER BY id ASC"):
            target = str(row["target"] or "").strip().lower()
            strokes = row["strokes_json"] or "[]"
            seq = flatten_strokes(strokes)
            nonzero = np.count_nonzero(np.abs(seq[:, :2]).sum(axis=1))
            if target and nonzero >= min_points:
                rows.append((target, seq))
    if not rows:
        raise RuntimeError("No usable stroke samples. Play/save several rounds first.")
    categories = sorted({r[0] for r in rows})
    label_to_idx = {label: i for i, label in enumerate(categories)}
    x = np.stack([r[1] for r in rows]).astype("float32")
    y = keras.utils.to_categorical([label_to_idx[r[0]] for r in rows], num_classes=len(categories))
    return x, y, categories


def build_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(MAX_LEN, 5), name="stroke_sequence")
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(48))(x)
    x = keras.layers.Dense(96, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = keras.Model(inputs, outputs, name="AirDrawVocabStrokeSequenceModel")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum classes with samples; warning only.")
    args = parser.parse_args()

    write_status("running", "Loading stroke dataset", mode="stroke")
    x, y, categories = load_samples(args.db)
    if len(categories) < 2:
        raise RuntimeError("Need at least 2 target classes to train a stroke classifier.")
    if len(x) < max(8, len(categories) * args.min_samples):
        print(f"WARNING: only {len(x)} samples for {len(categories)} classes. Model will be weak.")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    val_size = max(1, int(len(x) * 0.15)) if len(x) >= 10 else 0
    if val_size:
        x_train, y_train = x[:-val_size], y[:-val_size]
        x_val, y_val = x[-val_size:], y[-val_size:]
        validation_data = (x_val, y_val)
    else:
        x_train, y_train = x, y
        validation_data = None

    write_status("running", "Training stroke sequence model", mode="stroke", samples=len(x), classes=len(categories))
    model = build_model(len(categories))
    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy" if validation_data else "accuracy")]
    model.fit(x_train, y_train, validation_data=validation_data, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "stroke_sequence_model.keras")
    (MODELS_DIR / "stroke_categories.json").write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status("done", "Stroke model trained and saved", mode="stroke", samples=len(x), classes=len(categories), model=str(MODELS_DIR / "stroke_sequence_model.keras"))
    print("Saved:", MODELS_DIR / "stroke_sequence_model.keras")
    print("Classes:", categories)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("failed", str(exc), mode="stroke")
        raise
