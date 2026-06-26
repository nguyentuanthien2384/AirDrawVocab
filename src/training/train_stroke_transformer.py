"""Train a small Transformer encoder for AirDrawVocab stroke sequences."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import tensorflow as tf
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

from stroke_features import MAX_LEN, NUM_FEATURES, count_active_points, strokes_to_sequence
from src.utils.model_versioning import save_versioned_model

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
MODELS_DIR = ROOT / "models"
STATUS_PATH = Path(os.getenv("AIRDRAW_RETRAIN_STATUS_PATH", str(ROOT / "data" / "self_improving_loop" / "status" / "retrain_status.json")))
REPORTS_DIR = ROOT / "assets" / "reports" / "stroke_transformer"


def write_status(status: str, message: str, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({"status": status, "message": message, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_samples(db_path: Path, min_points: int = 4) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT target, strokes_json FROM stroke_samples ORDER BY id ASC"):
            label = str(row["target"] or "").strip().lower()
            if not label:
                continue
            seq = strokes_to_sequence(row["strokes_json"] or "[]")
            if count_active_points(seq) >= min_points:
                rows.append((label, seq))
    if not rows:
        raise RuntimeError("No usable stroke samples.")
    categories = sorted({r[0] for r in rows})
    label_to_idx = {label: i for i, label in enumerate(categories)}
    x = np.stack([r[1] for r in rows]).astype("float32")
    y_int = np.asarray([label_to_idx[r[0]] for r in rows], dtype="int64")
    return x, y_int, categories


def transformer_block(x, d_model: int, heads: int, ff_dim: int, dropout: float):
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=max(8, d_model // heads), dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ff = keras.Sequential([layers.Dense(ff_dim, activation="swish"), layers.Dropout(dropout), layers.Dense(d_model)])(x)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_model(num_classes: int, d_model: int = 64, heads: int = 4, ff_dim: int = 128, blocks: int = 2, dropout: float = 0.2, lr: float = 7e-4) -> keras.Model:
    inputs = keras.Input(shape=(MAX_LEN, NUM_FEATURES), name="stroke_sequence")
    mask = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Dense(d_model)(mask)
    positions = tf.range(start=0, limit=MAX_LEN, delta=1)
    pos_emb = layers.Embedding(input_dim=MAX_LEN, output_dim=d_model)(positions)
    x = x + pos_emb
    for _ in range(blocks):
        x = transformer_block(x, d_model, heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="swish")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="AirDrawVocabStrokeTransformer")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=["accuracy"])
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--min-points", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)
    write_status("running", "Loading stroke transformer dataset", mode="stroke_transformer")
    x, y_int, categories = load_samples(args.db, args.min_points)
    if len(categories) < 2:
        raise RuntimeError("Need at least two classes.")
    y = keras.utils.to_categorical(y_int, len(categories))
    idx = np.random.default_rng(42).permutation(len(x))
    x, y, y_int = x[idx], y[idx], y_int[idx]
    val_size = max(1, int(len(x) * 0.15)) if len(x) >= 20 else 0
    if val_size:
        x_train, y_train = x[:-val_size], y[:-val_size]
        x_val, y_val, y_val_int = x[-val_size:], y[-val_size:], y_int[-val_size:]
    else:
        x_train, y_train = x, y
        x_val = y_val = y_val_int = None
    write_status("running", "Training stroke transformer", mode="stroke_transformer", samples=len(x), classes=len(categories))
    model = build_model(len(categories), lr=args.lr)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy" if val_size else "accuracy", mode="max", patience=6, restore_best_weights=True)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val) if val_size else None, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    metrics = {"samples": int(len(x)), "classes": int(len(categories))}
    if val_size:
        probs = model.predict(x_val, verbose=0)
        metrics["val_accuracy"] = float((probs.argmax(1) == y_val_int).mean())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "stroke_transformer_model.keras")
    (MODELS_DIR / "stroke_transformer_categories.json").write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        save_versioned_model(model, base_name="stroke_transformer", metrics=metrics, params={"epochs": args.epochs}, extra={"categories": categories})
    except Exception as exc:
        metrics["versioning_warning"] = str(exc)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status("done", "Stroke transformer trained", mode="stroke_transformer", samples=len(x), classes=len(categories), metrics=metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_status("failed", str(exc), mode="stroke_transformer")
        raise
