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
import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# parents[2] vì file đã chuyển vào src/training/ (giữ ROOT trỏ về gốc dự án)
ROOT = Path(__file__).resolve().parents[2]

# bootstrap: cho phép import stroke_features (gốc) và src.utils.*
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

# Khi backend chạy script này dạng subprocess, stdout bị redirect -> Windows mặc
# định dùng cp1252 và CRASH khi in tiếng Việt ('charmap' codec). Ép UTF-8.
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from stroke_features import strokes_to_sequence, MAX_LEN, NUM_FEATURES, count_active_points
from src.utils.mlflow_utils import (
    start_mlflow_run, log_params, log_metrics, log_model, end_mlflow_run,
)
from src.utils.repro import set_global_seed, collect_environment
from src.utils.model_versioning import save_versioned_model

DB_PATH = ROOT / "data" / "airdrawvocab_app.sqlite3"
MODELS_DIR = ROOT / "models"
STATUS_PATH = Path(os.getenv("AIRDRAW_RETRAIN_STATUS_PATH", str(ROOT / "data" / "self_improving_loop" / "status" / "retrain_status.json")))
REPORTS_DIR = ROOT / "assets" / "reports" / "stroke"


def write_status(status: str, message: str, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "message": message, **extra}
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_samples(db_path: Path, min_points: int = 4) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    rows = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT target, strokes_json FROM stroke_samples ORDER BY id ASC"):
            target = str(row["target"] or "").strip().lower()
            strokes = row["strokes_json"] or "[]"
            seq = strokes_to_sequence(strokes)  # đặc trưng dùng chung với backend
            if target and count_active_points(seq) >= min_points:
                rows.append((target, seq))
    if not rows:
        raise RuntimeError("No usable stroke samples. Play/save several rounds first.")
    categories = sorted({r[0] for r in rows})
    label_to_idx = {label: i for i, label in enumerate(categories)}
    x = np.stack([r[1] for r in rows]).astype("float32")
    y = keras.utils.to_categorical([label_to_idx[r[0]] for r in rows], num_classes=len(categories))
    return x, y, categories


def build_model(num_classes: int, lr: float = 1e-3) -> keras.Model:
    """BiGRU 2 lớp nâng cao: nhiều unit hơn, LayerNorm, recurrent dropout,
    label smoothing. Input theo bộ đặc trưng dùng chung (NUM_FEATURES)."""
    inputs = keras.Input(shape=(MAX_LEN, NUM_FEATURES), name="stroke_sequence")
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(96, return_sequences=True, recurrent_dropout=0.1),
        name="bigru_1",
    )(x)
    x = keras.layers.LayerNormalization(name="ln_1")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(64), name="bigru_2")(x)
    x = keras.layers.LayerNormalization(name="ln_2")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="swish", name="head_dense")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = keras.Model(inputs, outputs, name="AirDrawVocabStrokeSequenceModel")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_int: np.ndarray, num_classes: int) -> dict:
    """Cân bằng lớp đơn giản: weight ~ tổng/(n_lớp * count)."""
    counts = np.bincount(y_int, minlength=num_classes).astype("float64")
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def evaluate_and_report(model, x_val, y_val_int, categories) -> dict:
    """Đánh giá trên tập val: accuracy + per-class report + confusion matrix."""
    probs = model.predict(x_val, verbose=0)
    pred = probs.argmax(1)
    acc = float((pred == y_val_int).mean())
    top3 = float(np.mean([t in row for t, row in zip(y_val_int, np.argsort(probs, 1)[:, -3:])])) \
        if probs.shape[1] >= 3 else acc

    summary = {"val_accuracy": acc, "val_top3_accuracy": top3, "val_samples": int(len(y_val_int))}
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        labels = list(range(len(categories)))
        report = classification_report(
            y_val_int, pred, labels=labels, target_names=categories, zero_division=0
        )
        (REPORTS_DIR / "stroke_classification_report.txt").write_text(report, encoding="utf-8")
        cm = confusion_matrix(y_val_int, pred, labels=labels)
        np.savetxt(REPORTS_DIR / "stroke_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        print("\n" + report)
        print(f"Đã lưu báo cáo đánh giá: {REPORTS_DIR}")
    except Exception as exc:  # sklearn không có hoặc lỗi -> bỏ qua an toàn
        print(f"[Eval] Bỏ qua per-class report ({exc})")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum classes with samples; warning only.")
    args = parser.parse_args()

    set_global_seed(42)
    env = collect_environment()

    write_status("running", "Loading stroke dataset", mode="stroke")
    x, y, categories = load_samples(args.db)
    if len(categories) < 2:
        raise RuntimeError("Need at least 2 target classes to train a stroke classifier.")
    if len(x) < max(8, len(categories) * args.min_samples):
        print(f"WARNING: only {len(x)} samples for {len(categories)} classes. Model will be weak.")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    y_int_all = y.argmax(1)
    val_size = max(1, int(len(x) * 0.15)) if len(x) >= 10 else 0
    if val_size:
        x_train, y_train = x[:-val_size], y[:-val_size]
        x_val, y_val = x[-val_size:], y[-val_size:]
        validation_data = (x_val, y_val)
    else:
        x_train, y_train = x, y
        x_val = y_val = None
        validation_data = None

    # ==================== MLflow ====================
    start_mlflow_run(
        experiment_name="AirDrawVocab_Stroke",
        run_name=f"bigru_{len(categories)}cls_{len(x)}samples",
        tags={"model_type": "BiGRU", "script": "train_stroke_model.py", "branch": "final_boss"},
    )
    log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": MAX_LEN,
        "num_features": NUM_FEATURES,
        "num_classes": len(categories),
        "total_samples": int(len(x)),
        "val_samples": int(val_size),
        "seed": 42,
        **{f"env_{k}": v for k, v in env.items()},
    })
    # ===============================================

    class_weight = compute_class_weights(y_train.argmax(1), len(categories))

    write_status("running", "Training stroke sequence model", mode="stroke", samples=len(x), classes=len(categories))
    model = build_model(len(categories), lr=args.lr)
    monitor = "val_accuracy" if validation_data else "accuracy"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if validation_data else "loss",
                                          factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        x_train, y_train, validation_data=validation_data,
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, class_weight=class_weight, verbose=2,
    )

    # ==================== Đánh giá + log ====================
    metrics = {"final_train_accuracy": float(history.history["accuracy"][-1])}
    if validation_data:
        metrics.update(evaluate_and_report(model, x_val, y_val.argmax(1), categories))
    log_metrics(metrics)
    log_model(model, model_name="stroke_bigru")
    save_versioned_model(
        model,
        base_name="stroke_bigru",
        metrics=metrics,
        params={"epochs": args.epochs, "num_classes": len(categories), "samples": int(len(x))},
        extra={"env": env, "categories": categories},
    )
    end_mlflow_run()
    # =======================================================

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
