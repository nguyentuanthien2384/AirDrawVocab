"""
Train sạch & đáng tin cậy cho AirDrawVocab (QuickDraw 28x28, số lớp theo config.CATEGORIES).

Khác với advanced_train_model.py (gặp lỗi val_accuracy ~ngẫu nhiên do
augmentation layers/BatchNorm trên Keras 3), script này:
- Không nhúng augmentation layer trong model (tránh lệch train/inference).
- Augmentation nhẹ thực hiện trong tf.data (gọi tường minh training=True).
- Kiến trúc CNN gọn, BN ổn định.
- Nhãn gán theo đúng thứ tự config.CATEGORIES => khớp models/categories.json.

Chạy (từ thư mục gốc dự án):
    python src/training/train_clean.py --epochs 18 --train-per-class 2000 --val-per-class 400 --test-per-class 400
"""
from __future__ import annotations

import argparse
import json
import os

# Dùng hết nhân CPU để train nhanh hơn (đặt trước khi TF khởi tạo runtime).
_cpu = os.cpu_count() or 4
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    tf.config.threading.set_intra_op_parallelism_threads(_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, _cpu // 2))
except Exception:
    pass

# --- bootstrap: thêm thư mục gốc dự án vào sys.path để import config/airdraw_models ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from config import (
    CATEGORIES, NUM_CLASSES, RANDOM_STATE,
    DATA_DIR, MODELS_DIR, CATEGORIES_PATH,
)
from src.utils.mlflow_utils import (
    start_mlflow_run, log_params, log_metrics, log_model, end_mlflow_run,
)
from src.utils.repro import set_global_seed, collect_environment
from src.utils.model_versioning import save_versioned_model


def load_split(train_pc: int, val_pc: int, test_pc: int, seed: int):
    need = train_pc + val_pc + test_pc
    rng = np.random.default_rng(seed)
    xs_tr, ys_tr, xs_va, ys_va, xs_te, ys_te = [], [], [], [], [], []
    for cid, cat in enumerate(CATEGORIES):
        path = DATA_DIR / f"{cat}.npy"
        data = np.load(path)
        data = data[data.sum(axis=1) > 0]
        if len(data) < need:
            raise ValueError(f"{cat}: chỉ có {len(data)} mẫu, cần {need}")
        idx = rng.permutation(len(data))[:need]
        data = data[idx].astype("float32") / 255.0
        data = data.reshape(-1, 28, 28, 1)
        xs_tr.append(data[:train_pc]); ys_tr.append(np.full(train_pc, cid))
        xs_va.append(data[train_pc:train_pc + val_pc]); ys_va.append(np.full(val_pc, cid))
        xs_te.append(data[train_pc + val_pc:need]); ys_te.append(np.full(test_pc, cid))
    x_tr = np.concatenate(xs_tr); y_tr = np.concatenate(ys_tr)
    x_va = np.concatenate(xs_va); y_va = np.concatenate(ys_va)
    x_te = np.concatenate(xs_te); y_te = np.concatenate(ys_te)
    return x_tr, y_tr, x_va, y_va, x_te, y_te


def build_model(dropout: float = 0.4) -> keras.Model:
    # Không dùng BatchNormalization: trên TF2.21/Keras3 nó gây lệch
    # train/inference (val ~ngẫu nhiên). VGG-style nhỏ + Dropout ổn định.
    inputs = keras.Input(shape=(28, 28, 1), name="drawing")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="airdraw_clean_cnn")


def augment(x, y):
    # dịch nhẹ + zoom nhẹ + xoay nhẹ bằng thao tác ảnh, giữ nền 0
    x = tf.image.random_flip_left_right(x) if False else x  # không lật (chữ/áo có hướng)
    # padding rồi random crop = dịch ảnh
    pad = 3
    xp = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    xp = tf.image.random_crop(xp, tf.shape(x))
    return xp, y


def make_ds(x, y, batch, training):
    y_cat = keras.utils.to_categorical(y, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((x, y_cat))
    if training:
        ds = ds.shuffle(len(x), seed=RANDOM_STATE, reshuffle_each_iteration=True)
        ds = ds.batch(batch).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    # val/test cố định -> cache trong RAM cho nhanh
    return ds.batch(batch).cache().prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    # Mặc định tối ưu cho train nhanh trên CPU mà vẫn đạt ~93-94%:
    # 2000 mẫu/lớp + batch 512 + early stopping sớm.
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--train-per-class", type=int, default=2000)
    ap.add_argument("--val-per-class", type=int, default=400)
    ap.add_argument("--test-per-class", type=int, default=400)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--out", default=str(MODELS_DIR / "airdrawvocab_best_advanced.keras"))
    args = ap.parse_args()

    # Reproducibility: cố định seed trước khi build/train.
    set_global_seed(RANDOM_STATE)
    env = collect_environment()

    # ==================== MLflow Setup ====================
    start_mlflow_run(
        experiment_name="AirDrawVocab_CNN",
        run_name=f"clean_cnn_{args.train_per_class}pc_ep{args.epochs}",
        tags={
            "model_type": "CNN_VGG_Style",
            "script": "train_clean.py",
            "dataset": "QuickDraw_28x28",
        },
    )
    log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "test_per_class": args.test_per_class,
        "learning_rate": args.lr,
        "patience": args.patience,
        "dropout": 0.4,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "num_classes": NUM_CLASSES,
        "seed": RANDOM_STATE,
        **{f"env_{k}": v for k, v in env.items()},
    })
    # =====================================================

    print(f"TensorFlow {tf.__version__}, classes={NUM_CLASSES}")
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_split(
        args.train_per_class, args.val_per_class, args.test_per_class, RANDOM_STATE)
    print(f"train={len(x_tr)} val={len(x_va)} test={len(x_te)}")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=args.patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        make_ds(x_tr, y_tr, args.batch_size, True),
        validation_data=make_ds(x_va, y_va, args.batch_size, False),
        epochs=args.epochs, callbacks=cbs, verbose=2,
    )

    # đánh giá test
    prob = model.predict(make_ds(x_te, y_te, args.batch_size, False), verbose=0)
    pred = prob.argmax(1)
    acc = (pred == y_te).mean()
    top3 = np.mean([t in row for t, row in zip(y_te, np.argsort(prob, 1)[:, -3:])])
    print(f"\nTest accuracy = {acc*100:.2f}%  top3 = {top3*100:.2f}%  ({len(y_te)} mẫu)")
    print("Per-class accuracy:")
    for cid, cat in enumerate(CATEGORIES):
        m = y_te == cid
        print(f"  {cat:10s}: {(pred[m] == cid).mean()*100:5.1f}%")

    # ==================== MLflow Logging + Versioning ====================
    metrics = {
        "test_accuracy": float(acc),
        "test_top3_accuracy": float(top3),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    log_metrics(metrics)
    log_model(model, model_name="airdraw_clean_cnn")

    save_versioned_model(
        model,
        base_name="airdraw_clean_cnn",
        metrics=metrics,
        params={
            "epochs": args.epochs,
            "train_per_class": args.train_per_class,
            "learning_rate": args.lr,
        },
        extra={"env": env},
    )
    end_mlflow_run()
    # ====================================================================

    model.save(args.out)
    CATEGORIES_PATH.write_text(json.dumps(CATEGORIES, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu model: {args.out}")
    print(f"Đã đồng bộ nhãn: {CATEGORIES_PATH}")


if __name__ == "__main__":
    main()
