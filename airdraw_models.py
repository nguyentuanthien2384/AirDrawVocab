"""
airdraw_models.py — "Model zoo" hiện đại cho AirDrawVocab (sketch 28x28, 19 lớp).

Thiết kế để VƯỢT cnn_clean (~95.3%) mà vẫn deploy được trên web/desktop hiện tại
(giữ nguyên input 28x28x1, output softmax 19 lớp, thứ tự nhãn = config.CATEGORIES).

Điểm khác biệt so với bản cũ:
  - GroupNormalization / LayerNormalization thay BatchNorm => ổn định ở inference
    trên TF2.21/Keras3 (tránh đúng lỗi val/inference ~ngẫu nhiên đã gặp).
  - Residual (ResNet) và ConvNeXt block (depthwise 7x7 + pointwise expand) =>
    kiến trúc hiện đại, sâu hơn nhưng không nghẽn gradient.
  - Recipe: augmentation mạnh hơn + label smoothing + Cosine LR + warmup + AdamW.
  - TTA (test-time augmentation) + ensemble => +0.5..1.5% gần như miễn phí.

Dùng:
    from airdraw_models import MODEL_BUILDERS, build_cnn_clean
    model = MODEL_BUILDERS["resnet_gn"]()
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from config import NUM_CLASSES
except Exception:
    NUM_CLASSES = 19


# ============================ AUGMENTATION ============================
def build_augmenter() -> keras.Sequential:
    """Augmentation bằng layer Keras (chạy GPU, ổn định). Không lật ngang."""
    return keras.Sequential([
        layers.RandomRotation(0.06, fill_mode="constant"),
        layers.RandomZoom(0.12, fill_mode="constant"),
        layers.RandomTranslation(0.10, 0.10, fill_mode="constant"),
    ], name="augment")


def cheap_augment(x, y):
    """Augmentation RẺ: chỉ dịch nhẹ ±3px (pad + random_crop). Nhanh hơn rotation/zoom."""
    pad = 3
    xp = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.image.random_crop(xp, tf.shape(x)), y


# ============================ KHỐI DÙNG CHUNG =========================
def _gn(x, groups: int = 8):
    f = x.shape[-1]
    return layers.GroupNormalization(groups=min(groups, f))(x)


def _conv_gn_act(x, f, k=3, s=1, act="gelu"):
    x = layers.Conv2D(f, k, strides=s, padding="same", use_bias=False)(x)
    x = _gn(x)
    return layers.Activation(act)(x)


def _res_block(x, f, s=1, drop=0.0):
    sc = x
    y = _conv_gn_act(x, f, 3, s)
    y = layers.Conv2D(f, 3, padding="same", use_bias=False)(y)
    y = _gn(y)
    if s != 1 or sc.shape[-1] != f:
        sc = layers.Conv2D(f, 1, strides=s, padding="same", use_bias=False)(sc)
        sc = _gn(sc)
    y = layers.add([y, sc])
    y = layers.Activation("gelu")(y)
    if drop > 0:
        y = layers.SpatialDropout2D(drop)(y)
    return y


def _convnext_block(x, dim, drop=0.0):
    sc = x
    y = layers.DepthwiseConv2D(7, padding="same", use_bias=False)(x)
    y = layers.GroupNormalization(groups=1)(y)        # ~LayerNorm theo kênh
    y = layers.Conv2D(dim * 4, 1, activation="gelu")(y)
    y = layers.Conv2D(dim, 1)(y)
    if drop > 0:
        y = layers.SpatialDropout2D(drop)(y)
    return layers.add([sc, y])


# ============================ KIẾN TRÚC =============================
def build_cnn_clean(dropout: float = 0.4) -> keras.Model:
    """Baseline đang deploy (VGG-style, không BatchNorm). ~586K params, ~95.3%."""
    inp = keras.Input((28, 28, 1), name="drawing")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inp, out, name="airdraw_clean_cnn")


def build_resnet_gn(dropout: float = 0.4) -> keras.Model:
    """ResNet nhỏ + GroupNorm cho sketch 28x28. ~732K params. Thường chính xác nhất."""
    inp = keras.Input((28, 28, 1), name="drawing")
    x = _conv_gn_act(inp, 32, 3, 1)
    x = _res_block(x, 32, 1, 0.05); x = _res_block(x, 32, 1, 0.05)
    x = _res_block(x, 64, 2, 0.10); x = _res_block(x, 64, 1, 0.10)
    x = _res_block(x, 128, 2, 0.15); x = _res_block(x, 128, 1, 0.15)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="gelu")(x); x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inp, out, name="resnet_gn_sketch")


def build_convnext_mini(dropout: float = 0.4) -> keras.Model:
    """ConvNeXt-mini (kiến trúc conv hiện đại 2022+) chỉnh cho 28x28. ~1.66M params."""
    inp = keras.Input((28, 28, 1), name="drawing")
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inp)
    x = layers.GroupNormalization(groups=1)(x)                          # stem @28
    x = _convnext_block(x, 64, 0.05); x = _convnext_block(x, 64, 0.05)
    x = layers.Conv2D(128, 2, strides=2, use_bias=False)(x)
    x = layers.GroupNormalization(groups=1)(x)                          # ->14
    x = _convnext_block(x, 128, 0.10); x = _convnext_block(x, 128, 0.10)
    x = layers.Conv2D(256, 2, strides=2, use_bias=False)(x)
    x = layers.GroupNormalization(groups=1)(x)                          # ->7
    x = _convnext_block(x, 256, 0.15); x = _convnext_block(x, 256, 0.15)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation="gelu")(x); x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inp, out, name="convnext_mini")


def build_cnn_wide_gap(dropout: float = 0.4) -> keras.Model:
    """CNN rộng + GroupNorm + GlobalAveragePooling head. ~1.2M params, robust."""
    inp = keras.Input((28, 28, 1), name="drawing")
    x = inp
    for f in (64, 128, 256):
        x = _conv_gn_act(x, f, act="relu")
        x = _conv_gn_act(x, f, act="relu")
        x = layers.MaxPooling2D()(x)
        x = layers.SpatialDropout2D(0.10)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inp, out, name="cnn_wide_gap")


MODEL_BUILDERS = {
    "cnn_clean": build_cnn_clean,        # baseline cũ (để so sánh)
    "resnet_gn": build_resnet_gn,        # mới — thường tốt nhất
    "convnext_mini": build_convnext_mini,  # mới — hiện đại
    "cnn_wide_gap": build_cnn_wide_gap,  # mới — robust
}


# ============================ RECIPE TRAIN =========================
def make_optimizer(num_train: int, batch: int, epochs: int,
                   base_lr: float = 1.5e-3, warmup_epochs: int = 3):
    steps = int(np.ceil(num_train / batch))
    total = steps * epochs
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr * 0.05,
        decay_steps=max(1, total - steps * warmup_epochs),
        warmup_target=base_lr,
        warmup_steps=steps * warmup_epochs,
        alpha=0.02,
    )
    return keras.optimizers.AdamW(learning_rate=sched, weight_decay=1e-4)


def compile_advanced(model: keras.Model, num_train: int, batch: int, epochs: int,
                     base_lr: float = 1.5e-3, label_smoothing: float = 0.05,
                     steps_per_execution: int = 1):
    model.compile(
        optimizer=make_optimizer(num_train, batch, epochs, base_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
        steps_per_execution=steps_per_execution,
    )
    return model


# ============================ TTA + ENSEMBLE =======================
def predict_tta(model: keras.Model, x: np.ndarray, n: int = 6, pad: int = 3) -> np.ndarray:
    """Trung bình dự đoán trên ảnh gốc + n bản dịch nhẹ -> ổn định/chính xác hơn."""
    probs = [model.predict(x, verbose=0)]
    for _ in range(n):
        xs = tf.image.random_crop(
            tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]), tf.shape(x))
        probs.append(model.predict(xs.numpy() if hasattr(xs, "numpy") else xs, verbose=0))
    return np.mean(probs, axis=0)


def ensemble_predict(models, x: np.ndarray, tta: int = 0) -> np.ndarray:
    """Trung bình xác suất của nhiều model (mỗi model có thể bật TTA)."""
    parts = []
    for m in models:
        parts.append(predict_tta(m, x, n=tta) if tta else m.predict(x, verbose=0))
    return np.mean(parts, axis=0)
