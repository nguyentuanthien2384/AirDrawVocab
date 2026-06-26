"""
train_highacc.py — Huấn luyện model nhận diện độ chính xác cao cho AirDrawVocab.
ĐỒNG BỘ với notebook AirDrawVocab_train_highacc_colab.ipynb:
  - Kiến trúc VGG + BatchNormalization (chạy tốt trên GPU; trên CPU vẫn train được nhưng chậm)
  - Label smoothing + augmentation + Cosine LR + EarlyStopping
  - Test-Time Augmentation (TTA) khi đánh giá  (khớp backend predict_proba)

Sau khi train xong, model lưu vào models/airdrawvocab_best_advanced.keras và
models/categories.json được đồng bộ -> backend tự nạp model mới khi khởi động lại.

Chạy (từ thư mục gốc dự án):
    # GPU (khuyến nghị, đạt ~96-98%):
    python src/training/train_highacc.py --per-class 12000 --epochs 40 --batch 512
    # CPU (nhanh, ~93-94%):
    python src/training/train_highacc.py --per-class 3500 --epochs 14 --batch 256
"""
from __future__ import annotations
import argparse, json, os, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import sys as _sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from config import CATEGORIES, NUM_CLASSES, DATA_DIR, MODELS_DIR, CATEGORIES_PATH, RANDOM_STATE

SEED = RANDOM_STATE


def load_split(per_class, val_pc, test_pc, seed):
    need = per_class + val_pc + test_pc
    rng = np.random.default_rng(seed)
    xt, yt, xv, yv, xe, ye = [], [], [], [], [], []
    for cid, c in enumerate(CATEGORIES):
        path = DATA_DIR / f"{c}.npy"
        d = np.load(path, mmap_mode="r")
        d = np.asarray(d)
        d = d[d.sum(1) > 0]
        if len(d) < need:
            raise ValueError(f"{c}: chỉ có {len(d)} mẫu < {need}. Giảm --per-class hoặc tải thêm dữ liệu.")
        idx = rng.permutation(len(d))[:need]
        d = d[idx].astype("float32") / 255.0
        d = d.reshape(-1, 28, 28, 1)
        xt.append(d[:per_class]); yt.append(np.full(per_class, cid))
        xv.append(d[per_class:per_class + val_pc]); yv.append(np.full(val_pc, cid))
        xe.append(d[per_class + val_pc:need]); ye.append(np.full(test_pc, cid))
    return (np.concatenate(xt), np.concatenate(yt), np.concatenate(xv),
            np.concatenate(yv), np.concatenate(xe), np.concatenate(ye))


def _aug(x, y):
    xp = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]])
    xp = tf.image.random_crop(xp, tf.shape(x))
    return xp, y


def make_ds(x, y, batch, training):
    yc = keras.utils.to_categorical(y, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((x, yc))
    if training:
        ds = ds.shuffle(len(x), seed=SEED).batch(batch).map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE)


def _conv_bn(x, f):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_model(use_bn=True):
    """VGG-style. use_bn=True (giống notebook, tốt trên GPU). use_bn=False: bản
    không BatchNorm ổn định trên CPU/Keras 3 (tránh lỗi val phân kỳ)."""
    inp = keras.Input((28, 28, 1))
    def block(x, f, drop):
        if use_bn:
            x = _conv_bn(x, f); x = _conv_bn(x, f)
        else:
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x); x = layers.Dropout(drop)(x)
        return x
    x = block(inp, 64, 0.25)
    x = block(x, 128, 0.25)
    x = block(x, 256, 0.35)
    x = layers.Flatten()(x)
    if use_bn:
        x = layers.Dense(512, use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    else:
        x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="AirDrawVGG")


def _top3(y, p):
    return float(np.mean([t in r for t, r in zip(y, np.argsort(p, 1)[:, -3:])]))


def tta_predict(model, xe, batch):
    """Test-Time Augmentation: gốc + 4 dịch chuyển 1px (khớp backend.predict_proba)."""
    preds = [model.predict(make_ds(xe, np.zeros(len(xe)), batch, False), verbose=0)]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        xs = np.roll(np.roll(xe, dx, axis=2), dy, axis=1)
        preds.append(model.predict(make_ds(xs, np.zeros(len(xs)), batch, False), verbose=0))
    return np.mean(preds, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12000, help="mẫu train mỗi lớp")
    ap.add_argument("--val-per-class", type=int, default=1000)
    ap.add_argument("--test-per-class", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--no-bn", action="store_true", help="tắt BatchNorm (khuyến nghị nếu train trên CPU)")
    ap.add_argument("--out", default=str(MODELS_DIR / "airdrawvocab_best_advanced.keras"))
    args = ap.parse_args()

    np.random.seed(SEED); tf.random.set_seed(SEED)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPU: {gpus or 'không có (train CPU sẽ chậm)'} | classes={NUM_CLASSES}")

    xt, yt, xv, yv, xe, ye = load_split(args.per_class, args.val_per_class, args.test_per_class, SEED)
    print(f"train={len(xt)} val={len(xv)} test={len(xe)}")
    dtr = make_ds(xt, yt, args.batch, True)
    dva = make_ds(xv, yv, args.batch, False)

    model = build_model(use_bn=not args.no_bn)
    steps = max(1, len(xt) // args.batch)
    lr = keras.optimizers.schedules.CosineDecay(1e-3, args.epochs * steps, alpha=0.02)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
                  metrics=["accuracy"])

    ckpt = str(MODELS_DIR / "_highacc_ckpt.keras")
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max",
                                        save_best_only=True, verbose=0),
    ]
    t0 = time.time()
    model.fit(dtr, validation_data=dva, epochs=args.epochs, callbacks=cbs, verbose=2)
    print(f"Train time: {time.time()-t0:.1f}s")

    # đánh giá plain + TTA
    p = model.predict(make_ds(xe, ye, args.batch, False), verbose=0)
    acc = float((p.argmax(1) == ye).mean()); t3 = _top3(ye, p)
    pt = tta_predict(model, xe, args.batch)
    acct = float((pt.argmax(1) == ye).mean()); t3t = _top3(ye, pt)
    print(f"\nTEST  plain acc={acc*100:.2f}% top3={t3*100:.2f}%")
    print(f"TEST  TTA   acc={acct*100:.2f}% top3={t3t*100:.2f}%")

    model.save(args.out)
    CATEGORIES_PATH.write_text(json.dumps(CATEGORIES, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu model: {args.out}")
    print(f"Đã đồng bộ nhãn: {CATEGORIES_PATH}")
    print("Khởi động lại backend để nạp model mới. Inference đã bật TTA sẵn (USE_TTA=1).")


if __name__ == "__main__":
    main()
