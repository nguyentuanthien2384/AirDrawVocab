"""
train_best.py — Train "model zoo" hiện đại cho AirDrawVocab rồi lưu model tốt
nhất để web/desktop dùng ngay (giữ nguyên MODEL_PATH + thứ tự nhãn).

Khác train_clean.py: dùng các kiến trúc mới (resnet_gn, convnext_mini, cnn_wide_gap),
recipe nâng cao (label smoothing + Cosine LR + warmup + AdamW + augmentation mạnh),
TTA + ensemble khi đánh giá. Mục tiêu vượt baseline ~95.3%.

Ví dụ:
    # Train kiến trúc tốt nhất với nhiều dữ liệu (khuyến nghị có GPU):
    python train_best.py --model resnet_gn --per-class 16000 --epochs 50

    # Train & so sánh tất cả rồi lưu cái tốt nhất + ensemble:
    python train_best.py --model all --per-class 12000 --epochs 45 --ensemble

    # Kiểm thử nhanh không cần mạng/GPU:
    python train_best.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from config import (CATEGORIES, NUM_CLASSES, DATA_DIR, MODELS_DIR, MODEL_PATH,
                    CATEGORIES_PATH, RANDOM_STATE)
import airdraw_models as Z
from urllib.parse import quote
from src.utils.mlflow_utils import (
    start_mlflow_run, log_params, log_metrics, log_model, end_mlflow_run,
)
from src.utils.repro import set_global_seed, collect_environment
from src.utils.model_versioning import save_versioned_model

QD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


# --------------------------- DỮ LIỆU ---------------------------
def download_partial(label: str, n: int) -> np.ndarray:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{label}.npy"
    if out.exists() and len(np.load(out, mmap_mode="r")) >= n:
        return np.load(out)
    nbytes = 256 + n * 784
    url = QD_URL.format(quote(label))  # mã hóa nhãn có dấu cách, vd "ice cream"
    r = requests.get(url, headers={"Range": f"bytes=0-{nbytes}"}, timeout=180)
    r.raise_for_status()
    raw = r.content
    assert raw[:6] == b"\x93NUMPY", f"{label}: file không hợp lệ"
    major = raw[6]
    data_off = (10 + int.from_bytes(raw[8:10], "little")) if major == 1 \
        else (12 + int.from_bytes(raw[8:12], "little"))
    avail = (len(raw) - data_off) // 784
    arr = np.frombuffer(raw[data_off:data_off + avail * 784], dtype=np.uint8).reshape(avail, 784)
    np.save(out, arr)
    return arr


def load_split(per_class, train_pc, val_pc, test_pc, smoke=False):
    rng = np.random.default_rng(RANDOM_STATE)
    need = train_pc + val_pc + test_pc
    Xtr, Ytr, Xva, Yva, Xte, Yte = [], [], [], [], [], []
    for cid, cat in enumerate(CATEGORIES):
        if smoke:
            data = (rng.random((need, 784)) * 255).astype("uint8")
        else:
            download_partial(cat, per_class)
            data = np.load(DATA_DIR / f"{cat}.npy")
            data = data[data.sum(axis=1) > 0]
            if len(data) < need:
                raise ValueError(f"{cat}: chỉ có {len(data)} mẫu, cần {need}.")
        idx = rng.permutation(len(data))[:need]
        data = (data[idx].astype("float32") / 255.0).reshape(-1, 28, 28, 1)
        Xtr.append(data[:train_pc]); Ytr.append(np.full(train_pc, cid))
        Xva.append(data[train_pc:train_pc + val_pc]); Yva.append(np.full(val_pc, cid))
        Xte.append(data[train_pc + val_pc:need]); Yte.append(np.full(test_pc, cid))
    Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
    # Xáo trộn toàn cục tập train (dữ liệu đang xếp theo lớp) -> mỗi batch đủ các lớp,
    # cho phép dùng shuffle buffer nhỏ ở chế độ --fast mà không hỏng train.
    p = rng.permutation(len(Xtr)); Xtr, Ytr = Xtr[p], Ytr[p]
    return (Xtr, Ytr,
            np.concatenate(Xva), np.concatenate(Yva),
            np.concatenate(Xte), np.concatenate(Yte))


def make_ds(x, y, batch, training, aug):
    yc = keras.utils.to_categorical(y, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((x, yc))
    if training:
        ds = ds.shuffle(len(x), seed=RANDOM_STATE, reshuffle_each_iteration=True)
        ds = ds.batch(batch).map(lambda a, b: (aug(a, training=True), b),
                                 num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    return ds.batch(batch).cache().prefetch(tf.data.AUTOTUNE)


def evaluate(prob, y_te, name=""):
    pred = prob.argmax(1)
    acc = float((pred == y_te).mean())
    top3 = float(np.mean([t in row for t, row in zip(y_te, np.argsort(prob, 1)[:, -3:])]))
    print(f"  [{name}] acc={acc*100:.2f}%  top3={top3*100:.2f}%")
    return acc, top3


# --------------------------- MAIN ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet_gn",
                    choices=list(Z.MODEL_BUILDERS) + ["all"])
    ap.add_argument("--per-class", type=int, default=12000)
    ap.add_argument("--train-pc", type=int, default=10000)
    ap.add_argument("--val-pc", type=int, default=1000)
    ap.add_argument("--test-pc", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--tta", type=int, default=6)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="data ngẫu nhiên, không cần mạng")
    ap.add_argument("--fast", action="store_true",
                    help="train nhanh: ít dữ liệu/epoch + augment rẻ + steps_per_execution")
    args = ap.parse_args()

    # Reproducibility
    set_global_seed(RANDOM_STATE)
    env = collect_environment()

    # ==================== MLflow Setup ====================
    start_mlflow_run(
        experiment_name="AirDrawVocab_Advanced",
        run_name=f"{args.model}_{args.per_class}pc",
        tags={
            "model": args.model,
            "script": "train_best.py",
            "fast_mode": args.fast,
            "ensemble": args.ensemble,
            "smoke_test": args.smoke,
        },
    )
    log_params({
        "model": args.model,
        "per_class": args.per_class,
        "train_pc": args.train_pc,
        "val_pc": args.val_pc,
        "test_pc": args.test_pc,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "tta": args.tta,
        "ensemble": args.ensemble,
        "fast_mode": args.fast,
        "smoke": args.smoke,
        "seed": RANDOM_STATE,
        **{f"env_{k}": v for k, v in env.items()},
    })
    # =====================================================

    spe = 1
    if args.fast:
        args.per_class = min(args.per_class, 5000)
        args.train_pc = min(args.train_pc, 3000)
        args.val_pc = min(args.val_pc, 500)     # val nhỏ -> đánh giá mỗi epoch nhanh hơn
        args.test_pc = min(args.test_pc, 1000)
        args.epochs = min(args.epochs, 20)
        spe = 16
        print("Chế độ FAST/TURBO: train=3000/lớp, val=500/lớp, epochs<=20, "
              "augment rẻ, shuffle buffer nhỏ, steps_per_execution=16")

    if args.smoke:
        args.per_class = 60; args.train_pc = 40; args.val_pc = 10; args.test_pc = 10
        args.epochs = 1; args.batch = 32; args.tta = 2

    if tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: BẬT")

    print("Nạp dữ liệu...")
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_split(
        args.per_class, args.train_pc, args.val_pc, args.test_pc, smoke=args.smoke)
    print(f"train={len(x_tr)} val={len(x_va)} test={len(x_te)}")

    aug = Z.build_augmenter()

    def make_ds_local(x, y, batch, training):
        if args.fast:
            yc = keras.utils.to_categorical(y, NUM_CLASSES)
            ds = tf.data.Dataset.from_tensor_slices((x, yc))
            if training:
                ds = ds.shuffle(min(len(x), 8192), seed=RANDOM_STATE, reshuffle_each_iteration=True)
                ds = ds.batch(batch).map(Z.cheap_augment, num_parallel_calls=tf.data.AUTOTUNE)
                return ds.prefetch(tf.data.AUTOTUNE)
            return ds.batch(batch).cache().prefetch(tf.data.AUTOTUNE)
        return make_ds(x, y, batch, training, aug)

    es_patience = 5 if args.fast else 8
    names = list(Z.MODEL_BUILDERS) if args.model == "all" else [args.model]
    names = [n for n in names if n != "cnn_clean"] or names  # ưu tiên kiến trúc mới

    trained, probs, scores = {}, {}, {}
    for name in names:
        print(f"\n==> Train {name} ...")
        model = Z.compile_advanced(Z.MODEL_BUILDERS[name](), len(x_tr), args.batch,
                                   args.epochs, steps_per_execution=spe)
        cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                            patience=es_patience, restore_best_weights=True, verbose=1)]
        t0 = time.time()
        model.fit(make_ds_local(x_tr, y_tr, args.batch, True),
                  validation_data=make_ds_local(x_va, y_va, args.batch, False),
                  epochs=args.epochs, callbacks=cb, verbose=2)
        print(f"  time={time.time()-t0:.1f}s params={model.count_params():,}")
        p = model.predict(make_ds_local(x_te, y_te, args.batch, False), verbose=0)
        acc, _ = evaluate(p, y_te, name)
        trained[name] = model; probs[name] = p; scores[name] = acc

    best = max(scores, key=scores.get)
    print(f"\nKiến trúc tốt nhất (plain): {best} ({scores[best]*100:.2f}%)")

    # TTA cho model tốt nhất
    if args.tta:
        print(f"==> TTA {best}:")
        p_tta = Z.predict_tta(trained[best], x_te, n=args.tta)
        evaluate(p_tta, y_te, f"{best}+TTA")

    # Ensemble
    if args.ensemble and len(trained) >= 2:
        print("==> Ensemble (+TTA):")
        p_ens = Z.ensemble_predict(list(trained.values()), x_te, tta=args.tta)
        evaluate(p_ens, y_te, "ensemble")
        for n, m in trained.items():
            m.save(MODELS_DIR / f"member_{n}.keras")
        print(f"Đã lưu {len(trained)} model thành viên (member_*.keras) cho ensemble.")

    # ==================== MLflow Logging + Versioning ====================
    metrics = {f"acc_{n}": float(s) for n, s in scores.items()}
    metrics["best_plain_accuracy"] = float(scores[best])
    log_metrics(metrics)
    log_model(trained[best], model_name=best)
    save_versioned_model(
        trained[best],
        base_name=best,
        metrics={"best_plain_accuracy": float(scores[best])},
        params={"per_class": args.per_class, "epochs": args.epochs, "model": args.model},
        extra={"env": env, "all_scores": {n: float(s) for n, s in scores.items()}},
    )
    end_mlflow_run()
    # ====================================================================

    # Lưu model deploy
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained[best].save(MODEL_PATH)
    CATEGORIES_PATH.write_text(json.dumps(CATEGORIES, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu model deploy -> {MODEL_PATH}")
    print("categories.json đồng bộ. Khởi động lại backend để dùng model mới.")


if __name__ == "__main__":
    main()
