"""
Advanced training pipeline for AirDrawVocab.

Why this script exists:
- The original train_model.py is a clear baseline CNN trainer.
- This script is a stronger "model zoo" trainer for the capstone project:
  residual CNN for 28x28 sketches, optional popular transfer-learning
  backbones, modern callbacks, label smoothing, optional MixUp, detailed
  reports, and automatic best-model export.

Recommended first run:
    python advanced_train_model.py --model resnet_sketch --epochs 60

Quick smoke test:
    python advanced_train_model.py --model resnet_sketch --epochs 1 --train-per-class 20 --val-per-class 5 --test-per-class 5

Optional comparisons:
    python advanced_train_model.py --model all --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from config import (
    ROOT, CATEGORIES, NUM_CLASSES, RANDOM_STATE,
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS, SAMPLES_PER_CLASS,
)
from mlflow_utils import (
    start_mlflow_run, log_params, log_metrics, log_model,
    end_mlflow_run, log_training_artifacts,
)
from repro import collect_environment

REPORTS_DIR = ROOT / "assets" / "reports" / "advanced_training"


def configure_runtime(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_dataset(samples_per_class: int = SAMPLES_PER_CLASS) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for class_id, category in enumerate(CATEGORIES):
        path = DATA_DIR / f"{category}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        data = np.load(path)
        data = data[data.sum(axis=1) > 0]
        if len(data) < samples_per_class:
            raise ValueError(f"{category} has {len(data)} valid samples, need {samples_per_class}.")
        data = data[:samples_per_class].astype("float32") / 255.0
        data = data.reshape(-1, 28, 28, 1)
        x_parts.append(data)
        y_parts.append(np.full(len(data), class_id, dtype=np.int32))
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    required = train_per_class + val_per_class + test_per_class
    for class_id in range(NUM_CLASSES):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) < required:
            raise ValueError(f"Class {CATEGORIES[class_id]} has {len(class_indices)} samples, need {required}.")
        rng.shuffle(class_indices)
        train_indices.extend(class_indices[:train_per_class])
        val_indices.extend(class_indices[train_per_class:train_per_class + val_per_class])
        test_indices.extend(class_indices[train_per_class + val_per_class:required])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return x[train_indices], x[val_indices], x[test_indices], y[train_indices], y[val_indices], y[test_indices]


def prepare_inputs(x: np.ndarray, model_name: str) -> np.ndarray:
    return x.astype("float32")


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    mixup_alpha: float = 0.0,
) -> tf.data.Dataset:
    y_cat = keras.utils.to_categorical(y, NUM_CLASSES).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y_cat))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if mixup_alpha > 0:
        ds = ds.map(lambda bx, by: mixup_batch(bx, by, mixup_alpha), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def mixup_batch(images: tf.Tensor, labels: tf.Tensor, alpha: float) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(images)[0]
    shuffled = tf.random.shuffle(tf.range(batch_size))
    images_2 = tf.gather(images, shuffled)
    labels_2 = tf.gather(labels, shuffled)

    # Approximate Beta(alpha, alpha) using Gamma samples. Works without extra deps.
    gamma_1 = tf.random.gamma([batch_size], alpha)
    gamma_2 = tf.random.gamma([batch_size], alpha)
    lam = gamma_1 / (gamma_1 + gamma_2)
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])
    return images * lam_x + images_2 * (1.0 - lam_x), labels * lam_y + labels_2 * (1.0 - lam_y)


def augmentation_layers(input_shape: tuple[int, int, int]) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom((-0.10, 0.12), (-0.10, 0.12), fill_mode="constant", fill_value=0.0),
        ],
        name=f"augmentation_{input_shape[0]}x{input_shape[1]}",
    )


def residual_block(x: tf.Tensor, filters: int, stride: int, drop_rate: float, name: str) -> tf.Tensor:
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("swish", name=f"{name}_act1")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_sepconv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"{name}_drop")(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_skip_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("swish", name=f"{name}_out")(x)


def build_resnet_sketch(input_shape: tuple[int, int, int], dropout: float) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="drawing")
    x = augmentation_layers(input_shape)(inputs)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("swish", name="stem_swish")(x)

    x = residual_block(x, 32, 1, 0.05, "stage1_block1")
    x = residual_block(x, 32, 1, 0.05, "stage1_block2")
    x = residual_block(x, 64, 2, 0.08, "stage2_block1")
    x = residual_block(x, 64, 1, 0.08, "stage2_block2")
    x = residual_block(x, 128, 2, 0.12, "stage3_block1")
    x = residual_block(x, 128, 1, 0.12, "stage3_block2")
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(192, activation="swish", name="classifier_dense")(x)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="resnet_sketch")


def build_efficientnetv2b0(input_shape: tuple[int, int, int], dropout: float) -> keras.Model:
    backbone_shape = (96, 96, 3)
    try:
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=backbone_shape,
            pooling="avg",
            include_preprocessing=False,
        )
    except Exception as exc:
        print(f"[WARN] Could not load ImageNet EfficientNetV2B0 weights: {exc}. Falling back to weights=None.")
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,
            input_shape=backbone_shape,
            pooling="avg",
            include_preprocessing=False,
        )
    inputs = keras.Input(shape=input_shape, name="drawing")
    x = augmentation_layers(input_shape)(inputs)
    x = layers.Resizing(96, 96, interpolation="bilinear", name="resize_to_96")(x)
    x = layers.Concatenate(name="gray_to_rgb")([x, x, x])
    # Keras EfficientNetV2 with include_preprocessing=False expects roughly [-1, 1].
    x = layers.Rescaling(2.0, offset=-1.0, name="efficientnet_rescale")(x)
    x = base(x, training=False)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="efficientnetv2b0_airdraw")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    return model


def build_mobilenetv3small(input_shape: tuple[int, int, int], dropout: float) -> keras.Model:
    backbone_shape = (96, 96, 3)
    try:
        base = keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=backbone_shape,
            pooling="avg",
            include_preprocessing=True,
        )
    except Exception as exc:
        print(f"[WARN] Could not load ImageNet MobileNetV3Small weights: {exc}. Falling back to weights=None.")
        base = keras.applications.MobileNetV3Small(
            include_top=False,
            weights=None,
            input_shape=backbone_shape,
            pooling="avg",
            include_preprocessing=True,
        )
    inputs = keras.Input(shape=input_shape, name="drawing")
    x = augmentation_layers(input_shape)(inputs)
    x = layers.Resizing(96, 96, interpolation="bilinear", name="resize_to_96")(x)
    x = layers.Concatenate(name="gray_to_rgb")([x, x, x])
    x = base(x, training=False)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv3small_airdraw")
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    return model


MODEL_BUILDERS: dict[str, tuple[tuple[int, int, int], Callable[[tuple[int, int, int], float], keras.Model]]] = {
    "resnet_sketch": ((28, 28, 1), build_resnet_sketch),
    "efficientnetv2b0": ((28, 28, 1), build_efficientnetv2b0),
    "mobilenetv3small": ((28, 28, 1), build_mobilenetv3small),
}


def compile_model(model: keras.Model, learning_rate: float, weight_decay: float, label_smoothing: float) -> None:
    try:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    except Exception:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def evaluate_and_save(
    model: keras.Model,
    model_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    history: keras.callbacks.History,
    run_dir: Path,
) -> dict:
    y_prob = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    top3 = float(np.mean([true in row for true, row in zip(y_test, np.argsort(y_prob, axis=1)[:, -3:])]))
    report_text = classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (run_dir / "classification_report.json").write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savetxt(run_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    wrong = np.where(y_test != y_pred)[0]
    with (run_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_index", "true_label", "predicted_label", "confidence", "top3_labels"])
        for idx in wrong:
            top_indices = np.argsort(y_prob[idx])[-3:][::-1]
            writer.writerow([
                int(idx),
                CATEGORIES[int(y_test[idx])],
                CATEGORIES[int(y_pred[idx])],
                float(np.max(y_prob[idx])),
                "|".join(CATEGORIES[int(i)] for i in top_indices),
            ])

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train")
    plt.plot(history.history.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(run_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary = {
        "model": model_name,
        "accuracy": float(accuracy),
        "top3_accuracy": top3,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "errors": int(len(wrong)),
        "test_samples": int(len(y_test)),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    return summary


def train_one_model(args: argparse.Namespace, model_name: str, x: np.ndarray, y: np.ndarray) -> dict:
    input_shape, builder = MODEL_BUILDERS[model_name]
    run_id = time.strftime(f"{model_name}_%Y%m%d_%H%M%S")
    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_full_split = (
        args.train_per_class == TRAIN_PER_CLASS
        and args.val_per_class == VAL_PER_CLASS
        and args.test_per_class == TEST_PER_CLASS
        and not args.smoke_test
    )

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x,
        y,
        args.seed,
        args.train_per_class,
        args.val_per_class,
        args.test_per_class,
    )
    x_train = prepare_inputs(x_train, model_name)
    x_val = prepare_inputs(x_val, model_name)
    x_test = prepare_inputs(x_test, model_name)

    train_ds = make_dataset(x_train, y_train, args.batch_size, True, args.seed, args.mixup_alpha)
    val_ds = make_dataset(x_val, y_val, args.batch_size, False, args.seed)

    model = builder(input_shape, args.dropout)
    compile_model(model, args.learning_rate, args.weight_decay, args.label_smoothing)

    if is_full_split:
        best_path = MODELS_DIR / f"{model_name}_best.keras"
        last_path = MODELS_DIR / f"{model_name}_last.keras"
    else:
        best_path = run_dir / f"{model_name}_best.keras"
        last_path = run_dir / f"{model_name}_last.keras"
    deploy_path = MODELS_DIR / "airdrawvocab_best_advanced.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]

    config = {
        "model": model_name,
        "input_shape": input_shape,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "deploy_threshold": args.deploy_threshold,
        "force_deploy": args.force_deploy,
        "seed": args.seed,
        "split": {
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "test_per_class": args.test_per_class,
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    # ==================== MLflow: bắt đầu run cho model này ====================
    start_mlflow_run(
        experiment_name="AirDrawVocab_AdvancedModels",
        run_name=f"{model_name}_{args.epochs}ep",
        tags={"model": model_name, "script": "advanced_train_model.py"},
    )
    log_params({
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "test_per_class": args.test_per_class,
        "seed": args.seed,
        **{f"env_{k}": v for k, v in collect_environment().items()},
    })
    # =========================================================================

    print(f"\n===== Training {model_name} =====")
    print(f"Reports: {run_dir}")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=1, callbacks=callbacks)

    if best_path.exists():
        model = keras.models.load_model(best_path, compile=False)
        compile_model(model, args.learning_rate, args.weight_decay, args.label_smoothing)

    summary = evaluate_and_save(model, model_name, x_test, y_test, history, run_dir)
    model.save(last_path)

    # Only a strong full-split model is allowed to replace the deploy model.
    # Smoke/subset runs are useful for debugging, but must not be loaded by the app.
    should_deploy = (
        is_full_split
        and input_shape == (28, 28, 1)
        and (args.force_deploy or summary["accuracy"] >= args.deploy_threshold)
    )
    if should_deploy:
        model.save(deploy_path)
        summary["deploy_model"] = str(deploy_path)
    elif not is_full_split:
        summary["deploy_model"] = None
        summary["note"] = "Subset/smoke run: deploy model was not updated."
    else:
        summary["deploy_model"] = None
        summary["note"] = (
            f"Full run finished, but accuracy {summary['accuracy']:.4f} is below "
            f"deploy threshold {args.deploy_threshold:.4f}. Existing app model was kept."
        )
    summary["best_model"] = str(best_path)
    summary["run_dir"] = str(run_dir)

    # ==================== MLflow: log kết quả & kết thúc run ====================
    log_metrics({
        "accuracy": summary["accuracy"],
        "top3_accuracy": summary["top3_accuracy"],
        "precision_weighted": summary["precision_weighted"],
        "recall_weighted": summary["recall_weighted"],
        "f1_weighted": summary["f1_weighted"],
    })
    log_model(model, model_name=model_name)
    log_training_artifacts(history, run_dir)
    end_mlflow_run()
    # ==========================================================================

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def write_comparison(summaries: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = sorted(summaries, key=lambda item: item["accuracy"], reverse=True)
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with (REPORTS_DIR / "model_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "accuracy", "top3_accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "errors", "test_samples", "best_model", "deploy_model", "run_dir"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced AirDrawVocab trainer with modern model zoo.")
    parser.add_argument("--model", choices=["resnet_sketch", "efficientnetv2b0", "mobilenetv3small", "all"], default="resnet_sketch")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Use 0 to disable; try 0.1 or 0.2 for stronger regularization.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    parser.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    parser.add_argument("--test-per-class", type=int, default=TEST_PER_CLASS)
    parser.add_argument("--smoke-test", action="store_true", help="Shortcut for a tiny 1-epoch run to verify the pipeline.")
    parser.add_argument("--deploy-threshold", type=float, default=0.985, help="Only deploy full-split models at or above this test accuracy.")
    parser.add_argument("--force-deploy", action="store_true", help="Deploy full-split model even if below threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.train_per_class = 20
        args.val_per_class = 5
        args.test_per_class = 5
        args.batch_size = min(args.batch_size, 32)
    configure_runtime(args.seed)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    samples_per_class = args.train_per_class + args.val_per_class + args.test_per_class
    x, y = load_dataset(samples_per_class=samples_per_class)
    selected = list(MODEL_BUILDERS) if args.model == "all" else [args.model]
    summaries = [train_one_model(args, model_name, x, y) for model_name in selected]
    write_comparison(summaries)
    print(f"\nModel comparison saved to: {REPORTS_DIR / 'model_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
