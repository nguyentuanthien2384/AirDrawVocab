"""
Evaluate any AirDrawVocab Keras model on the fixed QuickDraw split.

Example:
    python evaluate_model.py --model models/airdrawvocab_best_advanced.keras
"""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow import keras

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from config import CATEGORIES, ROOT, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset

REPORTS_DIR = ROOT / "assets" / "reports" / "evaluations"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an AirDrawVocab model.")
    parser.add_argument("--model", default="models/airdrawvocab_best_advanced.keras")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_dir = REPORTS_DIR / f"{model_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset()
    _, _, X_test, _, _, y_test = split_dataset(X, y, seed=args.seed)
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = keras.models.load_model(model_path, compile=False)
    probs = model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    top3 = float(np.mean([t in row for t, row in zip(y_test, np.argsort(probs, axis=1)[:, -3:])]))
    cm = confusion_matrix(y_test, y_pred)

    summary = {
        "model": str(model_path.name),
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "top3_accuracy": top3,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "errors": int(np.sum(y_test != y_pred)),
    }

    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])

    report_text = classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0)
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    np.savetxt(run_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    wrong = np.where(y_test != y_pred)[0]
    with (run_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "true_label", "predicted_label", "confidence"])
        for idx in wrong:
            w.writerow([idx, CATEGORIES[int(y_test[idx])], CATEGORIES[int(y_pred[idx])], f"{np.max(probs[idx]):.4f}"])

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f"Confusion Matrix - {model_path.name}", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
