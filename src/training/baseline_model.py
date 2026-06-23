"""
Baseline NearestCentroid cho AirDrawVocab.
Dùng cùng split 800/150/150 với CNN để so sánh khoa học.
"""
from __future__ import annotations

import json, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestCentroid

# --- bootstrap: thêm thư mục gốc dự án vào sys.path ---
import sys as _sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from config import CATEGORIES, RESULTS_DIR, REPORTS_DIR, RANDOM_STATE
from src.data.data_utils import load_dataset, split_dataset


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, seed=RANDOM_STATE)

    model = NearestCentroid()
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="weighted")
    report_text = classification_report(y_test, test_pred, target_names=CATEGORIES)

    metrics = {
        "model": "NearestCentroid baseline",
        "test_accuracy": float(test_accuracy),
        "weighted_f1": float(test_f1),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    with (REPORTS_DIR / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (REPORTS_DIR / "baseline_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title("Baseline Confusion Matrix - NearestCentroid", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
