"""Fit a scalar temperature for a release model on calibration split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import MODEL_PATH, MODELS_DIR
from src.evaluation.evaluate_release import load_jsonl, model_input_spec, prepare_x
from src.inference.calibration import TemperatureCalibrator, calibrate_probabilities, expected_calibration_error

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0)
    return float(-np.mean(np.log(p)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark" / "release_v1")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--categories", type=Path, default=MODELS_DIR / "categories.json")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "calibration" / "image_temperature.json")
    args = parser.parse_args()
    if load_model is None:
        raise RuntimeError("TensorFlow is not installed.")
    rows = load_jsonl(args.benchmark / "calibration.jsonl")
    if not rows:
        raise RuntimeError("No calibration split found. Run make_real_user_benchmark first.")
    model = load_model(args.model, compile=False)
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    label_to_idx = {label: i for i, label in enumerate(categories)}
    valid = [r for r in rows if r.get("target") in label_to_idx]
    size, channels = model_input_spec(model)
    x = prepare_x(valid, size, channels)
    y = np.asarray([label_to_idx[r["target"]] for r in valid], dtype="int64")
    probs = np.asarray(model.predict(x, verbose=0))
    best_t, best_loss = 1.0, nll(probs, y)
    for t in np.linspace(0.5, 4.0, 71):
        cal = calibrate_probabilities(probs, float(t))
        loss = nll(cal, y)
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    before = expected_calibration_error(probs.max(axis=1), probs.argmax(axis=1) == y)
    cal_probs = calibrate_probabilities(probs, best_t)
    after = expected_calibration_error(cal_probs.max(axis=1), cal_probs.argmax(axis=1) == y)
    payload = {"temperature": best_t, "nll_before": nll(probs, y), "nll_after": best_loss, "ece_before": before, "ece_after": after, "samples": int(len(valid)), "labels": categories}
    TemperatureCalibrator(best_t, labels=categories, metadata=payload).save(args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
