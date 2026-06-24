"""Evaluate current or candidate AirDrawVocab release on a fixed benchmark."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import MODEL_PATH, MODELS_DIR
from src.inference.calibration import expected_calibration_error

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

CANVAS_W = 960
CANVAS_H = 540


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strokes_to_bitmap(strokes: Any, size: int) -> np.ndarray:
    img = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    draw = ImageDraw.Draw(img)
    if isinstance(strokes, str):
        try:
            strokes = json.loads(strokes)
        except Exception:
            strokes = []
    if isinstance(strokes, list):
        for stroke in strokes:
            if not isinstance(stroke, list) or len(stroke) < 2:
                continue
            pts = [(float(p.get("x", 0)), float(p.get("y", 0))) for p in stroke if isinstance(p, dict)]
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=18, joint="curve")
    arr = np.asarray(img)
    coords = cv2.findNonZero((arr > 10).astype("uint8") * 255)
    if coords is None:
        return np.zeros((size, size), dtype="float32")
    x, y, w, h = cv2.boundingRect(coords)
    crop = arr[y:y+h, x:x+w]
    box = int(size * 0.78)
    scale = box / max(w, h, 1)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype="uint8")
    xs, ys = (size - nw) // 2, (size - nh) // 2
    canvas[ys:ys+nh, xs:xs+nw] = resized
    return canvas.astype("float32") / 255.0


def model_input_spec(model) -> tuple[int, int]:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    size = int(shape[1] or 28) if shape and len(shape) >= 3 else 28
    channels = int(shape[-1] or 1) if shape else 1
    return size, channels


def prepare_x(rows: list[dict], size: int, channels: int) -> np.ndarray:
    xs = []
    for row in rows:
        img = strokes_to_bitmap(row.get("strokes", []), size=size)
        if channels == 3:
            xs.append(np.repeat(img[..., None], 3, axis=-1))
        else:
            xs.append(img[..., None])
    return np.stack(xs).astype("float32") if xs else np.empty((0, size, size, channels), dtype="float32")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark" / "release_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--categories", type=Path, default=MODELS_DIR / "categories.json")
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "reports" / "releases" / "current")
    args = parser.parse_args()

    if load_model is None:
        raise RuntimeError("TensorFlow is not installed.")
    rows = load_jsonl(args.benchmark / f"{args.split}.jsonl")
    if not rows:
        raise RuntimeError(f"No benchmark rows found at {args.benchmark}/{args.split}.jsonl")
    model = load_model(args.model, compile=False)
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    label_to_idx = {label: i for i, label in enumerate(categories)}
    valid = [r for r in rows if r.get("target") in label_to_idx]
    if not valid:
        raise RuntimeError("Benchmark labels do not match model categories.")
    size, channels = model_input_spec(model)
    x = prepare_x(valid, size, channels)
    y_true = np.asarray([label_to_idx[r["target"]] for r in valid], dtype="int64")
    start = time.perf_counter()
    probs = np.asarray(model.predict(x, verbose=0))
    elapsed = time.perf_counter() - start
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    top3_idx = np.argsort(probs, axis=1)[:, -min(3, probs.shape[1]):]
    top3 = np.asarray([t in row for t, row in zip(y_true, top3_idx)])
    summary: Dict[str, Any] = {
        "benchmark": str(args.benchmark),
        "split": args.split,
        "model": str(args.model),
        "categories": str(args.categories),
        "samples": int(len(valid)),
        "classes": int(len(set(y_true.tolist()))),
        "top1_accuracy": float(correct.mean()),
        "top3_accuracy": float(top3.mean()),
        "avg_confidence": float(conf.mean()),
        "ece_10_bins": expected_calibration_error(conf, correct, bins=10),
        "latency_total_seconds": float(elapsed),
        "latency_ms_per_sample_avg": float((elapsed / max(1, len(valid))) * 1000),
    }
    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        labels = sorted(set(y_true.tolist()) | set(pred.tolist()))
        target_names = [categories[i] for i in labels]
        summary["macro_f1"] = float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0))
        report = classification_report(y_true, pred, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        cm = confusion_matrix(y_true, pred, labels=labels)
        np.savetxt(args.out / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    except Exception as exc:
        summary["metrics_warning"] = str(exc)
    confusions = []
    for row, yi, pi, ci in zip(valid, y_true, pred, conf):
        if yi != pi:
            confusions.append({"id": row.get("id"), "target": categories[int(yi)], "predicted": categories[int(pi)], "confidence": float(ci), "mode": row.get("mode")})
    summary["confusions"] = confusions[:50]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
