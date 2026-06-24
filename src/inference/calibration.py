"""Probability calibration helpers for AirDrawVocab.

Uses temperature scaling at inference time. If no calibration JSON exists,
functions are no-ops so gameplay keeps working.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature or 1.0), 1e-6)
    z = np.asarray(logits, dtype="float64") / temp
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype("float32")


def calibrate_probabilities(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to probabilities by converting to log-space."""
    p = np.asarray(probs, dtype="float64")
    p = np.clip(p, 1e-9, 1.0)
    logits = np.log(p)
    return softmax(logits, temperature=temperature)


def expected_calibration_error(confidences: Iterable[float], correct: Iterable[bool], bins: int = 10) -> float:
    conf = np.asarray(list(confidences), dtype="float64")
    corr = np.asarray(list(correct), dtype="float64")
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if not np.any(mask):
            continue
        acc = corr[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(acc - avg_conf)
    return float(ece)


class TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0, labels: list[str] | None = None, metadata: dict | None = None):
        self.temperature = max(float(temperature or 1.0), 1e-6)
        self.labels = labels or []
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path | None) -> "TemperatureCalibrator":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(float(data.get("temperature", 1.0)), data.get("labels") or [], data)
        except Exception:
            return cls()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"temperature": self.temperature, "labels": self.labels, **self.metadata}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def apply(self, probs: np.ndarray) -> np.ndarray:
        return calibrate_probabilities(probs, self.temperature)
