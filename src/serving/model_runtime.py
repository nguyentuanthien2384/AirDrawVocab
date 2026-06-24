"""Model runtime layer for AirDrawVocab.

Purpose:
- Keep model loading, metadata, calibration and hot reload separate from FastAPI handlers.
- Make release/evaluation/promotion easier to reason about.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model
except Exception:  # allows docs/tests without TensorFlow installed
    load_model = None

from src.inference.calibration import TemperatureCalibrator


@dataclass
class LoadedModel:
    branch: str
    model_path: Path
    categories_path: Path
    calibration_path: Optional[Path] = None
    model: Any = None
    categories: List[str] = field(default_factory=list)
    calibrator: TemperatureCalibrator = field(default_factory=TemperatureCalibrator)
    loaded_at: float = 0.0

    def load(self) -> "LoadedModel":
        if load_model is None:
            raise RuntimeError("TensorFlow/Keras is not installed in this environment.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.categories_path.exists():
            raise FileNotFoundError(f"Categories not found: {self.categories_path}")
        self.model = load_model(self.model_path, compile=False)
        self.categories = json.loads(self.categories_path.read_text(encoding="utf-8"))
        self.calibrator = TemperatureCalibrator.load(self.calibration_path)
        self.loaded_at = time.time()
        return self

    def predict(self, x: np.ndarray, calibrate: bool = True) -> Dict[str, Any]:
        if self.model is None:
            self.load()
        raw = np.asarray(self.model.predict(x, verbose=0))
        probs = raw[0] if raw.ndim == 2 else raw
        if calibrate:
            probs = self.calibrator.apply(probs.reshape(1, -1))[0]
        k = min(5, len(probs), len(self.categories))
        idx = np.argsort(probs)[::-1][:k]
        top5 = [{"label": self.categories[int(i)], "confidence": float(probs[int(i)])} for i in idx]
        return {
            "label": top5[0]["label"] if top5 else "unknown",
            "confidence": top5[0]["confidence"] if top5 else 0.0,
            "top5": top5,
            "branch": self.branch,
            "model_version": self.version_info(),
        }

    def version_info(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "model_path": str(self.model_path),
            "categories_path": str(self.categories_path),
            "calibration_path": str(self.calibration_path) if self.calibration_path else None,
            "num_categories": len(self.categories),
            "loaded_at": self.loaded_at,
        }


class ModelRuntimeManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._models: Dict[str, LoadedModel] = {}

    def register(self, loaded: LoadedModel, load_now: bool = False) -> None:
        with self._lock:
            if load_now:
                loaded.load()
            self._models[loaded.branch] = loaded

    def reload(self, branch: str, model_path: str | Path, categories_path: str | Path,
               calibration_path: str | Path | None = None) -> Dict[str, Any]:
        loaded = LoadedModel(branch, Path(model_path), Path(categories_path), Path(calibration_path) if calibration_path else None)
        loaded.load()
        with self._lock:
            self._models[branch] = loaded
        return loaded.version_info()

    def predict(self, branch: str, x: np.ndarray, calibrate: bool = True) -> Dict[str, Any]:
        with self._lock:
            loaded = self._models.get(branch)
        if loaded is None:
            raise KeyError(f"No model registered for branch {branch}")
        return loaded.predict(x, calibrate=calibrate)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {branch: loaded.version_info() for branch, loaded in self._models.items()}


RUNTIME = ModelRuntimeManager()
