"""
Smoke tests for AirDrawVocab.

Run after setup/training:
    python smoke_test_project.py
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app


ROOT = Path(__file__).resolve().parent


def make_sample_png(category: str = "apple") -> bytes:
    data_path = ROOT / "data" / "npy_28" / f"{category}.npy"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    arr = np.load(data_path)[0].reshape(28, 28).astype("uint8")
    image = Image.fromarray(arr, "L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200, health.text
    health_data = health.json()
    assert health_data["status"] == "ok"
    assert health_data["num_categories"] == 19
    assert "airdrawvocab" in health_data["model_path"].lower()
    print(f"[OK] Health endpoint, model={health_data['model_path']}")

    image_bytes = make_sample_png("apple")
    response = client.post("/predict", files={"file": ("apple.png", image_bytes, "image/png")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["label"] in health_data["categories"]
    assert result["enhanced_drawing"].startswith("data:image/png;base64,")
    print(f"[OK] Predict endpoint, label={result['label']}, confidence={result['confidence_percent']}%")

    gen = client.post("/image/generate", data={"label": result["label"]})
    assert gen.status_code == 200, gen.text
    gen_data = gen.json()
    assert gen_data["image"].startswith("data:image/png;base64,")
    print(f"[OK] Image reference endpoint, provider={gen_data['provider']}")

    preview = client.post("/image/reference", data={"label": result["label"]})
    assert preview.status_code == 200, preview.text
    preview_data = preview.json()
    assert preview_data["provider"] == "offline-pil-reference"
    assert preview_data["image"].startswith("data:image/png;base64,")
    print(f"[OK] Realtime preview endpoint, label={preview_data['label']}")

    print("[OK] Smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
