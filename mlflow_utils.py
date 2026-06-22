"""
mlflow_utils.py — Tiện ích logging MLflow cho AirDrawVocab (Phase 1, Task 1).

MLflow là TÙY CHỌN. Nếu chưa cài gói `mlflow` hoặc đặt biến môi trường
`MLFLOW_DISABLED=1`, mọi hàm dưới đây sẽ trở thành no-op (không làm gì) và in
một cảnh báo ngắn. Nhờ vậy các script train cũ vẫn chạy bình thường ngay cả khi
máy chưa cài MLflow.

Cách dùng nhanh trong script train:

    from mlflow_utils import (
        start_mlflow_run, log_params, log_metrics, log_model,
        end_mlflow_run, log_artifact, log_training_artifacts,
    )

    start_mlflow_run("AirDrawVocab_CNN", run_name="clean_cnn", tags={...})
    log_params({...})
    ...  # train
    log_metrics({...})
    log_model(model, "airdraw_clean_cnn")
    end_mlflow_run()

Xem kết quả:  mlflow ui   ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Phát hiện MLflow một cách an toàn
# ---------------------------------------------------------------------------
_DISABLED = os.environ.get("MLFLOW_DISABLED", "").strip().lower() in {"1", "true", "yes"}

mlflow = None
MLFLOW_AVAILABLE = False
_HAS_KERAS_FLAVOR = False
_REASON = ""

if _DISABLED:
    _REASON = "MLFLOW_DISABLED=1"
else:
    try:
        import mlflow as _mlflow  # type: ignore

        mlflow = _mlflow
        MLFLOW_AVAILABLE = True
        try:
            import mlflow.keras  # noqa: F401
            _HAS_KERAS_FLAVOR = True
        except Exception:
            _HAS_KERAS_FLAVOR = False
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        _REASON = f"không import được mlflow ({exc})"

_warned = False


def _warn_once() -> None:
    global _warned
    if not _warned:
        print(f"[MLflow] Bỏ qua experiment tracking ({_REASON or 'không khả dụng'}). "
              f"Cài bằng `pip install mlflow` để bật.")
        _warned = True


def is_enabled() -> bool:
    """True nếu MLflow đang hoạt động."""
    return MLFLOW_AVAILABLE


# ---------------------------------------------------------------------------
# API chính (giữ tên tương thích với hướng dẫn trong tài liệu)
# ---------------------------------------------------------------------------
def start_mlflow_run(experiment_name: str = "AirDrawVocab",
                     run_name: str | None = None,
                     tags: dict | None = None):
    """Bắt đầu một MLflow run. Trả về run hoặc None nếu MLflow tắt."""
    if not MLFLOW_AVAILABLE:
        _warn_once()
        return None
    mlflow.set_experiment(experiment_name)
    if run_name is None:
        run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run = mlflow.start_run(run_name=run_name)
    if tags:
        # MLflow yêu cầu value là str
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
    print(f"[MLflow] Bắt đầu run '{run_name}' (experiment='{experiment_name}').")
    return run


def log_params(params: dict) -> None:
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_params({k: (v if v is not None else "None") for k, v in params.items()})
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] Warning: không log được params - {exc}")


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if not MLFLOW_AVAILABLE:
        return
    try:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] Warning: không log được metrics - {exc}")


def log_model(model, model_name: str = "model") -> None:
    """Log model Keras. Nếu không có flavor keras thì bỏ qua an toàn."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        if _HAS_KERAS_FLAVOR:
            try:
                mlflow.keras.log_model(model, name=model_name)
            except TypeError:
                # API cũ dùng artifact_path
                mlflow.keras.log_model(model, artifact_path=model_name)
        else:
            print("[MLflow] Bỏ qua log_model (thiếu flavor mlflow.keras).")
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] Warning: không log được model - {exc}")


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log một file hoặc thư mục (ảnh, report, csv...)."""
    if not MLFLOW_AVAILABLE:
        return
    if not Path(local_path).exists():
        return
    try:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] Warning: không log được artifact {local_path} - {exc}")


def log_dict(data: dict, artifact_file: str) -> None:
    """Log một dict thành file JSON/YAML trong artifacts."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_dict(data, artifact_file)
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] Warning: không log được dict - {exc}")


def log_training_artifacts(history, results_dir: Path | str) -> None:
    """Log các artifact phổ biến sau khi train (history JSON + ảnh PNG)."""
    if not MLFLOW_AVAILABLE:
        return
    results_dir = Path(results_dir)
    try:
        if history is not None and hasattr(history, "history"):
            log_dict(history.history, "training_history.json")
        if results_dir.exists():
            for img_file in results_dir.glob("*.png"):
                log_artifact(str(img_file), artifact_path="plots")
    except Exception as exc:  # pragma: no cover
        print(f"[MLflow] Warning: không log được training artifacts - {exc}")


def end_mlflow_run() -> None:
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.end_run()
    except Exception:
        pass


# Alias ngắn gọn (tiện dùng)
start_run = start_mlflow_run
end_run = end_mlflow_run
