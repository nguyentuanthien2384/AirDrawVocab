import os

# Đảm bảo no-op để test an toàn dù máy có cài mlflow hay không
os.environ.setdefault("MLFLOW_DISABLED", "1")

from src.utils import mlflow_utils  # noqa: E402


def test_is_enabled_returns_bool():
    assert isinstance(mlflow_utils.is_enabled(), bool)


def test_graceful_noops_do_not_raise():
    # Khi tắt MLflow, mọi hàm phải an toàn (no-op), không ném lỗi
    run = mlflow_utils.start_mlflow_run(run_name="unit-test")
    mlflow_utils.log_params({"a": 1})
    mlflow_utils.log_metrics({"acc": 0.9})
    mlflow_utils.end_mlflow_run()
    assert run is None or run is not None  # chỉ cần không lỗi
