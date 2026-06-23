# MLOps & Production (Phase 3) — AirDrawVocab

Tài liệu này mô tả các thành phần MLOps được bổ sung ở **Phase 3**: kiểm thử ML,
model registry, train lại tự động, giám sát (monitoring), phát hiện data drift và
CI. Phase 1 (experiment tracking, versioning, reproducibility) và Phase 2 (feature
chung, benchmark, hard-negative mining) xem trong `TRAINING.md`.

> Mọi lệnh dưới đây chạy từ **thư mục gốc dự án**.

---

## 1. Unit & Integration Test cho ML (Task 19)

Bộ test nằm trong `tests/`, chạy bằng `pytest`:

```bash
pip install pytest
python -m pytest tests/ -v
```

Bao phủ các module logic thuần (không cần TensorFlow/OpenCV):

| File test | Module được kiểm thử |
|---|---|
| `test_stroke_features.py` | `stroke_features.py` (shape, xác định, pen-up, chuẩn hóa) |
| `test_repro.py` | `src/utils/repro.py` (seed lặp lại, môi trường) |
| `test_model_versioning.py` | `src/utils/model_versioning.py` (lưu/liệt kê version) |
| `test_model_registry.py` | `src/utils/model_registry.py` (registry, best, promote) |
| `test_data_versioning.py` | `src/data/data_versioning.py` (quét npy + DB) |
| `test_monitor.py` | `src/evaluation/monitor.py` (tổng hợp metric) |
| `test_data_drift.py` | `src/evaluation/data_drift.py` (PSI) |
| `test_auto_retrain.py` | `src/training/auto_retrain.py` (quyết định train) |
| `test_mlflow_utils.py` | `src/utils/mlflow_utils.py` (no-op an toàn) |
| `test_image_preprocess.py` | `image_preprocess.py` (*skip nếu thiếu OpenCV*) |
| `test_tta.py` | `airdraw_models.py` TTA (*skip nếu thiếu TensorFlow*) |

Các test cần thư viện nặng dùng `pytest.importorskip`, nên vẫn an toàn khi chạy ở
môi trường tối giản (CI).

---

## 2. Model Registry (Task 17)

`src/utils/model_registry.py` tổng hợp các bản model đã lưu version (bởi
`model_versioning.save_versioned_model`) trong `models/versions/` và hỗ trợ
"promote" một version lên làm model deploy chính.

```bash
# Quét versions -> ghi models/registry.json
python src/utils/model_registry.py rebuild

# Liệt kê các version (kèm metric)
python src/utils/model_registry.py list
python src/utils/model_registry.py list --base stroke_bigru

# Tìm version tốt nhất theo metric
python src/utils/model_registry.py best --metric val_accuracy

# Promote một version lên model deploy
python src/utils/model_registry.py promote --version v20260622_141530
python src/utils/model_registry.py promote --path models/versions/xxx.keras
```

---

## 3. Train lại tự động (Task 16)

`src/training/auto_retrain.py` kiểm tra số mẫu stroke người dùng đã thu thập trong
SQLite. Nếu số mẫu **mới** (so với lần train trước, lưu ở
`data/auto_retrain_state.json`) đạt ngưỡng, sẽ kích hoạt train lại rồi cập nhật
registry.

```bash
# Kiểm tra điều kiện, KHÔNG train
python src/training/auto_retrain.py --dry-run --min-new-samples 20

# Train lại model stroke khi có >= 20 mẫu mới
python src/training/auto_retrain.py --mode stroke --min-new-samples 20 --epochs 30
```

Có thể đặt vào Task Scheduler/cron để chạy định kỳ.

---

## 4. Monitoring model thực tế (Task 18)

`src/evaluation/monitor.py` đọc bảng `stroke_samples` (backend lưu mỗi lượt chơi:
target/predicted/confidence/correct/mode/created_at) và báo cáo accuracy thực tế,
confidence trung bình, tỉ lệ low-confidence, accuracy theo lớp/mode, và cảnh báo
"sai nhưng tự tin cao".

```bash
python src/evaluation/monitor.py
python src/evaluation/monitor.py --days 7 --out assets/reports/monitor.json
```

Đây là nguồn dữ liệu để quyết định cải tiến model (lớp nào accuracy thấp → ưu tiên
thu thập thêm dữ liệu / hard-negative mining).

---

## 5. Phát hiện Data Drift (Task 21)

`src/evaluation/data_drift.py` tính **PSI (Population Stability Index)** trên phân
phối lớp để phát hiện dữ liệu người dùng đang "trôi" so với baseline.

```bash
# So sánh nửa cũ vs nửa mới của stroke DB (theo thời gian)
python src/evaluation/data_drift.py --db

# So sánh 2 manifest dataset (tạo bởi data_versioning.py)
python src/evaluation/data_drift.py --baseline old_manifest.json --current new_manifest.json
```

Diễn giải PSI: `< 0.1` ổn định · `0.1–0.25` thay đổi vừa · `> 0.25` trôi mạnh.

---

## 6. CI (Task 20)

`.github/workflows/ci.yml` chạy bộ test trên Python 3.11 & 3.12 mỗi khi push/PR vào
`main`/`master`. CI chỉ cài deps gọn nhẹ (`numpy`, `pytest`, `pillow`); các test cần
TensorFlow/OpenCV tự động skip nên pipeline nhanh và ổn định.

---

## Tổng kết vòng đời MLOps

```
Thu thập dữ liệu (backend -> SQLite)
        │
        ▼
data_versioning.py / make_benchmark.py   (versioning dữ liệu, Phase 2)
        │
        ▼
auto_retrain.py ──► train_*.py (MLflow + versioning, Phase 1)
        │                     │
        │                     ▼
        │            model_registry.py  (promote bản tốt nhất)
        ▼                     │
monitor.py / data_drift.py ◄──┘   (giám sát thực tế -> phản hồi)
```
