# Hướng dẫn Training — AirDrawVocab (Phase 1)

Tài liệu này mô tả quy trình train chuẩn sau khi tích hợp **MLflow (experiment
tracking)**, **model versioning** và **reproducibility** ở Phase 1.

> Mục tiêu Phase 1: mọi lần train đều được log đầy đủ (tham số, metric, model,
> môi trường), dễ so sánh giữa các kiến trúc và dễ tái lập.

---

## 1. Cài đặt môi trường

Khuyến nghị Python 3.11/3.12 (64-bit).

```bash
pip install -r requirements.txt
```

MLflow là **tùy chọn**. Nếu không cài `mlflow`, các script train vẫn chạy bình
thường — phần tracking sẽ tự động bị bỏ qua (in cảnh báo ngắn). Để tắt MLflow có
chủ đích:

```bash
# Windows PowerShell
$env:MLFLOW_DISABLED = "1"
# Linux/macOS
export MLFLOW_DISABLED=1
```

---

## 2. Các thành phần Phase 1

| File | Vai trò |
|------|---------|
| `src/utils/mlflow_utils.py` | Tiện ích log MLflow (graceful, tùy chọn) |
| `src/utils/repro.py` | `set_global_seed()` + `collect_environment()` |
| `src/utils/model_versioning.py` | Lưu bản model có version + metadata vào `models/versions/` |
| `src/evaluation/compare_models.py` | So sánh nhiều model trên tập test cố định |

Các script train đã tích hợp sẵn: `src/training/train_clean.py`,
`src/training/train_best.py`, `src/training/advanced_train_model.py`.

---

## 3. Train model

### 3.1. CNN sạch (ổn định, nhanh)

```bash
python src/training/train_clean.py --epochs 16 --train-per-class 2000 --val-per-class 400 --test-per-class 400
```

Test nhanh:

```bash
python src/training/train_clean.py --epochs 2 --train-per-class 300 --val-per-class 100 --test-per-class 100
```

### 3.2. Model zoo hiện đại (ResNet-GN, ConvNeXt, CNN Wide GAP)

```bash
# Train kiến trúc tốt nhất
python src/training/train_best.py --model resnet_gn --per-class 16000 --epochs 50

# Train & so sánh tất cả + ensemble
python src/training/train_best.py --model all --per-class 12000 --epochs 45 --ensemble

# Kiểm thử nhanh, không cần mạng
python src/training/train_best.py --smoke
```

### 3.3. Pipeline nâng cao (ResNet Sketch / EfficientNet / MobileNet)

```bash
python src/training/advanced_train_model.py --model resnet_sketch --epochs 60
python src/training/advanced_train_model.py --model all --epochs 40
# smoke test
python src/training/advanced_train_model.py --smoke-test
```

> `advanced_train_model.py` tạo **một MLflow run cho mỗi model** khi chạy
> `--model all`, nên dễ so sánh các kiến trúc cạnh nhau.

---

## 4. Xem kết quả trên MLflow UI

Sau khi train, mở terminal tại thư mục dự án:

```bash
mlflow ui
```

Mở trình duyệt: <http://127.0.0.1:5000>

Bạn sẽ thấy các experiment:

- `AirDrawVocab_CNN` — từ `train_clean.py`
- `AirDrawVocab_Advanced` — từ `train_best.py`
- `AirDrawVocab_AdvancedModels` — từ `advanced_train_model.py`
- `AirDrawVocab_Comparison` — từ `compare_models.py`

Mỗi run gồm: **params** (hyperparameters + môi trường), **metrics**
(accuracy, top3, f1...), **tags** và **model artifact**.

---

## 5. Model Versioning

Mỗi lần train xong, ngoài model deploy `models/airdrawvocab_best_advanced.keras`,
một bản có version được lưu kèm metadata:

```
models/versions/
├── resnet_gn_v20260622_141530.keras
└── resnet_gn_v20260622_141530.json   # metrics, params, môi trường
```

Liệt kê các version bằng Python:

```python
from model_versioning import list_versions
for v in list_versions():
    print(v["version"], v["base_name"], v["metrics"])
```

> Thư mục `models/versions/` và `mlruns/` đã được thêm vào `.gitignore`.

---

## 6. So sánh model tự động

```bash
# So sánh mọi model trong models/
python src/evaluation/compare_models.py

# Gồm cả các bản version
python src/evaluation/compare_models.py --include-versions

# So sánh danh sách cụ thể
python src/evaluation/compare_models.py --models models/airdrawvocab_best_advanced.keras models/member_resnet_gn.keras
```

Kết quả: bảng xếp hạng in ra console + file CSV trong
`assets/reports/comparisons/`, và (mặc định) log vào MLflow experiment
`AirDrawVocab_Comparison`.

---

## 7. Reproducibility

- Tất cả script đặt seed qua `repro.set_global_seed(RANDOM_STATE)` (mặc định 42,
  cấu hình trong `config.py`).
- Thông tin môi trường (Python, TensorFlow, OS, GPU) được thu thập bằng
  `repro.collect_environment()` và log kèm params trên MLflow.
- Để tái lập chính xác hơn (chậm hơn): `set_global_seed(42, deterministic=True)`.

---

## 8. Phase 2 — Stroke BiGRU & Data Versioning

### 8.1. Đặc trưng stroke dùng chung

`stroke_features.py` (ở thư mục gốc) là **nguồn duy nhất** biến nét vẽ thành
tensor, dùng chung cho cả train (`src/training/train_stroke_model.py`) và
inference (`backend/app.py`) để tránh lệch train/inference.

Bộ đặc trưng nâng cấp (9 chiều/điểm): `x, y, dx, dy, speed, dir_cos, dir_sin,
pen_up, t` — thêm tốc độ, hướng di chuyển và cờ nhấc bút so với bản cũ (5 chiều).

### 8.2. Train Stroke BiGRU (nâng cấp)

```bash
python src/training/train_stroke_model.py --epochs 30
```

Cải tiến: BiGRU 2 lớp lớn hơn + LayerNorm + recurrent dropout, label smoothing,
**class weights** (cân bằng lớp), `ReduceLROnPlateau`, tích hợp MLflow
(experiment `AirDrawVocab_Stroke`) + versioning + **báo cáo đánh giá**
(per-class report + confusion matrix) lưu ở `assets/reports/stroke/`.

> Lưu ý: nếu trước đó đã có `models/stroke_sequence_model.keras` train bằng bộ
> đặc trưng cũ (5 chiều), cần train lại do đầu vào model giờ là 9 chiều.

### 8.3. Data Versioning (manifest)

```bash
python src/data/data_versioning.py            # có SHA1
python src/data/data_versioning.py --no-hash  # nhanh hơn
```

Ghi `data/dataset_manifest.json`: số mẫu/lớp, shape, dtype, SHA1 của từng file
`.npy`, và thống kê `stroke_samples` trong SQLite. So sánh manifest theo thời
gian để biết dữ liệu đã thay đổi thế nào.

### 8.4. Benchmark test set cố định

```bash
python src/data/make_benchmark.py
```

Đóng băng tập test (từ split cố định) thành `data/benchmark/benchmark_test.npz`
+ manifest (SHA1, counts) để MỌI model so sánh trên CÙNG dữ liệu:

```bash
python src/evaluation/compare_models.py --benchmark data/benchmark/benchmark_test.npz
```

### 8.5. Test-Time Augmentation (TTA) tốt hơn

`airdraw_models.predict_tta` nay dùng **lưới dịch tất định** (deterministic shift
grid) thay vì random crop: tái lập được, bao phủ đều các hướng, lấp viền bằng 0
(không tạo nét giả). Dùng trong `train_best.py` và `hard_negative_mining.py`.

### 8.6. Hard Negative Mining (phân tích lỗi)

```bash
python src/evaluation/hard_negative_mining.py --top 150
python src/evaluation/hard_negative_mining.py --benchmark data/benchmark/benchmark_test.npz --tta 6
```

Xuất ra `assets/reports/hard_negative/`: các mẫu sai tự tin nhất (CSV), **cặp lớp
hay nhầm** (vd `leaf -> diamond`), và **lớp recall thấp** — định hướng bổ sung dữ
liệu / oversample.

### 8.7. Tiền xử lý ảnh dùng chung

`image_preprocess.py` (gốc) là nguồn duy nhất cho tiền xử lý ảnh vẽ, dùng chung
bởi `backend/app.py` (production) và các script đánh giá → không lệch
production/evaluation. Hành vi mặc định **giữ nguyên** như backend cũ; có tùy chọn
`deskew=True` (chỉnh nghiêng, mặc định tắt).

---

## 9. Lộ trình tiếp theo

Theo kế hoạch trong Phase 1–3, các hạng mục còn lại:

- **Phase 1 – Task 2** *(ĐÃ XONG)*: Chuẩn hóa cấu trúc thư mục `src/`
  (training/evaluation/data/utils). Các module dùng chung cho web/desktop
  (`config.py`, `vocab_pairs.py`, `data_utils` nguồn, `airdraw_models.py`,
  `ai_assistant.py`, `face_auth.py`, `sample_generator.py`) được giữ ở thư mục
  gốc để không phá vỡ `backend/app.py`, `game.py`, `launcher.py` và deploy.
- **Phase 2** *(ĐÃ XONG)*: ✅ Data versioning (manifest), ✅ cải thiện Stroke BiGRU
  + feature engineering, ✅ evaluation per-class cho stroke, ✅ benchmark dataset cố
  định, ✅ TTA tất định, ✅ hard-negative mining, ✅ tách tiền xử lý ảnh dùng chung.
- **Phase 3**: Automated retraining, model registry, monitoring, testing, CI/CD,
  data drift detection.

### Cấu trúc thư mục sau Task 2

```
src/
├── training/      train_clean, train_best, advanced_train_model, train_model,
│                  baseline_model, train_stroke_model, self_improve_retrain
├── evaluation/    evaluate_model, error_analysis, compare_models
├── data/          data_utils
├── utils/         mlflow_utils, repro, model_versioning
├── inference/     (sẽ bổ sung)
└── models/        (sẽ bổ sung)
```

> Các script đã chuyển vào `src/` tự thêm thư mục gốc dự án vào `sys.path` nên
> vẫn chạy được bằng `python src/training/<script>.py` từ thư mục gốc.
