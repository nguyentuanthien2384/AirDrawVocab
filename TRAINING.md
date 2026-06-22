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
| `mlflow_utils.py` | Tiện ích log MLflow (graceful, tùy chọn) |
| `repro.py` | `set_global_seed()` + `collect_environment()` |
| `model_versioning.py` | Lưu bản model có version + metadata vào `models/versions/` |
| `compare_models.py` | So sánh nhiều model trên tập test cố định |

Các script train đã tích hợp sẵn: `train_clean.py`, `train_best.py`,
`advanced_train_model.py`.

---

## 3. Train model

### 3.1. CNN sạch (ổn định, nhanh)

```bash
python train_clean.py --epochs 16 --train-per-class 2000 --val-per-class 400 --test-per-class 400
```

Test nhanh:

```bash
python train_clean.py --epochs 2 --train-per-class 300 --val-per-class 100 --test-per-class 100
```

### 3.2. Model zoo hiện đại (ResNet-GN, ConvNeXt, CNN Wide GAP)

```bash
# Train kiến trúc tốt nhất
python train_best.py --model resnet_gn --per-class 16000 --epochs 50

# Train & so sánh tất cả + ensemble
python train_best.py --model all --per-class 12000 --epochs 45 --ensemble

# Kiểm thử nhanh, không cần mạng
python train_best.py --smoke
```

### 3.3. Pipeline nâng cao (ResNet Sketch / EfficientNet / MobileNet)

```bash
python advanced_train_model.py --model resnet_sketch --epochs 60
python advanced_train_model.py --model all --epochs 40
# smoke test
python advanced_train_model.py --smoke-test
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
python compare_models.py

# Gồm cả các bản version
python compare_models.py --include-versions

# So sánh danh sách cụ thể
python compare_models.py --models models/airdrawvocab_best_advanced.keras models/member_resnet_gn.keras
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

## 8. Lộ trình tiếp theo (chưa làm)

Theo kế hoạch trong Phase 1–3, các hạng mục còn lại:

- **Phase 1 – Task 2**: Chuẩn hóa cấu trúc thư mục `src/` (training/inference/
  data/evaluation/utils). *Chưa thực hiện* vì cần cập nhật đồng loạt import của
  `backend/app.py`, `game.py`, `launcher.py`... — nên làm trong một bước riêng.
- **Phase 2**: Data versioning, cải thiện Stroke BiGRU, evaluation pipeline mạnh,
  TTA tốt hơn, hard-negative mining.
- **Phase 3**: Automated retraining, model registry, monitoring, testing, CI/CD,
  data drift detection.
