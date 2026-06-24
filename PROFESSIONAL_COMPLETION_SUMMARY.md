# Professional Completion Summary

## Mục tiêu bản này

Nâng AirDrawVocab từ demo/game có self-improving loop thành project có pipeline deep learning thực tế hơn: có benchmark cố định, train candidate, evaluate, calibration, promotion gate, runtime layer, metrics và UI Production AI Ops.

## Những phần đã phát triển trong code

### 1. Benchmark thật từ người chơi

File mới:

- `src/data/make_real_user_benchmark.py`
- `configs/benchmark.yaml`
- `docs/BENCHMARK_POLICY.md`

Lệnh:

```bash
python src/data/make_real_user_benchmark.py --db data/airdrawvocab_app.sqlite3 --out data/benchmark/release_v1
```

Output:

- `data/benchmark/release_v1/train.jsonl`
- `data/benchmark/release_v1/calibration.jsonl`
- `data/benchmark/release_v1/test.jsonl`
- `data/benchmark/release_v1/manifest.json`

### 2. Train image model chuẩn candidate

File mới:

- `src/training/train_image_model.py`
- `configs/image_resnet_sketch.yaml`
- `configs/image_mobilenetv2.yaml`

Lệnh an toàn local/offline:

```bash
python src/training/train_image_model.py --config configs/image_resnet_sketch.yaml
```

Lệnh mạnh hơn nếu máy có internet/GPU để tải ImageNet weights:

```bash
python src/training/train_image_model.py --config configs/image_mobilenetv2.yaml
```

Output:

- `models/image_cnn_candidate.keras`
- `models/categories_candidate.json`
- `models/airdrawvocab_self_improved.keras`
- `models/categories_self_improved.json`

### 3. Stroke transformer challenger

File mới:

- `src/training/train_stroke_transformer.py`
- `configs/stroke_transformer.yaml`

Lệnh:

```bash
python src/training/train_stroke_transformer.py --epochs 20
```

Output:

- `models/stroke_transformer_model.keras`
- `models/stroke_transformer_categories.json`

### 4. Evaluation release

File mới:

- `src/evaluation/evaluate_release.py`

Lệnh:

```bash
python src/evaluation/evaluate_release.py --benchmark data/benchmark/release_v1 --out assets/reports/releases/current
```

Output:

- `assets/reports/releases/current/summary.json`
- `assets/reports/releases/current/classification_report.json`
- `assets/reports/releases/current/confusion_matrix.csv`

### 5. Calibration

File mới:

- `src/evaluation/calibrate_release.py`
- `src/inference/calibration.py`

Lệnh:

```bash
python src/evaluation/calibrate_release.py --benchmark data/benchmark/release_v1
```

Output:

- `models/calibration/image_temperature.json`

### 6. Runtime, TTA, reranker, metrics

File mới:

- `src/serving/model_runtime.py`
- `src/serving/ab_router.py`
- `src/inference/tta.py`
- `src/inference/reranker.py`
- `src/monitoring/metrics.py`

### 7. Promotion gate

File mới:

- `src/training/promote_candidate.py`
- `docs/PROMOTION_POLICY.md`

Lệnh kiểm tra không thay model:

```bash
python src/training/promote_candidate.py --dry-run --allow-if-weak-data
```

Lệnh promote thật sau khi evaluate đạt gate:

```bash
python src/training/promote_candidate.py --report assets/reports/releases/current/summary.json
```

### 8. Backend endpoint mới

Đã thêm vào `backend/app.py`:

- `GET /admin/pro/status`
- `POST /admin/benchmark/build`
- `POST /admin/evaluate/run`
- `GET /admin/promote/status`
- `POST /admin/promote/dry-run`
- `GET /metrics`

### 9. Frontend UI mới

Đã thêm khối:

- `Production AI Ops`
- nút `Build benchmark`
- nút `Evaluate`
- nút `Promote check`

Nó hiển thị benchmark, eval release, calibration, readiness và promotion.

## Kiểm tra đã chạy

```bash
python -m py_compile backend/app.py image_preprocess.py stroke_features.py camera_face_strokes.py
python -m py_compile src/data/make_real_user_benchmark.py src/training/train_image_model.py src/training/train_stroke_transformer.py src/evaluation/evaluate_release.py src/evaluation/calibrate_release.py src/training/promote_candidate.py src/serving/model_runtime.py src/monitoring/metrics.py
node --check frontend/app.js
python src/data/make_real_user_benchmark.py --db data/airdrawvocab_app.sqlite3 --out data/benchmark/release_v1
```

## Cảnh báo thực tế

Bản code đã tiến gần hơn cấu trúc production, nhưng độ chính xác chuyên nghiệp vẫn phụ thuộc dữ liệu. Benchmark hiện tại trong source chỉ có số mẫu nhỏ nên manifest sẽ báo `data_warning`. Muốn giảm nhầm `book/door/pants` ổn định, cần lưu thêm mẫu đúng và sai có chủ đích cho các cặp đó.
