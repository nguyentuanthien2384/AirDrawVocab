# BENCHMARK_POLICY.md

## Mục tiêu

AirDrawVocab chỉ được xem là nhận dạng ổn định khi model được kiểm tra trên một benchmark cố định, lấy từ dữ liệu người chơi thật và tách riêng khỏi dữ liệu train.

## Quy tắc dữ liệu

- Ground truth là từ được game yêu cầu vẽ, không phải nhãn AI đoán.
- Không dùng raw frame khuôn mặt để train hoặc lưu dài hạn; chỉ dùng stroke/landmark đã chuẩn hóa.
- Split phải cố định bằng seed và manifest.
- Nếu có user_id, không để cùng một người chơi xuất hiện đồng thời trong train và test khi đủ dữ liệu.
- Các mode `mouse`, `camera-hand`, `camera-face`, `face-strokes` phải được thống kê riêng.

## Split chuẩn

- Train: 70%
- Calibration: 15%
- Test: 15%

Khi dữ liệu còn ít, script vẫn tạo split nhưng đánh dấu `data_warning` trong manifest. Không nên promote model dựa trên benchmark quá nhỏ.

## Chỉ số cần xem

- Top-1 accuracy
- Top-3 accuracy
- Macro F1
- Per-class precision / recall / F1
- Confusion matrix
- Expected Calibration Error, nếu có calibration
- p50 / p95 / p99 latency nếu chạy benchmark runtime

## Lệnh

```bash
python src/data/make_real_user_benchmark.py --db data/airdrawvocab_app.sqlite3 --out data/benchmark/release_v1
python src/evaluation/evaluate_release.py --benchmark data/benchmark/release_v1 --out assets/reports/releases/current
```
