# AirDrawVocab - Production Deep Learning Upgrade

Bản này đã bổ sung lớp phát triển theo hướng nhận dạng chuyên nghiệp hơn:

## Đã thêm

1. Benchmark người chơi thật: `src/data/make_real_user_benchmark.py`.
2. Pipeline train image chuẩn hơn: `src/training/train_image_model.py` với `resnet_sketch` và `mobilenetv2`.
3. Pipeline stroke transformer challenger: `src/training/train_stroke_transformer.py`.
4. Evaluation release: `src/evaluation/evaluate_release.py`.
5. Calibration: `src/evaluation/calibrate_release.py`, `src/inference/calibration.py`.
6. Model runtime layer: `src/serving/model_runtime.py`.
7. Reranker tách riêng: `src/inference/reranker.py`.
8. TTA tách riêng: `src/inference/tta.py`.
9. Promotion gate: `src/training/promote_candidate.py`.
10. Monitoring metrics fallback: `src/monitoring/metrics.py`.
11. Backend admin endpoints: `/admin/pro/status`, `/admin/benchmark/build`, `/admin/evaluate/run`, `/admin/promote/status`, `/metrics`.
12. Frontend thêm khối `Production AI Ops` để xem trạng thái benchmark/eval/promotion ngay trong game.

## Điều quan trọng

Code pipeline đã hoàn chỉnh hơn, nhưng để đạt chất lượng thực tế chuyên nghiệp, bạn vẫn cần thu thập đủ dữ liệu người chơi thật. Mốc khuyến nghị:

- Ít nhất 50-100 mẫu sạch mỗi lớp để benchmark.
- 150+ mẫu mỗi lớp để train stroke ổn định.
- 300+ mẫu mỗi lớp nếu muốn giảm nhầm lẫn mạnh ở nhiều kiểu vẽ khác nhau.

## Lệnh chạy đề xuất

```bash
python src/data/make_real_user_benchmark.py --out data/benchmark/release_v1
python src/training/train_image_model.py --config configs/image_mobilenetv2.yaml
python src/training/train_stroke_model.py --config configs/stroke_bigru.yaml
python src/evaluation/evaluate_release.py --benchmark data/benchmark/release_v1 --out assets/reports/releases/current
python src/training/promote_candidate.py --report assets/reports/releases/current/summary.json --allow-if-weak-data
```

## Khi nào nhận dạng sẽ tốt hơn rõ rệt?

- Sau khi bạn lưu thêm mẫu train đúng cho các từ yếu trong `Skill Profile`.
- Sau khi `Export data` -> `Train stroke` -> `Train image` -> `Evaluate release` đạt gate.
- Sau khi benchmark có đủ hard negatives, ví dụ nhiều mẫu `book` bị nhầm `door/pants` để model học đúng đặc điểm phân biệt.
