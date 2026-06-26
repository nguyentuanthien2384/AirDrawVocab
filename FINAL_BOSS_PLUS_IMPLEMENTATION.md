# AirDrawVocab Final Boss Plus Implementation

Bản này tích hợp nốt các phần còn thiếu trong kiến trúc Final Boss:

## 1. Self-improving loop / retrain

Đã thêm:

- `GET /dataset/export`: export toàn bộ stroke_samples sang `data/self_improving_loop/exports/latest/stroke_samples.jsonl`.
- `GET /dataset/download/stroke_samples.jsonl`: tải dataset export.
- `POST /admin/retrain/start`: chạy retrain local bằng subprocess.
- `GET /admin/retrain/status`: xem trạng thái retrain.
- `self_improve_retrain.py`: rasterize stroke thành ảnh 28x28, trộn với QuickDraw `.npy` nếu có, train image CNN mới.
- `train_stroke_model.py`: train model chuỗi nét vẽ riêng bằng GRU.

Ghi chú: train thật vẫn nên chạy trên Colab/GPU nếu dữ liệu lớn. Local endpoint dùng cho demo/pipeline tự động.

## 2. Multiplayer / PvP / WebSocket

Đã thêm:

- `WebSocket /ws/pvp/{room}`.
- Room manager in-memory.
- Frontend panel `PvP WebSocket`: join/leave room, broadcast score/prediction/system event.

Đây là PvP sync layer mức localhost/demo. Muốn public nhiều người cần deploy server có WebSocket và persistent room store.

## 3. Stroke-based deep learning model riêng

Đã thêm:

- `POST /predict_stroke`: nhận `strokes_json`, dùng `models/stroke_sequence_model.keras` nếu đã train.
- `train_stroke_model.py`: train GRU/BiGRU model từ dữ liệu stroke đã lưu.
- Frontend realtime AI tự thử stroke model. Nếu stroke model có confidence cao hơn image CNN, UI hiển thị source là `stroke-sequence`.

Nếu chưa train stroke model, hệ thống fallback sang image CNN bình thường.

## 4. Production deploy / Docker / cloud

Đã thêm:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `Procfile`
- `render.yaml`

Chạy Docker local:

```bash
docker compose up --build
```

Mở:

```text
http://127.0.0.1:8000
```

## 5. Luồng dùng khuyến nghị

1. Chơi game và lưu stroke samples.
2. Export dataset bằng UI hoặc API.
3. Train stroke model:

```bash
python src/training/train_stroke_model.py --epochs 12
```

4. Hoặc train image self-improved model:

```bash
python src/training/self_improve_retrain.py --epochs 10
```

5. Copy/đổi model mới vào `models/` theo pipeline bạn chọn.

