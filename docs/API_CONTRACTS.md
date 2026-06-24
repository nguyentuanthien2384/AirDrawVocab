# API_CONTRACTS.md

## Runtime status

```http
GET /admin/pro/status
```

Trả về tình trạng benchmark, registry, model runtime và readiness.

## Build benchmark

```http
POST /admin/benchmark/build
```

Tạo `data/benchmark/release_v1` từ `stroke_samples` trong SQLite.

## Evaluate release

```http
POST /admin/evaluate/run
```

Chạy đánh giá release hiện tại trên benchmark nếu môi trường có TensorFlow.

## Existing gameplay endpoints

- `POST /predict_godmode`: ảnh canvas -> top5 image model + judge + rerank.
- `POST /predict_stroke`: strokes -> top5 stroke model.
- `POST /camera/face-strokes`: frame camera -> derived strokes, không lưu raw face frame.
- `GET /dataset/export`: xuất JSONL/CSV/manifest cho self-improving loop.
- `POST /admin/retrain/start`: train image hoặc stroke ở background.
- `POST /admin/model/reload`: reload model mới.
