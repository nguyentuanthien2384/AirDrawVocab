# Self-improving Loop storage update

Đã gom dữ liệu của khu vực **Self-improving Loop** vào một kho riêng:

```txt
data/self_improving_loop/
```

## Tự lưu khi thao tác

- Khi lưu mẫu vẽ qua `/game/stroke` hoặc nút lưu mẫu train:
  - append vào `samples/stroke_samples.jsonl`
  - lưu chi tiết từng mẫu vào `samples/by_label/<label>/<sample_id>.json`
  - ghi sự kiện vào `actions/events.jsonl`

- Khi bấm **Export data**:
  - ghi bản mới nhất vào `exports/latest/`
  - tạo thêm snapshot không bị ghi đè ở `exports/<timestamp>_export/`
  - ghi sự kiện export vào `actions/events.jsonl`

- Khi bấm **Train stroke** hoặc **Train image**:
  - tạo folder job riêng ở `jobs/<timestamp>_stroke/` hoặc `jobs/<timestamp>_image/`
  - lưu `job.json`
  - ghi log train vào `retrain_stroke.log` hoặc `retrain_image.log` trong folder job đó
  - ghi trạng thái mới nhất vào `status/retrain_status.json`
  - khi job kết thúc, lưu `final_status.json` và snapshot model/categories nếu có

- Khi reload model ảnh mới:
  - ghi sự kiện vào `actions/events.jsonl`

## Endpoint xem nhanh

```txt
GET /admin/self-improve/storage
```

Endpoint này trả về đường dẫn storage và danh sách file mới nhất.
