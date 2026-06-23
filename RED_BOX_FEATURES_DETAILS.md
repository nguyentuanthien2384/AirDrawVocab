# Chi tiết các chức năng khoanh đỏ sau khi chơi game

Tài liệu này mô tả nhiệm vụ của các khối chức năng được khoanh đỏ trên giao diện `Final Boss Game Arena` và luồng dữ liệu tương ứng trong source code.

## 1. Lưu mẫu train

Nút `Lưu mẫu train` dùng để lưu lại nét vẽ hiện tại vào SQLite để hệ thống học từ chính dữ liệu người chơi. Mỗi mẫu lưu gồm:

- `target`: từ cần vẽ, ví dụ `book`.
- `predicted`: nhãn AI đang đoán, ví dụ `door`.
- `confidence`: độ tin cậy của AI.
- `correct`: AI đoán đúng hay sai mục tiêu.
- `mode`: vẽ chuột hoặc camera.
- `strokes_json`: toàn bộ chuỗi điểm vẽ theo thời gian.
- `judge_json`: điểm Shape, Clarity, Stroke, Speed và feedback.
- `manual`: mẫu do người dùng bấm lưu thủ công hay do game tự lưu sau pass/fail.
- `point_count`: tổng số điểm stroke, dùng để lọc mẫu quá ít nét.

API xử lý: `POST /game/stroke`.
Bảng dữ liệu: `stroke_samples`.

## 2. Realtime AI

Khối `Realtime AI` chạy khi người chơi đang vẽ. Frontend chụp canvas định kỳ, gửi ảnh lên backend, backend tiền xử lý ảnh 28x28 và cho model CNN dự đoán top-5 nhãn.

Luồng chính:

```text
Canvas hiện tại
→ /predict_godmode
→ image_preprocess.preprocess_drawing
→ Keras CNN predict
→ top5 + confidence + target match
→ frontend hiển thị thanh xác suất
```

Nếu đã có model stroke-sequence, frontend còn gọi `POST /predict_stroke`. Nếu stroke model tự tin hơn CNN ảnh, giao diện sẽ chuyển `AI source` sang `stroke-sequence`.

## 3. AI Judge Mode

`AI Judge Mode` đánh giá chất lượng lượt vẽ theo 4 nhóm điểm:

- `Shape`: tổng hợp độ rõ hình, số nét và tốc độ.
- `Clarity`: dựa trên confidence của model.
- `Stroke`: dựa trên số điểm/nét đã vẽ.
- `Speed`: vẽ nhanh được điểm cao hơn, quá lâu sẽ giảm điểm.

Backend trả về `grade` từ S/A/B/C/D và feedback. Ví dụ khi mục tiêu là `book` nhưng AI nghiêng về `door`, feedback sẽ nhắc thêm gáy sách, đường trang sách để phân biệt.

API xử lý: `POST /predict_godmode`.
Hàm chính: `_judge_payload()`, `_teacher_feedback()`.

## 4. Skill Profile

`Skill Profile` là hồ sơ kỹ năng cá nhân. Sau khi game kết thúc, frontend gọi lại profile để cập nhật:

- số lượt chơi,
- điểm cao nhất,
- tổng số mẫu vẽ đã lưu,
- accuracy tổng,
- best streak,
- confidence trung bình,
- từ mạnh,
- từ yếu,
- practice plan,
- tình trạng sẵn sàng train.

API xử lý: `GET /game/profile`.
Bảng dữ liệu: `game_sessions`, `stroke_samples`.

## 5. Leaderboard

`Leaderboard` lấy điểm cao nhất của mỗi user từ SQLite và xếp hạng theo:

1. score cao nhất,
2. streak cao nhất,
3. level cao nhất.

API xử lý: `GET /game/leaderboard`.
Bảng dữ liệu: `game_sessions`.

## 6. PvP WebSocket

`PvP WebSocket` cho nhiều người vào cùng một phòng, xem điểm và trạng thái của nhau realtime. Mỗi người gửi event như `hello`, `prediction`, `score`, `final`. Server broadcast lại danh sách người chơi đã sort theo score.

API xử lý: `WS /ws/pvp/{room}`.
Bảng dữ liệu lịch sử: `pvp_matches`.

## 7. Self-improving Loop

Khối `Self-improving Loop` biến dữ liệu người chơi thành dữ liệu train.

### Export data

Xuất dữ liệu đã lưu thành:

- `stroke_samples.jsonl`: dùng cho training/Colab.
- `stroke_samples.csv`: xem nhanh bằng Excel/Sheets.
- `training_manifest.json`: thống kê số mẫu, số lớp, readiness.

API xử lý: `GET /dataset/export`.

### Train stroke

Train model chuỗi nét BiGRU từ `stroke_samples` trong SQLite. Model sinh ra:

- `models/stroke_sequence_model.keras`
- `models/stroke_categories.json`

API xử lý: `POST /admin/retrain/start` với `mode=stroke`.
Script chạy nền: `src/training/train_stroke_model.py`.

### Train image

Rasterize stroke thành ảnh 28x28, trộn với QuickDraw `.npy` nếu có, rồi train CNN ảnh self-improved. Model sinh ra:

- `models/airdrawvocab_self_improved.keras`
- `models/categories_self_improved.json`

API xử lý: `POST /admin/retrain/start` với `mode=image`.
Script chạy nền: `src/training/self_improve_retrain.py`.

Sau khi train image xong, frontend sẽ gọi `POST /admin/model/reload` để nạp model mới vào runtime nếu file model tồn tại.

## Luồng sau khi chơi xong

```text
endGame()
→ POST /game/session lưu điểm
→ Skill Profile refresh
→ Leaderboard refresh
→ Self-improving status refresh
→ Nếu người dùng export/train, dữ liệu stroke_samples được dùng để cải thiện model
```
