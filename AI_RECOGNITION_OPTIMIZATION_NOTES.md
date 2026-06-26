# AirDrawVocab - Tối ưu nhận diện AI và giải thích màn sau khi chơi

## Vì sao vẽ `book` nhưng AI nhầm sang `pants` hoặc `door`?

Model ảnh hiện tại nhận canvas bằng ảnh 28x28 kiểu QuickDraw. Khi một hình có nhiều nét dọc/ngang giống nhau, ví dụ `book`, `door`, `pants`, `square`, các chi tiết nhỏ như đường trang sách dễ bị mất khi ảnh bị ép xuống 28x28. Vì vậy CNN gốc có thể thấy hình chữ nhật + nét dọc và đoán nhầm sang `door` hoặc `pants`.

Ảnh trong game của bạn cho thấy CNN gốc nghiêng về `pants` 51%, `door` 41%, còn `book` 4%. Điều này không có nghĩa bạn vẽ sai hoàn toàn; nó cho thấy model ảnh gốc chưa đủ hiểu chi tiết `gáy sách + trang sách`.

## Đã tối ưu gì trong bản này?

### 1. Tiền xử lý ảnh tốt hơn

File: `image_preprocess.py`

- Giữ chi tiết nét nhỏ tốt hơn khi resize về 28x28.
- Dùng threshold ổn định hơn với nét vẽ web/camera.
- Làm dày nét rất nhẹ trước khi đưa vào model.
- Tăng khung target từ 20px lên 22px để bớt mất chi tiết trong hình như `book`.

### 2. TTA ổn định hơn

File: `backend/app.py`

Bản cũ dùng random crop khi realtime predict, nên cùng một hình có thể lúc thì ra `door`, lúc thì ra `pants`. Bản mới dùng deterministic TTA: dịch ảnh nhẹ theo các hướng cố định rồi lấy trung bình. Kết quả ổn định hơn và ít nhảy nhãn hơn.

### 3. Shape-rerank cho gameplay

File: `backend/app.py`

Trong game, backend biết mục tiêu hiện tại là từ nào. Vì vậy ngoài CNN, hệ thống kiểm tra thêm đặc trưng hình học của target:

- `book`: bìa chữ nhật, gáy dọc ở giữa, 2-3 đường trang sách bên trái.
- `door`: hình chữ nhật đứng, panel/cửa, ít đường trang sách.
- `pants`: hai ống quần, khoảng hở dưới giữa, không có đường trang sách.
- `square`: tỉ lệ vuông, ít chi tiết bên trong.
- `envelope`: khung + đường chéo phong bì.
- `lightning`: nhiều nét chéo/zigzag.

Nếu CNN nhầm nhưng shape check của target đủ cao, API `/predict_godmode` trả `ai_source = image-cnn+shape-rerank` và cho target lên trên.

### 4. UI hiển thị rõ lý do chỉnh kết quả

File: `frontend/app.js`

Khối Realtime AI bây giờ có thêm dòng:

- `AI source`: nguồn đang dùng, ví dụ `image-cnn`, `image-cnn+shape-rerank`, `stroke-sequence`.
- `Tối ưu nhận diện`: cho biết CNN gốc đã đoán gì trước khi rerank.
- `Shape check`: nếu shape chưa đủ mạnh để sửa kết quả CNN.

### 5. Guide vẽ `book` rõ hơn

File: `frontend/app.js`

Guide trên canvas cho `book` đã thêm 3 đường trang sách bên trái, vì đây là đặc điểm giúp phân biệt `book` với `door` và `pants`.

## Các khối sau khi chơi xong để làm gì?

### Realtime AI

Đây là bảng dự đoán live. Nó không lưu kết quả một mình, nhưng quyết định game có tự qua màn hay không. Nếu target đúng và confidence vượt ngưỡng trong 2 lần liên tiếp, game tự pass round.

### AI Judge Mode

Đây là bảng chấm chất lượng bản vẽ:

- Shape: hình tổng thể có giống mục tiêu không.
- Clarity: AI tự tin bao nhiêu.
- Stroke: số nét/điểm vẽ có đủ chưa.
- Speed: tốc độ hoàn thành.

Bảng này sinh feedback để người chơi biết nên sửa gì. Khi lưu mẫu train, nội dung judge được lưu vào database cùng stroke sample.

### Skill Profile

Đây là hồ sơ học của người chơi. Nó lấy dữ liệu từ `game_sessions` và `stroke_samples` trong SQLite. Nó cho biết bạn yếu từ nào, mạnh từ nào, cần luyện từ nào.

### Practice Plan

Đây là kế hoạch luyện tự động dựa trên những từ bạn sai nhiều hoặc confidence thấp. Nó không tự train model, nhưng chỉ ra nên lưu thêm mẫu nào để model học tốt hơn.

### Leaderboard

Bảng xếp hạng lấy lượt chơi tốt nhất từ SQLite. Khi game kết thúc, `/game/session` lưu score, level, streak, accuracy rồi bảng này cập nhật.

### PvP WebSocket

Chế độ chơi nhiều người theo phòng. Bấm Join sẽ mở WebSocket vào phòng, broadcast score/prediction/final event cho người cùng phòng.

### Self-improving Loop

Đây là phần dùng để biến dữ liệu chơi game thành dữ liệu train AI:

- `Export data`: xuất stroke samples ra JSONL/CSV/manifest.
- `Train stroke`: train model nhận diện theo chuỗi nét vẽ người chơi.
- `Train image`: train lại image CNN từ dữ liệu đã lưu và QuickDraw nếu có.

Bấm các nút này không "gen ảnh kết quả". Chúng thao tác với dữ liệu/model:

- Export tạo file dataset trong `data/self_improving_loop/exports/latest`.
- Train stroke tạo `models/stroke_sequence_model.keras`.
- Train image tạo `models/airdrawvocab_self_improved.keras`, sau đó frontend có thể reload model mới.

## Cách để AI nhận `book` tốt hơn ngay

Khi vẽ `book`, hãy vẽ theo thứ tự:

1. Vẽ bìa chữ nhật lớn.
2. Vẽ một đường gáy dọc hơi gần giữa.
3. Vẽ 2-3 đường ngang ngắn ở nửa trái.
4. Tránh vẽ dáng giống hai ống quần tách đáy.
5. Bấm `Lưu mẫu train` vài lần cho các mẫu đẹp và mẫu bị nhầm, rồi chạy `Train stroke`.
