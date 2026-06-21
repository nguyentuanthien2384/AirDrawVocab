# PDF COMPLIANCE REPORT - AirDrawVocab

## Kết luận cập nhật

Source hiện tại của dự án định nghĩa **40 từ vựng** trong `vocab_pairs.py`.
Báo cáo PDF đính kèm đang mô tả bản cũ với **19 từ vựng / Dense(19, Softmax)**.
Vì vậy, nếu lấy PDF làm chuẩn nộp bài thì PDF cần được sửa lại từ 19 thành 40 ở các phần mô tả dataset, kiến trúc output và kết quả thực nghiệm. Nếu lấy model đang đóng gói làm chuẩn chạy demo, artifact hiện tại vẫn là 19 lớp.

## Đối chiếu với PDF

Phần đúng theo PDF:

- Dự án là nền tảng học từ vựng tiếng Anh qua vẽ hình, AI nhận diện từ vựng và trả nghĩa tiếng Việt, IPA, ví dụ.
- Có mô hình CNN nhận diện hình vẽ 28x28 grayscale.
- Có game camera dùng MediaPipe để theo dõi tay.
- Có giao diện Pygame/camera như ảnh minh họa.
- Có TTS/pyttsx3 ở game desktop.

Phần đã phát triển thêm so với PDF:

- Source từ vựng mở rộng lên 40 từ trong `vocab_pairs.py`.
- Web Camera Game có theater mode giống ảnh demo: HUD phía trên, camera lớn, nút Menu/Clear/Submit overlay.
- Web game tự dùng danh sách từ vựng nhận diện được từ backend.

## Ghi chú kỹ thuật quan trọng

Trong zip hiện tại:

- `vocab_pairs.py` = 40 từ.
- `models/categories.json` = 19 nhãn.
- `models/airdrawvocab_best_advanced.keras` = model output 19 lớp.

Do đó, code đã được sửa theo hướng an toàn:

- `/vocab` trả đủ 40 từ, kèm cờ `recognition_supported`.
- Game chỉ chọn các từ mà model hiện tại nhận diện được.
- Khi bạn train/gắn model 40 lớp và cập nhật `models/categories.json`, web game và desktop game sẽ tự chuyển sang 40 từ.
