# Update merge notes

Đã cập nhật gói `AirDrawVocab_changed_files (2).zip` vào source code.

Các thay đổi đã có trong source:
- `frontend/index.html`: HUD game, controls nâng cao, nút chọn vẽ chuột / vẽ tay camera.
- `frontend/style.css`: bố cục game, chế độ `.mouse-mode`, style nút active.
- `frontend/app.js`: game 45s / 12 level / 5 lives, vẽ chuột trực tiếp, vẽ tay qua camera, nhận diện thủ công mới chuyển vòng, AI tự đoán chỉ để tham khảo.
- `backend/app.py`: preprocess ảnh theo kiểu MNIST/QuickDraw.

Phần sửa camera trước đó được giữ lại:
- Browser vẫn bắt buộc hỏi quyền camera lần đầu; không thể bỏ qua lớp quyền này.
- App kiểm tra secure origin/local origin và báo lỗi rõ hơn.
- App thử nhiều cấu hình `getUserMedia()` trước khi fallback.
- Backend gửi `Permissions-Policy: camera=(self), microphone=()`.

Cách chạy:
1. Mở CMD/PowerShell trong thư mục `AirDrawVocab`.
2. Chạy `run_web_chatbot.bat`.
3. Mở `http://127.0.0.1:8000`.
