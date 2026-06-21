# WEB_LOCALHOST_SYNC_NOTES

Bản này đồng bộ giao diện chạy trên web localhost theo các ảnh tham khảo, nhưng vẫn giữ đúng dự án AirDrawVocab.

## Đã cập nhật

- Màn chơi web `Camera Game Mode` được đổi sang dạng game screen/theater:
  - HUD lớn theo kiểu `Draw`, `Score`, `Time`, `Level`, `Lives`, `Streak`.
  - Khung chơi lớn 16:9, phù hợp chơi trên trình duyệt localhost.
  - Nút overlay `Menu`, `Start/Submit`, `Clear` nằm trên khung chơi.
  - Thanh chọn chế độ `Vẽ chuột` / `Vẽ tay (camera)` nằm ngay phía trên màn chơi.
- Giữ đầy đủ chức năng:
  - Vẽ chuột trực tiếp trên màn chơi.
  - Vẽ tay bằng camera qua MediaPipe.
  - AI tự dự đoán liên tục trong game.
  - Dashboard, đăng nhập, upload ảnh, result tabs và demo image vẫn hoạt động.
- Thêm cache-busting cho `style.css` và `app.js` để giảm lỗi trình duyệt dùng lại file CSS/JS cũ.

## Cách chạy

```powershell
cd "D:\AirDrawVocab"
.\.venv311\Scripts\activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Mở:

```text
http://127.0.0.1:8000
```

Nếu vẫn thấy giao diện cũ, hãy bấm `Ctrl + F5` hoặc mở tab ẩn danh. Đồng thời kiểm tra bạn đang chạy đúng thư mục của bản zip mới, không phải thư mục cũ.
