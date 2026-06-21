# Camera fix notes

Bản này đã sửa phần mở camera ở frontend:

- Kiểm tra đúng secure context trước khi gọi `navigator.mediaDevices.getUserMedia`.
- Thử nhiều cấu hình camera, fallback về `video: true` nếu camera không nhận constraint `facingMode` hoặc độ phân giải.
- Hiển thị lỗi cụ thể: sai origin, bị chặn quyền, không thấy webcam, webcam bị app khác chiếm.
- Thêm header `Permissions-Policy: camera=(self), microphone=()` ở backend.
- Nếu camera lỗi, app tự chuyển sang vẽ chuột để demo không bị chết.

Lưu ý bắt buộc của trình duyệt: website không thể bật webcam mà bỏ qua quyền camera của user. Lần đầu mở camera, Chrome/Edge vẫn phải hỏi quyền. Sau khi bấm Allow trên `http://127.0.0.1:8000` hoặc `http://localhost:8000`, trình duyệt thường sẽ nhớ quyền cho origin này.

Cách chạy khuyến nghị:

```bat
run_web_chatbot.bat
```

Hoặc:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Sau đó mở đúng URL:

```text
http://127.0.0.1:8000
```

Không mở bằng `file://`, `http://0.0.0.0:8000` hoặc IP LAN như `http://192.168.x.x:8000`, vì browser có thể không cấp API camera cho các origin đó nếu không dùng HTTPS.
