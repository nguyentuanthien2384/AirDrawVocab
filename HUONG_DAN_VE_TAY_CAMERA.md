# Tính năng mới: Vẽ tay bằng camera (shape_match)

Đã tích hợp sẵn vào dự án này. Có **2 chế độ**:

| Chế độ | Mục đích | Cử chỉ |
|---|---|---|
| **Vẽ tự do** (giống video YouTube) | vẽ bất kỳ, có thanh màu + tẩy + xóa | 1 ngón trỏ = vẽ · 2 ngón (trỏ+giữa) = chọn màu |
| **Tô theo hình mẫu** (chấm % khớp) | đồ theo outline mẫu, chấm điểm độ chính xác | ngón trỏ duỗi + ngón giữa gập = vẽ |

## Chạy trên WEB (đã gắn sẵn router vào backend)
1. Cài thư viện (nếu chưa): `pip install -r requirements.txt`
2. Chạy backend như bình thường: `run_web_chatbot.bat` (hoặc lệnh uvicorn bạn vẫn dùng).
3. Mở trình duyệt:
   - Tô theo hình mẫu (có chấm điểm): `http://127.0.0.1:8000/static/shape_trace.html`
   - Vẽ tự do giống video:           `http://127.0.0.1:8000/static/paint.html`

> Backend tự thêm 2 endpoint: `GET /shape/templates`, `POST /shape/score`.
> Nếu vì lý do gì đó không gắn được, app vẫn chạy bình thường (đã bọc try/except).

## Chạy thử bằng app DESKTOP (tùy chọn)
```bat
python -m shape_match.paint_app        # vẽ tự do giống video
python -m shape_match.trace_app        # tô theo hình mẫu
```

## Kiểm thử nhanh (không cần camera)
```bat
python -m shape_match.demo_offline
python -m shape_match.tests.test_matcher
```

Chi tiết kiến trúc, cách tinh chỉnh độ khó, thêm hình mẫu mới… xem `shape_match/README.md`.
