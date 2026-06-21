# Các file đã thay đổi (không có file tạo mới)

Chép đè 4 file vào project gốc theo đúng đường dẫn.

| File | Thay đổi chính |
|------|----------------|
| `frontend/index.html` | HUD (45s / 12 level / 5 tim). Gom cài đặt nâng cao vào `<details>`. Độ dày nét 8–44, mặc định 24. **Thêm 2 nút "🖱️ Vẽ chuột" / "✋ Vẽ tay (camera)" cho khu game.** |
| `frontend/style.css` | Bố cục gọn để không phải cuộn. **Thêm chế độ `.mouse-mode` cho khung game (vẽ chuột ngay trên khung).** Style nút active. |
| `frontend/app.js` | Lives 5, level 12, 45s. Camera: tự chấm chỉ khi top-1 ≥ 0.62; bấm Nhận diện thủ công thì nới lỏng (top-3). Bảng vẽ chuột: HiDPI sắc nét + làm mượt nét. **CHƠI ĐƯỢC CẢ 2 CÁCH:** vẽ chuột ngay trên khung game (mặc định, không cần camera) hoặc vẽ tay qua camera; cả hai tô lên cùng một bảng AI chấm. Map giữ tỉ lệ tránh méo hình. Camera bị chặn thì tự chuyển sang vẽ chuột. |
| `backend/app.py` | `preprocess_image`: chuẩn hoá kiểu MNIST/QuickDraw (cắt sát + căn trọng tâm 28×28). |

## Cách chơi sau cập nhật
1. Mặc định là **Vẽ chuột** — bấm **Bắt đầu**, xem từ ở ô **Draw**, rồi vẽ trực tiếp bằng chuột vào khung lớn phía trên. AI tự chấm mỗi ~1s hoặc bấm **Nhận diện**.
2. Muốn vẽ bằng ngón tay thì bấm **✋ Vẽ tay (camera)** và cho phép camera.

## Lưu ý
- Chưa đụng `models/categories.json` (vẫn 19 lớp) — đúng model đang deploy.

## Cập nhật mới nhất
- **Không tự chuyển hình nữa:** game chỉ chấm và sang vòng khi bạn tự bấm **Nhận diện**. Vòng tự động chạy ngầm chỉ hiển thị "AI đang đoán..." để bạn tham khảo, không tự nhảy.
- **Camera:** thêm thông báo lỗi rõ ràng (chặn quyền / không có webcam / bị app khác chiếm / sai địa chỉ truy cập). Lỗi "Permission denied" là do trình duyệt/Windows chặn quyền — bấm ổ khóa trên thanh địa chỉ → Camera → Cho phép, và bật quyền camera trong Windows Settings.
