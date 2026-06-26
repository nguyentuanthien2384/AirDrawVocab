# Panel Storage update

Bản cập nhật này thêm kho lưu tự động cho các panel ngoài Self-improving Loop.

## Folder mới

`data/panel_storage/`

Các thư mục con:

- `ai_recognition/`: mỗi lần bấm **Nhận diện**, backend lưu ảnh canvas `drawing.png` và `prediction.json`.
- `real_image_after_draw/by_label/<label>/`: mỗi lần bấm **Sinh hình thật**, backend lưu ảnh PNG sinh ra và metadata JSON. Đây là phần quan trọng cho panel **Hình thật sau khi vẽ**.
- `ai_judge_mode/`: lưu kết quả chấm điểm AI Judge.
- `skill_profile/`: lưu snapshot hồ sơ kỹ năng, mẫu vẽ liên quan và dữ liệu phiên chơi.
- `leaderboard/`: lưu snapshot bảng xếp hạng.
- `pvp_websocket/`: lưu sự kiện theo phòng PvP.
- `game_sessions/`: lưu JSON mỗi phiên chơi.
- `actions/events.jsonl`: log tổng hợp mọi thao tác lưu trữ.

## API mới

- `GET /admin/panel-storage`: xem nhanh toàn bộ kho lưu panel.
- `GET /admin/panel-storage?section=real_image_after_draw`: xem riêng kho ảnh thật sau khi vẽ.

## UI mới

Trong panel **Hình thật sau khi vẽ** có thêm:

- `Storage hình thật`: xem các ảnh thật đã sinh/lưu gần nhất.
- `Storage tất cả`: xem các file gần nhất của mọi panel.

## Luồng Hình thật sau khi vẽ

Frontend vẫn giữ logic thủ công:

1. Người dùng vẽ xong.
2. Nút **Sinh hình thật** mới được bật.
3. Khi bấm nút, frontend gọi `/image/generate`.
4. Backend sinh ảnh, trả ảnh cho UI và lưu bản PNG vào `data/panel_storage/real_image_after_draw/by_label/<label>/`.
5. Khi đang vẽ, hệ thống không tự sinh ảnh.
