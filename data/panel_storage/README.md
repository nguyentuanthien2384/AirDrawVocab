# AirDrawVocab panel storage

Thư mục này được backend tạo/cập nhật tự động để lưu dữ liệu của các panel bên phải giao diện.

## Tự lưu khi thao tác

- `ai_recognition/`: lưu `drawing.png` và `prediction.json` mỗi lần bấm **Nhận diện**.
- `real_image_after_draw/by_label/<label>/`: lưu ảnh PNG và metadata JSON mỗi lần bấm **Sinh hình thật**.
- `ai_judge_mode/`: lưu điểm Shape/Clarity/Stroke/Speed và feedback khi AI Judge chạy sau nhận diện.
- `skill_profile/`: lưu snapshot Skill Profile, phiên chơi và mẫu vẽ đã lưu train.
- `leaderboard/`: lưu snapshot Leaderboard mới nhất và lịch sử.
- `pvp_websocket/`: lưu sự kiện join/leave/message/score/final theo room PvP.
- `game_sessions/`: lưu JSON mỗi phiên chơi khi kết thúc game.
- `actions/events.jsonl`: nhật ký tổng hợp mọi thao tác đã ghi vào kho này.

Phần **Hình thật sau khi vẽ** không tự sinh khi đang vẽ; ảnh chỉ được tạo và lưu khi người dùng bấm nút **Sinh hình thật**.
