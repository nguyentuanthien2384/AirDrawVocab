# AirDrawVocab Final Boss Mode

Bản này đã refactor web localhost theo kiến trúc Single Page Game App:

- Màn đăng nhập chỉ còn form đăng nhập/đăng ký.
- Sau khi đăng nhập thành công, người dùng vào thẳng màn chơi chính.
- Không còn dashboard ngoài màn login, không còn notebook UI trong web.
- Game dùng realtime AI: vẽ tới đâu backend dự đoán tới đó qua `/predict_godmode`.
- AI tự qua level khi nhận đúng từ đủ ổn định.
- Có AI Judge Mode: shape score, clarity, stroke score, speed score, grade và feedback.
- Có lưu stroke dataset vào SQLite để sau này xuất dữ liệu train/retrain.
- Có Skill Profile và Leaderboard trong màn game chính.
- Có camera hand tracking bằng MediaPipe Hands: dùng ngón trỏ để vẽ, mở bàn tay để xóa.

## Các file chính đã sửa

- `frontend/index.html`: giao diện root mới, chỉ load SPA Final Boss.
- `frontend/style.css`: style game shell, login, HUD, arena, side panels.
- `frontend/app.js`: game engine realtime, camera tracking, auto judge, profile, leaderboard.
- `backend/app.py`: thêm bảng SQLite và API:
  - `POST /predict_godmode`
  - `GET /game/profile`
  - `GET /game/leaderboard`
  - `POST /game/session`
  - `POST /game/stroke`

## Chạy localhost

```powershell
cd "D:\AirDrawVocab"
.\.venv311\Scripts\activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

## Lưu ý model 40 từ

Source dự án giữ đủ 40 từ trong `vocab_pairs.py`. Nếu model hiện tại chỉ có 19 output class thì game sẽ ưu tiên các từ model đang nhận diện được để tránh level không thể qua. Khi thay bằng model 40 lớp và cập nhật `models/categories.json`, giao diện tự chạy đủ 40 từ.
