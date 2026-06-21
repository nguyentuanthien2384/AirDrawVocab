# 40 VOCAB UPDATE NOTES

Dự án hiện định nghĩa 40 từ vựng trong `vocab_pairs.py`.

Điểm cần phân biệt:

- `vocab_pairs.py`: nguồn từ vựng đầy đủ của project, hiện có 40 từ.
- `models/categories.json` và model `.keras` đang đóng gói trong bản này: vẫn là artifact cũ 19 lớp, nên AI hiện chỉ nhận diện chắc chắn 19 từ đầu.
- Code web/game đã được sửa để tự động đọc số lớp model hiện tại. Nếu model đang là 19 lớp thì game chỉ chọn 19 từ có thể nhận diện để tránh màn chơi không thể thắng. Sau khi train/gắn model 40 lớp và cập nhật `models/categories.json`, game sẽ tự chạy 40 từ.

Các file đã cập nhật:

- `backend/app.py`: tách `all_vocab_categories` 40 từ và `categories` nhận diện theo model hiện tại.
- `frontend/app.js`: danh sách/hint/guide mở rộng 40 từ, đồng bộ số từ chơi được từ `/vocab`.
- `game.py`: đặt `TOTAL_LEVELS = 40`, nhưng tự giới hạn theo số lớp model nhận diện được.
- `vocab_data.py`: bỏ hard-code 19 từ, lấy dữ liệu từ `vocab_pairs.py`.
- `PDF_COMPLIANCE_REPORT.md`: chỉnh lại ghi chú đối chiếu PDF/source.

Để AI nhận diện đủ 40 từ: tải đủ dữ liệu QuickDraw cho 40 lớp rồi chạy train lại, ví dụ:

```bat
python dev_download_quickdraw.py
python train_best.py
```

Sau khi train xong, kiểm tra `models/categories.json` phải có 40 nhãn và model output cuối phải là 40.
