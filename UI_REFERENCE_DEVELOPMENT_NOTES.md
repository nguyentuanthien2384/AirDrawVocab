# UI_REFERENCE_DEVELOPMENT_NOTES

Bản cập nhật này phát triển giao diện AirDrawVocab theo các ảnh tham khảo người dùng gửi, nhưng không đổi đề tài sang nhận diện bệnh mắt. Các ảnh tham khảo chỉ được dùng làm hướng bố cục UI: dashboard, đăng nhập, upload ảnh, khu vực demo, bảng kết quả/tab thông tin và confusion matrix.

## Phần đã phát triển

1. Dashboard người dùng
   - Thêm khu vực tổng quan: tổng số từ vựng, số lớp model hiện tại nhận diện được, tổng lượt dự đoán và lượt dự đoán của tài khoản đang đăng nhập.
   - Thêm biểu đồ phân bố nhóm từ vựng bằng canvas.
   - Thêm confusion matrix demo để trình bày bố cục đánh giá mô hình.
   - Thêm top dự đoán gần đây lấy từ SQLite.
   - Thêm gallery từ vựng demo từ `/vocab`.

2. Tài khoản người dùng
   - Thêm đăng ký, đăng nhập, đăng xuất.
   - Lưu tài khoản trong SQLite tại `data/airdrawvocab_app.sqlite3`.
   - Lưu session bằng cookie `airdrawvocab_session`.

3. Nhận diện từ ảnh upload
   - Thêm khung upload ảnh vẽ tay.
   - Dùng lại endpoint `/predict` và pipeline OpenCV/PIL hiện có.
   - Kết quả upload hiển thị chung với kết quả canvas: top 3, chatbot, ảnh vẽ rõ nét, ảnh tham khảo.

4. Tab kết quả tương tự giao diện tham khảo
   - Mô tả từ.
   - Ví dụ.
   - Gợi ý vẽ.
   - Đọc thông tin bằng Web Speech API của trình duyệt.

5. Giữ nguyên chức năng cốt lõi của dự án
   - Camera Game Mode vẫn dùng MediaPipe Hands.
   - Canvas QuickDraw vẫn dùng CNN/Keras model.
   - Từ vựng vẫn theo `vocab_pairs.py` với 40 từ.
   - Nếu model hiện tại mới có 19 output thì UI vẫn hiển thị 40 từ, nhưng nhận diện chỉ chạy trên các lớp model hỗ trợ.

## File chính đã sửa

- `backend/app.py`
- `frontend/index.html`
- `frontend/app.js`
- `frontend/style.css`

## Lưu ý

Confusion matrix trong dashboard hiện là demo layout. Khi train xong model 40 lớp, nên xuất confusion matrix thật từ `evaluate_model.py` hoặc script đánh giá model rồi thay endpoint `/analytics` để đọc kết quả thật.
