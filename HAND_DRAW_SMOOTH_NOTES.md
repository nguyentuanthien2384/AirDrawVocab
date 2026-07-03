# Vẽ tay camera mượt & không đứt nét khi vẽ qua khuôn mặt

Bản cập nhật này tập trung sửa đúng 2 vấn đề bạn mô tả ở chế độ **Vẽ tay camera (bút ngón trỏ)**:

1. **Nét bị đứt / không vẽ được khi ngón tay đi ngang qua khuôn mặt.**
2. **Nét chưa liền mạch, còn bị nguệch ngoạc/rung.**

Cách làm bám theo tinh thần của source **QuickDraw** bạn gửi (`QuickDraw-master/mediapipe_app.py`): điểm mấu chốt khiến QuickDraw vẽ "chuẩn chỉ" là danh sách điểm được **giữ liên tục qua các khung hình** — mất tay 1–2 khung chỉ là bỏ qua một khung, đường nét cũ vẫn còn và tự nối lại khi tay xuất hiện. Mình đã áp dụng đúng ý tưởng đó cho phiên bản web (MediaPipe Hands JavaScript) trong `frontend/app.js`, cộng thêm vài cải tiến làm nét mượt hơn.

Tất cả thay đổi nằm ở **`frontend/app.js`** (và bump version cache trong `frontend/index.html`). Không đổi backend, model hay dataset.

---

## 1. Không còn đứt nét khi vẽ qua khuôn mặt

Nguyên nhân gốc: khi bàn tay che lên mặt, nền da mặt làm MediaPipe **rớt tay vài khung**. Code cũ thấy mất tay là đếm 4 khung rồi **kết thúc nét ngay** → nét bị cắt đôi đúng chỗ khuôn mặt.

Cách sửa: khi mất tay, **KHÔNG kết thúc nét** mà **TẠM DỪNG** và giữ trong một khoảng ân hạn. Khi tay xuất hiện lại kịp thời và gần chỗ vừa mất, hệ thống **NỐI TIẾP đúng nét cũ** bằng cách bắc cầu nội suy qua vùng che mặt — nét đi xuyên qua mặt liền mạch.

Các mốc điều chỉnh được (đặt tên rõ trong code để bạn dễ tinh chỉnh):

- `HAND_LOST_GRACE_MS = 700` — mất tay trong ~0,7 giây thì vẫn giữ nét chờ nối tiếp.
- `RESUME_MAX_GAP_PX = 260` — tay hiện lại trong bán kính này thì nối vào nét cũ; xa hơn coi như bắt đầu nét mới.
- Hàm mới: `handleHandLost()`, `resumeStrokeBridge()`, `finishCurrentStroke()`.

Ngoài ra hạ ngưỡng bám tay của MediaPipe (`minTrackingConfidence` từ `0.4` → `0.25`) để nó **cố bám ngón tay lâu hơn** khi đi qua mặt, giảm số lần rớt tay ngay từ đầu.

## 2. Nét mượt hơn, hết nguệch ngoạc

Bốn thay đổi cộng lại:

1. **Sửa lỗi nội suy bị vọt điểm (quan trọng nhất).** Vòng lặp chèn điểm trung gian khi tay đi nhanh trước đây tính điểm giữa dựa trên `state.lastPoint` **đang bị cập nhật ngay trong vòng lặp** → offset bị **cộng dồn**, điểm giữa vọt quá đích rồi giật ngược lại (chính là cảm giác nét lượn/nguệch ngoạc khi vẩy tay nhanh). Nay nội suy dựa trên **điểm gốc cố định** nên các điểm cách đều, đúng đường thẳng. (Đo bằng test: khoảng hở lớn nhất trong một cú vẩy nhanh giảm từ ~398px xuống ~16px.)

2. **Lọc One-Euro chỉnh mượt hơn** cho nét vẽ: `minCutoff 1.2 → 1.0`, `beta 0.015 → 0.007`. Tay đi chậm thì lọc mạnh (hết rung), tay đi nhanh vẫn bám sát mà không trễ nhiều.

3. **Khử rung khi tay gần đứng yên** (`HAND_MIN_MOVE_PX = 1.6`): dịch nhỏ hơn ngưỡng này thì bỏ qua, không thêm điểm → nét không bị "râu" khi tay run nhẹ.

4. **Chống rung bật/tắt bút** (`PEN_UP_FRAMES = 3`): chỉ thật sự nhấc bút khi co ngón đủ 3 khung liên tục, nên một khung nhiễu không làm đứt nét li ti. Việc nhận biết "ngón trỏ duỗi để vẽ" cũng kết hợp thêm tín hiệu kiểu QuickDraw (đầu ngón cao hơn đốt) cho ổn định.

Cú vẫy nhanh giờ được nội suy thành nhiều điểm cách đều thay vì bị cắt vụn (`MAX_JUMP` nâng lên 220, `NEW_STROKE_JUMP_PX = 210`); chỉ khi nhảy thật xa (đổi chỗ chủ động) mới tách nét mới.

## 3. Cử chỉ giữ nguyên như cũ

- **Ngón trỏ duỗi**: hạ bút vẽ.
- **Co ngón trỏ**: nhấc bút.
- **Xòe cả bàn tay**: xóa nét.
- **Giơ 3 ngón (trỏ + giữa + áp út, gập ngón út)**: tự động nhận diện nét vừa vẽ (port từ QuickDraw).

## 4. Cách chạy để thấy thay đổi

```bat
run_web_chatbot.bat
```

Mở `http://localhost:8000` (dùng `localhost`, không dùng địa chỉ IP LAN để trình duyệt cho phép camera), vào chế độ **Vẽ tay camera** → **Bút tay**, rồi vẽ thử một nét đi ngang qua mặt: nét sẽ liền mạch, không còn đứt.

> Nếu trình duyệt vẫn nạp bản JS cũ, hãy tải lại trang bỏ cache (Ctrl+F5). Version cache đã được đổi thành `smooth-hand-draw-3` trong `frontend/index.html`.

## 5. File đã chỉnh

- `frontend/app.js` — toàn bộ logic vẽ tay ở trên.
- `frontend/index.html` — đổi version cache `?v=smooth-hand-draw-3`.
- `HAND_DRAW_SMOOTH_NOTES.md` — ghi chú này.

## 6. Kiểm thử đã chạy

Vì môi trường không có camera/trình duyệt thật, mình mô phỏng chuỗi khung hình MediaPipe (gồm cả tình huống "mất tay khi che mặt") và chạy trực tiếp state-machine vẽ tay của `app.js`:

- **Vẽ qua khuôn mặt (mất tay 5 khung giữa nét)** → vẫn là **một nét liền mạch**, khoảng hở lớn nhất ~24px (đã bắc cầu). ✅
- **Tay gần đứng yên có nhiễu ±1px suốt 40 khung** → nhiễu bị lọc, nét không xòe. ✅
- **Vẩy tay nhanh (~120px/khung)** → **một nét**, nội suy đều, khoảng hở lớn nhất ~16px (trước khi sửa là ~398px). ✅

`node --check frontend/app.js` cũng đã pass.
