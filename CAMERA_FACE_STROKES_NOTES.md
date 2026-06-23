# AirDrawVocab - nhận diện nét vẽ khuôn mặt khi chơi bằng camera

Bản này phát triển thêm chế độ **vẽ tay camera bằng khuôn mặt** dựa trên source AirDrawVocab và cách xử lý khuôn mặt trong source DeepShieldAI-Pro mà bạn gửi.

## 1. Mục tiêu chức năng

Khi người chơi chọn **Vẽ tay camera**, hệ thống không chỉ nhận ngón tay nữa mà có thêm chế độ nhận diện khuôn mặt để tạo nét vẽ:

- **Bút tay**: giữ luồng cũ, dùng MediaPipe Hands để vẽ bằng ngón trỏ.
- **Bút mặt/mũi**: dùng MediaPipe FaceMesh để lấy landmark khuôn mặt và điều khiển bút bằng đầu mũi.
- **Quét nét mặt**: chụp một frame webcam, gửi về backend, backend nhận diện khuôn mặt và chuyển thành các strokes khuôn mặt đưa vào canvas game.

## 2. Luồng Bút mặt/mũi realtime

Frontend xử lý trực tiếp trên trình duyệt:

1. Mở camera.
2. Chạy MediaPipe FaceMesh.
3. Lấy landmark mũi, mắt, miệng.
4. Mũi điều khiển vị trí con trỏ vẽ.
5. Người chơi **há miệng nhẹ để vẽ**.
6. Người chơi **khép miệng để nhấc bút**.
7. Người chơi **chớp cả hai mắt để xóa canvas**.
8. Nét được lưu vào `state.strokes` với source `face-nose`.
9. Khi AI realtime dự đoán, frontend gửi source `camera-face` để backend biết đây là nét vẽ từ khuôn mặt.

Các hàm chính trong `frontend/app.js`:

- `setCameraTool(tool)`
- `ensureFaceMeshReady()`
- `faceFrameLoop()`
- `onFaceResults(results)`
- `getFaceDrawingMetrics(landmarks)`
- `mapFacePointToCanvas(nose)`
- `smoothFacePoint(point)`
- `drawFaceOverlay(ctx, landmarks, metrics)`

## 3. Luồng Quét nét mặt

Nút **Quét nét mặt** dùng khi muốn chuyển khuôn mặt trong webcam thành một bản phác nét trên canvas:

1. Frontend chụp frame hiện tại từ video camera.
2. Gửi frame tới backend bằng endpoint `POST /camera/face-strokes`.
3. Backend dùng OpenCV Haar Cascade để tìm khuôn mặt lớn nhất.
4. Backend crop vùng mặt có padding, tương tự pattern trong DeepShieldAI-Pro.
5. Backend sinh hai nhóm stroke:
   - `semantic_strokes`: nét ổn định như mặt, mắt, mũi, miệng, lông mày.
   - `edge_strokes`: nét biên từ ảnh camera bằng Canny/contour.
6. Frontend vẽ các strokes này vào canvas game.
7. Các strokes được lưu vào `state.strokes` với source `camera-face-sketch`.
8. AI realtime được gọi lại để nhận diện hình sau khi nét mặt đã được thêm.

File backend mới:

- `camera_face_strokes.py`

Endpoint backend mới:

```text
POST /camera/face-strokes
```

Form data:

```text
file=<webcam frame jpeg/png>
canvas_width=960
canvas_height=540
mirror=1
preview=0
```

Response chính:

```json
{
  "ok": true,
  "face_detected": true,
  "detector": "opencv-haar",
  "bbox": {"x": 0, "y": 0, "width": 100, "height": 100},
  "face_bbox": {"x": 0, "y": 0, "width": 100, "height": 100},
  "strokes": [],
  "semantic_strokes": [],
  "edge_strokes": [],
  "stroke_count": 0,
  "point_count": 0,
  "quality": 0
}
```

## 4. UI mới

Trong vùng game có thêm bộ chọn camera:

- **Bút tay**: vẽ bằng ngón trỏ như cũ.
- **Bút mặt/mũi**: vẽ bằng mũi, điều khiển bằng biểu cảm miệng/mắt.
- **Quét nét mặt**: sinh strokes khuôn mặt từ frame webcam.

Khu vực camera có thêm overlay:

- `faceCanvas`: hiển thị khung/landmark khuôn mặt và cursor bút mũi.
- Chip trạng thái: hiển thị `chưa thấy mặt`, `há miệng để vẽ`, `đang vẽ`, `đang quét mặt`, ...

## 5. Tích hợp với AI game

Các phần đã nối vào gameplay:

- `currentInputMode()` trả về `camera-hand` hoặc `camera-face`.
- `/predict_godmode` nhận đúng source camera hiện tại.
- Nếu canvas có strokes từ `camera-face-sketch`, panel AI hiện thêm `+face-strokes`.
- Khi lưu mẫu train, trường `mode` ghi `camera-face` hoặc `camera-hand`.
- Dữ liệu này vẫn đi vào export/self-improving loop như các mẫu vẽ khác.

## 6. Quyền riêng tư

Endpoint `/camera/face-strokes` chỉ xử lý frame trong RAM:

- Không lưu frame camera ra file.
- Không ghi ảnh khuôn mặt vào database.
- Chỉ đưa các điểm stroke đã chuẩn hóa vào canvas/dataset khi người chơi dùng chức năng.

## 7. File đã chỉnh/thêm

- `frontend/app.js`
- `frontend/index.html`
- `frontend/style.css`
- `backend/app.py`
- `camera_face_strokes.py`
- `CAMERA_FACE_STROKES_NOTES.md`

## 8. Kiểm tra đã chạy

```bash
python -m py_compile backend/app.py camera_face_strokes.py
node --check frontend/app.js
```

Tôi chưa chạy webcam thật trong container vì môi trường này không có trình duyệt/camera thật, nhưng code frontend/backend đã được kiểm tra cú pháp và endpoint helper đã test với frame không có mặt.
