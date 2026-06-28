# Shape Trace Pro — Vẽ tay khớp hình mẫu (module `shape_match`)

Module bổ sung cho **AirDrawVocab**: hiển thị một **hình mẫu** (target), người dùng
**dùng đầu ngón tay vẽ theo qua camera**, hệ thống chấm điểm **độ khớp** giữa nét vẽ
và hình mẫu một cách nghiêm túc — đúng ý "khi vẽ phải chuẩn khớp với hình vẽ".

Đây là phần dự án bạn còn thiếu: AirDrawVocab cũ chỉ **nhận diện bạn vẽ con gì** (CNN
phân loại 19 lớp QuickDraw). Module này trả lời câu hỏi khác: **"bạn tô có ĐÚNG theo
hình mẫu này không, đúng bao nhiêu %?"** — giống các video tham khảo bạn gửi.

---

Module có **2 chế độ**, chọn theo nhu cầu:

| Chế độ | File | Mô tả | Cử chỉ |
|---|---|---|---|
| **Vẽ tự do** (giống video YouTube) | `paint_app.py` · `web/paint.html` | Vẽ bất kỳ, có thanh màu + tẩy + xóa. KHÔNG chấm điểm. | 1 ngón trỏ = vẽ · 2 ngón (trỏ+giữa) = chọn màu trên thanh |
| **Tô theo hình mẫu** (chấm % khớp) | `trace_app.py` · `web/shape_trace.html` | Đồ theo outline mẫu, chấm độ khớp nghiêm túc. | ngón trỏ duỗi + ngón giữa gập = vẽ |

> "Vẽ y hệt như video" → dùng **chế độ vẽ tự do** (mục 3b / 4c).
> "Vẽ phải chuẩn khớp với hình" → dùng **chế độ tô theo mẫu** (mục 4).

---



Pipeline đúng chuẩn các nghiên cứu shape-tracing (resample → normalize → align → score):

1. **Resample theo chiều dài cung** — đưa cả nét vẽ và hình mẫu về cùng số điểm cách
   đều. Nhờ vậy vẽ nhanh hay chậm, nhiều hay ít điểm đều được so công bằng (bỏ phụ
   thuộc tốc độ).
2. **Normalize (bất biến vị trí + tỉ lệ)** — dời trọng tâm về gốc, chia cho bán kính
   RMS. Đứng gần hay xa camera, vẽ to hay nhỏ, vẽ ở góc nào của khung hình đều chấm
   như nhau. (Đã kiểm chứng: vẽ vòng tròn nhỏ lệch tâm vẫn đạt 98% so với mẫu tròn lớn.)
3. **Align** — DTW (co giãn dọc đường) + với **hình kín** thì thử mọi điểm bắt đầu và
   cả chiều vẽ ngược, nên không bắt buộc bắt đầu đúng chỗ.
4. **Chấm điểm bằng TÍCH 4 thành phần** (dùng tích nên cả 4 phải tốt):

   | Thành phần | Ý nghĩa | Bắt lỗi gì |
   |---|---|---|
   | `shape_score` | quỹ đạo có giống mẫu không (DTW) | đi sai đường |
   | `coverage`    | đã vẽ **đủ** hình chưa | vẽ thiếu / bỏ dở |
   | `precision`   | có nét **thừa** ra ngoài không | nguệch ngoạc, lem |
   | `corner_score`| khớp **góc nhọn / đường cong** (turning function) | tròn ≠ vuông |

   `accuracy = 100 · shape^Ws · coverage^Wc · precision^Wp · corner^Wc2`

   Thêm **cổng cứng Hausdorff**: nếu có đoạn lệch quá xa thì tự động trượt.

Kết quả calibrate (xem `demo_offline.py` và ảnh `shape_match_proof.png`):

```
perfect       -> 100%  ĐẠT ***          tròn nhỏ vs mẫu tròn lớn -> 98% ĐẠT (bất biến tỉ lệ)
nhiễu nhẹ     -> 95-99% ĐẠT             vẽ vuông khi mẫu là tròn -> 79% CHƯA ĐẠT
vẽ một nửa    -> trượt (thiếu coverage) nguệch ngoạc            -> ~0-5% CHƯA ĐẠT
```

> Mặc định đặt **GẮT** (`pass_threshold = 80`). Muốn dễ hơn (trẻ em / người mới) thì
> hạ xuống 70-75 — xem mục 5.

---

## 2. Cấu trúc

```
shape_match/
├── __init__.py        # API: ShapeMatcher, MatchConfig, templates, geometry
├── geometry.py        # resample, normalize, DTW, Hausdorff, coverage/precision, turning function
├── templates.py       # thư viện hình mẫu (tròn, vuông, sao, tim, sóng, zigzag...)
├── matcher.py         # ShapeMatcher: gộp các thành phần -> accuracy + ĐẠT/CHƯA + sao
├── kalman.py          # làm mượt đầu ngón tay (Kalman, fallback EMA)
├── trace_app.py       # app camera desktop (tô theo mẫu, chạy thử nhanh)
├── paint_app.py       # app camera desktop VẼ TỰ DO (giống video YouTube)
├── web_endpoint.py    # tích hợp FastAPI: POST /shape/score, GET /shape/templates
├── web/
│   ├── shape_trace.html  # TRANG WEB tô-theo-hình-mẫu (chấm điểm; cần backend)
│   └── paint.html        # TRANG WEB vẽ tự do giống video (không cần backend)
├── demo_offline.py    # kiểm chứng chấm điểm KHÔNG cần camera
└── tests/
    └── test_matcher.py
```

---

## 4. Tích hợp WEB (cách dùng chính)

Frontend của bạn đã nạp MediaPipe Hands từ CDN và backend đã phục vụ `frontend/` tại
`/static/` (CORS đã mở). Làm 3 bước:

**Bước 1 — đặt module.** Chép thư mục `shape_match/` vào gốc project (cạnh `backend/`,
`frontend/`).

**Bước 2 — gắn router vào `backend/app.py`** (thêm 2 dòng, đặt sau khi tạo `app`):

```python
from shape_match.web_endpoint import router as shape_router
app.include_router(shape_router)
```

Có ngay 2 endpoint (đã test bằng TestClient, hoạt động):

```
GET  /shape/templates   -> danh sách hình mẫu + outline [0,1] để vẽ "ghost" lên canvas
POST /shape/score       body: {"target":"star","strokes":[[{"x","y","t"},...]],
                               "canvas_w":960,"canvas_h":540}
                        -> {"accuracy":99.9,"passed":true,"stars":3,
                            "shape_score":..,"coverage":..,"precision":..,"corner_score":..,
                            "message":"...","target":"star","label_vi":"Ngôi sao"}
```

**Bước 3 — đặt trang vẽ.** Chép `shape_match/web/shape_trace.html` vào `frontend/`. Mở:

```
http://127.0.0.1:8000/static/shape_trace.html
```

Trang này tự chạy: bật camera, theo dõi đầu ngón trỏ (landmark 8) + cử chỉ vẽ (ngón
trỏ duỗi, ngón giữa gập) giống app desktop, làm mượt bằng bộ lọc One-Euro, vẽ "ghost
outline" của hình mẫu, nhấc tay là tự gọi `/shape/score` và hiện %, ĐẠT/CHƯA ĐẠT, sao,
4 thanh thành phần. Có sẵn nút **Vẽ bằng chuột** nếu máy không có camera.

> Trang dùng đúng quy ước của bạn: canvas 960×540, strokes `[{x,y,t}]`. Không cần sửa
> file `app.js` 100KB. Muốn nhúng thành một tab trong giao diện chính thì gọi cùng 2
> API trên từ `app.js` và tái dùng phần camera sẵn có — logic chấm điểm nằm hết ở backend.

Gọi API trong code (không cần HTTP):

```python
from shape_match.web_endpoint import score_strokes
out = score_strokes(strokes, "circle")   # dict điểm
```

---

## 4b. Chạy thử nhanh bằng app desktop (tùy chọn)

```bat
python -m shape_match.trace_app
```
Phím: `C` xóa · `N`/`P` đổi hình mẫu · `Space` chấm điểm · `Q`/`Esc` thoát.
Nên dùng **Python 3.11/3.12** để MediaPipe ổn định.

---

## 4c. Chế độ VẼ TỰ DO — giống video YouTube (cử chỉ y hệt)

Cử chỉ theo mô hình "AI Virtual Painter" kinh điển mà các video dạng này dùng:

- **Chỉ ngón trỏ giơ lên** → **VẼ** (bút tại đầu ngón trỏ)
- **Ngón trỏ + ngón giữa cùng giơ** → **CHỌN** (đưa lên thanh trên cùng để đổi màu /
  chọn **Tẩy** / **Xóa**)
- Nắm tay / hạ ngón trỏ → nhấc bút. Ảnh lật gương, nét làm mượt, canvas giữ tới khi xóa.

**Desktop (giống định dạng video nhất):**
```bat
python -m shape_match.paint_app
python -m shape_match.paint_app --thickness 18 --eraser 60
```
Phím: `C` xóa · `S` lưu PNG · `Q`/`Esc` thoát.

**Web:** chép `shape_match/web/paint.html` vào `frontend/` rồi mở
`http://127.0.0.1:8000/static/paint.html` (chạy hoàn toàn ở trình duyệt, **không cần
backend**). Nút: Bật camera · Xóa hết · Lưu ảnh.

> Lưu ý trung thực: bản gốc trên video tôi **không mở được** để đối chiếu từng chi tiết,
> nên bộ cử chỉ trên là mô hình chuẩn của thể loại đó (gần như chắc chắn trùng). Nếu video
> bạn xem dùng cử chỉ khác (vd "chụm ngón cái–trỏ để vẽ", hay vẽ chỉ bằng 1 ngón không cần
> ngón giữa để chọn), nói tôi một câu là chỉnh lại trong vài phút.

---

## 5. Tinh chỉnh (MatchConfig)

```python
from shape_match import ShapeMatcher, MatchConfig
cfg = MatchConfig(
    pass_threshold=75,   # hạ để dễ hơn, nâng để gắt hơn
    tol=0.18,            # bán kính "gần mẫu" (chuẩn hóa); nhỏ = gắt
    sigma=0.45,          # độ gắt của quỹ đạo (DTW); nhỏ = gắt
    sigma_turn=0.58,     # độ gắt của góc/đường cong
    w_shape=1.0, w_coverage=0.8, w_precision=0.7, w_corner=0.7,
    rotate_invariant=False,  # True nếu cho phép xoay tự do (vd nhận dạng tự do)
)
matcher = ShapeMatcher(cfg)
r = matcher.score(user_points_xy, "star")
print(r.accuracy, r.passed, r.stars, r.message)
```

Thêm hình mẫu mới:

```python
from shape_match import templates
import numpy as np
# đường gấp khúc trong [0,1]; closed=True nếu là hình kín
templates.register("my_shape", np.array([[0.2,0.2],[0.8,0.2],[0.5,0.8]]), closed=True, label_vi="Hình của tôi")
```

---

## 6. Kiểm thử

```bat
python -m shape_match.demo_offline                 # bảng điểm các tình huống
python -m shape_match.tests.test_matcher           # 10 test (không cần pytest)
python -m pytest shape_match/tests/ -q             # nếu có pytest
```

---

## 7. Phụ thuộc

`numpy`, `scipy` (bắt buộc cho matcher); `opencv-python`, `mediapipe` (chỉ cho app camera).
Tất cả đã có trong `requirements.txt` của AirDrawVocab. Matcher vẫn chạy nếu thiếu
OpenCV/MediaPipe (Kalman tự fallback sang EMA, app camera mới cần camera).
