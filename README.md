# AirDrawVocab Unified AI

> **Ghi chú bản GitHub nhẹ:** bản zip này đã được dọn để lưu trữ/push lên GitHub. Các file cache, log, dataset `.npy`, ảnh/report kết quả train, slide, plan và model cũ đã bị loại bỏ. Bản này vẫn giữ source code, frontend/backend, requirements và model tốt nhất `models/airdrawvocab_best_advanced.keras` để demo nhận diện. Nếu muốn train lại, hãy tải/copy dataset QuickDraw `.npy` vào `data/npy_28/`.

## Chạy nhanh bản hoàn chỉnh

Trên Windows, mở CMD/PowerShell tại thư mục project rồi chạy theo thứ tự:

```bat
setup_training_env.bat
python check_environment.py
run_web_chatbot.bat
```

Sau đó mở:

```text
http://127.0.0.1:8000
```

Model chính hiện tại là `models/airdrawvocab_best_advanced.keras` (kiến trúc VGG-style CNN, 19 lớp). Model hiện tại được train với nhiều dữ liệu hơn (3.500 mẫu/lớp) và đánh giá có Test-Time Augmentation, đạt **test accuracy ≈ 93.9%, macro-F1 ≈ 0.94, Top-3 ≈ 97.2%** (19 lớp). Backend đã bật TTA sẵn (`USE_TTA=1`) nên nhận diện thực tế ổn định hơn. Thứ tự nhãn trong `models/categories.json` đã được đồng bộ đúng với model. Nếu chỉ muốn dùng project để demo hoặc nộp bài, bạn không cần train lại.

> **Lưu ý về số liệu (đã thống nhất, tránh mâu thuẫn):** các bản tài liệu cũ từng ghi 85.66%, 94.3% hoặc 99.33% với các thiết lập đo khác nhau (số lớp, số mẫu test, script train khác nhau). Con số chính thức của báo cáo là **≈ 90.4%** vì nó được đo lại trên cùng một split cố định và **so sánh công bằng** với baseline + nhiều kiến trúc khác (xem `assets/reports/benchmark_runs/`). Số 99.33% trước đây không tái lập được nên không dùng làm kết quả chính.

> **Quan trọng (đã sửa lỗi nhận diện):** model cũ trong bản zip ban đầu bị "suy biến" — nó đoán hầu hết hình vẽ thành `apple` do file `categories.json` lệch thứ tự nhãn so với model và pipeline train cũ (`advanced_train_model.py`) bị lỗi `BatchNormalization` trên TensorFlow 2.21/Keras 3 (train accuracy cao nhưng val/inference ~ngẫu nhiên). Đã khắc phục bằng `train_clean.py` (bỏ BatchNorm, augmentation trong tf.data) và đồng bộ lại nhãn. Xem mục 10.

Dự án này đã tổng hợp từ 2 source code bạn gửi thành một bản chung gồm:

1. **Game desktop AirDrawVocab**: vẽ bằng tay qua webcam/MediaPipe hoặc vẽ bằng chuột nếu máy không mở được camera.
2. **Nhận diện hình vẽ bằng CNN**: ưu tiên model `models/airdrawvocab_best_advanced.keras`, 19 lớp QuickDraw.
3. **Chatbot giải thích từ vựng**: web API trả về tên tiếng Anh, nghĩa tiếng Việt, top dự đoán, câu ví dụ và gợi ý vẽ rõ hơn.
4. **Nhận diện/xác thực khuôn mặt**: đăng ký khuôn mặt, train LBPH offline, xác thực bằng webcam hoặc qua web frontend.
5. **Camera Game Mode trên web**: dùng MediaPipe Hands JavaScript để lấy đầu ngón trỏ từ camera, vẽ nét cyan trên video, đoán realtime theo kiểu QuickDraw và hiện ảnh minh họa của vật thể AI đang thấy.
6. **Script train**: vẫn có đủ script train/evaluate; dataset `.npy` không đóng gói trong bản GitHub nhẹ, cần đặt lại vào `data/npy_28/` nếu train lại.
7. **Làm rõ ảnh vẽ và tạo ảnh tham khảo thực tế**: web trả thêm ảnh line-art rõ nét; nếu có `OPENAI_API_KEY` có thể tạo ảnh photorealistic theo nhãn nhận diện.
8. **Advanced model zoo**: có script train tối ưu với ResNet-style CNN cho sketch 28x28 và tùy chọn so sánh EfficientNetV2B0/MobileNetV3Small.

---

## 1. Cấu trúc dự án

```text
AirDrawVocab_Unified_AI/
├── backend/
│   ├── app.py                         # FastAPI: nhận diện hình vẽ + chatbot + face API
│   └── .env.example                   # Mẫu cấu hình API key
├── frontend/
│   ├── index.html                     # Web vẽ hình + nhận diện mặt
│   ├── app.js
│   └── style.css
├── models/
│   ├── airdrawvocab_best_advanced.keras  # Model CNN chính (VGG-style, ~90.4% test acc)
│   └── categories.json                   # 19 nhãn
├── data/
│   ├── dataset_info.json              # Metadata dataset
│   ├── dataset_summary.csv            # Tóm tắt dataset
│   └── npy_28/                        # Tự thêm QuickDraw .npy nếu muốn train lại
├── assets/                           # Tự sinh lại results/reports khi train/evaluate
├── face_data/
│   └── samples/                       # Mẫu khuôn mặt sau khi đăng ký
├── docs/                              # Tài liệu hướng dẫn bổ sung
├── game.py                            # Game desktop Pygame
├── src/                               # Mã nguồn ML (Phase 1 - chuẩn hóa cấu trúc)
│   ├── training/                      # train_clean, train_best, advanced_train_model,
│   │                                  #   train_model, baseline_model, train_stroke_model,
│   │                                  #   self_improve_retrain
│   ├── evaluation/                    # evaluate_model, error_analysis, compare_models
│   ├── data/                          # data_utils
│   └── utils/                         # mlflow_utils, repro, model_versioning
├── face_auth.py                       # Module nhận diện khuôn mặt (LBPH)
├── face_cli.py                        # CLI đăng ký/xác thực mặt
├── ai_assistant.py                    # Chatbot offline trong game
├── vocab_data.py                      # Nghĩa, IPA, ví dụ từ vựng
├── check_environment.py               # Kiểm tra môi trường
├── smoke_test_project.py              # Smoke test API
├── launcher.py                        # Menu chạy nhanh
├── requirements.txt
└── *.bat / *.sh                       # Scripts chạy nhanh trên Windows/Linux
```

---

## 2. Môi trường khuyến nghị

Nên dùng **Python 3.11 hoặc 3.12 64-bit**.
Không khuyến nghị Python 3.13 vì TensorFlow và MediaPipe thường chưa ổn định trên bản này, dễ gặp lỗi kiểu `mediapipe has no attribute solutions`.

### Cài nhanh trên Windows

Mở CMD/PowerShell trong thư mục dự án:

```bat
setup_training_env.bat
```

Hoặc cài thủ công:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip uninstall -y opencv-python opencv-contrib-python
pip install -r requirements.txt
```

> Dự án dùng `opencv-contrib-python` để có `cv2.face.LBPHFaceRecognizer_create` cho nhận diện khuôn mặt. Không nên cài đồng thời lung tung nhiều bản OpenCV.

---

## 3. Cách chạy dự án chung

### Cách 1: Chạy menu tổng hợp

```bat
python launcher.py
```

Menu có các lựa chọn:

- Chạy game desktop.
- Chạy game desktop có xác thực khuôn mặt.
- Đăng ký khuôn mặt.
- Test xác thực khuôn mặt.
- Chạy web chatbot API.

### Cách 2: Chạy game desktop

```bat
python game.py
```

Nếu muốn bắt buộc xác thực khuôn mặt trước khi vào game:

```bat
python game.py --face-login
```

Nếu muốn kiểm tra đúng một người cụ thể:

```bat
python game.py --face-login --face-user thien
```

### Cách 3: Chạy web chatbot + nhận diện mặt

```bat
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Sau đó mở:

```text
http://127.0.0.1:8000
```

Trên web có 2 phần:

- Nhận diện khuôn mặt: bật/tắt camera, đăng ký, xác thực.
- Bảng vẽ: vẽ hình, bấm nhận diện, chatbot giải thích kết quả, xem ảnh vẽ đã làm rõ nét và tạo ảnh tham khảo thực tế.
- Camera Game Mode: vẽ bằng ngón trỏ với nét cyan đã làm mượt, chỉnh độ dày/độ mượt, xem outline gợi ý theo từ `Draw` và AI đoán liên tục như QuickDraw.

---

## 4. Đăng ký và xác thực khuôn mặt

### Đăng ký mặt bằng webcam

```bat
python face_cli.py enroll thien
```

Nên đăng ký **35-50 mẫu**. Khi cửa sổ camera hiện lên, nhìn thẳng, xoay nhẹ trái/phải để hệ thống lấy nhiều góc mặt.

### Xác thực thử

```bat
python face_cli.py verify --username thien
```

Kết quả dạng:

```text
Xác thực THÀNH CÔNG: dự đoán=thien, score_trung_bình=..., ngưỡng=70.00
```

Với LBPH, **score càng thấp càng giống**. Ngưỡng mặc định là `70`. Nếu nhận sai người, giảm ngưỡng xuống 55-65 trong `FaceAuthManager(threshold=...)`.

---

## 5. Luồng xử lý tổng hợp

### Desktop game

```text
Webcam/chuột
→ MediaPipe lấy đầu ngón tay hoặc chuột vẽ nét
→ Canvas nhị phân nền đen nét trắng
→ Crop vùng có nét + padding
→ Resize 28x28
→ CNN dự đoán 19 lớp từ vựng
→ Hiển thị nghĩa tiếng Việt, IPA, ví dụ, phát âm TTS
→ AI assistant đưa gợi ý/hỗ trợ
```

### Web chatbot

```text
Canvas web
→ Upload ảnh PNG lên /predict
→ Backend tiền xử lý ảnh: grayscale, đảo màu nếu cần, crop nét, resize 28x28
→ CNN predict
→ Top 3 dự đoán
→ Chatbot trả lời: nhãn, nghĩa, ví dụ, gợi ý vẽ
→ Trả thêm ảnh vẽ đã làm rõ nét
→ Người dùng bấm Tạo ảnh để gọi /image/generate tạo ảnh tham khảo thực tế
```

### Face recognition

```text
Webcam/ảnh khuôn mặt
→ Haar Cascade phát hiện mặt lớn nhất
→ Crop + resize 200x200 + cân bằng sáng Histogram Equalization
→ Lưu nhiều mẫu theo user
→ Train LBPH
→ Verify bằng score trung bình nhiều frame
→ Cho phép mở game hoặc xác nhận danh tính trên web
```

---

## 6. Các điểm đã tối ưu để tăng độ chính xác

- Dùng model enhanced làm model chính thay vì model cũ.
- Tiền xử lý ảnh vẽ có crop vùng nét, padding và đưa ảnh về vuông trước khi resize, giúp hình không bị méo.
- Có pipeline làm rõ ảnh vẽ: crop nét, upscale Lanczos, morphology close, threshold lại và xuất PNG rõ nét để người học nhìn được bản vẽ sạch hơn.
- Có ảnh tham khảo thực tế theo nhãn: offline fallback bằng PIL để demo không cần key, hoặc photorealistic generation nếu cấu hình OpenAI Image API.
- Có top-3 prediction để người học biết các nhãn gần giống.
- Với khuôn mặt: dùng nhiều mẫu đăng ký, histogram equalization, lấy mặt lớn nhất, verify bằng nhiều frame rồi lấy điểm trung bình.
- Có threshold để tránh nhận nhầm khi score quá cao.
- Camera Game Mode có outline gợi ý theo từng nhãn, lọc mượt điểm ngón tay, vẽ đường cong bo tròn và nút tắt webcam khi không dùng.
- Game vẫn fallback sang vẽ chuột nếu camera/MediaPipe lỗi, tránh chết chương trình khi demo.

Không thể cam kết chính xác tuyệt đối 100% vì độ chính xác phụ thuộc camera, ánh sáng, chất lượng nét vẽ, số mẫu khuôn mặt và môi trường chạy. Tuy nhiên bản này đã thiết kế theo hướng ổn định và dễ demo nhất trong điều kiện local/offline.

---

## 7. Bật chatbot Foza/Claude nếu có API key

Mặc định chatbot nội bộ vẫn chạy không cần key. Nếu muốn gọi model ngoài, tạo file:

```text
backend/.env
```

Theo mẫu `backend/.env.example` rồi chạy lại backend.

---

## 8. Bật tạo ảnh photorealistic nếu có OpenAI API key

Mặc định web vẫn chạy offline và tạo ảnh tham khảo bằng PIL. Nếu muốn ảnh giống ngoài đời thực hơn, tạo `backend/.env` từ `backend/.env.example` rồi điền:

```text
OPENAI_API_KEY=sk-...
OPENAI_IMAGE_ENABLED=1
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_SIZE=1024x1024
OPENAI_IMAGE_QUALITY=medium
```

Sau khi nhận diện hình vẽ trên web, bấm **Tạo ảnh** ở phần “Ảnh tham khảo thực tế”.

Các endpoint liên quan:

```text
POST /image/enhance   # làm rõ nét ảnh vẽ, chạy offline
POST /image/generate  # tạo ảnh tham khảo theo nhãn, dùng OpenAI nếu có key, fallback offline nếu không có key
```

Lưu ý: tạo ảnh photorealistic cần internet, API key hợp lệ và có thể mất nhiều thời gian hơn nhận diện CNN.

---

## 9. Train, baseline và đánh giá theo yêu cầu capstone

> Bản GitHub nhẹ không kèm dataset train. Trước khi chạy `baseline_model.py`, `train_model.py` hoặc `advanced_train_model.py`, hãy tạo thư mục `data/npy_28/` và đặt các file QuickDraw `.npy` theo đúng 19 nhãn trong `config.py`.


Chạy baseline truyền thống để có mốc so sánh với CNN:

```bat
python src/training/baseline_model.py
```

Baseline dùng `NearestCentroid` trên vector pixel 28x28 và xuất kết quả vào:

```text
assets/reports/baseline_metrics.json
assets/reports/baseline_metrics.csv
assets/reports/baseline_classification_report.txt
assets/results/baseline_confusion_matrix.png
```

Train lại CNN và xuất đầy đủ báo cáo thực nghiệm:

```bat
python src/training/train_model.py
```

Script train mới dùng split cố định **800 train / 150 validation / 150 test mỗi lớp**, có `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` và lưu:

```text
models/airdrawvocab_best_model.h5
models/airdrawvocab_retrained_model.h5
assets/reports/training_log.csv
assets/reports/metrics_summary.csv
assets/reports/classification_report.*
assets/reports/error_analysis.csv
assets/results/confusion_matrix.png
```

Lưu ý: nên train/chạy demo bằng **Python 3.11 hoặc 3.12 64-bit**. Python 3.13 thường chưa phù hợp với TensorFlow/MediaPipe.

---

## 10. Train lại model (khuyến nghị dùng `train_clean.py`)

> **Khuyến nghị mới:** dùng `train_clean.py`. Script `advanced_train_model.py` bị lỗi `BatchNormalization` trên TensorFlow 2.21/Keras 3 khiến val/inference accuracy ~ngẫu nhiên dù train accuracy cao (đây là nguyên nhân model cũ nhận diện sai). `train_clean.py` đã bỏ BatchNorm, đưa augmentation vào tf.data, gán nhãn đúng thứ tự `config.CATEGORIES` và tự đồng bộ `models/categories.json`.

### Bước 1: tải dữ liệu QuickDraw (nếu chưa có)

```bat
.\.venv311\Scripts\python.exe dev_download_quickdraw.py --per-class 8000
```

Lệnh này tải một phần nhỏ mỗi lớp (~6MB/lớp) vào `data/npy_28/`, nhanh hơn nhiều so với tải full `.npy`.

### Cách MỚI (khuyến nghị) — train độ chính xác cao, đồng bộ với notebook Colab

Script `src/training/train_highacc.py` dùng đúng kiến trúc trong notebook Colab
(`AirDrawVocab_train_highacc_colab.ipynb`): VGG + BatchNorm, label smoothing,
cosine LR, augmentation và TTA khi đánh giá.

```bat
# GPU (khuyến nghị, ~96–98%):
python src/training/train_highacc.py --per-class 12000 --epochs 40 --batch 512

# CPU (nhanh hơn, ~93–94%, nên thêm --no-bn để ổn định trên Keras 3):
python src/training/train_highacc.py --per-class 3500 --epochs 14 --batch 256 --no-bn
```

Để đạt **Top-1 ≥ 97%** chắc chắn, hãy chạy notebook `AirDrawVocab_train_highacc_colab.ipynb`
trên Google Colab GPU (Runtime → GPU → Run all), rồi chép `airdrawvocab_best_advanced.keras`
và `categories.json` tải về vào thư mục `models/`.

### Bước 2: train (đã tối ưu tốc độ, ~8–12 phút trên CPU)

```bat
.\.venv311\Scripts\python.exe src/training/train_clean.py
```

Mặc định: 2000 mẫu/lớp, batch 512, early stopping sớm, dùng hết nhân CPU → train nhanh mà vẫn đạt ~93–94%. Muốn chính xác hơn (chậm hơn) thì tăng dữ liệu:

```bat
.\.venv311\Scripts\python.exe src/training/train_clean.py --train-per-class 4000 --val-per-class 600 --test-per-class 600 --epochs 22
```

Sau khi train xong, model lưu vào `models/airdrawvocab_best_advanced.keras` và `models/categories.json` được đồng bộ tự động. Khởi động lại web để nạp model mới.

### (Tham khảo) script nâng cao cũ — KHÔNG khuyến nghị do còn lỗi BatchNorm

```bat
python src/training/advanced_train_model.py --model resnet_sketch --epochs 60 --batch-size 64
```

Nếu dùng Windows, cách dễ nhất là chạy:

```bat
setup_training_env.bat
python src/training/advanced_train_model.py --model resnet_sketch --epochs 60 --batch-size 64
```

Model mặc định `resnet_sketch` là residual CNN tối ưu cho dữ liệu QuickDraw 28x28 grayscale. Đây là lựa chọn phù hợp nhất cho project vì:

- Dữ liệu là sketch nhỏ 28x28, không phải ảnh tự nhiên lớn.
- Residual connection giúp train sâu hơn CNN thường mà vẫn ổn định.
- Có augmentation xoay/dịch/zoom, AdamW, label smoothing, dropout, early stopping và checkpoint.
- Sau khi train xong, app tự ưu tiên `models/airdrawvocab_best_advanced.keras` nếu file này tồn tại.

Nếu muốn so sánh với các model phổ biến hiện nay:

```bat
python src/training/advanced_train_model.py --model all --epochs 40 --batch-size 64
```

Các model được hỗ trợ:

```text
resnet_sketch       # khuyến nghị cho deploy/game/web hiện tại
efficientnetv2b0    # model phổ biến, mạnh, có thể dùng pretrained ImageNet nếu tải được weights
mobilenetv3small    # model nhẹ, phổ biến cho mobile/edge
```

File kết quả:

```text
models/airdrawvocab_best_advanced.keras
models/resnet_sketch_best.keras
assets/reports/advanced_training/model_comparison.csv
assets/reports/advanced_training/<run_id>/metrics_summary.json
assets/reports/advanced_training/<run_id>/classification_report.txt
assets/reports/advanced_training/<run_id>/confusion_matrix.png
assets/reports/advanced_training/<run_id>/error_analysis.csv
```

Gợi ý tối ưu nếu có GPU:

```bat
python src/training/advanced_train_model.py --model resnet_sketch --epochs 100 --batch-size 128 --label-smoothing 0.03 --dropout 0.35
```

Nếu thấy validation accuracy dao động hoặc overfit, thử bật MixUp:

```bat
python src/training/advanced_train_model.py --model resnet_sketch --epochs 100 --mixup-alpha 0.1
```

Lưu ý thực tế: không có model nào cam kết “cao nhất tuyệt đối” trên mọi lần train. Cách tốt nhất là train nhiều cấu hình, so sánh validation/test metrics và dùng checkpoint tốt nhất.

### Kết quả advanced train đã chạy

Môi trường `.venv311` đã được tạo và kiểm tra thành công. Model mới đã được train bằng:

```bat
.\.venv311\Scripts\python.exe src/training/advanced_train_model.py --model resnet_sketch --epochs 12 --batch-size 128 --patience 5 --deploy-threshold 0.99
```

Kết quả đánh giá độc lập:

```text
Model: models/airdrawvocab_best_advanced.keras
Accuracy: ~90.4%      (benchmark tái lập, 15.200 mẫu test, 19 lớp)
Top-3 Accuracy: ~96.0%
Macro F1-score: ~0.90
Lưu ý: con số 99.33% trong các bản cũ KHÔNG tái lập được, không dùng làm kết quả chính.
```

So với model cũ `airdrawvocab_enhanced_model.h5`:

```text
So sánh kiến trúc (cùng split): baseline cổ điển ~61% < SmallCNN ~86.9% < ResNet-Sketch ~89.2% < VGG-style CNN ~90.4% (tốt nhất).
```

Chi tiết nằm ở:

```text
report/advanced_training_addendum.md
assets/reports/advanced_training/model_comparison.csv
assets/reports/evaluations/
```

Kiểm tra nhanh toàn project sau khi setup/train:

```bat
.\.venv311\Scripts\python.exe smoke_test_project.py
```

Smoke test sẽ kiểm tra `/health`, `/predict` và `/image/generate`.

---

## 11. Tài liệu nộp bài

Trong bản GitHub nhẹ, các file nộp bài như `report/final_report.pdf`, `slides/presentation.pptx`, `plan/project_plan.xlsx` và các output training đã được loại bỏ để repo gọn hơn. Nếu cần nộp hoặc trình bày, hãy lưu các file này ở release riêng, Google Drive hoặc thư mục ngoài repo.


---

## 12. Lỗi thường gặp

### `mediapipe has no attribute solutions`

Thường do môi trường Python/MediaPipe bị lệch. Cách xử lý:

```bat
pip uninstall -y mediapipe opencv-python opencv-contrib-python
pip install -r requirements.txt
```

Nên dùng Python 3.11/3.12.

### `cv2 has no attribute face`

Bạn đang dùng `opencv-python` thường. Cài lại:

```bat
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### Không mở được webcam

- Tắt app khác đang dùng camera.
- Bật quyền Camera trong Windows Settings.
- Thử camera index khác:

```bat
python game.py --camera 1
python face_cli.py verify --camera 1
```

---

## 13. Gợi ý demo bài báo cáo

1. Mở web: đăng ký khuôn mặt `thien`, xác thực thành công.
2. Vẽ `apple`, bấm nhận diện, chatbot giải thích nghĩa và ví dụ.
3. Mở desktop game bằng `python game.py --face-login --face-user thien`.
4. Vẽ bằng ngón tay/chuột, model nhận diện realtime và cho điểm.
5. Trình bày kiến trúc: CNN nhận diện hình vẽ + chatbot giải thích + face auth xác thực người học.

## Cập nhật UI theo ảnh tham khảo

Bản này đã phát triển thêm dashboard, đăng ký/đăng nhập SQLite, nhận diện ảnh upload, tab kết quả, đọc thông tin bằng giọng nói trình duyệt và khu vực demo/analytics. Chi tiết xem `UI_REFERENCE_DEVELOPMENT_NOTES.md`. Các chức năng này bám theo dự án AirDrawVocab 40 từ vựng, không chuyển hướng sang đề tài nhận diện bệnh mắt.


## Production Deep Learning Upgrade

Bản cập nhật này bổ sung pipeline nhận dạng chuyên nghiệp hơn cho Final Boss Mode:

```bash
python src/data/make_real_user_benchmark.py --out data/benchmark/release_v1
python src/training/train_image_model.py --config configs/image_resnet_sketch.yaml
python src/training/train_stroke_model.py --epochs 18
python src/evaluation/evaluate_release.py --benchmark data/benchmark/release_v1 --out assets/reports/releases/current
python src/evaluation/calibrate_release.py --benchmark data/benchmark/release_v1
python src/training/promote_candidate.py --dry-run --allow-if-weak-data
```

Trong giao diện web, khối **Production AI Ops** có thể build benchmark, evaluate release và kiểm tra promotion ngay sau khi chơi/lưu mẫu train.

Chi tiết xem `PRODUCTION_DEEP_LEARNING_UPGRADE.md` và `PROFESSIONAL_COMPLETION_SUMMARY.md`.
