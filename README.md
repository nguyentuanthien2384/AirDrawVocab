<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.6-3776AB?logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)

**Đồ án môn học Mạng Noron và Học sâu**

</div>

---

## 👥 Thông tin nhóm

|             | Thông tin              |
| ----------- | ---------------------- |
| **Nhóm**    | Nhóm 16               |
| **Môn học** | Deep Learning |

### Thành viên

| STT | Họ và tên         |   MSSV   |
| :-: | ----------------- | :------: |
|  1  | Nguyễn Tuấn Thiền | 23010571 |
|  2  | Đặng Việt Anh     | 23010689 |

---

# AirDrawVocab — Unified AI Vocabulary Learning Game

**AirDrawVocab** là một ứng dụng học từ vựng tiếng Anh tương tác độc đáo, kết hợp giữa việc vẽ hình (bằng chuột hoặc vẽ tay trong không gian qua camera) và trí tuệ nhân tạo (AI) để nhận diện hình vẽ và chấm điểm. Dự án còn tích hợp hệ thống xác thực khuôn mặt (Face Recognition) bảo mật cho người chơi và một quy trình MLOps/AI Ops hiện đại để huấn luyện, đánh giá, hiệu chuẩn và vận hành các mô hình học sâu (Deep Learning).

---

## 🌟 Các Tính Năng Chính

### 1. Trải Nghiệm Chơi Game Đa Dạng
*   **Desktop Game (Pygame):** Trải nghiệm game mượt mà với giao diện đồ họa Pygame, tích hợp giọng đọc tiếng Anh (pyttsx3) để hỗ trợ học phát âm từ vựng.
*   **Vẽ chuột (Mouse Drawing):** Chế độ mặc định sắc nét (hỗ trợ HiDPI) và làm mượt nét vẽ, chơi trực tiếp trên trình duyệt.
*   **Vẽ tay qua Camera (Air Drawing):** Dùng webcam để vẽ trong không trung thông qua thư viện MediaPipe nhận diện cử chỉ bàn tay:
    *   *Ngón trỏ duỗi:* Vẽ nét.
    *   *Hai ngón (trỏ + giữa) duỗi:* Chọn màu hoặc tẩy/xóa.
*   **Tô theo hình mẫu (Shape Tracing):** Đồ theo đường nét mẫu và tính điểm độ khớp (%) chính xác cao bằng các thuật toán so khớp hình học.

### 2. Xác Thực Khuôn Mặt (Face Authentication)
*   Đăng ký tài khoản người dùng bằng cách quét khuôn mặt trực tiếp qua camera.
*   Đăng nhập game thông qua nhận diện khuôn mặt (sử dụng thuật toán OpenCV LBPH Face Recognizer) để bảo mật thông tin và lịch sử chơi.

### 3. Pipeline MLOps & Production AI Ops
*   **Mô hình Nhận diện Kép (Dual AI Engine):**
    *   *Image Model (CNN/ResNet/MobileNetV2):* Phân loại hình vẽ từ canvas ảnh (28x28).
    *   *Stroke Model (BiGRU/Transformer):* Phân loại chuỗi nét vẽ động dựa trên tọa độ thời gian thực.
*   **Experiment Tracking:** Quản lý và theo dõi toàn bộ quá trình train (hyperparameters, metrics, artifacts) bằng **MLflow**.
*   **Model Versioning:** Tự động đánh phiên bản và lưu siêu dữ liệu (metadata) cho các model được huấn luyện.
*   **Temperature Calibration:** Hiệu chuẩn độ tự tin của AI trên tập calibration để tránh hiện tượng model đoán sai nhưng có xác suất tự tin quá cao.
*   **Promotion Gate:** Quy trình tự động đánh giá và cập nhật model challenger lên production nếu vượt qua benchmark quy chuẩn.
*   **Dashboard Giám Sát:** Giao diện quản trị viên giúp xây dựng tập benchmark, chạy đánh giá (evaluation) và thăng cấp model ngay trên nền tảng web.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
AirDrawVocab/
├── backend/                  # Mã nguồn FastAPI Web Server & API endpoints
├── frontend/                 # Giao diện Web Game (HTML, CSS, JS)
├── configs/                  # File cấu hình YAML cho training & benchmark
├── data/                     # Thư mục lưu SQLite DB, file .npy và tập benchmark
├── docs/                     # Quy định, chính sách và đặc tả API của dự án
├── models/                   # Lưu trữ các file model AI (.keras, .json)
├── shape_match/              # Module và ứng dụng phụ cho việc tô theo hình mẫu
├── src/                      # Mã nguồn Python cốt lõi của hệ thống AI
│   ├── data/                 # Xử lý dữ liệu và tạo benchmark
│   ├── training/             # Huấn luyện mô hình CNN, Stroke Transformer, BiGRU
│   ├── evaluation/           # Đánh giá, so sánh model và Hard Negative Mining
│   ├── inference/            # TTA (Test-Time Augmentation) và Reranker
│   ├── serving/              # Model Runtime layer và A/B Router
│   ├── monitoring/           # Prometheus metrics tracking
│   └── utils/                # Tiện ích MLflow, reproducibility và versioning
├── launcher.py               # Trình khởi chạy hợp nhất các thành phần
├── game.py                   # Game Desktop chính bằng Pygame
├── face_auth.py / face_cli.py # Hệ thống đăng ký & xác thực khuôn mặt
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── run_game.bat              # Script chạy nhanh Game Desktop
└── run_web_chatbot.bat       # Script chạy nhanh Web Server & Game Web
```

---

## ⚙️ Cài Đặt Môi Trường

### Yêu Cầu Hệ Thống
*   Khuyến nghị sử dụng **Python 3.11** hoặc **3.12 (64-bit)**.
*   *Lưu ý:* Tránh dùng Python 3.13 vì một số thư viện như TensorFlow hoặc MediaPipe chưa tương thích hoàn toàn.

### Các Bước Thực Hiện
Bạn có thể thiết lập nhanh bằng cách nhấp đúp vào file `setup_training_env.bat` trên Windows, hoặc chạy các lệnh sau trong terminal:

1.  **Tạo môi trường ảo:**
    ```bash
    python -m venv .venv311
    ```
2.  **Kích hoạt môi trường ảo:**
    *   *Windows (CMD/PowerShell):*
        ```powershell
        .venv311\Scripts\activate
        ```
    *   *Linux/macOS:*
        ```bash
        source .venv311/bin/activate
        ```
3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```
4.  **Kiểm tra môi trường:**
    ```bash
    python check_environment.py
    ```

---

## 🚀 Hướng Dẫn Chạy Ứng Dụng

Cách đơn giản nhất là chạy trình khởi chạy hợp nhất `launcher.py` để lựa chọn dịch vụ cần mở:
```bash
python launcher.py
```

### 1. Game Desktop (Pygame)
*   Để chạy game nhanh không có xác thực khuôn mặt:
    ```bash
    python game.py
    ```
    *(Hoặc chạy trực tiếp file `run_game.bat`)*
*   Để chạy game và bắt buộc đăng nhập bằng khuôn mặt:
    ```bash
    python game.py --face-login
    ```

### 2. Game Web & Chatbot API
Chạy file script `run_web_chatbot.bat` hoặc thực hiện lệnh:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Sau đó truy cập các đường dẫn trên trình duyệt:
*   **Trang chủ Game Web:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
*   **Vẽ tự do (Paint):** [http://127.0.0.1:8000/static/paint.html](http://127.0.0.1:8000/static/paint.html)
*   **Tô theo hình mẫu (Trace):** [http://127.0.0.1:8000/static/shape_trace.html](http://127.0.0.1:8000/static/shape_trace.html)

### 3. Xác Thực Khuôn Mặt (Face CLI)
*   **Đăng ký khuôn mặt mới:**
    ```bash
    python face_cli.py enroll <username>
    ```
*   **Kiểm tra nhận diện khuôn mặt thử:**
    ```bash
    python face_cli.py verify
    ```

---

## 📊 Vận Hành Quy Trình MLOps & Training

Dưới đây là các câu lệnh chính để phát triển và cập nhật mô hình AI trong dự án.

### 1. Tạo tập Benchmark từ người dùng thực tế
Trích xuất dữ liệu người chơi từ cơ sở dữ liệu SQLite để làm bộ kiểm thử cố định:
```bash
python src/data/make_real_user_benchmark.py --db data/airdrawvocab_app.sqlite3 --out data/benchmark/release_v1
```

### 2. Huấn Luyện Mô Hình
*   **Huấn luyện Image CNN Candidate:**
    ```bash
    python src/training/train_image_model.py --config configs/image_resnet_sketch.yaml
    ```
*   **Huấn luyện Stroke Transformer:**
    ```bash
    python src/training/train_stroke_transformer.py --epochs 20
    ```

### 3. Đánh Giá & Hiệu Chuẩn (Calibration)
*   **Đánh giá mô hình Challenger trên tập Benchmark:**
    ```bash
    python src/evaluation/evaluate_release.py --benchmark data/benchmark/release_v1 --out assets/reports/releases/current
    ```
*   **Hiệu chuẩn xác suất tự tin (Calibration):**
    ```bash
    python src/evaluation/calibrate_release.py --benchmark data/benchmark/release_v1
    ```

### 4. Xem Tiến Độ Trên MLflow Dashboard
Khởi động giao diện MLflow để so sánh các lần huấn luyện:
```bash
mlflow ui
```
Sau đó truy cập địa chỉ: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5. Thăng Cấp Mô Hình (Promotion Gate)
Kiểm tra xem mô hình mới có đạt chuẩn để thay thế mô hình cũ trên Production hay không:
*   Chạy thử (Dry run):
    ```bash
    python src/training/promote_candidate.py --dry-run --allow-if-weak-data
    ```
*   Thăng cấp chính thức:
    ```bash
    python src/training/promote_candidate.py --report assets/reports/releases/current/summary.json
    ```

---

## 🛠️ Công Nghệ Sử Dụng
*   **Backend:** FastAPI, Uvicorn, SQLite
*   **Frontend:** HTML5 Canvas, Vanilla CSS, JavaScript
*   **Desktop App:** Pygame, OpenCV, Pyttsx3
*   **AI/ML:** TensorFlow, MediaPipe, Scikit-learn, SciPy, NumPy, MLflow, Prometheus
