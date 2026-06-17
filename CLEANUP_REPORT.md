# Cleanup report

Bản này được dọn để phù hợp lưu trữ trên GitHub.

## Tổng quan

- File gốc: `deep_learning(2).zip`
- Dung lượng zip gốc: 11.20 MB
- Số file gốc: 74
- Số file giữ lại: 32
- Dữ liệu chưa nén bị loại bỏ: 18.48 MB

## Đã loại bỏ

### Large training dataset
- Số mục: 20
- Dung lượng chưa nén: 15.63 MB
- Ví dụ:
  - `data/npy_28/`
  - `data/npy_28/apple.npy`
  - `data/npy_28/baseball.npy`
  - `data/npy_28/book.npy`
  - `data/npy_28/bowtie.npy`
  - `data/npy_28/diamond.npy`
  - `data/npy_28/dog.npy`
  - `data/npy_28/door.npy`

### Generated training/evaluation image
- Số mục: 7
- Dung lượng chưa nén: 1.66 MB
- Ví dụ:
  - `assets/results/`
  - `assets/results/baseline_confusion_matrix.png`
  - `assets/results/confusion_matrix.png`
  - `assets/results/multiple_predictions.png`
  - `assets/results/prediction_sample.png`
  - `assets/results/sample_drawings.png`
  - `assets/results/training_history.png`

### Older model; kept best advanced .keras model
- Số mục: 1
- Dung lượng chưa nén: 0.69 MB
- Ví dụ:
  - `models/airdrawvocab_enhanced_model.h5`

### Non-source report deliverable
- Số mục: 2
- Dung lượng chưa nén: 0.38 MB
- Ví dụ:
  - `report/`
  - `report/final_report.pdf`

### Python cache/bytecode
- Số mục: 7
- Dung lượng chưa nén: 0.09 MB
- Ví dụ:
  - `backend/__pycache__/`
  - `backend/__pycache__/app.cpython-311.pyc`
  - `__pycache__/`
  - `__pycache__/ai_assistant.cpython-311.pyc`
  - `__pycache__/config.cpython-311.pyc`
  - `__pycache__/face_auth.cpython-311.pyc`
  - `__pycache__/vocab_data.cpython-311.pyc`

### Non-source presentation deliverable
- Số mục: 2
- Dung lượng chưa nén: 0.02 MB
- Ví dụ:
  - `slides/`
  - `slides/presentation.pptx`

### Non-source planning deliverable
- Số mục: 2
- Dung lượng chưa nén: 0.01 MB
- Ví dụ:
  - `plan/`
  - `plan/project_plan.xlsx`

### Generated training/evaluation report or log
- Số mục: 8
- Dung lượng chưa nén: 0.01 MB
- Ví dụ:
  - `assets/reports/`
  - `assets/reports/advanced_training/`
  - `assets/reports/advanced_training/model_comparison.csv`
  - `assets/reports/advanced_training/model_comparison.json`
  - `assets/reports/baseline_classification_report.txt`
  - `assets/reports/baseline_confusion_matrix.csv`
  - `assets/reports/baseline_metrics.csv`
  - `assets/reports/baseline_metrics.json`

### Runtime log
- Số mục: 2
- Dung lượng chưa nén: 0.00 MB
- Ví dụ:
  - `assets/reports/server_uvicorn.err.log`
  - `assets/reports/server_uvicorn.log`

### Submission metadata/TODO link; not source
- Số mục: 1
- Dung lượng chưa nén: 0.00 MB
- Ví dụ:
  - `github_link.txt`

### Duplicate categories.json; kept models/categories.json
- Số mục: 1
- Dung lượng chưa nén: 0.00 MB
- Ví dụ:
  - `data/categories.json`

## Đã giữ lại

- Source Python, FastAPI backend, frontend HTML/CSS/JS.
- `requirements.txt`, `.gitignore`, batch scripts.
- `models/airdrawvocab_best_advanced.keras` và `models/categories.json` để demo nhận diện mà không cần train lại.
- Metadata dataset nhỏ: `data/dataset_info.json`, `data/dataset_summary.csv`.

## Gợi ý GitHub

- Không commit `data/npy_28/`, cache, log và output training.
- Nếu model sau này lớn hơn nhiều, nên đưa model lên GitHub Releases hoặc dùng Git LFS.
- Nếu muốn train lại, tải/copy dataset QuickDraw `.npy` vào `data/npy_28/` local rồi chạy script train.
