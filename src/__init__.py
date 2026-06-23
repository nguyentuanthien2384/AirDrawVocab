"""Gói mã nguồn chính của AirDrawVocab (Phase 1 - chuẩn hóa cấu trúc).

Cấu trúc:
    src/training/    - các script huấn luyện model
    src/evaluation/  - đánh giá & so sánh model
    src/data/        - tải/chia dữ liệu
    src/utils/       - tiện ích (MLflow, reproducibility, versioning)
    src/inference/   - (dành cho code dự đoán dùng lại)
    src/models/      - (dành cho định nghĩa kiến trúc tách riêng)

Lưu ý: các module dùng chung cho web/desktop (config, vocab_pairs, data_utils
nguồn, airdraw_models, ai_assistant, face_auth, sample_generator) vẫn được giữ ở
thư mục gốc dự án để không phá vỡ backend/game/deploy.
"""
