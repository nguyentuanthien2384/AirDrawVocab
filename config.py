"""
Cấu hình tập trung cho toàn bộ dự án AirDrawVocab.
Tất cả hằng số dùng chung (categories, split, paths) được định nghĩa ở đây.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Danh sách lớp từ vựng QuickDraw (mở rộng) — nguồn duy nhất ở vocab_pairs.py
# Hiện tại: 40 lớp. Thêm/bớt từ vựng -> sửa vocab_pairs.VOCAB rồi train lại.
try:
    from vocab_pairs import CATEGORIES as _VOCAB_CATEGORIES
    CATEGORIES = list(_VOCAB_CATEGORIES)
except Exception:
    # Fallback: 19 lớp gốc nếu thiếu vocab_pairs.py
    CATEGORIES = [
        "apple", "baseball", "book", "bowtie", "diamond",
        "dog", "door", "envelope", "eye", "fish",
        "hat", "leaf", "lightning", "moon", "pants",
        "scissors", "square", "star", "t-shirt",
    ]
NUM_CLASSES = len(CATEGORIES)

# Chia dữ liệu cố định
TRAIN_PER_CLASS = 800
VAL_PER_CLASS = 150
TEST_PER_CLASS = 150
SAMPLES_PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
RANDOM_STATE = 42

# Đường dẫn
DATA_DIR = ROOT / "data" / "npy_28"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "assets" / "results"
REPORTS_DIR = ROOT / "assets" / "reports"
CATEGORIES_PATH = MODELS_DIR / "categories.json"

# Ưu tiên model: advanced > basic
MODEL_CANDIDATES = [
    MODELS_DIR / "airdrawvocab_best_advanced.keras",
    MODELS_DIR / "airdrawvocab_best_model.h5",
    MODELS_DIR / "airdrawvocab_enhanced_model.h5",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[-1])
