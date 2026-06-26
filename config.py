"""
Cấu hình tập trung cho toàn bộ dự án AirDrawVocab.
Tất cả hằng số dùng chung (categories, split, paths) được định nghĩa ở đây.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ============================================================================
# NHÃN LỚP (CATEGORIES)
# ----------------------------------------------------------------------------
# Nguyên tắc thống nhất (đã khắc phục mâu thuẫn 19 vs 40):
#   - CATEGORIES = nhãn THỰC TẾ mà model đang deploy đã được train, lấy từ
#     models/categories.json. Đây là nguồn chuẩn cho mọi script train/evaluate
#     để luôn khớp với artifact (.keras) và dữ liệu trong data/npy_28/.
#   - VOCAB_CATEGORIES = toàn bộ từ vựng định nghĩa trong vocab_pairs.py
#     (hiện 40 từ, dùng cho tra nghĩa/IPA/ví dụ ở backend & demo). Đây là
#     "hướng mở rộng": muốn nhận diện đủ 40 lớp thì phải bổ sung dữ liệu các
#     lớp mới vào data/npy_28/ rồi train lại, sau đó categories.json sẽ tự cập nhật.
# ============================================================================
import json as _json

# 1) Toàn bộ từ vựng (để tra nghĩa) — không nhất thiết = số lớp model.
try:
    from vocab_pairs import CATEGORIES as _VOCAB_CATEGORIES
    VOCAB_CATEGORIES = list(_VOCAB_CATEGORIES)
except Exception:
    VOCAB_CATEGORIES = []

# 2) Nhãn model thực tế (nguồn chuẩn cho train/evaluate).
_FALLBACK_19 = [
    "apple", "baseball", "book", "bowtie", "diamond",
    "dog", "door", "envelope", "eye", "fish",
    "hat", "leaf", "lightning", "moon", "pants",
    "scissors", "square", "star", "t-shirt",
]
_CATEGORIES_JSON = ROOT / "models" / "categories.json"
try:
    CATEGORIES = _json.loads(_CATEGORIES_JSON.read_text(encoding="utf-8"))
    assert isinstance(CATEGORIES, list) and len(CATEGORIES) > 0
except Exception:
    CATEGORIES = list(VOCAB_CATEGORIES) if VOCAB_CATEGORIES else list(_FALLBACK_19)

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
