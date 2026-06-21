"""
AirDrawVocab - dữ liệu từ vựng dùng cho game desktop.

Nguồn duy nhất là vocab_pairs.py. File này giữ API cũ VOCAB_DATA/CATEGORIES
nhưng không còn hard-code 19 từ; hiện dự án định nghĩa 40 từ vựng.
"""
from __future__ import annotations

try:
    from vocab_pairs import VOCAB, CATEGORIES
except Exception:
    VOCAB = {}
    CATEGORIES = []


def _to_legacy_item(item: dict) -> dict:
    return {
        "vietnamese": item.get("vi", ""),
        "ipa": item.get("ipa", ""),
        "example": item.get("ex", ""),
        "example_vi": item.get("ex_vi", ""),
        "hint": item.get("hint", ""),
    }


VOCAB_DATA = {label: _to_legacy_item(data) for label, data in VOCAB.items()}
NUM_VOCAB = len(CATEGORIES)
