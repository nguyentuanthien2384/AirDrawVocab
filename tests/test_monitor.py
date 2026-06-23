from src.evaluation.monitor import summarize

ROWS = [
    {"target": "cat", "predicted": "cat", "confidence": 0.9, "correct": 1, "mode": "mouse"},
    {"target": "cat", "predicted": "dog", "confidence": 0.85, "correct": 0, "mode": "mouse"},
    {"target": "dog", "predicted": "dog", "confidence": 0.4, "correct": 1, "mode": "hand"},
    {"target": "dog", "predicted": "cat", "confidence": 0.3, "correct": 0, "mode": "hand"},
]


def test_empty():
    assert summarize([]) == {"total": 0}


def test_basic_metrics():
    r = summarize(ROWS)
    assert r["total"] == 4
    assert r["accuracy"] == 0.5
    assert abs(r["mean_confidence"] - 0.6125) < 1e-6


def test_low_confidence_rate():
    r = summarize(ROWS, low_conf=0.5)
    # 2 mẫu < 0.5
    assert r["low_confidence_rate"] == 0.5


def test_confident_wrong():
    r = summarize(ROWS)
    # 1 mẫu sai với confidence >= 0.8 (cat->dog @0.85)
    assert r["confident_wrong_rate"] == 0.25


def test_per_class_accuracy():
    r = summarize(ROWS)
    assert r["per_class"]["cat"]["accuracy"] == 0.5
    assert r["per_class"]["cat"]["total"] == 2


def test_per_mode():
    r = summarize(ROWS)
    assert "mouse" in r["per_mode"]
    assert "hand" in r["per_mode"]
