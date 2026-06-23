import numpy as np

from stroke_features import (
    strokes_to_sequence, strokes_to_batch, count_active_points,
    MAX_LEN, NUM_FEATURES,
)

SAMPLE = [
    [{"x": 10, "y": 20, "t": 0}, {"x": 30, "y": 40, "t": 1}, {"x": 50, "y": 10, "t": 2}],
    [{"x": 100, "y": 100, "t": 3}, {"x": 120, "y": 140, "t": 4}],
]


def test_sequence_shape():
    seq = strokes_to_sequence(SAMPLE)
    assert seq.shape == (MAX_LEN, NUM_FEATURES)
    assert seq.dtype == np.float32


def test_batch_shape():
    b = strokes_to_batch(SAMPLE)
    assert b.shape == (1, MAX_LEN, NUM_FEATURES)


def test_empty_input():
    seq = strokes_to_sequence("[]")
    assert seq.shape == (MAX_LEN, NUM_FEATURES)
    assert float(seq.sum()) == 0.0


def test_deterministic():
    a = strokes_to_sequence(SAMPLE)
    b = strokes_to_sequence(SAMPLE)
    assert np.array_equal(a, b)


def test_count_active_points():
    seq = strokes_to_sequence(SAMPLE)
    # 5 điểm có chuyển động
    assert count_active_points(seq) == 5


def test_pen_up_flag_present():
    seq = strokes_to_sequence(SAMPLE)
    # cột pen_up (index 7) phải có ít nhất 1 điểm = 1.0 (cuối nét)
    assert seq[:, 7].max() == 1.0


def test_normalized_range():
    seq = strokes_to_sequence(SAMPLE)
    x, y = seq[:, 0], seq[:, 1]
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert y.min() >= 0.0 and y.max() <= 1.0
