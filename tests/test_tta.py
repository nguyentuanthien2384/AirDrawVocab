import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="cần TensorFlow")

from airdraw_models import _shift_batch, _tta_offsets  # noqa: E402


def test_offsets_start_with_zero():
    offs = _tta_offsets(6, max_shift=2)
    assert offs[0] == (0, 0)
    assert len(offs) == 6


def test_offsets_deterministic():
    assert _tta_offsets(6, 2) == _tta_offsets(6, 2)


def test_offsets_within_bounds():
    for dx, dy in _tta_offsets(9, max_shift=2):
        assert abs(dx) <= 2 and abs(dy) <= 2


def test_shift_zero_identity():
    x = np.random.rand(2, 8, 8, 1).astype("float32")
    assert np.array_equal(_shift_batch(x, 0, 0), x)


def test_shift_preserves_shape():
    x = np.random.rand(2, 8, 8, 1).astype("float32")
    out = _shift_batch(x, 1, -1)
    assert out.shape == x.shape
