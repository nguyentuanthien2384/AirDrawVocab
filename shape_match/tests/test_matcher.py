"""
test_matcher.py — Kiểm thử bộ chấm điểm khớp hình mẫu.

Chạy:  python -m pytest shape_match/tests/ -q
hoặc:  python -m shape_match.tests.test_matcher   (chạy thẳng không cần pytest)
"""
from __future__ import annotations

import numpy as np

try:
    from shape_match import templates
    from shape_match.matcher import ShapeMatcher, MatchConfig
    from shape_match import geometry as G
    from shape_match.web_endpoint import strokes_to_points, score_strokes
except ImportError:  # chạy trực tiếp
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from shape_match import templates
    from shape_match.matcher import ShapeMatcher, MatchConfig
    from shape_match import geometry as G
    from shape_match.web_endpoint import strokes_to_points, score_strokes


rng = np.random.default_rng(123)
M = ShapeMatcher(MatchConfig())


def _trace(name, scale=0.7, off=(120, 40), jitter=0.0):
    t = templates.get(name)
    p = G.resample(t.points, 140, closed=t.closed)
    p = templates.fit_unit_to_box(p, 960, 540, 0.82) * scale + np.array(off, dtype=float)
    if jitter:
        p = p + rng.normal(0, jitter, p.shape)
    return p


# --------------------------------------------------------------------------- #
def test_perfect_traces_pass_high():
    for name in ["circle", "square", "star", "triangle", "heart", "wave"]:
        r = M.score(_trace(name), name)
        assert r.passed, f"{name} perfect nên ĐẠT, được {r.accuracy:.1f}"
        assert r.accuracy >= 95, f"{name} perfect nên >=95, được {r.accuracy:.1f}"


def test_position_and_scale_invariance():
    base = M.score(_trace("star", scale=0.7, off=(120, 40)), "star").accuracy
    moved = M.score(_trace("star", scale=0.45, off=(400, 200)), "star").accuracy
    assert abs(base - moved) < 6, f"phải bất biến vị trí/tỉ lệ: {base:.1f} vs {moved:.1f}"


def test_small_noise_still_passes():
    for name in ["circle", "square", "triangle", "heart"]:
        r = M.score(_trace(name, jitter=5.0), name)
        assert r.accuracy >= 80, f"{name} nhiễu nhỏ nên >=80, được {r.accuracy:.1f}"


def test_scribble_fails():
    for name in ["circle", "square", "star"]:
        base = _trace(name)
        c = base.mean(axis=0)
        scribble = c + rng.normal(0, 90, base.shape)
        r = M.score(scribble, name)
        assert not r.passed and r.accuracy < 30, f"{name} nguệch ngoạc phải trượt, {r.accuracy:.1f}"


def test_partial_trace_fails_on_coverage():
    base = _trace("circle")
    partial = base[: int(len(base) * 0.5)]
    r = M.score(partial, "circle")
    assert not r.passed, "vẽ một nửa không được ĐẠT"
    assert r.coverage < 0.6, f"coverage phải thấp, được {r.coverage:.2f}"


def test_wrong_shape_fails():
    # tròn vs vuông và tam giác vs sao phải KHÔNG đạt
    assert not M.score(_trace("circle"), "square").passed
    assert not M.score(_trace("square"), "circle").passed
    assert not M.score(_trace("triangle"), "star").passed
    assert not M.score(_trace("line"), "wave").passed


def test_too_few_points():
    r = M.score(np.array([[10, 10], [12, 12], [14, 14]], dtype=float), "circle")
    assert not r.passed and r.accuracy == 0.0


def test_web_stroke_adapter():
    base = _trace("triangle")
    strokes = [[{"x": float(x), "y": float(y), "t": i} for i, (x, y) in enumerate(base)]]
    pts = strokes_to_points(strokes)
    assert pts.shape == (len(base), 2)
    out = score_strokes(strokes, "triangle")
    assert out["passed"] and out["accuracy"] >= 95
    assert out["target"] == "triangle" and "label_vi" in out


def test_geometry_resample_uniform():
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    r = G.resample(circle, 200, closed=True)
    seg = np.hypot(*np.diff(r, axis=0).T)
    assert seg.std() / seg.mean() < 0.02, "resample phải cách đều"


def test_normalize_invariance():
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    base = np.stack([np.cos(t), np.sin(t)], axis=1)
    a = G.normalize(base * 3.0 + np.array([100, -50]))
    b = G.normalize(base)
    assert np.allclose(a, b, atol=1e-6), "normalize phải bất biến vị trí + tỉ lệ"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} test đạt.")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
