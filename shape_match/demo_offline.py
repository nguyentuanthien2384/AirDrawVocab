"""
demo_offline.py — Kiểm chứng bộ chấm điểm KHÔNG cần camera.

Sinh các nét vẽ giả lập (hoàn hảo / có nhiễu / sai hình / nguệch ngoạc / vẽ thiếu)
rồi in điểm để xác nhận thuật toán phân biệt đúng các trường hợp.

Chạy: python -m shape_match.demo_offline   (từ thư mục cha của shape_match)
hoặc: python demo_offline.py               (nếu chạy trong thư mục shape_match)
"""
from __future__ import annotations

import numpy as np

try:
    from . import templates
    from .matcher import ShapeMatcher, MatchConfig
    from . import geometry as G
except ImportError:  # chạy trực tiếp trong thư mục
    import templates  # type: ignore
    from matcher import ShapeMatcher, MatchConfig  # type: ignore
    import geometry as G  # type: ignore


rng = np.random.default_rng(7)


def _to_pixels(norm_pts, w=960, h=540):
    return templates.fit_unit_to_box(norm_pts, w, h, margin=0.82)


def perfect_trace(name):
    t = templates.get(name)
    # đi đúng đường mẫu, ánh xạ isotropic, rồi DỊCH + THU NHỎ ĐỀU (scalar) để
    # kiểm tra tính bất biến vị trí + tỉ lệ của matcher.
    pts = G.resample(t.points, 140, closed=t.closed)
    pts = _to_pixels(pts) * 0.7 + np.array([120.0, 40.0])
    return pts


def noisy_trace(name, jitter=6.0):
    pts = perfect_trace(name)
    return pts + rng.normal(0, jitter, pts.shape)


def shaky_trace(name, jitter=16.0):
    pts = perfect_trace(name)
    return pts + rng.normal(0, jitter, pts.shape)


def partial_trace(name, frac=0.55):
    pts = perfect_trace(name)
    return pts[: int(len(pts) * frac)]


def scribble(name):
    pts = perfect_trace(name)
    n = len(pts)
    c = pts.mean(axis=0)
    return c + rng.normal(0, 90, (n, 2))


def main():
    matcher = ShapeMatcher(MatchConfig())
    header = f"{'case':22s} {'acc':>6s} {'pass':>5s} {'★':>2s} {'shape':>6s} {'cov':>5s} {'prec':>5s} {'corn':>5s} {'haus':>5s}"
    print(header)
    print("-" * len(header))

    def row(case, user, target):
        r = matcher.score(user, target)
        print(f"{case:22s} {r.accuracy:6.1f} {str(r.passed):>5s} {r.stars:>2d} "
              f"{r.shape_score:6.2f} {r.coverage:5.2f} {r.precision:5.2f} {r.corner_score:5.2f} {r.hausdorff:5.2f}")
        return r

    for shape in ["circle", "square", "star", "triangle", "heart", "wave"]:
        print(f"\n=== Mẫu: {shape} ===")
        row("perfect", perfect_trace(shape), shape)
        row("noisy(small)", noisy_trace(shape), shape)
        row("shaky(big)", shaky_trace(shape), shape)
        row("partial 55%", partial_trace(shape), shape)
        row("scribble", scribble(shape), shape)

    # Vẽ HÌNH SAI so với mục tiêu -> phải điểm thấp
    print("\n=== Vẽ sai hình (phải điểm thấp) ===")
    row("circle vs square", perfect_trace("circle"), "square")
    row("square vs circle", perfect_trace("square"), "circle")
    row("triangle vs star", perfect_trace("triangle"), "star")
    row("line vs wave", perfect_trace("line"), "wave")


if __name__ == "__main__":
    main()
