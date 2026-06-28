"""
templates.py — Thư viện hình mẫu (target shapes) để người dùng vẽ theo.

Mỗi hình là một đường gấp khúc trong hệ tọa độ chuẩn hóa [0, 1] x [0, 1]
(0,0 = góc trên-trái, giống ảnh/canvas). Có cờ `closed` cho biết hình kín hay hở.

Lý do dùng [0,1]: tách rời khỏi kích thước màn hình. Khi cần vẽ lên canvas WxH thì
gọi `template_to_pixels(name, w, h)`. Bộ matcher chuẩn hóa lại nên scale không ảnh hưởng.

Bạn có thể thêm hình mới bằng `register(name, points, closed)` hoặc nạp từ outline
[0,1] có sẵn trong frontend (drawGuide) — định dạng hoàn toàn tương thích.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Template:
    name: str
    points: np.ndarray  # (N, 2) trong [0, 1]
    closed: bool
    label_vi: str = ""

    def pixels(self, w: int, h: int, margin: float = 0.82) -> np.ndarray:
        """
        Ánh xạ tọa độ [0,1] -> pixel theo TỈ LỆ ĐỒNG NHẤT (isotropic), đặt vào một
        ô VUÔNG căn giữa khung WxH. Quan trọng: vì dùng cạnh vuông side=min(w,h),
        hình tròn vẫn là hình tròn (không bị méo thành elip trên màn 16:9). App vẽ
        outline mẫu và bộ matcher PHẢI dùng cùng phép ánh xạ này để khớp nhau.
        """
        return fit_unit_to_box(self.points, w, h, margin)


_REGISTRY: Dict[str, Template] = {}


def fit_unit_to_box(points: np.ndarray, w: int, h: int, margin: float = 0.82) -> np.ndarray:
    """
    Ánh xạ điểm [0,1]x[0,1] vào ô vuông căn giữa khung WxH theo tỉ lệ đồng nhất.
    Dùng chung cho cả việc VẼ outline mẫu lẫn tạo template pixel cho matcher.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    side = min(w, h) * float(margin)
    ox = (w - side) / 2.0
    oy = (h - side) / 2.0
    return pts * side + np.array([ox, oy])


def register(name: str, points, closed: bool, label_vi: str = "") -> Template:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    tpl = Template(name=name, points=pts, closed=closed, label_vi=label_vi)
    _REGISTRY[name] = tpl
    return tpl


def get(name: str) -> Template:
    if name not in _REGISTRY:
        raise KeyError(f"Không có hình mẫu '{name}'. Có sẵn: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> List[str]:
    return sorted(_REGISTRY)


def template_to_pixels(name: str, w: int, h: int) -> np.ndarray:
    return get(name).pixels(w, h)


# --------------------------------------------------------------------------- #
# Sinh các hình cơ bản (parametric) — sạch, mật độ điểm cao để resample tốt
# --------------------------------------------------------------------------- #
def _circle(cx=0.5, cy=0.5, r=0.32, n=240):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _polygon(sides, cx=0.5, cy=0.5, r=0.36, rot=-np.pi / 2, per_edge=40):
    verts = []
    for k in range(sides):
        a = rot + 2 * np.pi * k / sides
        verts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    verts.append(verts[0])
    pts = []
    for i in range(sides):
        a = np.array(verts[i])
        b = np.array(verts[i + 1])
        for f in np.linspace(0, 1, per_edge, endpoint=False):
            pts.append(a + f * (b - a))
    return np.asarray(pts)


def _star(points=5, cx=0.5, cy=0.5, r_out=0.38, r_in=0.16, rot=-np.pi / 2, per_edge=24):
    verts = []
    for k in range(points * 2):
        r = r_out if k % 2 == 0 else r_in
        a = rot + np.pi * k / points
        verts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    verts.append(verts[0])
    pts = []
    for i in range(len(verts) - 1):
        a = np.array(verts[i])
        b = np.array(verts[i + 1])
        for f in np.linspace(0, 1, per_edge, endpoint=False):
            pts.append(a + f * (b - a))
    return np.asarray(pts)


def _heart(cx=0.5, cy=0.52, scale=0.030, n=260):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([cx + scale * x, cy - scale * y], axis=1)
    return pts


def _triangle():
    return _polygon(3, r=0.40, per_edge=60)


def _zigzag(cx0=0.2, cy=0.5, w=0.6, amp=0.18, teeth=4, per=40):
    pts = []
    xs = np.linspace(cx0, cx0 + w, teeth + 1)
    for i in range(teeth):
        a = np.array([xs[i], cy + (amp if i % 2 == 0 else -amp)])
        b = np.array([xs[i + 1], cy + (-amp if i % 2 == 0 else amp)])
        for f in np.linspace(0, 1, per, endpoint=False):
            pts.append(a + f * (b - a))
    return np.asarray(pts)


def _wave(cx0=0.15, cy=0.5, w=0.7, amp=0.16, cycles=2.0, n=200):
    x = np.linspace(cx0, cx0 + w, n)
    y = cy + amp * np.sin(np.linspace(0, cycles * 2 * np.pi, n))
    return np.stack([x, y], axis=1)


def _line():
    return np.stack([np.linspace(0.18, 0.82, 80), np.full(80, 0.5)], axis=1)


def _diamond():
    return _polygon(4, r=0.40, rot=-np.pi / 2, per_edge=60)


# --------------------------------------------------------------------------- #
# Đăng ký mặc định
# --------------------------------------------------------------------------- #
def _bootstrap():
    register("circle", _circle(), closed=True, label_vi="Hình tròn")
    register("square", _polygon(4, rot=np.pi / 4, r=0.42, per_edge=60), closed=True, label_vi="Hình vuông")
    register("triangle", _triangle(), closed=True, label_vi="Tam giác")
    register("diamond", _diamond(), closed=True, label_vi="Hình thoi")
    register("pentagon", _polygon(5, r=0.38, per_edge=48), closed=True, label_vi="Ngũ giác")
    register("hexagon", _polygon(6, r=0.38, per_edge=40), closed=True, label_vi="Lục giác")
    register("star", _star(), closed=True, label_vi="Ngôi sao")
    register("heart", _heart(), closed=True, label_vi="Trái tim")
    register("line", _line(), closed=False, label_vi="Đường thẳng")
    register("wave", _wave(), closed=False, label_vi="Đường sóng")
    register("zigzag", _zigzag(), closed=False, label_vi="Đường zigzag")


_bootstrap()


if __name__ == "__main__":
    for nm in names():
        t = get(nm)
        print(f"{nm:10s} closed={t.closed} points={len(t.points):4d}  ({t.label_vi})")
