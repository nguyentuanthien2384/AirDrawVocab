"""
geometry.py — Các hàm hình học lõi cho việc so khớp nét vẽ với hình mẫu.

Tất cả thuật toán ở đây đều thuần NumPy/SciPy, không phụ thuộc camera, nên có thể
test offline và tái sử dụng cho cả app desktop lẫn web (backend FastAPI).

Pipeline chuẩn (giống các nghiên cứu shape-tracing):
    arc-length resample  ->  normalize (bất biến vị trí + tỉ lệ)  ->  align (DTW / cyclic shift)

Quy ước: "polyline" là mảng numpy shape (N, 2), mỗi hàng là một điểm (x, y).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # SciPy có sẵn trong project (requirements), nhưng vẫn fallback nếu thiếu.
    from scipy.spatial.distance import cdist, directed_hausdorff
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Tiện ích cơ bản
# --------------------------------------------------------------------------- #
def as_polyline(points) -> np.ndarray:
    """Ép đầu vào về mảng (N, 2) float32. Chấp nhận list[(x,y)], list[[x,y]], ndarray."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Polyline phải có shape (N, 2), nhận được {arr.shape}")
    return arr


def path_length(poly: np.ndarray, closed: bool = False) -> float:
    """Tổng chiều dài đường gấp khúc."""
    if len(poly) < 2:
        return 0.0
    seg = np.diff(poly, axis=0)
    total = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    if closed:
        d = poly[0] - poly[-1]
        total += float(np.hypot(d[0], d[1]))
    return total


def dedupe(poly: np.ndarray, min_dist: float = 1e-6) -> np.ndarray:
    """Bỏ các điểm trùng/sát nhau để resample ổn định."""
    if len(poly) <= 1:
        return poly
    keep = [0]
    for i in range(1, len(poly)):
        if np.hypot(*(poly[i] - poly[keep[-1]])) > min_dist:
            keep.append(i)
    out = poly[keep]
    return out if len(out) >= 2 else poly


# --------------------------------------------------------------------------- #
# Resample theo chiều dài cung (loại bỏ phụ thuộc tốc độ vẽ)
# --------------------------------------------------------------------------- #
def resample(poly: np.ndarray, n: int = 128, closed: bool = False) -> np.ndarray:
    """
    Lấy lại mẫu đường thành n điểm CÁCH ĐỀU NHAU theo chiều dài cung.

    Đây là bước then chốt: dù người dùng vẽ nhanh/chậm, nhiều/ít điểm, sau bước này
    hai đường luôn có cùng số điểm phân bố đều -> so sánh điểm-với-điểm mới công bằng.
    """
    poly = dedupe(as_polyline(poly))
    if len(poly) < 2:
        return np.repeat(poly if len(poly) else np.zeros((1, 2)), n, axis=0)[:n]

    pts = np.vstack([poly, poly[0]]) if closed else poly
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < _EPS:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg_len) - 1 and cum[j + 1] < t:
            j += 1
        seg_start, seg_end = cum[j], cum[j + 1]
        frac = 0.0 if seg_end - seg_start < _EPS else (t - seg_start) / (seg_end - seg_start)
        out[i] = pts[j] + frac * (pts[j + 1] - pts[j])
    return out


# --------------------------------------------------------------------------- #
# Chuẩn hóa: bất biến vị trí + tỉ lệ (tùy chọn bất biến xoay)
# --------------------------------------------------------------------------- #
def normalize(poly: np.ndarray, rotate_invariant: bool = False) -> np.ndarray:
    """
    Dời trọng tâm về gốc và chia cho bán kính RMS -> bất biến vị trí & kích thước.
    Nếu rotate_invariant=True thì xoay theo trục chính (PCA) để bất biến cả góc xoay.
    """
    poly = as_polyline(poly).astype(np.float64)
    centroid = poly.mean(axis=0)
    centered = poly - centroid

    if rotate_invariant and len(centered) >= 2:
        # Xoay sao cho trục có phương sai lớn nhất nằm ngang (ổn định hướng).
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        angle = -np.arctan2(axis[1], axis[0])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        centered = centered @ rot.T

    rms = np.sqrt(np.mean(np.sum(centered ** 2, axis=1))) + _EPS
    return centered / rms


# --------------------------------------------------------------------------- #
# Khoảng cách giữa hai tập điểm
# --------------------------------------------------------------------------- #
def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return cdist(a, b)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.2) -> float:
    """
    Dynamic Time Warping giữa hai chuỗi điểm đã resample/normalize.

    DTW cho phép "co giãn" dọc đường nên chịu được việc người dùng đi nhanh ở đoạn
    này, chậm ở đoạn kia. `band` là dải Sakoe-Chiba (0..1) để tránh warp quá đà và
    tăng tốc; 0.2 nghĩa là chỉ cho lệch tối đa 20% chiều dài chuỗi.
    """
    a = as_polyline(a)
    b = as_polyline(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = _pairwise(a, b)
    w = max(int(band * max(n, m)), abs(n - m)) + 1

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - w)
        j1 = min(m, i + w)
        for j in range(j0, j1 + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    # Chuẩn hóa theo độ dài đường warp (xấp xỉ n+m) để không thiên vị chuỗi dài.
    return float(acc[n, m] / (n + m))


def symmetric_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Hausdorff hai chiều: độ lệch tệ nhất giữa hai đường (đơn vị: không gian chuẩn hóa)."""
    a = as_polyline(a)
    b = as_polyline(b)
    if _HAS_SCIPY:
        return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))
    d = _pairwise(a, b)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def coverage(template: np.ndarray, user: np.ndarray, tol: float) -> float:
    """Tỉ lệ điểm MẪU có ít nhất một điểm NGƯỜI DÙNG ở gần (<= tol). -> vẽ đủ hình chưa."""
    d = _pairwise(as_polyline(template), as_polyline(user))
    return float(np.mean(d.min(axis=1) <= tol))


def precision(template: np.ndarray, user: np.ndarray, tol: float) -> float:
    """Tỉ lệ điểm NGƯỜI DÙNG nằm gần MẪU (<= tol). -> có vẽ thừa/nguệch ngoạc không."""
    d = _pairwise(as_polyline(template), as_polyline(user))
    return float(np.mean(d.min(axis=0) <= tol))


# --------------------------------------------------------------------------- #
# Căn chỉnh điểm bắt đầu / chiều vẽ cho hình KÍN
# --------------------------------------------------------------------------- #
def best_cyclic_alignment(
    user: np.ndarray,
    template: np.ndarray,
    try_reverse: bool = True,
    step: int = 2,
) -> Tuple[np.ndarray, float]:
    """
    Với hình kín (vòng tròn, vuông, sao...), người dùng có thể bắt đầu ở bất kỳ đâu
    và vẽ theo chiều nào. Hàm thử mọi điểm bắt đầu (dịch vòng) + cả chiều đảo ngược,
    trả về phiên bản `user` đã căn chỉnh tốt nhất cùng khoảng cách DTW nhỏ nhất.

    `step` để duyệt thưa cho nhanh; cả hai chuỗi giả định đã cùng độ dài n.
    """
    user = as_polyline(user)
    template = as_polyline(template)
    n = len(user)
    candidates = [user]
    if try_reverse:
        candidates.append(user[::-1])

    best_seq, best_d = user, float("inf")
    for cand in candidates:
        for shift in range(0, n, max(1, step)):
            rolled = np.roll(cand, -shift, axis=0)
            d = dtw_distance(rolled, template)
            if d < best_d:
                best_d, best_seq = d, rolled
    return best_seq, best_d


# --------------------------------------------------------------------------- #
# Hàm tiếp tuyến (turning function) — phân biệt GÓC NHỌN với ĐƯỜNG CONG
# --------------------------------------------------------------------------- #
def _smooth_1d(x: np.ndarray, k: np.ndarray, mode: str) -> np.ndarray:
    pad = len(k) // 2
    if mode == "wrap":
        xp = np.concatenate([x[-pad:], x, x[:pad]])
    else:  # nearest
        xp = np.concatenate([np.repeat(x[:1], pad), x, np.repeat(x[-1:], pad)])
    return np.convolve(xp, k, mode="valid")[: len(x)]


def smooth_points(poly: np.ndarray, win: int, closed: bool) -> np.ndarray:
    """Làm trơn đường (trung bình trượt trên x, y) để khử rung tay trước khi đo góc."""
    poly = as_polyline(poly)
    if win < 2 or len(poly) < win:
        return poly
    k = np.ones(win) / win
    mode = "wrap" if closed else "nearest"
    return np.stack([_smooth_1d(poly[:, 0], k, mode), _smooth_1d(poly[:, 1], k, mode)], axis=1)


def tangent_function(poly: np.ndarray, closed: bool = False, smooth: int = 3) -> np.ndarray:
    """
    Hàm góc tiếp tuyến theo chiều dài cung (turning function kinh điển của Arkin).
    Là TÍCH PHÂN của góc quay nên ít nhạy rung tay nhưng vẫn phân biệt mạnh "đường
    dốc đều" (tròn) với "bậc thang" (vuông/tam giác). Đã unwrap & trừ trung bình ->
    bất biến với góc xoay tổng thể.
    """
    poly = as_polyline(poly)
    n = len(poly)
    if n < 3:
        return np.zeros(max(n, 1))
    if closed:
        seg = poly - np.roll(poly, 1, axis=0)
    else:
        seg = np.diff(poly, axis=0)
        seg = np.vstack([seg[:1], seg])
    ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    if smooth and smooth > 1:
        ang = _smooth_1d(ang, np.ones(smooth) / smooth, "wrap" if closed else "nearest")
    return ang - ang.mean()


def turning_similarity(
    a: np.ndarray,
    b: np.ndarray,
    closed: bool = False,
    sigma: float = 0.58,
    coarse: int = 44,
    smooth_pts: int = 9,
) -> float:
    """
    So khớp KIỂU đường (góc nhọn vs đường cong) giữa hai chuỗi ĐÃ resample đều và ĐÃ
    căn chỉnh điểm-với-điểm (cùng độ dài). Vì đã căn chỉnh nên so hàm tiếp tuyến ở chế
    độ "mở" là đủ — không cần wrap (tránh tạo đoạn nối giả ở chỗ cắt của hình kín).

    Chống nhiễu: làm trơn ĐIỂM -> lấy mẫu THƯA -> mới tính hàm tiếp tuyến, nên rung
    tay bị triệt tiêu còn các góc lớn (đặc trưng hình) được giữ lại.

    Trả 1.0 nếu cùng kiểu; giảm khi khác (tròn vs vuông, thiếu/thừa đỉnh nhọn...).
    """
    a = as_polyline(a)
    b = as_polyline(b)
    a = smooth_points(a, smooth_pts, closed=False)
    b = smooth_points(b, smooth_pts, closed=False)
    m = min(len(a), len(b))
    if m < 4:
        return 0.0
    idx = np.linspace(0, m - 1, min(coarse, m)).astype(int)
    fa = tangent_function(a[idx], closed=False, smooth=3)
    fb = tangent_function(b[idx], closed=False, smooth=3)
    diff = float(np.sqrt(np.mean((fa - fb) ** 2)))  # RMS chênh lệch (radian)
    return float(np.exp(-(diff / sigma) ** 2))
