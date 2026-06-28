"""
matcher.py — Bộ chấm điểm độ khớp giữa nét vẽ của người dùng và hình mẫu.

Triết lý chấm điểm (nghiêm để không thể "vẽ đại cho qua"):

    accuracy = 100 * shape^Ws * coverage^Wc * precision^Wp

Vì dùng TÍCH (geometric), cả ba tiêu chí đều phải tốt thì điểm mới cao:
  - shape      : đường có đi đúng quỹ đạo mẫu không (DTW sau khi chuẩn hóa).
  - coverage   : đã vẽ ĐỦ hình mẫu chưa (có bỏ sót đoạn nào không).
  - precision  : có vẽ THỪA/nguệch ngoạc ra ngoài mẫu không.

Ngoài ra có "cổng cứng" Hausdorff: nếu lệch tệ nhất quá lớn -> tự động trượt,
tránh trường hợp điểm trung bình cao nhưng có một đoạn sai bét.

Tất cả tính trong không gian đã chuẩn hóa (bất biến vị trí + tỉ lệ), nên người
dùng đứng gần hay xa camera, vẽ to hay nhỏ đều được chấm công bằng.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from . import geometry as G
from .templates import Template, get as get_template


@dataclass
class MatchConfig:
    n_resample: int = 128          # số điểm sau khi resample mỗi đường
    rotate_invariant: bool = False # True nếu muốn bỏ qua góc xoay (vd nhận dạng tự do)
    tol: float = 0.18              # ngưỡng "gần mẫu" trong không gian chuẩn hóa
    sigma: float = 0.45            # độ rộng hàm điểm shape: nhỏ = chấm gắt hơn

    # Trọng số (mũ) cho tích điểm. Tổng không cần = 1.
    w_shape: float = 1.0
    w_coverage: float = 0.8
    w_precision: float = 0.7
    w_corner: float = 0.7          # phạt khi thiếu/thừa góc nhọn (tròn vs vuông...)
    sigma_turn: float = 0.58       # độ gắt của hồ sơ góc quay

    # Cổng cứng -> tự động fail nếu vi phạm
    max_hausdorff: float = 1.15
    min_points: int = 8

    # Ngưỡng đạt và mốc sao (đặt khá GẮT vì yêu cầu "phải khớp chuẩn"; chỉnh xuống
    # 70-75 nếu muốn dễ hơn cho trẻ em / người mới).
    pass_threshold: float = 80.0
    star2_threshold: float = 88.0
    star3_threshold: float = 94.0

    # Hình kín: thử mọi điểm bắt đầu + chiều vẽ
    cyclic_for_closed: bool = True
    cyclic_step: int = 3


@dataclass
class MatchResult:
    accuracy: float                # 0..100, điểm tổng để hiển thị
    passed: bool
    stars: int                     # 0..3
    shape_score: float             # 0..1
    coverage: float                # 0..1
    precision: float               # 0..1
    corner_score: float            # 0..1 (khớp góc nhọn/đường cong)
    hausdorff: float               # không gian chuẩn hóa
    dtw: float
    n_user_points: int
    message: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # làm tròn cho gọn khi trả JSON
        for k in ("accuracy", "shape_score", "coverage", "precision", "corner_score", "hausdorff", "dtw"):
            d[k] = round(float(d[k]), 4)
        return d


class ShapeMatcher:
    def __init__(self, config: Optional[MatchConfig] = None):
        self.cfg = config or MatchConfig()

    # ----------------------------------------------------------------- #
    def score(self, user_points, template) -> MatchResult:
        """
        user_points : đường người dùng vẽ (N, 2) — pixel hay chuẩn hóa đều được.
        template    : tên hình (str) HOẶC Template HOẶC mảng (M, 2). Nếu là mảng thì
                      mặc định coi là hình hở (closed=False).
        """
        cfg = self.cfg

        # --- chuẩn bị mẫu ---
        if isinstance(template, str):
            tpl = get_template(template)
            tpl_pts, closed = tpl.points, tpl.closed
        elif isinstance(template, Template):
            tpl_pts, closed = template.points, template.closed
        else:
            tpl_pts, closed = G.as_polyline(template), False

        user = G.as_polyline(user_points)
        n_user = len(user)
        if n_user < cfg.min_points:
            return MatchResult(
                accuracy=0.0, passed=False, stars=0, shape_score=0.0, coverage=0.0,
                precision=0.0, corner_score=0.0, hausdorff=float("inf"), dtw=float("inf"),
                n_user_points=n_user, message="Nét vẽ quá ngắn — hãy vẽ trọn hình mẫu.",
            )

        # --- resample đều theo chiều dài cung ---
        u = G.resample(user, cfg.n_resample, closed=False)
        t = G.resample(tpl_pts, cfg.n_resample, closed=closed)

        # --- chuẩn hóa (bất biến vị trí + tỉ lệ, tùy chọn xoay) ---
        un = G.normalize(u, rotate_invariant=cfg.rotate_invariant)
        tn = G.normalize(t, rotate_invariant=cfg.rotate_invariant)

        # --- căn chỉnh điểm bắt đầu / chiều cho hình kín ---
        if closed and cfg.cyclic_for_closed:
            un, dtw = G.best_cyclic_alignment(un, tn, try_reverse=True, step=cfg.cyclic_step)
        else:
            # hình hở: cho phép đảo chiều (vẽ từ trái hay phải) nhưng giữ điểm đầu/cuối
            d_fwd = G.dtw_distance(un, tn)
            d_rev = G.dtw_distance(un[::-1], tn)
            if d_rev < d_fwd:
                un, dtw = un[::-1], d_rev
            else:
                dtw = d_fwd

        # --- các thành phần điểm ---
        shape_score = float(np.exp(-(dtw / cfg.sigma) ** 2))
        cov = G.coverage(tn, un, cfg.tol)
        prec = G.precision(tn, un, cfg.tol)
        haus = G.symmetric_hausdorff(un, tn)
        corner = G.turning_similarity(un, tn, closed=closed, sigma=cfg.sigma_turn)

        # --- tổng hợp bằng tích có trọng số ---
        base = (
            (shape_score ** cfg.w_shape)
            * (cov ** cfg.w_coverage)
            * (prec ** cfg.w_precision)
            * (corner ** cfg.w_corner)
        )
        accuracy = 100.0 * float(base)

        # --- cổng cứng ---
        gated = accuracy
        hard_fail_msg = ""
        if haus > cfg.max_hausdorff:
            gated = min(gated, 45.0)
            hard_fail_msg = "Có đoạn lệch quá xa hình mẫu."
        accuracy = max(0.0, min(100.0, gated))

        passed = (
            accuracy >= cfg.pass_threshold
            and haus <= cfg.max_hausdorff
            and cov >= 0.6
            and prec >= 0.5
        )
        stars = 0
        if accuracy >= cfg.star3_threshold and passed:
            stars = 3
        elif accuracy >= cfg.star2_threshold and passed:
            stars = 2
        elif passed:
            stars = 1

        message = hard_fail_msg or self._feedback(accuracy, cov, prec, shape_score)

        return MatchResult(
            accuracy=accuracy, passed=passed, stars=stars,
            shape_score=shape_score, coverage=cov, precision=prec, corner_score=corner,
            hausdorff=haus, dtw=dtw, n_user_points=n_user, message=message,
            extra={"closed": bool(closed)},
        )

    # ----------------------------------------------------------------- #
    @staticmethod
    def _feedback(acc: float, cov: float, prec: float, shape: float) -> str:
        if acc >= 93:
            return "Tuyệt vời! Nét vẽ khớp gần như hoàn hảo."
        if acc >= 85:
            return "Rất tốt — chỉ lệch một chút."
        if acc >= 75:
            return "Đạt! Vẽ mượt hơn nữa sẽ điểm cao hơn."
        # gợi ý cụ thể theo điểm yếu nhất
        weakest = min(("cover", cov), ("prec", prec), ("shape", shape), key=lambda kv: kv[1])[0]
        if weakest == "cover":
            return "Chưa đạt: bạn còn bỏ sót một phần của hình — hãy vẽ trọn vẹn."
        if weakest == "prec":
            return "Chưa đạt: có nét thừa/lệch ra ngoài — bám sát đường mẫu hơn."
        return "Chưa đạt: quỹ đạo chưa giống mẫu — đi theo đúng hình dạng."
