"""
web_endpoint.py — Tích hợp bộ chấm điểm khớp-hình-mẫu vào web app (FastAPI) sẵn có.

Frontend của bạn đã thu nét dưới dạng:
    strokes = [ [ {"x":.., "y":.., "t":..}, ... ], ... ]   # canvas 960 x 540

File này cung cấp:
  - strokes_to_points(strokes): gộp về một polyline (N,2) để chấm điểm.
  - score_strokes(strokes, target): trả dict điểm khớp (dùng được không cần web).
  - router: APIRouter có sẵn POST /shape/score và GET /shape/templates.

Cách gắn vào backend/app.py (chỉ thêm 2 dòng):
    from shape_match.web_endpoint import router as shape_router
    app.include_router(shape_router)

Sau đó frontend gọi:
    POST /shape/score   body: {"target":"star", "strokes":[...], "canvas_w":960, "canvas_h":540}
    -> {"accuracy":.., "passed":.., "stars":.., "shape_score":.., "coverage":.., ...}
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

try:
    from . import templates
    from .matcher import ShapeMatcher, MatchConfig
except ImportError:  # pragma: no cover
    import templates  # type: ignore
    from matcher import ShapeMatcher, MatchConfig  # type: ignore


_matcher = ShapeMatcher(MatchConfig())


def strokes_to_points(strokes: Any) -> np.ndarray:
    """
    Gộp nhiều nét thành một polyline (N, 2) theo thứ tự thời gian.
    Với bài "tô theo hình mẫu" người dùng thường vẽ một nét liền; nếu nhiều nét thì
    nối lại theo đúng trình tự đã vẽ.
    """
    pts: List[List[float]] = []
    if not isinstance(strokes, list):
        return np.zeros((0, 2))
    for stroke in strokes:
        if not isinstance(stroke, list):
            continue
        for p in stroke:
            if isinstance(p, dict) and "x" in p and "y" in p:
                pts.append([float(p["x"]), float(p["y"])])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def score_strokes(strokes: Any, target: str, config: Optional[MatchConfig] = None) -> dict:
    """Chấm điểm trực tiếp từ định dạng strokes của frontend. Không phụ thuộc FastAPI."""
    pts = strokes_to_points(strokes)
    matcher = ShapeMatcher(config) if config else _matcher
    result = matcher.score(pts, target)
    out = result.to_dict()
    out["target"] = target
    tpl = templates.get(target)
    out["label_vi"] = tpl.label_vi
    out["closed"] = tpl.closed
    return out


# --------------------------------------------------------------------------- #
# Router FastAPI (tùy chọn — chỉ tạo nếu đã cài fastapi/pydantic)
# --------------------------------------------------------------------------- #
try:
    from fastapi import APIRouter
    from pydantic import BaseModel

    class ScoreRequest(BaseModel):
        target: str
        strokes: list
        canvas_w: int = 960
        canvas_h: int = 540
        pass_threshold: Optional[float] = None

    router = APIRouter(prefix="/shape", tags=["shape-trace"])

    @router.get("/templates")
    def list_templates():
        out = []
        for nm in templates.names():
            t = templates.get(nm)
            out.append({
                "name": nm,
                "label_vi": t.label_vi,
                "closed": t.closed,
                # outline chuẩn hóa [0,1] để frontend vẽ ghost outline khớp với matcher
                "points": t.points.round(4).tolist(),
            })
        return {"templates": out}

    @router.post("/score")
    def score(req: ScoreRequest):
        cfg = None
        if req.pass_threshold is not None:
            cfg = MatchConfig(pass_threshold=float(req.pass_threshold))
        return score_strokes(req.strokes, req.target, cfg)

except Exception:  # pragma: no cover
    router = None  # FastAPI chưa cài; vẫn dùng được score_strokes()
