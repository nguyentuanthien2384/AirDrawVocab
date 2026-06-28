"""
shape_match — Mô-đun "vẽ tay khớp hình mẫu" cho AirDrawVocab.

Cho phép: hiển thị một hình mẫu (target), người dùng vẽ theo bằng đầu ngón tay qua
camera, rồi chấm điểm độ khớp một cách nghiêm túc (bất biến vị trí/kích thước/tốc độ).

API nhanh:
    from shape_match import ShapeMatcher, MatchConfig, templates

    matcher = ShapeMatcher()
    result = matcher.score(user_points_xy, "star")
    print(result.accuracy, result.passed, result.stars)
"""
from .matcher import ShapeMatcher, MatchConfig, MatchResult
from . import templates
from . import geometry

__all__ = [
    "ShapeMatcher",
    "MatchConfig",
    "MatchResult",
    "templates",
    "geometry",
]
