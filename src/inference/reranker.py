"""Versioned lightweight confusion reranker.

This keeps geometric post-processing outside backend/app.py so it can be
evaluated and versioned separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RerankDecision:
    used: bool
    reason: str = ""
    original_label: str = ""
    final_label: str = ""
    target_shape_score: float = 0.0


def rerank_top5(top5: List[Dict], target: str, shape_scores: Dict[str, float] | None = None,
                min_shape: float = 0.58, max_override_conf: float = 0.72) -> tuple[List[Dict], RerankDecision]:
    """Promote target when shape evidence is strong and CNN is not too certain.

    This function is intentionally conservative. It should reduce obvious
    confusions such as book/door/pants but should not fight a confident model.
    """
    if not top5 or not target:
        return top5, RerankDecision(False)
    shape_scores = shape_scores or {}
    target_score = float(shape_scores.get(target, 0.0) or 0.0)
    original = str(top5[0].get("label", ""))
    original_conf = float(top5[0].get("confidence", 0.0) or 0.0)
    if original == target or target_score < min_shape or original_conf > max_override_conf:
        return top5, RerankDecision(False, original_label=original, final_label=original, target_shape_score=target_score)
    found = None
    for row in top5:
        if row.get("label") == target:
            found = row
            break
    pseudo = min(0.93, max(float((found or {}).get("confidence", 0.0) or 0.0), 0.34 + 0.55 * target_score))
    new_rows = [dict(row) for row in top5 if row.get("label") != target]
    new_rows.insert(0, {"label": target, "confidence": pseudo, "reranked": True})
    new_rows = sorted(new_rows, key=lambda r: float(r.get("confidence", 0.0)), reverse=True)[:5]
    return new_rows, RerankDecision(True, f"shape_match_{target}_{round(target_score * 100)}pct", original, new_rows[0]["label"], target_score)
