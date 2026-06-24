"""Simple deterministic A/B routing for model experiments."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class ABConfig:
    enabled: bool = False
    control: str = "image"
    treatment: str = "image_candidate"
    treatment_pct: int = 10


def assign_bucket(user_key: str, cfg: ABConfig) -> str:
    if not cfg.enabled or cfg.treatment_pct <= 0:
        return cfg.control
    digest = hashlib.sha1(str(user_key).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return cfg.treatment if bucket < cfg.treatment_pct else cfg.control
