"""Minimal metrics layer with prometheus-client fallback."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except Exception:
    Counter = Histogram = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


class InMemoryMetrics:
    def __init__(self):
        self.counters: Dict[str, float] = {}
        self.latency: Dict[str, list[float]] = {}

    def inc(self, name: str, amount: float = 1.0):
        self.counters[name] = self.counters.get(name, 0.0) + amount

    def observe(self, name: str, value: float):
        self.latency.setdefault(name, []).append(float(value))
        self.latency[name] = self.latency[name][-500:]

    def render(self) -> bytes:
        lines = []
        for k, v in sorted(self.counters.items()):
            lines.append(f"airdraw_{k}_total {v}")
        for k, vals in sorted(self.latency.items()):
            if vals:
                lines.append(f"airdraw_{k}_count {len(vals)}")
                lines.append(f"airdraw_{k}_avg {sum(vals) / len(vals)}")
        return ("\n".join(lines) + "\n").encode("utf-8")


if Counter and Histogram:
    REQUESTS = Counter("airdraw_requests_total", "AirDrawVocab requests", ["endpoint", "status"])
    PREDICT_LATENCY = Histogram("airdraw_predict_latency_seconds", "Prediction latency", ["branch"])
else:
    REQUESTS = None
    PREDICT_LATENCY = None

MEMORY_METRICS = InMemoryMetrics()


def inc_request(endpoint: str, status: str = "ok"):
    if REQUESTS is not None:
        REQUESTS.labels(endpoint=endpoint, status=status).inc()
    else:
        MEMORY_METRICS.inc(f"requests_{endpoint}_{status}".replace("/", "_"))


@contextmanager
def observe_latency(branch: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if PREDICT_LATENCY is not None:
            PREDICT_LATENCY.labels(branch=branch).observe(elapsed)
        else:
            MEMORY_METRICS.observe(f"latency_{branch}", elapsed)


def render_metrics() -> tuple[bytes, str]:
    if Counter and Histogram:
        return generate_latest(), CONTENT_TYPE_LATEST
    return MEMORY_METRICS.render(), CONTENT_TYPE_LATEST
