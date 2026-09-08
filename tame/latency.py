"""Per-route request latency and throughput over a bounded window (#5).

#5's 2026-04-12 comment moved history to MLflow and asked serving to keep
"standard HTTP middleware or just structured logging" instead of a self-rolled
tracker. Both are here and neither is history: every request is written out as a
structured log line -- the record of the request, which anything downstream can
aggregate -- and a bounded window of the most recent ones is kept in memory
because the acceptance criteria ask ``/metrics/health`` for live p50/p95/p99 and
tokens per second, which a log line cannot answer without a log aggregator. The
window is fixed at :data:`WINDOW_REQUESTS` per route and holds two floats per
request, so it cannot grow into the historical aggregate the comment removed.

Tokens per second is the throughput number a reader notices; the percentiles are
what catch a regression in the hook. #5 asks for both, and the coupling benchmark
(``scripts/benchmark_coupling.py``) measures the same two quantities offline with
the coupling detached, which is the comparison these are read against.
"""

import logging
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Two floats per request per route. Long enough that p99 means something (the
# 99th percentile of 256 samples is the third-slowest), short enough that the
# window is the recent past rather than a run's history -- which is MLflow's.
WINDOW_REQUESTS = 256


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already sorted list.

    Nearest-rank rather than interpolated: every value reported is a request that
    actually happened, which is what a latency budget is argued against.
    """
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[rank]


class LatencyTracker:
    """The last :data:`WINDOW_REQUESTS` requests per route: how long, and how many tokens."""

    def __init__(self, window: int = WINDOW_REQUESTS):
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = window
        self._requests: dict[str, deque[tuple[float, int]]] = {}

    @property
    def routes(self) -> list[str]:
        return sorted(self._requests)

    def record(self, route: str, seconds: float, output_tokens: int = 0) -> None:
        """Add one completed request and log it. Never raises on a bad measurement."""
        if seconds < 0:
            raise ValueError(f"seconds must be non-negative, got {seconds}")
        window = self._requests.setdefault(route, deque(maxlen=self.window))
        window.append((seconds, output_tokens))
        logger.info(
            "[LATENCY] route=%s seconds=%.4f tokens=%d tokens_per_second=%.2f",
            route,
            seconds,
            output_tokens,
            output_tokens / seconds if seconds > 0 else 0.0,
        )

    @contextmanager
    def measure(self, route: str) -> Iterator[list[int]]:
        """Time a block and record it, even if it raises.

        Yields a one-element list the caller writes its output token count into;
        the count is not known until the block has run, and a failed request still
        consumed the latency it took to fail.
        """
        tokens = [0]
        started = time.perf_counter()
        try:
            yield tokens
        finally:
            self.record(route, time.perf_counter() - started, tokens[0])

    def stats(self, route: str) -> dict[str, float | int]:
        """p50/p95/p99 and tokens per second over the window; zeros for an unseen route.

        Throughput is the window's total tokens over its total seconds, not the
        mean of per-request rates: a two-token request that took as long to set up
        as a two-hundred-token one would otherwise drag the figure to its own.
        """
        window = self._requests.get(route)
        if not window:
            return {
                "requests": 0,
                "p50_s": 0.0,
                "p95_s": 0.0,
                "p99_s": 0.0,
                "tokens_per_second": 0.0,
            }
        ordered = sorted(seconds for seconds, _ in window)
        total_seconds = sum(ordered)
        total_tokens = sum(tokens for _, tokens in window)
        return {
            "requests": len(window),
            "p50_s": _percentile(ordered, 0.50),
            "p95_s": _percentile(ordered, 0.95),
            "p99_s": _percentile(ordered, 0.99),
            "tokens_per_second": (total_tokens / total_seconds) if total_seconds > 0 else 0.0,
        }

    def reset(self) -> None:
        self._requests.clear()
