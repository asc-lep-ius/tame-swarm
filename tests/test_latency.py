"""The per-route latency window (#5).

The window is what makes p50/p95/p99 answerable without a log aggregator; the log
line is the record. Both are asserted, including the case the percentiles exist
for: a request that failed still cost what it took to fail.
"""

import logging

import pytest

from latency import WINDOW_REQUESTS, LatencyTracker


def test_an_unseen_route_reports_zeros_rather_than_raising():
    """A ``/metrics/health`` before the first generation must not 500."""
    stats = LatencyTracker().stats("/generate")
    assert stats == {
        "requests": 0,
        "p50_s": 0.0,
        "p95_s": 0.0,
        "p99_s": 0.0,
        "tokens_per_second": 0.0,
    }


def test_percentiles_are_of_requests_that_happened():
    """Nearest-rank: every value reported is a real request, which is what a budget is argued
    against."""
    tracker = LatencyTracker()
    for seconds in range(1, 101):
        tracker.record("/generate", float(seconds))

    stats = tracker.stats("/generate")
    assert stats["requests"] == 100
    assert stats["p50_s"] == 50.0
    assert stats["p95_s"] == 95.0
    assert stats["p99_s"] == 99.0


def test_the_p99_of_a_full_window_is_the_third_slowest_request():
    """The claim the window size is chosen for: at 256 samples p99 is a real request.

    Nearest-rank, `ceil(p*n) - 1`, so the p99 of 256 is index 253 -- three requests
    were slower. An interpolated percentile would report a duration nothing took.
    """
    tracker = LatencyTracker()
    for seconds in range(WINDOW_REQUESTS):
        tracker.record("/generate", float(seconds))

    slowest = sorted(range(WINDOW_REQUESTS), reverse=True)[:3]
    assert tracker.stats("/generate")["p99_s"] == float(slowest[-1])


def test_throughput_is_the_windows_tokens_over_its_seconds():
    """Not the mean of per-request rates: a two-token request would otherwise drag it to its
    own."""
    tracker = LatencyTracker()
    tracker.record("/generate", 1.0, output_tokens=2)
    tracker.record("/generate", 9.0, output_tokens=198)

    assert tracker.stats("/generate")["tokens_per_second"] == 20.0
    # The mean of the per-request rates would be (2 + 22) / 2 = 12.
    assert tracker.stats("/generate")["tokens_per_second"] != 12.0


def test_the_window_is_bounded_and_holds_the_most_recent_requests():
    tracker = LatencyTracker()
    for seconds in range(WINDOW_REQUESTS + 50):
        tracker.record("/generate", float(seconds))

    stats = tracker.stats("/generate")
    assert stats["requests"] == WINDOW_REQUESTS
    assert stats["p50_s"] > 50, "the earliest requests have left the window"


def test_routes_are_kept_apart():
    tracker = LatencyTracker()
    tracker.record("/generate", 2.0, output_tokens=10)
    tracker.record("/generate/stream", 1.0, output_tokens=100)

    assert tracker.routes == ["/generate", "/generate/stream"]
    assert tracker.stats("/generate")["tokens_per_second"] == 5.0
    assert tracker.stats("/generate/stream")["tokens_per_second"] == 100.0


def test_a_failed_request_still_costs_what_it_took_to_fail():
    """A p99 that silently drops the slow failures is the wrong number to hold a budget to."""
    tracker = LatencyTracker()
    with pytest.raises(RuntimeError), tracker.measure("/generate") as tokens:
        tokens[0] = 7
        raise RuntimeError("generation failed")

    stats = tracker.stats("/generate")
    assert stats["requests"] == 1
    assert stats["p50_s"] >= 0.0


def test_measure_records_the_token_count_the_block_wrote():
    tracker = LatencyTracker()
    with tracker.measure("/generate") as tokens:
        tokens[0] = 128

    assert tracker.stats("/generate")["tokens_per_second"] > 0


def test_every_request_is_logged_as_well_as_windowed(caplog):
    """The window is the live read; the log line is the record anything else aggregates."""
    tracker = LatencyTracker()
    with caplog.at_level(logging.INFO, logger="latency"):
        tracker.record("/generate", 2.0, output_tokens=64)

    assert "[LATENCY] route=/generate seconds=2.0000 tokens=64" in caplog.text
    assert "tokens_per_second=32.00" in caplog.text


def test_a_negative_measurement_is_rejected_rather_than_windowed():
    with pytest.raises(ValueError, match="non-negative"):
        LatencyTracker().record("/generate", -1.0)
