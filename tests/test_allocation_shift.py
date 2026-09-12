"""The allocation shift is one number per paired run, read against the re-running floor (#28)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from allocation_shift import (  # noqa: E402
    assert_identical_fingerprints,
    bootstrap_mean,
    excess_over_floor,
    format_report,
    paired_shifts,
    total_variation,
)


def _run(*shares: float, loss: float = 2.6) -> dict[str, float]:
    return {"eval/loss": loss, **{f"routing/win_share_e{i}": s for i, s in enumerate(shares)}}


def test_total_variation_is_half_the_l1_distance_over_the_shared_experts():
    a = _run(0.6, 0.5, 0.5, 0.4)
    b = _run(0.7, 0.4, 0.5, 0.4)

    assert total_variation(a, b) == pytest.approx(0.1)
    assert total_variation(a, a) == 0.0
    with pytest.raises(ValueError, match="different experts"):
        total_variation(a, _run(0.6, 0.5))


def test_a_run_without_win_shares_cannot_be_compared():
    with pytest.raises(ValueError, match="no win-share column"):
        total_variation({"eval/loss": 1.0}, _run(0.5, 0.5))


def test_shifts_are_paired_by_seed_and_the_excess_reads_against_the_floor():
    group_a = {"per_seed": {"0": _run(0.6, 0.4), "1": _run(0.5, 0.5), "9": _run(0.1, 0.9)}}
    group_b = {"per_seed": {"0": _run(0.8, 0.2), "1": _run(0.5, 0.5)}}
    floor_b = {"per_seed": {"0": _run(0.65, 0.35), "1": _run(0.55, 0.45)}}

    contrast = paired_shifts(group_a, group_b)
    floor = paired_shifts(group_a, floor_b)

    assert contrast == {"0": pytest.approx(0.2), "1": 0.0}
    assert floor == {"0": pytest.approx(0.05), "1": pytest.approx(0.05)}
    assert excess_over_floor(contrast, floor) == {
        "0": pytest.approx(0.15),
        "1": pytest.approx(-0.05),
    }
    with pytest.raises(ValueError, match="share no seed"):
        paired_shifts(group_a, {"per_seed": {"7": _run(0.5, 0.5)}})


def test_the_bootstrap_interval_brackets_the_mean_and_is_seeded():
    values = [0.1, 0.2, 0.3]

    mean, low, high = bootstrap_mean(values, resamples=2000, seed=1)

    assert mean == pytest.approx(0.2)
    assert 0.1 <= low <= mean <= high <= 0.3
    assert bootstrap_mean(values, resamples=2000, seed=1) == (mean, low, high)
    assert bootstrap_mean([0.4], resamples=10) == (0.4, 0.4, 0.4)


def test_at_three_pairs_the_interval_is_the_range_and_is_labelled_so():
    """A resample repeats one value with probability 1/27 > 2.5%: the percentiles are extremes."""
    contrast = {"0": -0.013, "1": 0.077, "2": 0.037}

    mean, low, high = bootstrap_mean(list(contrast.values()), resamples=5000, seed=0)
    report = format_report(contrast, None, resamples=5000, seed=0)

    assert (low, high) == (-0.013, 0.077)
    assert mean == pytest.approx(0.0337, abs=1e-4)
    assert "resampled-mean range" in report and "95%" not in report.splitlines()[0]
    assert "no 95% coverage" in report


def test_a_floor_pair_must_be_a_replication():
    prints = {"0": {"seed": 0, "router": "mob"}, "1": {"seed": 1, "router": "mob"}}
    same = {"fingerprints": prints}
    other = {"fingerprints": {**prints, "1": {"seed": 1, "router": "softmax"}}}

    assert_identical_fingerprints(same, same)
    with pytest.raises(ValueError, match="not a replication: seed 1 differs on \\['router'\\]"):
        assert_identical_fingerprints(same, other)
    with pytest.raises(ValueError, match="needs the arm fingerprints"):
        assert_identical_fingerprints({}, same)
