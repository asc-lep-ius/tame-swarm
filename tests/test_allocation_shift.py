"""The allocation shift is one number per paired run, read against the re-running floor (#28).

Since #57 the shift has four readouts and the recorded one is the default. Two
of the tests below are pins rather than checks: the default readout reproduces
#39's body shifts bit for bit from the win shares that run recorded, because a
readout that changes forward only (preregistration section 8, rule 2) must be
able to show that it did not change backward.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from allocation_shift import (  # noqa: E402
    DEFAULT_READOUT,
    READING_PREFIX,
    READOUTS,
    STEP_POST_PREFIX,
    STEP_PRE_PREFIX,
    SWEPT_TYPE_PREFIX,
    assert_identical_fingerprints,
    bootstrap_mean,
    excess_over_floor,
    format_report,
    paired_shifts,
    per_seed_readings,
    total_variation,
)

# #39's body sweep at 79c1b17: the recorded end-of-training win shares of the
# `value` arm at the two dose levels, per seed, and the per-seed total-variation
# shift PRIMARY.txt reports between them.
BODY_VALUE_R1 = {
    "0": (0.6529541015625, 0.53436279296875, 0.390350341796875, 0.422332763671875),
    "1": (0.5139617919921875, 0.5837554931640625, 0.498199462890625, 0.404083251953125),
    "2": (0.664947509765625, 0.6218719482421875, 0.416412353515625, 0.2967681884765625),
}
BODY_VALUE_R4 = {
    "0": (0.7654876708984375, 0.60552978515625, 0.27838134765625, 0.3506011962890625),
    "1": (0.602325439453125, 0.6048431396484375, 0.428924560546875, 0.3639068603515625),
    "2": (0.5075836181640625, 0.7127227783203125, 0.459381103515625, 0.3203125),
}
BODY_SHIFT = {
    "0": 0.1837005615234375,
    "1": 0.1094512939453125,
    "2": 0.1573638916015625,
}


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
    """The refusal names the column family, since #57 there being four of them."""
    with pytest.raises(ValueError, match="reports no 'routing/win_share_e' column"):
        total_variation({"eval/loss": 1.0}, _run(0.5, 0.5))
    with pytest.raises(ValueError, match=f"reports no '{SWEPT_TYPE_PREFIX}' column"):
        READOUTS["token-conditioned"].reading(_run(0.5, 0.5), _run(0.5, 0.5))


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
    with pytest.raises(ValueError, match="shares no seed"):
        assert_identical_fingerprints(same, {"fingerprints": {"7": prints["0"]}})
    extra = {"fingerprints": {**prints, "0": {**prints["0"], "extra": 1}}}
    with pytest.raises(ValueError, match=r"differs on \['extra'\]"):
        assert_identical_fingerprints(same, extra)


def test_the_default_readout_reproduces_the_recorded_body_shift_bit_for_bit():
    """#39's body numbers, from the win shares that run recorded (`PRIMARY.txt`)."""
    group_r1 = {"per_seed": {seed: _run(*shares) for seed, shares in BODY_VALUE_R1.items()}}
    group_r4 = {"per_seed": {seed: _run(*shares) for seed, shares in BODY_VALUE_R4.items()}}

    assert paired_shifts(group_r1, group_r4) == BODY_SHIFT
    assert paired_shifts(group_r1, group_r4, DEFAULT_READOUT) == BODY_SHIFT


def test_every_readout_reads_zero_between_a_run_and_itself():
    """The planted null: two groups at parity, whatever the readout underneath."""
    run = {
        **_run(0.6, 0.4),
        **{f"{SWEPT_TYPE_PREFIX}{i}": share for i, share in enumerate((0.7, 0.3))},
        **{f"{READING_PREFIX}350_e{i}": share for i, share in enumerate((0.5, 0.5))},
        **{f"{READING_PREFIX}400_e{i}": share for i, share in enumerate((0.6, 0.4))},
    }

    for readout in READOUTS.values():
        if not readout.within_run:
            assert readout.reading(run, run) == 0.0


def test_the_token_conditioned_readout_reads_the_swept_type_and_not_the_aggregate():
    """Candidate 2's whole claim: the aggregate averages the signature away."""
    columns = {
        "a": {"routing/win_share_e0": 0.5, "routing/win_share_e1": 0.5},
        "b": {"routing/win_share_e0": 0.5, "routing/win_share_e1": 0.5},
    }
    columns["a"].update({f"{SWEPT_TYPE_PREFIX}0": 0.9, f"{SWEPT_TYPE_PREFIX}1": 0.1})
    columns["b"].update({f"{SWEPT_TYPE_PREFIX}0": 0.3, f"{SWEPT_TYPE_PREFIX}1": 0.7})

    assert READOUTS["total-variation"].reading(columns["a"], columns["b"]) == 0.0
    assert READOUTS["token-conditioned"].reading(columns["a"], columns["b"]) == pytest.approx(0.6)


def test_the_integrated_readout_averages_over_the_readings_the_runs_share():
    a = {
        f"{READING_PREFIX}350_e0": 0.5,
        f"{READING_PREFIX}350_e1": 0.5,
        f"{READING_PREFIX}400_e0": 0.5,
        f"{READING_PREFIX}400_e1": 0.5,
    }
    b = {
        f"{READING_PREFIX}350_e0": 0.9,
        f"{READING_PREFIX}350_e1": 0.1,
        f"{READING_PREFIX}400_e0": 0.7,
        f"{READING_PREFIX}400_e1": 0.3,
    }

    assert READOUTS["integrated"].reading(a, b) == pytest.approx((0.4 + 0.2) / 2)
    with pytest.raises(ValueError, match="share no logged reading"):
        READOUTS["integrated"].reading(a, {"routing/win_share_e0": 1.0})


def test_a_within_run_readout_is_read_inside_the_run_and_says_so_when_it_is_not():
    """The two functions refuse each other's readout rather than return a wrong number."""
    stepped = {
        **{f"{STEP_PRE_PREFIX}{i}": share for i, share in enumerate((0.5, 0.5))},
        **{f"{STEP_POST_PREFIX}{i}": share for i, share in enumerate((0.8, 0.2))},
    }
    group = {"per_seed": {"0": stepped}}

    assert per_seed_readings(group, READOUTS["setpoint-step"]) == {"0": pytest.approx(0.3)}
    with pytest.raises(ValueError, match="read inside one run"):
        paired_shifts(group, group, READOUTS["setpoint-step"])
    with pytest.raises(ValueError, match="distance between two groups"):
        per_seed_readings(group, DEFAULT_READOUT)


def test_the_report_names_the_readout_when_it_is_not_the_recorded_one():
    shifts = {"0": 0.1, "1": 0.2, "2": 0.3}

    recorded = format_report(shifts, None, 200, 0)
    other = format_report(shifts, None, 200, 0, READOUTS["token-conditioned"])

    assert "allocation shift (TV)" in recorded
    assert "readout:" not in recorded
    assert "token-conditioned" in other
