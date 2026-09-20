"""What a contrast costs before it is worth GPU-hours (#56).

The numbers pinned here are the ones ``~/tame-runs/39-stakes-dial/body/POSTHOC.txt``
records, recomputed from the per-seed shifts that run wrote. They are inline
rather than read off disk: a test that skips when a run directory is missing is a
test that passed on nobody's machine, and these three values a seed are the whole
input to the calculation this issue exists to make routine.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from power import (  # noqa: E402
    Budget,
    _look_statistics,
    batch_looks,
    format_ceiling,
    load_shift,
    normal_approximation,
    null_calibration,
    paired_effect,
    paired_t_power,
    pairs_for_power,
    sequential_boundary,
    sequential_plan,
)

# #39's body sweep at 79c1b17: the total-variation allocation shift between the
# two dose levels, per seed, per arm (PRIMARY.txt).
BODY_SHIFTS = {
    "value": {"0": 0.1837005615234375, "1": 0.1094512939453125, "2": 0.1573638916015625},
    "decoupled": {"0": 0.251220703125, "1": 0.12640380859375, "2": 0.1607666015625},
    "shuffled": {"0": 0.136016845703125, "1": 0.108856201171875, "2": 0.1878509521484375},
}


def test_the_paired_t_reads_thirteen_seeds_where_the_normal_approximation_reads_ten():
    """#39's primary, and the reason section 8 names the exact calculation.

    The approximation drops the degrees of freedom, which is a fifth of the
    sweep at the count where the sweep is decided.
    """
    assert pairs_for_power(0.867) == 13
    assert paired_t_power(0.867, 13) == pytest.approx(0.818, abs=0.001)
    assert paired_t_power(0.867, 12) == pytest.approx(0.780, abs=0.001)
    assert normal_approximation(0.867) == pytest.approx(10.4, abs=0.05)


def test_the_self_reference_contrast_is_not_powerable_on_this_substrate():
    """dz 0.151 is 348 paired seeds, which at #39's cost is a month of the one GPU."""
    assert pairs_for_power(0.1507) == 348
    budget = Budget(348, runs_per_seed=2, arms=3, hours_per_run=8.2 / 24, ceiling=15.0)
    assert budget.runs_per_arm == 696
    assert budget.hours == pytest.approx(713, abs=1)
    assert not budget.fits


def test_the_recorded_shifts_reproduce_their_effect_sizes_and_the_pairing_that_bought_them():
    primary = paired_effect("value - decoupled", BODY_SHIFTS["value"], BODY_SHIFTS["decoupled"])
    self_reference = paired_effect(
        "value - shuffled", BODY_SHIFTS["value"], BODY_SHIFTS["shuffled"]
    )

    assert primary.dz == pytest.approx(-0.867, abs=0.001)
    assert self_reference.dz == pytest.approx(0.151, abs=0.001)
    # Sharma 2025's r/(1-rho): three paired seeds carried the precision of 34
    # unpaired ones, which is how dz reached 0.87 on an effect of 0.03 TV units.
    assert primary.rho == pytest.approx(0.913, abs=0.001)
    assert len(primary.deltas) / (1 - primary.rho) == pytest.approx(34.5, abs=0.1)


def test_a_readout_no_seed_count_reaches_is_capped_rather_than_searched_for_ever():
    assert pairs_for_power(0.001, maximum=200) == 200
    assert pairs_for_power(0.0, maximum=200) == 200


def test_an_effect_size_needs_two_shared_seeds():
    with pytest.raises(ValueError, match="at least two shared seeds"):
        paired_effect("one seed", {"0": 0.1}, {"0": 0.2, "9": 0.3})


def test_a_readout_that_is_not_the_allocation_shift_still_reads(tmp_path):
    """#57 will write per-seed values under another readout; the helper takes both shapes."""
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text('{"contrast": {"0": 1.0, "1": 2.0}, "floor": null}')
    bare = tmp_path / "bare.json"
    bare.write_text('{"0": 1.0, "1": 2.0}')

    assert load_shift(wrapped) == load_shift(bare) == {"0": 1.0, "1": 2.0}


def test_over_the_ceiling_the_plan_is_refused_and_the_design_goes_back_to_the_fixture():
    over = Budget(13, runs_per_seed=2, arms=3, hours_per_run=8.2 / 24, ceiling=15.0)
    assert not over.fits
    assert over.seeds_affordable == 7

    refusal = format_ceiling(over, over_ceiling=None)
    assert "REFUSED" in refusal
    assert "back to the fixture" in refusal
    assert "never a budget request" in refusal

    planned = format_ceiling(over, over_ceiling="a new section 7 row, dated")
    assert "REFUSED" not in planned
    assert "a new section 7 row, dated" in planned


def test_a_design_inside_the_ceiling_is_not_told_to_spend_up_to_it():
    inside = Budget(5, runs_per_seed=2, arms=2, hours_per_run=0.2, ceiling=15.0)

    assert inside.fits
    assert "dead band" in format_ceiling(inside, over_ceiling=None)


def test_the_batch_schedule_ends_on_the_preregistered_maximum():
    assert batch_looks(9, 3) == [3, 6, 9]
    # A maximum that is not a multiple of the batch still ends there: the last
    # batch is short, and the plan may not look past what it declared.
    assert batch_looks(7, 3) == [3, 6, 7]
    assert batch_looks(1, 3) == []


def test_the_sequential_boundary_holds_the_family_wise_error_over_every_look():
    looks = [3, 6, 9]
    boundary = sequential_boundary(looks, alpha=0.05, draws=8000, seed=0)

    # Checked against sequences the boundary was not fitted to.
    sample = np.random.default_rng(99).standard_normal((8000, max(looks)))
    crossed = (np.abs(_look_statistics(sample, looks)) > boundary).any(axis=1)
    assert crossed.mean() == pytest.approx(0.05, abs=0.01)

    # The price of looking three times: a single final test would read 2.31 here.
    assert boundary > 2.31


def test_the_plan_says_when_the_ceiling_and_not_the_batch_size_is_what_binds():
    plan = sequential_plan(0.867, batch_looks(7, 3), alpha=0.05, draws=8000, seed=0)

    assert plan.power < 0.5
    assert plan.expected_seeds < 7
    assert plan.stop_by == sorted(plan.stop_by)


def test_the_range_at_three_pairs_calls_a_difference_a_quarter_of_the_time_under_the_null():
    """Colas et al. 2018's false positive, and at n = 3 it is arithmetic.

    Below ``MIN_PAIRS_FOR_COVERAGE`` the percentile interval is the sample range,
    so it excludes zero exactly when every delta shares a sign: 2 x 0.5^3 = 0.25
    under any symmetric null, whatever the readout underneath. #39's primary was
    read at three seeds.
    """
    generator = np.random.default_rng(5)
    one_arm = list(generator.normal(0.15, 0.03, size=12))

    calibration = null_calibration(one_arm, pairs=3, splits=400, alpha=0.05, seed=0, resamples=500)

    assert calibration["bootstrap_false_positive_rate"] == pytest.approx(0.25, abs=0.06)
    assert calibration["paired_t_false_positive_rate"] == pytest.approx(0.05, abs=0.04)


def test_a_null_calibration_needs_the_readings_it_splits():
    with pytest.raises(ValueError, match="needs 12 readings"):
        null_calibration([0.1, 0.2], pairs=6, splits=10, alpha=0.05, seed=0, resamples=100)


def test_the_power_of_a_pair_count_no_one_can_reach_is_not_a_nan():
    """scipy's non-central t returns NaN far from any solution; the walk must not stop there."""
    assert not math.isnan(paired_t_power(0.15, 60_000))
    assert paired_t_power(0.15, 60_000) == pytest.approx(1.0, abs=1e-6)
