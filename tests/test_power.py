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
    BATCH_SEEDS,
    Budget,
    _look_statistics,
    _plan_block,
    batch_looks,
    build_parser,
    format_ceiling,
    format_plan,
    format_requirement,
    load_shift,
    main,
    maximum_grid,
    normal_approximation,
    null_calibration,
    paired_effect,
    paired_t_power,
    pairs_for_power,
    plan_maximum,
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


def test_a_maximum_named_past_the_ceiling_is_refused_even_when_the_fixed_design_fits():
    """The refusal reads the plan's own cost, not the fixed design's.

    ``--max-seeds`` names a maximum the fixed-n requirement never implied, so a
    gate on the requirement alone printed a 40-look schedule rising to 246
    GPU-hours directly beneath a ceiling line reading "12.3 against 15 -- inside
    it". Found by a reviewer running it; the acceptance criterion is that the
    helper refuses to plan past the ceiling without a preregistration row.
    """
    args = build_parser().parse_args(["--dz", "1.5", "--plan", "--max-seeds", "120"])
    budget = Budget(6, args.runs_per_seed, args.arms, args.hours_per_run, args.ceiling)
    assert budget.fits  # the fixed design is inside the ceiling; the plan is not

    blocks, record = _plan_block(1.5, budget, args)

    assert record == {}
    assert "REFUSED" in "\n".join(blocks)
    assert "look" not in "\n".join(blocks)
    assert "246.0 GPU-h against 15" in "\n".join(blocks)

    named = build_parser().parse_args(
        ["--dz", "1.5", "--plan", "--max-seeds", "120", "--over-ceiling", "row 9, dated"]
    )
    allowed, plan_record = _plan_block(1.5, budget, named)
    assert plan_record["looks"][-1] == 120
    assert "row 9, dated" in "\n".join(allowed)


def test_the_plan_maximum_is_what_the_sequence_needs_and_not_what_a_fixed_design_needs():
    """A Pocock boundary is higher than a single final test, so the fixed n under-powers.

    At dz 1.5 the fixed requirement is 6 paired seeds for 0.833, and a sequence
    that stops at 6 reaches 0.49. Defaulting the maximum to the fixed n is what
    made ``--plan`` report an under-powered design at every dz, and the message
    beneath it then named the readout as the lever when the maximum was the one
    still free to move.
    """
    fixed = pairs_for_power(1.5)
    assert fixed == 6
    assert sequential_plan(1.5, batch_looks(fixed, 3), 0.05, 8000, 0).power < 0.6

    chosen = plan_maximum(1.5, batch=3, alpha=0.05, draws=8000, seed=0, power=0.80, cap=48)

    assert chosen > fixed
    assert sequential_plan(1.5, batch_looks(chosen, 3), 0.05, 8000, 0).power >= 0.80
    # The smallest such maximum, not merely one that works: a batch below it
    # does not reach the target.
    assert sequential_plan(1.5, batch_looks(chosen - 3, 3), 0.05, 8000, 0).power < 0.80


def test_an_under_powered_plan_names_the_lever_that_is_still_free():
    short = Budget(6, runs_per_seed=2, arms=3, hours_per_run=8.2 / 24, ceiling=100.0)
    at_the_wall = Budget(6, runs_per_seed=2, arms=3, hours_per_run=8.2 / 24, ceiling=15.0)
    plan = sequential_plan(0.867, batch_looks(6, 3), alpha=0.05, draws=8000, seed=0)
    assert plan.power < 0.80

    room = format_plan(plan, short, batch=3, alpha=0.05, power=0.80)
    assert "Raise the maximum first" in room
    assert f"affords {short.seeds_affordable}" in room

    # And once the maximum is everything the ceiling affords, it is not: there
    # the substrate is the only thing left, which is section 8's rule 3.
    wall = format_plan(
        sequential_plan(0.867, batch_looks(7, 3), alpha=0.05, draws=8000, seed=0),
        at_the_wall,
        batch=3,
        alpha=0.05,
        power=0.80,
    )
    assert "Raise the maximum first" not in wall
    assert "already at or past everything the ceiling affords" in wall
    # The sentence names both numbers, because the maximum can sit *past* the
    # affordance under --over-ceiling and "already everything" was false there.
    assert "and this plan stops at 7" in wall


def test_a_discrete_readout_with_no_spread_is_a_call_and_not_a_skipped_split():
    """A zero-spread split with a nonzero mean is an infinite t, so it is a rejection.

    Dropping it from the numerator while keeping it in the denominator is what
    understates the rate this function exists to measure -- on exactly the
    readouts that produce ties, which is every count-valued one.
    """
    tied = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    calibration = null_calibration(tied, pairs=3, splits=200, alpha=0.05, seed=0, resamples=200)

    assert calibration["paired_t_false_positive_rate"] > 0.0


def test_the_pair_ceiling_the_message_quotes_is_the_one_the_search_used():
    budget = Budget(500, runs_per_seed=2, arms=3, hours_per_run=8.2 / 24, ceiling=15.0)

    assert "under 500 reaches this power" in format_requirement(
        0.01, budget, power=0.80, alpha=0.05, maximum=500
    )


def test_a_batch_below_three_is_refused_by_the_operators_decision_and_the_degrees_of_freedom():
    """#56 fixed the batch at three, and below it the first look has no spread worth a t."""
    for batch in (-3, 0, 1, 2):
        with pytest.raises(ValueError, match="no spread worth a t"):
            plan_maximum(1.0, batch, alpha=0.05, draws=2000, seed=0, power=0.80, cap=30)

    assert plan_maximum(1.5, BATCH_SEEDS, 0.05, 8000, 0, 0.80, 48) > 0


def test_the_maximum_is_the_smallest_by_construction_and_not_by_a_curve_assumption():
    """A binary search needed the sequence's power to rise with the maximum. It does not.

    The statistic is estimated by simulation, so where the power gradient is
    shallow the Monte Carlo noise wins: at 2000 draws the binary search returned
    48 and 51 where the smallest is 42, which at #39's body knobs is 12 to 18
    GPU-hours of preregistered maximum bought for nothing. Both cases are pinned
    here against brute force, at the batch the guard above allows.
    """

    def brute_force(dz, draws, seed, power, cap):
        # `maximum_grid` rather than a second description of it: the two agreed
        # only while `cap` was a multiple of the batch, and an oracle that walks
        # different points fails for reasons unrelated to what it guards.
        grid = maximum_grid(BATCH_SEEDS, cap)
        reached = (
            count
            for count in grid
            if sequential_plan(dz, batch_looks(count, BATCH_SEEDS), 0.05, draws, seed).power
            >= power
        )
        return next(reached, grid[-1])

    for seed in (1, 2):
        found = plan_maximum(0.6, BATCH_SEEDS, 0.05, 2000, seed, 0.50, 60)
        assert found == brute_force(0.6, 2000, seed, 0.50, 60) == 42

    # A cap that is not a multiple of the batch: the grid ends on it, and the
    # oracle and the code have to agree there too. They did not while the oracle
    # described the grid instead of sharing it.
    assert maximum_grid(BATCH_SEEDS, 47)[-1] == 47
    assert plan_maximum(0.6, BATCH_SEEDS, 0.05, 2000, 1, 0.50, 47) == brute_force(
        0.6, 2000, 1, 0.50, 47
    )


def test_a_target_the_cap_misses_returns_the_cap_and_not_a_smaller_point_that_reaches():
    """The probe is a decision about what may be preregistered, not a shortcut.

    Where the curve is not monotone there are smaller maxima that reach while
    the cap does not, and they are Monte Carlo noise: at 2000 draws, dz 0.3 and
    seed 5, a maximum of 6 reads 0.080 against the cap's 0.075. Preregistering
    that 6 would be preregistering the noise, so the answer is that the design
    does not reach. This is the one branch that is a choice rather than a
    search, and it is pinned here so it stays one.
    """
    reaching = sequential_plan(0.3, batch_looks(6, BATCH_SEEDS), 0.05, 2000, 5).power
    at_the_cap = sequential_plan(0.3, batch_looks(30, BATCH_SEEDS), 0.05, 2000, 5).power
    assert reaching > 0.0775 > at_the_cap

    assert plan_maximum(0.3, BATCH_SEEDS, 0.05, 2000, 5, 0.0775, 30) == 30

    # The same rule at a cap that is not a multiple of the batch, where it
    # decides between two different answers on the same design: 44 misses 0.50
    # so 44 is returned, and 47 reaches it so the scan runs and finds 42.
    assert plan_maximum(0.6, BATCH_SEEDS, 0.05, 2000, 1, 0.50, 44) == 44
    assert plan_maximum(0.6, BATCH_SEEDS, 0.05, 2000, 1, 0.50, 47) == 42

    # And a target nothing reaches at all still costs one probe and returns the
    # cap, rather than walking every point to say the same thing.
    assert plan_maximum(0.05, BATCH_SEEDS, 0.05, 2000, 0, 0.99, 30) == 30


def test_a_nonsense_batch_is_a_usage_error_and_not_a_sequence_wearing_the_label(
    monkeypatch, capsys
):
    """``--batch -3`` printed "batches of -3 paired seeds ... over 1 looks".

    At a boundary of 2.443 against a single final test's 2.447: a fixed-n test
    wearing the label of a family-wise-controlled sequence. argparse takes any
    integer, so the refusal is ``main``'s, and it has to name the flag -- the
    degenerate cases used to surface as `range() arg 3 must not be zero` or as
    nothing at all.
    """
    monkeypatch.setattr(sys, "argv", ["power.py", "--dz", "1.5", "--plan", "--batch", "-3"])

    with pytest.raises(SystemExit):
        main()

    assert "at least 3 paired seeds" in capsys.readouterr().err


def test_a_maximum_too_small_to_look_at_says_so_rather_than_printing_nothing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["power.py", "--dz", "1.5", "--plan", "--max-seeds", "1"])

    main()

    assert "a maximum of 1 cannot be looked at" in capsys.readouterr().out


def test_a_design_already_over_the_ceiling_does_not_pay_for_a_search_it_cannot_use():
    args = build_parser().parse_args(["--dz", "0.867", "--plan"])
    over = Budget(13, args.runs_per_seed, args.arms, args.hours_per_run, args.ceiling)
    assert not over.fits

    blocks, record = _plan_block(0.867, over, args)

    assert (blocks, record) == ([], {})
