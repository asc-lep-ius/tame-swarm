"""What #60's lever 1 turns, and what the fixture is invariant to when it is not turned.

Lever 2 -- a scored self-prediction each cell is paid for -- was built on this
branch and stripped from it. Its target was the value a cell realised if it held
the token and zero if it did not, which makes the score a function of the
allocation and therefore of the bid: the one thing a payment beside the auction
may not be. It is redesigned on dense counterfactual targets under #66, so what
is pinned here is the knob that survives -- how much of the token the cells own
-- and that leaving it alone leaves the recorded fixture exactly where it was.
"""

import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import measure_self_model  # noqa: E402
from measure_ledger_stability import (  # noqa: E402
    ClampedLedgerError,
    Derivation,
    DerivationPass,
)
from measure_self_model import (  # noqa: E402
    CLAMPED_COLUMN,
    CORRELATION_COLUMN,
    DERIVED,
    HAND_SET,
    READINGS_COLUMN,
    RECORDED_RATE,
    grid_row,
    pooled,
    rates_for,
    run_arm,
)
from measure_stakes_dial import FIXTURE, RATIOS, TAIL, fixture_fingerprint  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    shuffled,
)

from mob import PERSISTENCE_VALUE  # noqa: E402
from parity import ArmFingerprint, ParityError, assert_parity  # noqa: E402

from .arm_fingerprints import BASE


def economy(**overrides) -> DifferentiatedEconomy:
    cells = overrides.pop("cells", len(DEFAULT_COMPETENCE))
    scale = overrides.pop("contribution_scale", 1.0)
    competence = DEFAULT_COMPETENCE if cells == len(DEFAULT_COMPETENCE) else DEFAULT_COMPETENCE[::2]
    config = replace(BASE_CONFIG, num_experts=cells, **overrides)
    built = DifferentiatedEconomy(
        shuffled(competence, 0), seed=0, config=config, contribution_scale=scale
    )
    built.add_goal_field(0, 0.5, 0.25)
    built.add_goal_field(1, 0.5, 0.25)
    return built


def test_the_recorded_economy_is_bitwise_what_it_was_at_scale_one():
    """A knob whose default is the fixture itself, rather than a fixture beside it.

    The planted corrections are scaled where they are built, so an off-by-a-float
    default would move every recorded number on the differentiated fixture at
    once and nothing downstream would notice. Read as wealth, which is where a
    change of that kind lands.
    """

    def run(**overrides) -> torch.Tensor:
        built = economy(**overrides)
        for _ in range(10):
            built.step()
        return built.mob.expert_wealth.clone()

    assert torch.equal(run(), run(contribution_scale=1.0))
    assert not torch.equal(run(), run(contribution_scale=2.0))


def test_more_cells_than_slots_is_a_knob_and_the_recorded_count_is_the_default():
    assert BASE_CONFIG.num_experts == 8
    narrow = economy(cells=4)

    assert narrow.config.num_experts == 4
    assert narrow.mob.expert_wealth.numel() == 4
    for _ in range(3):
        narrow.step()


def test_the_contribution_scale_is_what_the_cells_own_of_the_output():
    """Lever 1's second ratio: the same allocation, a larger share of the token."""
    plain = economy()
    louder = economy(contribution_scale=4.0)

    assert torch.allclose(louder.type_corrections, 4.0 * plain.type_corrections, atol=1e-6)
    # And it reaches the loss, which is what makes it a ratio the organism can
    # notice rather than a number in a config.
    assert plain.step().loss != louder.step().loss


def test_parity_refuses_two_arms_that_own_different_amounts_of_the_token():
    """The confound that withdrew the grid, made unrepeatable.

    The scale multiplies realised value, reward and price and leaves the wealth
    band where it is, so two arms at different scales are two economies rather
    than two arms of one: as the scale rises the `value` arm climbs to the
    ceiling while `shuffled` piles at the floor. The occupancy table under README
    `#self-model` carries the numbers; nothing in the run record carried them,
    which is why the grid was read as a contrast for a day.
    """
    with pytest.raises(ParityError, match="contribution_scale"):
        assert_parity(
            [
                replace(BASE, persistence_coupling="value"),
                replace(BASE, persistence_coupling="shuffled", contribution_scale=2.0),
            ]
        )


def test_a_fingerprint_recorded_before_the_scale_existed_reads_as_the_recorded_fixture():
    """Every run before #60 owned exactly one unit of the token, so a legacy row is 1.0."""
    recorded = {key: value for key, value in asdict(BASE).items() if key != "contribution_scale"}

    assert ArmFingerprint(**recorded) == BASE
    assert BASE.contribution_scale == 1.0


def test_the_fixture_fingerprint_carries_the_grid_cell_it_actually_ran_at():
    """It carried the recorded fixture's cell count whatever the grid asked for.

    A run at four cells and twice the correction fingerprinted as eight cells at
    one, so two rows of #60's grid compared clean against each other and against
    every run recorded before the grid existed.
    """
    run = fixture_fingerprint(
        FIXTURE, seed=0, arm="value", doses=(0.25,), steps=10, cells=16, contribution_scale=2.0
    )

    assert run.num_experts == 16
    assert run.contribution_scale == 2.0
    with pytest.raises(ParityError, match="contribution_scale"):
        assert_parity(
            [
                run,
                fixture_fingerprint(
                    FIXTURE, seed=0, arm="shuffled", doses=(0.25,), steps=10, cells=16
                ),
            ]
        )


def test_the_occupancy_rate_is_over_the_window_actually_read():
    """The denominator was the tail length, not the steps the run had to give.

    A smoke run shorter than the tail counted its clamps over `cells x TAIL`
    cell-steps it never saw, so every occupancy came out scaled down by
    `steps / TAIL` — 3 of 400 where it was 3 of 160. The direction is the one
    that hides a clamp rather than inventing one, which is why it survived a
    reading of the printed table.
    """
    short = run_arm(PERSISTENCE_VALUE, RATIOS[0], seed=0, cells=4, scale=1.0, steps=7)
    full = run_arm(PERSISTENCE_VALUE, RATIOS[0], seed=0, cells=4, scale=1.0, steps=TAIL + 5)

    assert short["guardrail/cell_steps"] == 4 * 7
    assert full["guardrail/cell_steps"] == 4 * TAIL


def test_one_clamped_seed_does_not_swallow_the_seeds_that_had_a_correlation():
    """`fmean` propagates a single NaN over the whole pool.

    At 2x one seed in six can have every ledger on a bound while the other five
    are perfectly readable, and the pooled column printed `nan` for all six — so
    the guardrail went blind in exactly the regime #66, #67 and #69 are specified
    in. NaN survives only where nothing was readable.
    """

    def reading(correlation: float) -> dict[str, float]:
        return {
            "guardrail/ceiling_occupancy": 0.5,
            "guardrail/floor_occupancy": 0.25,
            "guardrail/interior_occupancy": 0.25,
            "guardrail/cell_steps": 800.0,
            CORRELATION_COLUMN: correlation,
        }

    mixed = pooled([reading(0.6), reading(float("nan")), reading(0.2)])

    assert mixed[CORRELATION_COLUMN] == pytest.approx(0.4)
    assert mixed[CLAMPED_COLUMN] == 1
    assert mixed[READINGS_COLUMN] == 3
    # And the occupancies still pool over every reading, clamped or not.
    assert mixed["guardrail/ceiling_occupancy"] == pytest.approx(0.5)

    blind = pooled([reading(float("nan")), reading(float("nan"))])

    assert math.isnan(blind[CORRELATION_COLUMN])
    assert blind[CLAMPED_COLUMN] == 2


# --- #73: the grid runs at the derived rate, with the hand-set pairing beside it ----------


def _stub_derivation(rate: float) -> Derivation:
    """A derivation record with the arithmetic already done, for the wiring tests."""
    final = DerivationPass(
        rate=rate,
        fixed_points={0: (1500.0,)},
        per_cell={0: (rate,)},
        derived={0: rate},
        cells_on_ceiling={0: 2},
        ceiling_occupancy={0: 0.25},
        floor_occupancy={0: 0.7},
        wealth_vs_competence={0: 0.8},
        saturated=False,
        next_rate=rate,
        next_rate_from="derived",
    )
    return Derivation(
        fixture="differentiated-fixture",
        contribution_scale=2.0,
        cells=8,
        seeds=(0,),
        steps=10,
        tail=5,
        recorded_rate=RECORDED_RATE,
        reference={0: (1500.0,)},
        reference_ceiling_occupancy={0: 0.25},
        reference_floor_occupancy={0: 0.7},
        passes=(final,),
        derived=rate,
        seed_spread=0.0,
        code_sha=None,
        code_dirty=None,
    )


def test_a_scaled_cell_runs_at_its_derived_rate_with_the_hand_set_pairing_beside_it(monkeypatch):
    """The withdrawn grid is reproduced as the pairing, never as the reading."""
    monkeypatch.setattr(
        measure_self_model, "derive_reward_scale", lambda *args, **kwargs: _stub_derivation(0.5)
    )

    rates, record = rates_for(8, 2.0, (0,), 10)

    assert list(rates) == [DERIVED, HAND_SET]
    assert rates[DERIVED] == (0.5, True)
    assert rates[HAND_SET] == (RECORDED_RATE, False)
    assert record["derived"]["derived"] == 0.5 and record["refused"] is None


def test_the_recorded_scale_has_one_rate_and_derives_nothing(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("1x must not run a derivation")

    monkeypatch.setattr(measure_self_model, "derive_reward_scale", never)

    rates, record = rates_for(8, 1.0, (0,), 10)

    assert rates == {HAND_SET: (RECORDED_RATE, False)}
    assert record == {"derived": None, "refused": None}


def test_a_refused_derivation_leaves_the_pairing_and_the_refusal(monkeypatch):
    """A cell read at the clamp is #60's grid; the record says so instead of running it as new."""

    def refuse(*args, **kwargs):
        raise ClampedLedgerError("every pass saturated")

    monkeypatch.setattr(measure_self_model, "derive_reward_scale", refuse)

    rates, record = rates_for(16, 4.0, (0,), 10)

    assert rates == {HAND_SET: (RECORDED_RATE, False)}
    assert record["derived"] is None
    assert "saturated" in record["refused"]


def test_a_grid_row_carries_the_rate_it_ran_at_in_every_fingerprint():
    row = grid_row(4, 2.0, 0.5, True, (0,), 7)

    assert row["reward_scale"] == 0.5 and row["reward_scale_derived"] is True
    for arm in ("value", "shuffled"):
        fingerprint = ArmFingerprint(**row["fingerprints"][arm]["0"])
        assert fingerprint.reward_scale == 0.5
        assert fingerprint.reward_scale_derived is True
        assert fingerprint.contribution_scale == 2.0
    with pytest.raises(ParityError, match="reward_scale"):
        assert_parity(
            [
                ArmFingerprint(**row["fingerprints"]["value"]["0"]),
                ArmFingerprint(
                    **grid_row(4, 2.0, 2.0, False, (0,), 7)["fingerprints"]["shuffled"]["0"]
                ),
            ]
        )
