"""What #60's lever 1 turns, and what the fixture is invariant to when it is not turned.

Lever 2 -- a scored self-prediction each cell is paid for -- was built on this
branch and stripped from it. Its target was the value a cell realised if it held
the token and zero if it did not, which makes the score a function of the
allocation and therefore of the bid: the one thing a payment beside the auction
may not be. It is redesigned on dense counterfactual targets under #66, so what
is pinned here is the knob that survives -- how much of the token the cells own
-- and that leaving it alone leaves the recorded fixture exactly where it was.
"""

import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from measure_stakes_dial import FIXTURE, fixture_fingerprint  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    shuffled,
)

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
    wide = economy(cells=4)

    assert wide.config.num_experts == 4
    assert wide.mob.expert_wealth.numel() == 4
    for _ in range(3):
        wide.step()


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
    than two arms of one: at 2x the `value` arm spends most of its cell-steps on
    the ceiling while `shuffled` piles at the floor. Nothing in the run record
    said so, which is why the grid was read as a contrast for a day.
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
