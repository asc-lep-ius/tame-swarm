"""#65's re-entry read: the arithmetic on hand-built ledgers, with no fixture in it."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from measure_ledger_stability import (  # noqa: E402
    HORIZON,
    CellReading,
    LedgerReading,
    inflow_to_cross_within,
    re_entry,
    steps_to_cross,
)
from synthetic_economy import BASE_CONFIG  # noqa: E402

from mob.ledger import LEDGER_DECAY, LOSS_REWARD_MULTIPLIER, PERSISTENCE_VALUE  # noqa: E402

DECAY = BASE_CONFIG.wealth_decay


def test_a_cell_past_the_threshold_needs_no_inflow_and_crosses_at_once():
    assert inflow_to_cross_within(20.0, 15.0, -0.3, DECAY, 0.0) == 0.0
    assert steps_to_cross(20.0, 15.0, 0.0, -0.3, DECAY, 0.0, 10) == 1


def test_the_map_with_no_inflow_never_climbs_and_a_large_inflow_climbs_in_one_step():
    assert steps_to_cross(15.0, 16.0, 0.0, 0.0, DECAY, 0.0, 1000) is None
    assert steps_to_cross(15.0, 16.0, 5.0, 0.0, DECAY, 0.0, 1000) == 1


def test_the_inflow_to_cross_is_the_least_that_crosses_inside_the_horizon():
    """Bisection lands on the boundary: just below it the map does not cross in time."""
    needed = inflow_to_cross_within(15.0, 18.0, -0.3, DECAY, 0.0, HORIZON)

    assert steps_to_cross(15.0, 18.0, needed, -0.3, DECAY, 0.0, HORIZON) is not None
    assert steps_to_cross(15.0, 18.0, 0.99 * needed, -0.3, DECAY, 0.0, HORIZON) is None
    # A winner's charge makes the same climb cost far more than a rebate does.
    assert inflow_to_cross_within(15.0, 18.0, 60.0, DECAY, 0.0, HORIZON) > 10 * needed


def _cell(share: float, reward: float, kappa: float, on_floor: bool) -> CellReading:
    rho = 1.0 - DECAY
    discriminant = reward**2 - 4.0 * rho * kappa
    root = (reward + discriminant**0.5) / (2.0 * rho) if discriminant >= 0 else math.nan
    ruin = (reward - discriminant**0.5) / (2.0 * rho) if discriminant >= 0 else math.nan
    return CellReading(
        competence=0.5,
        wealth=15.0 if on_floor else 750.0,
        share=share,
        reward=reward,
        price_coefficient=kappa,
        settles_at=root,
        ruined_below=ruin,
        from_flat_inflow=0.0,
        clamped=True,
        reconstruction_error=0.0,
        tail_at_ceiling=0.0 if on_floor else 1.0,
        tail_at_floor=1.0 if on_floor else 0.0,
    )


def test_the_read_sizes_the_gift_from_the_winners_charge_and_the_floors_inflow():
    """README #ledger-stability's shape: winners paying a charge whose ruin threshold sits above the floor.

    At kappa 80 the winners' lower root is 16.1, above the floor of 15, so a
    floor cell has a threshold to cross and the extra to cross it is not
    trivially zero; the floor cells carry a positive kappa so their own map
    does not cross on its inflow either.
    """
    cells = tuple(
        _cell(0.49, 5.0, 80.0, False) if index < 2 else _cell(0.002, 0.01, 0.2, True)
        for index in range(8)
    )
    reading = LedgerReading(
        mode=LEDGER_DECAY,
        coupling=PERSISTENCE_VALUE,
        seed=0,
        cells=cells,
        market_holders=2,
        least_share=0.002,
        wealth_vs_competence=0.8,
        tail_loss=0.01,
    )

    read = re_entry(reading)

    rho = 1.0 - DECAY
    assert read.winner_cells == (0, 1) and read.floor_cells == tuple(range(2, 8))
    assert read.threshold_inflow == pytest.approx(2.0 * math.sqrt(rho * 80.0))
    assert read.winner_ruin_threshold > BASE_CONFIG.min_wealth
    assert read.shortfall[2] == pytest.approx(read.threshold_inflow / 0.01)
    assert 90 < read.shortfall[2] < 110
    gift_share = BASE_CONFIG.exploration_rate / (BASE_CONFIG.num_experts - BASE_CONFIG.top_k)
    for cell in read.floor_cells:
        assert read.extra_inflow_to_cross[cell] > 0.0
        assert read.root_condition_extra[cell] == pytest.approx(read.threshold_inflow - 0.01)
        assert read.root_condition_per_slot[cell] == pytest.approx(
            read.root_condition_extra[cell] / gift_share
        )
        assert read.gift_per_explored_slot[cell] == pytest.approx(
            read.extra_inflow_to_cross[cell] / gift_share
        )
        # The worth to a deliberate loser is the whole lottery, rate x the per-slot amount.
        assert read.deviation_worth[cell] == pytest.approx(
            read.gift_per_explored_slot[cell] * BASE_CONFIG.exploration_rate
        )
        assert read.root_condition_worth[cell] == pytest.approx(
            read.root_condition_per_slot[cell] * BASE_CONFIG.exploration_rate
        )
        # And the config's units are credits over the exchange rate and the multiplier.
        assert read.root_condition_config_amount[cell] == pytest.approx(
            read.root_condition_per_slot[cell] / (BASE_CONFIG.reward_scale * LOSS_REWARD_MULTIPLIER)
        )
    assert 2.0 < read.root_condition_config_amount[2] < 3.5


def test_a_ledger_with_no_floor_cell_reads_as_nothing_to_carry_back():
    """A gift that lifts every cell off the floor is a reading, not an error."""
    cells = tuple(_cell(0.125, 1.0, 10.0, False) for _ in range(8))
    reading = LedgerReading(
        mode=LEDGER_DECAY,
        coupling=PERSISTENCE_VALUE,
        seed=0,
        cells=cells,
        market_holders=8,
        least_share=0.125,
        wealth_vs_competence=0.0,
        tail_loss=0.01,
    )

    read = re_entry(reading)

    assert read.floor_cells == () and read.floor_inflow == {}
    assert read.winner_cells == tuple(range(8))
    with pytest.raises(ValueError, match="winner"):
        re_entry(
            LedgerReading(
                mode=LEDGER_DECAY,
                coupling=PERSISTENCE_VALUE,
                seed=0,
                cells=tuple(_cell(0.001, 0.01, -0.3, True) for _ in range(8)),
                market_holders=0,
                least_share=0.001,
                wealth_vs_competence=0.0,
                tail_loss=0.01,
            )
        )
