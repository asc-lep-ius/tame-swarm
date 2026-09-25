"""#65's candidate 1, the settlement-sized gift: inert at zero, paid only on explored slots."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    SyntheticEconomy,
    shuffled,
)

from mob import MoBConfig  # noqa: E402
from mob.ledger import (  # noqa: E402
    LOSS_REWARD_MULTIPLIER,
    PERSISTENCE_DECOUPLED,
    RealisedValueReward,
    Settlement,
)
from parity import ParityError, assert_parity  # noqa: E402

from .arm_fingerprints import BASE  # noqa: E402

torch.set_num_threads(1)


def _trajectory(steps: int = 30, **overrides) -> tuple[torch.Tensor, list[torch.Tensor]]:
    economy = SyntheticEconomy(
        shuffled(DEFAULT_COMPETENCE, 0), seed=0, config=replace(BASE_CONFIG, **overrides)
    )
    routes = [economy.step().selected_experts.clone() for _ in range(steps)]
    return economy.mob.expert_wealth.clone(), routes


def test_the_shipped_economy_is_bitwise_the_one_at_a_gift_of_zero():
    recorded_wealth, recorded_routes = _trajectory()
    at_zero_wealth, at_zero_routes = _trajectory(re_entry_gift=0.0)
    gifted_wealth, _ = _trajectory(re_entry_gift=50.0)

    assert torch.equal(at_zero_wealth, recorded_wealth)
    assert all(torch.equal(a, b) for a, b in zip(at_zero_routes, recorded_routes, strict=True))
    assert not torch.equal(gifted_wealth, recorded_wealth)
    assert MoBConfig().re_entry_gift == 0.0


def _settlement(explored: torch.Tensor | None, valid: torch.Tensor | None = None) -> Settlement:
    selected = torch.tensor([[[0, 1], [2, 1], [3, 0], [1, 2]]])
    return Settlement(
        selected_experts=selected,
        routing_weights=torch.full((1, 4, 2), 0.5),
        confidences=torch.zeros(1, 4, 4),
        num_tokens=4,
        values=torch.zeros(1, 4, 2),
        valid_mask=torch.ones(1, 4, dtype=torch.bool) if valid is None else valid,
        explored=explored,
    )


def test_the_gift_is_paid_per_explored_slot_and_to_no_one_else():
    """Credited at 1/num_tokens a slot, through the exchange rate, keyed on the draw alone."""
    config = replace(BASE_CONFIG, num_experts=4, re_entry_gift=10.0)
    signal = RealisedValueReward(config)
    wealth = torch.full((4,), 100.0)
    explored = torch.zeros(1, 4, 2, dtype=torch.bool)
    explored[0, 0, 1] = True  # cell 1, token 0
    explored[0, 3, 0] = True  # cell 1, token 3

    gift = signal.gift(wealth, _settlement(explored))

    assert gift is not None
    unit = 10.0 * config.reward_scale * LOSS_REWARD_MULTIPLIER / 4
    assert gift.tolist() == pytest.approx([0.0, 2 * unit, 0.0, 0.0])
    # A padded token's explored slot is not a trade and pays nothing.
    valid = torch.tensor([[True, True, True, False]])
    masked = signal.gift(wealth, _settlement(explored, valid))
    assert masked is not None and masked.tolist() == pytest.approx([0.0, unit, 0.0, 0.0])
    # No draw, or no gift configured: nothing is paid.
    assert signal.gift(wealth, _settlement(None)) is None
    assert (
        RealisedValueReward(replace(config, re_entry_gift=0.0)).gift(wealth, _settlement(explored))
        is None
    )


def test_the_gift_reads_exactly_zero_on_decoupled():
    """The pinned arm's gate never reads the ledger the gift moves; the allocation is untouched."""
    pinned = {"persistence_coupling": PERSISTENCE_DECOUPLED}
    _, recorded = _trajectory(**pinned)
    _, gifted = _trajectory(**pinned, re_entry_gift=50.0)
    assert all(torch.equal(a, b) for a, b in zip(recorded, gifted, strict=True))

    _, live = _trajectory()
    _, live_gifted = _trajectory(re_entry_gift=50.0)
    assert any(not torch.equal(a, b) for a, b in zip(live, live_gifted, strict=True)), (
        "the live economy ignored the gift too; the pairing proves nothing"
    )


def test_a_loser_collects_the_exploration_share_of_the_gift():
    """The deviation bound's new term, measured: expected gift a step is share x amount."""
    config = replace(BASE_CONFIG, re_entry_gift=30.0)
    economy = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0, config=config)
    assert isinstance(economy.mob.wealth_updater.reward, RealisedValueReward)
    paid = torch.zeros(config.num_experts)
    steps = 400
    # Read the gift the way the settlement paid it: the explored slots per step
    # are what the wrapper below counts, without reaching into the draw.
    explored_slots = torch.zeros(config.num_experts)

    class Counting:
        def __init__(self, inner):
            self.inner = inner

        @property
        def multiplier(self):
            return self.inner.multiplier

        def __call__(self, wealth, settlement):
            return self.inner(wealth, settlement)

        def gift(self, wealth, settlement):
            gift = self.inner.gift(wealth, settlement)
            if gift is not None:
                paid.add_(gift)
                assert settlement.explored is not None
                for cell in range(config.num_experts):
                    explored_slots[cell] += float(
                        (settlement.explored & (settlement.selected_experts == cell)).sum()
                    )
            return gift

    economy.mob.wealth_updater = replace(
        economy.mob.wealth_updater, reward=Counting(economy.mob.wealth_updater.reward)
    )
    for _ in range(steps):
        economy.step()

    tokens = economy.batch_size * economy.seq_len
    unit = 30.0 * config.reward_scale * LOSS_REWARD_MULTIPLIER / tokens
    assert torch.allclose(paid, explored_slots * unit, atol=1e-4)
    share = config.exploration_rate / (config.num_experts - config.top_k)
    # The losers are every cell but the two the ledger seats.
    losers = paid[economy.mob.expert_wealth.argsort()[: config.num_experts - config.top_k]]
    assert losers.numel() == 6
    # Per step, a loser's expected gift is share x tokens x unit, within the draw's noise.
    expected = share * tokens * unit * steps
    assert losers.mean().item() == pytest.approx(expected, rel=0.35)


def test_arms_at_different_gifts_are_not_at_parity_and_a_legacy_run_paid_none():
    gifted = replace(BASE, re_entry_gift=30.0)
    with pytest.raises(ParityError, match="re_entry_gift"):
        assert_parity([BASE, replace(gifted, persistence_coupling=PERSISTENCE_DECOUPLED)])
    assert BASE.re_entry_gift == 0.0
