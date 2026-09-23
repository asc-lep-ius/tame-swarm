"""The blockade hooks do what they say, and the readouts read what they claim (#63).

Three things a blockade read depends on and none of which a diff shows: that
the output block leaves a contribution of *exactly* zero while touching neither
the head nor the ledger; that the ledger pin holds through a settlement rather
than being overwritten by it; and that the probe pairs -- two economies from
one seed walk the same steps bitwise, so an unblocked twin is a control and not
a different run. The readouts are then checked on hand-built shares where the
answer is arithmetic.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))

import blockade as driver  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    SyntheticEconomy,
    shuffled,
)

from individuation import (  # noqa: E402
    dominant_cell,
    gains,
    half_life,
    paired_t,
    planted_statistic,
    predicted_substitute,
    reconvergence_step,
    returned,
    type_shares,
    uptake,
    winners,
)
from mob import PERSISTENCE_DECOUPLED, LightweightExpert  # noqa: E402

torch.set_num_threads(1)
SETTLE = 40
WINDOW = 10


def economy(seed: int = 0, arm: str = "value") -> SyntheticEconomy:
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    return SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)


def head_state(economy: SyntheticEconomy) -> list[torch.Tensor]:
    return [p.detach().clone() for p in economy.mob.confidence_heads.parameters()]


def test_output_block_zeroes_the_contribution_exactly_and_touches_nothing_else():
    eco = economy()
    for _ in range(SETTLE):
        eco.step()
    cell = 0
    heads, wealth = head_state(eco), eco.mob.expert_wealth.clone()
    expert = eco.mob.experts[cell]
    assert isinstance(expert, LightweightExpert)
    planted = expert.down_adapter_B.weight.detach().clone()
    assert planted.abs().sum() > 0

    eco.block_output(cell)
    x = torch.randn(4, eco.config.hidden_dim)
    held, reference = expert.forward_with_reference(
        x, eco.mob.base_gate_proj, eco.mob.base_up_proj, eco.mob.base_down_proj
    )
    assert torch.equal(held, reference), "a blocked cell's output must be the base's, bitwise"
    assert all(torch.equal(a, b) for a, b in zip(heads, head_state(eco), strict=True))
    assert torch.equal(wealth, eco.mob.expert_wealth)

    eco.release_output(cell)
    assert torch.equal(expert.down_adapter_B.weight.detach(), planted)


def test_ledger_pin_holds_through_settlement_and_leaves_the_bid_alone():
    eco = economy()
    for _ in range(SETTLE):
        eco.step()
    cell = int(eco.mob.expert_wealth.argmax())
    assert float(eco.mob.expert_wealth[cell]) > eco.config.min_wealth
    heads = head_state(eco)

    eco.pin_wealth(cell)
    assert float(eco.mob.expert_wealth[cell]) == eco.config.min_wealth
    assert all(torch.equal(a, b) for a, b in zip(heads, head_state(eco), strict=True))
    for _ in range(WINDOW):
        record = eco.step()
        assert float(record.selected_experts.numel()) > 0
        assert float(eco.mob.expert_wealth[cell]) == eco.config.min_wealth
    eco.release_wealth(cell)
    for _ in range(WINDOW):
        eco.step()
    # Released, the ledger moves again: the pin was a hold, not a rewrite.
    assert float(eco.mob.expert_wealth[cell]) != eco.config.min_wealth


def test_two_economies_from_one_seed_walk_the_same_steps_bitwise():
    """The pairing every reading rests on: the control is the same tokens, unblocked.

    Run one after the other, never interleaved: the exploration draw reads the
    global stream, which a constructor reseeds, so two economies stepped in
    turn share one stream and diverge -- which is why ``read`` settles one
    economy at a time and never two.
    """
    a = economy(3)
    first = [a.step() for _ in range(SETTLE)]
    b = economy(3)
    second = [b.step() for _ in range(SETTLE)]
    for ra, rb in zip(first, second, strict=True):
        assert ra.loss == rb.loss
        assert torch.equal(ra.selected_experts, rb.selected_experts)
    assert torch.equal(a.mob.expert_wealth, b.mob.expert_wealth)


def test_observing_does_not_move_the_economy():
    """The probe never pays: counting a window leaves the trajectory the plain run's."""
    plain = economy(1)
    losses = [plain.step().loss for _ in range(SETTLE)]
    observed = economy(1)
    window = driver.observe(observed, SETTLE)
    assert window.loss_by_type[0] == pytest.approx(losses)
    assert torch.equal(plain.mob.expert_wealth, observed.mob.expert_wealth)


def test_the_ledger_blockade_does_not_reach_the_allocation_on_decoupled():
    """Under ``decoupled`` the gate reads a pinned snapshot: pinning the ledger changes no route."""
    control = economy(2, PERSISTENCE_DECOUPLED)
    for _ in range(SETTLE):
        control.step()
    routes = [control.step().selected_experts for _ in range(WINDOW)]
    pinned = economy(2, PERSISTENCE_DECOUPLED)
    for _ in range(SETTLE):
        pinned.step()
    pinned.pin_wealth(0)
    for route in routes:
        assert torch.equal(pinned.step().selected_experts, route)


def test_on_type_counts_split_the_slots_by_the_tokens_type():
    eco = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0)
    window = driver.observe(eco, WINDOW)
    total_slots = WINDOW * eco.batch_size * eco.seq_len * eco.config.top_k
    assert float(window.wins_by_type.sum()) == total_slots
    assert window.own_type_wins(eco.expert_types).shape == (eco.config.num_experts,)


# --- the readouts, on hand-built shares -------------------------------------------------

PRE = torch.tensor([0.5, 0.48, 0.01, 0.005, 0.005, 0.0, 0.0, 0.0])
COMPETENCE = torch.tensor([0.9, 0.7, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1])
ON_TYPE = torch.ones(8, dtype=torch.bool)


def test_dominant_cell_is_the_one_delivering_the_most_correction():
    wins = torch.tensor([100.0, 100.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert dominant_cell(wins, COMPETENCE) == 0
    # A more competent cell that is shut out does not dominate.
    assert dominant_cell(torch.tensor([0.0, 100.0, 90.0, 0, 0, 0, 0, 0.0]), COMPETENCE) == 1


def test_uptake_is_the_fraction_of_the_blocked_share_released():
    inside = torch.tensor([0.1, 0.48, 0.41, 0.005, 0.005, 0.0, 0.0, 0.0])
    assert uptake(PRE, inside, 0) == pytest.approx(0.8)
    assert uptake(PRE, PRE, 0) == 0.0
    assert uptake(PRE, torch.tensor([0.6, 0.38, 0.01, 0.005, 0.005, 0, 0, 0.0]), 0) < 0
    with pytest.raises(ValueError):
        uptake(PRE, inside, 7)


def test_predicted_substitute_is_the_best_cell_not_already_winning():
    assert winners(PRE, 2) == [0, 1]
    assert predicted_substitute(COMPETENCE, ON_TYPE, PRE, blocked=0, top_k=2) == 2
    # Two cells a type, one blocked and one already winning: nothing to predict.
    two = torch.tensor([True, True, False, False, False, False, False, False])
    assert predicted_substitute(COMPETENCE, two, PRE, blocked=0, top_k=2) is None


def test_planted_statistic_reads_positive_when_the_share_went_by_competence():
    right = gains(PRE, torch.tensor([0.1, 0.48, 0.41, 0.005, 0.005, 0.0, 0.0, 0.0]))
    scattered = gains(PRE, torch.tensor([0.1, 0.48, 0.09, 0.085, 0.085, 0.08, 0.08, 0.0]))
    wrong = gains(PRE, torch.tensor([0.1, 0.48, 0.01, 0.005, 0.005, 0.0, 0.0, 0.4]))
    assert planted_statistic(right, ON_TYPE, PRE, 0, 2, 2) == pytest.approx(0.4)
    assert abs(planted_statistic(scattered, ON_TYPE, PRE, 0, 2, 2) or 1.0) < 0.02
    assert planted_statistic(wrong, ON_TYPE, PRE, 0, 2, 2) < 0
    two = torch.tensor([True, True, True, False, False, False, False, False])
    assert planted_statistic(right, two, PRE, 0, 2, 2) is None


def test_returned_and_half_life():
    assert returned(PRE, torch.tensor([0.4, 0.5, 0.1, 0, 0, 0, 0, 0.0]), 0) == pytest.approx(0.8)
    assert half_life([0.5, 0.5, 0.0, 0.0, 0.1, 0.1], before=0.5, final=0.1) == 4
    assert half_life([0.5, 0.5, 0.5], before=0.5, final=0.5) is None


def test_reconvergence_step_reads_the_trailing_mean():
    losses = [1.0] * 10 + [0.2] * 10
    assert reconvergence_step(losses, target=0.2, factor=1.1, trailing=5) == 15
    assert reconvergence_step([1.0] * 10, target=0.2, factor=1.1, trailing=5) is None
    with pytest.raises(ValueError):
        reconvergence_step(losses, 0.2, 1.1, 0)


def test_type_shares_and_paired_t():
    assert type_shares(torch.tensor([3.0, 1.0])).tolist() == [0.75, 0.25]
    assert type_shares(torch.zeros(2)).tolist() == [0.0, 0.0]
    test = paired_t([0.4, 0.5, 0.6, 0.5, 0.45, 0.55])
    assert test.n == 6 and test.mean == pytest.approx(0.5) and test.p < 0.001
    flat = paired_t([0.3, 0.3, 0.3])
    assert flat.sd == 0.0 and flat.p == 0.0


def test_read_pairs_its_control_and_refuses_an_unpaired_one(monkeypatch):
    monkeypatch.setattr(driver, "SETTLE_STEPS", 120)
    blocked = driver.read(driver.Job(driver.QUALITY, "value", 0, driver.LEDGER, WINDOW))
    control = driver.read(driver.Job(driver.QUALITY, "value", 0, driver.NONE, WINDOW))
    assert blocked["pre_shares"] == control["pre_shares"]
    assert blocked["blocked_cell"] == control["blocked_cell"]
    # The pinned cell bids at the floor from the first blocked step, so it loses
    # nearly everything inside the window: the ledger blockade substitutes at once.
    assert blocked["uptake"] > 0.9
    deltas = driver.paired_against_control({"0": blocked}, {"0": control}, "uptake")
    assert deltas == {"0": pytest.approx(blocked["uptake"] - control["uptake"])}
    other = driver.read(driver.Job(driver.QUALITY, "value", 1, driver.NONE, WINDOW))
    with pytest.raises(AssertionError):
        driver.paired_against_control({"0": blocked}, {"0": other}, "uptake")
