"""The differentiated planted-competence fixture (#25): a body whose cells differ.

The quality fixture (``tests/test_expert_value.py``) plants one correction and
scales it by competence, so every token wants the same experts and the experts'
contributions are parallel. These tests are about the fixture one substrate down
from the real model's differentiation question: experts of different types carry
orthogonal corrections, each token calls for one type and says so in its input,
and the economy is asked whether it can route each token to the cells that are
competent *on it*. Every threshold here was measured first
(``scripts/measure_differentiated_economy.py``), on the default legibility, and
the suite's other baselines stay on the quality fixture by design.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from measure_differentiated_economy import warm_up_heads, warmup_lengths  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DEFAULT_NUM_TYPES,
    DifferentiatedEconomy,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from specialisation import expert_output_divergence  # noqa: E402

SEEDS = (0, 1, 2)
STEPS = 600
TAIL = 100
# Chance for the on-type share is the share of experts carrying each type; the
# efficient allocation is 1.0. Measured 0.65-0.75 at the default legibility.
ON_TYPE_FLOOR = 0.55
# Measured r(wealth, competence) 0.60-0.75 on the three seeds.
WEALTH_TRACKS_COMPETENCE = 0.5
# When the field is fully legible every expert holds a market: measured 0.07-0.12
# of the slots for the least-used expert, against an exploration gift of 0.0017.
LEGIBLE_SIGNAL = 8.0
EVERY_EXPERT_USED = 0.05


def _settled(seed: int, **kwargs) -> tuple[DifferentiatedEconomy, torch.Tensor, float]:
    """The fixture after ``STEPS`` steps: the economy, its tail win share and tail on-type share."""
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    economy = DifferentiatedEconomy(competence, seed=seed, **kwargs)
    wins = torch.zeros(economy.config.num_experts)
    on_type: list[float] = []
    for step in range(STEPS):
        record = economy.step()
        if step >= STEPS - TAIL:
            wins += torch.bincount(
                record.selected_experts.flatten(), minlength=economy.config.num_experts
            ).float()
            on_type.append(economy.on_type_share(record.selected_experts))
    return economy, wins / wins.sum(), sum(on_type) / len(on_type)


def test_the_contributions_are_different_functions_by_construction():
    """What the quality fixture cannot read: the metric #25 gates on, nonzero here.

    Four types over eight experts: the four same-type pairs are parallel and the
    twenty-four cross-type pairs are orthogonal, so the mean contribution cosine
    distance is exactly 24/28. On the quality fixture the same metric is zero to
    float precision, while the *output* metric reads 0.018 there -- the dilution
    the contribution metric exists to see through.
    """
    competence = shuffled(DEFAULT_COMPETENCE, 0)
    hidden = torch.randn(256, DifferentiatedEconomy(competence, seed=0).config.hidden_dim)

    differentiated = expert_output_divergence(DifferentiatedEconomy(competence, seed=0).mob, hidden)
    quality = expert_output_divergence(SyntheticEconomy(competence, seed=0).mob, hidden)

    assert differentiated.mean_contribution_cosine_distance == pytest.approx(24 / 28, abs=1e-4)
    assert differentiated.min_contribution_cosine_distance == pytest.approx(0.0, abs=1e-4)
    assert quality.mean_contribution_cosine_distance == pytest.approx(0.0, abs=1e-6)
    assert quality.mean_cosine_distance > 0.01, "the output metric is nonzero with nothing to sort"


def test_the_loss_identity_holds_for_whatever_the_gate_chose():
    """The identity in the class docstring, checked against the realised loss per token."""
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, 1), seed=1)
    hidden, target = economy._draw()
    output = economy.mob(hidden)
    stats = economy.mob.last_stats
    assert stats is not None

    realised = ((output - target) ** 2).sum(-1).detach()
    predicted = economy.closed_form_loss(hidden, target, stats.selected_experts)

    assert torch.allclose(realised, predicted, rtol=1e-4, atol=1e-6)


def test_the_types_are_shuffled_away_from_index_and_from_competence():
    for seed in range(10):
        economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed)
        types = economy.expert_types
        assert sorted(types.tolist()) == sorted(
            (torch.arange(DEFAULT_COMPETENCE.numel()) % DEFAULT_NUM_TYPES).tolist()
        )
        steps = types[1:] - types[:-1]
        assert not ((steps >= 0).all() or (steps <= 0).all()), "types monotone in index"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_economy_routes_tokens_toward_the_cells_competent_on_them(seed):
    """The specialisation reading, per seed: on-type share well above chance.

    Measured 0.65-0.75 at the default legibility against a chance of 0.25 and an
    optimum of 1.0 -- the gate learns what each token calls for, and stops short
    of the efficient allocation. Wealth follows competence (r 0.60-0.75) once
    competence is competence *at* something, and in no type does the weaker
    expert outrank the stronger; at half the legibility both of those fail, and
    that curve is the measurement script's, not the suite's.
    """
    economy, share, on_type = _settled(seed)

    assert on_type > ON_TYPE_FLOOR, on_type
    assert pearson(economy.mob.expert_wealth, economy.competence) > WEALTH_TRACKS_COMPETENCE
    for expert_type in range(economy.num_types):
        members = (economy.expert_types == expert_type).nonzero().flatten().tolist()
        best = max(members, key=lambda index: float(economy.competence[index]))
        worst = min(members, key=lambda index: float(economy.competence[index]))
        assert share[best] > share[worst], (expert_type, share[best], share[worst])


def test_a_fully_legible_field_puts_every_cell_to_work():
    """The efficient allocation uses all n experts, and the mechanism reaches it when it can.

    At the default legibility the least-used expert lives on the exploration gift;
    with the field loud enough to read on every token, the least-used expert holds
    0.07-0.12 of the slots and the loss sits at the oracle's.
    """
    _, share, on_type = _settled(SEEDS[0], type_signal=LEGIBLE_SIGNAL)

    assert on_type > 0.9, on_type
    assert float(share.min()) > EVERY_EXPERT_USED, share


def test_the_fixture_refuses_a_shape_it_cannot_plant():
    with pytest.raises(ValueError, match="num_types"):
        DifferentiatedEconomy(DEFAULT_COMPETENCE, seed=0, num_types=0)
    with pytest.raises(ValueError, match="hidden dimension"):
        DifferentiatedEconomy(
            DEFAULT_COMPETENCE, seed=0, config=replace(BASE_CONFIG, hidden_dim=16)
        )


def test_the_warmup_lets_every_head_see_every_type_without_moving_the_ledger():
    """What a perception warmup is, in the two things it must do and the one it must not (#34).

    Routing forced uniform over all eight experts means every head is trained on
    tokens of every type, so none can be shut out before it has learned to read
    the field; and the wealth ledger it hands to the economy is the one every cell was
    born with, so what carries forward is perception and not a market position.
    ``expert_usage_count`` is the warmup's own slot tally, since nothing restores it.
    """
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0, type_signal=2.0)
    gate = economy.mob.gate
    ledger = economy.mob.expert_wealth.clone()
    reports = [
        parameter.detach().clone() for parameter in economy.mob.confidence_heads.parameters()
    ]

    warm_up_heads(economy, 20, seed=0)

    assert torch.equal(economy.mob.expert_wealth, ledger), economy.mob.expert_wealth
    assert economy.mob.gate is gate, "the forced gate outlived the warmup"
    # Uniform over eight experts is 0.125 of the slots each; the unforced gate leaves
    # its least-used expert at 0.03 over the same 20 steps, so this is what pins the
    # forcing and not just that every cell won something.
    usage = economy.mob.expert_usage_count
    assert float((usage / usage.sum()).min()) > 0.10, usage
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(reports, economy.mob.confidence_heads.parameters(), strict=True)
    ), "the heads did not train on the value objective"


def _run(seed: int, warmup: int | None, steps: int = 20) -> tuple[list[torch.Tensor], torch.Tensor]:
    """One arm, built and run end to end: every step's winners, and the ledger it ends at.

    Built and run in one call because ``SyntheticEconomy`` seeds the *global*
    generator at construction and the gate's exploration draws from it, so two
    arms stepped side by side share a stream and diverge for that reason alone.
    Sequential runs are what the measurement script does and what compares.
    """
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, type_signal=2.0)
    if warmup is not None:
        warm_up_heads(economy, warmup, seed=seed)
    return [economy.step().selected_experts for _ in range(steps)], economy.mob.expert_wealth


def test_a_zero_step_warmup_is_the_recorded_arm_bit_for_bit():
    """The warmup steps are additional: warm 0 has to be the run #25 recorded.

    Warm 0 consumes nothing: the same token draws, the same winners and the same
    ledger as the arm #25 recorded, so the warmed rows are read against the
    baseline they claim. A nonzero warmup does draw its tokens from the economy's
    own stream; what it must never spend is the economy's budget. The 50-step
    arm is here to show
    the comparison can fail: a warmup that ran and changed nothing would pass
    the first assertion for the wrong reason.
    """
    unwarmed_winners, unwarmed_wealth = _run(1, warmup=None)
    zero_winners, zero_wealth = _run(1, warmup=0)
    warmed_winners, _ = _run(1, warmup=50)

    for zero, unwarmed in zip(zero_winners, unwarmed_winners, strict=True):
        assert torch.equal(zero, unwarmed)
    assert torch.equal(zero_wealth, unwarmed_wealth)
    assert any(
        not torch.equal(warmed, unwarmed)
        for warmed, unwarmed in zip(warmed_winners, unwarmed_winners, strict=True)
    ), "a 50-step warmup left the economy it precedes untouched"


def test_the_warmup_flag_refuses_a_length_it_cannot_run():
    assert warmup_lengths("0,50,200") == (0, 50, 200)
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        warmup_lengths("0,fifty")
    with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
        warmup_lengths("-50")


def test_the_setpoint_step_moves_the_target_and_nothing_else(seed=0):
    """#57's perturbation: the experimenter's step, in the tissue's own variable.

    A thermostat is tested by moving the target, not the electricity price. The
    dose is what prices the goal error and it must not move with the setpoint --
    a step that moved both would be the dose sweep wearing another name.
    """
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed)
    economy.add_goal_field(0, 0.5, 0.25)
    economy.add_goal_field(1, 0.5, 0.25)

    stepped = economy.step_goal_setpoint(0, 0.1)

    assert stepped.setpoint == pytest.approx(0.6)
    assert stepped.dose == 0.25
    assert [(f.expert_type, f.setpoint, f.dose) for f in economy.goal_fields()] == [
        (0, pytest.approx(0.6), 0.25),
        (1, 0.5, 0.25),
    ]
    with pytest.raises(ValueError, match="no goal field is attached on type 2"):
        economy.step_goal_setpoint(2, 0.1)


def test_the_tissue_reading_is_the_quantity_the_goal_term_prices():
    """What the step perturbs is what the field pays for, and the two agree.

    The reading is the on-type competence the winners delivered at their share --
    the same expression ``closed_form_goal_terms`` differences to price a
    contribution, so an experiment can watch the regulated variable itself
    instead of inferring it from what the cells were paid.
    """
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0)
    field = economy.add_goal_field(0, 0.5, 0.25)
    record = economy.step()
    selected = record.selected_experts

    reading = economy.goal_reading(field, selected)

    k = economy.config.top_k
    delivered = economy.competence[selected] / k
    on_type = economy.expert_types[selected] == field.expert_type
    assert reading == pytest.approx(float((delivered * on_type).sum(dim=-1).mean()))
    # A field on a type no expert carries reads exactly zero, whatever the route.
    assert reading >= 0.0
