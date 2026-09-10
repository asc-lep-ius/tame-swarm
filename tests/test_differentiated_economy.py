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

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

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
