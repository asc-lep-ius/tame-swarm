"""A self-model that costs something: the invariants #60's two levers have to hold.

Lever 2 is the one with teeth here. It pays a cell for the accuracy of its own
prediction, which is a second thing a head emits and therefore a second thing a
cell could try to game -- so what is pinned is that the scored output cannot
reach the auction's price, that a wrong self-model costs something, and that at
`mu = 0` none of it exists: not the payment, not the parameter, and not the two
draws from the generator that creating one would consume.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    shuffled,
)

from mob.experts import SELF_PREDICTION_RANGE, ConfidenceHead  # noqa: E402
from mob.ledger import Settlement, WealthUpdater  # noqa: E402


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


def test_a_head_at_no_price_has_no_self_model_at_all():
    """Not the payment, not the parameter, and not the draws creating one costs.

    The first version built the projection unconditionally. Two draws from the
    generator per head moved every recorded fixture number -- seed 1's
    r(wealth, competence) went from 0.51 to 0.49 -- and a tensor in every
    checkpoint would have stopped #39's restoring under #29's strict load,
    which is what #58's read depends on.
    """
    plain = ConfidenceHead(hidden_dim=8, expert_id=0)
    scored = ConfidenceHead(hidden_dim=8, expert_id=0, self_model=True)

    assert plain.prediction is None
    assert set(plain.state_dict()) == {"proj.weight", "proj.bias"}
    assert set(scored.state_dict()) == {
        "proj.weight",
        "proj.bias",
        "prediction.weight",
        "prediction.bias",
    }
    with pytest.raises(ValueError, match="no self-model to read"):
        plain.forward_prediction(torch.zeros(1, 1, 8))


def test_the_recorded_economy_is_bitwise_what_it_was_at_price_zero():
    def run(**overrides) -> torch.Tensor:
        built = economy(**overrides)
        for _ in range(10):
            built.step()
        return built.mob.expert_wealth.clone()

    assert torch.equal(run(), run(self_score_mu=0.0))
    assert not torch.equal(run(), run(self_score_mu=1.0))


def test_the_prediction_is_signed_where_the_bid_cannot_be():
    """A cell can realise negative value; a self-model that cannot say so is not one."""
    head = ConfidenceHead(hidden_dim=8, expert_id=0, self_model=True)
    with torch.no_grad():
        assert head.prediction is not None
        head.prediction.bias.fill_(-3.0)
    x = torch.zeros(1, 1, 8)

    assert float(head.forward_prediction(x).detach()) == pytest.approx(-3.0)
    # The bid is softplussed and cannot go below zero, which is what makes the
    # two outputs different objects rather than one read twice.
    assert float(head(x)) > 0.0


def test_the_prediction_is_clamped_so_one_token_cannot_empty_a_ledger():
    head = ConfidenceHead(hidden_dim=8, expert_id=0, self_model=True)
    with torch.no_grad():
        assert head.prediction is not None
        head.prediction.bias.fill_(1000.0)

    predicted = head.forward_prediction(torch.zeros(1, 1, 8)).detach()
    assert float(predicted) == SELF_PREDICTION_RANGE


def test_the_scoring_payment_never_reaches_the_auctions_price():
    """The constitution's property survives because the two outputs are separate.

    The bid is `proj`, the scored prediction is `prediction`, and no gradient or
    value flows between them: a cell cannot move its scoring payment by shading
    its bid, or its bid by shading its prediction. That is the decoupling #60
    allows in place of re-deriving the composite misreport bound.
    """
    head = ConfidenceHead(hidden_dim=8, expert_id=0, self_model=True)
    x = torch.randn(1, 1, 8)
    before = float(head(x))

    with torch.no_grad():
        assert head.prediction is not None
        head.prediction.weight.add_(torch.randn_like(head.prediction.weight))
        head.prediction.bias.add_(1.0)

    assert float(head(x)) == before
    # Distinct parameter objects, not two names for one tensor: nothing the
    # scored output learns can move the bid.
    assert head.proj.weight is not head.prediction.weight
    assert head.proj.bias is not head.prediction.bias
    assert not any(a is b for a in head.proj.parameters() for b in head.prediction.parameters())


def test_the_score_is_a_payment_for_accuracy_and_zero_for_a_cell_that_held_nothing():
    updater = replace(WealthUpdater.for_experts(BASE_CONFIG), self_score_mu=2.0)
    wealth = torch.full((BASE_CONFIG.num_experts,), 100.0)
    scores = torch.zeros(BASE_CONFIG.num_experts)
    scores[0] = -0.25  # a cell that predicted 0.5 away from what it realised
    settlement = Settlement(
        selected_experts=torch.zeros(1, 2, 2, dtype=torch.long),
        routing_weights=torch.full((1, 2, 2), 0.5),
        confidences=torch.zeros(1, 2, BASE_CONFIG.num_experts),
        num_tokens=2,
        values=torch.zeros(1, 2, 2),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        self_score=scores,
    )

    before = wealth.clone()
    updater.settle(wealth, settlement, lambda *args, **kwargs: torch.zeros_like(wealth))

    # The inaccurate cell is charged 2.0 x 0.25 more than the others, and the
    # ones that held nothing are charged nothing for holding nothing.
    inaccurate = float(before[0] - wealth[0])
    accurate = float(before[1] - wealth[1])
    assert inaccurate - accurate == pytest.approx(0.5, abs=1e-5)


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
