"""What an expert is worth, pinned from the mechanism statement (#15).

Value is an expert's contribution against the loss gradient at its layer: the
first-order change in the organism's loss from replacing what the expert did on
a token by what the shared base would have done. It is a counterfactual against
the tissue's default behaviour, not against the expert's own history -- the
definition it replaced averaged to zero by construction -- and it is what the
auction's externality price is denominated in.

The planted-competence fixture in ``scripts/synthetic_economy.py`` is what makes
the acceptance criteria testable: competence is known, shuffled away from expert
index, and the market can be read at its steady state.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from mob import LightweightExpert, MixtureOfBidders, MoBConfig  # noqa: E402

TINY = MoBConfig(
    num_experts=3,
    top_k=2,
    hidden_dim=16,
    intermediate_dim=32,
    adapter_rank=4,
    adapter_alpha=4.0,
)


def _contributing_layer(config: MoBConfig, seed: int = 3) -> MixtureOfBidders:
    torch.manual_seed(seed)
    mob = MixtureOfBidders(config)
    mob.train()
    with torch.no_grad():
        for name, param in mob.experts.named_parameters():
            if name.endswith("_B.weight"):
                param.normal_(std=0.1)
    return mob


def test_realised_value_is_the_contribution_against_the_loss_gradient():
    """Hand-compute the definition for every winner slot of every token.

    Backwarding ``(output * g).sum()`` makes the loss gradient at the layer output
    exactly ``g``, so the value the hook captures must be
    ``-scale * <g_t, expert_output - base_output>``.
    """
    mob = _contributing_layer(TINY)
    hidden = torch.randn(1, 5, TINY.hidden_dim)
    output = mob(hidden)
    gradient = torch.randn_like(output)
    (output * gradient).sum().backward()

    mob.update_wealth_from_loss(torch.ones(1, 5), loss_gradient_scale=2.0)

    values = mob.last_realised_values
    selected = mob.last_stats.selected_experts
    assert values is not None and values.shape == (1, 5, TINY.top_k)
    for token in range(5):
        for slot in range(TINY.top_k):
            expert = mob.experts[int(selected[0, token, slot])]
            assert isinstance(expert, LightweightExpert)
            held, reference = expert.forward_with_reference(
                hidden[0, token : token + 1],
                mob.base_gate_proj,
                mob.base_up_proj,
                mob.base_down_proj,
            )
            expected = -2.0 * ((held - reference)[0] * gradient[0, token]).sum()
            assert values[0, token, slot].item() == pytest.approx(expected.item(), abs=1e-5)


def test_an_untrained_expert_realises_exactly_zero():
    """Upcycling starts every expert as the base; a contribution of nothing is worth nothing."""
    torch.manual_seed(0)
    mob = MixtureOfBidders(TINY)
    mob.train()
    output = mob(torch.randn(1, 6, TINY.hidden_dim))
    (output * torch.randn_like(output)).sum().backward()

    mob.update_wealth_from_loss(torch.ones(1, 6))

    assert mob.last_realised_values is not None
    assert torch.equal(mob.last_realised_values, torch.zeros(1, 6, TINY.top_k))


def test_settlement_without_a_backward_pays_nothing(caplog):
    """Before the backward there is no gradient and therefore no value: refuse, loudly."""
    mob = _contributing_layer(TINY)
    mob(torch.randn(1, 4, TINY.hidden_dim))
    wealth_before = mob.expert_wealth.clone()

    with caplog.at_level("WARNING"):
        mob.update_wealth_from_loss(torch.ones(1, 4))

    assert "before the loss backward" in caplog.text
    assert torch.equal(mob.expert_wealth, wealth_before)
    assert not mob._loss_feedback_pending
    assert mob.get_confidence_calibration_loss().item() == 0.0


def test_reports_converge_to_the_mean_value_not_its_positive_part():
    """Unbiased reports, on a fixture where the two estimates differ.

    Every token's realised value is +1.0 with probability 0.6 and -0.5 otherwise,
    unpredictable from the input, so the best report is the mean, 0.4. A head
    regressed onto the clamped target this replaced would settle at the mean of
    the positive part, 0.6, and every price would follow it upward.
    """
    torch.manual_seed(0)
    config = replace(TINY, num_experts=2, top_k=2)
    mob = MixtureOfBidders(config)
    mob.train()
    optimizer = torch.optim.Adam(mob.confidence_heads.parameters(), lr=0.05)
    hidden = torch.randn(1, 64, config.hidden_dim)
    draws = torch.Generator().manual_seed(1)

    for _ in range(400):
        mob(hidden)
        realised = torch.where(torch.rand(1, 64, generator=draws) < 0.6, 1.0, -0.5)
        mob._cached_values = realised.unsqueeze(-1).expand(1, 64, config.top_k).clone()
        mob.update_wealth_from_loss(torch.ones(1, 64))
        mob.get_confidence_calibration_loss().backward()
        optimizer.step()
        optimizer.zero_grad()

    mob(hidden)
    mean_report = mob.last_stats.confidences.mean().item()
    assert mean_report == pytest.approx(0.4, abs=0.05)
    assert mean_report < 0.5, "the report tracks the positive part, not the mean"


def test_first_order_value_tracks_the_exact_counterfactual():
    """The gradient estimate is first order; check it against the exact replacement.

    With the planted competence scaled down the contributions are small relative
    to the gap they close, which is the regime a trained adapter sits in, and the
    second-order term is a few percent.
    """
    economy = SyntheticEconomy(DEFAULT_COMPETENCE * 0.2, seed=0)
    record = economy.step(with_exact_values=True)

    assert record.exact_values is not None
    assert torch.equal(record.realised_values.sign(), record.exact_values.sign())
    relative_error = (
        record.realised_values - record.exact_values
    ).abs() / record.exact_values.abs()
    assert relative_error.median().item() < 0.1
    assert (record.exact_values > 0).all(), "planted competence must make every win worth something"


def test_the_competence_fixture_is_shuffled_away_from_expert_index():
    """The measurement-hygiene rule from #15, enforced on the fixture itself."""
    for seed in range(20):
        competence = shuffled(DEFAULT_COMPETENCE, seed)
        steps = competence[1:] - competence[:-1]
        assert not ((steps >= 0).all() or (steps <= 0).all())
        assert sorted(competence.tolist()) == sorted(DEFAULT_COMPETENCE.tolist())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_winning_is_profitable_for_a_competent_expert(seed):
    """The #15 acceptance criteria, on competence shuffled away from index, per seed.

    At steady state a win carries a surplus, the experts that win are the ones that
    get rich, and wealth follows the planted competence rather than the expert
    index. As filed, the same measurement read a surplus of -0.22 per win and
    r(wealth, win share) = -0.28.
    """
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    summary = SyntheticEconomy(competence, seed=seed).run(400, window=100)

    assert summary.final_surplus > 0.0
    assert summary.final_realised_value > 0.0
    assert summary.wealth_vs_win_share > 0.5
    assert summary.wealth_vs_competence > 0.5


def test_trained_reports_equal_realised_value_at_steady_state():
    """On the fixture the winners' reports are the values they realise.

    The distinction from the clamped target is pinned by the unit test above on a
    fixture with negative realised values; this is the market-level check that the
    calibrated report is neither above nor below what a win is worth.
    """
    summary = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0).run(400, window=100)

    assert summary.final_realised_value > 0.0
    assert summary.final_report == pytest.approx(summary.final_realised_value, rel=0.1)


def test_the_fixture_reads_the_market_at_its_steady_state():
    summary = SyntheticEconomy(DEFAULT_COMPETENCE, seed=0).run(20, window=5)
    assert summary.final_report != summary.mean_report
    with pytest.raises(ValueError, match="window"):
        SyntheticEconomy(DEFAULT_COMPETENCE, seed=0).run(5, window=6)


def test_wealth_does_not_track_expert_index_by_construction():
    """The initialisation artefact: r(wealth, index) ~ -0.9 whatever the competence.

    With sorted competence the sign is now the competence's own; the invariant
    negative correlation the issue recorded is what must not come back.
    """
    correlations = []
    for seed in range(3):
        summary = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed).run(300)
        correlations.append(summary.wealth_vs_index)
        assert pearson(summary.wealth, summary.competence) > 0.5
    assert max(correlations) > -0.5, f"wealth still monotone in index: {correlations}"


def test_full_experts_price_their_whole_output():
    """Without a shared base the empty slot is the reference, so the contribution is the output."""
    config = replace(TINY, use_shared_base=False)
    torch.manual_seed(1)
    mob = MixtureOfBidders(config)
    mob.train()
    hidden = torch.randn(1, 4, config.hidden_dim)
    output = mob(hidden)
    gradient = torch.randn_like(output)
    (output * gradient).sum().backward()
    mob.update_wealth_from_loss(torch.ones(1, 4))

    values = mob.last_realised_values
    selected = mob.last_stats.selected_experts
    assert values is not None
    for token in range(4):
        for slot in range(config.top_k):
            expert_output = mob.experts[int(selected[0, token, slot])](hidden[0, token : token + 1])
            expected = -(expert_output[0] * gradient[0, token]).sum()
            assert values[0, token, slot].item() == pytest.approx(expected.item(), abs=1e-5)


def test_base_config_is_the_shared_synthetic_shape():
    assert BASE_CONFIG.num_experts == DEFAULT_COMPETENCE.numel()
