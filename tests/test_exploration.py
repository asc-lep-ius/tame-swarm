"""The exploration slot: developmental noise in the allocation, not in the reports.

A head is trained only on the tokens its expert holds. An expert whose truthful
report has fallen to zero therefore never holds another token, never sees another
target, and never comes back -- on the planted-competence fixture the market
collapsed to two of eight experts. The auction hands the last slot to a random
loser on a small fraction of training tokens so every head keeps sampling. What
must hold: the gift is report-independent, it costs the explorer nothing, and it
leaves every other slot and every other price exactly as the auction set them.
"""

import pytest
import torch

from mob import MixtureOfBidders, MoBConfig, VCGAuctioneer


def _auction(exploration_rate: float, num_experts: int = 6, top_k: int = 2) -> VCGAuctioneer:
    auctioneer = VCGAuctioneer(num_experts, top_k, exploration_rate=exploration_rate)
    auctioneer.train()
    return auctioneer


def test_exploration_is_off_in_evaluation():
    auctioneer = _auction(0.5)
    auctioneer.eval()
    outcome = auctioneer(torch.rand(4, 16, 6), torch.ones(6))
    assert outcome.explored is None


def test_exploration_is_off_at_zero_rate():
    outcome = _auction(0.0)(torch.rand(4, 16, 6), torch.ones(6))
    assert outcome.explored is None


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
def test_exploration_rate_outside_the_unit_interval_is_rejected(rate):
    with pytest.raises(ValueError, match="exploration_rate"):
        VCGAuctioneer(6, 2, exploration_rate=rate)
    with pytest.raises(ValueError, match="exploration_rate"):
        MoBConfig(exploration_rate=rate)


def test_explored_slots_go_to_losers_at_the_configured_rate_and_cost_nothing():
    torch.manual_seed(0)
    auctioneer = _auction(0.25)
    sold = VCGAuctioneer(6, 2)
    sold.eval()
    confidences = torch.rand(8, 64, 6)
    wealth = torch.rand(6) * 10 + 1

    outcome = auctioneer(confidences, wealth)
    reference = sold(confidences, wealth)
    assert outcome.explored is not None

    explored = outcome.explored
    assert explored[..., 0].sum() == 0, "only the last slot is ever handed out"
    fraction = explored[..., 1].float().mean().item()
    assert fraction == pytest.approx(0.25, abs=0.04)

    # The explorer is never an expert the auction had already sold a slot to.
    explorer = outcome.selected_experts[..., 1][explored[..., 1]]
    incumbent = reference.selected_experts[..., 0][explored[..., 1]]
    displaced = reference.selected_experts[..., 1][explored[..., 1]]
    assert (explorer != incumbent).all() and (explorer != displaced).all()

    assert (outcome.payments[..., 1][explored[..., 1]] == 0).all()
    # Every other slot and price is the auction's own.
    assert torch.equal(outcome.selected_experts[..., 0], reference.selected_experts[..., 0])
    assert torch.allclose(outcome.payments[..., 0], reference.payments[..., 0])
    untouched = ~explored[..., 1]
    assert torch.equal(
        outcome.selected_experts[..., 1][untouched], reference.selected_experts[..., 1][untouched]
    )
    assert torch.allclose(
        outcome.payments[..., 1][untouched], reference.payments[..., 1][untouched]
    )
    assert torch.allclose(outcome.rebates, reference.rebates), "rebates read bids, not slots"


def test_every_loser_is_equally_likely_to_be_handed_the_slot():
    torch.manual_seed(1)
    auctioneer = _auction(1.0 - 1e-9, num_experts=5, top_k=1)
    # Expert 0 wins every token outright; the four losers should share the slot.
    confidences = torch.full((16, 256, 5), 0.1)
    confidences[..., 0] = 5.0
    outcome = auctioneer(confidences, torch.ones(5))
    counts = torch.bincount(outcome.selected_experts.flatten(), minlength=5).float()
    assert counts[0] == 0
    assert (counts[1:] / counts[1:].sum()).allclose(torch.full((4,), 0.25), atol=0.03)


def test_a_losers_report_cannot_change_what_exploration_hands_it():
    """Report independence: the gift is drawn before any report is read."""
    auctioneer = _auction(0.5)
    wealth = torch.ones(6)
    confidences = torch.rand(4, 32, 6)
    confidences[..., 0] = 0.0  # expert 0 loses everywhere

    torch.manual_seed(7)
    baseline = auctioneer(confidences, wealth)
    for report in (0.05, 0.1, 0.2):
        torch.manual_seed(7)
        perturbed = confidences.clone()
        perturbed[..., 0] = report  # still below every other bid
        outcome = auctioneer(perturbed, wealth)
        assert torch.equal(outcome.explored, baseline.explored)
        assert torch.equal(outcome.selected_experts, baseline.selected_experts)


def test_no_loser_means_nothing_to_explore():
    outcome = _auction(0.9, num_experts=3, top_k=3)(torch.rand(2, 8, 3), torch.ones(3))
    assert outcome.explored is not None and not outcome.explored.any()


def test_explored_slots_are_not_counted_as_trades():
    """Surplus is a statement about the market; a gift is not a trade."""
    torch.manual_seed(3)
    config = MoBConfig(
        num_experts=4,
        top_k=2,
        hidden_dim=16,
        intermediate_dim=32,
        adapter_rank=4,
        adapter_alpha=4.0,
        exploration_rate=0.5,
    )
    mob = MixtureOfBidders(config)
    mob.train()
    with torch.no_grad():
        for name, param in mob.experts.named_parameters():
            if name.endswith("_B.weight"):
                param.normal_(std=0.1)

    output = mob(torch.randn(2, 32, 16))
    (output * torch.randn_like(output)).sum().backward()
    mob.update_wealth_from_loss(torch.ones(2, 32))

    explored = mob._cached_explored
    assert explored is not None and explored.any() and not explored.all()
    values = mob.last_realised_values
    confidences = mob.last_stats.confidences
    selected = mob.last_stats.selected_experts
    traded = ~explored
    expected_value = values[traded].mean()
    expected_report = torch.gather(confidences, -1, selected)[traded].mean()

    summary = mob.last_value_summary
    assert summary is not None
    assert summary.mean_realised_value.item() == pytest.approx(expected_value.item(), abs=1e-6)
    assert summary.mean_report.item() == pytest.approx(expected_report.item(), abs=1e-6)
