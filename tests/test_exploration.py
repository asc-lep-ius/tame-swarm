"""The exploration slot: developmental noise in the allocation, not in the reports.

A head is trained only on the tokens its expert holds. An expert whose truthful
report has fallen to zero therefore never holds another token, never sees another
target, and never comes back -- on the planted-competence fixture the market
collapsed to two of eight experts. The auction hands one slot to a random loser on
a small fraction of training tokens so every head keeps sampling. What must hold:
the gift is drawn before any report is read, it costs the explorer nothing, it is
funded out of the token's rebate, it leaves every other slot and price exactly as
the auction set them, and what it does to the incentive claim is bounded by the
rate and measured rather than asserted away.
"""

import pytest
import torch

from mob import MixtureOfBidders, MoBConfig, VCGAuctioneer

PAYMENT_TOLERANCE = 1e-5


def _auction(exploration_rate: float, num_experts: int = 6, top_k: int = 2) -> VCGAuctioneer:
    auctioneer = VCGAuctioneer(num_experts, top_k, exploration_rate=exploration_rate)
    auctioneer.train()
    return auctioneer


def _sold(num_experts: int = 6, top_k: int = 2) -> VCGAuctioneer:
    """The same auction with nothing handed out: exploration is off in evaluation."""
    auctioneer = VCGAuctioneer(num_experts, top_k)
    auctioneer.eval()
    return auctioneer


def _funding_factor(wealth: torch.Tensor, top_k: int) -> float:
    reciprocals = 1.0 / torch.topk(wealth, top_k).values
    return (reciprocals[: top_k - 1].sum() / reciprocals.sum()).item()


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
    confidences = torch.rand(8, 64, 6)
    wealth = torch.rand(6) * 10 + 1

    outcome = auctioneer(confidences, wealth)
    reference = _sold()(confidences, wealth)
    explored = outcome.explored
    assert explored is not None

    explored_token = explored.any(dim=-1)
    assert (explored.sum(dim=-1) <= 1).all(), "at most one slot per token is handed out"
    assert explored_token.float().mean().item() == pytest.approx(0.25, abs=0.04)

    # The explorer is never an expert the auction had already sold a slot to.
    explorer = outcome.selected_experts[explored]
    for slot in range(2):
        incumbent = (
            reference.selected_experts[..., slot].unsqueeze(-1).expand_as(explored)[explored]
        )
        assert (explorer != incumbent).all()

    assert (outcome.payments[explored] == 0).all()
    # Every other slot and price is the auction's own.
    untouched = ~explored
    assert torch.equal(outcome.selected_experts[untouched], reference.selected_experts[untouched])
    assert torch.allclose(outcome.payments[untouched], reference.payments[untouched])

    # The rebate is the auction's on unexplored tokens and scaled on explored ones.
    assert torch.allclose(outcome.rebates[~explored_token], reference.rebates[~explored_token])
    factor = _funding_factor(wealth, 2)
    assert 0.0 < factor < 1.0
    assert torch.allclose(
        outcome.rebates[explored_token], reference.rebates[explored_token] * factor, atol=1e-6
    )


def test_every_slot_is_equally_likely_to_be_explored():
    """A fixed last slot would let a marginal winner overreport into a safer one."""
    torch.manual_seed(2)
    outcome = _auction(0.5, num_experts=6, top_k=3)(torch.rand(16, 256, 6), torch.ones(6))
    assert outcome.explored is not None
    per_slot = outcome.explored.float().sum(dim=(0, 1))
    assert (per_slot / per_slot.sum()).allclose(torch.full((3,), 1 / 3), atol=0.03)


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
    """The gift is drawn before any report is read."""
    auctioneer = _auction(0.5)
    wealth = torch.ones(6)
    torch.manual_seed(3)
    confidences = 0.5 + 0.5 * torch.rand(4, 32, 6)
    confidences[..., 0] = 0.0  # expert 0 loses everywhere, whatever it reports below 0.5

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


@pytest.mark.parametrize("rate", [0.02, 0.3, 0.9])
@pytest.mark.parametrize(("num_experts", "top_k"), [(6, 2), (8, 3), (5, 1), (4, 2)])
def test_rebates_stay_affordable_when_slots_are_explored(rate, num_experts, top_k):
    """Budget feasibility, in credits, with the explored slot's payment gone.

    Without funding the explored token's rebate returns more than the token
    collected: measured at a rate of 0.3 on a flat wealth vector the transfer
    returned 110% of the collection. Every explored token must stay inside what
    its remaining payments cover.
    """
    torch.manual_seed(37)
    auctioneer = _auction(rate, num_experts=num_experts, top_k=top_k)
    wealth = torch.linspace(15.0, 750.0, num_experts)
    outcome = auctioneer(torch.rand(4, 64, num_experts), wealth)
    assert outcome.explored is not None and outcome.explored.any()

    collected = outcome.payments.sum(dim=-1)
    returned = outcome.rebates.sum(dim=-1)
    assert (returned <= collected + PAYMENT_TOLERANCE).all(), (
        f"n={num_experts} k={top_k} rate={rate}: rebate exceeds revenue on an explored token"
    )
    explored_token = outcome.explored.any(dim=-1)
    if top_k == 1:
        assert (returned[explored_token] == 0).all(), "a token with no collection rebates nothing"
    else:
        assert (returned[explored_token] > 0).any(), "funding must scale the rebate, not zero it"


def _expected_utility(
    auctioneer: VCGAuctioneer, report: float, true_value: float, draws: int
) -> float:
    """Expert 0's expected quasi-linear payoff, averaged over the exploration draws.

    The same token is tiled ``draws`` times so the average is taken in one forward.
    Influence is the share renormalised by an equal split, identically 1.0 under
    the uniform rule, so the payoff is ``value * 1[holds a slot] - payment``.
    """
    confidences = torch.tensor([report, 0.30, 0.40, 0.15, 0.20]).view(1, 1, -1).expand(1, draws, -1)
    wealth = torch.tensor([1.0, 1.25, 0.9, 1.4, 1.5])
    outcome = auctioneer(confidences.contiguous(), wealth)

    holds = outcome.selected_experts == 0
    influence = (outcome.routing_weights * auctioneer.top_k * holds).sum(dim=-1)
    paid = (outcome.payments * holds).sum(dim=-1)
    return (true_value * influence - paid).mean().item()


def test_deviation_gain_is_bounded_by_the_exploration_rate():
    """Strategyproof up to O(exploration_rate), measured rather than asserted.

    Exact strategyproofness is pinned at a rate of zero by
    ``test_truthful_reporting_maximises_expert_utility``. With a slot handed out,
    a winner faces a ``rate / k`` chance of displacement it cannot bid away and a
    loser an expected ``rate / (n - k)`` share of the gift, so a deviation can be
    worth at most ``rate x value`` to the deviator -- and a fixed last slot would
    have let a marginal winner take ``rate x (value - price)`` for free by
    overreporting into the slot that is never displaced.
    """
    rate = 0.1
    auctioneer = _auction(rate, num_experts=5, top_k=2)
    # Expert 0 takes a slot once its bid clears the second-highest rival bid,
    # 0.40 x 0.9, restated in its own units by its wealth of 1.0.
    critical_value = 0.40 * 0.9 / 1.0
    draws = 20000

    for true_value in (0.30, 0.365, 0.38, 0.45, 0.60):
        torch.manual_seed(5)
        truthful = _expected_utility(auctioneer, true_value, true_value, draws)
        assert (true_value > critical_value) or truthful >= -1e-6
        for report in torch.linspace(0.0, 1.0, 21).tolist():
            torch.manual_seed(5)
            deviation = _expected_utility(auctioneer, report, true_value, draws)
            assert deviation <= truthful + rate * true_value + 1e-3, (
                f"value {true_value:.2f}: reporting {report:.2f} gained "
                f"{deviation - truthful:.4f} against a bound of {rate * true_value:.4f}"
            )
            # Overreporting buys nothing: with the slot drawn uniformly, a
            # winner's chance of displacement does not depend on which slot it
            # holds, so there is no safer slot to climb into. A fixed last slot
            # paid rate x (value - price) for exactly that climb. Underreporting
            # is what the O(rate) bound above is for -- a marginal winner can
            # give up a thin surplus for the lottery's expected gift.
            if report > true_value:
                assert deviation <= truthful + 1e-3, (
                    f"value {true_value:.2f}: overreporting {report:.2f} gained "
                    f"{deviation - truthful:.4f}"
                )

    assert 0.30 < critical_value < 0.45, "the sweep must straddle the critical value"


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
