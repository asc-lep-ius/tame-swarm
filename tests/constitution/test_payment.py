"""Strategyproof payment: no report beats the truthful one, and the gate that
breaks it is kept as the negative (#9's defect, #38's first property)."""

import torch

from .helpers import _UTILITY_CRITICAL_VALUE, PAYMENT_TOLERANCE, _expert_zero_utility, _make_auction


def test_truthful_reporting_maximises_expert_utility():
    """The incentive statement itself, checked by exhaustive deviation.

    Sweeping expert 0's *report* across the whole range while its true value is held
    fixed, no misreport ever beats reporting truthfully. This is the property the
    README is allowed to claim, and it needs both halves of the mechanism: an
    undivided weighted price or an own-bid-dependent share each hand some deviation
    a strictly better payoff.
    """
    auctioneer = _make_auction(num_experts=5, top_k=2)
    auctioneer.eval()

    winning_outcomes = set()
    for true_value in torch.linspace(0.05, 0.95, 10).tolist():
        truthful = _expert_zero_utility(auctioneer, true_value, true_value)
        winning_outcomes.add(true_value > _UTILITY_CRITICAL_VALUE)

        assert truthful >= -PAYMENT_TOLERANCE, "truthful reporting must never lose money"

        for report in torch.linspace(0.0, 1.0, 41).tolist():
            deviation = _expert_zero_utility(auctioneer, report, true_value)
            assert deviation <= truthful + PAYMENT_TOLERANCE, (
                f"misreporting {report:.3f} beat truthful {true_value:.3f}: "
                f"{deviation:.6f} > {truthful:.6f}"
            )

    assert winning_outcomes == {True, False}, "sweep must straddle the critical value"


def test_proportional_baseline_rewards_overreporting():
    """Negative control for the deviation sweep above.

    The same utility, the same fixture, the same truthful report -- but with the
    own-bid-weighted gate restored there is a strictly profitable lie. Asserting the
    baseline *fails* the property is what stops the test above from passing for
    reasons unrelated to the mechanism.
    """
    auctioneer = _make_auction(num_experts=5, top_k=2, routing_share="proportional")
    auctioneer.eval()

    true_value = 0.5
    truthful = _expert_zero_utility(auctioneer, true_value, true_value)
    best_lie = max(
        _expert_zero_utility(auctioneer, report, true_value)
        for report in torch.linspace(0.0, 1.0, 41).tolist()
    )

    assert best_lie > truthful + PAYMENT_TOLERANCE
