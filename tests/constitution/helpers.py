"""Fixtures the constitution tests and the auction tests share."""

import torch

from mob import VCGAuctioneer

PAYMENT_TOLERANCE = 1e-5


def _make_auction(num_experts=4, top_k=2, differentiable=True, routing_share="uniform"):
    return VCGAuctioneer(num_experts, top_k, differentiable, routing_share=routing_share)


def _bids(confidences: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
    """The auction's bid rule, restated here so tests never import it."""
    return confidences * wealth.unsqueeze(0).unsqueeze(0)


# Wealth and rival reports chosen so expert 0's threshold lands mid-sweep. Rival
# bids are 2*0.8=1.6, 3*0.5=1.5, 1*0.9=0.9 and 5*0.28=1.4; with two slots, expert 0
# enters the allocation once 4*c_0 clears the second-highest rival bid of 1.5, so
# its critical value is 0.375 and the sweep below straddles it in both directions.
_UTILITY_WEALTH = torch.tensor([4.0, 2.0, 3.0, 1.0, 5.0])
_UTILITY_FIELD = torch.tensor([0.0, 0.8, 0.5, 0.9, 0.28])
_UTILITY_CRITICAL_VALUE = 0.375


def _expert_zero_utility(auctioneer, report: float, true_value: float) -> float:
    """Quasi-linear payoff for expert 0: what it banks, less what it is charged.

    Influence is the winner's share renormalised by an equal split, so it is
    identically 1.0 under the uniform rule and the expression reduces to the
    textbook ``v * 1[win] - p``. It is written this way so the same utility is
    defined for the proportional baseline, where a winner's slice does move with its own
    report and the deviation test below must be able to see that.
    """
    confidences = _UTILITY_FIELD.clone()
    confidences[0] = report
    selected, weights, payments, _, _ = auctioneer(confidences.view(1, 1, -1), _UTILITY_WEALTH)

    slots = (selected[0, 0] == 0).nonzero()
    if slots.numel() == 0:
        return 0.0

    slot = slots[0, 0]
    influence = weights[0, 0, slot].item() * auctioneer.top_k
    return true_value * influence - payments[0, 0, slot].item()
