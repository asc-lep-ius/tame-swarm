"""Budget-balanced rebate the recipient cannot influence (#38's second property)."""

import pytest
import torch

from .helpers import PAYMENT_TOLERANCE, _bids, _make_auction


def test_rebate_is_independent_of_the_recipients_own_bid():
    """The property the whole redistribution rests on.

    Cavallo pays expert i out of a quantity computed from everyone *but* i, so no
    report an expert can make moves the money it gets back. A rebate that did depend
    on it — an even split of the collected pot, say — would shift that expert's
    threshold away from its price, which is the Green–Laffont trade this rule exists
    to avoid.
    """
    auctioneer = _make_auction(num_experts=6, top_k=2)
    auctioneer.eval()

    torch.manual_seed(31)
    wealth = torch.rand(6) * 8.0 + 1.0
    confidences = torch.rand(1, 1, 6)

    baseline = auctioneer(confidences, wealth).rebates[0, 0, 0].item()

    for own_bid in torch.linspace(0.0, 1.0, 21).tolist():
        perturbed = confidences.clone()
        perturbed[0, 0, 0] = own_bid
        rebate = auctioneer(perturbed, wealth).rebates[0, 0, 0].item()
        assert rebate == pytest.approx(baseline, abs=PAYMENT_TOLERANCE), (
            f"reporting {own_bid:.2f} moved expert 0's own rebate"
        )


def test_rebate_never_exceeds_what_the_auction_collected():
    """Budget feasibility, in the currency the wealth ledger actually uses.

    Both sides are the per-expert quantities the wealth update consumes: payments
    already divided by each winner's own wealth, rebates divided by the harmonic
    mean of the k richest. Checking this in bid units instead — multiplying wealth
    back in — tests an inequality that holds even when the ledger's does not, which
    is exactly how a rebate that over-paid by 7.4x passed a feasibility test.
    """
    torch.manual_seed(37)
    for num_experts, top_k in ((6, 2), (8, 3), (5, 1), (4, 2)):
        auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
        auctioneer.eval()

        # Spanning the configured min_wealth..max_wealth band, not a narrow
        # random range: a per-recipient divisor only over-rebates once the
        # spread is wide, so a tight fixture cannot see it.
        wealth = torch.linspace(15.0, 750.0, num_experts)
        confidences = torch.rand(2, 6, num_experts)
        outcome = auctioneer(confidences, wealth)

        collected = outcome.payments.sum(dim=-1)
        returned = outcome.rebates.sum(dim=-1)

        assert (returned <= collected + PAYMENT_TOLERANCE).all(), (
            f"n={num_experts} k={top_k}: rebate exceeds revenue in credits"
        )
        assert (returned > 0).any(), "fixture returns nothing; feasibility is vacuous"

        # The classical Cavallo bound, which does not depend on the divisor: every
        # reference is at most b_(k+1), so sum_i (k/n) * ref_i <= k * b_(k+1).
        # Measured slack, credit vs bid-unit: 16.6/2.5, 24.7/2.4, 2.0/2.0, 72.4/61.6.
        # So this bound is the tight one wherever k >= 2 and n > k + 2 -- a 3%
        # reference inflation trips it while the credit assertion sleeps through.
        # All four fixtures share one wealth spread, so the k/n relation is what
        # separates them, not the spread. On n=5 k=1 the two coincide, because a lone winner
        # collapses the harmonic mean onto its own wealth and the tight token is
        # one the richest expert takes; on n=4 k=2 both are slack, because
        # k + 2 == n leaves the reference at the bottom of the bid vector.
        richest = torch.topk(wealth, top_k).values
        payout_in_bid_units = (outcome.rebates * (top_k / (1.0 / richest).sum())).sum(dim=-1)
        displaced = torch.sort(_bids(confidences, wealth), dim=-1, descending=True)[0][..., top_k]
        # Relative, not absolute: these are bid-unit quantities of order 100, where
        # a 1e-5 absolute tolerance is really no tolerance at all.
        bound = top_k * displaced
        assert (payout_in_bid_units <= bound * (1 + PAYMENT_TOLERANCE)).all(), (
            f"n={num_experts} k={top_k}: exclusion rule returned too large a reference"
        )
