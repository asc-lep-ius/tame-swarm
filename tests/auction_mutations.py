"""Broken auctions, for pairing every mechanism property with the defect it catches.

Each mutant reproduces one way the auction has been, or could be, wrong. A
property test that passes on the real auction and fails under the mutant that
breaks it is a test; one that passes under both is documentation with a green
checkmark.

Mutants come in two shapes. Most are methods, in the form
``monkeypatch.setattr(VCGAuctioneer, ...)`` expects. The band mutants at the
bottom are configs instead, because what the wealth bounds do cannot be removed
by patching a method -- it is the band itself that is the mechanism. #16 added
them after running the same experiment by hand over the whole suite and finding
that lowering ``min_wealth`` by 15000x broke nothing: the floor was inert with
respect to every behavioural assertion in the repository.
"""

import math
from dataclasses import replace

import torch

from mob import MoBConfig
from mob.auction import AuctionOutcome


def pre_nine_payments(self, bids, selected_experts, wealth):
    """The defect #9 fixed: an exclusion set of ``k - 1``, and the clamp that hid it."""
    batch, seq_len, _ = bids.shape
    k = self.top_k
    winner_bids = torch.gather(bids, -1, selected_experts)
    other_winner_welfare = winner_bids.sum(dim=-1, keepdim=True) - winner_bids
    payments = torch.zeros(batch, seq_len, k, device=bids.device, dtype=bids.dtype)
    for j in range(k):
        winner_j = selected_experts[:, :, j : j + 1]
        masked = bids.scatter(-1, winner_j, torch.full_like(winner_j, -math.inf, dtype=bids.dtype))
        top_without_j = torch.topk(masked, k - 1, dim=-1).values
        payments[:, :, j] = top_without_j.sum(dim=-1) - other_winner_welfare[:, :, j]
    return payments.clamp(min=0) / wealth[selected_experts]


def undivided_payments(original):
    """The #10 defect: the externality in bid units, never restated in the winner's own."""

    def mutant(self, bids, selected_experts, wealth):
        return original(self, bids, selected_experts, wealth) * wealth[selected_experts]

    return mutant


def first_price_payments(self, bids, selected_experts, wealth):
    """Pay your own bid: the rule whose price does move with the report."""
    return torch.gather(bids, -1, selected_experts) / wealth[selected_experts]


def wealth_blind_forward(self, confidences, wealth):
    """Allocate on reports alone, as if every expert were equally rich."""
    _, selected = torch.topk(confidences, self.top_k, dim=-1)
    weights = torch.full_like(confidences[..., : self.top_k], 1.0 / self.top_k)
    return AuctionOutcome(
        selected, weights, torch.zeros_like(weights), torch.zeros_like(confidences), None
    )


# --- Band mutants ------------------------------------------------------------------


def lowered_floor(config: MoBConfig, factor: float = 1000.0) -> MoBConfig:
    """The band with its floor dropped, which is a mutation of the *ratio*.

    Selection is ``argtopk(confidence x wealth)``, so ``max_wealth / min_wealth``
    is the report advantage the market demands of its poorest expert, and dropping
    the floor raises that demand. It does not leave the price division unguarded:
    at the default factor the floor is still ten orders of magnitude above
    ``auction.WEALTH_EPSILON``, so the guard is weakened, not removed.

    #16 measured that the floor's *height* is inert either way -- raised 5x, so
    that a ruined expert is restored to full ``initial_wealth`` on the next clamp,
    its win share stays at 0.0012-0.0019, a hundredfold below chance. Use this to
    move the ratio, not to claim anything about recovery.
    """
    return replace(config, min_wealth=config.min_wealth / factor)


def flat_band(config: MoBConfig) -> MoBConfig:
    """Every expert equally rich, so the ledger decides nothing.

    ``min_wealth == max_wealth == initial_wealth`` pins wealth at a constant, and a
    constant multiplier cannot reorder ``confidence x wealth``: selection becomes
    the report ranking alone.

    **This removes the ledger, not merely its ratio.** Prices and rebates are still
    computed -- unlike ``wealth_blind_forward``, which returns zeros for both -- but
    the clamp restores every wealth to the same constant on the very next
    settlement, so every credit collected is annihilated and none of it can reach a
    later allocation. Read it as "the economy is switched off", not as "the band is
    narrower". #16 used it to establish that a cause other than the ledger is
    *sufficient* to stop a market re-forming after a forced episode; it cannot
    apportion the damage between the two, because pinning the ledger also moves the
    undamaged steady state it would be measured against.

    Under the **ruin** protocol it is degenerate outright: ``ruin()`` zeroes an
    expert's wealth and the clamp restores it to the constant before anything reads
    it, so the damage never happens and the resulting win share is not a recovery.
    """
    return replace(
        config,
        min_wealth=config.initial_wealth,
        max_wealth=config.initial_wealth,
    )
