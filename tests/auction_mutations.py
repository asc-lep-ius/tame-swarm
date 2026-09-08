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
    """The band with its floor dropped, leaving the price division unguarded.

    What ``min_wealth`` buys is a bound on the report advantage the market can
    demand of its poorest expert: selection is ``argtopk(confidence x wealth)``,
    so an expert at the floor needs a report ``max_wealth / min_wealth`` times the
    richest expert's to win at all. Dropping the floor raises that demand without
    touching anything else, which is the one way to make a recovering expert
    unable to climb back on merit.
    """
    return replace(config, min_wealth=config.min_wealth / factor)


def flat_band(config: MoBConfig) -> MoBConfig:
    """Every expert equally rich, so the ledger decides nothing and the auction still prices.

    ``min_wealth == max_wealth == initial_wealth`` pins wealth at a constant, and a
    constant multiplier cannot reorder ``confidence x wealth``: selection becomes
    the report ranking alone. It is the wealth-blind arm without
    ``wealth_blind_forward``'s side effect of zeroing every payment and rebate too,
    which matters when the question is what the *band* costs rather than what the
    whole auction does. #16 used it to show that the band is not what stops a
    market re-forming after a forced episode.
    """
    return replace(
        config,
        min_wealth=config.initial_wealth,
        max_wealth=config.initial_wealth,
    )
