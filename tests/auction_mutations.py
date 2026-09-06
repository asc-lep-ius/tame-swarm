"""Broken auctions, for pairing every mechanism property with the defect it catches.

Each mutant reproduces one way the auction has been, or could be, wrong, in the
shape ``monkeypatch.setattr(VCGAuctioneer, ...)`` expects. A property test that
passes on the real auction and fails under the mutant that breaks it is a test;
one that passes under both is documentation with a green checkmark.
"""

import math

import torch

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
