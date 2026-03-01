import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VCGAuctioneer(nn.Module):
    """
    VCG (Vickrey-Clarke-Groves) auction mechanism for expert routing.

    Implements second-price auction that incentivizes truthful bidding.
    Supports differentiable routing via straight-through estimator.
    """

    def __init__(self, num_experts: int, top_k: int = 2, differentiable: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.differentiable = differentiable

    def forward(
        self,
        confidences: torch.Tensor,
        wealth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the VCG auction.

        Args:
            confidences: Expert confidence scores (batch, seq_len, num_experts)
            wealth: Current wealth of each expert (num_experts,)

        Returns:
            selected_experts: Indices of winning experts (batch, seq_len, top_k)
            routing_weights: Normalized weights for selected experts (batch, seq_len, top_k)
            payments: What each winning expert pays (batch, seq_len, top_k)
        """
        bids = confidences * wealth.unsqueeze(0).unsqueeze(0)

        top_bids, selected_experts = torch.topk(bids, self.top_k, dim=-1)

        if self.top_k < self.num_experts:
            all_sorted, _ = torch.sort(bids, dim=-1, descending=True)
            payments = all_sorted[:, :, self.top_k : self.top_k + 1].expand(
                -1, -1, self.top_k
            )
        else:
            payments = torch.zeros_like(top_bids)

        if self.differentiable and self.training:
            soft_weights = F.softmax(bids, dim=-1)

            hard_mask = torch.zeros_like(bids)
            hard_mask.scatter_(-1, selected_experts, 1.0)

            differentiable_mask = hard_mask + (soft_weights - soft_weights.detach())

            routing_weights_full = differentiable_mask * F.softmax(bids, dim=-1)
            routing_weights = torch.gather(routing_weights_full, -1, selected_experts)
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-8
            )
        else:
            routing_weights = F.softmax(top_bids, dim=-1)

        return selected_experts, routing_weights, payments
