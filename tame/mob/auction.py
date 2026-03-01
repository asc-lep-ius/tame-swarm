import torch
import torch.nn as nn
import torch.nn.functional as F


class VCGAuctioneer(nn.Module):
    def __init__(self, num_experts: int, top_k: int = 2, differentiable: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.differentiable = differentiable

    def forward(
        self,
        confidences: torch.Tensor,
        wealth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bids = confidences * wealth.unsqueeze(0).unsqueeze(0)
        top_bids, selected_experts = torch.topk(bids, self.top_k, dim=-1)
        payments = self._compute_vcg_payments(bids, selected_experts)

        if self.differentiable and self.training:
            routing_weights = self._differentiable_routing(bids, selected_experts)
        else:
            routing_weights = F.softmax(top_bids, dim=-1)

        return selected_experts, routing_weights, payments

    def _compute_vcg_payments(
        self, bids: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, _ = bids.shape
        k = self.top_k

        if k >= self.num_experts:
            return torch.zeros(batch, seq_len, k, device=bids.device, dtype=bids.dtype)

        winner_bids = torch.gather(bids, -1, selected_experts)
        other_winner_welfare = winner_bids.sum(dim=-1, keepdim=True) - winner_bids

        payments = torch.zeros(batch, seq_len, k, device=bids.device, dtype=bids.dtype)
        for j in range(k):
            winner_j_idx = selected_experts[:, :, j : j + 1]
            masked_bids = bids.scatter(
                -1, winner_j_idx, torch.full_like(winner_j_idx, float("-inf"), dtype=bids.dtype)
            )
            top_without_j, _ = torch.topk(masked_bids, k - 1, dim=-1) if k > 1 else (
                torch.zeros(batch, seq_len, 0, device=bids.device), None
            )
            welfare_without_j = top_without_j.sum(dim=-1)
            payments[:, :, j] = (welfare_without_j - other_winner_welfare[:, :, j]).clamp(min=0)

        return payments

    def _differentiable_routing(
        self, bids: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        soft_weights = F.softmax(bids, dim=-1)

        hard_mask = torch.zeros_like(bids)
        hard_mask.scatter_(-1, selected_experts, 1.0)

        differentiable_mask = hard_mask + (soft_weights - soft_weights.detach())

        routing_weights_full = differentiable_mask * F.softmax(bids, dim=-1)
        routing_weights = torch.gather(routing_weights_full, -1, selected_experts)
        return routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
