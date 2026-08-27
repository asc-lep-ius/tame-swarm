import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative headroom for float error when differencing two welfare sums. Payments
# are always accumulated in float32, so this does not vary with the model dtype.
PAYMENT_NEGATIVITY_TOLERANCE = 1e-5


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
        wealth_snapshot = wealth.detach().clone()
        bids = confidences * wealth_snapshot.unsqueeze(0).unsqueeze(0)
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
        """Charge each winner the externality it imposes on the experts it displaced.

        Removing winner *j* leaves all *k* slots open, so the counterfactual
        allocation is the top *k* of the remaining bids -- not the top *k-1*. With
        *k-1* the exclusion set is exactly the other winners, the difference is
        identically zero, and the mechanism has no price at all.

        Under this top-*k* unit-slot rule the result collapses to the (k+1)-th
        highest bid for every winner, but the welfare difference is kept in its
        general form: it is the VCG definition, and it stays correct if the
        allocation rule ever stops being a plain top-*k*.
        """
        batch, seq_len, _ = bids.shape
        k = self.top_k
        out_dtype = bids.dtype

        if k >= self.num_experts:
            return torch.zeros(batch, seq_len, k, device=bids.device, dtype=out_dtype)

        # The price is a difference of two welfare sums of magnitude k * max_bid,
        # while the result is only b_(k+1). bfloat16 is the default training dtype
        # and expert_wealth is converted to it, so in production that cancellation
        # costs far more than the price is worth: a true price of 0.08 quantises to
        # 0.0, silently restoring the zero payments this function was fixed to stop.
        # Accumulate in float32 and cast back at the boundary.
        bids = bids.to(torch.promote_types(bids.dtype, torch.float32))

        winner_bids = torch.gather(bids, -1, selected_experts)
        other_winner_welfare = winner_bids.sum(dim=-1, keepdim=True) - winner_bids

        payments = torch.zeros(batch, seq_len, k, device=bids.device, dtype=bids.dtype)
        for j in range(k):
            winner_j_idx = selected_experts[:, :, j : j + 1]
            masked_bids = bids.scatter(
                -1, winner_j_idx, torch.full_like(winner_j_idx, float("-inf"), dtype=bids.dtype)
            )
            top_without_j, _ = torch.topk(masked_bids, k, dim=-1)
            payments[:, :, j] = top_without_j.sum(dim=-1) - other_winner_welfare[:, :, j]

        if __debug__:
            self._assert_payments_well_formed(payments, bids)

        return payments.to(out_dtype)

    @staticmethod
    def _assert_payments_well_formed(payments: torch.Tensor, bids: torch.Tensor) -> None:
        """Fail loudly on a broken mechanism invariant rather than repairing it.

        Non-negativity holds whenever bids do (sigmoid confidence x positive wealth).
        A clamp here is what let the identically-zero payments this function was
        fixed to stop pass as plausible. Finiteness is reported separately so that
        NaN bids are not surfaced as a negative price.

        Both checks read from a single device-host sync. The ``if __debug__`` at the
        call site is what keeps that sync out of tuned runs: once the asserts live
        in a method, the call itself is not an assert and -O would otherwise strip
        the checks while still paying for the sync.
        """
        low, high, max_bid = torch.stack(
            [payments.detach().min(), payments.detach().max(), bids.detach().abs().amax()]
        ).tolist()

        assert math.isfinite(low) and math.isfinite(high), (
            "VCG payment is not finite; the bid vector contains NaN or inf"
        )
        tolerance = PAYMENT_NEGATIVITY_TOLERANCE * max(max_bid, 1.0)
        assert low >= -tolerance, (
            f"VCG payment is negative ({low:.3e}); the auction's welfare accounting is inconsistent"
        )

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
