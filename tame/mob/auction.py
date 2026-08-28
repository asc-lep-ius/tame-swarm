import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative headroom for float error when differencing two welfare sums. Payments
# are always accumulated in float32, so this does not vary with the model dtype.
PAYMENT_NEGATIVITY_TOLERANCE = 1e-5

# Guards the division that converts a weighted-welfare externality back into the
# winner's own value units. An expert at zero wealth bids zero, so it can only win
# when the displaced bid is also zero -- the numerator vanishes with the
# denominator and the clamp returns a price of zero rather than an infinity.
#
# Negative wealth never reaches this clamp. Reports are non-negative, so a negative
# wealth means a negative bid, which can only win when every bid is negative -- and
# that case trips the payment-negativity assert below first. MoBConfig rejects
# non-positive wealth bounds at the boundary, which is where the real guard lives;
# an assert here would be unreachable.
WEALTH_EPSILON = 1e-12


class AuctionOutcome(NamedTuple):
    """What one token's auction produced.

    ``rebates`` is per *expert*, not per winner slot: every expert is rebated,
    including the ones that lost, which is what keeps the rebate independent of a
    winner's own report.
    """

    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    payments: torch.Tensor
    rebates: torch.Tensor


ROUTING_SHARE_UNIFORM = "uniform"
ROUTING_SHARE_SOFTMAX = "softmax"
SUPPORTED_ROUTING_SHARES = frozenset({ROUTING_SHARE_UNIFORM, ROUTING_SHARE_SOFTMAX})


class VCGAuctioneer(nn.Module):
    """Top-*k* unit-slot auction over reported confidence, weighted by wealth.

    Under ``routing_share="uniform"`` the mechanism is strategyproof in the
    per-token stage game: the allocation is monotone in an expert's own report,
    every winner is charged its critical value, and a winner's share of the output
    does not depend on what it reported. ``"softmax"`` restores the own-bid-weighted
    gate the auction previously used -- retained as the gate-swap baseline, and
    *not* incentive compatible, because a winner can enlarge its own share by
    overreporting while its price stays fixed.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        differentiable: bool = True,
        routing_share: str = ROUTING_SHARE_UNIFORM,
    ):
        super().__init__()
        if routing_share not in SUPPORTED_ROUTING_SHARES:
            shares = ", ".join(sorted(SUPPORTED_ROUTING_SHARES))
            raise ValueError(f"Unsupported routing share '{routing_share}'. Supported: {shares}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.differentiable = differentiable
        self.routing_share = routing_share

    def forward(
        self,
        confidences: torch.Tensor,
        wealth: torch.Tensor,
    ) -> "AuctionOutcome":
        wealth_snapshot = wealth.detach().clone()
        bids = confidences * wealth_snapshot.unsqueeze(0).unsqueeze(0)
        top_bids, selected_experts = torch.topk(bids, self.top_k, dim=-1)
        payments = self._compute_vcg_payments(bids, selected_experts, wealth_snapshot)
        rebates = self._compute_rebates(bids, wealth_snapshot)
        routing_weights = self._compute_routing_weights(bids, top_bids, selected_experts)

        return AuctionOutcome(selected_experts, routing_weights, payments, rebates)

    def _compute_rebates(self, bids: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
        """Return most of the collected payment without touching anyone's incentives.

        VCG prices have no recipient here. Burned, they make the expert economy an
        open system: with payments correctly scaled the outflow dwarfs the reward
        inflow and every expert converges on ``min_wealth``.

        Handing the money back is not free -- Green-Laffont says budget balance,
        strategyproofness and efficiency cannot hold together, and a naive even split
        would make an expert's rebate depend on whether it won. This is the Cavallo
        (2006) / Guo-Conitzer redistribution instead: expert *i* is rebated from the
        (k+1)-th highest bid **among the others**, a quantity *i* cannot move at all,
        so every threshold stays exactly where the payment rule put it.

        Excluding *i* lifts the ranking by one place precisely when *i* sits in the
        top k+1, which is what the two branches below are. The rebate is always
        affordable: it sums to at most the collected total, with equality only when
        the (k+1)-th and (k+2)-th bids coincide. Whatever is left over is the
        unavoidable residual of budget balance, not a leak.
        """
        batch, seq_len, _ = bids.shape
        k, n = self.top_k, self.num_experts

        if k >= n or n < k + 2:
            return torch.zeros(batch, seq_len, n, device=bids.device, dtype=bids.dtype)

        accumulate_dtype = torch.promote_types(bids.dtype, torch.float32)
        ranked, ranked_idx = torch.topk(bids.to(accumulate_dtype), k + 2, dim=-1)

        displaced = ranked[..., k].unsqueeze(-1)  # (k+1)-th highest overall
        displaced_without_self = ranked[..., k + 1].unsqueeze(-1)  # (k+2)-th

        # Removing i only shifts the ranking for the k+1 experts above the cut.
        in_top_k_plus_one = torch.zeros_like(bids, dtype=torch.bool)
        in_top_k_plus_one.scatter_(-1, ranked_idx[..., : k + 1], True)

        reference = torch.where(in_top_k_plus_one, displaced_without_self, displaced)
        rebates = (k / n) * reference / wealth.to(accumulate_dtype).clamp_min(WEALTH_EPSILON)

        return rebates.to(bids.dtype)

    def _compute_routing_weights(
        self,
        bids: torch.Tensor,
        top_bids: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Split the output across winners without reading their own reports.

        Incentive compatibility constrains the *share*, not just the price. A
        softmax over own bids makes the allocation continuous and strictly
        increasing in a winner's own report while the VCG price is by construction
        independent of it, so overreporting buys output influence for free. A flat
        1/k split is the allocation the top-*k* unit-slot theorem is stated over.

        ``full_like`` also severs the gradient from the language-modelling loss back
        into the confidence heads: under this share the heads are trained only by
        their own value objective, which is what makes them agents rather than
        slices of a central planner.
        """
        if self.routing_share == ROUTING_SHARE_UNIFORM:
            return torch.full_like(top_bids, 1.0 / self.top_k)

        if self.differentiable and self.training:
            return self._differentiable_routing(bids, selected_experts)
        return F.softmax(top_bids, dim=-1)

    def _compute_vcg_payments(
        self, bids: torch.Tensor, selected_experts: torch.Tensor, wealth: torch.Tensor
    ) -> torch.Tensor:
        """Charge each winner the externality it imposes, in its own value units.

        Removing winner *j* leaves all *k* slots open, so the counterfactual
        allocation is the top *k* of the remaining bids -- not the top *k-1*. With
        *k-1* the exclusion set is exactly the other winners, the difference is
        identically zero, and the mechanism has no price at all.

        The allocation maximises ``confidence x wealth``, a *weighted* welfare, so
        the raw welfare difference is denominated in bid units rather than in the
        units an expert reports. Weighted VCG is truthful only once each winner's
        externality is divided by its own weight, which is what converts the price
        into the same currency as the report::

            p_j = b_(k+1) / w_j

        That is exactly *j*'s critical value: *j* wins iff ``c_j * w_j > b_(k+1)``,
        i.e. iff ``c_j > p_j``. A monotone allocation paid at its critical value is
        strategyproof, so misreporting confidence cannot raise *j*'s utility.

        Under this top-*k* unit-slot rule the numerator collapses to the (k+1)-th
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

        winner_wealth = wealth.to(payments.dtype)[selected_experts]

        # Checked on the weighted welfare difference, before the division rescales
        # it: that is the quantity the accounting produces, and the tolerance below
        # is stated relative to the bid magnitudes it is differencing.
        if __debug__:
            self._assert_payments_well_formed(payments, bids)

        payments = payments / winner_wealth.clamp_min(WEALTH_EPSILON)

        return payments.to(out_dtype)

    @staticmethod
    def _assert_payments_well_formed(payments: torch.Tensor, bids: torch.Tensor) -> None:
        """Fail loudly on a broken mechanism invariant rather than repairing it.

        Non-negativity holds whenever bids do (softplus report x positive wealth).
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
        # WEALTH_EPSILON exists to keep a zero-wealth winner from dividing by zero,
        # not to absorb a negative one: clamping a negative wealth up to 1e-12 turns
        # a valid numerator into an enormous positive price with nothing complaining.

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
