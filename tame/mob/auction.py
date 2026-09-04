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
# Negative wealth never reaches this clamp in production, though not because it
# cannot win: with mixed signs a negative bid can still place in the top k. A
# negative-wealth winner sits in the top k, so b_(k+1) is at most its own negative
# bid, and the payment-negativity assert below fires on the numerator -- unless
# b_(k+1) itself is within the assert's tolerance of zero. That tolerance is relative to the
# largest bid, so at the configured max_wealth the assert stays silent down to a
# wealth of about -6e-3, and it is the middle of that window that hurts: -1e-7
# underflows to a price of exactly zero, while -1e-4 divides through the clamp to
# about -1.2e8.
# MoBConfig rejects non-positive bounds, an inverted band, and an initial_wealth
# that is non-positive or out of band -- that last pair is what makes the unclamped
# constructor write in core.py safe -- and the three update paths and load_mob_state
# clamp. No writer can produce it; the boundary is where the real guard lives, and
# an assert here would be unreachable.
WEALTH_EPSILON = 1e-12


class AuctionOutcome(NamedTuple):
    """What a gate produced, per token.

    ``rebates`` is per *expert*, not per winner slot: every expert is rebated,
    including the ones that lost, which is what keeps the rebate independent of a
    winner's own report.

    ``payments`` and ``rebates`` are optional because the #12 control arm
    (:class:`~mob.softmax_router.SoftmaxRouter`) runs no auction. ``None`` there
    says *no transfer was computed*, which the wealth paths must not confuse with
    an auction that computed zeros -- the defect #9 fixed looked exactly like the
    latter.

    ``explored`` marks, per winner slot, the slots the auction handed out rather
    than sold -- see ``VCGAuctioneer.exploration_rate``. ``None`` means no slot
    was, which is every gate but a training auction with exploration switched on.
    """

    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    payments: torch.Tensor | None
    rebates: torch.Tensor | None
    explored: torch.Tensor | None = None


ROUTING_SHARE_UNIFORM = "uniform"
ROUTING_SHARE_PROPORTIONAL = "proportional"
SUPPORTED_ROUTING_SHARES = frozenset({ROUTING_SHARE_UNIFORM, ROUTING_SHARE_PROPORTIONAL})

# Floor under a bid before it enters the log domain. A production bid cannot reach
# it: the confidence logit is clamped at -20, so softplus bottoms out near 2.1e-9,
# and MoBConfig rejects a non-positive min_wealth. It exists so that a bid of
# exactly zero -- reachable only in a test that feeds one, or under a dtype whose
# softplus underflows -- becomes a vanishing share rather than a NaN.
BID_LOG_FLOOR = 1e-30

# Floor under a routing weight before its log enters the entropy below. A weight
# can legitimately reach zero -- a floored bid earns no share -- and 0*log(0) is 0
# in the limit, which this recovers without a special case. Distinct from
# BID_LOG_FLOOR: that one floors a *bid* on its way into the gate, this one floors
# a *probability* on its way into a diagnostic.
ENTROPY_PROBABILITY_FLOOR = 1e-30

# A token routed through a gate this sharp has spent top_k experts' compute and
# used one. Reported as a fraction rather than folded into the mean, because a
# distribution with 90% of its mass at 1.0 and a long left tail has an
# unremarkable mean.
ROUTING_SATURATION_THRESHOLD = 0.99


class RoutingDiagnostics(NamedTuple):
    """What the gate actually did to a batch, as opposed to what top_k configures.

    Every field is a 0-dim tensor rather than a float. Reducing to a Python scalar
    forces a device sync, and these are computed on every forward of every MoB
    layer while a training step wants at most one sync at the end of it.

    ``effective_experts`` is ``exp(entropy(routing_weights))`` averaged over tokens:
    the number of experts the output was genuinely mixed from. It equals ``top_k``
    exactly under the uniform share and falls to 1.0 when the gate has collapsed
    onto a single winner, which is the failure ``top_k`` alone cannot show.
    """

    top1_mean: torch.Tensor
    top1_median: torch.Tensor
    top1_saturated_fraction: torch.Tensor
    effective_experts: torch.Tensor


def routing_diagnostics(routing_weights: torch.Tensor) -> RoutingDiagnostics:
    """Summarise a batch of routing weights without leaving the device.

    ``routing_weights`` is ``(batch, seq, top_k)`` and normalised over its last
    dimension, so the entropy below is taken over the winners only -- the experts
    that lost contribute no compute and are not part of the mixture being measured.
    """
    weights = routing_weights.detach().float()
    top1 = weights.amax(dim=-1).reshape(-1)

    safe = weights.clamp_min(ENTROPY_PROBABILITY_FLOOR)
    entropy = -(safe * safe.log()).sum(dim=-1)

    return RoutingDiagnostics(
        top1_mean=top1.mean(),
        top1_median=top1.median(),
        top1_saturated_fraction=(top1 > ROUTING_SATURATION_THRESHOLD).float().mean(),
        effective_experts=entropy.exp().mean(),
    )


class VCGAuctioneer(nn.Module):
    """Top-*k* unit-slot auction over reported confidence, weighted by wealth.

    Under ``routing_share="uniform"`` the mechanism is strategyproof in the
    per-token stage game: the allocation is monotone in an expert's own report,
    every winner is charged its critical value, and a winner's share of the output
    does not depend on what it reported. ``"proportional"`` restores an
    own-bid-weighted gate as the gate-swap baseline -- *not* incentive compatible,
    because a winner can enlarge its own share by overreporting while its price
    stays fixed, which is exactly the property that baseline exists to isolate.

    ``temperature`` sharpens or flattens that baseline gate. It is applied in the
    log domain, so a winner's share is ``bid ** (1 / temperature)`` normalised over
    the winners: 1.0 is plain bid-proportional, below 1.0 approaches argmax, above
    1.0 approaches the uniform split. The uniform share ignores it.

    ``exploration_rate`` is the fraction of tokens on which, in training, one
    slot -- drawn uniformly over the *k* -- is handed to a uniformly random loser
    instead of sold. A head is trained only on the value its expert realises on
    the tokens it holds, so an expert that has fallen to a truthful report of zero
    would otherwise never hold another token, never see another target, and never
    come back however much its adapter later learns -- measured on the
    planted-competence fixture as a market that collapsed to two of eight experts
    with the other six at the wealth floor. The slot is a gift from the tissue
    rather than a trade: the explorer pays nothing, and the token's rebate is
    scaled down so the gift is funded from the payments that remain (see
    ``_compute_rebates``).

    What this does to the incentive claim, exactly. Whether a token is explored,
    and which slot, is drawn before any report is read; which loser receives it is
    uniform over the losers. A loser therefore cannot raise its chance of the
    gift by any report, and a winner can only reach the lottery by giving up its
    win. Drawing the slot uniformly is what removes the deviation a fixed last
    slot would create -- a marginal winner overreporting into a slot that is never
    displaced, at an unchanged price. What remains is that a winner faces a
    ``rate / k`` chance of displacement it cannot bid away, and a loser an
    expected ``rate / (n - k)`` share of the gift: the stage game is strategyproof
    up to ``O(exploration_rate)``, with any deviation worth at most
    ``exploration_rate x value`` to the deviator, and exactly strategyproof at a
    rate of zero. ``test_deviation_gain_is_bounded_by_the_exploration_rate``
    pins the bound. The bid stays the truthful value estimate; the noise that
    keeps every cell sampling its environment lives here, in the allocation.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        differentiable: bool = True,
        routing_share: str = ROUTING_SHARE_UNIFORM,
        temperature: float = 1.0,
        exploration_rate: float = 0.0,
    ):
        super().__init__()
        if routing_share not in SUPPORTED_ROUTING_SHARES:
            shares = ", ".join(sorted(SUPPORTED_ROUTING_SHARES))
            raise ValueError(f"Unsupported routing share '{routing_share}'. Supported: {shares}")
        # A non-positive temperature is not a sharper gate; it is a division by zero
        # or a sign flip that ranks the abstaining expert first.
        if temperature <= 0:
            raise ValueError(f"Routing temperature must be positive, got {temperature}")
        # 1.0 would hand every last slot to chance and leave nothing for the
        # auction to decide there.
        if not 0.0 <= exploration_rate < 1.0:
            raise ValueError(f"exploration_rate must lie in [0, 1), got {exploration_rate}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.differentiable = differentiable
        self.routing_share = routing_share
        self.temperature = temperature
        self.exploration_rate = exploration_rate

    def forward(
        self,
        confidences: torch.Tensor,
        wealth: torch.Tensor,
    ) -> "AuctionOutcome":
        wealth_snapshot = wealth.detach().clone()
        # The ledger is float32 whatever the model runs in; the bid takes the
        # report's dtype so the routing weights match the expert outputs they scale.
        bids = confidences * wealth_snapshot.to(confidences.dtype).unsqueeze(0).unsqueeze(0)
        top_bids, selected_experts = torch.topk(bids, self.top_k, dim=-1)
        payments = self._compute_vcg_payments(bids, selected_experts, wealth_snapshot)
        rebates = self._compute_rebates(bids, wealth_snapshot)

        explored: torch.Tensor | None = None
        if self.training and self.exploration_rate > 0.0:
            selected_experts, payments, explored = self._explore(bids, selected_experts, payments)
            rebates = self._fund_exploration(rebates, explored, wealth_snapshot)
            top_bids = torch.gather(bids, -1, selected_experts)

        routing_weights = self._compute_routing_weights(bids, top_bids, selected_experts)

        return AuctionOutcome(selected_experts, routing_weights, payments, rebates, explored)

    def _explore(
        self, bids: torch.Tensor, selected_experts: torch.Tensor, payments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hand one slot to a random loser on an ``exploration_rate`` fraction of tokens.

        Three draws, in this order and before any report is consulted: which
        tokens, which of the *k* slots, and -- masked to the experts that did not
        win -- which loser, as the argmax of independent uniforms. The displaced
        winner is not charged for a slot it no longer holds, and the explorer is
        not charged for one it did not bid for. The other winners' prices are the
        auction's prices: their externality was computed on the bids, and the bids
        have not changed.

        With ``top_k >= num_experts`` there is no loser to hand a slot to.
        """
        batch, seq_len, num_experts = bids.shape
        k = self.top_k
        if k >= num_experts:
            return selected_experts, payments, torch.zeros_like(selected_experts, dtype=torch.bool)

        explore_token = torch.rand(batch, seq_len, device=bids.device) < self.exploration_rate
        slot = torch.randint(0, k, (batch, seq_len), device=bids.device)
        explored = torch.zeros_like(selected_experts, dtype=torch.bool)
        explored.scatter_(-1, slot.unsqueeze(-1), explore_token.unsqueeze(-1))

        is_winner = torch.zeros_like(bids, dtype=torch.bool).scatter_(-1, selected_experts, True)
        draw = torch.rand(batch, seq_len, num_experts, device=bids.device).masked_fill(
            is_winner, -1.0
        )
        explorer = draw.argmax(dim=-1, keepdim=True).expand_as(selected_experts)

        selected_experts = torch.where(explored, explorer, selected_experts)
        payments = payments.masked_fill(explored, 0.0)
        return selected_experts, payments, explored

    def _fund_exploration(
        self, rebates: torch.Tensor, explored: torch.Tensor, wealth: torch.Tensor
    ) -> torch.Tensor:
        """Shrink the rebate on an explored token to what the remaining payments cover.

        An explored slot collects nothing, so the token's rebate -- computed
        against a full collection -- would otherwise return more than came in,
        and at a large enough rate the transfer would become a money pump.
        Scaling the row by ``sum_{richest k-1} 1/w / sum_{richest k} 1/w`` keeps
        the feasibility argument in ``_compute_rebates`` intact with one slot
        fewer: the payout is then at most ``b_(k+1) * sum_{richest k-1} 1/w_i``,
        and no ``k-1`` sold winners have a smaller sum of reciprocals. At
        ``top_k == 1`` the numerator is empty and the explored token rebates
        nothing, which is what a token with no collection can afford.

        The factor reads wealth and the exploration draw, neither of which any
        report can move, so the rebate stays independent of every recipient's own
        report.
        """
        k = self.top_k
        accumulate_dtype = torch.promote_types(rebates.dtype, torch.float32)
        reciprocals = 1.0 / torch.topk(wealth.to(accumulate_dtype), k).values.clamp_min(
            WEALTH_EPSILON
        )
        factor = reciprocals[: k - 1].sum() / reciprocals.sum()
        explored_token = explored.any(dim=-1, keepdim=True)
        return torch.where(explored_token, rebates * factor.to(rebates.dtype), rebates)

    def _compute_rebates(self, bids: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
        """Return part of the collected payment without touching anyone's incentives.

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
        top k+1, which is what the two branches below are.

        The divisor is the **harmonic mean of the k largest wealths**, not the
        recipient's own. Dividing by ``w_i`` is right for a *price* -- it restates
        an externality in the winner's own units, which is why
        ``_compute_vcg_payments`` does it -- and wrong for a lump-sum rebate: it
        pays the poorest expert the most, and feasibility then holds only in bid
        units rather than in the credits the wealth ledger is denominated in.
        Measured at the configured wealth band that over-rebates by up to 7.4x and
        turns the transfer into a money pump that flattens the very spread the
        economy exists to create.

        Against the harmonic mean ``H`` of the k richest it is affordable by
        construction, up to dtype epsilon. Every rebate reference is at most
        ``b_(k+1)``, so the payout is at most ``k * b_(k+1) / H``, which is
        ``b_(k+1) * sum_{i in richest k} 1/w_i``; the collection is
        ``b_(k+1) * sum_{j in winners} 1/w_j``, and no k experts have a smaller sum
        of reciprocals than the k richest. The bound is tight when the winners are
        the k richest and the (k+1)-th and (k+2)-th bids coincide, so in bf16
        rounding can cross it by a fraction of a percent -- bounded noise, not a
        leak.

        The divisor reads the recipient's own wealth whenever it is among the k
        richest. That is still report-independent, for a different reason than the
        exclusion rule: wealth is accumulated state read from a detached snapshot,
        not something an expert reports this token.

        Exploration takes one payment out of a token's collection; the rebate on
        such a token is scaled down by ``_fund_exploration`` so the same argument
        holds with one slot fewer.

        This replaced the pool's largest wealth, which was also safe and simpler
        but under-rebated by whatever the richest expert stood above the rest:
        against ``w_max`` the returned fraction was 94% on a flat wealth vector,
        68% across the configured band and 3.6% with one expert at ``max_wealth``
        and the rest at the floor -- weakest in exactly the monopoly regime where
        the drain bites hardest. The regimes test in ``tests/test_auction.py``
        pins the fractions this divisor returns.
        """
        batch, seq_len, _ = bids.shape
        k, n = self.top_k, self.num_experts

        # Subsumes k >= n: there is no (k+2)-th bid to rebate out of either way.
        if n < k + 2:
            return torch.zeros(batch, seq_len, n, device=bids.device, dtype=bids.dtype)

        accumulate_dtype = torch.promote_types(bids.dtype, torch.float32)
        ranked, ranked_idx = torch.topk(bids.to(accumulate_dtype), k + 2, dim=-1)

        displaced = ranked[..., k].unsqueeze(-1)  # (k+1)-th highest overall
        displaced_without_self = ranked[..., k + 1].unsqueeze(-1)  # (k+2)-th

        # Removing i only shifts the ranking for the k+1 experts above the cut.
        in_top_k_plus_one = torch.zeros_like(bids, dtype=torch.bool)
        in_top_k_plus_one.scatter_(-1, ranked_idx[..., : k + 1], True)

        reference = torch.where(in_top_k_plus_one, displaced_without_self, displaced)
        richest = torch.topk(wealth.to(accumulate_dtype), k).values.clamp_min(WEALTH_EPSILON)
        harmonic_mean = k / (1.0 / richest).sum()
        rebates = (k / n) * reference / harmonic_mean

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

        The baseline share is taken in the log domain rather than over the bids
        themselves, and that is not a reformulation of the same gate. Softmax is not
        scale invariant, so ``softmax(bids)`` reads the *absolute* wealth scale as a
        sharpness knob: measured over softplus reports at default initialisation,
        the top-1 weight has median ~0.99 at ``initial_wealth`` and 1.000 at
        ``max_wealth``, and across the configured band the effective expert count is
        1.000 -- ``top_k=2`` paying for two experts and using one. Wealth drifts
        during training, so under that gate every published number would be read
        through a sharpness moving independently of the variable under test.
        In the log domain a uniform rescaling of all wealth is a constant shift that
        softmax absorbs exactly, so the gate answers only to *relative* wealth --
        which is the quantity the economy exists to move.
        """
        if self.routing_share == ROUTING_SHARE_UNIFORM:
            return torch.full_like(top_bids, 1.0 / self.top_k)

        if self.differentiable and self.training:
            return self._differentiable_routing(bids, selected_experts)
        return F.softmax(self._log_bids(top_bids), dim=-1).to(top_bids.dtype)

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
        # not to absorb a negative one: a negative-wealth winner drags b_(k+1) below
        # zero too, so clamping its wealth up to 1e-12 divides a negative numerator
        # into an enormous negative price with nothing complaining.

    def _log_bids(self, bids: torch.Tensor) -> torch.Tensor:
        """Bids in the log domain, tempered, ready for a softmax.

        Normalised by the row's largest bid *before* the log, not shifted after it.
        The two are the same constant shift in the algebra -- softmax absorbs
        either -- but not in float32. ``log(b * s)`` carries the wealth scale as an
        additive offset of ``log(s)``, which at ``s = 1e8`` is 18.4 against log-bids
        of order 1, so the low bits that distinguish two nearby bids are spent
        representing an offset that is about to cancel; dividing by ``temperature``
        then multiplies what is left by ``1 / tau``. Measured at ``tau = 0.1`` under
        a wealth rescale of 1e8, over 500 draws of shape (2, 32, 8): shifting after
        the log exceeds 1e-6 on every draw and peaks above 5e-6, while normalising
        before it stays under 1e-6 on all of them and peaks around 6e-7. That is the
        difference between an invariance claim that degrades as the gate sharpens
        and one that does not.

        The *ratio* is clamped rather than the bid, which keeps every sign case
        finite: a row of non-positive bids collapses onto the floor and shares out
        evenly, the right answer for a token no expert wants.

        Accumulated in float32 for the same reason the payments are: bf16 is the
        training dtype, and a log ratio between two nearby bids is a small
        difference of two similar magnitudes.
        """
        accumulate_dtype = torch.promote_types(bids.dtype, torch.float32)
        scaled = bids.to(accumulate_dtype)
        largest = scaled.amax(dim=-1, keepdim=True).clamp_min(BID_LOG_FLOOR)
        return (scaled / largest).clamp_min(BID_LOG_FLOOR).log() / self.temperature

    def _differentiable_routing(
        self, bids: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """The straight-through gate, over the same log-domain shares.

        The renormalisation below needs no epsilon. ``soft_weights`` sums to one
        over all ``num_experts``, and the selected experts are the ``top_k``
        largest, so their share is at least ``top_k / num_experts`` -- bounded away
        from zero by the shape of the tensor rather than by a guard. Keeping it
        exact is what lets this path and the eval path agree to float tolerance
        instead of to 1e-8.
        """
        soft_weights = F.softmax(self._log_bids(bids), dim=-1)

        hard_mask = torch.zeros_like(soft_weights)
        hard_mask.scatter_(-1, selected_experts, 1.0)

        differentiable_mask = hard_mask + (soft_weights - soft_weights.detach())

        routing_weights_full = differentiable_mask * soft_weights
        routing_weights = torch.gather(routing_weights_full, -1, selected_experts)
        normalised = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return normalised.to(bids.dtype)
