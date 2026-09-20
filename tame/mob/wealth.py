"""The MoB layer's side of the ledger: what a winner realised, and what it is charged.

:mod:`mob.ledger` owns the settlement -- one :class:`~mob.ledger.WealthUpdater`
relaxing a ledger, paying a reward signal and holding the result at a floor. This
module is what a MoB layer brings to it: the value every winner realised on every
token (:func:`realised_values`, captured by a hook the loss backward fires), the
VCG charge in the reward's own units, the caches a settlement is assembled from,
and the two diagnostics the ledger does not read.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F

from .experts import SELF_PREDICTION_RANGE
from .ledger import (
    PERSISTENCE_DECOUPLED,
    PERSISTENCE_SHUFFLED,
    Settlement,
    WealthUpdater,
)

if TYPE_CHECKING:
    from .mob_config import MoBConfig

logger = logging.getLogger(__name__)


class ValueSummary(NamedTuple):
    """What the last loss-feedback update paid for, averaged over the slots it priced.

    Every field is a 0-dim tensor on the wealth's device, for the same reason the
    routing diagnostics are: these are produced on every step of every layer, and
    the training loop wants one host sync at the end of a step rather than one
    per statistic.

    ``mean_surplus`` is ``value - price`` per win, the sign of the wealth transfer
    under the uniform share (where every winner's share is the ``1/top_k`` the
    charge coefficient assumes; under the proportional baseline the two diverge).
    It is the quantity #15 was opened over: negative means winning is a
    loss-making trade and the economy rewards abstention. ``mean_report`` beside
    ``mean_realised_value`` is the calibration check -- a trained head's report
    should sit at the mean of the value it realises, not above it, which the
    clamped target this replaced could not deliver. Slots handed out by
    exploration are excluded from all six: they were not traded, so they say
    nothing about the market.

    ``mean_goal_term`` and ``goal_share`` are #54's, and they exist because
    without them a null cannot be told from a mechanism that never mattered. The
    goal term (#33) is *already inside* ``mean_realised_value`` by the time this
    is built, so the share is the fraction of what a cell was paid that the goal
    accounted for, not a ratio of two independent quantities. Both are absolute
    values: the term is signed per slot and the signs would otherwise cancel into
    a small mean over a large effect. Zero with no field attached, which is every
    arm before #54 and the correct reading of one.
    """

    mean_realised_value: torch.Tensor
    mean_report: torch.Tensor
    mean_price: torch.Tensor
    mean_surplus: torch.Tensor
    mean_goal_term: torch.Tensor
    goal_share: torch.Tensor


def realised_values(contributions: torch.Tensor, output_gradient: torch.Tensor) -> torch.Tensor:
    """The value each winner realised on each token: its contribution against the loss gradient.

    ``contributions`` is ``(batch, seq, top_k, hidden)`` -- for each winner slot,
    what that expert's output differed from the tissue default by -- and
    ``output_gradient`` is ``(batch, seq, hidden)``, the gradient of the loss at
    the layer output. Their inner product is the first-order change in loss from
    that contribution, and its negative is the loss *reduction* the expert
    delivered: a positive value means the organism's loss would have been higher
    without this expert on this token.

    This is a counterfactual against the shared base, not against the expert's own
    history. The definition it replaced -- an expert's own EMA loss minus its loss
    on the tokens it won -- asked whether the expert was surprised by itself, and
    in steady state nothing is surprised by itself, so it averaged to zero and the
    economy had nothing to allocate on. What is measured here is what the expert
    adds to what the tissue would have done anyway, which is also exactly the
    externality the auction prices.

    Accumulated in float32: a bf16 inner product over a hidden dimension of
    several thousand loses the small differences that distinguish two experts.
    """
    return -(contributions.float() * output_gradient.float().unsqueeze(-2)).sum(dim=-1)


class WealthUpdateMixin:
    # Declared for type checking — provided by MixtureOfBidders.__init__()
    config: MoBConfig
    wealth_updater: WealthUpdater
    expert_wealth: torch.Tensor
    expert_usage_count: torch.Tensor
    expert_baseline_loss: torch.Tensor
    expert_performance_ema: torch.Tensor
    expert_steps_since_held: torch.Tensor
    _cached_selected_experts: torch.Tensor | None
    _cached_routing_weights: torch.Tensor | None
    _cached_confidences: torch.Tensor | None
    _live_confidences: torch.Tensor | None
    _cached_payments: torch.Tensor | None
    _cached_rebates: torch.Tensor | None
    _cached_explored: torch.Tensor | None
    _cached_values: torch.Tensor | None
    _cached_goal_terms: torch.Tensor | None
    _cached_stress: torch.Tensor | None
    # What #59's charge took from every cell at the last settlement.
    last_stress_charge: float
    # #60: each cell's prediction of the value it would realise, captured in the
    # forward where the hidden states are, and scored in the settlement where
    # the realised values are.
    _cached_predictions: torch.Tensor | None
    _loss_feedback_pending: bool
    _cached_calibration_loss: torch.Tensor | None
    last_value_summary: ValueSummary | None
    last_realised_values: torch.Tensor | None
    last_goal_terms: torch.Tensor | None

    def allocation_wealth(self) -> torch.Tensor:
        """The wealth the gate reads: the ledger, or under ``decoupled`` a pinned one.

        Which of the two is the ledger's own mode (:attr:`WealthUpdater.pinned_at`)
        rather than a branch here, so that a ledger settling as a shadow is the
        same class as one the gate reads and not a fourth wealth path.
        """
        return self.wealth_updater.allocation_wealth(self.expert_wealth)

    def allocation_staleness(self) -> torch.Tensor | None:
        """The staleness the exploration draw reads; ``None`` draws uniformly (#38).

        Under ``decoupled`` re-entry is uniform: a cell that has been shut out is
        re-sampled no faster than one that has not, so nothing the ledger records
        reaches the allocation by this channel either. The ledger still ages, for
        the shadow. Read off the dial rather than off the ledger because the draw
        belongs to the gate: what #40 folded into the ledger is the pinned wealth,
        which the ledger is what the gate reads *instead of*.
        """
        if self.config.persistence_coupling == PERSISTENCE_DECOUPLED:
            return None
        return self.expert_steps_since_held

    def _value_target_sources(self) -> torch.Tensor:
        """Which expert's realised values each head regresses onto.

        Its own, except under ``shuffled``: a cyclic shift by a fresh draw in
        ``[1, num_experts)`` each step, which is a derangement in one draw -- no
        head ever sees its own value, and each sees every other expert's equally
        often over steps. The draw is from the global stream, as the exploration
        draw is, so the shuffled arm's trajectory diverges from the live arm's from
        the first settlement and never by a hidden constant.
        """
        num_experts = self.config.num_experts
        experts = torch.arange(num_experts, device=self.expert_wealth.device)
        if self.config.persistence_coupling != PERSISTENCE_SHUFFLED:
            return experts
        shift = int(torch.randint(1, num_experts, (1,)).item())
        return (experts + shift) % num_experts

    def _transfer_coefficient(self, reward_multiplier: float) -> float:
        """The single scale that makes ``reward - charge`` a quasi-linear utility.

        Quasi-linearity means one coefficient multiplies both halves. A reward is
        ``value x share x reward_scale x reward_multiplier`` and a charge is
        ``price x payment_scale x <this>``, so the charge has to carry the reward's
        own scale for the two to be in one currency.

        Getting this wrong is not cosmetic. With a reward coefficient of 100 against
        a charge coefficient of 0.3, an expert maximising wealth wins whenever
        ``value > 0.006 x price`` while the auction only lets it win when
        ``report > price`` -- so overreporting pays, and the mechanism's
        strategyproofness says nothing about the economy that realises the payoff.

        ``reward_multiplier`` is passed per call because the three reward signals
        use different ones; a single constant would only be quasi-linear for
        whichever path it was derived from. ``payment_scale`` survives as a
        dimensionless deviation from the balanced point, so 1.0 is the value the
        theory picks and anything else is a deliberate over- or under-pricing.
        """
        return (
            self.config.payment_scale
            * self.config.reward_scale
            * reward_multiplier
            / self.config.top_k
        )

    def _vcg_charges(
        self,
        payments: torch.Tensor | None,
        selected_experts: torch.Tensor,
        num_tokens: int,
        reward_multiplier: float,
        rebates: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-expert VCG transfer, in the same units and at the same scale as rewards.

        The utility model is quasi-linear: wealth moves by ``reward - charge``, with
        the payment subtracted rather than scaling the reward. Quasi-linearity is a
        precondition for every VCG result, so the multiplicative haircut this
        replaces could not support an incentive claim of any kind.

        Payments and rewards are both denominated in loss reduction -- the price is
        the weighted externality divided by the winner's own wealth, and the report
        that produced the bid is itself a loss-reduction estimate. Each expert pays
        its mean per-token price weighted by its share of the batch, so an expert
        that wins few tokens is charged proportionally less than one that wins many.

        ``valid_mask`` drops padding from both the charge and the rebate, as the
        reward path drops it: a padded position is not a trade, and charging for
        one while paying nothing on it would make padding a pure drain.
        """
        charges = torch.zeros_like(self.expert_wealth)
        if not self.config.use_vcg_payments or payments is None:
            return charges

        for slot in range(self.config.top_k):
            for expert_idx in range(self.config.num_experts):
                mask = selected_experts[:, :, slot] == expert_idx
                if valid_mask is not None:
                    mask = mask & valid_mask
                if not mask.any():
                    continue
                mean_payment = payments[:, :, slot][mask].mean()
                token_share = mask.sum().float() / num_tokens
                charges[expert_idx] += mean_payment * token_share

        if rebates is not None:
            # Cavallo redistribution: every expert is rebated, winners and losers
            # alike, from a quantity none of them can influence. Netting it here
            # keeps the wealth update a single transfer rather than two passes.
            if valid_mask is None:
                charges = charges - rebates.mean(dim=(0, 1))
            elif valid_mask.any():
                charges = charges - rebates[valid_mask].mean(dim=0)

        return charges * self._transfer_coefficient(reward_multiplier)

    def get_confidence_calibration_loss(self) -> torch.Tensor:
        if self._cached_calibration_loss is None:
            return torch.tensor(0.0, device=self.expert_wealth.device)
        return self._cached_calibration_loss

    def _compute_and_cache_calibration_loss(
        self,
        values: torch.Tensor,
        selected_experts: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Calibrate every confidence head against its own realised value.

        This is the objective that makes an expert an agent. Under the uniform
        routing share the language-modelling loss reaches the expert adapters but
        not the confidence heads, so a head is trained by nothing except the
        outcomes it personally realised: the value it delivered on the tokens it
        personally held, measured as its contribution against the loss gradient.

        Expert *i*'s term reads ``confidences[:, :, i]``, which only
        ``confidence_heads[i]`` produces, and the routing path reads *detached*
        hidden states, so the gradient reaches that head and the coupling module and
        stops there. It never reaches the backbone: every head is fed from the same
        representation, so an undetached path would make each expert's private
        objective a shared auxiliary loss on everything below the layer -- the
        central planner arriving through the back door. Held by
        ``test_value_objective_does_not_train_the_backbone``.

        Reporting that value truthfully is also the utility-maximising thing for the
        head to do. The auction pays each winner its critical value and hands it a
        share that ignores its own report, so the mechanism is strategyproof and an
        expert's optimal report *is* its value; the discrete utility has zero
        gradient almost everywhere, and regressing onto the value is the tractable
        form of the same optimum.

        The target is the realised value as measured, negative included. The
        clamped target this replaced trained each head onto the *positive part* of
        its value, so every report was an overestimate and prices followed reports
        above what any win was worth -- the bias #15 was opened over. Abstention
        needs no clamp: the report is a softplus, so the least-squares fit of a
        non-negative report to a target whose mean is negative is a report at zero,
        and the head abstains exactly when the mean of what it realises says it
        should.

        One honest limit remains. A head sees a target only on tokens it held, so
        the objective carries the selection bias of any bandit-feedback signal -- and
        a head that has fallen to zero would never see another. The exploration slot
        in the auction is what keeps every head sampling.
        """
        live_confidences = self._live_confidences
        self._live_confidences = None

        zero = torch.zeros((), device=self.expert_wealth.device)
        weight = self.config.confidence_calibration_weight
        if live_confidences is None or weight == 0.0:
            self._cached_calibration_loss = zero
            return

        seq_len = values.size(1)
        if live_confidences.size(1) < seq_len:
            self._cached_calibration_loss = zero
            return
        live_confidences = live_confidences[:, :seq_len, :]

        sources = self._value_target_sources()
        expert_terms = []
        for expert_idx in range(self.config.num_experts):
            # Under the shuffled arm the head is trained on another expert's
            # tokens and targets (#39); otherwise ``source`` is the head's own.
            source = int(sources[expert_idx])
            held_slots = selected_experts == source
            held = held_slots.any(dim=-1) & valid_mask
            if not held.any():
                continue

            target = (values * held_slots).sum(dim=-1)[held]
            prediction = live_confidences[:, :, expert_idx][held].float()
            expert_terms.append(F.mse_loss(prediction, target))

        if not expert_terms:
            self._cached_calibration_loss = zero
            return

        objective = torch.stack(expert_terms).mean()
        if not torch.isfinite(objective):
            self._cached_calibration_loss = zero
            return

        self._cached_calibration_loss = objective * weight

    def _self_scores(
        self,
        values: torch.Tensor,
        selected_experts: torch.Tensor,
        valid_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor | None:
        """Each cell's Brier score on the tokens it held: ``-(prediction - realised)^2``.

        Strictly proper, so the prediction that maximises it is the cell's own
        expectation of what it will realise -- which is what makes a wrong
        self-model cost something rather than merely look wrong. Bounded by the
        clamp both sides carry (``SELF_PREDICTION_RANGE``), because an unbounded
        quadratic lets one outlying token empty a ledger.

        Zero for a cell that held nothing: a cell with no tokens made no
        prediction anybody can score, and charging it for that would be charging
        it for losing the auction.
        """
        if self.config.self_score_mu <= 0.0 or self._cached_predictions is None:
            return None
        predictions = self._cached_predictions[:, :seq_len, :]
        scores = torch.zeros(self.config.num_experts, device=values.device)
        clamped_values = values.clamp(-SELF_PREDICTION_RANGE, SELF_PREDICTION_RANGE)
        for expert_idx in range(self.config.num_experts):
            held_slots = selected_experts == expert_idx
            held = held_slots.any(dim=-1) & valid_mask
            if not held.any():
                continue
            realised = (clamped_values * held_slots).sum(dim=-1)[held]
            predicted = predictions[..., expert_idx][held]
            scores[expert_idx] = -((predicted - realised) ** 2).mean()
        return scores

    def _stress_for(self, seq_len: int) -> torch.Tensor | None:
        """This step's transmitted stress, trimmed to the tokens the loss reached (#59).

        The forward caches it over the whole sequence and the loss can arrive
        shorter, exactly as the values and goal terms can; trimmed the same way
        and for the same reason.
        """
        if self._cached_stress is None:
            return None
        return self._cached_stress[:, :seq_len]

    def _discard_loss_feedback(self) -> None:
        self._loss_feedback_pending = False
        self._live_confidences = None
        self._cached_calibration_loss = None
        self._cached_values = None
        self._cached_goal_terms = None
        self._cached_stress = None
        self._cached_predictions = None

    def update_wealth_from_loss(
        self,
        per_token_loss: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        loss_gradient_scale: float = 1.0,
    ):
        """Settle the step: pay every winner what it realised, charge it what it bid past.

        Must run *after* the loss backward. A winner's value is its contribution
        against the loss gradient at this layer, captured by a hook when the
        backward passes through, so before the backward there is nothing to pay.

        ``loss_gradient_scale`` restates the captured gradient in per-token loss
        units. The trainer backwards a mean over ``N`` valid tokens divided by the
        gradient-accumulation count, so its gradient is that many times smaller
        than the gradient of the summed per-token loss; passing ``N x accumulation``
        puts every value on the scale the reward constants, the wealth band and the
        prices were derived on.

        ``per_token_loss`` no longer defines value. It still feeds
        ``expert_baseline_loss``, an EMA of the loss on the tokens each expert held,
        which survives as a diagnostic and in the checkpoint format.
        """
        # The #12 control arm has no economy to pay. Returning quietly rather than
        # warning is deliberate: the trainer calls this on every arm so the two call
        # sites stay identical, and a warning per step would train the reader to
        # ignore the one that means a real dropped forward pass.
        if not self.config.has_economy:
            return

        if not self._loss_feedback_pending or self._cached_selected_experts is None:
            logger.warning("update_wealth_from_loss called without pending forward pass")
            self._discard_loss_feedback()
            return

        if self._cached_values is None:
            logger.warning(
                "update_wealth_from_loss called before the loss backward reached this "
                "layer; a winner's value is its contribution against the loss gradient, "
                "so there is nothing to pay. Backward the loss first"
            )
            self._discard_loss_feedback()
            return

        with torch.no_grad():
            selected_experts = self._cached_selected_experts
            routing_weights = self._cached_routing_weights
            confidences = self._cached_confidences
            payments = self._cached_payments
            rebates = self._cached_rebates
            explored = self._cached_explored
            values = self._cached_values

            assert routing_weights is not None
            assert confidences is not None

            batch_size, cached_seq_len, _ = confidences.shape

            if per_token_loss.dim() == 1:
                loss_seq_len = per_token_loss.numel() // batch_size
                per_token_loss = per_token_loss.view(batch_size, loss_seq_len)

            seq_len = per_token_loss.size(1)

            if seq_len > cached_seq_len:
                logger.warning(
                    f"Loss seq_len ({seq_len}) > cached seq_len ({cached_seq_len}), "
                    f"skipping wealth update"
                )
                self._discard_loss_feedback()
                return

            selected_experts = selected_experts[:, :seq_len, :]
            routing_weights = routing_weights[:, :seq_len, :]
            confidences = confidences[:, :seq_len, :]
            values = values[:, :seq_len, :].float() * loss_gradient_scale
            # The goal term (#33, ``mob.goal``) is already in per-token loss units
            # -- the dose prices a unit of goal error in them -- so it joins the
            # value after the gradient has been restated in those units, not before.
            goal_terms = self._cached_goal_terms
            if goal_terms is not None:
                goal_terms = goal_terms[:, :seq_len, :]
                values = values + goal_terms
            if payments is not None:
                payments = payments[:, :seq_len, :]
            if rebates is not None:
                rebates = rebates[:, :seq_len, :]
            if explored is not None:
                explored = explored[:, :seq_len, :]

            valid_mask = self._valid_token_mask(token_mask, batch_size, seq_len)

            self.last_stress_charge = self.wealth_updater.settle(
                self.expert_wealth,
                Settlement(
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    confidences=confidences,
                    num_tokens=batch_size * seq_len,
                    payments=payments,
                    rebates=rebates,
                    valid_mask=valid_mask,
                    values=values,
                    usage_count=self.expert_usage_count,
                    stress=self._stress_for(seq_len),
                    self_score=self._self_scores(values, selected_experts, valid_mask, seq_len),
                ),
                self._vcg_charges,
            )
            self._update_expert_baselines(values, per_token_loss, selected_experts, valid_mask)

            self.last_realised_values = values
            self.last_goal_terms = goal_terms
            self.last_value_summary = self._summarise_values(
                values, confidences, selected_experts, payments, explored, valid_mask, goal_terms
            )
            self._loss_feedback_pending = False
            self._cached_values = None
            self._cached_goal_terms = None
            self._cached_stress = None
            self._cached_predictions = None

        self._compute_and_cache_calibration_loss(values, selected_experts, valid_mask)

    def _update_expert_baselines(
        self,
        values: torch.Tensor,
        per_token_loss: torch.Tensor,
        selected_experts: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Two EMAs over the tokens each expert held, neither of which the ledger reads.

        ``expert_performance_ema`` tracks the value an expert realises and
        ``expert_baseline_loss`` the organism's loss on the tokens it held. Both
        are diagnostics that survive in the checkpoint format, kept out of the
        reward signal for that reason: what a cell is paid for should be readable
        without reading what is merely recorded about it.
        """
        decay = self.config.loss_ema_decay
        for expert_idx in range(self.config.num_experts):
            held_slots = selected_experts == expert_idx
            held = held_slots.any(dim=-1) & valid_mask
            if not held.any():
                continue

            expert_value = (values * held_slots).sum(dim=-1)
            self.expert_performance_ema[expert_idx] = (
                decay * self.expert_performance_ema[expert_idx]
                + (1 - decay) * expert_value[held].mean()
            )
            self.expert_baseline_loss[expert_idx] = (
                decay * self.expert_baseline_loss[expert_idx]
                + (1 - decay) * per_token_loss[held].float().mean()
            )

    def _valid_token_mask(
        self, token_mask: torch.Tensor | None, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Padding is dropped from every sum, never scored.

        A masked position scored as a token of zero loss would read as the largest
        loss reduction an expert can achieve. It is dropped from rewards, charges,
        the calibration targets and the diagnostics alike.
        """
        if token_mask is None:
            return torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=self.expert_wealth.device
            )
        if token_mask.dim() == 1:
            token_mask = token_mask.view(batch_size, -1)
        if token_mask.size(1) > seq_len:
            token_mask = token_mask[:, :seq_len]
        elif token_mask.size(1) < seq_len:
            token_mask = F.pad(token_mask, (0, seq_len - token_mask.size(1)), value=0)
        return token_mask > 0

    def _summarise_values(
        self,
        values: torch.Tensor,
        confidences: torch.Tensor,
        selected_experts: torch.Tensor,
        payments: torch.Tensor | None,
        explored: torch.Tensor | None,
        valid_mask: torch.Tensor,
        goal_terms: torch.Tensor | None = None,
    ) -> ValueSummary:
        traded = valid_mask.unsqueeze(-1).expand_as(values)
        if explored is not None:
            traded = traded & ~explored

        zero = torch.zeros((), device=values.device)
        if not traded.any():
            return ValueSummary(zero, zero, zero, zero, zero, zero)

        realised = values[traded]
        reports = torch.gather(confidences.float(), -1, selected_experts)[traded]
        if payments is not None and self.config.use_vcg_payments:
            price = payments.float()[traded] * self.config.payment_scale
        else:
            price = torch.zeros_like(realised)

        # Absolute, and against the absolute value the term is already part of:
        # "of what this cell was paid, how much was the goal". A share near zero
        # on a run launched to price a goal is the state #54 exists to make
        # visible -- the term present, attached, fingerprinted and too small to
        # decide anything, which reads in every other column exactly like the
        # extrinsic-teleology answer.
        goal = goal_terms.float()[traded].abs() if goal_terms is not None else None
        mean_goal = goal.mean() if goal is not None else zero
        scale = realised.abs().mean()
        share = mean_goal / scale if goal is not None and float(scale) > 0 else zero

        return ValueSummary(
            mean_realised_value=realised.mean(),
            mean_report=reports.mean(),
            mean_price=price.mean(),
            mean_surplus=(realised - price).mean(),
            mean_goal_term=mean_goal,
            goal_share=share,
        )

    def _settle_in_forward(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        confidences: torch.Tensor,
        payments: torch.Tensor | None,
        rebates: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        """The settlement for a layer no loss reaches: the same ledger, a different signal.

        ``update_wealth_from_loss`` cannot run here -- a winner's value is its
        contribution against the loss gradient, and inside the forward there is no
        gradient yet -- so the layer settles on whichever proxy its configuration
        selected. Which one that is belongs to ``WealthUpdater.for_experts``; this
        only assembles the step.
        """
        with torch.no_grad():
            batch_size, seq_len, _ = confidences.shape
            self.last_stress_charge = self.wealth_updater.settle(
                self.expert_wealth,
                Settlement(
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    confidences=confidences,
                    num_tokens=batch_size * seq_len,
                    payments=payments,
                    rebates=rebates,
                    output=output,
                    usage_count=self.expert_usage_count,
                ),
                self._vcg_charges,
            )
