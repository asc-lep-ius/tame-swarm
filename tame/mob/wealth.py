from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .mob_config import MoBConfig

logger = logging.getLogger(__name__)

LOSS_REWARD_MULTIPLIER = 50.0
LOCAL_REWARD_MULTIPLIER = 5.0
PARTICIPATION_REWARD_MULTIPLIER = 10.0
COMPETITIVE_BONUS_FACTOR = 0.5
WEALTH_EPSILON = 1e-6


class WealthUpdateMixin:
    # Declared for type checking — provided by MixtureOfBidders.__init__()
    config: MoBConfig
    expert_wealth: torch.Tensor
    expert_usage_count: torch.Tensor
    expert_baseline_loss: torch.Tensor
    expert_performance_ema: torch.Tensor
    _cached_selected_experts: torch.Tensor | None
    _cached_routing_weights: torch.Tensor | None
    _cached_confidences: torch.Tensor | None
    _live_confidences: torch.Tensor | None
    _cached_payments: torch.Tensor | None
    _cached_expert_token_masks: list[torch.Tensor] | None
    _loss_feedback_pending: bool
    _cached_calibration_loss: torch.Tensor | None

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

        ``reward_multiplier`` is passed per call because the three wealth paths use
        different ones; a single constant would only be quasi-linear for whichever
        path it was derived from. ``payment_scale`` survives as a dimensionless
        deviation from the balanced point, so 1.0 is the value the theory picks and
        anything else is a deliberate over- or under-pricing.
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
        """
        charges = torch.zeros_like(self.expert_wealth)
        if not self.config.use_vcg_payments or payments is None:
            return charges

        for slot in range(self.config.top_k):
            for expert_idx in range(self.config.num_experts):
                mask = selected_experts[:, :, slot] == expert_idx
                if not mask.any():
                    continue
                mean_payment = payments[:, :, slot][mask].mean()
                token_share = mask.sum().float() / num_tokens
                charges[expert_idx] += mean_payment * token_share

        return charges * self._transfer_coefficient(reward_multiplier)

    def get_confidence_calibration_loss(self) -> torch.Tensor:
        if self._cached_calibration_loss is None:
            return torch.tensor(0.0, device=self.expert_wealth.device)
        return self._cached_calibration_loss

    def _compute_and_cache_calibration_loss(
        self,
        per_token_loss: torch.Tensor,
        selected_experts: torch.Tensor,
        baselines: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        """Calibrate every confidence head against its own realised value.

        This is the objective that makes an expert an agent. Under the uniform
        routing share the language-modelling loss reaches the expert adapters but
        not the confidence heads, so a head is trained by nothing except the
        outcomes it personally realised: the value it delivered on the tokens it
        personally won, priced against its own loss baseline.

        Among the *heads*, expert *i*'s term is local: it reads
        ``confidences[:, :, i]``, which only ``confidence_heads[i]`` produces. It is
        not local to the layer. Every head is fed from the same
        ``confidence_hidden_states``, so the objective also backpropagates into
        whatever produced the hidden states -- the coupling module and the backbone
        below it. That is a real auxiliary objective on the trunk, weighted by
        ``confidence_calibration_weight`` and summed over every MoB layer.

        Reporting that value truthfully is also the utility-maximising thing for the
        head to do. The auction pays each winner its critical value and hands it a
        share that ignores its own report, so the mechanism is strategyproof and an
        expert's optimal report *is* its value; the discrete utility has zero
        gradient almost everywhere, and regressing onto the value is the tractable
        form of the same optimum.

        One honest limit. Value is observed only where the expert won, so the
        targets carry the selection bias of any bandit-feedback signal -- a head
        learns what its wins were worth, not what its losses would have been worth.
        """
        live_confidences = self._live_confidences
        self._live_confidences = None

        zero = torch.zeros((), device=self.expert_wealth.device)
        weight = self.config.confidence_calibration_weight
        if live_confidences is None or weight == 0.0:
            self._cached_calibration_loss = zero
            return

        seq_len = per_token_loss.size(1)
        if live_confidences.size(1) < seq_len:
            self._cached_calibration_loss = zero
            return
        live_confidences = live_confidences[:, :seq_len, :]

        expert_terms = []
        for expert_idx in range(self.config.num_experts):
            won = (selected_experts == expert_idx).any(dim=-1)
            if valid_mask is not None:
                won = won & valid_mask
            if not won.any():
                continue

            realised_value = (baselines[expert_idx] - per_token_loss[won]).detach()
            # Clamped at zero, not squashed: a non-positive realised value means the
            # expert would rather not have won, and the truthful report of that is
            # an abstaining bid. Clamping keeps report and reward in one currency,
            # which a sigmoid target would break -- and on the winning set, where
            # the reward is actually paid, clamp(0) is the identity.
            target = realised_value.float().clamp_min(0.0)
            prediction = live_confidences[:, :, expert_idx][won].float()
            expert_terms.append(F.mse_loss(prediction, target))

        if not expert_terms:
            self._cached_calibration_loss = zero
            return

        objective = torch.stack(expert_terms).mean()
        if not torch.isfinite(objective):
            self._cached_calibration_loss = zero
            return

        self._cached_calibration_loss = objective * weight

    def update_wealth_from_loss(
        self,
        per_token_loss: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ):
        if not self._loss_feedback_pending or self._cached_selected_experts is None:
            logger.warning("update_wealth_from_loss called without pending forward pass")
            self._live_confidences = None
            return

        with torch.no_grad():
            selected_experts = self._cached_selected_experts
            routing_weights = self._cached_routing_weights
            confidences = self._cached_confidences
            payments = self._cached_payments

            assert routing_weights is not None
            assert confidences is not None

            batch_size, cached_seq_len, _ = confidences.shape

            if per_token_loss.dim() == 1:
                loss_seq_len = per_token_loss.numel() // batch_size
                per_token_loss = per_token_loss.view(batch_size, loss_seq_len)

            loss_seq_len = per_token_loss.size(1)

            if loss_seq_len != cached_seq_len:
                if loss_seq_len < cached_seq_len:
                    selected_experts = selected_experts[:, :loss_seq_len, :]
                    routing_weights = routing_weights[:, :loss_seq_len, :]
                    confidences = confidences[:, :loss_seq_len, :]
                    if payments is not None:
                        payments = payments[:, :loss_seq_len, :]
                else:
                    logger.warning(
                        f"Loss seq_len ({loss_seq_len}) > cached seq_len ({cached_seq_len}), "
                        f"skipping wealth update"
                    )
                    self._loss_feedback_pending = False
                    self._live_confidences = None
                    self._cached_calibration_loss = None
                    return

            seq_len = loss_seq_len

            # The value objective reads the losses as measured. The reward path
            # below zeroes masked positions instead of dropping them, which would
            # read as a padding token of zero loss -- i.e. maximum realised value.
            unmasked_loss = per_token_loss
            valid_mask: torch.Tensor | None = None

            if token_mask is not None:
                if token_mask.dim() == 1:
                    token_mask = token_mask.view(batch_size, -1)
                if token_mask.size(1) > seq_len:
                    token_mask = token_mask[:, :seq_len]
                elif token_mask.size(1) < seq_len:
                    pad_size = seq_len - token_mask.size(1)
                    token_mask = F.pad(token_mask, (0, pad_size), value=0)
                valid_mask = token_mask > 0
                per_token_loss = per_token_loss * token_mask

            # Each expert's report is calibrated against the baseline it held when
            # it bid, not the one the loop below leaves behind.
            baselines = self.expert_baseline_loss.clone()

            self.expert_wealth *= self.config.wealth_decay

            expert_rewards = torch.zeros_like(self.expert_wealth)

            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = selected_experts[:, :, k] == expert_idx
                    if not mask.any():
                        continue

                    expert_losses = per_token_loss[mask]
                    mean_loss = expert_losses.mean()
                    token_count = mask.sum().float()

                    baseline = self.expert_baseline_loss[expert_idx]

                    loss_reduction = baseline - mean_loss

                    mean_weight = routing_weights[:, :, k][mask].mean()
                    reward = loss_reduction * mean_weight * token_count / (batch_size * seq_len)

                    expert_rewards[expert_idx] += (
                        reward * self.config.reward_scale * LOSS_REWARD_MULTIPLIER
                    )

                    self.expert_baseline_loss[expert_idx] = (
                        self.config.loss_ema_decay * baseline
                        + (1 - self.config.loss_ema_decay) * mean_loss
                    )

                    self.expert_performance_ema[expert_idx] = (
                        self.config.loss_ema_decay * self.expert_performance_ema[expert_idx]
                        + (1 - self.config.loss_ema_decay) * loss_reduction
                    )

            # Charge before the bonus, so the bonus rewards surplus rather than
            # gross activity. On gross it subsidises winning a token the expert was
            # charged for, which moves wealth's break-even below the auction's price
            # -- the same "paid for winning, never charged" defect on a second axis.
            # On the net it cannot: at value == price every net is zero, so the
            # guard below does not fire and the crossing sits exactly at the price.
            expert_rewards -= self._vcg_charges(
                payments, selected_experts, batch_size * seq_len, LOSS_REWARD_MULTIPLIER
            )

            if expert_rewards.abs().max() > WEALTH_EPSILON:
                reward_std = (
                    expert_rewards.std(correction=0)
                    if expert_rewards.numel() >= 2
                    else torch.tensor(WEALTH_EPSILON, device=expert_rewards.device)
                )
                normalized_rewards = (expert_rewards - expert_rewards.mean()) / (
                    reward_std + WEALTH_EPSILON
                )
                expert_rewards += (
                    F.relu(normalized_rewards)
                    * expert_rewards.abs().mean()
                    * COMPETITIVE_BONUS_FACTOR
                )

            self.expert_wealth += expert_rewards
            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)

            self._loss_feedback_pending = False

        self._compute_and_cache_calibration_loss(
            unmasked_loss, selected_experts, baselines, valid_mask
        )

    def _update_wealth_local_quality(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        confidences: torch.Tensor,
        payments: torch.Tensor | None,
        output: torch.Tensor,
    ):
        with torch.no_grad():
            batch_size, seq_len, hidden_dim = output.shape
            num_tokens = batch_size * seq_len

            is_inference = not self.config.use_loss_feedback

            decay_rate = (
                self.config.inference_wealth_decay if is_inference else self.config.wealth_decay
            )
            self.expert_wealth *= decay_rate

            expert_rewards = torch.zeros_like(self.expert_wealth)
            output_norms = output.norm(dim=-1)
            global_mean_norm = output_norms.mean()

            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = selected_experts[:, :, k] == expert_idx
                    if not mask.any():
                        continue

                    expert_output_norms = output_norms[mask]

                    if expert_output_norms.numel() >= 2:
                        norm_std = expert_output_norms.std(correction=0)
                    else:
                        norm_std = torch.tensor(0.0, device=expert_output_norms.device)
                    consistency_reward = 1.0 / (1.0 + norm_std)

                    norm_mean = expert_output_norms.mean()
                    magnitude_diff = (norm_mean - global_mean_norm).abs()
                    magnitude_reward = 1.0 / (1.0 + magnitude_diff)

                    quality = (consistency_reward + magnitude_reward) / 2.0

                    mean_confidence = confidences[:, :, expert_idx][mask].mean()
                    mean_weight = routing_weights[:, :, k][mask].mean()
                    selection_fraction = mask.sum().float() / num_tokens

                    reward = quality * mean_confidence * mean_weight * selection_fraction
                    expert_rewards[expert_idx] += (
                        reward * self.config.reward_scale * LOCAL_REWARD_MULTIPLIER
                    )

            expert_rewards -= self._vcg_charges(
                payments, selected_experts, num_tokens, LOCAL_REWARD_MULTIPLIER
            )

            # See the loss path: the bonus is paid on surplus, not on gross wins.
            mean_reward = expert_rewards.mean()
            if mean_reward > 0:
                competitive_bonus = (expert_rewards - mean_reward) * COMPETITIVE_BONUS_FACTOR
                expert_rewards += competitive_bonus.clamp(min=0)

            if is_inference and self.config.inference_exploration_bonus > 0:
                mean_usage = self.expert_usage_count.mean()
                if mean_usage > 0:
                    usage_ratio = self.expert_usage_count / (mean_usage + WEALTH_EPSILON)
                    exploration_bonus = (1.0 - usage_ratio).clamp(
                        min=0
                    ) * self.config.inference_exploration_bonus
                    exploration_bonus = exploration_bonus * self.expert_wealth.mean()
                    expert_rewards += exploration_bonus

            self.expert_wealth += expert_rewards
            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)

    def _update_wealth_participation(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        confidences: torch.Tensor,
        payments: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            batch_size, seq_len, _ = confidences.shape
            num_tokens = batch_size * seq_len

            self.expert_wealth *= self.config.wealth_decay

            expert_rewards = torch.zeros_like(self.expert_wealth)

            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = selected_experts[:, :, k] == expert_idx
                    if mask.any():
                        selection_count = mask.sum().float()
                        selection_fraction = selection_count / num_tokens
                        mean_confidence = confidences[:, :, expert_idx][mask].mean()
                        mean_weight = routing_weights[:, :, k][mask].mean()

                        base_reward = selection_fraction * mean_confidence * mean_weight
                        expert_rewards[expert_idx] += (
                            base_reward * self.config.reward_scale * PARTICIPATION_REWARD_MULTIPLIER
                        )

            expert_rewards -= self._vcg_charges(
                payments, selected_experts, num_tokens, PARTICIPATION_REWARD_MULTIPLIER
            )

            # See the loss path: the bonus is paid on surplus, not on gross wins.
            mean_reward = expert_rewards.mean()
            if mean_reward > 0:
                competitive_bonus = (expert_rewards - mean_reward) * COMPETITIVE_BONUS_FACTOR
                expert_rewards += competitive_bonus.clamp(min=0)

            self.expert_wealth += expert_rewards

            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)
