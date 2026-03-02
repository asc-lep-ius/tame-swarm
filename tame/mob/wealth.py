import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

CONFIDENCE_SIGMOID_SCALE = 5.0
LOSS_REWARD_MULTIPLIER = 50.0
LOCAL_REWARD_MULTIPLIER = 5.0
PARTICIPATION_REWARD_MULTIPLIER = 10.0
COMPETITIVE_BONUS_FACTOR = 0.5
PAYMENT_COST_FACTOR = 0.1
WEALTH_EPSILON = 1e-6
LOSS_PAYMENT_CLAMP_MAX = 0.3
LOCAL_PAYMENT_CLAMP_MAX = 0.5


class WealthUpdateMixin:

    def get_confidence_calibration_loss(self) -> torch.Tensor:
        if self._cached_calibration_loss is None:
            return torch.tensor(0.0, device=self.expert_wealth.device)
        return self._cached_calibration_loss

    def _compute_and_cache_calibration_loss(self, confidences: torch.Tensor):
        target_confidence = torch.sigmoid(self.expert_performance_ema * CONFIDENCE_SIGMOID_SCALE)

        mean_confidences = confidences.mean(dim=(0, 1))

        if torch.isnan(mean_confidences).any() or torch.isnan(target_confidence).any():
            self._cached_calibration_loss = torch.tensor(0.0, device=self.expert_wealth.device)
            return

        calibration_loss = F.mse_loss(mean_confidences, target_confidence.detach())

        if torch.isnan(calibration_loss):
            self._cached_calibration_loss = torch.tensor(0.0, device=self.expert_wealth.device)
            return

        self._cached_calibration_loss = calibration_loss * self.config.confidence_calibration_weight

    def update_wealth_from_loss(
        self,
        per_token_loss: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ):
        if not self._loss_feedback_pending or self._cached_selected_experts is None:
            logger.warning("update_wealth_from_loss called without pending forward pass")
            return

        with torch.no_grad():
            selected_experts = self._cached_selected_experts
            routing_weights = self._cached_routing_weights
            confidences = self._cached_confidences
            payments = self._cached_payments

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
                    return

            seq_len = loss_seq_len

            if token_mask is not None:
                if token_mask.dim() == 1:
                    token_mask = token_mask.view(batch_size, -1)
                if token_mask.size(1) > seq_len:
                    token_mask = token_mask[:, :seq_len]
                elif token_mask.size(1) < seq_len:
                    pad_size = seq_len - token_mask.size(1)
                    token_mask = F.pad(token_mask, (0, pad_size), value=0)
                per_token_loss = per_token_loss * token_mask

            self.expert_wealth *= self.config.wealth_decay

            expert_rewards = torch.zeros_like(self.expert_wealth)
            expert_token_counts = torch.zeros_like(self.expert_wealth)

            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
                    if not mask.any():
                        continue

                    expert_losses = per_token_loss[mask]
                    mean_loss = expert_losses.mean()
                    token_count = mask.sum().float()
                    expert_token_counts[expert_idx] += token_count

                    baseline = self.expert_baseline_loss[expert_idx]

                    loss_reduction = baseline - mean_loss

                    mean_weight = routing_weights[:, :, k][mask].mean()
                    reward = loss_reduction * mean_weight * token_count / (batch_size * seq_len)

                    expert_rewards[expert_idx] += reward * self.config.reward_scale * LOSS_REWARD_MULTIPLIER

                    self.expert_baseline_loss[expert_idx] = (
                        self.config.loss_ema_decay * baseline
                        + (1 - self.config.loss_ema_decay) * mean_loss
                    )

                    self.expert_performance_ema[expert_idx] = (
                        self.config.loss_ema_decay * self.expert_performance_ema[expert_idx]
                        + (1 - self.config.loss_ema_decay) * loss_reduction
                    )

            if expert_rewards.abs().max() > WEALTH_EPSILON:
                reward_std = (
                    expert_rewards.std(correction=0)
                    if expert_rewards.numel() >= 2
                    else torch.tensor(WEALTH_EPSILON, device=expert_rewards.device)
                )
                normalized_rewards = (expert_rewards - expert_rewards.mean()) / (reward_std + WEALTH_EPSILON)
                competitive_bonus = F.relu(normalized_rewards) * expert_rewards.abs().mean() * COMPETITIVE_BONUS_FACTOR
                expert_rewards += competitive_bonus

            if self.config.use_vcg_payments and payments is not None:
                for k in range(self.config.top_k):
                    for expert_idx in range(self.config.num_experts):
                        mask = (selected_experts[:, :, k] == expert_idx)
                        if mask.any():
                            mean_payment = payments[:, :, k][mask].mean()
                            payment_fraction = mean_payment / (self.expert_wealth[expert_idx] + WEALTH_EPSILON)
                            expert_rewards[expert_idx] *= 1.0 - payment_fraction.clamp(0, LOSS_PAYMENT_CLAMP_MAX)

            self.expert_wealth += expert_rewards
            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)

            self._compute_and_cache_calibration_loss(confidences)

            self._loss_feedback_pending = False

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

            decay_rate = self.config.inference_wealth_decay if is_inference else self.config.wealth_decay
            self.expert_wealth *= decay_rate

            expert_rewards = torch.zeros_like(self.expert_wealth)
            output_norms = output.norm(dim=-1)
            global_mean_norm = output_norms.mean()

            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
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
                    expert_rewards[expert_idx] += reward * self.config.reward_scale * LOCAL_REWARD_MULTIPLIER

            mean_reward = expert_rewards.mean()
            if mean_reward > 0:
                competitive_bonus = (expert_rewards - mean_reward) * COMPETITIVE_BONUS_FACTOR
                expert_rewards += competitive_bonus.clamp(min=0)

            if self.config.use_vcg_payments and payments is not None:
                for k in range(self.config.top_k):
                    for expert_idx in range(self.config.num_experts):
                        mask = (selected_experts[:, :, k] == expert_idx)
                        if mask.any():
                            mean_payment = payments[:, :, k][mask].mean()
                            payment_cost = mean_payment * PAYMENT_COST_FACTOR / (self.expert_wealth[expert_idx] + WEALTH_EPSILON)
                            expert_rewards[expert_idx] *= 1.0 - payment_cost.clamp(0, LOCAL_PAYMENT_CLAMP_MAX)

            if is_inference and self.config.inference_exploration_bonus > 0:
                mean_usage = self.expert_usage_count.mean()
                if mean_usage > 0:
                    usage_ratio = self.expert_usage_count / (mean_usage + WEALTH_EPSILON)
                    exploration_bonus = (1.0 - usage_ratio).clamp(min=0) * self.config.inference_exploration_bonus
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
            expert_selections = torch.zeros_like(self.expert_wealth)

            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
                    if mask.any():
                        selection_count = mask.sum().float()
                        expert_selections[expert_idx] += selection_count

                        selection_fraction = selection_count / num_tokens
                        mean_confidence = confidences[:, :, expert_idx][mask].mean()
                        mean_weight = routing_weights[:, :, k][mask].mean()

                        base_reward = selection_fraction * mean_confidence * mean_weight
                        expert_rewards[expert_idx] += base_reward * self.config.reward_scale * PARTICIPATION_REWARD_MULTIPLIER

            mean_reward = expert_rewards.mean()
            if mean_reward > 0:
                competitive_bonus = (expert_rewards - mean_reward) * COMPETITIVE_BONUS_FACTOR
                expert_rewards += competitive_bonus.clamp(min=0)

            if self.config.use_vcg_payments and payments is not None:
                for k in range(self.config.top_k):
                    for expert_idx in range(self.config.num_experts):
                        mask = (selected_experts[:, :, k] == expert_idx)
                        if mask.any():
                            mean_payment = payments[:, :, k][mask].mean()
                            payment_cost = mean_payment * PAYMENT_COST_FACTOR / (self.expert_wealth[expert_idx] + WEALTH_EPSILON)
                            expert_rewards[expert_idx] *= 1.0 - payment_cost.clamp(0, LOCAL_PAYMENT_CLAMP_MAX)

            self.expert_wealth += expert_rewards

            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)
