import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

try:
    from ..coupling import CouplingMetrics, SteeringCoupling, SteeringCouplingConfig
except ImportError:
    if __package__ != "mob":
        raise
    from coupling import CouplingMetrics, SteeringCoupling, SteeringCouplingConfig

from .auction import VCGAuctioneer
from .experts import ConfidenceHead, Expert, LightweightExpert
from .mob_config import MoBConfig
from .wealth import WealthUpdateMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoBStats:
    confidence_logits: torch.Tensor
    confidences: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    expert_wealth: torch.Tensor
    expert_usage: torch.Tensor
    expert_performance: torch.Tensor
    router_z_loss: torch.Tensor
    coupling_metrics: CouplingMetrics | None = None


class MixtureOfBidders(WealthUpdateMixin, nn.Module):
    def __init__(self, config: MoBConfig):
        super().__init__()
        self.config = config
        self.use_shared_base = config.use_shared_base

        if self.use_shared_base:
            self.base_gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.base_up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.base_down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

            self.experts = nn.ModuleList(
                [
                    LightweightExpert(
                        config.hidden_dim,
                        config.intermediate_dim,
                        rank=config.adapter_rank,
                        alpha=config.adapter_alpha,
                    )
                    for _ in range(config.num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [
                    Expert(config.hidden_dim, config.intermediate_dim)
                    for _ in range(config.num_experts)
                ]
            )

        self.confidence_heads = nn.ModuleList(
            [
                ConfidenceHead(config.hidden_dim, expert_id=i, num_experts=config.num_experts)
                for i in range(config.num_experts)
            ]
        )

        self.auctioneer = VCGAuctioneer(
            config.num_experts,
            config.top_k,
            differentiable=config.use_differentiable_routing,
        )

        self.register_buffer(
            "expert_wealth",
            torch.full((config.num_experts,), config.initial_wealth),
        )

        self.register_buffer(
            "expert_usage_count",
            torch.zeros(config.num_experts),
        )

        self.register_buffer(
            "expert_baseline_loss",
            torch.ones(config.num_experts),
        )

        self.register_buffer(
            "expert_performance_ema",
            torch.zeros(config.num_experts),
        )

        self.last_stats: MoBStats | None = None

        self.wealth_history: list[list[float]] = []
        self._track_wealth: bool = False

        self._cached_selected_experts: torch.Tensor | None = None
        self._cached_routing_weights: torch.Tensor | None = None
        self._cached_confidences: torch.Tensor | None = None
        self._cached_payments: torch.Tensor | None = None
        self._cached_expert_token_masks: list[torch.Tensor] | None = None
        self._loss_feedback_pending: bool = False
        self._cached_calibration_loss: torch.Tensor | None = None
        self._last_coupling_metrics: CouplingMetrics | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        update_wealth: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the MoB layer."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        confidence_hidden_states = hidden_states
        coupling_metrics: CouplingMetrics | None = None
        coupling = self._get_coupling()
        if coupling is not None:
            confidence_hidden_states = coupling(hidden_states)
            coupling_metrics = coupling.last_metrics

        confidence_logits = torch.stack(
            [
                cast(ConfidenceHead, confidence_head)
                .forward_logits(confidence_hidden_states)
                .squeeze(-1)
                for confidence_head in self.confidence_heads
            ],
            dim=-1,
        )
        confidences = torch.sigmoid(confidence_logits)
        router_z_loss = self._compute_router_z_loss(confidence_logits)

        selected_experts, routing_weights, payments = self.auctioneer(
            confidences, self.expert_wealth
        )

        output = torch.zeros_like(hidden_states)

        if self.training:
            output = self._forward_training(
                hidden_states, output, selected_experts, routing_weights, update_wealth
            )
        else:
            output = self._forward_inference(
                hidden_states, output, selected_experts, routing_weights, update_wealth
            )

        if output.dtype == torch.bfloat16 or output.dtype == torch.float16:
            output = torch.nan_to_num(output, nan=0.0, posinf=65000.0, neginf=-65000.0)

        if self.training and self.config.use_loss_feedback:
            self._cached_selected_experts = selected_experts.detach()
            self._cached_routing_weights = routing_weights.detach()
            self._cached_confidences = confidences.detach()
            self._cached_payments = payments.detach() if payments is not None else None
            self._loss_feedback_pending = True

        if update_wealth and self.config.use_local_quality and not self.config.use_loss_feedback:
            self._update_wealth_local_quality(
                selected_experts, routing_weights, confidences, payments, output
            )
        elif update_wealth and self.training:
            if self.config.use_local_quality and not self.config.use_loss_feedback:
                self._update_wealth_local_quality(
                    selected_experts, routing_weights, confidences, payments, output
                )
            elif not self.config.use_local_quality:
                self._update_wealth_participation(
                    selected_experts, routing_weights, confidences, payments
                )

        self.last_stats = MoBStats(
            confidence_logits=confidence_logits.detach(),
            confidences=confidences.detach(),
            selected_experts=selected_experts.detach(),
            routing_weights=routing_weights.detach(),
            expert_wealth=self.expert_wealth.detach().clone(),
            expert_usage=self.expert_usage_count.detach().clone(),
            expert_performance=self.expert_performance_ema.detach().clone(),
            router_z_loss=router_z_loss.detach(),
            coupling_metrics=coupling_metrics,
        )
        self._last_coupling_metrics = coupling_metrics

        if self._track_wealth:
            self.wealth_history.append(self.expert_wealth.cpu().tolist())

        return output

    def attach_coupling(
        self,
        steering_direction: torch.Tensor,
        config: SteeringCouplingConfig | None = None,
    ) -> SteeringCoupling:
        coupling_config = config or SteeringCouplingConfig(hidden_dim=self.config.hidden_dim)
        if coupling_config.hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"Coupling hidden_dim {coupling_config.hidden_dim} does not match "
                f"MoB hidden_dim {self.config.hidden_dim}"
            )

        if hasattr(self, "coupling"):
            self.detach_coupling()

        coupling = SteeringCoupling(coupling_config, steering_direction)
        reference_parameter = next(self.confidence_heads.parameters())
        coupling.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        self.add_module("coupling", coupling)
        self._last_coupling_metrics = None
        return coupling

    def detach_coupling(self) -> None:
        if hasattr(self, "coupling"):
            delattr(self, "coupling")
        self._last_coupling_metrics = None
        self.last_stats = None

    def set_coupling_step(self, step: int) -> "MixtureOfBidders":
        if step < 0:
            raise ValueError("coupling step must be non-negative")

        coupling = self._get_coupling()
        if coupling is not None:
            coupling.set_coupling_step(step)
        return self

    def _get_coupling(self) -> SteeringCoupling | None:
        return cast(SteeringCoupling | None, getattr(self, "coupling", None))

    def _compute_router_z_loss(self, confidence_logits: torch.Tensor) -> torch.Tensor:
        log_z = torch.logsumexp(confidence_logits.float(), dim=-1)
        return log_z.square().mean() * self.config.confidence_z_loss_weight

    def _forward_training(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        update_wealth: bool,
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_output = output.reshape(-1, hidden_dim)

        for k in range(self.config.top_k):
            flat_expert_indices = selected_experts[:, :, k].reshape(-1)
            flat_weights = routing_weights[:, :, k].reshape(-1)

            for expert_idx in range(self.config.num_experts):
                mask = flat_expert_indices == expert_idx
                if not mask.any():
                    continue

                expert_input = flat_hidden[mask]

                if self.use_shared_base:
                    expert_output = self.experts[expert_idx](
                        expert_input,
                        self.base_gate_proj,
                        self.base_up_proj,
                        self.base_down_proj,
                    )
                else:
                    expert_output = self.experts[expert_idx](expert_input)

                weighted = expert_output * flat_weights[mask].unsqueeze(-1)
                token_indices = mask.nonzero(as_tuple=False).squeeze(-1)
                flat_output.index_add_(0, token_indices, weighted)

                if update_wealth:
                    self.expert_usage_count[expert_idx] += mask.sum().float()

        return flat_output.reshape_as(hidden_states)

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        update_wealth: bool,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]

        for k in range(self.config.top_k):
            expert_indices = selected_experts[:, :, k]
            weights = routing_weights[:, :, k : k + 1]

            for expert_idx in range(self.config.num_experts):
                mask = expert_indices == expert_idx
                if not mask.any():
                    continue

                expert_input = hidden_states[mask]

                if self.use_shared_base:
                    expert_output = self.experts[expert_idx](
                        expert_input,
                        self.base_gate_proj,
                        self.base_up_proj,
                        self.base_down_proj,
                    )
                else:
                    expert_output = self.experts[expert_idx](expert_input)

                weight_vals = weights.squeeze(-1)[mask]
                weighted_expert_output = expert_output * weight_vals.unsqueeze(-1)

                mask_indices = mask.nonzero(as_tuple=False)
                flat_indices = mask_indices[:, 0] * seq_len + mask_indices[:, 1]
                output_flat = output.view(-1, hidden_dim)
                output_flat.index_add_(0, flat_indices, weighted_expert_output)

                if update_wealth:
                    self.expert_usage_count[expert_idx] += mask.sum().float()

        return output

    def start_tracking(self):
        self._track_wealth = True
        self.wealth_history = []

    def stop_tracking(self):
        self._track_wealth = False

    def get_wealth_history(self) -> list[list[float]]:
        return self.wealth_history.copy()

    def reset_tracking(self):
        """Reset the wealth history without disabling tracking."""
        self.wealth_history = []

    @classmethod
    def from_pretrained_ffn(
        cls,
        ffn_module: nn.Module,
        config: MoBConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MixtureOfBidders":
        """
        Initialize MoB by upcycling from a pretrained FFN.

        Args:
            ffn_module: The original FFN module (e.g., from Mistral)
            config: MoB configuration
            device: Target device (auto-detected from ffn_module if None)
            dtype: Target dtype (auto-detected from ffn_module if None)

        Returns:
            Initialized MoB module with weights copied from FFN
        """
        if device is None or dtype is None:
            if hasattr(ffn_module, "gate_proj"):
                ref_param = cast(nn.Linear, ffn_module.gate_proj).weight
            elif hasattr(ffn_module, "up_proj"):
                ref_param = cast(nn.Linear, ffn_module.up_proj).weight
            else:
                ref_param = next(ffn_module.parameters())

            if device is None:
                device = ref_param.device
                if device.type == "meta":
                    device = (
                        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                    )
                    logger.warning(
                        f"Detected meta device from lazy loading, forcing device={device}"
                    )
            if dtype is None:
                dtype = ref_param.dtype

        mode_str = "shared-base" if config.use_shared_base else "full-expert"
        logger.info(f"Creating MoB ({mode_str}) on device={device}, dtype={dtype}")

        mob = cls(config)

        with torch.no_grad():
            if config.use_shared_base:
                if hasattr(ffn_module, "gate_proj"):
                    ffn_gate = cast(nn.Linear, ffn_module.gate_proj)
                    ffn_up = cast(nn.Linear, ffn_module.up_proj)
                    ffn_down = cast(nn.Linear, ffn_module.down_proj)
                    mob.base_gate_proj.weight.copy_(ffn_gate.weight.cpu())
                    mob.base_up_proj.weight.copy_(ffn_up.weight.cpu())
                    mob.base_down_proj.weight.copy_(ffn_down.weight.cpu())

                for i, expert_mod in enumerate(mob.experts):
                    lw = cast(LightweightExpert, expert_mod)
                    lw.gate_adapter_A.weight.add_(
                        torch.randn_like(lw.gate_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
                    lw.up_adapter_A.weight.add_(
                        torch.randn_like(lw.up_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
                    lw.down_adapter_A.weight.add_(
                        torch.randn_like(lw.down_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
            else:
                ffn_gate_w: torch.Tensor | None = None
                ffn_up_w: torch.Tensor | None = None
                ffn_down_w: torch.Tensor | None = None
                if hasattr(ffn_module, "gate_proj"):
                    ffn_gate_w = cast(nn.Linear, ffn_module.gate_proj).weight.cpu()
                    ffn_up_w = cast(nn.Linear, ffn_module.up_proj).weight.cpu()
                    ffn_down_w = cast(nn.Linear, ffn_module.down_proj).weight.cpu()

                for expert_mod in mob.experts:
                    exp = cast(Expert, expert_mod)
                    if ffn_gate_w is not None and ffn_up_w is not None and ffn_down_w is not None:
                        exp.gate_proj.weight.copy_(ffn_gate_w)
                        exp.up_proj.weight.copy_(ffn_up_w)
                        exp.down_proj.weight.copy_(ffn_down_w)

                    exp.gate_proj.weight.add_(
                        torch.randn_like(exp.gate_proj.weight) * config.jitter_std
                    )
                    exp.up_proj.weight.add_(
                        torch.randn_like(exp.up_proj.weight) * config.jitter_std
                    )
                    exp.down_proj.weight.add_(
                        torch.randn_like(exp.down_proj.weight) * config.jitter_std
                    )

        mob = mob.to(device=device, dtype=dtype)

        logger.info(
            f"Upcycled FFN to MoB with {config.num_experts} experts, "
            f"top-k={config.top_k}, mode={mode_str}"
        )
        return mob


def apply_mob_to_model(
    model: nn.Module,
    mob_config: MoBConfig,
    layers_to_modify: list[int] | None = None,
) -> nn.Module:
    layers: nn.ModuleList | None = None
    if hasattr(model, "model"):
        model_inner = cast(nn.Module, model.model)
        if hasattr(model_inner, "layers"):
            layers = cast(nn.ModuleList, model_inner.layers)
    if layers is None and hasattr(model, "layers"):
        layers = cast(nn.ModuleList, model.layers)
    if layers is None:
        raise ValueError("Cannot find transformer layers in model")

    num_layers = len(layers)
    if layers_to_modify is None:
        layers_to_modify = list(range(4, num_layers - 4))

    for layer_idx in layers_to_modify:
        layer = layers[layer_idx]

        ffn_attr: str | None = None
        if hasattr(layer, "mlp"):
            ffn_attr = "mlp"
        elif hasattr(layer, "feed_forward"):
            ffn_attr = "feed_forward"
        else:
            logger.warning(f"Layer {layer_idx}: Cannot find FFN module, skipping")
            continue

        ffn = cast(nn.Module, getattr(layer, ffn_attr))
        mob = MixtureOfBidders.from_pretrained_ffn(ffn, mob_config)
        setattr(layer, ffn_attr, mob)

        logger.info(f"Layer {layer_idx}: Replaced FFN with MoB")

    return model
