"""The steering-economy coupling: the goal direction offered to routing as a perceived feature.

#2's design is perception modulation. The field does not bias bids after the fact;
it shifts what every confidence head *sees* before it reports, so a head that
finds the goal direction valuable bids higher on its own and the auction runs
unchanged. Per token,

    perceived = h + beta_effective * (u . h) * d

where ``d`` is the goal direction the residual-stream hooks inject at this layer
and ``u`` is the *receptor*: the one learned vector that says which local feature
of the stream this layer reads as "the field is here". #14 replaced #2's
``hidden_dim x hidden_dim`` projection with ``u`` because the projection only ever
entered the forward through its product with ``d`` -- the signal was
``h . (W^T d)`` -- and every gradient from the zero initialisation lay in
``span(d) (x) R^n``. The two are the same function class; this one names the
object being learned and costs a dot product per token instead of a matmul.

The delta is capped at ``max_coupling_fraction`` of the hidden norm per token, and
ramps in over ``warmup_steps`` so the coupling arrives in the routing dynamics as a
gradient rather than a step. It is zero-initialised: at day zero the perceived
stream is the stream, exactly as upcycling preserves the FFN. That makes an inert
coupling indistinguishable from a fresh one, which is why the receptor's norm and
``beta_effective`` are both reported, a training-mode forward refuses to run until
the step has been set, and the tests assert the mechanism where it is non-trivial.
"""

import logging
import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_COUPLING_BETA = 0.1
# #2's specification. The ramp is what keeps the coupling from arriving in the
# routing dynamics as a step change; the value of 1 that shipped had removed it.
DEFAULT_WARMUP_STEPS = 100
# Below this many steps the ramp has too few distinct levels to be one: the
# economy settles once per step, so a warmup of a handful of steps is a step
# change with extra bookkeeping. Warned about, not rejected -- a fixture that wants
# the coupling at full strength at once says so on purpose.
MIN_WARMUP_STEPS_FOR_RAMP = 10
DEFAULT_MAX_COUPLING_FRACTION = 0.1
DEFAULT_EPS = 1e-8
SUPPORTED_COUPLING_MODES = frozenset({"perception"})


@dataclass(frozen=True)
class SteeringCouplingConfig:
    hidden_dim: int
    coupling_beta: float = DEFAULT_COUPLING_BETA
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    max_coupling_fraction: float = DEFAULT_MAX_COUPLING_FRACTION
    mode: str = "perception"
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.coupling_beta < 0.0:
            raise ValueError("coupling_beta must be non-negative")
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if self.max_coupling_fraction < 0.0:
            raise ValueError("max_coupling_fraction must be non-negative")
        if self.mode not in SUPPORTED_COUPLING_MODES:
            modes = ", ".join(sorted(SUPPORTED_COUPLING_MODES))
            raise ValueError(f"Unsupported coupling mode '{self.mode}'. Supported modes: {modes}")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.warmup_steps < MIN_WARMUP_STEPS_FOR_RAMP:
            logger.warning(
                "warmup_steps=%d is below %d: beta_effective reaches coupling_beta within "
                "%d steps, so the coupling arrives as a step change rather than a ramp",
                self.warmup_steps,
                MIN_WARMUP_STEPS_FOR_RAMP,
                self.warmup_steps,
            )


@dataclass(frozen=True)
class CouplingMetrics:
    active: torch.Tensor
    beta_effective: torch.Tensor
    delta_norm_fraction_mean: torch.Tensor
    delta_norm_fraction_max: torch.Tensor
    steering_alignment_mean: torch.Tensor
    # The receptor's norm: the one number that separates a coupling that has
    # learned something from one that is attached and inert.
    detector_norm: torch.Tensor


class SteeringCoupling(nn.Module):
    def __init__(self, config: SteeringCouplingConfig, steering_direction: torch.Tensor):
        super().__init__()
        self.config = config
        self.detector = nn.Parameter(torch.zeros(config.hidden_dim))

        self.steering_direction: torch.Tensor
        self.register_buffer(
            "steering_direction",
            self._normalize_direction(steering_direction, config),
            persistent=True,
        )

        self._coupling_step: torch.Tensor
        self.register_buffer(
            "_coupling_step",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )
        # Whether anything has ever set the step. A coupling whose counter never
        # advances is a no-op that looks attached, so a training forward refuses
        # to run in that state rather than route uncoupled under a coupled name.
        self._step_was_set = False

        self.last_metrics: CouplingMetrics | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"Expected hidden size {self.config.hidden_dim}, got {hidden_states.shape[-1]}"
            )
        if self.training and not self._step_was_set:
            raise RuntimeError(
                "SteeringCoupling forwarded in training mode before set_coupling_step was "
                "called: the warmup would never advance and the coupling would stay inert"
            )

        beta_effective = self._effective_beta(hidden_states)
        direction = self.steering_direction.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        direction_view = direction.view(*([1] * (hidden_states.ndim - 1)), -1)

        steering_signal = (hidden_states @ self.detector.to(hidden_states.dtype)).unsqueeze(-1)
        raw_delta = beta_effective * steering_signal * direction_view
        delta = self._cap_delta(raw_delta, hidden_states)

        self.last_metrics = self._build_metrics(hidden_states, delta, direction, beta_effective)
        return hidden_states + delta

    def set_coupling_step(self, step: int) -> "SteeringCoupling":
        if step < 0:
            raise ValueError("coupling step must be non-negative")
        self._coupling_step.fill_(step)
        self._step_was_set = True
        self.last_metrics = None
        return self

    def _effective_beta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        step = self._coupling_step.to(device=hidden_states.device, dtype=hidden_states.dtype)
        warmup_steps = torch.full_like(step, float(self.config.warmup_steps))
        warmup_fraction = (step / warmup_steps).clamp(max=1.0)
        return warmup_fraction * self.config.coupling_beta

    def _cap_delta(self, delta: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_norm = hidden_states.norm(dim=-1, keepdim=True)
        delta_norm = delta.norm(dim=-1, keepdim=True)
        max_delta_norm = self.config.max_coupling_fraction * hidden_norm
        scale = torch.where(
            delta_norm > max_delta_norm,
            max_delta_norm / delta_norm.clamp_min(self.config.eps),
            torch.ones_like(delta_norm),
        )
        return delta * scale

    def _build_metrics(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        direction: torch.Tensor,
        beta_effective: torch.Tensor,
    ) -> CouplingMetrics:
        with torch.no_grad():
            detached_hidden = hidden_states.detach()
            detached_delta = delta.detach()
            hidden_norm = detached_hidden.norm(dim=-1)
            delta_norm = detached_delta.norm(dim=-1)
            delta_fraction = torch.where(
                hidden_norm > 0,
                delta_norm / hidden_norm.clamp_min(self.config.eps),
                torch.zeros_like(hidden_norm),
            )

            direction_float = direction.detach().float()
            direction_view = direction_float.view(*([1] * (detached_hidden.ndim - 1)), -1)
            alignment = F.cosine_similarity(
                detached_hidden.float(),
                direction_view.expand_as(detached_hidden.float()),
                dim=-1,
                eps=self.config.eps,
            )
            detector_norm = self.detector.detach().float().norm()

        return CouplingMetrics(
            active=(beta_effective.detach() > 0.0),
            beta_effective=beta_effective.detach(),
            delta_norm_fraction_mean=delta_fraction.mean().detach(),
            delta_norm_fraction_max=delta_fraction.max().detach(),
            steering_alignment_mean=alignment.mean().detach(),
            detector_norm=detector_norm,
        )

    @staticmethod
    def _normalize_direction(
        steering_direction: torch.Tensor,
        config: SteeringCouplingConfig,
    ) -> torch.Tensor:
        direction = steering_direction.detach().clone().float().reshape(-1)
        if direction.numel() != config.hidden_dim:
            raise ValueError(
                f"Expected steering direction with {config.hidden_dim} values, "
                f"got {direction.numel()}"
            )

        norm = float(direction.norm().item())
        if not math.isfinite(norm) or norm <= config.eps:
            return torch.zeros_like(direction)
        return direction / norm


def attach_coupling(
    module: nn.Module,
    steering_direction: torch.Tensor,
    config: SteeringCouplingConfig | None = None,
) -> SteeringCoupling:
    attach = getattr(module, "attach_coupling", None)
    if not callable(attach):
        raise TypeError("module does not support coupling attachment")
    return cast(SteeringCoupling, attach(steering_direction, config))


def detach_coupling(module: nn.Module) -> None:
    detach = getattr(module, "detach_coupling", None)
    if not callable(detach):
        raise TypeError("module does not support coupling detachment")
    detach()
