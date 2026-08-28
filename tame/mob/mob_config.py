from dataclasses import dataclass

from .auction import ROUTING_SHARE_UNIFORM, SUPPORTED_ROUTING_SHARES


@dataclass
class MoBConfig:
    """Configuration for Mixture of Bidders module."""

    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    initial_wealth: float = 75.0
    wealth_decay: float = 0.997
    min_wealth: float = 15.0
    max_wealth: float = 750.0
    jitter_std: float = 0.08
    reward_scale: float = 2.0
    use_vcg_payments: bool = True
    # Dimensionless deviation from the balanced transfer, not a unit conversion.
    # Reward and charge share one coefficient derived from reward_scale, the path's
    # reward multiplier and top_k, so 1.0 is the quasi-linear point the VCG results
    # require and anything else deliberately over- or under-prices the auction.
    # scripts/sweep_payment_scale.py sweeps around it.
    payment_scale: float = 1.0
    use_shared_base: bool = True
    adapter_rank: int = 64
    adapter_alpha: float = 16.0
    use_loss_feedback: bool = True
    use_local_quality: bool = True
    # "uniform" splits the output 1/top_k across winners, which is what makes the
    # auction strategyproof and keeps the language-modelling loss out of the
    # confidence heads. "softmax" restores the own-bid-weighted gate as the
    # gate-swap baseline; use_differentiable_routing only applies in that mode.
    routing_share: str = ROUTING_SHARE_UNIFORM
    use_differentiable_routing: bool = True
    confidence_calibration_weight: float = 0.15
    confidence_z_loss_weight: float = 0.0001
    loss_ema_decay: float = 0.92
    inference_wealth_decay: float = 0.98
    inference_exploration_bonus: float = 0.03
    inference_wealth_compression: float = 0.4

    def __post_init__(self) -> None:
        # The auction divides each winner's externality by its own wealth to price
        # it in the winner's own units. A non-positive wealth makes that division
        # meaningless, and the clamp guarding it would turn a valid numerator into
        # an astronomically large price with no invariant firing.
        if self.min_wealth <= 0:
            raise ValueError(f"min_wealth must be positive, got {self.min_wealth}")
        if self.initial_wealth <= 0:
            raise ValueError(f"initial_wealth must be positive, got {self.initial_wealth}")
        # An inverted band is worse than a merely odd one: clamp_(min=15, max=-5)
        # returns -5, so every clamp that exists to keep wealth positive would write
        # a negative wealth instead -- the one way the auction's "no writer can
        # produce it" could be false.
        if self.max_wealth < self.min_wealth:
            raise ValueError(
                f"max_wealth ({self.max_wealth}) must be at least min_wealth ({self.min_wealth})"
            )
        # Otherwise every expert starts outside the band and the first wealth update
        # yanks them all to a bound -- a step-zero discontinuity that reads as a
        # training artefact rather than a config error.
        if not self.min_wealth <= self.initial_wealth <= self.max_wealth:
            raise ValueError(
                f"initial_wealth ({self.initial_wealth}) must lie within "
                f"[{self.min_wealth}, {self.max_wealth}]"
            )

        if self.routing_share not in SUPPORTED_ROUTING_SHARES:
            shares = ", ".join(sorted(SUPPORTED_ROUTING_SHARES))
            raise ValueError(
                f"Unsupported routing share '{self.routing_share}'. Supported: {shares}"
            )
