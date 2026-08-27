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
    # Converts VCG payments into reward units for the quasi-linear wealth update.
    # Prices are denominated in an expert's own value units -- the weighted
    # externality divided by its own wealth -- so they are far smaller than the
    # bid-unit prices the previous 0.02 was swept for. Re-swept over 400 steps x 3
    # seeds: 0.3 charges ~8% of reward flow, matching the fraction the bid-unit
    # sweep settled on, while >=1.5 collapses the wealth spread on every seed.
    # Between 0.05 and 1.4 the seed-to-seed spread in Gini exceeds the difference
    # between scales, so this is pinned by the charge fraction, not by Gini.
    payment_scale: float = 0.3
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
        if self.routing_share not in SUPPORTED_ROUTING_SHARES:
            shares = ", ".join(sorted(SUPPORTED_ROUTING_SHARES))
            raise ValueError(
                f"Unsupported routing share '{self.routing_share}'. Supported: {shares}"
            )
