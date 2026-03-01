from dataclasses import dataclass


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
    use_shared_base: bool = True
    adapter_rank: int = 64
    adapter_alpha: float = 16.0
    use_loss_feedback: bool = True
    use_local_quality: bool = True
    use_differentiable_routing: bool = True
    confidence_calibration_weight: float = 0.15
    loss_ema_decay: float = 0.92
    inference_wealth_decay: float = 0.98
    inference_exploration_bonus: float = 0.03
    inference_wealth_compression: float = 0.4
