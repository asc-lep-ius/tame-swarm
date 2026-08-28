import logging
from dataclasses import MISSING, Field, dataclass, fields
from typing import Any

from .auction import ROUTING_SHARE_UNIFORM, SUPPORTED_ROUTING_SHARES
from .softmax_router import ROUTER_AUCTION, SUPPORTED_ROUTERS

logger = logging.getLogger(__name__)

# Fields nothing reads unless the auction gate is the one running. Two groups: the
# economy proper, and the auction's own share/sharpness parameters, which are passed
# to VCGAuctioneer and have no counterpart in SoftmaxRouter. Tuning any of them under
# the softmax gate is otherwise silent -- the defect class #12 exists to remove.
#
# ``jitter_std`` is deliberately absent: it perturbs the expert adapters in
# ``from_pretrained_ffn``, which every arm runs, so it steers the control arm too.
AUCTION_ONLY_FIELDS = (
    "initial_wealth",
    "wealth_decay",
    "min_wealth",
    "max_wealth",
    "reward_scale",
    "use_vcg_payments",
    "payment_scale",
    "use_loss_feedback",
    "use_local_quality",
    "loss_ema_decay",
    "inference_wealth_decay",
    "inference_exploration_bonus",
    "inference_wealth_compression",
    "routing_share",
    "routing_temperature",
    "use_differentiable_routing",
    # Reached only below the has_economy early return in update_wealth_from_loss, so
    # under the softmax gate the cached loss stays None and the trainer adds zero.
    "confidence_calibration_weight",
)


def _declared_default(spec: "Field[Any]") -> Any:
    """The default a field was declared with, however it was declared."""
    if spec.default is not MISSING:
        return spec.default
    if spec.default_factory is not MISSING:
        return spec.default_factory()
    return MISSING


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
    # confidence heads. "proportional" restores an own-bid-weighted gate as the
    # gate-swap baseline; use_differentiable_routing only applies in that mode.
    routing_share: str = ROUTING_SHARE_UNIFORM
    use_differentiable_routing: bool = True
    # Sharpness of the "proportional" gate, applied in the log domain: a winner's
    # share is bid ** (1 / routing_temperature), normalised over the winners. 1.0 is
    # plain bid-proportional and is the default because it introduces no constant
    # that has to be re-tuned when anything else moves. Below 1.0 approaches argmax,
    # above 1.0 approaches the uniform split; every value is invariant to a uniform
    # rescaling of wealth, so this is a sharpness choice and not, as the raw bid
    # scale was, a sharpness side effect. Exact in the algebra, and measured under
    # 1e-6 in float32 down to tau=0.1 -- see _log_bids, which normalises before the
    # log precisely so that bound does not degrade as the gate sharpens. Ignored
    # under the uniform share.
    routing_temperature: float = 1.0
    # Which gate turns reports into an allocation. "auction" is MoB. "softmax" is
    # the #12 control arm: the same confidence heads, softmaxed, with the whole
    # economy switched off -- no wealth read, no payment, no rebate, no value
    # objective. It is not a variant of the mechanism, it is the thing the
    # mechanism is being compared against, so every economy path checks
    # has_economy rather than assuming there is one.
    router: str = ROUTER_AUCTION
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

        # Zero divides, and a negative temperature inverts the ranking so the gate
        # would hand the largest share to the expert that bid least.
        if self.routing_temperature <= 0:
            raise ValueError(
                f"routing_temperature must be positive, got {self.routing_temperature}"
            )

        if self.routing_share not in SUPPORTED_ROUTING_SHARES:
            shares = ", ".join(sorted(SUPPORTED_ROUTING_SHARES))
            raise ValueError(
                f"Unsupported routing share '{self.routing_share}'. Supported: {shares}"
            )

        if self.router not in SUPPORTED_ROUTERS:
            routers = ", ".join(sorted(SUPPORTED_ROUTERS))
            raise ValueError(f"Unsupported router '{self.router}'. Supported: {routers}")

        self._warn_about_ignored_auction_settings()

    def _warn_about_ignored_auction_settings(self) -> None:
        """Say so when an auction-only field was tuned under a gate that never reads it."""
        if self.has_economy:
            return

        defaults = {spec.name: _declared_default(spec) for spec in fields(self)}
        ignored = [name for name in AUCTION_ONLY_FIELDS if getattr(self, name) != defaults[name]]
        if ignored:
            logger.warning(
                f"Router '{self.router}' does not run the auction, so {', '.join(ignored)} "
                "will not be read; none of them affect this arm"
            )

    @property
    def has_economy(self) -> bool:
        """Whether wealth, payments and the value objective are live.

        The control arm shares this config object with the auction arm rather than
        having its own, so every economy path is guarded on this one predicate.
        ``__post_init__`` warns when an auction-only field is set away from its
        default under the softmax gate, because a silently-ignored
        ``use_vcg_payments=True`` would be the same class of defect as the
        ``eval_steps`` that #12 was opened over: a field that looks like it steers
        the experiment and does not.
        """
        return self.router == ROUTER_AUCTION
