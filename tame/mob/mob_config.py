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
    "exploration_rate",
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
    # The four wealth constants, re-derived under the economy #9, #11 and #15 left
    # (#16). All four are *retained*; `scripts/sweep_wealth_bounds.py` re-runs the
    # evidence, and two of the reasons are that the constant does less than it
    # looks like it does.
    #
    # What they shape, now that #11 has taken wealth out of gate sharpness:
    # selection (winners are argtopk(confidence x wealth), so the band's *ratio* is
    # the report advantage the market demands of its poorest expert) and, through
    # the wealth spread, the size of the prices and rebates a given step produces.
    # Prices and rebates are not shaped by the band's *scale*: a price is
    # b_(k+1)/w_j with b_(k+1) itself proportional to a wealth, so it is a ratio of
    # wealths and exactly invariant to rescaling the ledger -- measured unchanged
    # to 7 significant figures at 0.1x and 10x. The reward is the one quantity
    # carrying no wealth at all, so the inflow is an absolute number of credits.
    #
    # **What decides the retention: the share a shut-out expert holds is a closed
    # form the band cannot reach.** The exploration slot is drawn uniformly over the
    # losers *before any report is read*, so a shut-out expert's expected share is
    # exploration_rate / top_k / (num_experts - top_k) = 0.02/2/6 = 0.0017 whatever
    # the band is. Measured 0.0009-0.0046 across bands and seeds. No setting of
    # these four constants can move it, which is why #16 changed none of them and
    # #26 is about the exploration slot.
    #
    # The concentration itself is *not* evidence of a defect, and the earlier
    # drafts of this comment read it as though it were. On the planted-competence
    # fixture an expert closes the same fraction of the gap on every token it holds
    # (see scripts/synthetic_economy.py), so competence is token-independent, there
    # is nothing to specialise on, and the efficient allocation is always the same
    # top_k experts. A two-expert market is the *optimum* here. What the fixture can
    # say is whether the mechanism picks the right two and whether anyone else can
    # ever get back in -- not whether concentration is bad.
    #
    # On picking the right two the band earns its keep, and the shipped one wins:
    # at [15, 750] the two monopolists are the top two by competence on all three
    # seeds, while at [37.5, 150] and [23.7, 237.2] they come out as ranks {4,2},
    # {2,1}, {3,2} and {4,1}. Narrowing the band makes the allocation *less*
    # competence-ordered, not more.
    #
    # Everything else the band moves is bookkeeping. At decay 0.997 it changes only
    # where the six losers' wealth sits: on the floor at [15, 750] (74% floor
    # occupancy, 0.7% of expert-steps in the band's interior), in the interior at
    # [37.5, 150] (0% floor, 75% interior). That takes wealth Gini from 0.693 to
    # 0.124 and leaves the same six on the exploration slot.
    #
    # Ceiling occupancy is top_k/num_experts -- 25.0% -- at every ratio swept *at
    # this decay*, because the equilibrium sits above every ceiling swept and the
    # two winners are the only experts clamped there. That is the shipped decay's
    # number, not a universal: across the grid ceiling occupancy runs 0.0% to 100.0%.
    #
    # The concentration column varies across the decay rows too -- three to eight
    # experts clear 1% of the slots at 0.98 against two at 0.997 -- but that
    # comparison is confounded and the sweep contains its own control. The budget
    # scales with the decay (eight memory horizons), so the fast rows are also the
    # short ones; and at ratio 1 the ledger is pinned by construction, so decay
    # cannot act at all, yet `win>1%` still falls from 7.7 to 2.3 as the budget goes
    # from 600 to 2667 steps. The heads keep calibrating long after the ledger has
    # settled, so concentration tracks training time rather than decay. Anything
    # read off a report -- concentration, overturn, the report advantage -- is not
    # comparable across those rows.
    #
    # So Gini here is a closed form rather than a measurement: 0.693 is exactly the
    # Gini of (750, 750, 15 x 6), which is why its spread over three seeds is
    # +-0.000. Read it beside interior occupancy, never alone.
    #
    # Sets the transient, and the settled wealth *distribution* is not its to move.
    # Holding this band and moving only the start from 25 to 750 leaves the settled
    # Gini at 0.691/0.693/0.693/0.692 and the mean wealth at 199 in every case, and
    # the monopoly is the same two experts on 99% of slots throughout. What does
    # move is how much of the tail the economy spends off its bounds -- interior
    # occupancy runs 0.7% to 24.7% across those starts -- and time-to-first-clamp,
    # 343 steps against 1. The equilibrium is a fixed point of
    # `w = decay*w + net(w)` and is an absolute wealth, so what has to contain it
    # is the *band*, not the starting point. The constraint on this constant is
    # therefore only that it sit inside the band and away from either bound, so a
    # run does not begin clamped: at 750 every expert starts pinned to the ceiling
    # with a 333-step decay away from it, which is why that mutant breaks seven
    # tests that run 200-600 steps.
    initial_wealth: float = 75.0
    # Bounded above and below by two different tests, and the margins are narrow
    # enough to state exactly. Above: at 1.0 the ledger never forgets, and
    # `test_the_market_re_forms_around_a_senescent_expert` fails -- on one seed of
    # three, on the share statistic, 0.0191 against the 0.01 ceiling. Below: at
    # 0.995 `test_frozen_heads_leave_the_senescent_expert_in_the_market` fails by
    # 1% on the loss comparison (0.990 of its gate) while the dead expert still
    # plainly holds the market at 4.9% of slots; only at 0.99 does drainage
    # actually remove it, at 0.56%. So the honest statement is not "below 0.997
    # decay does the value objective's job" but "0.997 is the nearest value at
    # which that pairing is unambiguous" -- and the pairing is what makes #15's
    # senescence claim attributable to the objective rather than to the ledger
    # draining. 1/(1 - decay) = 333 steps of memory.
    wealth_decay: float = 0.997
    # A guard, and inert as economics -- which is the finding, not an omission.
    # The auction prices an externality in the winner's own units by dividing by
    # its wealth, and a non-positive wealth makes that meaningless; __post_init__
    # rejects one and this keeps every writer clear. Beyond that it does nothing
    # measurable: lowered 15000x, no behavioural assertion in the repository fails;
    # raised 5x, so that a ruined expert is restored to a full initial_wealth by
    # the next clamp, its win share stays at 0.0012-0.0019, a hundredfold below
    # chance. Its height decides neither the healthy economy nor recovery from the
    # one damage protocol it might have been expected to govern. What the floor
    # does participate in is the band's *ratio*, held by
    # `test_the_band_ratio_bounds_the_report_advantage_demanded_of_the_poorest`;
    # no test pins this value itself, deliberately, because there is nothing true
    # left to pin it with.
    min_wealth: float = 15.0
    # Retained as the bound on the domain over which the auction's payment
    # properties are asserted, and honestly that is all the evidence supports.
    # Raised to 1e9 it breaks both coupling property tests and
    # `test_payments_are_strictly_positive_whenever_a_bid_is_displaced`
    # -- but those tests draw their
    # markets log-uniformly from `WEALTH_BAND`, which *is* this constant, so
    # widening it widens the tests' own input distribution until differencing two
    # welfare sums loses the float32 precision that keeps a displaced price above
    # zero. That is a real limit on how far apart two wealths may be, and it is
    # also self-referential in the way #16 discounted the value-pinning
    # `test_default_values_match_expected`
    # for; no production path reads this constant except the clamps.
    #
    # What it is *not* is a lever on the monopoly. Narrowing the band to [37.5, 150]
    # takes wealth Gini from 0.693 to 0.124 and how often wealth overturns a report
    # from 26.6% to 1.0%, and leaves two experts holding 99% of the slots with the
    # other six shut out -- it relocates the losers' wealth into the band's interior
    # without readmitting them, and the two it leaves in charge are a worse-chosen
    # pair by competence than the shipped band's. The limit case is
    # unambiguous the other way: at decay 1.0 every band whose bounds differ ends
    # with 100% of expert-steps at the ceiling, every expert equally and maximally
    # rich. Neither end is a cap on inequality, which is why #16 changed no value
    # here and #26 is about the exploration slot instead.
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
    # Fraction of training tokens whose last slot goes to a random loser instead
    # of being sold -- see VCGAuctioneer. A head learns only from the tokens its
    # expert holds, so without this an expert whose truthful report has fallen to
    # zero never holds another token and never recovers. 0.02 is a floor, not a
    # tuning: at eight experts and top-2 it hands each loser about one token in
    # 300, enough for every head to keep a target every step at a real batch
    # size, while displacing the marginal winner on 2% of tokens.
    exploration_rate: float = 0.02
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

        if not 0.0 <= self.exploration_rate < 1.0:
            raise ValueError(f"exploration_rate must lie in [0, 1), got {self.exploration_rate}")

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
