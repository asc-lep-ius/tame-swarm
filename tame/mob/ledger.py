"""One ledger class, and the three plugs that make it serve two scales (#40).

Every wealth path in this project settles the same way -- relax the ledger, pay
what the step earned, charge what the auction priced, hold the result at a floor
-- and until #40 each of the three wrote that sequence out again. The defects #9
(a charge that scaled the reward instead of transferring against it) and #16
(constants that turned out to do less than they looked like they did) were each
paid for once and would have been paid for again by a fourth copy, which is what
the organism's budget ledger (#43) would have been.

So the sequence is :class:`WealthUpdater` and the differences are plugs: a
**reward signal** (what the step earned), a **floor** (what happens to a ledger
that runs out), and a **mode** -- whether the relaxation pulls the ledger toward
zero (``decay``, the economy as recorded) or toward a setpoint (``setpoint``,
#26's ledger, derived here and not adopted). #39's shadow ledger is a mode too:
under ``decoupled`` the ledger settles exactly as it always did and the gate
reads a pinned wealth instead of it.

Why modes rather than more constants: with the relaxation written as one rate,
the ledger's fixed point and the condition for reaching it monotonically are
closed forms, so the constants a second ledger needs are *derived* rather than
tuned. README ``#ledger-stability`` carries the derivation and what it fixes;
:meth:`WealthUpdater.equilibrium` and :meth:`WealthUpdater.cannot_oscillate`
are the two formulas themselves.

``tame/mob/wealth.py`` is the MoB layer's side of this: the value hook, the VCG
charge and the caches a settlement is assembled from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from .mob_config import MoBConfig

# What each reward signal's inflow is multiplied by. Per signal rather than one
# constant because the charge has to carry the reward's own scale for the two to
# be a single quasi-linear utility (``WealthUpdateMixin._transfer_coefficient``).
LOSS_REWARD_MULTIPLIER = 50.0
LOCAL_REWARD_MULTIPLIER = 5.0
PARTICIPATION_REWARD_MULTIPLIER = 10.0
WEALTH_EPSILON = 1e-6


# The stakes dial (#39): whether a cell's continuation depends on its realised
# value. "value" is the economy as it has always run -- the gate reads the ledger,
# so what a cell reports changes whether it keeps holding tokens. "decoupled" is
# the control: the gate reads ``initial_wealth`` for every expert, so relative
# wealth is equal and the allocation is the report ranking alone; the exploration
# draw is uniform, so re-entry owes nothing to the ledger either; and the ledger
# itself still settles every step -- payments, rebates, rewards, the clamp -- as a
# **shadow ledger**, the wealth a cell would have had, logged and never read by
# the gate. The head's value regression is untouched: the cell perceives value and
# is not paid in continuation, which is the whole experiment. "shuffled" keeps the
# economy live and permutes the regression's targets across experts each step,
# so the cell is paid in continuation for a value signal that is noise about
# itself -- the control for a signature that is really the head's regression.
PERSISTENCE_VALUE = "value"
PERSISTENCE_DECOUPLED = "decoupled"
PERSISTENCE_SHUFFLED = "shuffled"
SUPPORTED_PERSISTENCE_COUPLINGS = frozenset(
    {PERSISTENCE_VALUE, PERSISTENCE_DECOUPLED, PERSISTENCE_SHUFFLED}
)

# What the ledger relaxes toward between settlements (#40). "decay" is the economy
# as recorded: wealth is multiplied by ``wealth_decay`` every step, so an expert
# that earns nothing sinks toward zero and is held off it by the floor alone.
# "setpoint" is #26's ledger as a mode of the same arithmetic -- the ledger
# relaxes toward ``initial_wealth`` at the same rate -- which moves the ruin
# threshold down rather than moving the floor up. Measured in README
# ``#ledger-stability``; ``decay`` remains the default, and nothing recorded was
# run under anything else.
LEDGER_DECAY = "decay"
LEDGER_SETPOINT = "setpoint"
SUPPORTED_LEDGER_MODES = frozenset({LEDGER_DECAY, LEDGER_SETPOINT})
# --- One settlement, three plugs (#40) ---------------------------------------------------


@dataclass(frozen=True)
class Settlement:
    """What one step handed the ledger: the holdings, and what the auction priced them at.

    Everything a reward signal is allowed to read. Deliberately not the ledger
    itself -- that is passed beside it, because the ledger is what a settlement
    *moves* and the class exists so that a second ledger at a second scale (#43)
    can be moved by the same code. ``valid_mask`` is ``None`` on the paths that
    never see a token mask, and is what both the reward and the charge drop
    padding with when it is not.
    """

    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    confidences: torch.Tensor
    num_tokens: int
    payments: torch.Tensor | None = None
    rebates: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    # The loss path's realised values, and the local-quality path's layer output:
    # each is the one quantity its own signal prices, and neither exists on the
    # other paths.
    values: torch.Tensor | None = None
    output: torch.Tensor | None = None
    # #59's transmitted stress for this step, per token, or None on a layer that
    # transmits none. Not per expert: every cell at the layer is charged the
    # same, which is the mechanism rather than an implementation shortcut.
    stress: torch.Tensor | None = None
    # #60's Brier score per expert, over the tokens it held: negative, bounded,
    # and zero for a cell that held nothing. Per *expert*, unlike the stress,
    # because a self-model is exactly the thing a cell owns.
    self_score: torch.Tensor | None = None
    # How often each expert has held a token. A ledger-side count rather than a
    # step quantity, read only by the inference path's re-entry gift.
    usage_count: torch.Tensor | None = None


class RewardSignal(Protocol):
    """What a cell is paid for.

    The invariant every signal plugged in here carries, and the one #33 settled:
    **a reward is a stress, never a direction score.** What the cell is paid is
    the counterfactual reduction in an error the tissue is holding -- what would
    have gone wrong without this cell, minus what went wrong with it -- so a cell
    that does not move the reading earns exactly zero at any error, a cell that
    pushes a tissue already at its setpoint *off* it is charged, and a cell
    aligned past the setpoint earns less than one that stopped at it. A score on
    how well a contribution points along a goal direction has none of those three
    properties: it pays for alignment whether or not there was anything to fix,
    and it pays more the harder the cell pushes.

    #43 and #44 plug their own signals in here. A signal that declares a
    ``setpoint`` and a ``reduction`` is probed at construction and refused if it
    is a direction score -- see :func:`refuse_a_direction_score`, which is where
    the wrong shape fails.

    ``multiplier`` is the path's own reward constant. It is per signal rather than
    per config because the charge has to carry the reward's scale for the two to
    be one quasi-linear utility, and a single constant would only be quasi-linear
    for whichever path it was derived from.
    """

    @property
    def multiplier(self) -> float:
        """What this signal's inflow is scaled by, and what its charge is priced at."""
        ...

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        """Per-expert inflow, in the ledger's units, against a ledger already relaxed."""
        ...

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        """An inflow that prices no holding, paid after the transfer rather than in it.

        ``None`` for a signal that pays only for what was held. The inference
        path's exploration bonus is the one that is not: it is a credit to an
        under-used cell, which is the shape #43's dormancy rule will want, and it
        is added after the charge because it is not part of the trade.
        """
        ...


class Floor(Protocol):
    """What happens to a ledger that runs out.

    A plug rather than a constant because the two scales want different things
    from it. The expert ledger clamps into a band (:class:`BandFloor`): the floor
    is an absorbing barrier that keeps the auction's division by a winner's wealth
    meaningful, and README ``#ledger-stability`` derives why its *height* decides
    almost nothing. The organism's budget (#43) wants a floor that also declares
    dormancy, which is a different rule over the same arithmetic.
    """

    def __call__(self, wealth: torch.Tensor) -> None:
        """Hold ``wealth`` inside whatever this floor admits, in place."""
        ...


class Charge(Protocol):
    """The VCG transfer, as the updater asks for it: see ``WealthUpdateMixin._vcg_charges``."""

    def __call__(
        self,
        payments: torch.Tensor | None,
        selected_experts: torch.Tensor,
        num_tokens: int,
        reward_multiplier: float,
        rebates: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class BandFloor:
    """The expert ledger's floor: clamp into ``[minimum, maximum]``.

    Both bounds, because an inverted band is worse than a merely odd one and the
    clamp is the one writer that could produce a negative wealth. ``MoBConfig``
    refuses an inverted band at construction; this is what enforces it every step.
    """

    minimum: float
    maximum: float

    def __call__(self, wealth: torch.Tensor) -> None:
        wealth.clamp_(min=self.minimum, max=self.maximum)


class StressSignal(Protocol):
    """A reward signal that declares the tissue setpoint it prices error against (#33).

    The shape #43 and #44 plug in, and the one :func:`refuse_a_direction_score`
    can probe. ``reduction`` is the atom of ``mob.goal.error_relieved``: given the
    tissue's ``reading`` with this cell's ``push`` already in it, what the cell
    relieved. A signal that declares neither member is left alone -- the three the
    expert economy runs on price holdings rather than a tissue error.
    """

    @property
    def setpoint(self) -> float:
        """What the tissue is asked to hold, in the units the reading is in."""
        ...

    def reduction(self, reading: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        """Error without this cell, minus error with it."""
        ...


# Magnitude of the push the stress probes use. Any positive number does; 1.0 keeps
# the numbers in the refusal messages readable.
PROBE_PUSH = 1.0


def refuse_a_direction_score(reward: object) -> None:
    """Probe a plugged stress signal at three states, and refuse a direction score.

    A signal that declares a ``setpoint`` and a ``reduction(reading, push)`` is
    claiming to be #33's shape, where ``reading`` is the tissue's reading *with*
    this cell's ``push`` already in it. Three states separate that shape from a
    score on how well the cell points along a goal direction, and each is a
    property ``mob/goal.py`` states and the cosine term #33 rejected does not have:

    1. **Idle.** A cell whose push is zero earns exactly zero, at any error.
    2. **At the setpoint.** A tissue that was already holding its goal, and a cell
       that pushes it off: the cell is charged. A direction score pays it.
    3. **Saturation.** Closing the error exactly earns at least as much as
       overshooting it by the same amount again. A direction score pays the
       bigger push more.

    Signals that declare neither are left alone -- the three the expert economy
    runs on price holdings rather than a tissue error, and have no setpoint to be
    probed at. Raises ``ValueError``, at construction, where it is cheap to read.

    **What this does not reach**, so that #43 and #44 read it as a guard and not
    as a guarantee. It is opt-*in* by shape: a signal that spells its members
    anything other than ``setpoint`` and ``reduction``, or one behind a
    delegating wrapper that does not forward them, is never probed. And it
    validates ``reduction`` while :meth:`WealthUpdater.settle` pays through
    ``__call__``, with nothing binding the two -- ``__call__`` takes a
    :class:`Settlement` rather than a reading and a push, so the three states
    cannot be posed to it. What is checked is that the signal's *stated* atom is
    a stress; that the signal pays what its atom prices is still the plugger's to
    get right.
    """
    if not hasattr(reward, "setpoint") or not callable(getattr(reward, "reduction", None)):
        return
    signal = cast("StressSignal", reward)
    setpoint = signal.setpoint

    def probe(reading: float, push: float) -> float:
        relieved = signal.reduction(torch.tensor([reading]), torch.tensor([push]))
        return float(relieved.reshape(()))

    name = type(reward).__name__
    idle = probe(setpoint + PROBE_PUSH, 0.0)
    if idle != 0.0:
        raise ValueError(
            f"{name} is not a stress: a cell that does not move the reading earned "
            f"{idle}, not zero. What a cell is paid is the error it relieved, so "
            "holding a tissue that was already holding its goal pays nothing"
        )

    pushed_off = probe(setpoint + PROBE_PUSH, PROBE_PUSH)
    if pushed_off >= 0.0:
        raise ValueError(
            f"{name} is not a stress: a cell that pushed a tissue off its setpoint "
            f"earned {pushed_off}, which is not a charge. A signal that pays for "
            "alignment whether or not there was an error to fix is a direction "
            "score, and #33 rejected it"
        )

    closed = probe(setpoint, PROBE_PUSH)
    overshot = probe(setpoint + PROBE_PUSH, 2.0 * PROBE_PUSH)
    if overshot > closed:
        raise ValueError(
            f"{name} is not a stress: overshooting the setpoint earned {overshot} "
            f"against {closed} for closing the error exactly. The reduction has to "
            "peak where the push closes the error and fall past it, or the cell is "
            "paid for pushing harder rather than for the error it relieved"
        )


# --- The three signals the expert economy runs on ----------------------------------------


@dataclass(frozen=True)
class RealisedValueReward:
    """#15's value path: every winner is paid what it realised, at the share it held.

    The stress the expert economy prices is the organism's own loss -- a winner's
    value is its contribution against the loss gradient, the counterfactual
    against what the tissue would have done anyway (:func:`realised_values`), plus
    #33's goal term when a field is attached. It carries no setpoint of its own,
    so :func:`refuse_a_direction_score` has nothing to probe: the setpoint it is
    measured against is the tissue's behaviour without the cell, which is already
    inside the value.
    """

    config: MoBConfig
    multiplier: float = LOSS_REWARD_MULTIPLIER

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        values = settlement.values
        valid_mask = settlement.valid_mask
        assert values is not None and valid_mask is not None

        rewards = torch.zeros_like(wealth)
        for expert_idx in range(self.config.num_experts):
            held_slots = settlement.selected_experts == expert_idx
            held = held_slots.any(dim=-1) & valid_mask
            if not held.any():
                continue

            expert_value = (values * held_slots).sum(dim=-1)
            expert_share = (settlement.routing_weights.float() * held_slots).sum(dim=-1)
            credited = (expert_value * expert_share)[held].sum() / settlement.num_tokens
            rewards[expert_idx] += credited * self.config.reward_scale * self.multiplier
        return rewards

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        return None


@dataclass(frozen=True)
class LocalQualityReward:
    """The fallback when no loss reaches the layer: pay for output consistency.

    A proxy, and named as one. With no gradient there is no counterfactual to
    price, so what is left is whether an expert's outputs are self-consistent and
    sit near the layer's own scale -- neither of which is a stress the tissue is
    under. It runs at inference, where the alternative is a ledger that only ever
    decays.
    """

    config: MoBConfig
    # Whether this layer's settlement is the inference one. True wherever this
    # signal is reachable -- the path runs only when loss feedback is off -- and
    # written as the predicate rather than as its value so the reason survives.
    at_inference: bool = True
    multiplier: float = LOCAL_REWARD_MULTIPLIER

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        output = settlement.output
        assert output is not None

        rewards = torch.zeros_like(wealth)
        output_norms = output.norm(dim=-1)
        global_mean_norm = output_norms.mean()

        for k in range(self.config.top_k):
            for expert_idx in range(self.config.num_experts):
                mask = settlement.selected_experts[:, :, k] == expert_idx
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

                mean_confidence = settlement.confidences[:, :, expert_idx][mask].mean()
                mean_weight = settlement.routing_weights[:, :, k][mask].mean()
                selection_fraction = mask.sum().float() / settlement.num_tokens

                reward = quality * mean_confidence * mean_weight * selection_fraction
                rewards[expert_idx] += reward * self.config.reward_scale * self.multiplier
        return rewards

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        """The inference re-entry gift: a credit to the cells the gate has been ignoring."""
        usage = settlement.usage_count
        if not self.at_inference or self.config.inference_exploration_bonus <= 0 or usage is None:
            return None

        mean_usage = usage.mean()
        if mean_usage <= 0:
            return None

        usage_ratio = usage / (mean_usage + WEALTH_EPSILON)
        bonus = (1.0 - usage_ratio).clamp(min=0) * self.config.inference_exploration_bonus
        return bonus * wealth.mean()


@dataclass(frozen=True)
class ParticipationReward:
    """The last fallback: pay for having been selected, at the report and share it won on.

    The weakest of the three and the only one that prices nothing at all about
    what the cell did -- kept because a layer with neither a loss nor an output to
    read still has to settle, and a ledger that only decays is a ledger nobody can
    re-enter.
    """

    config: MoBConfig
    multiplier: float = PARTICIPATION_REWARD_MULTIPLIER

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        rewards = torch.zeros_like(wealth)
        for k in range(self.config.top_k):
            for expert_idx in range(self.config.num_experts):
                mask = settlement.selected_experts[:, :, k] == expert_idx
                if mask.any():
                    selection_count = mask.sum().float()
                    selection_fraction = selection_count / settlement.num_tokens
                    mean_confidence = settlement.confidences[:, :, expert_idx][mask].mean()
                    mean_weight = settlement.routing_weights[:, :, k][mask].mean()

                    base_reward = selection_fraction * mean_confidence * mean_weight
                    rewards[expert_idx] += base_reward * self.config.reward_scale * self.multiplier
        return rewards

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        return None


# --- The ledger itself -------------------------------------------------------------------


@dataclass(frozen=True)
class WealthUpdater:
    """One ledger's settlement: relax, pay, charge, hold at the floor.

    The sequence is the same at both scales, and so is the arithmetic that makes
    it analysable. Written as one map over a ledger ``w`` with a net inflow ``n``::

        w <- (1 - rho) w + rho S + n(w)

    where ``rho = 1 - decay`` is the relaxation rate and ``S`` the setpoint the
    ledger relaxes toward -- zero under ``decay``, ``initial_wealth`` under
    ``setpoint``, so **the decay ledger is the setpoint ledger at a setpoint of
    zero** and one derivation covers both. What that buys is in README
    ``#ledger-stability``: the fixed point, the condition under which the approach
    to it is monotone rather than oscillatory, and the threshold below which a
    cell is ruined -- each a closed form in ``rho``, ``S`` and the price
    coefficient, which is what lets #43 derive its constants instead of tuning
    them.

    ``pinned_at`` is #39's shadow ledger as a mode of the same class rather than a
    fourth path: the ledger settles exactly as it always did and the gate is
    handed a constant wealth instead of it.
    """

    reward: RewardSignal
    floor: Floor
    decay: float
    mode: str = LEDGER_DECAY
    setpoint: float = 0.0
    # What the gate reads instead of this ledger, or ``None`` when it reads the
    # ledger. Pinned at ``initial_wealth`` under ``decoupled`` rather than at any
    # other constant so that the decoupled arm's first auction is the live arm's:
    # at step 0 every ledger holds ``initial_wealth`` anyway, and prices are ratios
    # of wealths, so the shadow ledger's first settlement is identically the live
    # one's -- the invariant
    # ``test_the_shadow_ledger_is_the_live_ledger_at_step_zero_and_not_after`` pins.
    pinned_at: float | None = None
    # #59's metabolic charge, in wealth per unit of transmitted stress per
    # token. Zero in every recorded arm, and ``settle`` adds exactly nothing
    # when it is.
    stress_lambda: float = 0.0
    # #60's price on a cell's self-model. Zero in every recorded arm.
    self_score_mu: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_LEDGER_MODES:
            modes = ", ".join(sorted(SUPPORTED_LEDGER_MODES))
            raise ValueError(f"Unsupported ledger mode '{self.mode}'. Supported: {modes}")
        if self.mode == LEDGER_DECAY and self.setpoint != 0.0:
            raise ValueError(
                f"the decay ledger relaxes toward zero, so it cannot carry a setpoint of "
                f"{self.setpoint}; use mode '{LEDGER_SETPOINT}'"
            )
        refuse_a_direction_score(self.reward)

    @property
    def rate(self) -> float:
        """``rho``: the fraction of the gap to the setpoint the ledger closes per step.

        ``1 / rho`` is the ledger's memory in steps -- 333 at the shipped decay of
        0.997 -- and it is the one constant both the equilibrium and the
        oscillation condition are written in.
        """
        return 1.0 - self.decay

    def cannot_oscillate(self, price_coefficient: float = 0.0, wealth: float = 1.0) -> bool:
        """Whether the approach to equilibrium is monotone rather than alternating.

        The map's slope is ``f'(w) = decay + kappa / w^2``, and alternating
        approach is exactly ``f'(w) < 0``. So the condition has two halves and the
        second is easy to state wrongly:

        - A *winner* pays ``b_(k+1) / w``, so its ``kappa`` is non-negative and
          ``rho <= 1`` alone settles it: no price a winner pays can make its
          ledger oscillate. That is the bound #43's relaxation rate has to
          respect, and at ``wealth_decay`` 0.997 the expert ledger is 333 times
          inside it.
        - A *shut-out* cell's ``kappa`` is **negative** -- the Cavallo rebate
          exceeds the payments it never makes -- and a negative one lowers the
          slope rather than raising it, so ``rho <= 1`` no longer settles it on
          its own. Oscillation then needs ``kappa < -decay * w^2``, which is about
          -224 at the floor against the -0.28 to -0.36 measured across three seeds
          (README ``#ledger-stability``): the shipped conclusion survives by a
          factor of roughly 660, and it survives on a measurement rather than on
          the sign of ``kappa``.

        Named for what it tests and nothing more. A ledger with ``decay > 1``
        diverges monotonically, and this still answers ``True``, because it did
        not oscillate on the way out.
        """
        return self.decay + price_coefficient / wealth**2 >= 0.0

    def equilibrium(self, net_inflow: float) -> float:
        """The ledger's fixed point at a net inflow that does not depend on the ledger.

        ``w* = S + n / rho``. Exact under #39's ``decoupled`` arm, where every
        price is computed from the pinned wealth and the inflow therefore carries
        no ``w`` at all, so the shadow ledger is a linear filter with a
        closed-form steady state. Under a live economy the inflow does depend on
        the ledger and this is its leading term; README ``#ledger-stability``
        carries the quadratic that replaces it.

        Held in two halves rather than one, deliberately.
        ``test_the_pinned_arms_inflow_does_not_read_its_own_ledger`` pins the
        independence that makes the formula *apply* to the pinned arm, and
        ``test_a_ledger_at_a_constant_inflow_settles_where_the_closed_form_says``
        runs the formula itself out to its fixed point. They are not composed into
        one check on the fixture because the pinned arm's inflow is independent of
        its ledger but not *stationary* over the ledger's own 333-step memory --
        the confidence heads are still calibrating at every budget the fixture
        affords -- so a steady-state comparison there pins that drift and not this
        formula. ``scripts/measure_ledger_stability.py --coupling decoupled``
        reports how far, and README ``#ledger-stability`` records the range.
        """
        return self.setpoint + net_inflow / self.rate

    def relax(self, wealth: torch.Tensor) -> None:
        """Move the ledger one step toward what it relaxes toward, in place."""
        wealth *= self.decay
        if self.mode == LEDGER_SETPOINT:
            wealth += self.rate * self.setpoint

    def allocation_wealth(self, wealth: torch.Tensor) -> torch.Tensor:
        """The wealth the gate reads: this ledger, or the constant pinned over it."""
        if self.pinned_at is None:
            return wealth
        return torch.full_like(wealth, self.pinned_at)

    def settle(self, wealth: torch.Tensor, settlement: Settlement, charge: Charge) -> float:
        """Relax the ledger, pay what the step earned, charge what it was priced at.

        The charge is passed in rather than held because it is the layer's --
        looked up at call time, so a test that spies on ``_vcg_charges`` sees the
        call the settlement makes. Wealth moves by a single transfer: the rebate
        is netted inside the charge, and the gift is added after it because it
        prices no holding and is not part of the trade.

        Returns what the stress charged every cell this step, which the layer
        records: the calibration that prices the two channels against each
        other reads it, and so does the sweep that reports what the charge
        cost. Returned rather than stored because this class is frozen -- a
        ledger whose parameters can be written after construction is one the
        fingerprint no longer describes.

        #59's metabolic charge is the last term and it is the same number for
        every cell: ``lambda`` times the layer's transmitted stress summed over
        the step's tokens, paid winner or not, so the only way to lower it is to
        act. It is a term in this one settlement rather than a fourth update
        path, for the reason this class exists at all -- and at
        ``stress_lambda = 0``, which is every recorded arm, it adds exactly
        nothing and the arithmetic is the one that was recorded.
        """
        self.relax(wealth)

        transfer = self.reward(wealth, settlement)
        transfer -= charge(
            settlement.payments,
            settlement.selected_experts,
            settlement.num_tokens,
            self.reward.multiplier,
            settlement.rebates,
            settlement.valid_mask,
        )
        gift = self.reward.gift(wealth, settlement)
        if gift is not None:
            transfer += gift
        stress = self.stress_charge(settlement)
        transfer -= stress
        if self.self_score_mu > 0.0 and settlement.self_score is not None:
            # A payment for the accuracy of a cell's own prediction, added to
            # the same transfer: #60 makes the self-model load-bearing by
            # making a wrong one cost something, and a cost that arrives
            # through a fourth update path is the thing #40 removed.
            transfer = transfer + self.self_score_mu * settlement.self_score

        wealth += transfer
        self.floor(wealth)
        return stress

    def stress_charge(self, settlement: Settlement) -> float:
        """What the tissue's error costs every cell at this layer this step (#59)."""
        if self.stress_lambda <= 0.0 or settlement.stress is None:
            return 0.0
        stress = settlement.stress
        if settlement.valid_mask is not None:
            stress = stress * settlement.valid_mask.to(stress.dtype)
        return self.stress_lambda * float(stress.sum())

    @classmethod
    def for_experts(cls, config: MoBConfig) -> WealthUpdater:
        """The ledger one MoB layer settles, with the signal its configuration selects.

        Exactly one of the three signals is reachable for a given config, which is
        what the three separate paths this replaces were: the loss path when there
        is a loss to price, the local-quality proxy when there is not, and
        participation when there is not even an output to read. ``at_inference``
        is therefore constant per layer, and the decay it selects with it.
        """
        at_inference = not config.use_loss_feedback
        reward: RewardSignal
        if config.use_loss_feedback:
            reward = RealisedValueReward(config)
            decay = config.wealth_decay
        elif config.use_local_quality:
            reward = LocalQualityReward(config, at_inference=at_inference)
            decay = config.inference_wealth_decay if at_inference else config.wealth_decay
        else:
            reward = ParticipationReward(config)
            decay = config.wealth_decay

        return cls(
            reward=reward,
            floor=BandFloor(config.min_wealth, config.max_wealth),
            decay=decay,
            mode=config.ledger_mode,
            setpoint=config.initial_wealth if config.ledger_mode == LEDGER_SETPOINT else 0.0,
            pinned_at=(
                config.initial_wealth
                if config.persistence_coupling == PERSISTENCE_DECOUPLED
                else None
            ),
            stress_lambda=config.stress_lambda,
            self_score_mu=config.self_score_mu,
        )
