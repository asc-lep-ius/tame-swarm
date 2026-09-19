"""One ledger, three plugs: the class both scales settle through (#40).

``tame/mob/ledger.py`` is one :class:`WealthUpdater` -- relax, pay, charge, hold
at a floor -- with the differences as plugs: a reward signal, a floor, and
whether the relaxation points at zero or at a setpoint. These are the tests of
the class itself: that each plug is a plug, that the mode changes what the ledger
does, and that the fixed point the README derives is the one the map settles at.

What the class must *not* have changed is in
``tests/test_wealth_updates.py::test_the_value_path_reproduces_the_recorded_fixture_numbers``,
which pins the recorded economy point by point.
"""

import sys
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    SyntheticEconomy,
    shuffled,
)

from mob import MixtureOfBidders, MoBConfig  # noqa: E402
from mob.goal import error_relieved  # noqa: E402
from mob.ledger import (  # noqa: E402
    LEDGER_DECAY,
    LEDGER_SETPOINT,
    LOSS_REWARD_MULTIPLIER,
    PERSISTENCE_DECOUPLED,
    PERSISTENCE_VALUE,
    BandFloor,
    LocalQualityReward,
    ParticipationReward,
    RealisedValueReward,
    RewardSignal,
    Settlement,
    WealthUpdater,
    refuse_a_direction_score,
)
from parity import ParityError, assert_parity  # noqa: E402

from .arm_fingerprints import BASE  # noqa: E402

STABILITY_CONFIG = MoBConfig(
    num_experts=2,
    top_k=1,
    hidden_dim=32,
    intermediate_dim=64,
    adapter_rank=4,
    adapter_alpha=4.0,
    use_shared_base=True,
    use_vcg_payments=True,
    use_differentiable_routing=True,
    use_loss_feedback=True,
    use_local_quality=True,
)


def _build_training_mob(config: MoBConfig = STABILITY_CONFIG) -> MixtureOfBidders:
    """A layer whose experts have something to sell; see ``tests/test_wealth_updates.py``."""
    mob = MixtureOfBidders(config)
    mob.train()
    with torch.no_grad():
        for name, param in mob.experts.named_parameters():
            if name.endswith("_B.weight"):
                param.normal_(std=0.1)
    return mob


def _settle(mob: MixtureOfBidders, hidden: torch.Tensor) -> None:
    """Forward, backward a synthetic per-token loss on the output, then settle."""
    output = mob(hidden)
    target = torch.randn_like(output)
    per_token = ((output - target) ** 2).sum(dim=-1)
    per_token.mean().backward()
    mob.update_wealth_from_loss(per_token.detach(), loss_gradient_scale=float(per_token.numel()))


LEDGER_CONFIG = replace(STABILITY_CONFIG, num_experts=4, top_k=1)

# A band no unit test of the ledger's own dynamics can reach, so what those tests
# read is the map and not the clamp.
WIDE = BandFloor(-1e9, 1e9)


def _empty_settlement(num_experts: int = 1) -> Settlement:
    """A step with no holdings, for the tests that drive the ledger and not the economy."""
    return Settlement(
        selected_experts=torch.zeros(1, 1, 1, dtype=torch.long),
        routing_weights=torch.ones(1, 1, 1),
        confidences=torch.zeros(1, 1, num_experts),
        num_tokens=1,
    )


# The ledger the layer keeps is float32, where 0.997 is 0.99699997901916504 and
# the map's own fixed point sits 1e-5 below the formula's. These tests are about
# the formula, so they drive the ledger in float64 and that representation error
# is not what they measure.
LEDGER_DTYPE = torch.float64


def _ledger_wealth(value: float) -> torch.Tensor:
    return torch.full((1,), value, dtype=LEDGER_DTYPE)


def _no_charge(payments, selected_experts, num_tokens, reward_multiplier, rebates=None, mask=None):
    del payments, selected_experts, num_tokens, reward_multiplier, rebates, mask
    return torch.zeros(1, dtype=LEDGER_DTYPE)


@dataclass(frozen=True)
class _ConstantInflow:
    """Pays the same credits every step, whatever happened: the map without an economy."""

    inflow: float
    multiplier: float = LOSS_REWARD_MULTIPLIER

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        return torch.full_like(wealth, self.inflow)

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        return None


@dataclass(frozen=True)
class _GoalStress:
    """A signal in #33's shape: what the cell relieved of the tissue's goal error."""

    setpoint: float = 0.4
    multiplier: float = LOSS_REWARD_MULTIPLIER

    def reduction(self, reading: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        return error_relieved(reading, push, self.setpoint)

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        return torch.zeros_like(wealth)

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        return None


@dataclass(frozen=True)
class _WrongShape(_GoalStress):
    """A signal that declares a setpoint and prices something that is not a stress."""

    shape: str = "pays an idle cell"

    def reduction(self, reading: torch.Tensor, push: torch.Tensor) -> torch.Tensor:
        error = reading - self.setpoint
        if self.shape == "pays an idle cell":
            return torch.ones_like(push)
        if self.shape == "scores the direction":
            # The cosine term #33 rejected, in one dimension: alignment pays,
            # whether or not there was an error for the cell to fix.
            return push
        # Superlinear in the cell's own push: it is charged for a small push past
        # the setpoint and paid for a large one, which is a reward for pushing.
        return push**3 - 3.0 * push * error.abs()


@dataclass(frozen=True)
class _DormancyFloor:
    """What #43's budget wants of a floor: a ledger that runs out is parked, not clamped."""

    park_at: float

    def __call__(self, wealth: torch.Tensor) -> None:
        wealth[wealth < self.park_at] = self.park_at


def _ledger_at(mode: str, decay: float, setpoint: float = 0.0, inflow: float = 0.0):
    return WealthUpdater(
        reward=_ConstantInflow(inflow), floor=WIDE, decay=decay, mode=mode, setpoint=setpoint
    )


@pytest.mark.parametrize(
    ("mode", "decay", "setpoint", "inflow"),
    [
        (LEDGER_DECAY, 0.997, 0.0, 0.6),
        (LEDGER_DECAY, 0.9, 0.0, -0.2),
        (LEDGER_SETPOINT, 0.997, 75.0, 0.6),
        (LEDGER_SETPOINT, 0.997, 75.0, 0.0),
    ],
)
def test_a_ledger_at_a_constant_inflow_settles_where_the_closed_form_says(
    mode, decay, setpoint, inflow
):
    """``w* = S + n / rho``: the formula README #ledger-stability derives, run out.

    The one case the closed form is exact for -- an inflow that does not depend on
    the ledger -- which is also #39's pinned arm, and the reason the analysis can
    be written down at all. A ledger whose equilibrium sat somewhere else would
    make every constant derived from it a guess.
    """
    ledger = _ledger_at(mode, decay, setpoint, inflow)
    predicted = ledger.equilibrium(inflow)

    # A fixed point is a point the map leaves alone, so that is what is asserted
    # first and exactly. The second half is that it is the *attracting* one: a
    # formula naming a repeller would derive constants nothing settles at.
    resting = _ledger_wealth(predicted)
    for _ in range(50):
        ledger.settle(resting, _empty_settlement(), _no_charge)
    assert resting.item() == pytest.approx(predicted, rel=1e-12)

    approaching = _ledger_wealth(75.0)
    for _ in range(5000):
        ledger.settle(approaching, _empty_settlement(), _no_charge)
    assert approaching.item() == pytest.approx(predicted, rel=1e-5)


def test_the_decay_ledger_is_the_setpoint_ledger_at_a_setpoint_of_zero():
    """Why one analysis covers both modes: they are one map with two setpoints."""
    decaying = _ledger_at(LEDGER_DECAY, 0.9, inflow=0.5)
    relaxing = _ledger_at(LEDGER_SETPOINT, 0.9, setpoint=0.0, inflow=0.5)

    assert relaxing.equilibrium(0.5) == decaying.equilibrium(0.5)
    assert _ledger_at(LEDGER_SETPOINT, 0.9, setpoint=40.0, inflow=0.5).equilibrium(0.5) > (
        decaying.equilibrium(0.5)
    )


def test_a_ledger_refuses_a_setpoint_it_does_not_relax_toward():
    with pytest.raises(ValueError, match="relaxes toward zero"):
        _ledger_at(LEDGER_DECAY, 0.9, setpoint=75.0)
    with pytest.raises(ValueError, match="Unsupported ledger mode"):
        WealthUpdater(reward=_ConstantInflow(0.0), floor=WIDE, decay=0.9, mode="drain")


def test_an_over_relaxed_ledger_oscillates_and_the_shipped_one_cannot():
    """The oscillation condition, and the constant it fixes: ``rho <= 1``.

    The map's slope is ``1 - rho + kappa / w^2`` and the price coefficient
    ``kappa`` is non-negative, so no economy can make a ledger oscillate while it
    closes at most the whole gap to its setpoint in one step. Past that it
    alternates about the equilibrium, which is what a relaxation rate chosen
    without the derivation would buy.
    """
    shipped = _ledger_at(LEDGER_DECAY, 0.997, inflow=0.6)
    over_relaxed = _ledger_at(LEDGER_SETPOINT, -0.5, setpoint=10.0)

    assert shipped.cannot_oscillate()
    assert not over_relaxed.cannot_oscillate()

    # The winner's case is not the only case. A shut-out cell's price coefficient
    # is negative -- the rebate exceeds the payments it never makes -- and a
    # negative one lowers the slope rather than raising it, so rho <= 1 does not
    # settle the question on its own. The -0.34 measured at the floor is far
    # inside the bound; a coefficient past -decay*w^2 is not, and the default
    # argument would have said so either way.
    assert shipped.cannot_oscillate(price_coefficient=-0.34, wealth=15.0)
    assert not shipped.cannot_oscillate(price_coefficient=-300.0, wealth=15.0)

    wealth = _ledger_wealth(0.0)
    errors = []
    for _ in range(6):
        over_relaxed.settle(wealth, _empty_settlement(), _no_charge)
        errors.append(wealth.item() - over_relaxed.setpoint)

    assert all(a * b < 0 for a, b in zip(errors, errors[1:], strict=False)), errors
    assert abs(errors[-1]) < abs(errors[0]), "an oscillation that does not settle is divergence"


def test_the_reward_slot_takes_a_stress():
    """#33's shape passes every probe, so a signal built on it plugs in unremarked."""
    ledger = WealthUpdater(reward=_GoalStress(), floor=WIDE, decay=0.997)

    assert ledger.reward.multiplier == LOSS_REWARD_MULTIPLIER


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ("pays an idle cell", "does not move the reading"),
        ("scores the direction", "direction score"),
        ("pays for pushing harder", "overshooting the setpoint"),
    ],
)
def test_the_reward_slot_refuses_a_signal_that_is_not_a_stress(shape, message):
    """The #33 invariant, enforced where a signal is plugged in rather than measured.

    #43 and #44 plug their own signals into this slot. A reward that pays for
    alignment whether or not there was an error to fix, or that pays more the
    harder a cell pushes, is a direction score wearing a stress's name -- and #32
    measured what a goal a cell is not *paid* for buys, which is nothing. Making
    the wrong shape fail at construction is cheaper than finding it in a run.
    """
    with pytest.raises(ValueError, match=message):
        WealthUpdater(reward=_WrongShape(shape=shape), floor=WIDE, decay=0.997)


def test_a_signal_with_no_setpoint_is_left_alone():
    """The three the expert economy runs on price holdings, not a tissue error."""
    refuse_a_direction_score(_ConstantInflow(1.0))
    refuse_a_direction_score(RealisedValueReward(LEDGER_CONFIG))


def test_the_floor_is_a_plug_and_the_band_is_one_of_them():
    """#43 wants a floor that parks a spent ledger rather than clamping it into a band."""
    parked = WealthUpdater(
        reward=_ConstantInflow(-5.0), floor=_DormancyFloor(park_at=2.0), decay=1.0
    )
    banded = WealthUpdater(reward=_ConstantInflow(-5.0), floor=BandFloor(2.0, 1e9), decay=1.0)

    wealth = _ledger_wealth(10.0)
    for _ in range(4):
        parked.settle(wealth, _empty_settlement(), _no_charge)
    assert wealth.item() == 2.0

    banded_wealth = _ledger_wealth(10.0)
    for _ in range(4):
        banded.settle(banded_wealth, _empty_settlement(), _no_charge)
    assert banded_wealth.item() == 2.0

    # The two agree on where a spent ledger rests and not on what a ceiling does,
    # which is the difference the slot exists for.
    rich = _ledger_wealth(1e10)
    BandFloor(2.0, 750.0)(rich)
    assert rich.item() == 750.0
    _DormancyFloor(park_at=2.0)(rich)
    assert rich.item() == 750.0


def _layer_reward_signal(**overrides):
    return type(MixtureOfBidders(replace(LEDGER_CONFIG, **overrides)).wealth_updater.reward)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, RealisedValueReward),
        ({"use_loss_feedback": False, "use_local_quality": True}, LocalQualityReward),
        ({"use_loss_feedback": False, "use_local_quality": False}, ParticipationReward),
    ],
)
def test_the_layer_holds_one_updater_and_the_config_picks_its_signal(overrides, expected):
    """The three wealth paths were this: one ledger, and which signal is reachable."""
    assert _layer_reward_signal(**overrides) is expected


@dataclass
class _Recording:
    """Wraps a reward signal and keeps what it paid, so a plug can be read as one."""

    inner: RewardSignal
    paid: list[torch.Tensor] = dc_field(default_factory=list)

    @property
    def multiplier(self) -> float:
        return self.inner.multiplier

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        paid = self.inner(wealth, settlement)
        self.paid.append(paid.clone())
        return paid

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        return self.inner.gift(wealth, settlement)


def _inflows_under(coupling: str, start: torch.Tensor, steps: int = 3) -> list[torch.Tensor]:
    """What the reward paid on each of ``steps`` settlements, from a given ledger."""
    torch.manual_seed(5)
    mob = _build_training_mob(replace(LEDGER_CONFIG, persistence_coupling=coupling))
    recording = _Recording(mob.wealth_updater.reward)
    mob.wealth_updater = replace(mob.wealth_updater, reward=recording)
    with torch.no_grad():
        mob.expert_wealth.copy_(start)

    hidden = torch.randn(1, 8, 32)
    torch.manual_seed(7)
    for _ in range(steps):
        _settle(mob, hidden)
    return recording.paid


def test_the_pinned_arms_inflow_does_not_read_its_own_ledger():
    """Why the closed form is exact under ``decoupled``, and only there.

    The equilibrium ``S + n / rho`` assumes an inflow that does not depend on the
    ledger. Under the pinned arm every bid, price and rebate is computed from
    ``initial_wealth``, so the shadow ledger is a linear filter of an inflow it
    cannot influence and the formula is its steady state exactly. Under the live
    economy a winner pays ``b_(k+1) / w``, the inflow carries a ``1/w``, and
    README #ledger-stability's quadratic replaces the closed form -- which is a
    claim about the arms and is what this pins on both of them.
    """
    flat = torch.full((LEDGER_CONFIG.num_experts,), LEDGER_CONFIG.initial_wealth)
    spread = torch.linspace(20.0, 300.0, LEDGER_CONFIG.num_experts)

    for pinned, elsewhere in zip(
        _inflows_under(PERSISTENCE_DECOUPLED, flat),
        _inflows_under(PERSISTENCE_DECOUPLED, spread),
        strict=True,
    ):
        assert torch.equal(pinned, elsewhere)

    live = zip(
        _inflows_under(PERSISTENCE_VALUE, flat),
        _inflows_under(PERSISTENCE_VALUE, spread),
        strict=True,
    )
    assert any(not torch.equal(paid, elsewhere) for paid, elsewhere in live), (
        "the live economy's inflow ignored the ledger too; the pairing proves nothing"
    )


def _quality_ledger(steps: int = 30, **overrides) -> torch.Tensor:
    economy = SyntheticEconomy(
        shuffled(DEFAULT_COMPETENCE, 0), seed=0, config=replace(BASE_CONFIG, **overrides)
    )
    for _ in range(steps):
        economy.step()
    return economy.mob.expert_wealth.clone()


def test_the_setpoint_ledger_is_a_mode_and_the_recorded_one_is_the_default():
    """#26's ledger, derived and measured: it moves the economy, and it is not on."""
    recorded = _quality_ledger()

    assert torch.equal(_quality_ledger(ledger_mode=LEDGER_DECAY), recorded)
    assert not torch.equal(_quality_ledger(ledger_mode=LEDGER_SETPOINT), recorded), (
        "the setpoint ledger relaxed toward the same place the decay ledger does"
    )
    assert MoBConfig().ledger_mode == LEDGER_DECAY


def test_two_arms_that_relax_toward_different_things_are_not_at_parity():
    """The #25 failure, kept from recurring: MoBConfig is invisible to ArmFingerprint.

    A ledger with a different fixed point and a different ruin threshold is not a
    comparison partner, so the mode is a confound rather than a variable under
    test -- and it has to reach the fingerprint through ``TrainingConfig`` for the
    check to be able to fire at all.
    """
    decoupled = replace(BASE, persistence_coupling=PERSISTENCE_DECOUPLED)
    assert_parity([BASE, decoupled])

    with pytest.raises(ParityError, match="ledger_mode"):
        assert_parity([BASE, replace(decoupled, ledger_mode=LEDGER_SETPOINT)])

    assert BASE.ledger_mode == LEDGER_DECAY, "a fingerprint recorded before the mode is a decay one"


# --- The recorded ledger-stability row (README #ledger-stability) -------------------------

# The quality fixture at the shipped constants, 2667 steps -- eight memory
# horizons -- read over a 333-step tail, under the setpoint ledger at seed 0. The
# row rests on the closed form predicting a *live* economy's settled ledger, so
# what is pinned is the prediction's error and not the wealth: a fixture whose
# inflow moved would move both together, and only the error says the algebra is
# right. Six of eight cells are unclamped under this mode, which is what makes
# the check non-empty; the two monopolists rest on the ceiling with their
# equilibrium above it.
RECORDED_CLOSED_FORM_ERROR = 0.02
RECORDED_RECONSTRUCTION_ERROR = 1e-4
RECORDED_UNCLAMPED_CELLS = 6


@pytest.mark.slow
def test_the_closed_form_predicts_the_setpoint_arms_settled_ledger():
    """README #ledger-stability's measured claim, at seed 0.

    Solved from the run's own reward and price coefficient, not from constants
    chosen to fit: a cell that is not held by a bound settles where
    ``rho w^2 - (rho S + R) w + kappa = 0`` says it does.
    """
    from measure_ledger_stability import measure

    reading = measure(LEDGER_SETPOINT, seed=0)

    unclamped = [cell for cell in reading.cells if not cell.clamped]
    assert len(unclamped) == RECORDED_UNCLAMPED_CELLS, [cell.wealth for cell in reading.cells]
    for cell in unclamped:
        assert cell.relative_error < RECORDED_CLOSED_FORM_ERROR, cell

    for cell in reading.cells:
        if cell.clamped:
            assert cell.settles_at > BASE_CONFIG.max_wealth, cell
        else:
            # The reward and the charge are the whole settlement: run back
            # through the map they rebuild the ledger. A reading that had missed
            # a term would still solve a quadratic, and it would be the wrong one.
            assert cell.reconstruction_error < RECORDED_RECONSTRUCTION_ERROR, cell

    # The other half of the row: the ledger moved and the allocation did not.
    assert reading.market_holders == 2
    assert reading.least_share < 0.01
    assert reading.wealth_vs_competence > 0.5
