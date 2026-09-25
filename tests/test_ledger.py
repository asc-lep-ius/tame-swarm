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

import math
import sys
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_ledger_stability  # noqa: E402
from measure_ledger_stability import (  # noqa: E402
    DIFFERENTIATED,
    QUALITY,
    CellReading,
    ClampedLedgerError,
    LedgerReading,
    derive_reward_scale,
    rate_placing,
)
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


# --- #73: the exchange rate, derived from the settlement ---------------------------------

# Short enough to run in the gate and long enough for a market to form: what
# these pin is the derivation's arithmetic and its refusal, not a settled
# number, which the slow test at the bottom and README #ledger-stability carry.
SHORT_STEPS, SHORT_TAIL = 120, 40


def test_rate_placing_is_the_identity_on_a_cells_own_root():
    """A cell read at rate ``s`` is placed at its own upper root by ``s`` exactly."""
    rho, setpoint, rate = 0.003, 0.0, 2.0
    reward, price_coefficient = 5.1023, 67.36
    linear = rho * setpoint + reward
    root = (linear + (linear**2 - 4 * rho * price_coefficient) ** 0.5) / (2 * rho)

    assert rate_placing(root, reward, price_coefficient, rate, rho, setpoint) == pytest.approx(
        rate, abs=1e-9
    )
    # Twice the target wants a little under twice the rate: the raw charge is a
    # fixed offset against the raw inflow, and it matters less the higher the root.
    doubled = rate_placing(2 * root, reward, price_coefficient, rate, rho, setpoint)
    assert 1.9 * rate < doubled < 2 * rate
    # No positive rate places a cell whose raw inflow at the target is under its raw charge.
    assert math.isnan(rate_placing(1.0, reward, price_coefficient, rate, rho, setpoint))


@pytest.mark.parametrize("fixture", [QUALITY, DIFFERENTIATED])
def test_the_derivation_returns_the_recorded_rate_at_the_recorded_scale(fixture):
    """1x is the recorded constant, in one pass, with the recorded economy untouched.

    The reference run *is* the first pass, so every winner's rate is the identity
    of ``rate_placing`` and the number that comes back is 2.0 to float rounding
    rather than to a tolerance chosen to fit.
    """
    derivation = derive_reward_scale(fixture, 1.0, seeds=(0,), steps=SHORT_STEPS, tail=SHORT_TAIL)

    assert derivation.derived == BASE_CONFIG.reward_scale
    assert derivation.final.next_rate == pytest.approx(BASE_CONFIG.reward_scale, abs=1e-6)
    assert len(derivation.passes) == 1
    assert not derivation.final.saturated
    assert derivation.final.next_rate_from == "derived"
    assert derivation.final.fixed_points[0] == derivation.reference[0]
    assert derivation.seed_spread == 0.0
    assert set(derivation.reference_wealth_vs_competence) == {0}


# --- the refusal paths, on hand-built readings ------------------------------------------


def _cell(share: float, reward: float, price_coefficient: float, on_ceiling: bool) -> CellReading:
    """A cell whose root follows from its R and kappa at the shipped decay."""
    rho = 1.0 - BASE_CONFIG.wealth_decay
    discriminant = reward**2 - 4.0 * rho * price_coefficient
    root = (reward + discriminant**0.5) / (2.0 * rho) if discriminant >= 0.0 else math.nan
    return CellReading(
        competence=0.5,
        wealth=BASE_CONFIG.max_wealth if on_ceiling else 20.0,
        share=share,
        reward=reward,
        price_coefficient=price_coefficient,
        settles_at=root,
        ruined_below=0.0,
        from_flat_inflow=0.0,
        clamped=on_ceiling,
        reconstruction_error=0.0,
        tail_at_ceiling=1.0 if on_ceiling else 0.0,
        tail_at_floor=0.0,
    )


def _reading(
    seed: int, scale: float, rate: float, winners: int, winner_reward: float = 5.0
) -> LedgerReading:
    """``winners`` cells resting on the ceiling with a market share, the rest shut out."""
    cells = tuple(
        _cell(0.45, winner_reward, 60.0, True)
        if index < winners
        else _cell(0.002, 0.01, -0.3, False)
        for index in range(BASE_CONFIG.num_experts)
    )
    return LedgerReading(
        mode=LEDGER_DECAY,
        coupling=PERSISTENCE_VALUE,
        seed=seed,
        cells=cells,
        market_holders=winners,
        least_share=0.002,
        wealth_vs_competence=0.8,
        tail_loss=0.01,
        contribution_scale=scale,
        reward_scale=rate,
    )


def _stub_measure(monkeypatch, scaled):
    """The reference is the recorded two-up-six-down lattice; ``scaled`` answers every other run."""

    def fake(mode, seed, steps, tail, fixture, contribution_scale, reward_scale, cells, config):
        if contribution_scale == 1.0 and reward_scale == BASE_CONFIG.reward_scale:
            return _reading(seed, 1.0, reward_scale, winners=2)
        return scaled(seed, contribution_scale, reward_scale)

    monkeypatch.setattr(measure_ledger_stability, "measure", fake)


def test_a_configuration_whose_every_pass_is_saturated_is_refused(monkeypatch):
    """Finite R and kappa on a clamped ledger derive a number, and the number is never returned."""
    _stub_measure(monkeypatch, lambda seed, scale, rate: _reading(seed, scale, rate, winners=5))

    with pytest.raises(ClampedLedgerError, match="no unsaturated, settled rate"):
        derive_reward_scale(QUALITY, 2.0, seeds=(0,), steps=10, tail=5, max_passes=3)


def test_a_saturated_pass_that_cannot_solve_halves_the_trial_rate(monkeypatch):
    """No positive rate places a winner whose inflow is under its charge: halve and go on."""

    def scaled(seed, scale, rate):
        if rate == BASE_CONFIG.reward_scale:
            # Saturated, and the winners' R is too small for any rate to place them.
            return _reading(seed, scale, rate, winners=5, winner_reward=0.001)
        return _reading(seed, scale, rate, winners=2)

    _stub_measure(monkeypatch, scaled)

    derivation = derive_reward_scale(QUALITY, 2.0, seeds=(0,), steps=10, tail=5)

    first = derivation.passes[0]
    assert first.saturated and first.next_rate_from == "halved"
    assert first.next_rate == BASE_CONFIG.reward_scale / 2.0
    assert derivation.derived == first.next_rate, (
        "the rate returned is the one the final pass ran at"
    )
    assert not derivation.final.saturated


def test_an_unsaturated_pass_that_cannot_place_the_winners_is_refused_at_once(monkeypatch):
    _stub_measure(
        monkeypatch,
        lambda seed, scale, rate: _reading(seed, scale, rate, winners=2, winner_reward=0.001),
    )

    with pytest.raises(ClampedLedgerError, match="does not cover the raw charge"):
        derive_reward_scale(QUALITY, 2.0, seeds=(0,), steps=10, tail=5)


def test_winners_pair_by_rank_and_a_rootless_rank_keeps_the_others_aligned(monkeypatch):
    """The scaled run may seat other cells; a rootless reference winner is skipped, not squeezed."""
    reading = _reading(0, 1.0, BASE_CONFIG.reward_scale, winners=3)
    cells = list(reading.cells)
    # Rank 2 earns less than rank 0, so its root differs and a pairing that
    # squeezed rank 2 onto rank 1's target could not return the identity.
    cells[2] = _cell(0.44, 4.0, 60.0, True)
    reading = replace(reading, cells=tuple(cells))
    # Rank 1's root is not real: kappa too large for its reward. The same share
    # as rank 0, so the stable sort keeps it at rank 1.
    cells[1] = _cell(0.45, 0.01, 60.0, True)
    rootless = replace(reading, cells=tuple(cells))
    targets = measure_ledger_stability._winner_targets(rootless)

    assert sorted(targets) == [0, 1, 2]
    assert math.isfinite(targets[0]) and math.isnan(targets[1]) and math.isfinite(targets[2])
    # The scaled reading's rank-2 winner is placed at the reference's rank-2 root,
    # not shifted onto rank 1's missing one.
    _, per_rank, derived = measure_ledger_stability._solve(
        reading, targets, BASE_CONFIG.reward_scale, BASE_CONFIG
    )
    assert sorted(per_rank) == [0, 2]
    assert per_rank[2] == pytest.approx(BASE_CONFIG.reward_scale, abs=1e-9)
    assert derived == pytest.approx(BASE_CONFIG.reward_scale, abs=1e-9)


def test_a_scaled_run_with_fewer_winners_than_the_reference_cannot_be_placed(monkeypatch):
    _stub_measure(monkeypatch, lambda seed, scale, rate: _reading(seed, scale, rate, winners=1))
    targets = measure_ledger_stability._winner_targets(
        _reading(0, 1.0, BASE_CONFIG.reward_scale, winners=2)
    )

    _, _, derived = measure_ledger_stability._solve(
        _reading(0, 2.0, 2.0, winners=1), targets, 2.0, BASE_CONFIG
    )
    assert math.isnan(derived)
    with pytest.raises(ClampedLedgerError, match="fewer winners"):
        derive_reward_scale(QUALITY, 2.0, seeds=(0,), steps=10, tail=5)


def test_the_derivation_refuses_a_ledger_the_band_is_holding():
    """A band one credit wide holds every winner on the ceiling; there is no rate to read."""
    narrow = replace(BASE_CONFIG, max_wealth=BASE_CONFIG.initial_wealth + 1.0)

    with pytest.raises(ClampedLedgerError, match="ceiling"):
        derive_reward_scale(
            QUALITY,
            2.0,
            seeds=(0,),
            steps=SHORT_STEPS,
            tail=SHORT_TAIL,
            config=narrow,
            max_passes=2,
        )


def test_the_derivation_refuses_a_scale_that_is_not_positive():
    with pytest.raises(ValueError, match="positive"):
        derive_reward_scale(QUALITY, 0.0, seeds=(0,), steps=SHORT_STEPS, tail=SHORT_TAIL)
    with pytest.raises(ValueError, match="positive"):
        SyntheticEconomy(DEFAULT_COMPETENCE, seed=0, contribution_scale=-1.0)


def _quality_trajectory(steps: int = 30, **economy_overrides) -> tuple[torch.Tensor, list]:
    config = replace(BASE_CONFIG, **economy_overrides.pop("config", {}))
    economy = SyntheticEconomy(
        shuffled(DEFAULT_COMPETENCE, 0), seed=0, config=config, **economy_overrides
    )
    routed = [economy.step().selected_experts.clone() for _ in range(steps)]
    return economy.mob.expert_wealth.clone(), routed


def test_the_recorded_quality_fixture_is_bitwise_the_one_at_scale_one():
    """Multiplying the planted correction by 1.0 is exact; by 2.0 it is another fixture."""
    recorded_wealth, recorded_routes = _quality_trajectory()
    at_one_wealth, at_one_routes = _quality_trajectory(contribution_scale=1.0)
    at_two_wealth, _ = _quality_trajectory(contribution_scale=2.0)

    assert torch.equal(at_one_wealth, recorded_wealth)
    assert all(torch.equal(a, b) for a, b in zip(at_one_routes, recorded_routes, strict=True))
    assert not torch.equal(at_two_wealth, recorded_wealth)


def test_the_rate_reads_exactly_zero_on_decoupled():
    """Under the pinned arm the gate never reads the ledger the rate moves, so nothing moves.

    The pairing on the live arm is what makes the zero a reading rather than a
    fixture that ignores the rate everywhere.
    """
    decoupled = {"persistence_coupling": PERSISTENCE_DECOUPLED}
    _, recorded = _quality_trajectory(config=decoupled)
    _, halved = _quality_trajectory(config={**decoupled, "reward_scale": 0.5})
    assert all(torch.equal(a, b) for a, b in zip(recorded, halved, strict=True))

    _, live = _quality_trajectory()
    _, live_halved = _quality_trajectory(config={"reward_scale": 0.5})
    assert any(not torch.equal(a, b) for a, b in zip(live, live_halved, strict=True)), (
        "the live economy ignored the rate too; the pairing proves nothing"
    )


def test_a_hand_set_rate_is_not_at_parity_with_a_derived_one():
    """The withdrawn grid is the derivation's pairing, never its twin.

    Two arms at one scale and two rates have two fixed points; and two at one
    rate, one derived and one set by hand, are a measurement and its control
    rather than a comparison -- which is why both fields are asserted equal.
    """
    derived = replace(BASE, contribution_scale=2.0, reward_scale=0.5, reward_scale_derived=True)
    hand_set = replace(derived, persistence_coupling=PERSISTENCE_DECOUPLED, reward_scale=2.0)
    coincident = replace(hand_set, reward_scale=0.5, reward_scale_derived=False)

    assert_parity([derived, replace(derived, persistence_coupling=PERSISTENCE_DECOUPLED)])
    with pytest.raises(ParityError, match="reward_scale"):
        assert_parity([derived, hand_set])
    with pytest.raises(ParityError, match="reward_scale_derived"):
        assert_parity([derived, coincident])
    # Every run before #73 ran at the hand-set constant, and a 1x run after it does too.
    assert BASE.reward_scale == BASE_CONFIG.reward_scale
    assert BASE.reward_scale_derived is False


# The differentiated fixture at twice the recorded correction, seeds 0-2, 2667
# steps: the configuration #60's grid was withdrawn on. What is pinned is the
# guardrail #73 names as the number that would show the derivation failing --
# ceiling occupancy at 2x outside the 1x spread -- with one cell of tolerance,
# and that the recorded rate at 2x is the saturated pairing the grid measured.
ONE_CELL = 1.0 / BASE_CONFIG.num_experts


@pytest.mark.slow
def test_a_2x_run_at_the_derived_rate_keeps_the_ceiling_inside_the_1x_spread():
    derivation = derive_reward_scale(DIFFERENTIATED, 2.0)

    assert derivation.passes[0].rate == BASE_CONFIG.reward_scale
    assert derivation.passes[0].saturated, "the hand-set rate at 2x is no longer the clamp"
    assert not derivation.final.saturated
    assert 0.0 < derivation.derived < BASE_CONFIG.reward_scale
    low = min(derivation.reference_ceiling_occupancy.values()) - ONE_CELL
    high = max(derivation.reference_ceiling_occupancy.values()) + ONE_CELL
    for seed, occupancy in derivation.final.ceiling_occupancy.items():
        assert low <= occupancy <= high, (seed, occupancy, derivation.reference_ceiling_occupancy)
