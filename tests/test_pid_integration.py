"""The tissue against the plant #4 measured, and the bounds that plant justifies (#6).

``tests/test_pid_controller.py`` proves the controller's properties on a generic
static-delay plant; ``tests/test_homeostat.py`` proves the tissue's wiring on
identity fakes. Neither runs the tissue on the plant the gains were derived for.
This does: a simulated tissue with the Qwen3-1.7B calibration from #4 -- per-cell
lifts of 0 to 0.58 sigma per unit of strength across 13, 16-21 and the readout
22, reference strength 4 inside the [2, 6] band, a 9-token sensor filter and the
two-token consensus delay -- so every bound below is the one that measurement
implies, not one assumed:

- The consensus the shared integrator regulates weights each cell by its gain
  (#21), so the tissue gain is ``sum(g^2) / sum(g) = 0.50`` sigma per unit rather
  than the plain mean 0.42 that counted the blind cell, and SIMC gives
  ``kp = 1.63, ki = 0.18`` (1.97 and 0.22 on the plain mean, as #4 recorded); the
  closed-loop time constant is the filter's 9 tokens, so a step settles to 5% in
  about ``3 x (9 + 2) = 33`` tokens, and the recorded settling under the served
  sampling was 16.
- The P-only loop this replaced sat against a permanent error of 0.69-0.79 with
  the setpoint unreachable; on a reachable deficit P control leaves
  ``d / (1 + K kp)`` of it -- 0.27 sigma of 0.5 whatever the gain, since SIMC fixes
  ``K kp`` at ``tau / (tau_c + theta)`` -- and PI leaves none.

Each property is paired with the state that breaks it: the integral removed, the
integral gain four times its derived value (inside the API's stability bound and
still past the overshoot bound), the accumulator freed from conditional
integration, the derivative filter removed, the derivative moved onto the error,
and the blind cell counted as one of eight. The setpoint-kick pairing is the one
that found a defect: the tissue used to hand the controller ``setpoint -
consensus`` as its process variable, so a setpoint change kicked a derivative that
was designed to ignore it.
"""

from dataclasses import replace

import pytest
import torch

from homeostat import AdaptiveHomeostat
from homeostat_calibration import AlignmentCalibration, ConsensusWeighting, LayerCalibration
from steering import SteeringConfig

from .wired_system import build_wired_system

# #4's calibration of the served truthful tissue on Qwen3-1.7B: lift per unit of
# strength in each cell's own slow sigma. Layer 13 is the bottom actuator with
# nothing below it; 22 is the sensor-only readout.
QWEN_CELL_LIFTS: dict[int, float] = {
    13: 0.0,
    16: 0.22,
    17: 0.46,
    18: 0.55,
    19: 0.58,
    20: 0.56,
    21: 0.51,
    22: 0.44,
}
QWEN_ACTUATORS: tuple[int, ...] = (13, 16, 17, 18, 19, 20, 21)
QWEN_READOUT = 22
QWEN_CELLS: tuple[int, ...] = (*QWEN_ACTUATORS, QWEN_READOUT)
QWEN_REFERENCE_STRENGTH = 4.0
QWEN_BAND = (2.0, 6.0)
QWEN_FILTER_ALPHA = 0.1
# The consensus gain the shared gains derive from: the gain-weighted mean of the cell
# lifts above, sum(g^2) / sum(g) (#21), and the SIMC gains it gives.
QWEN_TISSUE_GAIN = 0.502
QWEN_SIMC_GAINS = (1.63, 0.18)
# The plain mean of the lifts, which counts the blind cell's zero: the tissue gain #4
# recorded (the README rounds it to 0.42) and the gains it derived, reproduced by
# the uniform consensus as the pairing.
QWEN_MEAN_LIFT = 0.415
QWEN_RECORDED_GAINS = (1.97, 0.22)
FILTER_TOKENS = (1 - QWEN_FILTER_ALPHA) / QWEN_FILTER_ALPHA
CONSENSUS_DEAD_TIME = 2
SETTLE_BOUND = int(3 * (FILTER_TOKENS + CONSENSUS_DEAD_TIME))  # 33
OVERSHOOT_BOUND = 0.2
BAND_FRACTION = 0.05
ANTI_WINDUP_RECOVERY = 50
# A deficit the band can carry: 0.5 sigma needs 1.2 units of strength on top of
# the reference 4, inside the band's top of 6. One sigma would not be.
DEFICIT = 0.5
# A deficit the band cannot carry, so the output pins at its top and the
# accumulator has something to wind up on.
PINNING_DEFICIT = 2.0

DIRECTION = torch.eye(8)[0]


def qwen_calibration(
    reference_strength: float = QWEN_REFERENCE_STRENGTH, weighting: ConsensusWeighting = "gain"
) -> AlignmentCalibration:
    return AlignmentCalibration(
        layers={
            layer: LayerCalibration(resting_mean=0.0, resting_sigma=1.0, token_sigma=1.0, lift=lift)
            for layer, lift in QWEN_CELL_LIFTS.items()
        },
        actuators=QWEN_ACTUATORS,
        sensors=(*QWEN_ACTUATORS, QWEN_READOUT),
        reference_strength=reference_strength,
        num_passages=24,
        weighting=weighting,
    )


def qwen_config(**overrides) -> SteeringConfig:
    settings = dict(
        steering_layers=list(QWEN_ACTUATORS),
        readout_layer=QWEN_READOUT,
        base_strength=QWEN_REFERENCE_STRENGTH,
        min_strength=QWEN_BAND[0],
        max_strength=QWEN_BAND[1],
        adaptive=True,
        measurement_filter_alpha=QWEN_FILTER_ALPHA,
        orthogonal_projection=False,
    )
    settings.update(overrides)
    return SteeringConfig(**settings)


class MeasuredTissue:
    """The measured plant, cell by cell: content plus each cell's lift at the strength below it.

    A cell's lift was calibrated with every actuator at the reference strength, so
    it is a gain per unit of the *common* strength; the simulation reads it against
    the mean strength of the actuators below the cell, which is that calibration's
    plant. Readings are in sigma directly (resting mean 0, sigma 1), the injection
    is the passthrough #4 found it to be, and the token noise is optional.
    """

    def __init__(self, homeostat: AdaptiveHomeostat, content: float = 0.0):
        self.homeostat = homeostat
        self.content: dict[int, float] = dict.fromkeys(QWEN_CELLS, content)
        self.strengths: dict[int, float] = dict.fromkeys(QWEN_ACTUATORS, 0.0)
        self.noise: torch.Generator | None = None

    def set_content(self, content: float) -> None:
        self.content = dict.fromkeys(self.content, content)

    def reading(self, layer: int) -> float:
        below = [self.strengths[actuator] for actuator in QWEN_ACTUATORS if actuator < layer]
        lift = QWEN_CELL_LIFTS[layer] * (sum(below) / len(below) if below else 0.0)
        noise = float(torch.randn(1, generator=self.noise)) if self.noise is not None else 0.0
        return self.content[layer] + lift + noise

    def step(self) -> float:
        for layer in QWEN_CELLS:
            hidden = (self.reading(layer) * DIRECTION).view(1, 1, -1)
            strength = self.homeostat.sense(layer, hidden, DIRECTION)
            if layer in QWEN_ACTUATORS:
                self.strengths[layer] = strength
        return self.homeostat.error

    def run(self, passes: int) -> list[float]:
        return [self.step() for _ in range(passes)]

    def max_d_term(self) -> float:
        return max(
            abs(self.homeostat.controller.snapshot(self.homeostat._key(layer)).d_term)
            for layer in QWEN_CELLS
        )


def settled_at(errors: list[float], tolerance: float) -> int | None:
    """First pass after which the error stays inside ``tolerance``, or None."""
    for index, _ in enumerate(errors):
        if all(abs(error) <= tolerance for error in errors[index:]):
            return index + 1
    return None


def make(content: float = 0.0, **overrides) -> tuple[AdaptiveHomeostat, MeasuredTissue]:
    homeostat = AdaptiveHomeostat(qwen_config(**overrides), qwen_calibration())
    return homeostat, MeasuredTissue(homeostat, content)


def pin_pid_config(homeostat: AdaptiveHomeostat, **changes) -> None:
    """Rebuild the tissue's controller with fields the SteeringConfig does not expose."""
    original = homeostat._pid_config
    homeostat._pid_config = lambda: replace(original(), **changes)  # type: ignore[method-assign]
    homeostat.controller.config = homeostat._pid_config()


# --- The gains are the measured plant's -------------------------------------------


def test_the_derived_gains_follow_the_consensus_gain_of_the_recorded_lifts():
    """The gains derive from the gain the integrator sees; #4's plain mean is the pairing."""
    homeostat, _ = make()
    kp, ki = homeostat.gains()

    assert kp == pytest.approx(QWEN_SIMC_GAINS[0], rel=0.02)
    assert ki == pytest.approx(QWEN_SIMC_GAINS[1], rel=0.02)
    assert homeostat.calibration is not None
    assert homeostat.calibration.gain_z == pytest.approx(QWEN_TISSUE_GAIN, abs=0.005)
    assert homeostat.setpoint == pytest.approx(QWEN_TISSUE_GAIN * QWEN_REFERENCE_STRENGTH, abs=0.01)

    recorded = AdaptiveHomeostat(qwen_config(), qwen_calibration(weighting="uniform"))
    assert recorded.calibration is not None
    assert recorded.calibration.gain_z == pytest.approx(QWEN_MEAN_LIFT, abs=0.005)
    assert recorded.gains() == pytest.approx(QWEN_RECORDED_GAINS, rel=0.02)


# --- Step response, with the bound the plant justifies -----------------------------


def _setpoint_step(
    homeostat: AdaptiveHomeostat, tissue: MeasuredTissue
) -> tuple[list[float], float]:
    """Settle at the reference, then raise the reference strength by one unit."""
    tissue.run(200)
    before = homeostat.setpoint
    homeostat.calibration = qwen_calibration(QWEN_REFERENCE_STRENGTH + 1.0)
    step = homeostat.setpoint - before
    return tissue.run(100), step


def test_a_setpoint_step_settles_within_the_plant_bound_without_overshoot():
    """Three closed-loop time constants plus the delay: 33 tokens (measured: 26), overshoot 1.4%."""
    homeostat, tissue = make()
    errors, step = _setpoint_step(homeostat, tissue)

    settled = settled_at(errors, BAND_FRACTION * step)
    assert settled is not None and settled <= SETTLE_BOUND, settled
    assert max(-error for error in errors) / step <= OVERSHOOT_BOUND


def test_the_integral_removed_never_settles_the_step():
    """P control leaves ``1 / (1 + K kp)`` of the step -- 0.55 on the lumped plant, whatever K.

    The tissue's cells have different lifts and its bottom cell none, so the lumped
    prediction is approximate on it, and in a known direction: the bottom actuator's
    setpoint is zero before and after the step, so it pushes nothing extra, and the
    cell above it, which reads that actuator alone, is left with the whole step.
    Measured: 0.64 of the step (0.60 under the plain mean, where the blind cell's
    own zero error masked that lag). Either way, more than half never closes.
    """
    homeostat, tissue = make(ki=0.0)
    errors, step = _setpoint_step(homeostat, tissue)

    assert settled_at(errors, BAND_FRACTION * step) is None
    p_only_residual = 1 / (1 + QWEN_TISSUE_GAIN * homeostat.gains()[0])
    assert errors[-1] / step == pytest.approx(p_only_residual, rel=0.2)
    assert errors[-1] / step > p_only_residual
    assert errors[-1] / step > 0.5


def test_four_times_the_derived_integral_gain_breaks_the_overshoot_bound():
    """Inside the API's stability bound (half the critical gain), outside the overshoot bound."""
    homeostat, tissue = make(ki=4 * QWEN_SIMC_GAINS[1])
    limit = homeostat.max_stable_ki()
    assert limit is not None and homeostat.gains()[1] < limit
    errors, step = _setpoint_step(homeostat, tissue)

    assert max(-error for error in errors) / step > OVERSHOOT_BOUND


# --- PI against the recorded P-only baseline -----------------------------------------


def test_pi_removes_the_steady_state_error_the_p_only_loop_leaves():
    """On a 0.5 sigma deficit P control leaves ``d / (1 + K kp)``, 0.27 sigma; PI leaves none.

    The recorded baseline was worse still -- the P loop sat 0.69-0.79 from a
    setpoint it could never reach -- but that number is not a property of the
    controller, it is a property of an unreachable setpoint; the reachable case is
    the comparison the controller can be held to. On the tissue the P-only residual
    comes out below the lumped 0.27 (measured 0.24): the bottom cell's deficit is
    one nothing removes, so it pushes hardest of all, and every cell above reads
    that push.
    """
    pi, pi_tissue = make(content=-DEFICIT)
    p_only, p_tissue = make(content=-DEFICIT, ki=0.0)
    pi_tissue.run(300)
    p_tissue.run(300)

    predicted_p_error = DEFICIT / (1 + QWEN_TISSUE_GAIN * p_only.gains()[0])
    assert abs(pi.error) < 0.01
    assert p_only.error == pytest.approx(predicted_p_error, rel=0.15)
    assert 0 < p_only.error < predicted_p_error
    assert abs(p_only.error) > 20 * abs(pi.error)
    assert pi.current_strength > p_only.current_strength


# --- Anti-windup through the tissue ---------------------------------------------------


def _pin_then_release(
    homeostat: AdaptiveHomeostat, tissue: MeasuredTissue
) -> tuple[float, int | None]:
    tissue.run(100)
    tissue.set_content(-PINNING_DEFICIT)
    tissue.run(100)
    assert homeostat.status()["integral_saturated"], "the fixture must pin the band"
    integral = homeostat.controller.snapshot(homeostat._key(QWEN_ACTUATORS[1])).integral
    tissue.set_content(0.0)
    return integral, settled_at(tissue.run(400), BAND_FRACTION)


def test_the_tissue_unwinds_within_fifty_tokens_of_a_pinning_deficit_lifting():
    """Measured: the accumulator at 4.3 when the deficit lifts, and back in band at pass 40."""
    homeostat, tissue = make()
    integral, recovered = _pin_then_release(homeostat, tissue)

    assert recovered is not None and recovered <= ANTI_WINDUP_RECOVERY, recovered
    assert integral < 10.0


def test_a_free_integrator_winds_up_and_recovers_three_times_slower():
    """Measured: the accumulator at 95 against 4.3, and back in band at pass 137 against 40."""
    conditional, conditional_tissue = make()
    conditional_integral, conditional_recovered = _pin_then_release(conditional, conditional_tissue)
    homeostat, tissue = make()
    pin_pid_config(homeostat, anti_windup=False, integral_limit=None)
    integral, recovered = _pin_then_release(homeostat, tissue)

    assert conditional_recovered is not None
    assert integral > 10 * conditional_integral, (integral, conditional_integral)
    assert recovered is None or recovered > 3 * conditional_recovered, (
        recovered,
        conditional_recovered,
    )


# --- The derivative: filtered, and on the reading --------------------------------------


def _d_term_spread(alpha: float) -> float:
    homeostat, tissue = make(kd=1.0, derivative_filter_alpha=alpha)
    tissue.noise = torch.Generator().manual_seed(0)
    tissue.run(50)
    d_terms = []
    for _ in range(200):
        tissue.step()
        d_terms.append(homeostat.controller.snapshot(homeostat._key(QWEN_READOUT)).d_term)
    return float(torch.tensor(d_terms).std())


def test_the_derivative_is_filtered_against_per_token_noise():
    """Unit-sigma noise on every reading; the filtered derivative is a fraction of the raw one."""
    filtered, raw = _d_term_spread(QWEN_FILTER_ALPHA), _d_term_spread(1.0)

    assert filtered < 0.25 * raw, (filtered, raw)


def _kick_at_setpoint_step(derivative_on_pv: bool) -> tuple[float, float]:
    """``|d_term|`` on the first pass after a setpoint step, and its peak after a reading step."""
    homeostat, tissue = make(kd=1.0)
    if not derivative_on_pv:
        pin_pid_config(homeostat, derivative_on_pv=False)
    tissue.run(100)
    homeostat.calibration = qwen_calibration(QWEN_REFERENCE_STRENGTH + 1.0)
    tissue.step()
    setpoint_kick = tissue.max_d_term()

    reference, reference_tissue = make(kd=1.0)
    reference_tissue.run(100)
    reference_tissue.set_content(-QWEN_TISSUE_GAIN)
    reading_response = 0.0
    for _ in range(10):
        reference_tissue.step()
        reading_response = max(reading_response, reference_tissue.max_d_term())
    return setpoint_kick, reading_response


def test_a_setpoint_step_does_not_kick_the_derivative_but_a_reading_step_moves_it():
    setpoint_kick, reading_response = _kick_at_setpoint_step(derivative_on_pv=True)

    assert setpoint_kick < 0.1 * reading_response, (setpoint_kick, reading_response)
    assert reading_response > 0.005


def test_the_derivative_on_the_error_kicks_at_a_setpoint_step():
    setpoint_kick, reading_response = _kick_at_setpoint_step(derivative_on_pv=False)

    assert setpoint_kick > 3 * reading_response, (setpoint_kick, reading_response)


def test_every_cell_carries_the_same_controller_state_from_the_first_pass():
    """The first pass is seeded with the tissue setpoint, so the switch to the consensus is no step.

    Seeded with its own setpoint, each cell would remember a different process
    variable and read the shared one on pass two as a jump -- measured at 0.17 of
    d_term at ``kd = 1``, four times the setpoint kick the tissue was fixed for.
    """
    homeostat, tissue = make(kd=1.0)
    tissue.run(2)

    states = {homeostat.controller.snapshot(homeostat._key(layer)) for layer in QWEN_CELLS}
    assert len(states) == 1, "one shared memory, every cell"
    assert tissue.max_d_term() < 1e-9


# --- The blind cell: sensed, reported, and out of the shared memory (#21) --------------

BLIND_DEFICIT = 0.5


def _blind_deficit(weighting: ConsensusWeighting) -> tuple[AdaptiveHomeostat, dict[int, dict]]:
    """Content at cell 13 alone, settled; the tissue and its per-cell status by layer."""
    homeostat = AdaptiveHomeostat(qwen_config(), qwen_calibration(weighting=weighting))
    tissue = MeasuredTissue(homeostat)
    tissue.run(100)
    tissue.content[QWEN_ACTUATORS[0]] = -BLIND_DEFICIT
    tissue.run(300)
    return homeostat, {cell["layer"]: cell for cell in homeostat.status()["cells"]}


def _blind_push_bound(layer: int, kp: float, blind_error: float) -> float:
    """What the blind cell's own proportional push can move ``layer`` by.

    The push, ``kp`` times the blind error, is one actuator's; ``layer`` reads the
    mean strength of the actuators below it through its own lift.
    """
    below = [actuator for actuator in QWEN_ACTUATORS if actuator < layer]
    return QWEN_CELL_LIFTS[layer] * kp * blind_error / len(below)


def test_a_deficit_at_the_blind_cell_alone_leaves_the_survivors_consensus_at_setpoint():
    """Cell 13 reads half a sigma nothing can correct; the other seven no longer pay for it.

    The blind cell's weight is zero, so its error is reported unchanged, the tissue
    still senses it, and the consensus over the survivors settles at zero. What
    remains is the blind cell's own proportional push -- 0.85 units above the other
    actuators -- which cell 16 reads alone and the cells above read diluted over
    the actuators below them, while the integrator trims the common strength in
    return: the survivors spread from -0.14 sigma at 16 to +0.03 at the top, each
    inside the bound that push implies.
    """
    homeostat, cells = _blind_deficit("gain")
    kp = homeostat.gains()[0]
    blind = cells[QWEN_ACTUATORS[0]]
    survivors = [cells[layer] for layer in QWEN_CELLS[1:]]

    assert blind["weight"] == 0.0
    assert blind["error"] == pytest.approx(BLIND_DEFICIT, abs=1e-3)
    assert abs(homeostat.error) < 1e-3
    assert homeostat.sensed_error > 0.0
    weighted = sum(c["weight"] * c["error"] for c in survivors) / sum(
        c["weight"] for c in survivors
    )
    assert abs(weighted) < 1e-3
    for cell in survivors:
        assert abs(cell["error"]) < _blind_push_bound(cell["layer"], kp, blind["error"]), cell
    assert blind["output"] > cells[QWEN_ACTUATORS[1]]["output"] + 0.5


def test_the_uniform_consensus_dilutes_the_blind_deficit_over_the_survivors():
    """The pairing: counted as one of eight, the blind error is zeroed by pushing the other seven.

    The plain mean goes to zero and every survivor ends past its own setpoint, at
    ``-blind / 7`` on average: -0.07 sigma, the dilution #6 pinned on the fixture.
    """
    homeostat, cells = _blind_deficit("uniform")
    blind = cells[QWEN_ACTUATORS[0]]
    survivors = [cells[layer] for layer in QWEN_CELLS[1:]]

    assert blind["weight"] == 1.0
    assert abs(homeostat.error) < 1e-3
    mean_survivor_error = sum(c["error"] for c in survivors) / len(survivors)
    assert mean_survivor_error == pytest.approx(-blind["error"] / len(survivors), rel=0.02)
    assert all(cell["error"] < 0 for cell in survivors)


# --- Multi-goal independence, on the wired system --------------------------------------


def test_two_goal_tissues_on_one_model_keep_independent_state():
    """A deficit along one goal's direction moves that loop and leaves the other's alone.

    Two tissues on the same tiny model, orthogonal directions, each with its own
    calibration, controller keys and strength; content dragged along the first
    direction only. The first recovers by raising its strength; the second's error
    stays inside its band and its strength at the reference throughout.
    """
    system = build_wired_system(goals=("alpha", "beta"))
    system.run(60)
    beta_setpoint = system.setpoint("beta")
    beta_reference = system.tissue("beta").config.base_strength

    system.set_content("alpha", -1.0)
    beta_errors = []
    for _ in range(200):
        system.step()
        beta_errors.append(abs(system.error("beta")))

    assert abs(system.error("alpha")) <= BAND_FRACTION * system.setpoint("alpha")
    assert system.strength("alpha") > beta_reference + 0.2
    assert max(beta_errors) <= BAND_FRACTION * beta_setpoint
    assert system.strength("beta") == pytest.approx(beta_reference, abs=0.01)
    alpha_keys = set(system.tissue("alpha").controller.states)
    beta_keys = set(system.tissue("beta").controller.states)
    assert alpha_keys and beta_keys and not alpha_keys & beta_keys
