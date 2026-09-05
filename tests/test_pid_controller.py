"""PID controller in isolation, against the plant #4's characterisation measured.

The plant is a static gain with a one-token delay: the alignment lift at the
readout responds to the injected strength within the same token and then stays
there, and the controller acts on the reading from the *previous* token. Every
property below is stated from control theory for that plant -- never from the
implementation -- and each test fails when the mechanism it names is disabled.
"""

import math

import pytest

from pid_controller import (
    PIDConfig,
    PIDController,
    PIDState,
    pid_step,
    simc_pi_gains,
)


class StaticDelayPlant:
    """``y_t = baseline + disturbance + gain * u_{t-1}``; the measured shape."""

    def __init__(self, gain: float, baseline: float = 0.0, disturbance: float = 0.0):
        self.gain = gain
        self.baseline = baseline
        self.disturbance = disturbance
        self._previous_u = 0.0

    def measure(self) -> float:
        return self.baseline + self.disturbance + self.gain * self._previous_u

    def apply(self, u: float) -> None:
        self._previous_u = u


def run_loop(
    controller: PIDController,
    plant: StaticDelayPlant,
    setpoint: float,
    steps: int,
    key: str = "goal",
    bias: float = 0.0,
    setpoint_schedule=None,
) -> list[PIDState]:
    history = []
    for step in range(steps):
        target = setpoint_schedule(step) if setpoint_schedule else setpoint
        output, state = controller.step(key, target, plant.measure(), bias=bias)
        plant.apply(output)
        history.append(state)
    return history


def test_config_rejects_negative_gains_and_bad_filter():
    with pytest.raises(ValueError, match="kp"):
        PIDConfig(kp=-0.1)
    with pytest.raises(ValueError, match="ki"):
        PIDConfig(kp=0.1, ki=-1.0)
    with pytest.raises(ValueError, match="kd"):
        PIDConfig(kp=0.1, kd=-1.0)
    with pytest.raises(ValueError, match="derivative_filter_alpha"):
        PIDConfig(kp=0.1, derivative_filter_alpha=0.0)
    with pytest.raises(ValueError, match="output_limits"):
        PIDConfig(kp=0.1, output_limits=(1.0, 0.0))
    with pytest.raises(ValueError, match="dt"):
        PIDConfig(kp=0.1, dt=0.0)


def test_step_is_pure_and_returns_a_new_state():
    config = PIDConfig(kp=1.0, ki=0.5)
    before = PIDState()
    output, after = pid_step(config, before, setpoint=1.0, process_variable=0.0)

    assert before == PIDState()
    assert after is not before
    assert after.step_count == 1
    assert output == pytest.approx(after.output)


def test_p_only_leaves_the_steady_state_error_theory_predicts():
    """On a static plant P control settles at ``r / (1 + K kp)`` -- the baseline PI must beat."""
    gain, kp, setpoint = 0.4, 0.5, 2.0
    controller = PIDController(PIDConfig(kp=kp))
    history = run_loop(controller, StaticDelayPlant(gain), setpoint, steps=200)

    predicted_error = setpoint / (1 + gain * kp)
    assert history[-1].error == pytest.approx(predicted_error, rel=1e-3)
    assert predicted_error > 0.5 * setpoint


def test_pi_eliminates_steady_state_error_under_constant_disturbance():
    gain = 0.4
    kp, ki = simc_pi_gains(process_gain=gain, dead_time=1.0, time_constant=0.0, closed_loop_tau=4.0)
    controller = PIDController(PIDConfig(kp=kp, ki=ki))
    plant = StaticDelayPlant(gain, baseline=-0.5, disturbance=-1.0)

    history = run_loop(controller, plant, setpoint=1.5, steps=100)

    assert abs(history[-1].error) < 1e-3
    assert plant.measure() == pytest.approx(1.5, abs=1e-3)


def test_pi_converges_within_the_bound_the_tuning_promises():
    """SIMC with ``closed_loop_tau`` tokens settles to 5% in about three of them."""
    gain, tau_c = 0.4, 4.0
    kp, ki = simc_pi_gains(
        process_gain=gain, dead_time=1.0, time_constant=0.0, closed_loop_tau=tau_c
    )
    controller = PIDController(PIDConfig(kp=kp, ki=ki))
    history = run_loop(controller, StaticDelayPlant(gain), setpoint=1.0, steps=60)

    within_band = [abs(state.error) <= 0.05 for state in history]
    first_inside = within_band.index(True)
    assert first_inside <= 3 * (tau_c + 1)
    assert all(within_band[first_inside:])
    overshoot = max(-state.error for state in history)
    assert overshoot <= 0.2


def test_conditional_integration_stops_winding_up_when_saturated():
    """Unreachable setpoint: the integral must stop growing once the output is pinned."""
    config = PIDConfig(kp=0.0, ki=0.5, output_limits=(0.0, 1.0))
    controller = PIDController(config)
    plant = StaticDelayPlant(gain=0.1)

    history = run_loop(controller, plant, setpoint=10.0, steps=100)

    assert all(state.saturated for state in history[3:])
    # Once saturated the accumulator holds; without anti-windup it would be ~1000.
    assert history[-1].integral == pytest.approx(history[10].integral)
    assert history[-1].integral < 3.0


def test_anti_windup_recovers_faster_than_a_wound_up_integrator():
    """The property that justifies conditional integration over a clamp: bounded recovery."""

    def recovery_steps(anti_windup: bool) -> int:
        config = PIDConfig(kp=0.0, ki=0.5, output_limits=(0.0, 1.0), anti_windup=anti_windup)
        controller = PIDController(config)
        plant = StaticDelayPlant(gain=0.5)
        # Sustained disturbance makes the setpoint unreachable for 100 tokens ...
        plant.disturbance = -10.0
        run_loop(controller, plant, setpoint=0.25, steps=100)
        # ... then it is removed, and we count tokens until the loop is back in band.
        plant.disturbance = 0.0
        history = run_loop(controller, plant, setpoint=0.25, steps=400)
        for index, state in enumerate(history):
            if abs(state.error) < 0.01:
                return index
        return len(history)

    with_anti_windup = recovery_steps(anti_windup=True)
    without = recovery_steps(anti_windup=False)
    assert with_anti_windup <= 50
    assert without > 2 * with_anti_windup


def test_integral_limit_clamps_the_accumulator():
    config = PIDConfig(kp=0.0, ki=1.0, integral_limit=2.0, anti_windup=False)
    controller = PIDController(config)
    for _ in range(50):
        controller.step("goal", setpoint=1.0, process_variable=0.0)
    assert controller.snapshot("goal").integral == pytest.approx(2.0)


def test_derivative_on_pv_ignores_a_setpoint_step():
    """A setpoint change must not kick the derivative term; derivative-on-error does."""
    plant_pv = 0.3

    def d_terms(derivative_on_pv: bool) -> list[float]:
        config = PIDConfig(
            kp=0.0, kd=1.0, derivative_filter_alpha=1.0, derivative_on_pv=derivative_on_pv
        )
        controller = PIDController(config)
        terms = []
        for step in range(6):
            setpoint = 0.0 if step < 3 else 5.0
            _, state = controller.step("goal", setpoint, plant_pv)
            terms.append(state.d_term)
        return terms

    assert all(term == 0.0 for term in d_terms(derivative_on_pv=True))
    assert max(abs(term) for term in d_terms(derivative_on_pv=False)) > 1.0


def test_derivative_on_pv_has_the_documented_sign():
    config = PIDConfig(kp=0.0, kd=1.0, derivative_filter_alpha=1.0)
    controller = PIDController(config)
    controller.step("goal", 0.0, 0.0)
    _, rising = controller.step("goal", 0.0, 1.0)
    assert rising.d_term == pytest.approx(-1.0)


def test_ema_filter_smooths_a_noisy_derivative():
    import random

    rng = random.Random(0)
    noise = [rng.gauss(0.0, 1.0) for _ in range(2000)]

    def d_std(alpha: float) -> float:
        controller = PIDController(PIDConfig(kp=0.0, kd=1.0, derivative_filter_alpha=alpha))
        terms = [controller.step("goal", 0.0, pv)[1].d_term for pv in noise]
        tail = terms[200:]
        mean = sum(tail) / len(tail)
        return math.sqrt(sum((t - mean) ** 2 for t in tail) / len(tail))

    raw, filtered = d_std(alpha=1.0), d_std(alpha=0.1)
    # The raw derivative of white noise is an MA(1) sequence with lag-one
    # autocorrelation -1/2; an EMA with weight alpha scales its standard deviation
    # by alpha / sqrt(2 - alpha), not by the white-noise sqrt(alpha / (2 - alpha)).
    expected_ratio = 0.1 / math.sqrt(1.9)
    assert filtered / raw == pytest.approx(expected_ratio, rel=0.25)


def test_output_limits_and_bias_are_applied_to_the_total():
    config = PIDConfig(kp=1.0, output_limits=(0.0, 1.5))
    controller = PIDController(config)
    output, state = controller.step("goal", setpoint=10.0, process_variable=0.0, bias=0.3)
    assert output == 1.5
    assert state.saturated
    output, state = controller.step("goal", setpoint=0.0, process_variable=0.0, bias=0.3)
    assert output == pytest.approx(0.3)
    assert not state.saturated


def test_goals_keep_independent_state():
    controller = PIDController(PIDConfig(kp=0.0, ki=1.0))
    for _ in range(5):
        controller.step("truthful", 1.0, 0.0)
    controller.step("safe", 1.0, 0.5)
    assert controller.snapshot("truthful").integral == pytest.approx(5.0)
    assert controller.snapshot("safe").integral == pytest.approx(0.5)
    assert set(controller.states) == {"truthful", "safe"}

    controller.reset("truthful")
    assert controller.snapshot("truthful") == PIDState()
    assert controller.snapshot("safe").integral == pytest.approx(0.5)
    controller.reset()
    assert controller.states == {}


def test_state_round_trips_through_a_dict():
    controller = PIDController(PIDConfig(kp=0.2, ki=0.1, kd=0.05))
    for pv in (0.0, 0.1, 0.3):
        controller.step("goal", 1.0, pv)
    state = controller.snapshot("goal")

    restored = PIDState.from_dict(state.to_dict())
    assert restored == state
    controller.restore("goal", restored)
    assert controller.snapshot("goal") == state


def test_set_gains_applies_immediately_and_validates():
    controller = PIDController(PIDConfig(kp=0.1))
    controller.set_gains(kp=1.0, ki=0.2)
    assert controller.config.kp == 1.0
    assert controller.config.ki == 0.2
    _, state = controller.step("goal", 1.0, 0.0)
    assert state.p_term == pytest.approx(1.0)
    with pytest.raises(ValueError):
        controller.set_gains(kd=-1.0)


def test_simc_gains_for_a_static_delay_plant_are_integral_only():
    kp, ki = simc_pi_gains(process_gain=0.4, dead_time=1.0, time_constant=0.0, closed_loop_tau=4.0)
    assert kp == 0.0
    assert ki == pytest.approx(1.0 / (0.4 * 5.0))


def test_simc_gains_for_a_first_order_plant():
    kp, ki = simc_pi_gains(process_gain=2.0, dead_time=1.0, time_constant=10.0, closed_loop_tau=1.0)
    assert kp == pytest.approx(10.0 / (2.0 * 2.0))
    assert ki == pytest.approx(kp / min(10.0, 8.0))
    with pytest.raises(ValueError):
        simc_pi_gains(process_gain=0.0, dead_time=1.0, time_constant=0.0, closed_loop_tau=1.0)
