"""Discrete PI(D) control for the homeostat, as a pure function over immutable state.

The homeostat regulates how far the residual stream sits along a goal direction by
choosing an injection strength. #4's plant characterisation found that plant to be
a *static gain with a one-token delay*: the reading responds to the strength within
the same token and holds, and the controller acts on the previous token's reading.
On such a plant a proportional controller settles at ``error = r / (1 + K kp)``,
so the integral term is what removes the offset; the derivative is available but
adds noise sensitivity and ships disabled.

Design points the tests pin down:

- ``pid_step`` is pure: it takes a :class:`PIDState` and returns a new one, so a
  state can be snapshotted, serialised and restored without touching the
  controller.
- **Conditional integration** anti-windup: when the output is pinned at a limit
  and the error would drive it further, the accumulator stops growing -- and is
  snapped to the value that puts the output exactly at the limit, so recovery
  starts from the boundary rather than one step past it. It resumes the moment
  the error reverses.
- **Derivative on the process variable**, not the error, so a setpoint change
  does not kick the output; EMA-filtered so per-token noise does not either.
- One :class:`PIDController` holds a state per key (a goal), sharing gains.

:func:`simc_pi_gains` derives the gains from the measured plant (Skogestad's SIMC
rules), which is how the defaults in ``steering.SteeringConfig`` were obtained.
"""

import math
from dataclasses import asdict, dataclass, replace

_UNBOUNDED = (-math.inf, math.inf)


@dataclass(frozen=True)
class PIDConfig:
    """Gains and limits. ``dt`` is one token; ``output_limits`` clamp the total output."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 1.0
    # Absolute clamp on the accumulator, a second guard beside conditional
    # integration for the case where the output limits are unbounded.
    integral_limit: float | None = None
    derivative_filter_alpha: float = 0.1
    output_limits: tuple[float, float] = _UNBOUNDED
    derivative_on_pv: bool = True
    anti_windup: bool = True

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "kd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not 0 < self.derivative_filter_alpha <= 1:
            raise ValueError(
                f"derivative_filter_alpha must be in (0, 1], got {self.derivative_filter_alpha}"
            )
        if self.integral_limit is not None and self.integral_limit <= 0:
            raise ValueError(f"integral_limit must be positive, got {self.integral_limit}")
        low, high = self.output_limits
        if low > high:
            raise ValueError(f"output_limits must be ordered, got {self.output_limits}")


@dataclass(frozen=True)
class PIDState:
    """One loop's memory plus the terms of its last step, for status and checkpoints."""

    integral: float = 0.0
    previous_pv: float | None = None
    previous_error: float | None = None
    filtered_derivative: float = 0.0
    step_count: int = 0
    error: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    output: float = 0.0
    saturated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PIDState":
        return cls(**data)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _raw_derivative(
    config: PIDConfig, state: PIDState, error: float, process_variable: float
) -> float:
    if config.derivative_on_pv:
        if state.previous_pv is None:
            return 0.0
        # Minus sign: a rising process variable calls for less output.
        return -(process_variable - state.previous_pv) / config.dt
    if state.previous_error is None:
        return 0.0
    return (error - state.previous_error) / config.dt


def _accumulate(config: PIDConfig, integral: float, error: float) -> float:
    candidate = integral + error * config.dt
    if config.integral_limit is None:
        return candidate
    return _clamp(candidate, -config.integral_limit, config.integral_limit)


def _condition_integral(
    config: PIDConfig,
    previous: float,
    candidate: float,
    error: float,
    rest: float,
) -> float:
    """Conditional integration: stop at the boundary when the error pushes past a limit.

    ``rest`` is the output without the integral term. If the candidate accumulator
    would saturate the output in the direction the error is pushing, the accumulator
    is placed where the output meets the limit exactly -- never moved against the
    error, so a proportional term that saturates on its own simply holds the
    accumulator where it was. An error opposing the saturation integrates in full,
    which is what unwinds the loop the moment the disturbance lifts.
    """
    if not config.anti_windup or config.ki == 0:
        return candidate
    low, high = config.output_limits
    unsaturated = rest + config.ki * candidate
    if unsaturated > high and error > 0:
        boundary = (high - rest) / config.ki
    elif unsaturated < low and error < 0:
        boundary = (low - rest) / config.ki
    else:
        return candidate
    return _clamp(boundary, min(previous, candidate), max(previous, candidate))


def pid_step(
    config: PIDConfig,
    state: PIDState,
    setpoint: float,
    process_variable: float,
    bias: float = 0.0,
) -> tuple[float, PIDState]:
    """One controller step: ``(output, new_state)``. ``bias`` is the feed-forward term.

    The output is ``bias + P + I + D`` clamped to ``config.output_limits``; the
    saturation flag and the anti-windup logic both read the *total*, which is why
    the bias enters here rather than being added by the caller.
    """
    error = setpoint - process_variable
    p_term = config.kp * error

    raw_derivative = _raw_derivative(config, state, error, process_variable)
    alpha = config.derivative_filter_alpha
    filtered = alpha * raw_derivative + (1 - alpha) * state.filtered_derivative
    d_term = config.kd * filtered

    rest = bias + p_term + d_term
    candidate = _accumulate(config, state.integral, error)
    integral = _condition_integral(config, state.integral, candidate, error, rest)
    i_term = config.ki * integral

    low, high = config.output_limits
    unsaturated = rest + i_term
    output = _clamp(unsaturated, low, high)
    saturated = unsaturated >= high or unsaturated <= low

    return output, PIDState(
        integral=integral,
        previous_pv=process_variable,
        previous_error=error,
        filtered_derivative=filtered,
        step_count=state.step_count + 1,
        error=error,
        p_term=p_term,
        i_term=i_term,
        d_term=d_term,
        output=output,
        saturated=saturated,
    )


class PIDController:
    """Shared gains, one :class:`PIDState` per key (a goal)."""

    def __init__(self, config: PIDConfig):
        self.config = config
        self._states: dict[str, PIDState] = {}

    @property
    def states(self) -> dict[str, PIDState]:
        return dict(self._states)

    def snapshot(self, key: str) -> PIDState:
        return self._states.get(key, PIDState())

    def restore(self, key: str, state: PIDState) -> None:
        self._states[key] = state

    def step(
        self, key: str, setpoint: float, process_variable: float, bias: float = 0.0
    ) -> tuple[float, PIDState]:
        output, state = pid_step(self.config, self.snapshot(key), setpoint, process_variable, bias)
        self._states[key] = state
        return output, state

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._states.clear()
        else:
            self._states.pop(key, None)

    def set_gains(
        self, kp: float | None = None, ki: float | None = None, kd: float | None = None
    ) -> PIDConfig:
        """Replace any of the gains; validation is the config's, so bad values raise."""
        changes = {
            name: value for name, value in (("kp", kp), ("ki", ki), ("kd", kd)) if value is not None
        }
        self.config = replace(self.config, **changes)
        return self.config


def simc_pi_gains(
    process_gain: float,
    dead_time: float,
    time_constant: float,
    closed_loop_tau: float,
) -> tuple[float, float]:
    """PI gains from a first-order-plus-dead-time plant by Skogestad's SIMC rules.

    ``kp = tau / (K (tau_c + theta))`` and ``ki = kp / min(tau, 4 (tau_c + theta))``.
    For the measured plant the time constant is zero -- the reading responds within
    the token -- and the rules reduce to integral-only control with
    ``ki = 1 / (K (tau_c + theta))``: the controller *is* the loop's dynamics, and
    ``closed_loop_tau`` is the number of tokens over which a deviation is corrected.
    """
    if process_gain <= 0:
        raise ValueError(f"process_gain must be positive, got {process_gain}")
    if dead_time < 0 or time_constant < 0 or closed_loop_tau <= 0:
        raise ValueError(
            "dead_time and time_constant must be non-negative, closed_loop_tau positive"
        )
    horizon = closed_loop_tau + dead_time
    if time_constant == 0:
        return 0.0, 1.0 / (process_gain * horizon)
    kp = time_constant / (process_gain * horizon)
    integral_time = min(time_constant, 4 * horizon)
    return kp, kp / integral_time
