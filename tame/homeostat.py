"""The cognitive homeostat: a tissue of coupled cells, one per steered layer, per goal.

#4 measured the loop the P-controller had been closing and found it never regulated
anything: ``cos(h, v)`` sits near zero unsteered, a unit of strength moves it by
about a hundredth, and ``target_alignment = 0.7`` was unreachable by a factor of
thirty, so the output was a near-constant offset. The redesign follows the README's
own metaphor -- a pH *buffer* -- and TAME's own architecture -- a tissue, not a
controller with subordinates:

- **Cells.** Every steered layer is a cell. It reads its own stream (the projection
  of the last position onto the goal direction, EMA-filtered, as a z-score against
  the resting distribution calibrated for *that layer*), holds its own setpoint
  (the lift it reads when every actuator injects the certified reference strength),
  and runs its own integrator. A cell may also be sensor-only: the readout above
  the top actuator senses and votes but injects nothing.
- **Gap junctions.** The slow state is shared and the fast response is local. Every
  cell's integrator accumulates the *tissue's* mean error over the live cells, so
  all cells hold the same memory -- a coupled tissue is isopotential for slow
  signals -- and the bottom cell, which has exactly zero gain from its own action,
  cannot wind up on a deficit only the cells above it can correct. Each cell's
  proportional term acts on its *own* fresh error, so a cell that senses a deficit
  now pushes harder now, without memory and without fighting.
- **Damage.** Liveness is a property of firing, not of wiring: a cell that misses
  a pass drops out of the consensus and rejoins when it fires again. There is no
  fallback path, because the local rule *is* the rule.

Gains are derived from the measured tissue gain by SIMC (``pid_controller``)
unless the config pins them. Without a calibration the loop keeps the legacy
contract -- one implicit cell regulating cosine toward ``target_alignment`` with a
proportional gain -- so probes and tests that never calibrate behave as before.
"""

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pid_controller import PIDConfig, PIDController, PIDState, simc_pi_gains
from steering import (
    CAPABILITY_CORPUS,
    SteeringConfig,
    SteeringVector,
    estimate_capability_subspace,
    project_steering_direction,
)

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 10_000
# The proportional gain the pre-#4 controller shipped with; used only while a loop
# is uncalibrated, where cosine toward ``target_alignment`` is still the contract.
LEGACY_KP = 0.5
# The implicit cell an uncalibrated loop regulates through ``compute_strength``.
LEGACY_CELL = -1
# Shortest closed-loop time constant the derivation will accept, in tokens: with an
# unfiltered reading SIMC would otherwise ask for a loop that corrects a deviation
# faster than the one-token dead time allows.
MIN_CLOSED_LOOP_TAU = 1.0
# Tokens between the tissue's action and its integrator reading the consequence: a
# cell's own reading is one token old, and the mean error every cell integrates is
# frozen at the start of the pass, so it is a token older still.
SHARED_DEAD_TIME = 2.0
# Position 0 carries the attention-sink activation (norm ~17,000 against a median of
# 60-1,400 on Qwen3-1.7B); it is not a sample of the resting distribution.
FIRST_CALIBRATION_POSITION = 1
SIGMA_FLOOR = 1e-6


@dataclass(frozen=True)
class LayerCalibration:
    """One cell's resting state, and the lift it reads when every actuator injects."""

    resting_mean: float
    # Standard deviation of per-passage means: the slow variability the filtered
    # reading actually has, and the unit the setpoint and gains are expressed in.
    resting_sigma: float
    token_sigma: float
    # Projection units per unit of strength this cell reads with every actuator
    # injecting -- the passthrough from the layers below plus the network's
    # response, its own injection excluded. Zero for the lowest actuator.
    lift: float = 0.0

    @property
    def sigma(self) -> float:
        return max(self.resting_sigma, SIGMA_FLOOR)

    @property
    def gain_z(self) -> float:
        return self.lift / self.sigma


@dataclass(frozen=True)
class AlignmentCalibration:
    """Per-cell resting states and lifts; the tissue gain the shared gains derive from."""

    layers: dict[int, LayerCalibration]
    actuators: tuple[int, ...]
    sensors: tuple[int, ...]
    reference_strength: float
    num_passages: int

    def z(self, layer: int, projection: float) -> float:
        calibration = self.layers[layer]
        return (projection - calibration.resting_mean) / calibration.sigma

    def setpoint_z(self, layer: int) -> float:
        return self.layers[layer].gain_z * self.reference_strength

    @property
    def gain_z(self) -> float:
        """Tissue gain: mean cell lift per unit of strength, in each cell's own sigma."""
        return float(np.mean([self.layers[layer].gain_z for layer in self.sensors]))

    @property
    def readout_layer(self) -> int:
        return max(self.sensors)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm()
    return vector if norm == 0 else vector / norm


def _last_position(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states[:, -1, :].float().mean(dim=0)


class AdaptiveHomeostat:
    """One goal's tissue: the cells, their shared memory, and the shared gains."""

    def __init__(
        self,
        config: SteeringConfig,
        calibration: AlignmentCalibration | None = None,
        goal: str = "goal",
    ):
        self.config = config
        self.calibration = calibration
        self.goal = goal
        self.controller = PIDController(self._pid_config())
        self.alignment_history: deque[float] = deque(maxlen=MAX_HISTORY_LENGTH)
        self.strength_history: deque[float] = deque(maxlen=MAX_HISTORY_LENGTH)
        self._filtered: dict[int, float] = {}
        self._pv: dict[int, float] = {}
        self._error: dict[int, float] = {}
        self._strength: dict[int, float] = {}
        self._saturated: dict[int, bool] = {}
        self._seen: dict[int, int] = {}
        self._pass = 0
        self._recorded_pass = 0
        self._last_layer: int | None = None
        self._consensus: float | None = None

    # --- configuration -----------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    @property
    def readout_layer(self) -> int | None:
        return self.calibration.readout_layer if self.calibration else None

    @property
    def cells(self) -> tuple[int, ...]:
        return self.calibration.sensors if self.calibration else ()

    def cell_setpoint(self, layer: int) -> float:
        if self.calibration is not None and layer in self.calibration.layers:
            return self.calibration.setpoint_z(layer)
        return self.config.target_alignment

    @property
    def setpoint(self) -> float:
        """The tissue setpoint: the mean cell setpoint, or the legacy cosine target."""
        if self.calibration is None:
            return self.config.target_alignment
        return float(np.mean([self.cell_setpoint(layer) for layer in self.cells]))

    @property
    def filter_time_constant(self) -> float:
        alpha = self.config.measurement_filter_alpha
        return (1.0 - alpha) / alpha

    @property
    def closed_loop_tau(self) -> float:
        if self.config.closed_loop_tau is not None:
            return self.config.closed_loop_tau
        return max(self.filter_time_constant, MIN_CLOSED_LOOP_TAU)

    @property
    def dead_time(self) -> float:
        return SHARED_DEAD_TIME

    def gains(self) -> tuple[float, float]:
        """``(kp, ki)``: pinned by the config where given, otherwise derived from the plant."""
        kp, ki = self.config.kp, self.config.ki
        if kp is not None and ki is not None:
            return kp, ki
        if self.calibration is None:
            return (LEGACY_KP if kp is None else kp), (0.0 if ki is None else ki)
        derived_kp, derived_ki = simc_pi_gains(
            process_gain=self.calibration.gain_z,
            dead_time=self.dead_time,
            time_constant=self.filter_time_constant,
            closed_loop_tau=self.closed_loop_tau,
        )
        return (derived_kp if kp is None else kp), (derived_ki if ki is None else ki)

    def max_stable_ki(self) -> float | None:
        """Stability bound for the integral gain on the measured plant.

        With a static gain ``K`` and a dead time of two tokens the integral-only loop
        is stable iff ``K ki < 1``; conservative once the filter adds its own pole.
        """
        if self.calibration is None:
            return None
        return 2.0 / (self.calibration.gain_z * self.dead_time)

    def max_kd(self) -> float:
        """A one-sigma-per-token swing of the reading may not move the output past the band."""
        return self.config.max_strength - self.config.min_strength

    def set_gains(
        self, kp: float | None = None, ki: float | None = None, kd: float | None = None
    ) -> None:
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        ki_limit = self.max_stable_ki()
        if ki is not None and ki_limit is not None and ki > ki_limit:
            raise ValueError(f"ki {ki} exceeds the stability bound {ki_limit:.4g} for this plant")
        if kd is not None and kd > self.max_kd():
            raise ValueError(f"kd {kd} exceeds the noise bound {self.max_kd():.4g}")
        if kp is not None:
            self.config.kp = kp
        if ki is not None:
            self.config.ki = ki
        if kd is not None:
            self.config.kd = kd
        self.controller.config = self._pid_config()

    def _pid_config(self) -> PIDConfig:
        kp, ki = self.gains()
        span = self.config.max_strength - self.config.min_strength
        return PIDConfig(
            kp=kp,
            ki=ki,
            kd=self.config.kd,
            output_limits=(self.config.min_strength, self.config.max_strength),
            # The accumulator alone may never ask for more than the whole band.
            integral_limit=(span / ki) if ki > 0 and span > 0 else None,
            derivative_filter_alpha=self.config.derivative_filter_alpha,
        )

    # --- sensing and acting --------------------------------------------------------

    def _key(self, layer: int) -> str:
        return f"{self.goal}@{layer}"

    def _injects(self, layer: int) -> bool:
        return self.calibration is None or layer in self.calibration.actuators

    def reading(self, hidden_states: torch.Tensor, steering_vector: torch.Tensor) -> float:
        """Projection of the last position onto the goal direction (cosine when uncalibrated)."""
        state = _last_position(hidden_states)
        direction = _unit(steering_vector.float().to(state.device))
        if self.calibration is None:
            return float(F.cosine_similarity(state, direction, dim=0).item())
        return float((state @ direction).item())

    def sense(
        self, layer: int, hidden_states: torch.Tensor, steering_vector: torch.Tensor
    ) -> float:
        """One cell's step: read, filter, blend with the tissue, integrate; its strength."""
        if not self.config.adaptive:
            return self.config.base_strength
        self._advance(layer)

        reading = self.reading(hidden_states, steering_vector)
        alpha = self.config.measurement_filter_alpha
        previous = self._filtered.get(layer)
        filtered = reading if previous is None else alpha * reading + (1 - alpha) * previous
        self._filtered[layer] = filtered

        if self.calibration is not None and layer in self.calibration.layers:
            pv = self.calibration.z(layer, filtered)
        else:
            pv = filtered
        setpoint = self.cell_setpoint(layer)
        error = setpoint - pv
        self._pv[layer], self._error[layer] = pv, error

        # The controller integrates the tissue's mean error (frozen for this pass);
        # the cell's own deviation from it enters through the proportional term only.
        shared = 0.0 if self._consensus is None else self._consensus
        tissue_strength, state = self.controller.step(
            self._key(layer), setpoint, setpoint - shared, bias=self.config.base_strength
        )
        local = self.gains()[0] * (error - shared)
        low, high = self.config.min_strength, self.config.max_strength
        strength = min(high, max(low, tissue_strength + local))
        self._strength[layer] = strength
        self._saturated[layer] = state.saturated or strength <= low or strength >= high
        self._record()
        return strength

    def compute_strength(self, hidden_states: torch.Tensor, steering_vector: torch.Tensor) -> float:
        """The legacy single-loop entry point: one implicit cell."""
        return self.sense(LEGACY_CELL, hidden_states, steering_vector)

    def _advance(self, layer: int) -> None:
        """Cells fire in layer order; a non-increasing layer opens a new pass.

        The tissue's mean error is frozen here, over the cells alive at the end of
        the previous pass, so every cell in this pass blends in the same consensus.
        """
        if self._last_layer is None or layer <= self._last_layer:
            live = [cell for cell in self.live_cells() if cell in self._error]
            self._consensus = float(np.mean([self._error[cell] for cell in live])) if live else None
            self._pass += 1
        self._last_layer = layer
        self._seen[layer] = self._pass

    def live_cells(self) -> list[int]:
        """Cells that fired in this pass or the previous one."""
        return sorted(cell for cell, seen in self._seen.items() if seen >= self._pass - 1)

    def _live_actuators(self) -> list[int]:
        return [cell for cell in self.live_cells() if self._injects(cell)]

    @property
    def current_strength(self) -> float:
        actuators = [cell for cell in self._live_actuators() if cell in self._strength]
        if not actuators:
            return self.config.base_strength
        return float(np.mean([self._strength[cell] for cell in actuators]))

    @property
    def error(self) -> float:
        live = [cell for cell in self.live_cells() if cell in self._error]
        return float(np.mean([self._error[cell] for cell in live])) if live else 0.0

    def _record(self) -> None:
        """One history entry per pass, updated as the pass's cells fire."""
        live = [cell for cell in self.live_cells() if cell in self._pv]
        alignment = float(np.mean([self._pv[cell] for cell in live]))
        if self._recorded_pass == self._pass and self.alignment_history:
            self.alignment_history[-1] = alignment
            self.strength_history[-1] = self.current_strength
            return
        self.alignment_history.append(alignment)
        self.strength_history.append(self.current_strength)
        self._recorded_pass = self._pass

    # --- state -----------------------------------------------------------------------

    def _cell_status(self, layer: int, live: set[int]) -> dict[str, Any]:
        state = self.controller.snapshot(self._key(layer))
        setpoint = self.cell_setpoint(layer)
        pv = self._pv.get(layer, setpoint)
        return {
            "layer": layer,
            "injects": self._injects(layer),
            "alive": layer in live,
            "setpoint": setpoint,
            "process_variable": pv,
            "error": setpoint - pv,
            "p_term": self.gains()[0] * (setpoint - pv),
            "i_term": state.i_term,
            "d_term": state.d_term,
            "output": self._strength.get(layer, self.config.base_strength),
            "saturated": self._saturated.get(layer, False),
            "step_count": state.step_count,
        }

    def status(self) -> dict[str, Any]:
        live = set(self.live_cells())
        cells = [self._cell_status(layer, live) for layer in self.cells]
        active = [self._cell_status(layer, live) for layer in sorted(live)]
        kp, ki = self.gains()

        def mean_of(name: str) -> float:
            return float(np.mean([cell[name] for cell in active])) if active else 0.0

        return {
            "goal": self.goal,
            "calibrated": self.calibrated,
            "readout_layer": self.readout_layer,
            "alive_cells": len(live),
            "setpoint": self.setpoint,
            "process_variable": mean_of("process_variable"),
            "error": self.error,
            "p_term": mean_of("p_term"),
            "i_term": mean_of("i_term"),
            "d_term": mean_of("d_term"),
            "output": self.current_strength,
            "integral_saturated": any(cell["saturated"] and cell["injects"] for cell in active),
            "step_count": max((cell["step_count"] for cell in active), default=0),
            "kp": kp,
            "ki": ki,
            "kd": self.config.kd,
            "cells": cells,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "pass": self._pass,
            "recorded_pass": self._recorded_pass,
            "last_layer": self._last_layer,
            "consensus": self._consensus,
            "cells": {
                str(layer): {
                    "pid": self.controller.snapshot(self._key(layer)).to_dict(),
                    "filtered": self._filtered.get(layer),
                    "pv": self._pv.get(layer),
                    "error": self._error.get(layer),
                    "strength": self._strength.get(layer),
                    "saturated": self._saturated.get(layer, False),
                    "seen": seen,
                }
                for layer, seen in self._seen.items()
            },
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.reset()
        self.goal = snapshot["goal"]
        self._pass = snapshot["pass"]
        self._recorded_pass = snapshot["recorded_pass"]
        self._last_layer = snapshot["last_layer"]
        self._consensus = snapshot["consensus"]
        for key, cell in snapshot["cells"].items():
            layer = int(key)
            self.controller.restore(self._key(layer), PIDState.from_dict(cell["pid"]))
            self._seen[layer] = cell["seen"]
            self._saturated[layer] = cell["saturated"]
            for name, store in (
                ("filtered", self._filtered),
                ("pv", self._pv),
                ("error", self._error),
                ("strength", self._strength),
            ):
                if cell[name] is not None:
                    store[layer] = cell[name]

    def reset(self) -> None:
        self.controller.reset()
        self.alignment_history = deque(maxlen=MAX_HISTORY_LENGTH)
        self.strength_history = deque(maxlen=MAX_HISTORY_LENGTH)
        self._filtered, self._pv, self._error, self._strength = {}, {}, {}, {}
        self._saturated, self._seen = {}, {}
        self._pass = 0
        self._recorded_pass = 0
        self._last_layer = None
        self._consensus = None


class SteeringHook:
    """Forward hook on one decoder layer: a cell that senses, and injects if it is an actuator."""

    def __init__(
        self,
        steering_vector: SteeringVector,
        config: SteeringConfig,
        homeostat: AdaptiveHomeostat | None = None,
        capability_subspace: torch.Tensor | None = None,
        injects: bool = True,
    ):
        self.steering_vector = steering_vector
        self.config = config
        self.homeostat = homeostat or AdaptiveHomeostat(config)
        self.capability_subspace = capability_subspace
        self.injects = injects
        self._last_strength = config.base_strength
        self._direction_cache: torch.Tensor | None = None
        self._direction_key: tuple[torch.device, torch.dtype] | None = None

    @property
    def layer(self) -> int:
        return self.steering_vector.layer

    def __call__(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        if isinstance(output, tuple):
            hidden_states, rest = output[0], output[1:]
        else:
            hidden_states, rest = output, ()

        direction = self._direction(hidden_states.device, hidden_states.dtype)
        strength = self.homeostat.sense(self.layer, hidden_states, direction)
        modified = hidden_states
        if self.injects:
            self._last_strength = strength
            modified = hidden_states + strength * direction.unsqueeze(0).unsqueeze(0)

        if rest:
            return (modified,) + rest
        return modified

    def _direction(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The direction this hook injects, projected and cached.

        The projection depends only on the vector and the subspace, so recomputing
        it per forward pass would repeat a Gram-Schmidt sweep on every token.
        """
        key = (device, dtype)
        if self._direction_key == key and self._direction_cache is not None:
            return self._direction_cache

        steer_vec = self.steering_vector.vector.to(device=device, dtype=dtype)
        if self.config.orthogonal_projection and self.capability_subspace is not None:
            steer_vec, _ = project_steering_direction(
                steer_vec,
                self.capability_subspace.to(device=device, dtype=dtype),
                layer=self.layer,
                rank=self.config.capability_subspace_rank,
            )

        self._direction_cache = steer_vec
        self._direction_key = key
        return steer_vec


# --- calibration ---------------------------------------------------------------------


def _transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return cast(nn.ModuleList, getattr(model.model, "layers"))  # noqa: B009
    if hasattr(model, "layers"):
        return cast(nn.ModuleList, getattr(model, "layers"))  # noqa: B009
    raise ValueError("Cannot find transformer layers")


def _input_device(model: nn.Module) -> torch.device:
    inner = getattr(model, "model", model)
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None:
        return embed.weight.device
    return next(model.parameters()).device


@dataclass
class _ProjectionRecorder:
    """Per-position projections onto each layer's direction, read before any injection there."""

    directions: dict[int, torch.Tensor]
    inject: dict[int, float] = field(default_factory=dict)
    records: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    def hook(self, layer: int):
        def _hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            direction = self.directions[layer].to(hidden.device, hidden.dtype)
            start = FIRST_CALIBRATION_POSITION if hidden.shape[1] > 1 else 0
            projection = (hidden[0, start:].float() @ direction.float()).detach().cpu()
            self.records.setdefault(layer, []).append(projection)
            if layer not in self.inject:
                return output
            hidden = hidden + self.inject[layer] * direction
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

        return _hook

    def run(self, model: nn.Module, tokenizer, texts: Sequence[str], max_length: int) -> None:
        layers = _transformer_layers(model)
        handles = [
            layers[layer].register_forward_hook(self.hook(layer)) for layer in self.directions
        ]
        device = _input_device(model)
        try:
            for text in texts:
                inputs = tokenizer(
                    text, return_tensors="pt", max_length=max_length, truncation=True
                ).to(device)
                with torch.no_grad():
                    model(**inputs)
        finally:
            for handle in handles:
                handle.remove()


def _layer_statistics(records: list[torch.Tensor], lift: float) -> LayerCalibration:
    tokens = torch.cat(records)
    passage_means = torch.stack([record.mean() for record in records])
    return LayerCalibration(
        resting_mean=float(tokens.mean().item()),
        resting_sigma=float(passage_means.std().item()) if len(records) > 1 else 0.0,
        token_sigma=float(tokens.std().item()) if tokens.numel() > 1 else 0.0,
        lift=lift,
    )


def resolve_readout_layer(config: SteeringConfig, vector_layers: Sequence[int]) -> int:
    """The sensor-only cell: configured, else the layer above the top actuator, else the top."""
    actuators = [layer for layer in vector_layers if layer in config.steering_layers]
    if not actuators:
        raise ValueError(
            f"no steering vector at any configured steering layer {config.steering_layers}"
        )
    top = max(actuators)
    if config.readout_layer is not None:
        if config.readout_layer not in vector_layers:
            raise ValueError(f"no steering vector at readout layer {config.readout_layer}")
        return config.readout_layer
    return top + 1 if top + 1 in vector_layers else top


def calibrate_alignment(
    model: nn.Module,
    tokenizer,
    vectors: dict[int, SteeringVector],
    config: SteeringConfig,
    texts: Sequence[str] | None = None,
    max_length: int = 128,
) -> AlignmentCalibration:
    """Measure every cell's resting state and the lift it reads at the reference strength.

    Two passes over the corpus: unsteered, recording every cell, then steered at
    ``config.base_strength`` on every actuator, recording every cell again before
    its own injection. Each cell's lift is its mean rise per unit of strength.
    Position 0 is excluded from both; see :data:`FIRST_CALIBRATION_POSITION`.
    """
    if config.base_strength <= 0:
        raise ValueError("calibration needs a positive base_strength to measure the lift against")
    corpus = list(texts) if texts is not None else list(CAPABILITY_CORPUS)
    if not corpus:
        raise ValueError("calibration corpus is empty")

    readout = resolve_readout_layer(config, list(vectors))
    actuators = tuple(sorted(layer for layer in vectors if layer in config.steering_layers))
    sensors = tuple(sorted({*actuators, readout}))
    directions = {layer: _unit(vectors[layer].vector.float()) for layer in sensors}

    resting = _ProjectionRecorder(directions)
    resting.run(model, tokenizer, corpus, max_length)
    steered = _ProjectionRecorder(directions, inject=dict.fromkeys(actuators, config.base_strength))
    steered.run(model, tokenizer, corpus, max_length)

    layers = {}
    for layer in sensors:
        rest_mean = float(torch.cat(resting.records[layer]).mean().item())
        steered_mean = float(torch.cat(steered.records[layer]).mean().item())
        lift = (steered_mean - rest_mean) / config.base_strength
        layers[layer] = _layer_statistics(resting.records[layer], lift)

    calibration = AlignmentCalibration(
        layers=layers,
        actuators=actuators,
        sensors=sensors,
        reference_strength=config.base_strength,
        num_passages=len(corpus),
    )
    logger.info(
        "Alignment calibration over %d passages: cells %s, tissue gain %.4f sigma/unit, "
        "cell setpoints %s at strength %.2f",
        len(corpus),
        sensors,
        calibration.gain_z,
        {layer: round(calibration.setpoint_z(layer), 3) for layer in sensors},
        config.base_strength,
    )
    return calibration


# --- coordinator -----------------------------------------------------------------------


class CognitiveHomeostat(nn.Module):
    """Holds the goal's directions and subspaces, wires the tissue into a model."""

    def __init__(self, config: SteeringConfig):
        super().__init__()
        self.config = config
        self.steering_vectors: dict[int, SteeringVector] = {}
        self.capability_subspaces: dict[int, torch.Tensor] = {}
        self.hooks: dict[int, SteeringHook] = {}
        self._registered_hooks: list = []
        self.calibration: AlignmentCalibration | None = None
        self.homeostat = AdaptiveHomeostat(config)

    @property
    def goal(self) -> str:
        return self.homeostat.goal

    def add_steering_vector(self, layer: int, vector: SteeringVector):
        self.steering_vectors[layer] = vector
        self.homeostat.goal = vector.name
        logger.info(f"Added steering vector '{vector.name}' to layer {layer}")

    def add_steering_vectors(self, vectors: dict[int, SteeringVector]):
        for layer, vector in vectors.items():
            self.add_steering_vector(layer, vector)

    @property
    def actuator_layers(self) -> list[int]:
        return sorted(
            layer for layer in self.steering_vectors if layer in self.config.steering_layers
        )

    @property
    def readout_layer(self) -> int:
        return resolve_readout_layer(self.config, list(self.steering_vectors))

    def set_capability_subspaces(self, subspaces: dict[int, torch.Tensor]) -> None:
        """Install per-layer capability bases; hooks pick them up on the next attach."""
        for layer, subspace in subspaces.items():
            if subspace.ndim != 2:
                raise ValueError(
                    f"Layer {layer}: capability subspace must be (rank, hidden_dim), "
                    f"got shape {tuple(subspace.shape)}"
                )
            steering_vector = self.steering_vectors.get(layer)
            if steering_vector is not None and subspace.shape[-1] != steering_vector.vector.numel():
                raise ValueError(
                    f"Layer {layer}: capability subspace has hidden_dim "
                    f"{subspace.shape[-1]}, steering vector has "
                    f"{steering_vector.vector.numel()}"
                )

        self.capability_subspaces = dict(subspaces)
        logger.info(f"Installed capability subspaces for layers {sorted(subspaces)}")

    def estimate_capability_subspaces(
        self,
        model: nn.Module,
        tokenizer,
        texts: list[str] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Estimate and install the capability subspace for every steered layer."""
        subspaces = estimate_capability_subspace(
            model,
            tokenizer,
            layers=sorted(self.steering_vectors) or list(self.config.steering_layers),
            texts=texts,
            rank=self.config.capability_subspace_rank,
        )
        self.set_capability_subspaces(subspaces)
        return subspaces

    def calibrate(
        self, model: nn.Module, tokenizer, texts: Sequence[str] | None = None
    ) -> AlignmentCalibration:
        """Measure the cells' resting states and lifts, and rebuild the tissue around them."""
        self.calibration = calibrate_alignment(
            model, tokenizer, self.steering_vectors, self.config, texts=texts
        )
        self.homeostat = AdaptiveHomeostat(self.config, self.calibration, goal=self.goal)
        for hook in self.hooks.values():
            hook.homeostat = self.homeostat
        return self.calibration

    def attach_to_model(self, model: nn.Module):
        layers = _transformer_layers(model)
        readout = self.readout_layer
        cells = {layer: True for layer in self.actuator_layers}
        cells.setdefault(readout, False)

        for layer_idx, injects in sorted(cells.items()):
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range, skipping")
                continue
            hook = SteeringHook(
                steering_vector=self.steering_vectors[layer_idx],
                config=self.config,
                homeostat=self.homeostat,
                capability_subspace=self.capability_subspaces.get(layer_idx),
                injects=injects,
            )
            self.hooks[layer_idx] = hook
            self._registered_hooks.append(layers[layer_idx].register_forward_hook(hook))

        logger.info(
            "Attached %d cells: actuators %s, readout %d",
            len(self._registered_hooks),
            self.actuator_layers,
            readout,
        )

    def detach_from_model(self):
        for handle in self._registered_hooks:
            handle.remove()
        self._registered_hooks = []
        self.hooks = {}
        logger.info("Detached all steering hooks")

    def projected_direction(self, layer: int) -> tuple[torch.Tensor, float]:
        """The direction actually injected at ``layer``, and its retained norm share.

        Anything else that consumes the goal direction must read it from here rather
        than from the raw steering vector. ``SteeringCoupling`` in particular keeps
        its own copy in a buffer, so seeding it from the unprojected vector would
        leave the routing coupling steering toward a direction the residual-stream
        injection has already decided not to use.
        """
        vector = self.steering_vectors[layer].vector
        subspace = self.capability_subspaces.get(layer)
        if subspace is None or not self.config.orthogonal_projection:
            return vector, 1.0

        return project_steering_direction(
            vector,
            subspace.to(device=vector.device, dtype=vector.dtype),
            layer=layer,
            rank=self.config.capability_subspace_rank,
        )

    def get_capability_retention(self) -> dict[int, float]:
        """Share of each steering vector's norm surviving its capability projection.

        A diagnostic, not a guarantee: it reports how much of the goal direction was
        orthogonal to general-task variation, not whether capability was preserved.
        Measuring that needs a held-out benchmark.
        """
        return {
            layer: self.projected_direction(layer)[1]
            for layer in self.steering_vectors
            if layer in self.capability_subspaces
        }

    def set_gains(
        self, kp: float | None = None, ki: float | None = None, kd: float | None = None
    ) -> dict[str, Any]:
        self.homeostat.set_gains(kp=kp, ki=ki, kd=kd)
        return self.pid_status()

    def pid_status(self) -> dict[str, Any]:
        return self.homeostat.status()

    def snapshot(self) -> dict[str, Any]:
        return self.homeostat.snapshot()

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.homeostat.restore(snapshot)

    def get_alignment_stats(self) -> dict[str, Any]:
        history = self.homeostat.alignment_history
        if not history:
            return {}
        strength_history = self.homeostat.strength_history

        stats: dict[str, Any] = {
            "current_alignment": history[-1],
            "mean_alignment": float(np.mean(list(history))),
            "min_alignment": min(history),
            "max_alignment": max(history),
            "setpoint": self.homeostat.setpoint,
            "current_strength": self.homeostat.current_strength,
            "alignment_history": list(history),
            "strength_history": list(strength_history),
            "pid": self.pid_status(),
        }
        if strength_history:
            stats["mean_strength"] = float(np.mean(list(strength_history)))
            stats["max_strength"] = max(strength_history)
            stats["min_strength"] = min(strength_history)
        return stats

    def reset(self):
        self.homeostat.reset()
