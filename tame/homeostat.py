"""The cognitive homeostat: one calibrated loop per goal, actuators at every steered layer.

#4 measured the loop the P-controller had been closing and found it never regulated
anything: ``cos(h, v)`` sits near zero unsteered, a unit of strength moves it by
about a hundredth, and ``target_alignment = 0.7`` was unreachable by a factor of
thirty, so the output was a near-constant offset. The redesign follows the README's
own metaphor -- a pH *buffer* -- rather than a thermostat:

- **Sensor.** One reading per token at a *readout* layer above the injections: the
  projection of the last position onto the goal direction, EMA-filtered, expressed
  as a z-score against the resting distribution measured unsteered on a calibration
  corpus (:func:`calibrate_alignment`). Per-token content noise is filtered; the
  slow tone of the stream is what is regulated.
- **Setpoint.** The lift the reference strength produces at the readout, in the same
  units -- the one bridge from geometry to a behaviour the gate certified. On
  resting content the loop settles at the reference strength; a stream dragged below
  its resting alignment is pushed harder, one carried above it less.
- **Actuator.** One strength per goal, broadcast to every steered layer and applied
  from the next token (a stated one-token dead time), clamped to the strength band
  the gate passes.
- **Local autonomy.** The pattern is stored distributively: each actuator keeps its
  own direction and resting calibration, and when the sensor stops firing it falls
  back to a local proportional rule on its own reading, discounting the passthrough
  of the layers below it so upper layers do not undo what lower ones injected.

Gains are derived from the measured plant by SIMC (``pid_controller.simc_pi_gains``)
unless the config pins them. Without a calibration the loop keeps the legacy
contract -- cosine toward ``target_alignment`` with a proportional gain -- so probes
and tests that never calibrate behave as before.
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
# Shortest closed-loop time constant the derivation will accept, in tokens: with an
# unfiltered reading SIMC would otherwise ask for a loop that corrects a deviation
# faster than the one-token dead time allows.
MIN_CLOSED_LOOP_TAU = 1.0
# Position 0 carries the attention-sink activation (norm ~17,000 against a median of
# 60-1,400 on Qwen3-1.7B); it is not a sample of the resting distribution.
FIRST_CALIBRATION_POSITION = 1
SIGMA_FLOOR = 1e-6


@dataclass(frozen=True)
class LayerCalibration:
    """Resting projection of the stream onto the goal direction at one layer, unsteered."""

    resting_mean: float
    # Standard deviation of per-passage means: the slow variability the filtered
    # reading actually has, and the unit the setpoint and gains are expressed in.
    resting_sigma: float
    token_sigma: float

    @property
    def sigma(self) -> float:
        return max(self.resting_sigma, SIGMA_FLOOR)


@dataclass(frozen=True)
class AlignmentCalibration:
    """Resting state per layer and the actuator's gain at the readout."""

    layers: dict[int, LayerCalibration]
    readout_layer: int
    # Projection units at the readout per unit of strength, every steered layer
    # injecting -- the additive passthrough plus whatever the network does with it.
    gain: float
    reference_strength: float
    num_passages: int

    def z(self, layer: int, projection: float) -> float:
        calibration = self.layers[layer]
        return (projection - calibration.resting_mean) / calibration.sigma

    @property
    def gain_z(self) -> float:
        """Setpoint units (readout sigmas) per unit of strength."""
        return self.gain / self.layers[self.readout_layer].sigma

    @property
    def setpoint_z(self) -> float:
        return self.gain_z * self.reference_strength


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm()
    return vector if norm == 0 else vector / norm


def _last_position(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states[:, -1, :].float().mean(dim=0)


class AdaptiveHomeostat:
    """One goal's loop: sensor filter, PID, broadcast strength and the local fallback."""

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
        self._filtered: float | None = None
        self._strength = config.base_strength
        self._sensor_ticks = 0
        self._actuator_ticks = 0
        self._steered: list[int] = []
        self._passthrough: dict[int, dict[int, float]] = {}
        self._applied: dict[int, float] = {}

    # --- configuration -----------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    @property
    def readout_layer(self) -> int | None:
        return self.calibration.readout_layer if self.calibration else None

    @property
    def setpoint(self) -> float:
        if self.calibration is not None:
            return self.calibration.setpoint_z
        return self.config.target_alignment

    @property
    def filter_time_constant(self) -> float:
        alpha = self.config.measurement_filter_alpha
        return (1.0 - alpha) / alpha

    def gains(self) -> tuple[float, float]:
        """``(kp, ki)``: pinned by the config where given, otherwise derived from the plant."""
        kp, ki = self.config.kp, self.config.ki
        if kp is not None and ki is not None:
            return kp, ki
        if self.calibration is None:
            return (LEGACY_KP if kp is None else kp), (0.0 if ki is None else ki)
        tau_filter = self.filter_time_constant
        tau_closed = self.config.closed_loop_tau
        if tau_closed is None:
            tau_closed = max(tau_filter, MIN_CLOSED_LOOP_TAU)
        derived_kp, derived_ki = simc_pi_gains(
            process_gain=self.calibration.gain_z,
            dead_time=1.0,
            time_constant=tau_filter,
            closed_loop_tau=tau_closed,
        )
        return (derived_kp if kp is None else kp), (derived_ki if ki is None else ki)

    def max_stable_ki(self) -> float | None:
        """Stability bound for the integral gain on the measured plant.

        With a one-token delay and a static gain ``K`` the integral-only loop has
        error ratio ``1 - K ki`` per token, so it is stable iff ``K ki < 2``. Exact
        for an unfiltered reading; conservative once the filter adds its own pole.
        """
        if self.calibration is None:
            return None
        return 2.0 / self.calibration.gain_z

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

    # --- sensing -------------------------------------------------------------------

    def reading(self, hidden_states: torch.Tensor, steering_vector: torch.Tensor) -> float:
        """Projection of the last position onto the goal direction (cosine when uncalibrated)."""
        state = _last_position(hidden_states)
        direction = _unit(steering_vector.float().to(state.device))
        if self.calibration is None:
            return float(F.cosine_similarity(state, direction, dim=0).item())
        return float((state @ direction).item())

    def compute_strength(self, hidden_states: torch.Tensor, steering_vector: torch.Tensor) -> float:
        """The sensor step: filter the reading, step the loop, broadcast the next strength."""
        if not self.config.adaptive:
            return self.config.base_strength

        reading = self.reading(hidden_states, steering_vector)
        alpha = self.config.measurement_filter_alpha
        self._filtered = (
            reading if self._filtered is None else alpha * reading + (1 - alpha) * self._filtered
        )
        if self.calibration is not None:
            process_variable = self.calibration.z(self.calibration.readout_layer, self._filtered)
        else:
            process_variable = self._filtered

        strength, _ = self.controller.step(
            self.goal, self.setpoint, process_variable, bias=self.config.base_strength
        )
        self.alignment_history.append(process_variable)
        self.strength_history.append(strength)
        self._strength = strength
        self._sensor_ticks += 1
        return strength

    # --- actuation -----------------------------------------------------------------

    def bind_layers(
        self, steered_layers: Sequence[int], directions: dict[int, torch.Tensor]
    ) -> None:
        """Tell the loop which layers inject, and how their directions overlap.

        The overlaps are what the local rule subtracts: an actuator's raw reading
        contains its lower neighbours' injections, and those are effort already spent
        rather than a surplus of content to correct.
        """
        self._steered = sorted(steered_layers)
        self._passthrough = {
            upper: {
                lower: float(
                    F.cosine_similarity(
                        directions[lower].float(), directions[upper].float(), dim=0
                    ).item()
                )
                for lower in self._steered
                if lower < upper
            }
            for upper in self._steered
        }

    @property
    def sensor_alive(self) -> bool:
        """True while the readout has fired within one pass of the actuators."""
        return self._actuator_ticks - self._sensor_ticks <= 1

    @property
    def current_strength(self) -> float:
        return self._strength

    def actuate(
        self, layer: int, hidden_states: torch.Tensor, steering_vector: torch.Tensor
    ) -> float:
        """Strength an actuator at ``layer`` applies this pass."""
        if not self.config.adaptive:
            return self.config.base_strength
        if not self._steered or layer == self._steered[0]:
            self._actuator_ticks += 1
            self._applied = {}
        strength = (
            self._strength
            if self.sensor_alive
            else self._local_strength(layer, hidden_states, steering_vector)
        )
        self._applied[layer] = strength
        return strength

    def _local_strength(
        self, layer: int, hidden_states: torch.Tensor, steering_vector: torch.Tensor
    ) -> float:
        """Local autonomy: a proportional rule on this layer's reading, minus what lies below."""
        reading = self.reading(hidden_states, steering_vector)
        low, high = self.config.min_strength, self.config.max_strength
        kp = self.config.kp
        if self.calibration is None or layer not in self.calibration.layers:
            gain = LEGACY_KP if kp is None else kp
            error = self.config.target_alignment - reading
            return min(high, max(low, self.config.base_strength + gain * error))
        passthrough = sum(
            applied * self._passthrough.get(layer, {}).get(lower, 0.0)
            for lower, applied in self._applied.items()
        )
        deficit = -self.calibration.z(layer, reading - passthrough)
        # One sigma of local deficit asks for the strength that lifts the readout by one.
        gain = (1.0 / self.calibration.gain_z) if kp is None else kp
        return min(high, max(low, self.config.base_strength + gain * deficit))

    # --- state -----------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        state = self.controller.snapshot(self.goal)
        kp, ki = self.gains()
        return {
            "goal": self.goal,
            "calibrated": self.calibrated,
            "sensor_alive": self.sensor_alive,
            "readout_layer": self.readout_layer,
            "setpoint": self.setpoint,
            "process_variable": self.setpoint - state.error,
            "error": state.error,
            "p_term": state.p_term,
            "i_term": state.i_term,
            "d_term": state.d_term,
            "output": state.output,
            "integral_saturated": state.saturated,
            "step_count": state.step_count,
            "kp": kp,
            "ki": ki,
            "kd": self.config.kd,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "pid": self.controller.snapshot(self.goal).to_dict(),
            "filtered": self._filtered,
            "strength": self._strength,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.goal = snapshot["goal"]
        self.controller.restore(self.goal, PIDState.from_dict(snapshot["pid"]))
        self._filtered = snapshot["filtered"]
        self._strength = snapshot["strength"]

    def reset(self) -> None:
        self.controller.reset()
        self.alignment_history = deque(maxlen=MAX_HISTORY_LENGTH)
        self.strength_history = deque(maxlen=MAX_HISTORY_LENGTH)
        self._filtered = None
        self._strength = self.config.base_strength
        self._sensor_ticks = 0
        self._actuator_ticks = 0
        self._applied = {}


class SteeringHook:
    """Forward hook on one decoder layer: inject the goal direction, and/or read the stream."""

    def __init__(
        self,
        steering_vector: SteeringVector,
        config: SteeringConfig,
        homeostat: AdaptiveHomeostat | None = None,
        capability_subspace: torch.Tensor | None = None,
        injects: bool = True,
        measures: bool = True,
    ):
        self.steering_vector = steering_vector
        self.config = config
        self.homeostat = homeostat or AdaptiveHomeostat(config)
        self.capability_subspace = capability_subspace
        self.injects = injects
        self.measures = measures
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
        modified = hidden_states
        if self.injects:
            strength = self.homeostat.actuate(self.layer, hidden_states, direction)
            self._last_strength = strength
            modified = hidden_states + strength * direction.unsqueeze(0).unsqueeze(0)
        if self.measures and self.config.adaptive:
            self.homeostat.compute_strength(modified, direction)

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
    """Per-position projections onto each layer's direction, injecting a fixed strength if told."""

    directions: dict[int, torch.Tensor]
    inject: dict[int, float] = field(default_factory=dict)
    records: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    def hook(self, layer: int):
        def _hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            direction = self.directions[layer].to(hidden.device, hidden.dtype)
            if layer in self.inject:
                hidden = hidden + self.inject[layer] * direction
            start = FIRST_CALIBRATION_POSITION if hidden.shape[1] > 1 else 0
            projection = (hidden[0, start:].float() @ direction.float()).detach().cpu()
            self.records.setdefault(layer, []).append(projection)
            if layer in self.inject:
                return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
            return output

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


def _layer_statistics(records: list[torch.Tensor]) -> LayerCalibration:
    tokens = torch.cat(records)
    passage_means = torch.stack([record.mean() for record in records])
    return LayerCalibration(
        resting_mean=float(tokens.mean().item()),
        resting_sigma=float(passage_means.std().item()) if len(records) > 1 else 0.0,
        token_sigma=float(tokens.std().item()) if tokens.numel() > 1 else 0.0,
    )


def resolve_readout_layer(config: SteeringConfig, vector_layers: Sequence[int]) -> int:
    """The layer the sensor reads: configured, else the one above the top actuator, else the top."""
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
    """Measure the resting alignment per layer and the actuator's gain at the readout.

    Two passes over the corpus: unsteered, recording every steered layer and the
    readout, then steered at ``config.base_strength`` recording the readout again.
    The gain is the mean lift per unit strength. Position 0 is excluded from both;
    see :data:`FIRST_CALIBRATION_POSITION`.
    """
    if config.base_strength <= 0:
        raise ValueError("calibration needs a positive base_strength to measure the gain against")
    corpus = list(texts) if texts is not None else list(CAPABILITY_CORPUS)
    if not corpus:
        raise ValueError("calibration corpus is empty")

    readout = resolve_readout_layer(config, list(vectors))
    actuators = sorted(layer for layer in vectors if layer in config.steering_layers)
    directions = {layer: _unit(vectors[layer].vector.float()) for layer in {*actuators, readout}}

    resting = _ProjectionRecorder(directions)
    resting.run(model, tokenizer, corpus, max_length)
    layers = {layer: _layer_statistics(records) for layer, records in resting.records.items()}

    steered = _ProjectionRecorder(
        {readout: directions[readout], **{layer: directions[layer] for layer in actuators}},
        inject=dict.fromkeys(actuators, config.base_strength),
    )
    steered.run(model, tokenizer, corpus, max_length)
    lift = float(torch.cat(steered.records[readout]).mean().item()) - layers[readout].resting_mean

    calibration = AlignmentCalibration(
        layers=layers,
        readout_layer=readout,
        gain=lift / config.base_strength,
        reference_strength=config.base_strength,
        num_passages=len(corpus),
    )
    logger.info(
        "Alignment calibration: readout layer %d, resting %.3f +/- %.3f (slow), gain %.4f/unit "
        "-> setpoint %.3f sigma at strength %.2f",
        readout,
        layers[readout].resting_mean,
        layers[readout].sigma,
        calibration.gain,
        calibration.setpoint_z,
        config.base_strength,
    )
    return calibration


# --- coordinator -----------------------------------------------------------------------


class CognitiveHomeostat(nn.Module):
    """Holds the goal's directions and subspaces, wires the loop into a model."""

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
        """Measure the resting state and gain, and rebuild the loop around them."""
        self.calibration = calibrate_alignment(
            model, tokenizer, self.steering_vectors, self.config, texts=texts
        )
        self.homeostat = AdaptiveHomeostat(self.config, self.calibration, goal=self.goal)
        for hook in self.hooks.values():
            hook.homeostat = self.homeostat
        if self.hooks:
            self._bind()
        return self.calibration

    def _bind(self) -> None:
        self.homeostat.bind_layers(
            self.actuator_layers,
            {
                layer: self.hooks[layer]._direction(torch.device("cpu"), torch.float32)
                for layer in self.actuator_layers
            },
        )

    def attach_to_model(self, model: nn.Module):
        layers = _transformer_layers(model)
        readout = self.readout_layer
        roles = {layer: (True, layer == readout) for layer in self.actuator_layers}
        if readout not in roles:
            roles[readout] = (False, True)

        for layer_idx, (injects, measures) in sorted(roles.items()):
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range, skipping")
                continue
            hook = SteeringHook(
                steering_vector=self.steering_vectors[layer_idx],
                config=self.config,
                homeostat=self.homeostat,
                capability_subspace=self.capability_subspaces.get(layer_idx),
                injects=injects,
                measures=measures,
            )
            self.hooks[layer_idx] = hook
            self._registered_hooks.append(layers[layer_idx].register_forward_hook(hook))

        self._bind()
        logger.info(
            "Attached %d steering hooks: actuators %s, readout %d",
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
