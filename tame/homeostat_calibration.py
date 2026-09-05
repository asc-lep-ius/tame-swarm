"""Calibrate a goal tissue: each cell's resting state and the lift it reads when steered.

Two teacher-forced passes over a corpus that samples the *served* regime: one
unsteered, recording every cell, and one with every actuator injecting the
reference strength, recording every cell again *before* its own injection --
which is what a cell reads at runtime. Each cell's setpoint is its lift at the
reference strength; the tissue gain the shared gains derive from is the mean
cell lift in each cell's own slow sigma.

The directions measured here must be the ones the hooks inject and read. With
the capability projection on, that is the projected direction, not the raw
steering vector: the two differ by a percent of norm and by several sigma of
reading, because the components the projection removes are the ones that carry
the stream's large resting offset.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from steering import SteeringConfig, SteeringVector

logger = logging.getLogger(__name__)

# Position 0 carries the attention-sink activation (norm ~17,000 against a median of
# 60-1,400 on Qwen3-1.7B); it is not a sample of the resting distribution.
FIRST_CALIBRATION_POSITION = 1
SIGMA_FLOOR = 1e-6
# Below this many passages the slow sigma is a guess and the derived gains with it.
MIN_CALIBRATION_PASSAGES = 4


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
    # The unit direction each cell was measured along -- what the hook must inject.
    directions: dict[int, torch.Tensor] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.num_passages < MIN_CALIBRATION_PASSAGES:
            raise ValueError(
                f"calibration needs at least {MIN_CALIBRATION_PASSAGES} passages, "
                f"got {self.num_passages}"
            )

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


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm()
    return vector if norm == 0 else vector / norm


def transformer_layers(model: nn.Module) -> nn.ModuleList:
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
        layers = transformer_layers(model)
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
    texts: Sequence[str],
    directions: dict[int, torch.Tensor] | None = None,
    max_length: int = 128,
) -> AlignmentCalibration:
    """Measure every cell's resting state and the lift it reads at the reference strength.

    Two passes over the corpus: unsteered, recording every cell, then steered at
    ``config.base_strength`` on every actuator, recording every cell again before
    its own injection. Each cell's lift is its mean rise per unit of strength.
    Position 0 is excluded from both; see :data:`FIRST_CALIBRATION_POSITION`.

    ``texts`` must sample the served regime (``steering_pipeline.calibration_texts``);
    there is no default because the general capability corpus was measured to sit
    about two sigma off it. ``directions`` overrides the unit steering vectors with
    the directions the hooks actually inject (the capability-projected ones).
    """
    if config.base_strength <= 0:
        raise ValueError("calibration needs a positive base_strength to measure the lift against")
    corpus = list(texts)
    if len(corpus) < MIN_CALIBRATION_PASSAGES:
        raise ValueError(
            f"calibration needs at least {MIN_CALIBRATION_PASSAGES} passages, got {len(corpus)}"
        )

    readout = resolve_readout_layer(config, list(vectors))
    actuators = tuple(sorted(layer for layer in vectors if layer in config.steering_layers))
    sensors = tuple(sorted({*actuators, readout}))
    measured = {
        layer: unit_vector((directions or {}).get(layer, vectors[layer].vector).float().cpu())
        for layer in sensors
    }

    resting = _ProjectionRecorder(measured)
    resting.run(model, tokenizer, corpus, max_length)
    steered = _ProjectionRecorder(measured, inject=dict.fromkeys(actuators, config.base_strength))
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
        directions=measured,
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
