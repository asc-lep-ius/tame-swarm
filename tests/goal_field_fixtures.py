"""A tiny body with goal fields on it: the substrate #54's wiring is pinned on.

``scripts/synthetic_economy.py`` is the fixture #39 read the dial on, and its
``TypeGoalField`` prices a planted correction rather than a direction in a
residual stream. What #54 adds is the other path -- a real decoder stack with
converted layers, a calibration measured along a direction, and
``goal_field.attach_goal_fields`` between them -- so the properties #33 pinned
on the fixture are pinned here too, on the thing the GPU arms actually run.

The adapters are perturbed on purpose. A freshly built expert has a zero output
projection, so every contribution is identically zero, the goal term is zero at
any dose, and a test that asked "does the dose change anything" would pass
against a layer that had been handed no field at all.
"""

import torch
import torch.nn as nn

from goal_field import attach_goal_fields
from homeostat_calibration import AlignmentCalibration, LayerCalibration, unit_vector
from mob import MoBConfig, apply_mob_to_model, get_mob_layers, update_all_mob_from_loss
from mob.wealth import ValueSummary

from .conftest import TINY_HIDDEN_DIM, TINY_INTERMEDIATE_DIM, build_tiny_causal_lm

# Two converted layers out of the four the tiny stack has, so "certified at a
# layer the conversion does not cover" is a state this fixture can be put in.
BODY_LAYERS = (1, 2)
BODY_CONFIG = MoBConfig(
    hidden_dim=TINY_HIDDEN_DIM,
    intermediate_dim=TINY_INTERMEDIATE_DIM,
    adapter_rank=4,
    adapter_alpha=4.0,
)
REFERENCE_STRENGTH = 4.0
CALIBRATION_PASSAGES = 8


def body_calibration(
    layers=BODY_LAYERS, seed: int = 7, lift: float = 0.6, readout: int | None = None
) -> AlignmentCalibration:
    """A calibration of the shape ``calibrate_alignment`` returns, without a corpus.

    The numbers are made up and their *relations* are not: the direction is a
    unit vector, as the real one is after ``unit_vector``, and the setpoint the
    fields are built at is ``resting_mean + lift * reference_strength``, which is
    what ``setpoint_projection`` has to reproduce.

    ``readout`` is the shape a real calibration has and this fixture otherwise
    cannot produce: ``extract_steering_vectors`` extracts a vector at
    ``readout_layer`` as well as at the actuators, so ``sensors`` strictly
    exceeds ``actuators`` and ``calibration.layers`` holds a cell the behavioural
    gate never passed. Without it every test here runs on a calibration whose
    sensors and actuators coincide, which is exactly the case that cannot catch
    a missing certification gate.
    """
    generator = torch.Generator().manual_seed(seed)
    sensors = tuple(sorted({*layers, *(() if readout is None else (readout,))}))
    return AlignmentCalibration(
        layers={
            layer: LayerCalibration(
                resting_mean=0.25 * layer, resting_sigma=0.5, token_sigma=0.9, lift=lift
            )
            for layer in sensors
        },
        actuators=tuple(layers),
        sensors=sensors,
        reference_strength=REFERENCE_STRENGTH,
        num_passages=CALIBRATION_PASSAGES,
        directions={
            layer: unit_vector(torch.randn(TINY_HIDDEN_DIM, generator=generator))
            for layer in sensors
        },
    )


def paid_body(
    goals=(), doses=(), calibrations=None, certified=None, seed: int = 0, device=None
) -> nn.Module:
    """A converted tiny stack, deterministic in ``seed``, paying for ``goals`` at ``doses``.

    ``device`` is placed before the fields go on, as the trainer places the model
    before ``_attach_goal_fields``: the calibration's directions live on the CPU
    and have to reach the device their layer ended up on. ``certified`` is the
    behavioural gate the trainer hands in; it defaults to each calibration's
    actuators, which is what the gate passes for a real goal, so a test that
    cares about the readout cell has to say so.
    """
    torch.manual_seed(seed)
    model = apply_mob_to_model(
        build_tiny_causal_lm(), BODY_CONFIG, layers_to_modify=list(BODY_LAYERS)
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("_B.weight"):
                parameter.normal_(std=0.1)
    model.train()
    if device is not None:
        model = model.to(device)
    if goals:
        calibrations = calibrations or {}
        attach_goal_fields(
            model,
            goals,
            doses,
            calibrations,
            certified or {goal: calibrations[goal].actuators for goal in goals},
        )
    return model


def settle_body(model: nn.Module, seed: int = 1) -> list[torch.Tensor]:
    """One forward, the loss backward, the settlement: every layer's realised values.

    The backward is the real one -- summed per-token cross-entropy, which is what
    ``loss_gradient_scale=1`` means -- because the value a winner realises is read
    by a hook on the layer's own output gradient, and a loss that does not reach
    that tensor settles an economy in which nobody realised anything.
    """
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 64, (2, 8), device=next(model.parameters()).device)
    logits = model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
    ).logits
    per_token = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(input_ids.shape[0], -1)
    per_token.sum().backward()
    update_all_mob_from_loss(model, per_token.detach(), torch.ones_like(per_token))
    values = []
    for mob in get_mob_layers(model):
        assert mob.last_realised_values is not None
        values.append(mob.last_realised_values.clone())
    return values


def value_summaries(model: nn.Module) -> list[ValueSummary]:
    """Each layer's last settlement summary; valid only after :func:`settle_body`."""
    summaries = []
    for mob in get_mob_layers(model):
        assert mob.last_value_summary is not None
        summaries.append(mob.last_value_summary)
    return summaries
