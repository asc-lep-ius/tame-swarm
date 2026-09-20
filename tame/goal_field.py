"""The goal field on the body (#54): what a cell is paid for, at the tissue's own setpoint.

#33 built the goal term and #39 read it on a fixture where the setpoint was a
script parameter. On the real model there is a setpoint already: the homeostat's
calibration measures, per cell, where that cell's projection sits when every
actuator injects at the certified reference strength
(``homeostat_calibration.calibrate_alignment``). That number pre-exists any
reward, which is the whole reason to use it -- a setpoint chosen to make the
reward interesting is a goal the experimenter holds, not one the tissue does.
``AlignmentCalibration.setpoint_projection`` is it, in the raw projection units
the goal term reads rather than the sigma units the homeostat filters in.

**The two readings are not the same quantity, and nothing here pretends they
are.** The homeostat reads the residual stream's whole projection at the last
position, filtered; the goal term reads the coordinate of what the experts
*added* at one converted layer, per token, with no filter. The setpoint is a
stream-level number and the reading is a layer-level delta, so on the body the
error ``setpoint - reading`` is large and a cell's push moves it only slightly:
the exact counterfactual in ``mob/goal.py`` is evaluated in the near-linear part
of its own absolute value, where a unit of push is worth about a unit of error
relieved whatever the error already is. The saturation the fixture exercises --
a cell aligned past the setpoint earning nothing more -- is a property of the
term that this substrate does not reach, and the README says so beside the run.
What survives on the body is the sign and the price: a contribution that carries
the layer's delta toward the calibrated level is paid, one that carries it away
is charged, one that does not move it earns exactly zero, and the rate is the
dose.

The dose is not free either. One unit of goal error is priced at what the
injection of that goal costs the model on held-out loss -- #28 measured +0.017
nats on the untrained body -- so a cell paid the full unit for relieving the
tissue's whole error is paid exactly what the goal is worth to the loss. Above
that the goal buys routing at the expense of the objective, which is the reward
farming #14's scale rule names.

Attaching is deliberately separate from injecting (``train.TAMETrainer._attach_field``,
#28). A field here is a thing cells are *paid* for, not a thing pushed into the
stream: the preregistered run is pay-only for both goals, and the calibration's
own steered pass is the only injection anywhere in this path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from homeostat_calibration import AlignmentCalibration
from mob import ConstantGoalField, MixtureOfBidders, mob_layers_by_index

logger = logging.getLogger(__name__)

# What one unit of goal error is worth, in held-out loss. #28 measured the
# ``truthful`` injection costing the untrained body +0.017 nats (3.274 -> 3.291,
# eleven run-to-run floors), which is the price of the goal being held rather
# than a guess at what it should be worth; #33's record fixes it as the unit the
# relative dose is expressed in. A sweep that wants another unit says so in its
# preregistration row rather than editing this.
GOAL_ERROR_UNIT_COST = 0.017


@dataclass(frozen=True)
class AttachedGoalField:
    """One goal field on one converted layer, with the calibration it was built from.

    Recorded rather than only logged: the setpoint is a measurement taken at
    training start, on that run's own model and corpus, and a run whose arithmetic
    is later questioned cannot recover it from the config -- the config carries
    the dose, and the dose is the only half of the price that was chosen.
    """

    goal: str
    layer: int
    dose: float
    setpoint: float
    setpoint_z: float
    resting_mean: float
    resting_sigma: float
    lift: float
    reference_strength: float
    num_passages: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_goal_doses(specs: Sequence[str]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """``<goal>=<dose>`` entries, repeated or comma-separated, into the two config tuples.

    Per goal rather than positional: the doses are the swept axis of #39's
    signature 1, they end up in the fingerprint as a bare tuple of floats, and a
    launch line reading ``--goal_dose 0.068,0.017`` leaves which field was swept
    to whoever remembers the argument order. Order is preserved, because the
    fingerprint's two tuples are parallel.
    """
    goals: list[str] = []
    doses: list[float] = []
    for spec in specs:
        for entry in spec.split(","):
            if not entry.strip():
                continue
            goal, separator, dose = entry.partition("=")
            if not separator:
                raise ValueError(
                    f"--goal_dose takes <goal>=<dose>, got {entry.strip()!r}: a bare number "
                    "leaves which goal it prices to whoever remembers the argument order"
                )
            try:
                doses.append(float(dose))
            except ValueError as exc:
                raise ValueError(
                    f"the dose of goal {goal.strip()!r} is not a number: {dose.strip()!r}"
                ) from exc
            goals.append(goal.strip())
    return tuple(goals), tuple(doses)


def paying_layers(calibration: AlignmentCalibration, converted: Sequence[int]) -> tuple[int, ...]:
    """Where a goal can be paid: a cell the calibration measured that is also a MoB layer.

    The certification decides the first half -- a layer the behavioural gate never
    passed is not a place this direction means anything (``certified_coupling_layers``)
    -- and the conversion range the second. A goal with no layer in both is refused
    by the caller rather than attached to nothing, the way ``_seed_coupling`` refuses
    to seed nothing: a run that pays no cell for a goal it was launched to price is
    a null by absence of mechanism.
    """
    return tuple(sorted(set(calibration.layers) & set(calibration.directions) & set(converted)))


def _layer_device(mob: MixtureOfBidders) -> torch.device:
    return next(mob.parameters()).device


def attach_goal_fields(
    model: nn.Module,
    goals: Sequence[str],
    doses: Sequence[float],
    calibrations: Mapping[str, AlignmentCalibration],
) -> tuple[AttachedGoalField, ...]:
    """Pay every converted layer for every goal certified there, at that layer's setpoint.

    Append-only on the layer (``MixtureOfBidders.attach_goal_field``), so whatever a
    previous call left behind comes off first: two goals certified at one layer are
    meant to stack there -- ``truthful`` and ``safe`` share layer 18, and the term
    sums over fields -- while the same goal attached twice is a silent doubling of
    its dose.

    The direction is the one the calibration measured along, taken from the
    calibration rather than re-derived, so the reading and the setpoint are
    expressed in the same units by construction and not by two call sites agreeing.
    """
    converted = mob_layers_by_index(model)
    for mob in converted.values():
        mob.detach_goal_fields()

    records: list[AttachedGoalField] = []
    for goal, dose in zip(goals, doses, strict=True):
        calibration = calibrations[goal]
        layers = paying_layers(calibration, list(converted))
        if not layers:
            raise ValueError(
                f"goal {goal!r} is calibrated at cells {sorted(calibration.layers)}, none of "
                f"which is a converted layer ({sorted(converted)}): no cell could be paid for it"
            )
        for layer in layers:
            mob = converted[layer]
            cell = calibration.layers[layer]
            field = ConstantGoalField(
                vector=calibration.directions[layer].to(
                    device=_layer_device(mob), dtype=torch.float32
                ),
                setpoint=calibration.setpoint_projection(layer),
                dose=dose,
            )
            mob.attach_goal_field(field)
            records.append(
                AttachedGoalField(
                    goal=goal,
                    layer=layer,
                    dose=dose,
                    setpoint=field.setpoint,
                    setpoint_z=calibration.setpoint_z(layer),
                    resting_mean=cell.resting_mean,
                    resting_sigma=cell.resting_sigma,
                    lift=cell.lift,
                    reference_strength=calibration.reference_strength,
                    num_passages=calibration.num_passages,
                )
            )
        logger.info(
            "Goal field %r: paid at MoB layers %s, dose %.5f per unit of goal error, "
            "setpoints %s in raw projection units (%s in the cells' own sigma), from a "
            "calibration over %d passages at reference strength %.2f",
            goal,
            list(layers),
            dose,
            {record.layer: round(record.setpoint, 3) for record in records if record.goal == goal},
            {
                record.layer: round(record.setpoint_z, 3)
                for record in records
                if record.goal == goal
            },
            calibration.num_passages,
            calibration.reference_strength,
        )
    return tuple(records)
