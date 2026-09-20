"""The goal in the cell's value (#33, built for #39): a counterfactual on the tissue's goal error.

Realised value is the counterfactual loss reduction of an expert's contribution
against the shared base (``wealth.realised_values``). The goal enters it as a
second counterfactual in the same units: the reduction in the *tissue's goal
error* attributable to that contribution, priced at ``dose`` per unit of error.
The error is the homeostat's -- setpoint minus reading, ``tame/homeostat.py`` --
where the reading is the projection coordinate of what the tissue adds on this
token (every winner's contribution at its share) along the field's direction.

Three properties the cosine term #33 rejected does not have, each pinned in
``tests/test_stakes_dial.py``. A contribution that does not move the reading
earns exactly zero whatever the error: a tissue already holding its goal pays
nothing for holding it. A cell aligned beyond the setpoint earns nothing more:
the reduction peaks when the cell's push closes the error exactly and falls past
it, so overshoot is not rewarded and a large overshoot is charged. And with no
field attached, or a field at dose zero, realised value is bitwise what it was.

The reduction is computed exactly rather than to first order. A linearised
``|error|`` has a constant gradient, so its first-order term would pay a cell
the same for the last unit of error as for the first and keep paying past the
setpoint -- the saturation the operator's pre-mortem asked for is the whole
point, and it needs the absolute value evaluated on both sides.

A field gives its direction per token, ``(batch, seq, hidden)``, or once for
every token, ``(hidden,)``. The direction need not be a unit vector: the reading
is the coordinate ``delta . d / |d|^2``, so along a unit direction it is the plain
projection and along the fixture's planted correction it is the fraction of that
correction the tissue delivered (``scripts/synthetic_economy.py``). The residual
stream's own projection -- what a real-model field would add as an offset to the
reading -- is the GPU arm's to define and is deliberately not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class GoalField(Protocol):
    """What the layer needs from a goal field: a setpoint, a dose, and a direction per token.

    Both read-only, so a frozen field satisfies this: the layer never writes
    either, and a mutable attribute satisfies a read-only protocol member while
    the reverse is not true -- ``ConstantGoalField`` is frozen because a dose that
    can be edited after attachment is a dose the fingerprint no longer describes.
    """

    @property
    def setpoint(self) -> float: ...

    @property
    def dose(self) -> float: ...

    def direction(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ConstantGoalField:
    """One direction for every token: the shape a served steering direction has."""

    vector: torch.Tensor
    setpoint: float
    dose: float

    def __post_init__(self) -> None:
        if self.vector.dim() != 1:
            raise ValueError(
                f"a constant goal direction is one vector, got shape {self.vector.shape}"
            )
        if self.dose < 0.0:
            raise ValueError(f"dose must be non-negative, got {self.dose}")

    def direction(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.vector


def error_relieved(reading: torch.Tensor, push: torch.Tensor, setpoint: float) -> torch.Tensor:
    """The atom of every stress signal: error without the cell, minus error with it.

    ``reading`` is what the tissue holds *with* this cell's ``push`` already in
    it, so removing the cell removes its own term and leaves the other winners in
    place. Named and separate because it is the shape the reward-signal slot in
    ``mob/ledger.py`` probes a plugged signal against (#40): a signal built on
    this is a stress by construction, and one that is not has to say why.
    """
    error_with = (setpoint - reading).abs()
    error_without = (setpoint - (reading - push)).abs()
    return error_without - error_with


def goal_error_reduction(
    contributions: torch.Tensor,
    routing_weights: torch.Tensor,
    direction: torch.Tensor,
    setpoint: float,
) -> torch.Tensor:
    """``|error without the slot| - |error with it``, per winner slot, at the slot's share.

    ``contributions`` is ``(batch, seq, top_k, hidden)`` per unit share, as the
    value hook sees it; ``routing_weights`` is ``(batch, seq, top_k)``, the share
    each slot actually held. The reading is the sum over slots of share times
    coordinate, and removing one slot removes its own term -- the counterfactual
    is against the tissue without that cell, the other winners left in place.
    Accumulated in float32 for the reason ``realised_values`` is.
    """
    d = direction.to(torch.float32)
    if d.dim() == 1:
        d = d.view(1, 1, -1)
    norm_sq = (d * d).sum(dim=-1, keepdim=True)
    dots = (contributions.float() * d.unsqueeze(-2)).sum(dim=-1)
    # A zero direction on a token is no field on that token, not an infinity.
    coordinates = torch.where(
        norm_sq > 0,
        dots / norm_sq.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(dots),
    )
    pushes = routing_weights.float() * coordinates
    reading = pushes.sum(dim=-1, keepdim=True)
    return error_relieved(reading, pushes, setpoint)


def goal_terms(
    contributions: torch.Tensor,
    routing_weights: torch.Tensor,
    fields: Sequence[GoalField],
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Every field's priced reduction, summed, per winner slot and per unit share.

    Per unit share so that it adds to ``realised_values`` on the same footing: the
    reward multiplies value by the slot's share, and share times this term is the
    priced reduction the slot actually bought. ``None`` with no field attached, so
    the caller can leave realised value untouched rather than add a zero to it.
    """
    if not fields:
        return None
    weights = routing_weights.float()
    total = torch.zeros(contributions.shape[:-1], dtype=torch.float32, device=contributions.device)
    for field in fields:
        reduction = goal_error_reduction(
            contributions, weights, field.direction(hidden_states), field.setpoint
        )
        per_unit_share = torch.where(weights > 0, reduction / weights.clamp_min(1e-30), reduction)
        total = total + field.dose * per_unit_share
    return total
