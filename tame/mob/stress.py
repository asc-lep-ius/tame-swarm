"""The tissue's error as the cell's own stress: unsigned, gated, paracrine (#59).

#33's goal term pays a cell the counterfactual reduction *it* caused in the
tissue's goal error. #39's body run then found that a cell paid in continuation
for its own value behaves like one paid for another cell's, and the literature
the redesign rests on says why that might be the wrong channel: what binds a
sub-agent to a collective in TAME is a **shared stress it must reduce**, not an
attributed payment it receives (Levin 2019; Pio-Lopez et al. 2023, where binding
was dispensable under a static objective and selected under change; Yoshida &
Man 2025, where prosociality appeared under affective coupling and not under
observation). This module is that channel.

Four properties, each one a line of the definition and a test in
``tests/test_stress_coupling.py``:

**Unsigned.** ``sigma = |e| / sigma_hat`` is a magnitude in units of the layer's
own resting spread, never a direction. A signed error fed into wealth is a
reward for a direction wearing a new name, and it is farmable by a
neighbourhood; the pre-mortem on this plan named that as the failure to guard
against, and the absolute value is the guard.

**Gated.** A stress inside one resting spread is indistinguishable from noise
and is not transmitted. Too much sharing is soup (Levin 2019) and stress is
informative only in a window (Pio-Lopez); the gate is where that window starts.

**Paracrine.** A cell pays for its neighbourhood's stress as well as its own, at
the homeostat's own consensus weights. This is the term that makes it a tissue
rather than four cells: it recruits cells to an error they did not cause and no
cell's counterfactual accounts for.

**Ownership wiped.** Every cell at a layer receives the same value, whoever
pushed. Nothing here reads which cell caused the error, and nothing here reads a
report -- a cell lowers its charge by *acting*, never by saying something. That
is invariant 2, and it is what keeps the constitution's strategyproofness
untouched: the charge is not a payment for a report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

# Which fraction of a layer's transmitted stress is its neighbourhood's, and
# where the gate opens. Both are fingerprinted, because an arm that shares more
# of its neighbours' error is not at parity with one that shares less.
DEFAULT_GATE_SIGMA = 1.0
DEFAULT_GAMMA = 0.0
DEFAULT_LAMBDA = 0.0

# Whether the paracrine fraction is a constant or a function of the tissue's own
# state (Levin 2019's selective coupling: a stressed tissue couples its cells
# more tightly than a resting one, which is the top-down half of nesting).
STRESS_GATE_FIXED = "fixed"
STRESS_GATE_STATE = "state"
SUPPORTED_GATE_MODES = frozenset({STRESS_GATE_FIXED, STRESS_GATE_STATE})

# Which channel a cell's continuation is charged through. The arms of #59, and a
# varying field: two arms that differ here are the experiment.
STRESS_ATTRIBUTED = "attributed"
STRESS_SHARED = "shared"
STRESS_MIXED = "mixed"
SUPPORTED_STRESS_COUPLINGS = frozenset({STRESS_ATTRIBUTED, STRESS_SHARED, STRESS_MIXED})


@dataclass(frozen=True)
class StressConfig:
    """What the charge costs, how far it spreads, and where the gate opens.

    Frozen for the reason ``ConstantGoalField`` is: a coupling that can be
    edited after attachment is one the fingerprint no longer describes.
    """

    stress_lambda: float = DEFAULT_LAMBDA
    gamma: float = DEFAULT_GAMMA
    gate_sigma: float = DEFAULT_GATE_SIGMA
    mode: str = STRESS_GATE_FIXED

    def __post_init__(self) -> None:
        if self.stress_lambda < 0.0:
            raise ValueError(f"stress_lambda must be non-negative, got {self.stress_lambda}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1], got {self.gamma}")
        if self.gate_sigma < 0.0:
            raise ValueError(f"gate_sigma must be non-negative, got {self.gate_sigma}")
        if self.mode not in SUPPORTED_GATE_MODES:
            raise ValueError(f"unknown stress gate mode {self.mode!r}")

    @property
    def live(self) -> bool:
        """Whether this coupling charges anything at all.

        ``stress_lambda = 0`` is the recorded economy: invariant 4 is that it
        reproduces it bitwise, and the cheapest way to keep that promise is for
        nothing to run.
        """
        return self.stress_lambda > 0.0


class StressSettings(Protocol):
    """The five fields a configuration carries this channel in."""

    @property
    def stress_coupling(self) -> str: ...

    @property
    def stress_lambda(self) -> float: ...

    @property
    def stress_gamma(self) -> float: ...

    @property
    def stress_gate_sigma(self) -> float: ...

    @property
    def stress_gate_mode(self) -> str: ...


def stress_from_config(config: StressSettings) -> StressConfig:
    """The coupling a configuration selects, refusing one whose name and price disagree.

    An arm called ``shared`` that charges nothing is the defect
    ``tests/test_no_silent_noops.py`` exists for -- a field that looks like it
    steers the mechanism and does not -- and it would produce a table in which
    two arms wear different names and ran the same economy. So the name binds:
    ``attributed`` charges nothing and the other two charge something.
    """
    coupling, price = config.stress_coupling, config.stress_lambda
    if coupling == STRESS_ATTRIBUTED and price > 0.0:
        raise ValueError(
            f"stress_coupling {coupling!r} is the arm that pays the attributed goal term and "
            f"charges no stress, but stress_lambda is {price}"
        )
    if coupling in (STRESS_SHARED, STRESS_MIXED) and price <= 0.0:
        raise ValueError(
            f"stress_coupling {coupling!r} charges the tissue's stress, but stress_lambda is "
            f"{price}: the arm would run the recorded economy under another name"
        )
    return StressConfig(
        stress_lambda=price,
        gamma=config.stress_gamma,
        gate_sigma=config.stress_gate_sigma,
        mode=config.stress_gate_mode,
    )


def cell_stress(reading: torch.Tensor, setpoint: float, resting_sigma: float) -> torch.Tensor:
    """``|setpoint - reading| / resting_sigma``, per token: a magnitude, in sigma.

    Not the squared error. A magnitude in sigma keeps the charge in the units
    the homeostat's ``z`` already uses, so a dose is interpretable as "this many
    resting spreads of error cost this much wealth", and it keeps the charge
    linear in the error where a square would make a large error dominate every
    small one.
    """
    if resting_sigma <= 0.0:
        raise ValueError(f"resting_sigma must be positive, got {resting_sigma}")
    return (setpoint - reading).abs() / resting_sigma


def transmitted(stress: torch.Tensor, gate_sigma: float) -> torch.Tensor:
    """The stress that passes the gate; zero where it does not.

    A hard gate rather than a soft one: the claim is that a stress inside the
    noise is *not information*, and a soft gate would pass a little of it and
    make the threshold a tuning knob rather than a statement.
    """
    return torch.where(stress > gate_sigma, stress, torch.zeros_like(stress))


def paracrine_share(
    own: torch.Tensor,
    neighbours: Sequence[torch.Tensor],
    weights: Sequence[float],
    gamma: float,
) -> torch.Tensor:
    """``(1 - gamma) * own + gamma * sum(w * neighbour)``, the weights renormalised.

    A neighbourhood that is empty, or whose consensus weights are all zero,
    contributes nothing and the share is the layer's own stress whatever
    ``gamma`` says -- the existing rule that a cell with no live actuator does
    not vote, met here from the other side. Renormalising rather than requiring
    normalised weights keeps the caller's arithmetic out of this: the homeostat's
    consensus weights are over all its actuators, and a neighbourhood is a subset
    of them.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    total = float(sum(weights))
    if not neighbours or total <= 0.0 or gamma == 0.0:
        return own
    if len(neighbours) != len(weights):
        raise ValueError(
            f"{len(neighbours)} neighbours and {len(weights)} weights are not a neighbourhood"
        )
    mixed = torch.zeros_like(own)
    for neighbour, weight in zip(neighbours, weights, strict=True):
        mixed = mixed + (weight / total) * neighbour
    return (1.0 - gamma) * own + gamma * mixed


def state_gamma(gamma: float, recent_stress: float, gate_sigma: float) -> float:
    """The paracrine fraction under ``state``: a stressed tissue couples tighter.

    The collective's own state is upstream of how tightly its cells are coupled,
    which is the top-down half of nesting and the reason this mode exists. The
    form is the smallest one that says it: the configured ``gamma`` scaled by how
    far the tissue's recent mean transmitted stress sits above the gate, saturating
    at one. At rest the cells are as independent as ``gamma`` allows; under stress
    they are coupled as tightly as the mechanism permits.
    """
    if gate_sigma <= 0.0:
        return gamma
    excess = max(0.0, recent_stress - gate_sigma) / gate_sigma
    return float(min(1.0, gamma * (1.0 + excess)))


def layer_stress(
    readings: Sequence[torch.Tensor],
    setpoints: Sequence[float],
    resting_sigmas: Sequence[float],
    config: StressConfig,
    neighbour_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """One layer's transmitted stress per token: its own field's, and its neighbourhood's.

    The first reading is the layer's own field and the rest are the
    neighbourhood's, in the order the caller's consensus weights are given. On
    the body that neighbourhood is the nearest converted layers above and below
    that carry the same goal. **On the differentiated fixture it is the layer's
    other goal field**, which is the same structure at the substrate's own scale:
    one tissue, two conflicting setpoints, and a cell charged for an error it did
    not cause. The fixture cannot have the body's neighbourhood -- it has one
    converted layer -- and an analogue is said to be one rather than left to read
    as the thing itself.
    """
    if not readings:
        raise ValueError("a layer with no goal field has no stress to transmit")
    if not len(readings) == len(setpoints) == len(resting_sigmas):
        raise ValueError("every reading needs its own setpoint and resting spread")
    gated = [
        transmitted(cell_stress(reading, setpoint, sigma), config.gate_sigma)
        for reading, setpoint, sigma in zip(readings, setpoints, resting_sigmas, strict=True)
    ]
    own, neighbours = gated[0], gated[1:]
    weights = list(neighbour_weights or [1.0] * len(neighbours))
    gamma = config.gamma
    if config.mode == STRESS_GATE_STATE:
        gamma = state_gamma(gamma, float(own.mean()), config.gate_sigma)
    return paracrine_share(own, neighbours, weights, gamma)
