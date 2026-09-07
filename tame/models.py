from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    steering_strength: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Hold the injection at this strength for the request. None = the served "
            "configuration (constant certified strength, or the loop when adaptive)."
        ),
    )
    goal: str | None = Field(
        default="truthful",
        description="Behavioral goal: truthful, reasoning, safe",
    )
    return_stats: bool = Field(
        default=False,
        description="Include MoB routing statistics in response",
    )


class GenerateResponse(BaseModel):
    response: str
    usage: dict[str, int]
    # The loop's stats: scalars plus histories and the per-cell PID status.
    homeostasis: dict[str, Any] | None = None
    mob_stats: dict[str, Any] | None = None


class SwarmStatus(BaseModel):
    num_experts: int
    expert_wealth: list[float]
    expert_usage: list[float]
    layers_modified: int


class HealthResponse(BaseModel):
    status: str
    gpu: str
    model_id: str
    architecture: str
    mob_active: bool
    steering_active: bool


class CellStatus(BaseModel):
    """One layer's cell: its own reading, setpoint and controller terms.

    ``weight`` is the cell's share of the tissue consensus: its calibrated gain
    per unit of strength while some actuator below it is live, and zero once none
    is -- so a cell the tissue's effort cannot move, calibrated so or made so by
    damage, reports its reading here with a weight of zero.
    """

    layer: int
    injects: bool
    alive: bool
    weight: float
    setpoint: float
    process_variable: float
    error: float
    p_term: float
    i_term: float
    d_term: float
    output: float
    saturated: bool
    step_count: int


class PIDStatus(BaseModel):
    """One goal's tissue as its consensus over the live cells, plus every cell.

    ``setpoint``, ``process_variable`` and ``error`` are the controllability-weighted
    means over the live cells that the shared integrator regulates, so ``error`` is
    ``setpoint - process_variable`` after damage as well as before it; ``setpoint``
    is the calibrated one, ``gain_z`` times the reference strength, whenever every
    cell is live. ``sensed_error`` is the plain mean error over the same cells,
    which still counts a cell nothing can correct.
    ``error`` is also zero while no live cell can be moved -- ``alive_cells`` and
    ``sensed_error`` tell that from a tissue at setpoint -- and ``p_term``,
    ``i_term`` and ``d_term`` are plain means over the live cells, so ``p_term``
    tracks ``kp * sensed_error`` rather than ``kp * error``.
    """

    goal: str
    calibrated: bool
    readout_layer: int | None
    alive_cells: int
    setpoint: float
    process_variable: float
    error: float
    sensed_error: float
    p_term: float
    i_term: float
    d_term: float
    output: float
    integral_saturated: bool
    step_count: int
    kp: float
    ki: float
    kd: float
    cells: list[CellStatus] = Field(default_factory=list)


class GainUpdate(BaseModel):
    """Runtime gain change; upper bounds are checked against the calibrated plant."""

    goal: str | None = Field(default=None, description="Goal loop to tune; default: the loaded one")
    kp: float | None = Field(default=None, ge=0.0)
    ki: float | None = Field(default=None, ge=0.0)
    kd: float | None = Field(default=None, ge=0.0)
    adaptive: bool | None = Field(
        default=None, description="Turn the loop on or off without re-extracting"
    )
