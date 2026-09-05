from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    steering_strength: float | None = Field(
        default=None,
        description="Override steering strength (0.0-1.5). None = adaptive.",
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
    homeostasis: dict[str, float] | None = None
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
    """One layer's cell: its own reading, setpoint and controller terms."""

    layer: int
    injects: bool
    alive: bool
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
    """One goal's tissue, as the mean over its live cells, plus every cell."""

    goal: str
    calibrated: bool
    readout_layer: int | None
    alive_cells: int
    setpoint: float
    process_variable: float
    error: float
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
