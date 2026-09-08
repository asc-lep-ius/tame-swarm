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
    damage, reports its reading here with a weight of zero. ``resting_sigma`` is
    the unit the cell's numbers are in (the slow sigma of its resting projection)
    and ``gain`` its calibrated lift per unit of strength in that unit; both are
    ``None`` on an uncalibrated loop.
    """

    layer: int
    injects: bool
    alive: bool
    weight: float
    resting_sigma: float | None
    gain: float | None
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
    ``setpoint - process_variable`` after damage as well as before it, as long as
    some live cell can still be moved; ``setpoint`` is the calibrated one,
    ``gain_z`` times the reference strength, whenever every cell is live.
    ``sensed_error`` is the plain mean error over the same cells, which still
    counts a cell nothing can correct. While no live cell can be moved ``error``
    is zero by its own rule, ``setpoint`` and ``process_variable`` fall back to the
    plain means over the live cells, and ``sensed_error`` is their gap --
    ``alive_cells`` and ``sensed_error`` tell that state from a tissue at
    setpoint. ``p_term``, ``i_term`` and ``d_term`` are plain means over the live
    cells, so ``p_term`` tracks ``kp * sensed_error`` rather than ``kp * error``.

    ``dispersion`` is how far the cells disagree: the controllability-weighted RMS
    of their errors about the consensus, in sigma. The consensus is a compromise
    the cells make, not a reading any one of them takes -- on the served tissue
    they read one continuation several sigma apart (#23) -- and this is the number
    that says so beside an ``error`` that may sit near zero. It is **not a target**:
    a healthy served tissue reads 2.0-2.3 sigma, and driving it toward zero would
    mean making the cells read alike, which no strength can do.

    ``sensed_dispersion`` is its pairing, as ``sensed_error`` is ``error``'s: the
    plain RMS over the same cells, counting the ones no action can move. The
    weighted number is what the controller is left with; the plain one is what the
    tissue is actually feeling. A low ``dispersion`` with a high
    ``sensed_dispersion`` is a tissue whose stuck cells disagree loudly and whose
    consensus cannot hear them -- and ``dispersion`` alone reads zero both when the
    cells agree and when exactly one of them can be moved.

    The four ``*_window`` fields are the loop's recent behaviour rather than its
    instantaneous state, computed over the last ``window_passes`` entries of the
    alignment history (see ``observability.convergence``): ``convergence_rate``
    positive means the error is shrinking, ~0 that it is holding, negative that it
    is growing, and ``error_rms_window`` beside it is what tells "0 because
    converged" from "0 because the loop is not acting". They are ``None`` on a
    history too short to split, and default to ``None`` here because
    ``AdaptiveHomeostat.status()`` -- which knows the tissue but not the window --
    does not produce them; ``observability.pid_status`` adds them.
    """

    goal: str
    calibrated: bool
    readout_layer: int | None
    alive_cells: int
    setpoint: float
    process_variable: float
    error: float
    sensed_error: float
    dispersion: float
    sensed_dispersion: float = 0.0
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
    convergence_rate: float | None = None
    error_mean_window: float | None = None
    error_rms_window: float | None = None
    window_passes: int = 0


class GainUpdate(BaseModel):
    """Runtime gain change; upper bounds are checked against the calibrated plant."""

    goal: str | None = Field(default=None, description="Goal loop to tune; default: the loaded one")
    kp: float | None = Field(default=None, ge=0.0)
    ki: float | None = Field(default=None, ge=0.0)
    kd: float | None = Field(default=None, ge=0.0)
    adaptive: bool | None = Field(
        default=None, description="Turn the loop on or off without re-extracting"
    )


class LayerCouplingStatus(BaseModel):
    """One MoB layer's coupling: whether it is there, how far it has ramped, how much it moves.

    ``delta_norm_fraction_mean``/``max`` are the share of the hidden norm the
    perceived stream differs from the stream by, on the last forward -- the
    coupling's share of what the confidence heads see. That share, not a
    per-bid attribution, is what this surface reports: attributing a bid to the
    coupling needs a second head forward per token on the unshifted stream, which
    is a measurement to run offline (``scripts/benchmark_coupling.py``) rather
    than a cost to put in every served token.

    ``detector_norm`` is the one number that separates a coupling that has learned
    something from one that is attached and inert: the receptor is zero-initialised,
    so a norm of zero means the perceived stream is the stream.
    """

    layer: int
    attached: bool
    active: bool
    coupling_step: int
    warmup_steps: int
    warmup_progress: float
    beta_effective: float
    delta_norm_fraction_mean: float | None
    delta_norm_fraction_max: float | None
    steering_alignment_mean: float | None
    detector_norm: float | None


class CouplingStatus(BaseModel):
    """Whether the goal is shaping which experts activate, right now.

    ``coupling_mode`` is ``off`` when no :class:`~coupling.SteeringCoupling` is
    attached anywhere, ``inert`` when one is but its receptor norm is zero,
    ``warming`` while the ramp is still below ``coupling_beta``, and ``active``
    once it is not.

    ``steering_routing_correlation`` is, per expert, the Pearson correlation over
    the trace window between a token's alignment with the **injected goal
    direction** and whether that expert won a slot on it, averaged over the MoB
    layers that carry a direction. It does not require a coupling: the tissue's
    hooks put the goal direction into the stream the confidence heads read either
    way, so with ``coupling_mode="off"`` this is the *additive* baseline -- what
    the goal does to routing without perception modulation -- which is exactly the
    number the coupled arm has to beat for Phase 1's multiplicative claim.
    ``correlation_basis`` says which of the two a reader is looking at. ``None``
    when no goal direction reaches a MoB layer, or the window is too short.
    """

    goal: str
    coupling_mode: str
    layers: list[LayerCouplingStatus] = Field(default_factory=list)
    beta_effective_mean: float | None = None
    warmup_progress: float | None = None
    steering_routing_correlation: list[float | None] | None = None
    correlation_basis: str = ""
    goal_alignment_mean: float | None = None
    trace_tokens: int = 0


class RoutingLayerHealth(BaseModel):
    """One MoB layer's gate over the trace window. ``win_share`` sums to ``top_k``, not to 1."""

    layer: int
    tokens: int
    top1_mean: float
    top1_median: float
    top1_saturated_fraction: float
    effective_experts: float
    win_share: list[float] = Field(default_factory=list)


class RoutingHealthMetrics(BaseModel):
    """Is the gate degenerate? The confound #11 named, as a live reading.

    ``effective_experts`` is ``exp(entropy(routing_weights))`` over the winners: it
    equals ``top_k`` under a uniform share and falls to 1.0 when the gate has
    collapsed onto a single winner. ``degenerate`` is the flag: the pooled count
    within ``DEGENERATE_EFFECTIVE_MARGIN`` of one expert, or more than
    ``DEGENERATE_SATURATION_FRACTION`` of tokens routed at a top-1 weight above
    ``ROUTING_SATURATION_THRESHOLD``. It matters here because coupling's measured
    influence on routing is read *through* the gate: near-argmax, the influence
    and the ablation that tests it are both uninterpretable.
    """

    top_k: int
    num_experts: int
    tokens: int
    top1_mean: float
    top1_median: float
    top1_saturated_fraction: float
    effective_experts: float
    win_share: list[float] = Field(default_factory=list)
    degenerate: bool = False
    layers: list[RoutingLayerHealth] = Field(default_factory=list)


class CertifiedEffect(BaseModel):
    """The behavioural gate's measurement for the served goal, from the certification record.

    Not a live number and never recomputed here: it is what
    ``scripts/validate_steering.py`` measured at ``layers`` and ``strength`` on the
    certified model. ``control_effect`` is ``None`` for a configuration whose
    instruction-prefix control was not re-quoted at those layers. The live
    counterpart is ``/metrics/outcome``.
    """

    effect: float
    random_max: float | None
    control_effect: float | None
    layers: list[int] = Field(default_factory=list)
    strength: float | None = None
    model: str | None = None


class LayerSteeringQuality(BaseModel):
    """One steered layer: the injected direction's norm, and the diagnostics beside it.

    ``norm`` is of the direction actually injected -- after the capability
    projection -- so it is ``capability_retention`` rather than the unit norm the
    extractor guarantees. ``separability`` is PCA separability of the two arms'
    reads, carried as a diagnostic and read by nothing that decides (#3): prompt
    surface features separate especially cleanly, so it cannot certify quality.
    """

    layer: int
    injects: bool
    norm: float
    separability: float | None = None
    capability_retention: float | None = None


class SteeringQualityMetrics(BaseModel):
    """Are the served vectors what the gate passed, and how do the goals sit relative to each other?

    ``inter_goal_cosine`` is over ``goals`` -- every goal extracted in this
    process, which on a freshly started server is the one it serves -- at
    ``cosine_layer``, the highest layer all of them have a vector at.
    """

    goal: str
    certified: bool
    source: str
    pair_format: str
    pair_count: int
    tier_counts: dict[str, int] = Field(default_factory=dict)
    extraction_layers: list[int] = Field(default_factory=list)
    readout_layer: int | None = None
    base_strength: float | None = None
    strength_band: list[float] | None = None
    layers: list[LayerSteeringQuality] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    cosine_layer: int | None = None
    inter_goal_cosine: list[list[float]] = Field(default_factory=list)
    certified_effect: CertifiedEffect | None = None


class PCAPoint(BaseModel):
    goal: str
    layer: int
    coordinates: list[float]


class PCAProjection(BaseModel):
    """Goal directions projected onto their own principal components. A diagnostic, not a gate.

    Separability here certifies nothing (#3): prompt-surface features separate
    especially cleanly, and the gate that decides is behavioural. ``note`` carries
    what the reader has to know to read the picture -- above all that a server
    which has extracted one goal is showing that goal's *layers* spread out, not
    goals separating from each other.
    """

    components: int
    points: list[PCAPoint] = Field(default_factory=list)
    explained_variance_ratio: list[float] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    note: str = ""


class OutcomeArm(BaseModel):
    """One arm of the outcome probe on the goal's held-out pairs."""

    mean_log_odds: float
    accuracy: float
    mean_strength: float | None = None


class OutcomeMetrics(BaseModel):
    """Did regulating the variable change what the organism does? The falsifiable one.

    Every other metric on this surface is internal telemetry: it can say the wire
    is connected, never that connecting it helped. This runs the served goal's
    held-out pairs on the arms that differ only in the intervention -- ``served``
    as configured, ``unsteered`` with the hooks off, and ``constant`` at the
    certified fixed strength when the loop is adaptive (#4's value test) -- and
    reports the deltas.

    Every delta is **paired**: the same pair is scored in each arm and the
    differences averaged, so the between-pair variance -- most of the variance,
    since the pairs differ in topic and difficulty -- cancels.
    ``served_minus_unsteered_standard_error`` is the standard error of that paired
    mean, and is ``None`` when an arm dropped a pair the others kept, which breaks
    the pairing.

    ``beats_random`` compares ``served_minus_unsteered_log_odds`` against the
    certification's strongest matched random direction. **A delta at or below that
    floor is the value that reads "steering did not help"**, and it is reachable:
    #6's coupling ablation already found the held-out effect null at 500 steps. It
    is ``None`` -- not ``False`` -- when the floor does not describe this process,
    with ``floor_not_applicable`` saying why: the floor was measured at one model,
    one set of layers and one strength, and a served process can differ on all
    three. ``num_pairs`` defaults small enough to run inside a request, so the
    error bars are wide: **this is a smoke number**, read with its standard error,
    and the certification's 200-pair measurement is the one that certifies.

    ``adaptive_minus_constant_log_odds`` is #4's value test as a live contrast, and
    it conflates two things it cannot separate: the adaptive arm may win because
    the loop regulated, or simply because it injected more. ``mean_strength`` per
    arm is what tells them apart -- an arm that drifted well above the constant's
    reference strength was not measured at the same dose.

    ``stale`` is true once the served goal changed after the probe ran.
    """

    goal: str
    probed_at: str
    num_pairs: int
    arms: dict[str, OutcomeArm] = Field(default_factory=dict)
    served_minus_unsteered_log_odds: float
    served_minus_unsteered_standard_error: float | None = None
    served_minus_unsteered_accuracy: float = 0.0
    adaptive_minus_constant_log_odds: float | None = None
    certified_random_max: float | None = None
    beats_random: bool | None = None
    floor_not_applicable: str | None = None
    held_out_perplexity_steered: float | None = None
    held_out_perplexity_unsteered: float | None = None
    stale: bool = False


class LatencyStats(BaseModel):
    """One route's window of recent requests. Not history -- that is MLflow's (#5, #7)."""

    route: str
    requests: int
    p50_s: float
    p95_s: float
    p99_s: float
    tokens_per_second: float


class SystemHealthStatus(BaseModel):
    """Every current-state snapshot in one call. Composite of live reads, no history.

    Each component is ``None`` when the thing it describes is not there -- steering
    disabled, no MoB layer, no coupling, nothing probed yet -- rather than a zeroed
    model, so "absent" and "at rest" stay distinguishable.
    """

    health: HealthResponse
    swarm: SwarmStatus | None = None
    pid: PIDStatus | None = None
    coupling: CouplingStatus | None = None
    routing: RoutingHealthMetrics | None = None
    steering: SteeringQualityMetrics | None = None
    outcome: OutcomeMetrics | None = None
    latency: list[LatencyStats] = Field(default_factory=list)
