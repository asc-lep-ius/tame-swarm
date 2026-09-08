"""Every metrics snapshot the API serves, built from live state (#5).

The rule the whole surface is written to: **a route is a read, not a
measurement.** Everything here is a pure function from :class:`app.TAMEApplication`
to a typed model, reading state the forward pass already produced -- the tissue's
own status, the coupling's last metrics, the routing trace's window, the
extraction record -- with no torch compute beyond one device read of a trace and,
for the PCA, an SVD of a matrix with at most a few dozen rows. Nothing here runs a
forward pass. The one thing on this surface that does is the outcome probe, and it
is a ``POST`` for exactly that reason.

The split that matters for reading the numbers: `dispersion`, the goal-routing
correlation and the outcome probe are claims the organism could fail -- the cells
disagree, the goal does or does not shape which experts activate, regulating the
variable does or does not change behaviour. Norms, PCA, latency and the routing
histogram are engineering telemetry. They are labelled as such and none of them
gates anything (#3's rule about PCA, kept).
"""

import logging
from datetime import datetime, timezone

import torch

from app import TAMEApplication
from contrastive_data import certification_for
from homeostat import AdaptiveHomeostat
from mob import MixtureOfBidders, RoutingTraceSummary, mob_layers_by_index
from mob.auction import ROUTING_SATURATION_THRESHOLD
from models import (
    CertifiedEffect,
    CouplingStatus,
    HealthResponse,
    LatencyStats,
    LayerCouplingStatus,
    LayerSteeringQuality,
    PCAPoint,
    PCAProjection,
    PIDStatus,
    RoutingHealthMetrics,
    RoutingLayerHealth,
    SteeringQualityMetrics,
    SwarmStatus,
    SystemHealthStatus,
)
from steering_pipeline import goal_similarity_matrix

logger = logging.getLogger(__name__)

# The rolling window, in passes, as a multiple of the loop's own settling scale
# (closed-loop tau plus the consensus dead time): 44 tokens on the served loop.
# Four of them is long enough for the earliest and latest quarters to be different
# regimes and short enough that the answer is about the request in flight.
CONVERGENCE_WINDOW_TIME_CONSTANTS = 4
# Below this many passes the quarters hold one or two readings each and the ratio
# between them is noise, so the rate is None rather than a number.
MIN_CONVERGENCE_PASSES = 8

# A gate is degenerate when the mixture has collapsed onto one expert. Allow each
# extra slot a tenth of an expert of mixing before saying so: at top_k = 2 the
# threshold is 1.1 effective experts, against the 2.0 a uniform share gives.
DEGENERATE_EFFECTIVE_MARGIN = 0.1
# Or when almost every token is routed at a top-1 weight above the saturation
# threshold, which is the near-argmax #11 is about even if the entropy is not
# quite collapsed.
DEGENERATE_SATURATION_FRACTION = 0.9

CORRELATION_BASIS_COUPLED = (
    "per-token alignment with the injected goal direction against expert wins, "
    "with a steering coupling attached: the perception-modulation effect"
)
CORRELATION_BASIS_BASELINE = (
    "per-token alignment with the injected goal direction against expert wins, "
    "no coupling attached: the additive baseline a coupled arm has to beat"
)


# --- the tissue ---------------------------------------------------------------------


def convergence(tissue: AdaptiveHomeostat) -> dict[str, float | int | None]:
    """Is the error shrinking, holding or growing over the loop's own settling scale?

    The rate is ``1 - mean|e|_late / mean|e|_early`` over the window's first and
    last quarters: positive means the error is shrinking, about zero that it is
    holding, negative that it is growing. It is deliberately not an identified
    time constant -- the 9-token measurement filter alone gives a lag-1
    autocorrelation near 0.9 whether the loop is open or closed, so an
    autocorrelation time cannot tell a converging loop from an inert one.

    Read it beside ``error_rms_window``. A loop sitting on resting content at its
    setpoint reads a rate near zero because there is nothing to converge to, and a
    loop that is not acting at all reads the same; the RMS is what separates them.
    The error is taken against the *current* setpoint, which after damage is the
    survivors' -- so a window spanning a removal is measured against the tissue
    that exists now, not the one that took the earlier readings.
    """
    history = list(tissue.alignment_history)
    window = CONVERGENCE_WINDOW_TIME_CONSTANTS * max(
        1, round(tissue.closed_loop_tau + tissue.dead_time)
    )
    recent = history[-window:]
    if len(recent) < MIN_CONVERGENCE_PASSES:
        return {
            "convergence_rate": None,
            "error_mean_window": None,
            "error_rms_window": None,
            "window_passes": len(recent),
        }

    setpoint = tissue.setpoint
    errors = [setpoint - reading for reading in recent]
    quarter = max(1, len(errors) // 4)
    early = sum(abs(error) for error in errors[:quarter]) / quarter
    late = sum(abs(error) for error in errors[-quarter:]) / quarter
    return {
        "convergence_rate": (1.0 - late / early) if early > 0 else None,
        "error_mean_window": sum(errors) / len(errors),
        "error_rms_window": (sum(error * error for error in errors) / len(errors)) ** 0.5,
        "window_passes": len(errors),
    }


def pid_status(app: TAMEApplication) -> PIDStatus | None:
    """The loop's status with the rolling window added. ``None`` while steering is off.

    The tissue produces its own instantaneous status and knows nothing about
    windows; the window is added here so ``PIDStatus`` is one model wherever it is
    served -- ``/homeostasis/status``, ``PUT /steering/gains``, ``/steering/update``
    and ``/metrics/pid`` all go through this.
    """
    if app.homeostat is None:
        return None
    tissue = app.homeostat.homeostat
    return PIDStatus(**app.homeostat.pid_status(), **convergence(tissue))  # pyright: ignore[reportArgumentType] # convergence returns the four rolling fields by name


# --- the coupling and the gate --------------------------------------------------------


def _mob_layers(app: TAMEApplication) -> dict[int, MixtureOfBidders]:
    return mob_layers_by_index(app.model)  # pyright: ignore[reportArgumentType] # AutoModelForCausalLM is an nn.Module at runtime


def _trace_summaries(app: TAMEApplication) -> dict[int, RoutingTraceSummary]:
    return {
        layer: mob.routing_trace.summary()
        for layer, mob in _mob_layers(app).items()
        if mob.routing_trace is not None
    }


def _layer_coupling(layer: int, mob: MixtureOfBidders) -> LayerCouplingStatus:
    coupling = mob.coupling_or_none()
    if coupling is None:
        return LayerCouplingStatus(
            layer=layer,
            attached=False,
            active=False,
            coupling_step=0,
            warmup_steps=0,
            warmup_progress=0.0,
            beta_effective=0.0,
            delta_norm_fraction_mean=None,
            delta_norm_fraction_max=None,
            steering_alignment_mean=None,
            detector_norm=None,
        )
    step = int(coupling._coupling_step.item())
    warmup = coupling.config.warmup_steps
    metrics = coupling.last_metrics
    detector_norm = float(coupling.detector.detach().float().norm())
    beta = min(1.0, step / warmup) * coupling.config.coupling_beta
    return LayerCouplingStatus(
        layer=layer,
        attached=True,
        # Attached and ramped is not the same as acting: a zero-norm receptor
        # leaves the perceived stream equal to the stream, which is where every
        # coupling starts.
        active=beta > 0.0 and detector_norm > 0.0,
        coupling_step=step,
        warmup_steps=warmup,
        warmup_progress=min(1.0, step / warmup),
        beta_effective=beta,
        delta_norm_fraction_mean=(
            float(metrics.delta_norm_fraction_mean) if metrics is not None else None
        ),
        delta_norm_fraction_max=(
            float(metrics.delta_norm_fraction_max) if metrics is not None else None
        ),
        steering_alignment_mean=(
            float(metrics.steering_alignment_mean) if metrics is not None else None
        ),
        detector_norm=detector_norm,
    )


def _coupling_mode(layers: list[LayerCouplingStatus]) -> str:
    attached = [layer for layer in layers if layer.attached]
    if not attached:
        return "off"
    if any(layer.active for layer in attached):
        return "active" if all(layer.warmup_progress >= 1.0 for layer in attached) else "warming"
    return "inert"


def _mean_correlation(
    summaries: dict[int, RoutingTraceSummary], num_experts: int
) -> list[float | None] | None:
    """Average each expert's goal-routing correlation over the layers that measured one.

    Averaged rather than pooled: the layers do not see the same distribution of
    alignments (the injection enters at each actuator's block, so a cell above
    reads the pushes below it), and pooling would weight whichever layer happened
    to have the widest spread.
    """
    measured = [
        summary.goal_correlation
        for summary in summaries.values()
        if summary.goal_correlation is not None
    ]
    if not measured:
        return None
    averaged: list[float | None] = []
    for expert in range(num_experts):
        values = [value for row in measured if (value := row[expert]) is not None]
        averaged.append(sum(values) / len(values) if values else None)
    return averaged


def coupling_status(app: TAMEApplication) -> CouplingStatus | None:
    """Is the goal shaping which experts activate? ``None`` while steering is off."""
    if app.homeostat is None:
        return None
    layers = [_layer_coupling(layer, mob) for layer, mob in sorted(_mob_layers(app).items())]
    mode = _coupling_mode(layers)
    summaries = _trace_summaries(app)
    alignments = [
        value
        for summary in summaries.values()
        if (value := summary.goal_alignment_mean) is not None
    ]
    betas = [layer.beta_effective for layer in layers if layer.attached]
    progress = [layer.warmup_progress for layer in layers if layer.attached]
    correlation = _mean_correlation(summaries, app.mob_config.num_experts)
    return CouplingStatus(
        goal=app.homeostat.goal,
        coupling_mode=mode,
        layers=layers,
        beta_effective_mean=(sum(betas) / len(betas)) if betas else None,
        warmup_progress=(sum(progress) / len(progress)) if progress else None,
        steering_routing_correlation=correlation,
        correlation_basis=(
            ""
            if correlation is None
            else (CORRELATION_BASIS_BASELINE if mode == "off" else CORRELATION_BASIS_COUPLED)
        ),
        goal_alignment_mean=(sum(alignments) / len(alignments)) if alignments else None,
        trace_tokens=min((summary.tokens for summary in summaries.values()), default=0),
    )


def routing_health(app: TAMEApplication) -> RoutingHealthMetrics | None:
    """The gate's histogram over the trace window. ``None`` when no MoB layer is traced.

    Pooled by token count, so a layer whose window is still filling does not weigh
    as much as one that is full; the per-layer rows are there because a gate can be
    degenerate at one depth and healthy at another.
    """
    summaries = _trace_summaries(app)
    if not summaries:
        return None
    rows = [
        RoutingLayerHealth(
            layer=layer,
            tokens=summary.tokens,
            top1_mean=summary.top1_mean,
            top1_median=summary.top1_median,
            top1_saturated_fraction=summary.top1_saturated_fraction,
            effective_experts=summary.effective_experts,
            win_share=summary.win_share,
        )
        for layer, summary in sorted(summaries.items())
    ]
    total = sum(row.tokens for row in rows)
    num_experts = app.mob_config.num_experts

    def pooled(value) -> float:
        if total == 0:
            return 0.0
        return sum(value(row) * row.tokens for row in rows) / total

    win_share = [
        pooled(lambda row, expert=expert: row.win_share[expert] if row.win_share else 0.0)
        for expert in range(num_experts)
    ]
    effective = pooled(lambda row: row.effective_experts)
    saturated = pooled(lambda row: row.top1_saturated_fraction)
    return RoutingHealthMetrics(
        top_k=app.mob_config.top_k,
        num_experts=num_experts,
        tokens=min(row.tokens for row in rows),
        top1_mean=pooled(lambda row: row.top1_mean),
        top1_median=pooled(lambda row: row.top1_median),
        top1_saturated_fraction=saturated,
        effective_experts=effective,
        win_share=win_share,
        degenerate=bool(
            total > 0
            and (
                effective < 1.0 + DEGENERATE_EFFECTIVE_MARGIN * (app.mob_config.top_k - 1)
                or saturated > DEGENERATE_SATURATION_FRACTION
            )
        ),
        layers=rows,
    )


# --- the vectors ---------------------------------------------------------------------


def _certified_effect(goal: str) -> CertifiedEffect | None:
    certification = certification_for(goal)
    if certification is None or certification.effect is None:
        return None
    return CertifiedEffect(
        effect=certification.effect,
        random_max=certification.random_max,
        control_effect=certification.control_effect,
        layers=list(certification.layers or ()),
        strength=certification.strength,
        model=certification.model,
    )


def _shared_cosine_layer(app: TAMEApplication) -> int | None:
    """The highest layer every extracted goal has a vector at, or ``None`` if there is none."""
    per_goal = [set(extraction.vectors) for extraction in app.extractions.values()]
    if not per_goal:
        return None
    shared = set.intersection(*per_goal)
    return max(shared) if shared else None


def steering_quality(app: TAMEApplication) -> SteeringQualityMetrics | None:
    """The served vectors' provenance, norms and geometry. ``None`` while steering is off."""
    if app.homeostat is None:
        return None
    homeostat, config = app.homeostat, app.steering_config
    goal = homeostat.goal
    extraction = app.extractions.get(goal)
    retention = homeostat.get_capability_retention()
    layers = [
        LayerSteeringQuality(
            layer=layer,
            injects=layer in config.steering_layers,
            # The direction the hooks actually inject: the extractor's vectors are
            # unit-norm by construction, so what is worth reporting is what the
            # capability projection left of one.
            norm=float(homeostat.projected_direction(layer)[0].float().norm()),
            separability=(extraction.separability.get(layer) if extraction else None),
            capability_retention=retention.get(layer),
        )
        for layer in sorted(homeostat.steering_vectors)
    ]

    cosine_layer = _shared_cosine_layer(app)
    cosine: list[list[float]] = []
    goals: list[str] = []
    if cosine_layer is not None:
        goals, matrix = goal_similarity_matrix(
            {name: record.vectors for name, record in app.extractions.items()}, cosine_layer
        )
        cosine = matrix.tolist()

    return SteeringQualityMetrics(
        goal=goal,
        certified=bool(extraction.certified) if extraction else False,
        source=extraction.source if extraction else "unknown",
        pair_format=extraction.pair_format if extraction else "unknown",
        pair_count=extraction.pair_count if extraction else 0,
        tier_counts=dict(extraction.tier_counts) if extraction else {},
        extraction_layers=list(extraction.layers) if extraction else [],
        readout_layer=homeostat.readout_layer,
        base_strength=config.base_strength,
        strength_band=[config.min_strength, config.max_strength],
        layers=layers,
        goals=goals,
        cosine_layer=cosine_layer,
        inter_goal_cosine=cosine,
        certified_effect=_certified_effect(goal),
    )


def steering_pca(app: TAMEApplication, components: int = 2) -> PCAProjection:
    """Every extracted (goal, layer) direction projected onto its own principal components.

    A diagnostic, never a gate (#3): prompt-surface features separate especially
    cleanly, so a picture in which the goals stand apart certifies nothing. On a
    server that has extracted one goal -- which is every freshly started one --
    what the picture shows is that goal's *layers* spread out, not goals separating
    from each other, and ``note`` says so.
    """
    if components not in (2, 3):
        raise ValueError(f"components must be 2 or 3, got {components}")
    labels: list[tuple[str, int]] = []
    rows: list[torch.Tensor] = []
    for goal, extraction in sorted(app.extractions.items()):
        for layer, vector in sorted(extraction.vectors.items()):
            flat = vector.vector.detach().float().reshape(-1).cpu()
            labels.append((goal, layer))
            rows.append(flat / flat.norm().clamp_min(1e-12))

    goals = sorted(app.extractions)
    if len(rows) < components + 1:
        return PCAProjection(
            components=components,
            goals=goals,
            note=(
                f"{len(rows)} direction(s) extracted in this process: fewer than the "
                f"{components + 1} a {components}-component projection needs. Nothing is "
                "projected; install another goal to populate this."
            ),
        )

    stacked = torch.stack(rows)
    centred = stacked - stacked.mean(dim=0, keepdim=True)
    _, singular, right = torch.linalg.svd(centred, full_matrices=False)
    variance = singular.square()
    coordinates = centred @ right[:components].T
    note = (
        "goal separation: the directions of "
        + ", ".join(goals)
        + ", at every layer each was extracted at"
        if len(goals) > 1
        else (
            f"one goal ({goals[0]}) is extracted in this process, so what separates here "
            "is its layers, not its goals. A diagnostic either way: separability does "
            "not certify a direction, the behavioural gate does (#3)."
        )
    )
    return PCAProjection(
        components=components,
        points=[
            PCAPoint(goal=goal, layer=layer, coordinates=coordinates[index].tolist())
            for index, (goal, layer) in enumerate(labels)
        ],
        explained_variance_ratio=(variance / variance.sum().clamp_min(1e-12))[:components].tolist(),
        goals=goals,
        note=note,
    )


# --- latency ---------------------------------------------------------------------------


def latency_stats(app: TAMEApplication) -> list[LatencyStats]:
    return [
        LatencyStats(route=route, **app.latency.stats(route))  # pyright: ignore[reportArgumentType] # stats() returns the model's fields by name
        for route in app.latency.routes
    ]


def probed_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- the composite --------------------------------------------------------------------


def health_response(app: TAMEApplication) -> HealthResponse:
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except RuntimeError:  # pragma: no cover - a driver that answers is_available but not the name
        gpu_name = "Unknown"
    return HealthResponse(
        status="alive",
        gpu=gpu_name,
        model_id=app.model_id,
        architecture="TAME (Mixture of Bidders + Cognitive Homeostasis)",
        mob_active=bool(_mob_layers(app)),
        steering_active=app.homeostat is not None,
    )


def swarm_status(app: TAMEApplication) -> SwarmStatus:
    """Wealth averaged over the MoB layers, usage summed over them, as the economy stands."""
    layers = list(_mob_layers(app).values())
    num_experts = app.mob_config.num_experts
    if not layers:
        return SwarmStatus(
            num_experts=num_experts,
            expert_wealth=[0.0] * num_experts,
            expert_usage=[0.0] * num_experts,
            layers_modified=0,
        )
    wealth = torch.zeros(num_experts)
    usage = torch.zeros(num_experts)
    for mob in layers:
        wealth += mob.expert_wealth.detach().float().cpu()
        usage += mob.expert_usage_count.detach().float().cpu()
    return SwarmStatus(
        num_experts=num_experts,
        expert_wealth=(wealth / len(layers)).tolist(),
        expert_usage=usage.tolist(),
        layers_modified=len(layers),
    )


def system_health(app: TAMEApplication) -> SystemHealthStatus:
    """Every current-state snapshot in one call, each ``None`` where its subject is absent."""
    outcome = app.outcome
    if outcome is not None and app.homeostat is not None and outcome.goal != app.homeostat.goal:
        outcome = outcome.model_copy(update={"stale": True})
    return SystemHealthStatus(
        health=health_response(app),
        swarm=swarm_status(app),
        pid=pid_status(app),
        coupling=coupling_status(app),
        routing=routing_health(app),
        steering=steering_quality(app),
        outcome=outcome,
        latency=latency_stats(app),
    )


__all__ = [
    "CONVERGENCE_WINDOW_TIME_CONSTANTS",
    "DEGENERATE_EFFECTIVE_MARGIN",
    "DEGENERATE_SATURATION_FRACTION",
    "MIN_CONVERGENCE_PASSES",
    "ROUTING_SATURATION_THRESHOLD",
    "convergence",
    "coupling_status",
    "health_response",
    "latency_stats",
    "pid_status",
    "probed_at",
    "routing_health",
    "steering_pca",
    "steering_quality",
    "swarm_status",
    "system_health",
]
