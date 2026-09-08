"""The metrics surface: seven reads and one measurement (#5).

Every ``GET`` here is a thin read of state the forward pass already produced --
see ``observability``, which holds the builders and the argument for why each
number is worth serving. The one ``POST`` is the outcome probe, which runs forward
passes and is a ``POST`` for exactly that reason; it caches, and ``GET
/metrics/outcome`` returns the cache.

Included into the main router rather than mounted separately so a client (and
``tests/test_api.py``) sees one application.
"""

import logging
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query

import observability
from app import TAMEApplication
from dependencies import get_tame_app
from models import (
    CouplingStatus,
    OutcomeMetrics,
    PCAProjection,
    PIDStatus,
    RoutingHealthMetrics,
    SteeringQualityMetrics,
    SystemHealthStatus,
)
from outcome_probe import DEFAULT_PROBE_PAIRS, MAX_PROBE_PAIRS, ProbeUnavailable, probe_outcome

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])

# The dependency in the annotation rather than the default: FastAPI's current
# idiom, and it keeps ``Depends()`` out of a mutable default argument.
TameApp = Annotated[TAMEApplication, Depends(get_tame_app)]

STEERING_OFF = "steering is not active on this process"

# 3.10 is the floor (pyrightconfig), so a TypeVar rather than PEP 695 syntax.
Snapshot = TypeVar("Snapshot")


def _require_steering(tame: TAMEApplication) -> None:
    if tame.homeostat is None:
        raise HTTPException(status_code=404, detail=STEERING_OFF)


def _built(snapshot: Snapshot | None, what: str) -> Snapshot:
    """A builder returned None after its precondition was checked: report it, do not assert it.

    ``assert`` is stripped under ``-O``, and a stripped guard here would turn a
    404 into a 500 from response-model validation.
    """
    if snapshot is None:
        raise HTTPException(status_code=500, detail=f"{what} could not be built")
    return snapshot


@router.get("/pid", response_model=PIDStatus)
def metrics_pid(tame: TameApp):
    """The tissue's consensus, every cell, and the rolling window over recent passes."""
    _require_steering(tame)
    return _built(observability.pid_status(tame), "pid status")


@router.get("/coupling", response_model=CouplingStatus)
def metrics_coupling(tame: TameApp):
    """Is the goal shaping which experts activate, right now?

    ``coupling_mode`` says whether a routing coupling mediates it. With none
    attached the correlation is still measured, against the direction the tissue's
    hooks inject, and ``correlation_basis`` labels it as the additive baseline.
    """
    _require_steering(tame)
    return _built(observability.coupling_status(tame), "coupling status")


@router.get("/routing", response_model=RoutingHealthMetrics)
def metrics_routing(tame: TameApp):
    """Is the gate degenerate? ``degenerate`` is the flag; the confound is #11's."""
    health = observability.routing_health(tame)
    if health is None:
        raise HTTPException(
            status_code=404,
            detail="no MoB layer on this process is recording a routing trace",
        )
    return health


@router.get("/steering/quality", response_model=SteeringQualityMetrics)
def metrics_steering_quality(tame: TameApp):
    """Provenance, norms and geometry of the served vectors, with the gate's own verdict."""
    _require_steering(tame)
    return _built(observability.steering_quality(tame), "steering quality")


@router.get("/steering/pca", response_model=PCAProjection)
def metrics_steering_pca(
    tame: TameApp,
    components: int = Query(default=2, ge=2, le=3),
):
    """Goal directions on their own principal components. A diagnostic, never a gate (#3)."""
    _require_steering(tame)
    return observability.steering_pca(tame, components)


@router.get("/outcome", response_model=OutcomeMetrics)
def metrics_outcome(tame: TameApp):
    """The last outcome probe. 404 until one has run: this surface does not invent a number."""
    if tame.outcome is None:
        raise HTTPException(
            status_code=404,
            detail="no outcome probe has run on this process; POST /metrics/outcome/probe",
        )
    outcome = tame.outcome
    if tame.homeostat is not None and outcome.goal != tame.homeostat.goal:
        return outcome.model_copy(update={"stale": True})
    return outcome


@router.post("/outcome/probe", response_model=OutcomeMetrics)
def metrics_outcome_probe(
    tame: TameApp,
    num_pairs: int = Query(default=DEFAULT_PROBE_PAIRS, ge=1, le=MAX_PROBE_PAIRS),
):
    """Measure whether steering changes held-out behaviour. Runs forward passes.

    Synchronous on purpose, like ``/steering/update``: a ``def`` endpoint runs in
    the threadpool, where blocking on the GPU stalls one worker rather than the
    event loop and every other request with it.
    """
    # The probe detaches the hooks and flips the loop's mode, and this endpoint is a
    # sync ``def`` running in a 40-worker threadpool: two overlapping probes would
    # double-register the steering hooks (attach is append-only) and could leave the
    # economy frozen for the life of the process. 409 rather than queueing, so a
    # double-click does not stack minutes of GPU work behind the first.
    if not tame.state_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="another probe or goal install is in flight on this process",
        )
    try:
        return probe_outcome(tame, num_pairs=num_pairs)
    except ProbeUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Outcome probe failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="outcome probe failed") from exc
    finally:
        tame.state_lock.release()


@router.get("/health", response_model=SystemHealthStatus)
def metrics_health(tame: TameApp):
    """Every current-state snapshot in one call; each component ``None`` when absent.

    Current state only, by design: #5's 2026-04-12 comment gave history to MLflow,
    so nothing here aggregates over time except the bounded latency window and the
    trace's own.
    """
    return observability.system_health(tame)
