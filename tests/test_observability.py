"""The metrics surface on the wired system (#5).

``tests/wired_system.py`` is the smallest thing that has all the parts a metric
here describes: MoB blocks with a live gate, a calibrated tissue with cells that
disagree, and the routing coupling seeded from the same direction the hooks
inject. Every builder is exercised against it and against the state that empties
it -- no steering, no trace, nothing probed -- because a metrics surface whose
failure mode is a confident zero is worse than none.

The 10 ms budget is asserted on the builders rather than on the HTTP round trip:
the criterion is that a route does no heavy computation, and ``TestClient`` adds
its own ASGI overhead to a number that is meant to be about this code.
"""

import time

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import observability
from app import TAMEApplication
from contrastive_data import CERTIFIED
from models import OutcomeArm, OutcomeMetrics

from .wired_system import (
    ACTUATORS,
    BELOW_ACTUATORS,
    MOB_LAYERS,
    WiredSystem,
    build_wired_app,
)

SETTLE_PASSES = 60
CONTENT_DEFICIT = -1.0
# The acceptance criterion, per builder. Timed as the best of a few repetitions:
# a metrics route competing with a garbage collection is not what the budget is
# about, and the claim is that the work is a read rather than a computation.
BUDGET_SECONDS = 0.010
BUDGET_REPETITIONS = 3


@pytest.fixture
def loaded() -> tuple[TAMEApplication, WiredSystem]:
    tame, system = build_wired_app()
    system.run(SETTLE_PASSES)
    return tame, system


@pytest.fixture
def client(loaded) -> TestClient:
    from routes import router

    tame, _ = loaded
    app = FastAPI()
    app.include_router(router)
    app.state.tame = tame
    return TestClient(app)


@pytest.fixture
def unsteered_client() -> TestClient:
    """A process whose steering failed to build: the degraded mode ``from_profile`` allows."""
    from routes import router

    tame, _ = build_wired_app()
    tame.homeostat = None
    app = FastAPI()
    app.include_router(router)
    app.state.tame = tame
    return TestClient(app)


# --- the tissue ---------------------------------------------------------------------


def test_the_pid_snapshot_carries_the_cells_the_dispersion_and_the_window(loaded):
    tame, _ = loaded
    status = observability.pid_status(tame)

    assert status is not None
    assert len(status.cells) == len(ACTUATORS) + 1
    assert all(cell.resting_sigma is not None and cell.gain is not None for cell in status.cells)
    assert status.window_passes >= observability.MIN_CONVERGENCE_PASSES
    assert status.sensed_dispersion >= 0.0
    assert status.error_rms_window is not None and status.convergence_rate is not None


def test_convergence_is_positive_while_the_loop_is_recovering_and_flat_once_it_has():
    """ "Zero" is ambiguous on its own, which is why the RMS is reported beside it."""
    tame, system = build_wired_app()
    system.run(SETTLE_PASSES)
    settled = observability.convergence(system.tissue())
    assert settled["convergence_rate"] is not None
    assert abs(settled["convergence_rate"]) < 0.5  # pyright: ignore[reportOperatorIssue]
    assert settled["error_rms_window"] < 0.05 * system.setpoint()  # pyright: ignore[reportOperatorIssue]

    system.set_content("truthful", CONTENT_DEFICIT)
    system.run(12)
    recovering = observability.convergence(system.tissue())
    assert recovering["error_rms_window"] > settled["error_rms_window"]  # pyright: ignore[reportOperatorIssue]

    system.run(60)
    recovered = observability.convergence(system.tissue())
    assert recovered["convergence_rate"] is not None and recovered["convergence_rate"] > 0.5  # pyright: ignore[reportOperatorIssue]


def test_an_inert_loop_under_a_sustained_push_does_not_read_as_converging():
    """The pairing: cells that sense but cannot act leave the error where it is."""
    tame, system = build_wired_app(kp=0.0, ki=0.0)
    system.run(SETTLE_PASSES)
    system.set_content("truthful", CONTENT_DEFICIT)
    system.run(72)

    inert = observability.convergence(system.tissue())
    assert inert["convergence_rate"] is not None and inert["convergence_rate"] < 0.5  # pyright: ignore[reportOperatorIssue]
    assert inert["error_rms_window"] > 0.05 * system.setpoint()  # pyright: ignore[reportOperatorIssue]


def test_the_dispersion_rises_when_content_reaches_the_cells_unevenly():
    """The #23 observable: the consensus is a compromise, and this is what says so."""
    tame, system = build_wired_app()
    system.run(SETTLE_PASSES)
    settled = system.tissue().dispersion

    system.set_content("truthful", CONTENT_DEFICIT, layer=BELOW_ACTUATORS)
    system.run(20)

    status = observability.pid_status(tame)
    assert status is not None
    assert status.dispersion > settled
    assert status.dispersion > 0.0


# --- the coupling and the gate ---------------------------------------------------------


def test_the_coupling_status_reports_every_mob_layer_and_the_ramp(loaded):
    tame, _ = loaded
    status = observability.coupling_status(tame)

    assert status is not None
    assert [layer.layer for layer in status.layers] == list(MOB_LAYERS)
    assert all(layer.attached and layer.active for layer in status.layers)
    assert status.coupling_mode == "active"
    assert status.beta_effective_mean is not None and status.beta_effective_mean > 0
    assert all(layer.detector_norm and layer.detector_norm > 0 for layer in status.layers)


def test_an_uncoupled_process_still_measures_the_goals_effect_on_routing():
    """The answer to Q3: no coupling is not no measurement, it is the additive baseline."""
    tame, system = build_wired_app(coupled=False)
    system.run(400)
    status = observability.coupling_status(tame)

    assert status is not None
    assert status.coupling_mode == "off"
    assert all(not layer.attached for layer in status.layers)
    assert status.goal_alignment_mean is not None
    assert status.steering_routing_correlation is not None
    assert len(status.steering_routing_correlation) == tame.mob_config.num_experts
    assert "additive baseline" in status.correlation_basis


def test_a_coupled_process_labels_its_correlation_as_the_coupling_s(loaded):
    tame, system = loaded
    system.run(400)
    status = observability.coupling_status(tame)

    assert status is not None
    assert status.coupling_mode == "active"
    assert status.steering_routing_correlation is not None
    assert "perception-modulation" in status.correlation_basis


def test_an_attached_but_inert_coupling_is_labelled_as_the_baseline_it_is(loaded):
    """A zero-norm receptor adds a zero delta, so the stream is the stream.

    The one misreading `correlation_basis` exists to prevent is calling that number
    the perception-modulation effect, and "attached" is not the test for it.
    """
    tame, system = loaded
    for mob in system.mobs:
        coupling = mob.coupling_or_none()
        assert coupling is not None
        with torch.no_grad():
            coupling.detector.zero_()
    system.run(400)

    status = observability.coupling_status(tame)
    assert status is not None
    assert status.coupling_mode == "inert"
    assert all(layer.attached and not layer.active for layer in status.layers)
    assert "additive baseline" in status.correlation_basis
    assert "perception-modulation" not in status.correlation_basis


def test_a_coupling_still_ramping_is_labelled_a_mixture_of_the_two(loaded):
    """Half a ramp is neither the baseline nor the effect, and saying so is the point."""
    tame, system = loaded
    for mob in system.mobs:
        mob.set_coupling_step(1)
    system.run(400)

    status = observability.coupling_status(tame)
    assert status is not None
    assert status.coupling_mode == "warming"
    assert "mixture" in status.correlation_basis


def test_the_blind_cells_disagreement_reaches_the_pairing_the_consensus_discounts(loaded):
    """`dispersion` hears only cells the tissue can move; `sensed_dispersion` hears all of them.

    Content below the bottom actuator lands on a cell nothing can correct. #21 took
    that cell out of the shared memory on purpose -- it must not steer the others --
    but a reader still has to be able to see it, which is what the pairing is for.
    """
    tame, system = build_wired_app()
    system.run(SETTLE_PASSES)
    system.set_content("truthful", -8.0, layer=BELOW_ACTUATORS)
    system.run(30)

    tissue = system.tissue()
    blind = [cell for cell in tissue.cells if tissue.cell_weight(cell) == 0.0]
    assert blind, "the fixture must have a cell no action can move"

    status = observability.pid_status(tame)
    assert status is not None
    assert status.sensed_dispersion > status.dispersion
    stuck = next(cell for cell in status.cells if cell.layer == blind[0])
    assert stuck.weight == 0.0 and abs(stuck.error) > status.dispersion


def test_routing_health_pools_the_layers_and_flags_a_healthy_gate_as_not_degenerate(loaded):
    tame, _ = loaded
    health = observability.routing_health(tame)

    assert health is not None
    assert health.tokens > 0 and len(health.layers) == len(MOB_LAYERS)
    assert health.top_k == tame.mob_config.top_k
    assert 1.0 <= health.effective_experts <= health.top_k
    # The value, not the formula restated: re-deriving the threshold here would
    # agree with a wrong implementation as readily as a right one.
    assert health.degenerate is False
    assert health.effective_experts > 1.0 + observability.DEGENERATE_EFFECTIVE_MARGIN
    assert health.top1_saturated_fraction < observability.DEGENERATE_SATURATION_FRACTION


def test_a_collapsed_gate_is_flagged_degenerate():
    """The pairing for the flag: a gate that always routes to one expert reads as such."""
    tame, system = build_wired_app()
    for mob in system.mobs:
        trace = mob.routing_trace
        assert trace is not None
        trace.clear()
        # One winner at full weight, on every token: exp(entropy) is 1.0.
        weights = torch.zeros(1, 200, tame.mob_config.top_k)
        weights[..., 0] = 1.0
        experts = torch.zeros(1, 200, tame.mob_config.top_k, dtype=torch.long)
        experts[..., 1] = 1
        trace.record(weights, experts, torch.randn(1, 200, mob.config.hidden_dim))

    health = observability.routing_health(tame)
    assert health is not None and health.degenerate
    assert health.effective_experts == pytest.approx(1.0, abs=0.01)
    assert health.top1_saturated_fraction == pytest.approx(1.0)


def test_a_process_with_no_trace_reports_no_routing_health():
    tame, system = build_wired_app()
    for mob in system.mobs:
        mob.disable_routing_trace()

    assert observability.routing_health(tame) is None


# --- the vectors ---------------------------------------------------------------------


def test_steering_quality_carries_the_provenance_the_norms_and_the_gate_s_verdict(loaded):
    tame, _ = loaded
    quality = observability.steering_quality(tame)

    assert quality is not None
    assert quality.goal == "truthful" and quality.pair_count == 8
    assert [layer.layer for layer in quality.layers] == sorted(
        tame.homeostat.steering_vectors  # pyright: ignore[reportOptionalMemberAccess]
    )
    assert all(layer.norm == pytest.approx(1.0, abs=1e-5) for layer in quality.layers)
    assert quality.inter_goal_cosine == [[pytest.approx(1.0, abs=1e-5)]]
    assert quality.goals == ["truthful"]
    # The fixture's direction is random, so the extraction is uncertified -- but
    # the *goal* is one the gate certified, and what is quoted is that record's
    # own measurement, labelled as a record and never recomputed here.
    assert quality.certified is False
    assert quality.certified_effect is not None
    assert quality.certified_effect.effect == CERTIFIED["truthful"].effect
    assert quality.certified_effect.layers == list(CERTIFIED["truthful"].layers or ())


def test_a_goal_the_gate_never_passed_quotes_no_effect():
    """The pairing: an uncertified goal has no measurement to quote, and says so."""
    tame, system = build_wired_app()
    tame.homeostat.homeostat.goal = "not-a-goal"  # pyright: ignore[reportOptionalMemberAccess]

    quality = observability.steering_quality(tame)
    assert quality is not None and quality.certified_effect is None


def test_the_inter_goal_cosine_is_over_the_goals_this_process_extracted():
    tame, system = build_wired_app(goals=("truthful", "safe"))
    quality = observability.steering_quality(tame)

    assert quality is not None
    assert quality.goals == ["safe", "truthful"]
    assert len(quality.inter_goal_cosine) == 2
    # The fixture's directions are built mutually orthogonal, so the off-diagonal
    # is zero and a non-zero one would mean the goals share a direction.
    assert quality.inter_goal_cosine[0][1] == pytest.approx(0.0, abs=1e-5)


def test_the_pca_projects_what_is_there_and_says_what_it_is(loaded):
    tame, _ = loaded
    projection = observability.steering_pca(tame, components=2)

    assert projection.components == 2
    assert len(projection.points) == len(tame.extractions["truthful"].vectors)
    assert len(projection.explained_variance_ratio) == 2
    assert sum(projection.explained_variance_ratio) <= 1.0 + 1e-5
    assert "its layers, not its goals" in projection.note


def test_the_pca_refuses_to_invent_components_it_does_not_have():
    tame, _ = build_wired_app()
    tame.extractions = {}

    projection = observability.steering_pca(tame, components=3)
    assert projection.points == [] and "fewer than the 4" in projection.note

    with pytest.raises(ValueError, match="components must be 2 or 3"):
        observability.steering_pca(tame, components=5)


# --- the composite and the routes -------------------------------------------------------


def test_system_health_composes_every_snapshot(loaded):
    tame, _ = loaded
    health = observability.system_health(tame)

    assert health.health.steering_active and health.health.mob_active
    assert health.swarm is not None and health.swarm.layers_modified == len(MOB_LAYERS)
    assert health.pid is not None and health.coupling is not None
    assert health.routing is not None and health.steering is not None
    assert health.outcome is None, "nothing has been probed on this process"


def test_every_metrics_route_returns_a_valid_model(client):
    for path in (
        "/metrics/pid",
        "/metrics/coupling",
        "/metrics/routing",
        "/metrics/steering/quality",
        "/metrics/steering/pca",
        "/metrics/health",
    ):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        assert response.json(), path

    assert client.get("/metrics/steering/pca", params={"components": 3}).status_code == 200
    assert client.get("/metrics/steering/pca", params={"components": 9}).status_code == 422


def test_the_steering_routes_are_absent_rather_than_empty_without_steering(unsteered_client):
    """404, not a zeroed model: "absent" and "at rest" must stay distinguishable."""
    for path in ("/metrics/pid", "/metrics/coupling", "/metrics/steering/quality"):
        assert unsteered_client.get(path).status_code == 404, path

    health = unsteered_client.get("/metrics/health")
    assert health.status_code == 200
    body = health.json()
    assert body["pid"] is None and body["coupling"] is None and body["steering"] is None
    assert body["routing"] is not None, "the gate is still there when the tissue is not"


def test_outcome_is_absent_until_probed_and_stale_after_a_goal_change(client, loaded):
    tame, _ = loaded
    assert client.get("/metrics/outcome").status_code == 404

    tame.outcome = OutcomeMetrics(
        goal="truthful",
        probed_at=observability.probed_at(),
        num_pairs=20,
        arms={"served": OutcomeArm(mean_log_odds=0.4, accuracy=0.7)},
        served_minus_unsteered_log_odds=0.13,
        served_minus_unsteered_accuracy=0.05,
        certified_random_max=0.087,
        beats_random=True,
    )
    fresh = client.get("/metrics/outcome").json()
    assert fresh["stale"] is False and fresh["beats_random"] is True

    tame.homeostat.homeostat.goal = "safe"  # pyright: ignore[reportOptionalMemberAccess]
    assert client.get("/metrics/outcome").json()["stale"] is True
    assert client.get("/metrics/health").json()["outcome"]["stale"] is True


def test_a_delta_at_the_random_floor_is_the_value_that_reads_did_not_help():
    """#5's falsifiability criterion, as a property of the model rather than a hope."""
    at_floor = OutcomeMetrics(
        goal="truthful",
        probed_at=observability.probed_at(),
        num_pairs=20,
        arms={"served": OutcomeArm(mean_log_odds=0.1, accuracy=0.5)},
        served_minus_unsteered_log_odds=0.05,
        served_minus_unsteered_accuracy=0.0,
        certified_random_max=0.087,
        beats_random=0.05 > 0.087,
    )
    assert at_floor.beats_random is False


def test_the_generation_routes_are_recorded_in_the_latency_window(loaded):
    tame, _ = loaded
    with tame.latency.measure("/generate") as tokens:
        tokens[0] = 64

    stats = observability.latency_stats(tame)
    assert [row.route for row in stats] == ["/generate"]
    assert stats[0].requests == 1 and stats[0].tokens_per_second > 0


# --- the budget ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        observability.pid_status,
        observability.coupling_status,
        observability.routing_health,
        observability.steering_quality,
        observability.steering_pca,
        observability.system_health,
    ],
    ids=lambda builder: builder.__name__,
)
def test_every_builder_reads_cached_state_inside_the_ten_millisecond_budget(loaded, builder):
    tame, _ = loaded
    best = min(_timed(builder, tame) for _ in range(BUDGET_REPETITIONS))
    assert best < BUDGET_SECONDS, f"{builder.__name__} took {best * 1000:.2f} ms"


def _timed(builder, tame) -> float:
    started = time.perf_counter()
    builder(tame)
    return time.perf_counter() - started
