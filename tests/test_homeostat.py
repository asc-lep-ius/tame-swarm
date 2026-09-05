"""The goal tissue: coupled per-layer cells, a shared pattern, one rule with no fallback.

Every steered layer is a cell that senses its own reading and runs its own
integrator; the integrators are diffusively coupled, so the tissue-level loop is
the consensus they converge on rather than a privileged sensor. Where a property
can be stated against the measured plant it is asserted with a simulated tissue;
where it concerns the wiring it runs on the identity-layer fakes, whose response
to an injection is exactly the additive passthrough, so every lift the
calibration should recover is known in closed form.
"""

import pytest
import torch
import torch.nn as nn

from homeostat import (
    AdaptiveHomeostat,
    AlignmentCalibration,
    CognitiveHomeostat,
    LayerCalibration,
    SteeringHook,
    calibrate_alignment,
    critical_integral_gain,
)
from steering import SteeringConfig, SteeringVector

from .steering_fakes import MonotonicModel, SimpleCharTokenizer

HIDDEN = 8
ACTUATORS = (1, 2)
CALIBRATION_TEXTS = ["abc", "defg", "hij", "klmn"]
READOUT = 3
# Identity layers with one shared direction: the lift a cell reads per unit of
# strength is the number of actuators below it.
LIFTS = {1: 0.0, 2: 1.0, 3: 2.0}


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


def calibration(base: float = 2.0, sigma: float = 1.0) -> AlignmentCalibration:
    return AlignmentCalibration(
        layers={
            layer: LayerCalibration(
                resting_mean=0.0, resting_sigma=sigma, token_sigma=sigma, lift=LIFTS[layer]
            )
            for layer in (*ACTUATORS, READOUT)
        },
        actuators=ACTUATORS,
        sensors=(*ACTUATORS, READOUT),
        reference_strength=base,
        num_passages=4,
    )


class Tissue:
    """The measured plant, cell by cell: each reading is content plus the effort below it."""

    def __init__(self, homeostat: AdaptiveHomeostat, direction: torch.Tensor, content=0.0):
        self.homeostat, self.direction = homeostat, direction
        if isinstance(content, dict):
            self.content = dict(content)
        else:
            self.content = dict.fromkeys((*ACTUATORS, READOUT), float(content))
        self.strengths: dict[int, float] = dict.fromkeys(ACTUATORS, 0.0)
        self.dead: set[int] = set()

    def hidden(self, layer: int) -> torch.Tensor:
        below = sum(self.strengths[lower] for lower in ACTUATORS if lower < layer)
        return ((self.content[layer] + below) * self.direction).view(1, 1, -1)

    def step(self) -> None:
        for layer in (*ACTUATORS, READOUT):
            if layer in self.dead:
                # A removed hook neither senses nor injects.
                if layer in ACTUATORS:
                    self.strengths[layer] = 0.0
                continue
            strength = self.homeostat.sense(layer, self.hidden(layer), self.direction)
            if layer in ACTUATORS:
                self.strengths[layer] = strength

    def run(self, passes: int) -> None:
        for _ in range(passes):
            self.step()


def tissue_config(**overrides) -> SteeringConfig:
    defaults = dict(
        steering_layers=list(ACTUATORS),
        readout_layer=READOUT,
        base_strength=2.0,
        min_strength=0.0,
        max_strength=4.0,
        adaptive=True,
        measurement_filter_alpha=1.0,
        orthogonal_projection=False,
    )
    defaults.update(overrides)
    return SteeringConfig(**defaults)


def make(content=0.0, **overrides) -> tuple[AdaptiveHomeostat, Tissue]:
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(tissue_config(**overrides), calibration=calibration())
    return homeostat, Tissue(homeostat, direction, content)


def test_uncalibrated_loop_keeps_the_legacy_cosine_contract():
    """Without a calibration the loop regulates cos(h, v) toward target_alignment, as before."""
    config = SteeringConfig(adaptive=True, target_alignment=0.99, base_strength=0.3, max_strength=5)
    homeostat = AdaptiveHomeostat(config)
    direction = unit(torch.randn(HIDDEN))
    orthogonal = unit(torch.randn(HIDDEN))
    orthogonal = unit(orthogonal - (orthogonal @ direction) * direction)

    strength = homeostat.compute_strength(orthogonal.view(1, 1, -1), direction)

    assert strength > config.base_strength
    assert homeostat.setpoint == pytest.approx(0.99)
    assert not homeostat.calibrated


def test_per_cell_setpoints_are_the_lift_each_cell_reads_at_the_reference_strength():
    homeostat, _ = make()
    assert homeostat.cell_setpoint(1) == 0.0
    assert homeostat.cell_setpoint(2) == pytest.approx(2.0)
    assert homeostat.cell_setpoint(3) == pytest.approx(4.0)
    # The tissue's gain is the mean cell gain, which is what the shared gains derive from.
    assert homeostat.calibration is not None
    assert homeostat.calibration.gain_z == pytest.approx(1.0)


def test_tissue_settles_at_the_reference_strength_on_resting_content():
    homeostat, tissue = make()
    tissue.run(80)

    for layer in ACTUATORS:
        assert tissue.strengths[layer] == pytest.approx(2.0, abs=1e-2)
    assert abs(homeostat.status()["error"]) < 1e-2


def test_tissue_compensates_a_content_deficit_in_consensus():
    """Every cell reads the same deficit; the tissue's mean error goes to zero, not each cell's.

    The bottom cell can never see its own effect, so its error stays; the cells above
    it absorb the correction. With a shared integrator both actuators carry the same
    effort, and ``2a + b = 1.5`` with ``a == b`` puts each half a unit above the reference.
    """
    homeostat, tissue = make(content=-0.5)
    tissue.run(120)

    assert abs(homeostat.status()["error"]) < 2e-2
    for layer in ACTUATORS:
        assert tissue.strengths[layer] == pytest.approx(2.5, abs=5e-2)


def test_tissue_saturates_at_the_band_and_reports_it():
    homeostat, tissue = make(content=-5.0, max_strength=2.5)
    tissue.run(60)

    assert all(tissue.strengths[layer] == pytest.approx(2.5) for layer in ACTUATORS)
    assert homeostat.status()["integral_saturated"] is True


def test_integrators_are_shared_across_cells():
    """The slow state is isopotential: every cell carries the same memory."""
    homeostat, tissue = make(content={1: -1.0, 2: 0.5, 3: -0.3})
    tissue.run(30)
    integrals = [cell["i_term"] for cell in homeostat.status()["cells"]]
    assert integrals[0] == pytest.approx(integrals[1], abs=1e-9)
    assert integrals[1] == pytest.approx(integrals[2], abs=1e-9)


def test_local_proportional_term_pushes_where_the_deficit_is_without_winding_up():
    """A blind cell's deficit shows in its own effort through P only; the integral stays shared."""
    homeostat, tissue = make(content={1: -1.0, 2: 0.0, 3: 0.0}, kp=0.5)
    tissue.run(120)

    assert tissue.strengths[1] > tissue.strengths[2]
    assert tissue.strengths[1] < homeostat.config.max_strength
    cells = homeostat.status()["cells"]
    assert cells[0]["i_term"] == pytest.approx(cells[1]["i_term"], abs=1e-9)
    assert abs(homeostat.status()["error"]) < 2e-2


def test_a_removed_cell_drops_out_of_the_consensus_and_rejoins():
    """Undesigned damage: a hook that stops firing must not freeze the tissue on its stale state."""
    homeostat, tissue = make(content=-0.5, max_strength=8.0)
    tissue.run(40)
    steady = dict(tissue.strengths)

    tissue.dead.add(2)
    tissue.run(3)
    alive = {cell["layer"]: cell["alive"] for cell in homeostat.status()["cells"]}
    assert alive == {1: True, 2: False, 3: True}
    # The survivors keep regulating: the tissue error is computed over live cells only,
    # the dead cell's stale error no longer enters the consensus, and the remaining
    # actuator takes up the effort the removed one used to supply.
    tissue.run(80)
    assert abs(homeostat.status()["error"]) < 5e-2
    assert tissue.strengths[1] > steady[1] + 1.0
    assert homeostat.current_strength == pytest.approx(tissue.strengths[1])

    tissue.dead.clear()
    tissue.run(3)
    assert all(cell["alive"] for cell in homeostat.status()["cells"])


def test_gains_are_derived_from_the_calibration_unless_pinned():
    derived = AdaptiveHomeostat(tissue_config(), calibration=calibration())
    kp, ki = derived.gains()
    assert ki > 0
    # A filter with alpha 1 has no time constant, so SIMC prescribes integral-only control.
    assert kp == 0.0

    pinned = AdaptiveHomeostat(tissue_config(kp=0.3, ki=0.05), calibration=calibration())
    assert pinned.gains() == (0.3, 0.05)

    filtered = AdaptiveHomeostat(
        tissue_config(measurement_filter_alpha=0.1), calibration=calibration()
    )
    assert filtered.gains()[0] > 0


def simulate_integral_loop(gain: float, ki: float, delay: int, steps: int = 600) -> float:
    """Integral-only control of ``y_t = gain * u_(t - delay)``; the final |error|."""
    outputs = [0.0] * delay
    integral, error = 0.0, 1.0
    for _ in range(steps):
        reading = gain * outputs[-delay]
        error = 1.0 - reading
        integral += error
        outputs.append(ki * integral)
    return abs(error)


def test_critical_integral_gain_separates_decay_from_oscillation():
    critical = critical_integral_gain(1.0, delay=2)
    assert critical == pytest.approx(1.0)
    assert critical_integral_gain(1.0, delay=3) == pytest.approx(0.618, abs=1e-3)
    assert simulate_integral_loop(1.0, 0.9 * critical, delay=2) < 1e-3
    assert simulate_integral_loop(1.0, 1.1 * critical, delay=2) > 0.1


def test_set_gains_rejects_an_integral_gain_the_plant_cannot_stabilise():
    homeostat = AdaptiveHomeostat(tissue_config(), calibration=calibration())
    limit = homeostat.max_stable_ki()
    assert homeostat.calibration is not None
    critical = critical_integral_gain(homeostat.calibration.gain_z, 2)
    assert limit is not None and limit < critical
    # The accepted bound still converges on the modelled delay loop.
    assert simulate_integral_loop(homeostat.calibration.gain_z, limit, delay=2) < 1e-3
    with pytest.raises(ValueError, match="ki"):
        homeostat.set_gains(ki=limit * 1.01)
    homeostat.set_gains(ki=limit * 0.5, kp=0.1)
    assert homeostat.gains() == (0.1, pytest.approx(limit * 0.5))


def test_gain_derivation_reads_the_closed_loop_and_filter_time_constants():
    slow = AdaptiveHomeostat(tissue_config(closed_loop_tau=20.0), calibration=calibration())
    fast = AdaptiveHomeostat(tissue_config(closed_loop_tau=5.0), calibration=calibration())
    assert slow.gains()[1] < fast.gains()[1]
    filtered = AdaptiveHomeostat(
        tissue_config(measurement_filter_alpha=0.2), calibration=calibration()
    )
    assert filtered.filter_time_constant == pytest.approx(4.0)
    assert filtered.gains()[0] > 0


def test_reading_filters_over_passes_and_takes_the_last_position():
    homeostat, _ = make(measurement_filter_alpha=0.5)
    direction = unit(torch.randn(HIDDEN))
    hidden = torch.stack([torch.zeros(HIDDEN), torch.zeros(HIDDEN), 5.0 * direction]).view(1, 3, -1)
    assert homeostat.reading(hidden, direction) == pytest.approx(5.0)

    homeostat.sense(3, (2.0 * direction).view(1, 1, -1), direction)
    homeostat.sense(3, torch.zeros(1, 1, HIDDEN), direction)
    # Second reading is 0 but the filtered process variable is halfway back to 2.
    assert homeostat.status()["cells"][2]["process_variable"] == pytest.approx(1.0)


def test_local_push_past_the_band_is_reported_as_saturation():
    homeostat, tissue = make(content={1: -10.0, 2: 0.0, 3: 0.0}, kp=0.5)
    tissue.run(20)
    cells = homeostat.status()["cells"]
    assert cells[0]["saturated"] is True
    assert cells[0]["i_term"] == pytest.approx(cells[1]["i_term"], abs=1e-9)


def test_band_less_goal_is_never_reported_saturated():
    homeostat, tissue = make(content=-5.0, min_strength=2.0, max_strength=2.0)
    tissue.run(10)
    assert homeostat.status()["integral_saturated"] is False
    with pytest.raises(ValueError, match="constant strength"):
        homeostat.set_gains(kd=0.1)


def test_reset_clears_cells_and_histories():
    homeostat, tissue = make(content=-0.5)
    tissue.run(3)
    assert len(homeostat.alignment_history) == 3
    assert len(homeostat.strength_history) == 3

    homeostat.reset()
    assert len(homeostat.alignment_history) == 0
    assert homeostat.controller.states == {}
    assert homeostat.current_strength == 2.0


def test_snapshot_round_trips_through_a_dict():
    homeostat, tissue = make(content=-0.5)
    tissue.run(5)

    restored = AdaptiveHomeostat(tissue_config(), calibration=calibration())
    restored.restore(homeostat.snapshot())
    assert restored.controller.states == homeostat.controller.states
    assert restored.current_strength == pytest.approx(homeostat.current_strength)
    assert restored.status()["process_variable"] == pytest.approx(
        homeostat.status()["process_variable"]
    )
    # The filter state came across too: one more identical pass agrees exactly.
    twin = Tissue(restored, tissue.direction, -0.5)
    twin.strengths = dict(tissue.strengths)
    tissue.step()
    twin.step()
    assert twin.strengths == pytest.approx(tissue.strengths)


# --- calibration and wiring on the identity-layer fakes


def identity_vectors(direction: torch.Tensor, layers) -> dict[int, SteeringVector]:
    return {layer: SteeringVector("goal", direction.clone(), layer) for layer in layers}


def test_calibration_recovers_the_lift_every_cell_reads():
    """Identity layers pass an injection straight up: a cell's lift is the actuators below it."""
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    tokenizer = SimpleCharTokenizer()
    direction = unit(torch.eye(HIDDEN)[1])
    vectors = identity_vectors(direction, layers=[1, 2, 3])
    config = SteeringConfig(steering_layers=[1, 2], base_strength=2.0, readout_layer=3)

    result = calibrate_alignment(model, tokenizer, vectors, config, texts=CALIBRATION_TEXTS)

    assert result.actuators == (1, 2)
    assert result.sensors == (1, 2, 3)
    for layer, lift in LIFTS.items():
        assert result.layers[layer].lift == pytest.approx(lift, abs=1e-4)
        assert result.layers[layer].resting_mean == pytest.approx(0.0, abs=1e-5)
    assert result.reference_strength == 2.0
    assert set(result.directions) == {1, 2, 3}
    with pytest.raises(ValueError, match="passages"):
        calibrate_alignment(model, tokenizer, vectors, config, texts=["abc", "de"])


def test_calibration_is_measured_along_the_direction_the_hooks_inject():
    """With the capability projection on, the hook injects the projected direction; so must
    the calibration read it, or the resting state is measured on a different variable."""
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    tokenizer = SimpleCharTokenizer()
    raw = unit(torch.eye(HIDDEN)[1] + 0.5 * torch.eye(HIDDEN)[0])
    config = tissue_config(orthogonal_projection=True)
    coordinator = CognitiveHomeostat(config)
    coordinator.add_steering_vectors(identity_vectors(raw, layers=[1, 2, 3]))
    coordinator.set_capability_subspaces({layer: torch.eye(HIDDEN)[:1] for layer in (1, 2, 3)})
    coordinator.calibrate(model, tokenizer, texts=CALIBRATION_TEXTS)
    coordinator.attach_to_model(model)

    assert coordinator.calibration is not None
    for layer, hook in coordinator.hooks.items():
        injected = hook._direction(torch.device("cpu"), torch.float32)
        assert torch.allclose(coordinator.calibration.directions[layer], injected, atol=1e-6)
        assert not torch.allclose(injected, raw, atol=1e-3)
    with pytest.raises(RuntimeError, match="detach"):
        coordinator.calibrate(model, tokenizer, texts=CALIBRATION_TEXTS)
    coordinator.detach_from_model()


def test_cognitive_homeostat_attaches_a_cell_per_layer_and_one_sensor_only_readout():
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    tokenizer = SimpleCharTokenizer()
    direction = unit(torch.eye(HIDDEN)[1])
    config = tissue_config()
    coordinator = CognitiveHomeostat(config)
    coordinator.add_steering_vectors(identity_vectors(direction, layers=[1, 2, 3]))
    coordinator.calibrate(model, tokenizer, texts=CALIBRATION_TEXTS)
    coordinator.attach_to_model(model)

    assert {layer for layer, hook in coordinator.hooks.items() if hook.injects} == {1, 2}
    assert not coordinator.hooks[3].injects

    ids = tokenizer("abcd", return_tensors="pt")["input_ids"]
    for _ in range(3):
        model(ids)
    stats = coordinator.get_alignment_stats()
    assert stats["pid"]["step_count"] == 3
    assert [cell["layer"] for cell in stats["pid"]["cells"]] == [1, 2, 3]
    assert config.min_strength <= stats["current_strength"] <= config.max_strength
    coordinator.detach_from_model()


def test_top_actuator_is_the_top_cell_when_no_readout_vector_exists():
    config = SteeringConfig(steering_layers=[1, 2], adaptive=True, orthogonal_projection=False)
    coordinator = CognitiveHomeostat(config)
    coordinator.add_steering_vectors(identity_vectors(unit(torch.randn(HIDDEN)), layers=[1, 2]))
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    coordinator.attach_to_model(model)

    assert set(coordinator.hooks) == {1, 2}
    assert all(hook.injects for hook in coordinator.hooks.values())
    coordinator.detach_from_model()


def test_standalone_hook_with_adaptive_off_injects_base_strength():
    """The probe path (behavioural validation) must keep a constant, unmeasured injection."""
    direction = unit(torch.eye(HIDDEN)[0])
    hook = SteeringHook(
        SteeringVector("goal", direction, layer=0),
        SteeringConfig(adaptive=False, base_strength=1.5, orthogonal_projection=False),
    )
    hidden = torch.zeros(1, 2, HIDDEN)
    injected = hook(nn.Identity(), (hidden,), hidden)
    assert torch.allclose(injected[0, -1], 1.5 * direction)
