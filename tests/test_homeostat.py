"""The goal loop: calibrated setpoint, one strength per goal, local fallback after damage.

Where a property can be stated against the measured plant it is asserted with a
simulated one; where it concerns the wiring it runs on the identity-layer fakes,
whose response to an injection is exactly the additive passthrough, so the gain
the calibration should recover is known in closed form.
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
)
from steering import SteeringConfig, SteeringVector

from .steering_fakes import MonotonicModel, SimpleCharTokenizer

HIDDEN = 8


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


def calibration(
    gain: float, sigma: float = 1.0, base: float = 2.0, layers=(1, 2)
) -> AlignmentCalibration:
    return AlignmentCalibration(
        layers={
            layer: LayerCalibration(resting_mean=0.0, resting_sigma=sigma, token_sigma=sigma)
            for layer in (*layers, max(layers) + 1)
        },
        readout_layer=max(layers) + 1,
        gain=gain,
        reference_strength=base,
        num_passages=4,
    )


class Plant:
    """The measured shape: the readout sees ``resting + content + gain * last strength``."""

    def __init__(self, gain: float, direction: torch.Tensor, content: float = 0.0):
        self.gain, self.direction, self.content = gain, direction, content
        self.last_strength = 0.0

    def hidden(self) -> torch.Tensor:
        projection = self.content + self.gain * self.last_strength
        return (projection * self.direction).view(1, 1, -1)


def loop_config(**overrides) -> SteeringConfig:
    defaults = dict(
        steering_layers=[1, 2],
        base_strength=2.0,
        min_strength=0.0,
        max_strength=4.0,
        adaptive=True,
        measurement_filter_alpha=1.0,
        orthogonal_projection=False,
    )
    defaults.update(overrides)
    return SteeringConfig(**defaults)


def test_uncalibrated_loop_keeps_the_legacy_cosine_contract():
    """Without a calibration the loop regulates cos(h, v) toward target_alignment, as before."""
    config = SteeringConfig(
        adaptive=True, target_alignment=0.99, base_strength=0.3, max_strength=5.0
    )
    homeostat = AdaptiveHomeostat(config)
    direction = unit(torch.randn(HIDDEN))
    orthogonal = unit(torch.randn(HIDDEN) - 0 * direction)
    orthogonal = unit(orthogonal - (orthogonal @ direction) * direction)

    strength = homeostat.compute_strength(orthogonal.view(1, 1, -1), direction)

    assert strength > config.base_strength
    assert homeostat.setpoint == pytest.approx(0.99)
    assert not homeostat.calibrated


def test_calibrated_loop_settles_at_the_reference_strength_on_resting_content():
    """The setpoint is the lift the reference strength produces: no deficit, no correction."""
    config = loop_config()
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    plant = Plant(gain=0.5, direction=direction)

    for _ in range(60):
        plant.last_strength = homeostat.compute_strength(plant.hidden(), direction)

    assert plant.last_strength == pytest.approx(config.base_strength, abs=1e-3)
    assert abs(homeostat.controller.snapshot(homeostat.goal).error) < 1e-3


def test_calibrated_loop_compensates_a_content_deficit_and_removes_the_error():
    """A stream below its resting alignment is pushed harder until the lift is restored."""
    config = loop_config()
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    plant = Plant(gain=0.5, direction=direction, content=-0.5)

    for _ in range(80):
        plant.last_strength = homeostat.compute_strength(plant.hidden(), direction)

    # Restoring a 0.5 deficit through a gain of 0.5 takes one extra unit of strength.
    assert plant.last_strength == pytest.approx(config.base_strength + 1.0, abs=1e-2)
    assert abs(homeostat.controller.snapshot(homeostat.goal).error) < 1e-2


def test_calibrated_loop_saturates_at_the_band_and_reports_it():
    config = loop_config(max_strength=2.5)
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    plant = Plant(gain=0.5, direction=direction, content=-5.0)

    for _ in range(40):
        plant.last_strength = homeostat.compute_strength(plant.hidden(), direction)

    assert plant.last_strength == pytest.approx(2.5)
    assert homeostat.status()["integral_saturated"] is True


def test_gains_are_derived_from_the_calibration_unless_pinned():
    direction_free = AdaptiveHomeostat(loop_config(), calibration=calibration(gain=0.5))
    kp, ki = direction_free.gains()
    assert ki > 0
    # A filter with alpha 1 has no time constant, so SIMC prescribes integral-only control.
    assert kp == 0.0

    pinned = AdaptiveHomeostat(loop_config(kp=0.3, ki=0.05), calibration=calibration(gain=0.5))
    assert pinned.gains() == (0.3, 0.05)

    filtered = AdaptiveHomeostat(
        loop_config(measurement_filter_alpha=0.1), calibration=calibration(gain=0.5)
    )
    kp_filtered, _ = filtered.gains()
    assert kp_filtered > 0


def test_set_gains_rejects_an_integral_gain_the_plant_cannot_stabilise():
    homeostat = AdaptiveHomeostat(loop_config(), calibration=calibration(gain=0.5))
    limit = homeostat.max_stable_ki()
    assert limit == pytest.approx(2.0 / 0.5)
    with pytest.raises(ValueError, match="ki"):
        homeostat.set_gains(ki=limit * 1.01)
    homeostat.set_gains(ki=limit * 0.5, kp=0.1)
    assert homeostat.gains() == (0.1, pytest.approx(limit * 0.5))


def test_broadcast_strength_is_shared_and_falls_back_when_the_sensor_is_silent():
    """Local autonomy: the sensor's strength while it fires, a local rule when it stops."""
    config = loop_config()
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    homeostat.bind_layers([1, 2], {1: direction, 2: direction})

    deficit_hidden = (-3.0 * direction).view(1, 1, -1)
    # Pass 1: nothing has been measured yet, both actuators inject the base strength.
    assert homeostat.actuate(1, deficit_hidden, direction) == config.base_strength
    assert homeostat.actuate(2, deficit_hidden, direction) == config.base_strength
    homeostat.compute_strength(deficit_hidden, direction)
    # Pass 2: the sensor's verdict is broadcast to every layer.
    broadcast = homeostat.actuate(1, deficit_hidden, direction)
    assert broadcast > config.base_strength
    assert homeostat.actuate(2, deficit_hidden, direction) == broadcast
    assert homeostat.sensor_alive

    # The sensor stops firing; by pass 3 the actuators notice and regulate locally.
    local = homeostat.actuate(1, deficit_hidden, direction)
    assert not homeostat.sensor_alive
    assert local > config.base_strength
    assert local <= config.max_strength
    resting_hidden = torch.zeros(1, 1, HIDDEN)
    assert homeostat.actuate(1, resting_hidden, direction) == pytest.approx(config.base_strength)


def test_local_rule_discounts_the_passthrough_of_lower_layers():
    """An upper layer must not read its lower neighbour's injection as a surplus to undo."""
    config = loop_config()
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    homeostat.bind_layers([1, 2], {1: direction, 2: direction})
    # Two passes without a sensor step: by the third the actuators regulate locally.
    for _ in range(2):
        homeostat.actuate(1, torch.zeros(1, 1, HIDDEN), direction)

    lower = homeostat.actuate(1, torch.zeros(1, 1, HIDDEN), direction)
    # Layer 2 sees exactly what layer 1 injected, and nothing else.
    upper = homeostat.actuate(2, (lower * direction).view(1, 1, -1), direction)
    assert upper == pytest.approx(lower)


def test_reset_clears_loop_memory_and_histories():
    config = loop_config()
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    for _ in range(3):
        homeostat.compute_strength((-1.0 * direction).view(1, 1, -1), direction)
    assert len(homeostat.alignment_history) == 3

    homeostat.reset()
    assert len(homeostat.alignment_history) == 0
    assert len(homeostat.strength_history) == 0
    assert homeostat.controller.states == {}
    assert homeostat.current_strength == config.base_strength


def test_snapshot_round_trips_through_a_dict():
    config = loop_config()
    direction = unit(torch.randn(HIDDEN))
    homeostat = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    for _ in range(3):
        homeostat.compute_strength((-1.0 * direction).view(1, 1, -1), direction)

    snapshot = homeostat.snapshot()
    restored = AdaptiveHomeostat(config, calibration=calibration(gain=0.5))
    restored.restore(snapshot)
    assert restored.controller.snapshot(restored.goal) == homeostat.controller.snapshot(
        homeostat.goal
    )
    assert restored.current_strength == homeostat.current_strength


# --- calibration and wiring on the identity-layer fakes


def identity_vectors(direction: torch.Tensor, layers) -> dict[int, SteeringVector]:
    return {layer: SteeringVector("goal", direction.clone(), layer) for layer in layers}


def test_calibration_recovers_the_additive_passthrough_gain():
    """Identity layers pass an injection straight up, so the gain is the summed cosines."""
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    tokenizer = SimpleCharTokenizer()
    direction = unit(torch.eye(HIDDEN)[1])
    vectors = identity_vectors(direction, layers=[1, 2, 3])
    config = SteeringConfig(steering_layers=[1, 2], base_strength=2.0, readout_layer=3)

    result = calibrate_alignment(model, tokenizer, vectors, config, texts=["abc", "defg", "hij"])

    assert result.readout_layer == 3
    assert set(result.layers) == {1, 2, 3}
    assert result.gain == pytest.approx(2.0, abs=1e-4)
    assert result.layers[3].resting_mean == pytest.approx(0.0, abs=1e-5)
    assert result.reference_strength == 2.0


def test_cognitive_homeostat_attaches_actuators_and_one_sensor():
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    tokenizer = SimpleCharTokenizer()
    direction = unit(torch.eye(HIDDEN)[1])
    config = SteeringConfig(
        steering_layers=[1, 2],
        readout_layer=3,
        base_strength=2.0,
        adaptive=True,
        measurement_filter_alpha=1.0,
        orthogonal_projection=False,
    )
    coordinator = CognitiveHomeostat(config)
    coordinator.add_steering_vectors(identity_vectors(direction, layers=[1, 2, 3]))
    coordinator.calibrate(model, tokenizer, texts=["abc", "defg", "hij"])
    coordinator.attach_to_model(model)

    assert {layer for layer, hook in coordinator.hooks.items() if hook.injects} == {1, 2}
    assert coordinator.hooks[3].measures and not coordinator.hooks[3].injects

    ids = tokenizer("abcd", return_tensors="pt")["input_ids"]
    for _ in range(3):
        model(ids)
    stats = coordinator.get_alignment_stats()
    assert stats["pid"]["step_count"] == 3
    assert stats["setpoint"] == pytest.approx(coordinator.homeostat.setpoint)
    assert config.min_strength <= stats["current_strength"] <= config.max_strength
    coordinator.detach_from_model()


def test_top_actuator_doubles_as_sensor_when_no_readout_vector_exists():
    config = SteeringConfig(steering_layers=[1, 2], adaptive=True, orthogonal_projection=False)
    coordinator = CognitiveHomeostat(config)
    coordinator.add_steering_vectors(identity_vectors(unit(torch.randn(HIDDEN)), layers=[1, 2]))
    model = MonotonicModel(vocab_size=32, hidden_dim=HIDDEN, num_layers=4)
    coordinator.attach_to_model(model)

    assert coordinator.hooks[2].injects and coordinator.hooks[2].measures
    assert coordinator.hooks[1].injects and not coordinator.hooks[1].measures
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
