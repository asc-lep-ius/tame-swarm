import torch
import torch.nn as nn

from steering import SteeringConfig, SteeringVector, AdaptiveHomeostat, CognitiveHomeostat


def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.base_strength == 0.3
    assert cfg.adaptive is True
    assert cfg.target_alignment == 0.7
    assert cfg.kp == 0.5
    assert cfg.max_strength == 1.5
    assert cfg.min_strength == 0.0
    assert cfg.orthogonal_projection is True


def test_steering_vector_normalization():
    raw = torch.tensor([3.0, 4.0])
    sv = SteeringVector(name="test", vector=raw, layer=0)
    assert torch.allclose(sv.vector.norm(), torch.tensor(1.0), atol=1e-6)


def test_adaptive_homeostat_strength_range():
    cfg = SteeringConfig(
        base_strength=0.3,
        min_strength=0.0,
        max_strength=1.5,
        adaptive=True,
        kp=0.5,
        target_alignment=0.7,
    )
    homeostat = AdaptiveHomeostat(cfg)

    hidden = torch.randn(1, 4, 32)
    steer_vec = torch.randn(32)
    steer_vec = steer_vec / steer_vec.norm()

    for _ in range(50):
        strength = homeostat.compute_strength(hidden, steer_vec)
        assert cfg.min_strength <= strength <= cfg.max_strength


def test_adaptive_homeostat_increases_strength_on_low_alignment():
    cfg = SteeringConfig(
        base_strength=0.3,
        adaptive=True,
        target_alignment=0.99,
        kp=0.5,
        min_strength=0.0,
        max_strength=5.0,
    )
    homeostat = AdaptiveHomeostat(cfg)

    steer_vec = torch.randn(32)
    steer_vec = steer_vec / steer_vec.norm()

    orthogonal = torch.randn(32)
    orthogonal = orthogonal - (orthogonal @ steer_vec) * steer_vec
    orthogonal = orthogonal / orthogonal.norm()
    hidden = orthogonal.unsqueeze(0).unsqueeze(0)

    strength = homeostat.compute_strength(hidden, steer_vec)
    assert strength > cfg.base_strength, (
        f"Strength should be above base_strength when alignment is low, got {strength}"
    )


def test_adaptive_homeostat_reset_clears_history():
    cfg = SteeringConfig(adaptive=True)
    homeostat = AdaptiveHomeostat(cfg)

    hidden = torch.randn(1, 4, 32)
    steer_vec = torch.randn(32)
    steer_vec = steer_vec / steer_vec.norm()

    homeostat.compute_strength(hidden, steer_vec)
    assert len(homeostat.alignment_history) > 0
    assert len(homeostat.strength_history) > 0

    homeostat.reset()
    assert len(homeostat.alignment_history) == 0
    assert len(homeostat.strength_history) == 0


def test_cognitive_homeostat_attach_detach():
    cfg = SteeringConfig(steering_layers=[0, 1])

    class FakeLayer(nn.Module):
        def forward(self, x):
            return x

    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(), FakeLayer(), FakeLayer()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeTransformer()

    model = FakeModel()

    sv0 = SteeringVector(name="test0", vector=torch.randn(32), layer=0)
    sv1 = SteeringVector(name="test1", vector=torch.randn(32), layer=1)

    homeostat = CognitiveHomeostat(cfg)
    homeostat.add_steering_vectors({0: sv0, 1: sv1})

    homeostat.attach_to_model(model)
    assert len(homeostat._registered_hooks) == 2

    homeostat.detach_from_model()
    assert len(homeostat._registered_hooks) == 0
    assert len(homeostat.hooks) == 0
