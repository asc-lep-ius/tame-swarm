import pytest
import torch
import torch.nn as nn

from homeostat import AdaptiveHomeostat, CognitiveHomeostat, SteeringHook
from steering import (
    SteeringConfig,
    SteeringVector,
    SteeringVectorExtractor,
    estimate_capability_subspace,
    project_out_subspace,
)


def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.base_strength == 0.3
    assert cfg.adaptive is True
    assert cfg.target_alignment == 0.7
    assert cfg.kp is None
    assert cfg.ki is None
    assert cfg.kd == 0.0
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


class _TinyBlock(nn.Module):
    """One transformer-ish layer: enough surface for a forward hook to fire on."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        return (torch.tanh(self.proj(hidden_states)),)


class _TinyModel(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_layers: int = 4, vocab: int = 40):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden_dim)
        self.layers = nn.ModuleList(_TinyBlock(hidden_dim) for _ in range(num_layers))

    def forward(self, input_ids, **_):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class _TinyTokenizer:
    def __call__(self, text, return_tensors=None, max_length=None, truncation=False):
        ids = [(ord(ch) % 40) for ch in text][: max_length or len(text)]
        return _TinyBatch({"input_ids": torch.tensor([ids or [0]])})


class _TinyBatch(dict):
    def to(self, _device):
        return self


def _tiny_corpus(count: int = 12) -> list[str]:
    return [f"passage number {i} about assorted ordinary subject matter" for i in range(count)]


def test_project_out_subspace_removes_only_subspace_components():
    subspace = torch.eye(4)[:2]
    vector = torch.tensor([3.0, 4.0, 5.0, 6.0])

    projected = project_out_subspace(vector, subspace)

    assert torch.allclose(projected, torch.tensor([0.0, 0.0, 5.0, 6.0]), atol=1e-6)
    assert torch.allclose(projected @ subspace.T, torch.zeros(2), atol=1e-6)


def test_estimate_capability_subspace_returns_orthonormal_basis():
    model = _TinyModel()
    subspaces = estimate_capability_subspace(
        model, _TinyTokenizer(), layers=[1, 2], texts=_tiny_corpus(), rank=3
    )

    assert set(subspaces) == {1, 2}
    for basis in subspaces.values():
        assert basis.shape == (3, 16)
        assert torch.allclose(basis @ basis.T, torch.eye(3), atol=1e-4)


def test_estimate_capability_subspace_spans_general_activation_variance():
    """The basis has to be the corpus's own principal axes, not an arbitrary frame.

    Centred activations projected onto the basis must retain more variance than any
    equally sized random frame does; otherwise the "capability subspace" is a name
    for noise.
    """
    torch.manual_seed(5)
    model = _TinyModel()
    extractor = SteeringVectorExtractor(model, _TinyTokenizer(), [2])
    tokens = extractor.collect_token_activations(_tiny_corpus(24))[2]
    centred = tokens - tokens.mean(dim=0, keepdim=True)

    basis = estimate_capability_subspace(
        model, _TinyTokenizer(), layers=[2], texts=_tiny_corpus(24), rank=4
    )[2]

    captured = (centred @ basis.T).pow(2).sum()
    random_frame = torch.linalg.qr(torch.randn(16, 4))[0].T
    random_captured = (centred @ random_frame.T).pow(2).sum()

    assert captured > random_captured


def test_estimate_capability_subspace_rejects_nonpositive_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        estimate_capability_subspace(
            _TinyModel(), _TinyTokenizer(), layers=[1], texts=_tiny_corpus(), rank=0
        )


def test_steering_hook_injects_a_direction_orthogonal_to_the_subspace():
    """The whole point: what reaches the residual stream carries no capability component."""
    hidden_dim = 8
    subspace = torch.eye(hidden_dim)[:2]
    vector = SteeringVector("goal", torch.tensor([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), layer=0)

    config = SteeringConfig(adaptive=False, base_strength=1.0, orthogonal_projection=True)
    hook = SteeringHook(vector, config, capability_subspace=subspace)

    hidden = torch.zeros(1, 1, hidden_dim)
    injected = hook(nn.Identity(), (hidden,), hidden)[0, 0]

    assert torch.allclose(injected @ subspace.T, torch.zeros(2), atol=1e-6)
    assert injected.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_steering_hook_leaves_the_vector_alone_when_projection_is_disabled():
    hidden_dim = 8
    subspace = torch.eye(hidden_dim)[:2]
    raw = torch.tensor([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vector = SteeringVector("goal", raw, layer=0)

    config = SteeringConfig(adaptive=False, base_strength=1.0, orthogonal_projection=False)
    hook = SteeringHook(vector, config, capability_subspace=subspace)

    hidden = torch.zeros(1, 1, hidden_dim)
    injected = hook(nn.Identity(), (hidden,), hidden)[0, 0]

    assert torch.allclose(injected, raw / raw.norm(), atol=1e-6)


def test_steering_hook_falls_back_when_the_subspace_swallows_the_goal():
    """A goal lying inside the capability subspace must not be renormalised noise.

    Projecting leaves floating-point dust; scaling that back to unit norm would
    inject an essentially random direction with full confidence.
    """
    hidden_dim = 8
    vector = SteeringVector("goal", torch.eye(hidden_dim)[0].clone(), layer=0)
    subspace = torch.eye(hidden_dim)[:2]

    config = SteeringConfig(adaptive=False, base_strength=1.0, orthogonal_projection=True)
    hook = SteeringHook(vector, config, capability_subspace=subspace)

    hidden = torch.zeros(1, 1, hidden_dim)
    injected = hook(nn.Identity(), (hidden,), hidden)[0, 0]

    assert torch.allclose(injected, torch.eye(hidden_dim)[0], atol=1e-6)


def test_attach_to_model_hands_each_hook_its_layer_subspace():
    """The wiring itself: a subspace on the homeostat has to reach the live hook.

    This is the step that was missing -- SteeringHook accepted the parameter and
    nothing ever passed one, so the guard inside it could never fire.
    """
    model = _TinyModel(hidden_dim=16, num_layers=4)
    homeostat = CognitiveHomeostat(SteeringConfig(steering_layers=[1, 2]))
    for layer in (1, 2):
        homeostat.add_steering_vector(layer, SteeringVector("goal", torch.randn(16), layer))

    subspaces = homeostat.estimate_capability_subspaces(model, _TinyTokenizer(), _tiny_corpus())
    homeostat.attach_to_model(model)

    assert set(homeostat.hooks) == {1, 2}
    for layer, hook in homeostat.hooks.items():
        assert hook.capability_subspace is not None
        assert torch.equal(hook.capability_subspace, subspaces[layer])

    homeostat.detach_from_model()


def test_set_capability_subspaces_rejects_a_mismatched_hidden_dim():
    homeostat = CognitiveHomeostat(SteeringConfig(steering_layers=[1]))
    homeostat.add_steering_vector(1, SteeringVector("goal", torch.randn(16), 1))

    with pytest.raises(ValueError, match="hidden_dim"):
        homeostat.set_capability_subspaces({1: torch.randn(4, 8)})

    with pytest.raises(ValueError, match="must be"):
        homeostat.set_capability_subspaces({1: torch.randn(16)})


def test_projected_direction_is_what_the_hook_injects():
    """Other consumers of the goal direction must read the same vector as the hook.

    SteeringCoupling keeps its own copy in a buffer; seeding it from the raw vector
    would leave routing steering toward a direction the injection already rejected.
    """
    model = _TinyModel(hidden_dim=16, num_layers=4)
    homeostat = CognitiveHomeostat(
        SteeringConfig(steering_layers=[2], adaptive=False, base_strength=1.0)
    )
    homeostat.add_steering_vector(2, SteeringVector("goal", torch.randn(16), 2))
    homeostat.estimate_capability_subspaces(model, _TinyTokenizer(), _tiny_corpus())
    homeostat.attach_to_model(model)

    direction, retained = homeostat.projected_direction(2)
    hidden = torch.zeros(1, 1, 16)
    injected = homeostat.hooks[2](nn.Identity(), (hidden,), hidden)[0, 0]

    assert torch.allclose(injected, direction, atol=1e-6)
    assert 0.0 < retained <= 1.0
    assert homeostat.get_capability_retention() == {2: retained}

    homeostat.detach_from_model()
