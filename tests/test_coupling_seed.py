"""Seeding the routing coupling: only where the direction is certified, only as injected.

#4's layer sweep found the ``truthful`` direction steering *toward* falsehood
below layer 13 and reading prompt wording at 12, while the MoB range on the same
profile starts at 6. A coupling seeded across the MoB range would therefore
inject, into routing, a direction the gate rejects at most of those layers. The
rule these tests pin: a MoB layer is coupled only when the goal's certification
names that layer on this model and the extraction is the certified pair, and the
direction it is coupled to is the one the hooks inject, after the capability
projection -- never the raw vector.
"""

import pytest
import torch

from contrastive_data import (
    BUILTIN_SOURCE,
    CERTIFIED,
    CERTIFIED_MODEL,
    MULTIPLE_CHOICE_FORMAT,
    TRUTHFUL_LAYERS,
    Certification,
)
from homeostat import CognitiveHomeostat
from mob import SteeringCouplingConfig, apply_mob_to_model, get_mob_layers
from steering import SteeringConfig, SteeringVector
from steering_pipeline import (
    SteeringExtraction,
    UncertifiedDirectionError,
    certified_coupling_layers,
    seed_coupling,
)

from .conftest import TINY_HIDDEN_DIM

MOB_LAYERS = [1, 2, 3]
VECTOR_LAYERS = [2, 3]


# --- The gate ----------------------------------------------------------------


def test_certified_layers_come_from_the_record():
    assert certified_coupling_layers("truthful", CERTIFIED_MODEL) == TRUTHFUL_LAYERS
    assert certified_coupling_layers("truthful") == TRUTHFUL_LAYERS


def test_a_goal_the_gate_never_passed_cannot_seed():
    with pytest.raises(UncertifiedDirectionError, match="no certified layers"):
        certified_coupling_layers("deliberation")


def test_a_certification_without_layers_cannot_seed(monkeypatch):
    monkeypatch.setitem(CERTIFIED, "bare", Certification(BUILTIN_SOURCE, MULTIPLE_CHOICE_FORMAT))

    with pytest.raises(UncertifiedDirectionError, match="no certified layers"):
        certified_coupling_layers("bare")


def test_a_certification_measured_on_another_model_cannot_seed():
    """The hooks serve another model with a warning; the coupling's alternative is to stay off."""
    with pytest.raises(UncertifiedDirectionError, match="certified on"):
        certified_coupling_layers("truthful", "google/gemma-2-2b-it")


# --- Seeding -----------------------------------------------------------------


def _extraction(vectors: dict[int, SteeringVector], certified: bool) -> SteeringExtraction:
    return SteeringExtraction(
        goal="truthful",
        vectors=vectors,
        pair_count=8,
        source="truthful_qa" if certified else BUILTIN_SOURCE,
        layers=sorted(vectors),
        tier_counts={},
        pair_format=MULTIPLE_CHOICE_FORMAT,
        certified=certified,
        fallback_reason=None if certified else "truthful_qa needs the train extra",
    )


@pytest.fixture
def seeded_model(tiny_causal_lm, tiny_mob_config):
    """A tiny model with MoB at layers 1-3 and a homeostat holding vectors at 2 and 3."""
    model = apply_mob_to_model(tiny_causal_lm, tiny_mob_config, layers_to_modify=MOB_LAYERS)
    torch.manual_seed(3)
    vectors = {
        layer: SteeringVector("truthful", torch.randn(TINY_HIDDEN_DIM), layer)
        for layer in VECTOR_LAYERS
    }
    homeostat = CognitiveHomeostat(
        SteeringConfig(steering_layers=VECTOR_LAYERS, adaptive=False, orthogonal_projection=True)
    )
    homeostat.add_steering_vectors(vectors)
    # A basis the raw vectors overlap, so the projected direction is a different
    # vector from the raw one and the test can tell which was seeded.
    basis = torch.linalg.qr(torch.randn(TINY_HIDDEN_DIM, 4))[0].T
    homeostat.set_capability_subspaces({layer: basis for layer in VECTOR_LAYERS})
    return model, homeostat, vectors


def _mob_by_layer(model):
    return dict(zip(MOB_LAYERS, get_mob_layers(model), strict=True))


def test_seed_couples_only_the_certified_mob_layers_with_the_injected_direction(seeded_model):
    model, homeostat, vectors = seeded_model
    config = SteeringCouplingConfig(hidden_dim=TINY_HIDDEN_DIM, warmup_steps=50)

    seeded = seed_coupling(model, homeostat, _extraction(vectors, True), (2, 3), config)

    mobs = _mob_by_layer(model)
    assert sorted(seeded) == [2, 3]
    assert not hasattr(mobs[1], "coupling"), "layer 1 is a MoB layer the gate never passed"
    for layer in (2, 3):
        coupling = mobs[layer].coupling
        assert seeded[layer] is coupling
        assert coupling.config.warmup_steps == 50
        injected, retained = homeostat.projected_direction(layer)
        assert retained < 1.0
        assert torch.allclose(coupling.steering_direction, injected, atol=1e-6)
        raw = vectors[layer].vector
        assert not torch.allclose(coupling.steering_direction, raw / raw.norm(), atol=1e-3)


def test_seed_skips_certified_layers_that_carry_no_mob_layer_or_no_vector(seeded_model):
    """The readout above the top actuator is certified but not a MoB layer: skipped, no error."""
    model, homeostat, vectors = seeded_model
    config = SteeringCouplingConfig(hidden_dim=TINY_HIDDEN_DIM)

    seeded = seed_coupling(model, homeostat, _extraction(vectors, True), (0, 1, 2, 9), config)

    assert sorted(seeded) == [2]
    assert not hasattr(_mob_by_layer(model)[1], "coupling"), "a MoB layer with no vector"


def test_seed_refuses_an_uncertified_extraction(seeded_model):
    """The right layer numbers do not launder a fallback vector."""
    model, homeostat, vectors = seeded_model
    config = SteeringCouplingConfig(hidden_dim=TINY_HIDDEN_DIM)

    with pytest.raises(UncertifiedDirectionError, match="uncertified"):
        seed_coupling(model, homeostat, _extraction(vectors, False), (2, 3), config)

    assert not any(hasattr(mob, "coupling") for mob in get_mob_layers(model))
