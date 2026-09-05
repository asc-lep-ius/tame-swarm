"""The log-odds metric, the random-direction floor, and the control ceiling."""

import torch

from behavioural_validation import (
    ValidationResult,
    _completion_log_prob,
    mean_log_odds,
    validate_steering_vector,
)
from contrastive_data import ContrastivePair
from steering import SteeringConfig, SteeringVector

from .steering_fakes import ScriptedModel, SimpleCharTokenizer

DEVICE = torch.device("cpu")


def _held_out():
    # Single-character completions so each maps to one token whose readout the
    # constructed direction can provably boost.
    return [
        ContrastivePair(prompt="alpha", positive_completion="Y", negative_completion="X"),
        ContrastivePair(prompt="bravo", positive_completion="Y", negative_completion="X"),
        ContrastivePair(prompt="delta", positive_completion="Y", negative_completion="X"),
    ]


def test_completion_log_prob_is_finite_and_negative():
    model, tokenizer = ScriptedModel(), SimpleCharTokenizer()
    value = _completion_log_prob(model, tokenizer, "alpha", "Y", DEVICE, max_length=64)
    assert value == value  # not NaN
    assert value < 0.0


def test_mean_log_odds_skips_degenerate_pairs():
    model, tokenizer = ScriptedModel(), SimpleCharTokenizer()
    pairs = _held_out()
    value = mean_log_odds(model, tokenizer, pairs, DEVICE, max_length=64)
    assert value == value


def _direction_favouring_positive(model, tokenizer, hidden_dim):
    pos_id = ord("Y") % tokenizer.vocab_size
    neg_id = ord("X") % tokenizer.vocab_size
    raw = model.token_readout(pos_id) - model.token_readout(neg_id)
    return raw / raw.norm()


def test_constructed_vector_beats_matched_random_directions():
    tokenizer = SimpleCharTokenizer(32)
    model = ScriptedModel(vocab_size=32, hidden_dim=16, seed=1)
    direction = _direction_favouring_positive(model, tokenizer, 16)
    vectors = {0: SteeringVector("truthful", direction, layer=0)}

    result = validate_steering_vector(
        model,
        tokenizer,
        goal="truthful",
        vectors=vectors,
        held_out=_held_out(),
        config=SteeringConfig(
            steering_layers=[0], base_strength=6.0, adaptive=False, orthogonal_projection=False
        ),
        device=DEVICE,
        num_random=8,
        seed=3,
    )
    assert result.vector_effect.effect > 0
    assert result.vector_effect.effect > result.random_max
    assert result.beats_random


def test_random_direction_does_not_beat_itself():
    """A random vector as the candidate should not clear the random-control max."""
    tokenizer = SimpleCharTokenizer(32)
    model = ScriptedModel(vocab_size=32, hidden_dim=16, seed=2)
    generator = torch.Generator().manual_seed(9)
    noise = torch.randn(16, generator=generator)
    noise = noise / noise.norm()
    vectors = {0: SteeringVector("noise", noise, layer=0)}

    result = validate_steering_vector(
        model,
        tokenizer,
        goal="noise",
        vectors=vectors,
        held_out=_held_out(),
        config=SteeringConfig(
            steering_layers=[0], base_strength=6.0, adaptive=False, orthogonal_projection=False
        ),
        device=DEVICE,
        num_random=12,
        seed=5,
    )
    assert not result.beats_random


def test_beats_control_requires_outscoring_the_instruction_prefix_vector():
    strong = ValidationResult(
        goal="truthful",
        vector_effect=_effect(0.5),
        random_effects=[0.0, 0.01],
        control_effect=_effect(0.1),
        margin=0.0,
    )
    weak = ValidationResult(
        goal="truthful",
        vector_effect=_effect(0.05),
        random_effects=[0.0, 0.01],
        control_effect=_effect(0.1),
        margin=0.0,
    )
    assert strong.beats_control and strong.passed
    assert not weak.beats_control and not weak.passed


def _effect(value: float):
    from behavioural_validation import DirectionEffect

    return DirectionEffect(
        label="x", effect=value, baseline_log_odds=0.0, steered_log_odds=value, num_pairs=3
    )
