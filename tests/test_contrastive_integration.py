"""The contrastive pipeline end to end on a model whose geometry is known (#6).

``tests/test_contrastive_data.py`` checks the pairs, ``tests/test_steering_extraction.py``
the read position, ``tests/test_behavioural_validation.py`` the metric -- each on
its own. This runs the pipeline as the server does: register pairs, extract a
vector at the completion position, and put it through the behavioural gate
against matched random directions and the instruction-prefix control.

The model is the identity-layer fake with its readout **tied** to its embedding.
On it every claim below is provable rather than hoped for: the read at the last
completion token *is* that token's embedding, so the diff-in-means over pairs
that end in ``Y`` versus ``X`` is exactly ``embed[Y] - embed[X]``, injecting it
raises ``logit[Y] - logit[X]`` at every position, and the prefix control -- whose
two arms both end in a full stop -- reads identically on both sides and yields
an inert vector. Whether the same holds on Qwen3-1.7B is what the recorded
certification (README, "Steering validation") and ``tests/test_real_model.py``
measure.

The pairing is #3's lesson itself: read the prompt instead of the completion and
the vector is the direction that separates two prompts, which does not beat a
random one.
"""

import logging
import math
import random
from typing import cast

import pytest
import torch
import torch.nn as nn

from behavioural_validation import pca_separability, validate_steering_vector
from contrastive_data import (
    COMPLETION_FORMAT,
    ContrastivePair,
    clear_custom_pairs,
    load_instruction_prefix_control,
    register_contrastive_pairs,
)
from contrastive_templates import TIERS
from steering import SteeringConfig, SteeringVectorExtractor
from steering_pipeline import extract_steering_vectors

from .steering_fakes import ScriptedModel, SimpleCharTokenizer

logger = logging.getLogger(__name__)

GOAL = "safe"
VOCAB = 128
HIDDEN = 16
LAYER = 0
DEVICE = torch.device("cpu")
POSITIVE, NEGATIVE = "Y", "X"
EXTRACTION_PER_TIER = 15
HELD_OUT_PER_TIER = 5
ALPHABET = "abcdefghijklmnopqrstuvw"


class TiedReadoutModel(ScriptedModel):
    """The scripted fake with ``logits = hidden @ embed.T``: what it reads is what it writes."""

    def __init__(self, seed: int = 0):
        super().__init__(vocab_size=VOCAB, hidden_dim=HIDDEN, seed=seed)
        self.unembed = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.unembed.weight = cast(nn.Embedding, self.model.embed_tokens).weight


def _pairs(count_per_tier: int, seed: int) -> list[ContrastivePair]:
    """Shared prompts of varied content, the same two one-token completions."""
    rng = random.Random(seed)
    pairs = []
    for tier in TIERS:
        for _ in range(count_per_tier):
            prompt = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 8)))
            pairs.append(
                ContrastivePair(
                    prompt=prompt,
                    positive_completion=POSITIVE,
                    negative_completion=NEGATIVE,
                    tier=tier,
                )
            )
    return pairs


@pytest.fixture
def pipeline():
    clear_custom_pairs()
    register_contrastive_pairs(GOAL, _pairs(EXTRACTION_PER_TIER, seed=1), replace_existing=True)
    model = TiedReadoutModel()
    tokenizer = SimpleCharTokenizer(VOCAB)
    config = SteeringConfig(
        steering_layers=[LAYER], base_strength=6.0, adaptive=False, orthogonal_projection=False
    )
    yield model, tokenizer, config, _pairs(HELD_OUT_PER_TIER, seed=2)
    clear_custom_pairs()


def _validate(model, tokenizer, config, held_out):
    extraction = extract_steering_vectors(
        model, tokenizer, goal=GOAL, config=config, source="custom", pair_format=COMPLETION_FORMAT
    )
    extractor = SteeringVectorExtractor(model, tokenizer, [LAYER])
    control = extractor.extract_from_pairs(list(load_instruction_prefix_control(GOAL)))
    result = validate_steering_vector(
        model,
        tokenizer,
        goal=GOAL,
        vectors=extraction.vectors,
        held_out=held_out,
        config=config,
        device=DEVICE,
        control_vectors=control,
        num_random=8,
        seed=3,
    )
    return extraction, result


def test_the_extracted_vector_beats_random_directions_and_the_prefix_control(pipeline):
    model, tokenizer, config, held_out = pipeline
    extraction, result = _validate(model, tokenizer, config, held_out)

    assert extraction.pair_count == EXTRACTION_PER_TIER * len(TIERS)
    assert extraction.pair_format == COMPLETION_FORMAT
    assert extraction.tier_counts == dict.fromkeys(TIERS, EXTRACTION_PER_TIER)
    assert result.vector_effect.effect > 0.0
    assert result.beats_random and result.beats_control and result.passed
    assert result.control_effect is not None
    assert result.control_effect.effect == pytest.approx(0.0, abs=1e-6)


def test_the_extracted_vector_is_the_known_direction_at_unit_norm(pipeline):
    model, tokenizer, config, _ = pipeline
    extraction = extract_steering_vectors(
        model, tokenizer, goal=GOAL, config=config, source="custom", pair_format=COMPLETION_FORMAT
    )
    embed = model.model.embed_tokens.weight
    expected = embed[ord(POSITIVE) % VOCAB] - embed[ord(NEGATIVE) % VOCAB]

    vector = extraction.vectors[LAYER].vector
    assert torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(vector, expected / expected.norm(), atol=1e-5)


def test_reading_the_prompt_instead_of_the_completion_fails_the_gate(pipeline, monkeypatch):
    """#3's defect, replayed: a vector read where the behaviour is described, not produced."""
    model, tokenizer, config, held_out = pipeline
    original = SteeringVectorExtractor._read_completion_activations

    def read_the_prompt(self, prompt, completion, read_position, max_length, input_device):
        # The extractor clamps every read into the completion, so the prompt's last
        # token is reached by moving the boundary one character to the left.
        return original(self, prompt[:-1], prompt[-1:] + completion, 0, max_length, input_device)

    monkeypatch.setattr(SteeringVectorExtractor, "_read_completion_activations", read_the_prompt)
    _, result = _validate(model, tokenizer, config, held_out)

    assert not result.beats_random


def test_pca_separability_is_reported_and_is_not_the_gate(pipeline):
    """The diagnostic #3 demoted: computed, logged, and read by nothing that decides.

    On this fake the behavioural pairs read as two points, so their separability
    is bounded only by float rounding, and the prefix control as one point, zero.
    The numbers are reported so a reader can compare them with the gate's verdicts;
    the gate's own result carries no such field and does not change if the
    diagnostic does.
    """
    model, tokenizer, _, _ = pipeline
    extractor = SteeringVectorExtractor(model, tokenizer, [LAYER])

    positives, negatives = extractor.read_pairs(_pairs(HELD_OUT_PER_TIER, seed=2))
    behavioural = pca_separability(positives[LAYER], negatives[LAYER])
    positives, negatives = extractor.read_pairs(list(load_instruction_prefix_control(GOAL)))
    control = pca_separability(positives[LAYER], negatives[LAYER])
    logger.info(
        "PCA separability: behavioural %.3f, instruction-prefix control %.3f", behavioural, control
    )

    assert behavioural > 1e3
    assert control == 0.0
    assert not hasattr(validate_steering_vector, "separability")


def test_pca_separability_on_noisy_arms_is_a_finite_ratio():
    generator = torch.Generator().manual_seed(0)
    positives = torch.randn(40, HIDDEN, generator=generator) + 3.0 * torch.eye(HIDDEN)[0]
    negatives = torch.randn(40, HIDDEN, generator=generator)

    separability = pca_separability(positives, negatives)

    assert math.isfinite(separability)
    assert 1.5 < separability < 6.0
    assert pca_separability(positives, positives) == 0.0
