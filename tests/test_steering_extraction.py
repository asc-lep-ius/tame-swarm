"""Completion-position extraction and the pipeline that normalises and reports."""

import torch

from contrastive_data import ContrastivePair
from steering import SteeringConfig, SteeringVectorExtractor
from steering_pipeline import (
    extract_steering_vectors,
    goal_similarity_matrix,
    log_goal_similarity,
)

from .steering_fakes import MonotonicModel, SimpleCharTokenizer


def test_reads_the_last_completion_token_not_a_prompt_token():
    """read_position=-1 must land on the final completion token.

    The MonotonicModel encodes the token id at each position into component 0, so
    the read value names the position that was read. Prompt "abcd" then completion
    "XY": the last completion token is 'Y', id ord('Y')%32.
    """
    model = MonotonicModel(vocab_size=32, hidden_dim=8)
    extractor = SteeringVectorExtractor(model, SimpleCharTokenizer(32), layers=[0, 1])

    extractor._register_hooks()
    try:
        acts = extractor._read_completion_activations(
            "abcd", "XY", read_position=-1, max_length=64, input_device=torch.device("cpu")
        )
    finally:
        extractor._remove_hooks()
    assert acts is not None
    assert acts[0][0].item() == ord("Y") % 32


def test_read_position_zero_lands_on_the_first_completion_token():
    model = MonotonicModel(vocab_size=32, hidden_dim=8)
    extractor = SteeringVectorExtractor(model, SimpleCharTokenizer(32), layers=[0])

    extractor._register_hooks()
    try:
        acts = extractor._read_completion_activations(
            "abcd", "XY", read_position=0, max_length=64, input_device=torch.device("cpu")
        )
    finally:
        extractor._remove_hooks()
    assert acts is not None
    assert acts[0][0].item() == ord("X") % 32


def test_extract_from_pairs_differences_the_completion_reads():
    model = MonotonicModel(vocab_size=32, hidden_dim=8)
    extractor = SteeringVectorExtractor(model, SimpleCharTokenizer(32), layers=[0])
    pairs = [
        ContrastivePair(prompt="abcd", positive_completion="Y", negative_completion="X"),
    ]
    vectors = extractor.extract_from_pairs(pairs)
    # Diff-in-means at component 0 is (ord Y - ord X); the vector is normalised, so
    # its component 0 must be positive and it must be unit norm.
    assert vectors[0].vector[0].item() > 0
    assert torch.allclose(vectors[0].vector.norm(), torch.tensor(1.0), atol=1e-5)


def test_extracted_vectors_are_l2_normalised():
    model = MonotonicModel(vocab_size=32, hidden_dim=8)
    extractor = SteeringVectorExtractor(model, SimpleCharTokenizer(32), layers=[0, 1])
    pairs = [
        ContrastivePair(prompt="hello", positive_completion=" yes", negative_completion=" no"),
        ContrastivePair(prompt="world", positive_completion=" up", negative_completion=" down"),
    ]
    vectors = extractor.extract_from_pairs(pairs)
    for vector in vectors.values():
        assert torch.allclose(vector.vector.norm(), torch.tensor(1.0), atol=1e-5)


def test_pipeline_extracts_builtin_goal_with_metadata():
    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    extraction = extract_steering_vectors(
        model,
        SimpleCharTokenizer(64),
        goal="truthful",
        config=SteeringConfig(steering_layers=[1, 2]),
        source="builtin",
    )
    assert extraction.goal == "truthful"
    assert extraction.pair_count >= 60
    assert set(extraction.vectors) == {1, 2}
    assert sum(extraction.tier_counts.values()) == extraction.pair_count
    for vector in extraction.vectors.values():
        assert vector.name == "truthful"


def test_goal_similarity_matrix_is_symmetric_with_unit_diagonal():
    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    tokenizer = SimpleCharTokenizer(64)
    config = SteeringConfig(steering_layers=[2])
    vectors_by_goal = {
        goal: extract_steering_vectors(model, tokenizer, goal=goal, config=config).vectors
        for goal in ("truthful", "reasoning", "safe")
    }
    goals, matrix = goal_similarity_matrix(vectors_by_goal, layer=2)
    assert goals == ["reasoning", "safe", "truthful"]
    assert torch.allclose(matrix, matrix.T, atol=1e-5)
    assert torch.allclose(matrix.diagonal(), torch.ones(3), atol=1e-5)

    pairwise = log_goal_similarity(vectors_by_goal, layer=2)
    assert set(pairwise) == {("reasoning", "safe"), ("reasoning", "truthful"), ("safe", "truthful")}
