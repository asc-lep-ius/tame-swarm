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
    # Explicit built-in source: the certified truthful source is a HF dataset, and a
    # unit test must not reach for the network or the cache.
    vectors_by_goal = {
        goal: extract_steering_vectors(
            model, tokenizer, goal=goal, config=config, source="builtin"
        ).vectors
        for goal in ("truthful", "reasoning", "safe")
    }
    goals, matrix = goal_similarity_matrix(vectors_by_goal, layer=2)
    assert goals == ["reasoning", "safe", "truthful"]
    assert torch.allclose(matrix, matrix.T, atol=1e-5)
    assert torch.allclose(matrix.diagonal(), torch.ones(3), atol=1e-5)

    pairwise = log_goal_similarity(vectors_by_goal, layer=2)
    assert set(pairwise) == {("reasoning", "safe"), ("reasoning", "truthful"), ("safe", "truthful")}


def test_steering_vector_keeps_a_zero_diff_inert_instead_of_nan():
    """Identical completions give a zero diff-in-means; normalising it must not NaN."""
    from steering import SteeringVector

    sv = SteeringVector("degenerate", torch.zeros(8), layer=0)
    assert torch.all(sv.vector == 0)
    assert torch.isfinite(sv.vector).all()


def _tqa_rows(count):
    return [
        {
            "question": f"Q{i}?",
            "mc1_targets": {"choices": [f"right{i}", f"wrong{i}"], "labels": [1, 0]},
        }
        for i in range(count)
    ]


def test_pipeline_default_source_is_the_certified_one():
    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    extraction = extract_steering_vectors(
        model,
        SimpleCharTokenizer(64),
        goal="truthful",
        config=SteeringConfig(steering_layers=[2]),
        load_dataset=lambda *a, **k: _tqa_rows(12),
    )
    assert extraction.source == "truthful_qa"
    assert extraction.pair_format == "multiple_choice"
    assert extraction.certified and extraction.fallback_reason is None
    assert "certified" in extraction.vectors[2].description


def test_pipeline_falls_back_to_builtin_and_flags_the_vector_uncertified():
    def offline(*_a, **_k):
        raise OSError("no cache")

    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    extraction = extract_steering_vectors(
        model,
        SimpleCharTokenizer(64),
        goal="truthful",
        config=SteeringConfig(steering_layers=[2]),
        load_dataset=offline,
    )
    assert extraction.source == "builtin"
    assert extraction.pair_format == "multiple_choice"
    assert not extraction.certified
    assert extraction.fallback_reason and "truthful_qa unavailable" in extraction.fallback_reason
    assert "UNCERTIFIED" in extraction.vectors[2].description


def test_pipeline_explicit_override_is_certified_only_when_it_names_the_certified_pair():
    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    tokenizer = SimpleCharTokenizer(64)
    config = SteeringConfig(steering_layers=[2])
    certified = extract_steering_vectors(
        model, tokenizer, goal="safe", config=config, source="builtin"
    )
    assert certified.certified and certified.pair_format == "completion"
    overridden = extract_steering_vectors(
        model,
        tokenizer,
        goal="safe",
        config=config,
        source="builtin",
        pair_format="multiple_choice",
    )
    assert not overridden.certified and overridden.pair_format == "multiple_choice"


def test_pipeline_reports_a_goal_outside_certified_as_uncertified():
    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    extraction = extract_steering_vectors(
        model,
        SimpleCharTokenizer(64),
        goal="deliberation",
        config=SteeringConfig(steering_layers=[2]),
    )
    assert not extraction.certified
    assert extraction.fallback_reason and "no certified" in extraction.fallback_reason
    assert "UNCERTIFIED" in extraction.vectors[2].description


def test_pipeline_rejects_max_pairs_zero():
    import pytest

    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    with pytest.raises(ValueError, match="no pairs"):
        extract_steering_vectors(
            model,
            SimpleCharTokenizer(64),
            goal="safe",
            config=SteeringConfig(steering_layers=[2]),
            source="builtin",
            max_pairs=0,
        )


def test_pipeline_extracts_the_readout_layer_alongside_the_actuators():
    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    extraction = extract_steering_vectors(
        model,
        SimpleCharTokenizer(64),
        goal="safe",
        config=SteeringConfig(steering_layers=[1, 2], readout_layer=3),
        source="builtin",
    )
    assert sorted(extraction.vectors) == [1, 2, 3]
    assert extraction.layers == [1, 2, 3]


def test_serving_config_takes_layers_and_band_from_the_certification():
    from steering_pipeline import serving_config

    template = SteeringConfig(steering_layers=[6, 7], base_strength=0.3, max_strength=1.5)

    truthful = serving_config("truthful", template)
    assert truthful.steering_layers == [13, 16, 17, 18, 19, 20, 21]
    assert truthful.readout_layer == 22
    assert (truthful.base_strength, truthful.min_strength, truthful.max_strength) == (4.0, 2.0, 6.0)

    # Certified at one strength with no swept band: served constant at that strength.
    reasoning = serving_config("reasoning", template)
    assert reasoning.steering_layers == [14, 18, 22]
    assert reasoning.min_strength == reasoning.max_strength == 4.0

    # No certification at all: the template stands, and extraction reports it uncertified.
    assert serving_config("deliberation", template).steering_layers == [6, 7]
    assert template.steering_layers == [6, 7]


def test_calibration_texts_sample_the_goal_prompts_in_the_served_format():
    from steering_pipeline import calibration_texts

    model = MonotonicModel(vocab_size=64, hidden_dim=8, num_layers=6)
    texts = calibration_texts(model, SimpleCharTokenizer(64), goal="safe", num_prompts=6)

    assert len(texts) == 6
    assert len(set(texts)) == 6
    # No chat template and no generate on the fake: the prompts themselves, unchanged.
    assert all("Human:" in text or "Q:" in text for text in texts)
