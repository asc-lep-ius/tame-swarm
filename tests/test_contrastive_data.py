"""The pair format, the built-in content, loaders, and the quality checks."""

import pytest
import torch

from contrastive_data import (
    MIN_PAIRS_PER_GOAL,
    MIN_PAIRS_PER_TIER,
    ContrastivePair,
    ContrastivePairSet,
    HFContrastiveLoader,
    _split_shared_prefix,
    available_goals,
    clear_custom_pairs,
    load_contrastive_dataset,
    load_instruction_prefix_control,
    register_contrastive_pairs,
)
from contrastive_templates import BUILTIN_PAIRS, TIERS


@pytest.fixture(autouse=True)
def _clear_custom():
    clear_custom_pairs()
    yield
    clear_custom_pairs()


def test_pair_is_shared_prompt_plus_two_completions_with_a_read_position():
    pair = ContrastivePair(
        prompt="Q: capital of Australia?\nA:",
        positive_completion=" Canberra",
        negative_completion=" Sydney",
    )
    assert pair.read_position == -1
    assert pair.positive_text == "Q: capital of Australia?\nA: Canberra"
    assert pair.negative_text == "Q: capital of Australia?\nA: Sydney"


def test_pair_rejects_empty_prompt_or_completion():
    with pytest.raises(ValueError):
        ContrastivePair(prompt="  ", positive_completion="a", negative_completion="b")
    with pytest.raises(ValueError):
        ContrastivePair(prompt="p", positive_completion="", negative_completion="b")


def test_pair_rejects_unknown_tier():
    with pytest.raises(ValueError, match="tier"):
        ContrastivePair(
            prompt="p", positive_completion="a", negative_completion="b", tier="trivial"
        )


@pytest.mark.parametrize("goal", sorted(BUILTIN_PAIRS))
def test_builtin_meets_count_and_tier_floors(goal):
    pair_set = load_contrastive_dataset(goal, source="builtin")
    report = pair_set.quality_report()
    assert report.pair_count >= MIN_PAIRS_PER_GOAL
    for tier in TIERS:
        assert report.tier_counts[tier] >= MIN_PAIRS_PER_TIER, (goal, tier)
    assert report.meets_count and report.meets_tier_coverage


@pytest.mark.parametrize("goal", sorted(BUILTIN_PAIRS))
def test_builtin_pairs_are_not_instruction_prefixes(goal):
    """The completions must differ per pair -- an instruction prefix reuses one stem."""
    pair_set = load_contrastive_dataset(goal, source="builtin")
    positives = {pair.positive_completion for pair in pair_set}
    assert len(positives) > 0.8 * len(pair_set), "completions look templated, not behavioural"


def test_builtin_pairs_have_no_exact_duplicates():
    for goal in BUILTIN_PAIRS:
        report = load_contrastive_dataset(goal, source="builtin").quality_report()
        assert report.duplicate_pairs == []


def test_quality_report_flags_identical_completions():
    pair_set = ContrastivePairSet.from_pairs(
        "x",
        [
            ContrastivePair(prompt="p1", positive_completion=" yes", negative_completion=" YES "),
            ContrastivePair(prompt="p2", positive_completion=" a", negative_completion=" b"),
        ],
    )
    report = pair_set.quality_report()
    assert report.duplicate_pairs == [0]
    assert any("identical" in warning for warning in report.warnings)


def test_quality_report_flags_high_embedding_similarity():
    pair_set = ContrastivePairSet.from_pairs(
        "x",
        [
            ContrastivePair(prompt="p1", positive_completion=" one", negative_completion=" two"),
            ContrastivePair(prompt="p2", positive_completion=" three", negative_completion=" four"),
        ],
    )

    def embedder(texts):
        # First pair's completions map to nearly-parallel vectors; second to orthogonal.
        table = {
            "p1 one": [1.0, 0.0],
            "p1 two": [0.999, 0.045],
            "p2 three": [1.0, 0.0],
            "p2 four": [0.0, 1.0],
        }
        return torch.tensor([table[text] for text in texts])

    report = pair_set.quality_report(embedder=embedder)
    flagged = {index for index, _ in report.high_similarity_pairs}
    assert flagged == {0}


def test_from_pairs_deduplicates():
    pair = ContrastivePair(prompt="p", positive_completion=" a", negative_completion=" b")
    pair_set = ContrastivePairSet.from_pairs("x", [pair, pair])
    assert len(pair_set) == 1


def test_register_and_load_custom_pairs():
    register_contrastive_pairs(
        "myprobe",
        [{"prompt": "p", "positive": " a", "negative": " b", "tier": "easy"}],
    )
    assert "myprobe" in available_goals(source="custom")
    loaded = load_contrastive_dataset("myprobe", source="custom")
    assert len(loaded) == 1 and loaded.pairs[0].source == "custom"


def test_load_unknown_builtin_goal_raises():
    with pytest.raises(ValueError, match="unknown goal"):
        load_contrastive_dataset("nope", source="builtin")


def test_instruction_prefix_control_is_available_for_every_goal():
    for goal in BUILTIN_PAIRS:
        control = load_instruction_prefix_control(goal)
        assert len(control) >= 1
        assert control.source == "instruction-prefix-control"


def test_split_shared_prefix_recovers_prompt_and_divergent_tails():
    chosen = "\n\nHuman: hi\n\nAssistant: I can help safely."
    rejected = "\n\nHuman: hi\n\nAssistant: I can help you cause harm."
    prompt, pos, neg = _split_shared_prefix(chosen, rejected)
    assert prompt == "\n\nHuman: hi\n\nAssistant: I can help "
    assert pos == "safely." and neg == "you cause harm."


def test_hf_loader_converts_truthful_qa_rows_without_network():
    rows = [
        {
            "question": "What is the capital of Australia?",
            "mc1_targets": {"choices": ["Canberra", "Sydney", "Melbourne"], "labels": [1, 0, 0]},
        }
    ]

    def fake_load_dataset(name, config=None, split=None):
        assert name == "truthful_qa"
        return rows

    loader = HFContrastiveLoader(load_dataset=fake_load_dataset)
    pair_set = loader.load("truthful", "truthful_qa")
    assert len(pair_set) == 1
    pair = pair_set.pairs[0]
    assert pair.prompt == "Q: What is the capital of Australia?\nA:"
    assert pair.positive_completion == " Canberra"
    assert pair.negative_completion == " Sydney"
    assert pair.tier == "hard"


def test_hf_loader_rejects_unknown_dataset():
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: [])
    with pytest.raises(ValueError, match="no HuggingFace converter"):
        loader.load("truthful", "some_unknown_dataset")
