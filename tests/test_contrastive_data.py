"""The pair format, the built-in content, loaders, and the quality checks."""

import pytest
import torch

from contrastive_data import (
    CERTIFIED,
    COMPLETION_FORMAT,
    MC_ANSWER_CUE,
    MIN_PAIRS_PER_GOAL,
    MIN_PAIRS_PER_TIER,
    MULTIPLE_CHOICE_FORMAT,
    ContrastivePair,
    ContrastivePairSet,
    available_goals,
    clear_custom_pairs,
    letter_counts,
    load_contrastive_dataset,
    load_instruction_prefix_control,
    register_contrastive_pairs,
    resolve_pair_format,
    to_multiple_choice,
)
from contrastive_sources import HFContrastiveLoader, _split_shared_prefix
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
    pair_set = load_contrastive_dataset(goal, source="builtin", pair_format=COMPLETION_FORMAT)
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
    assert prompt == "\n\nHuman: hi\n\nAssistant: I can help"
    assert pos == " safely." and neg == " you cause harm."


def test_split_shared_prefix_backs_up_to_a_word_boundary():
    """'Russia' vs 'Romania' share 'R'; the tails must not start mid-word."""
    prompt, pos, neg = _split_shared_prefix("X is in Russia.", "X is in Romania.")
    assert prompt == "X is in"
    assert pos == " Russia." and neg == " Romania."


def test_identical_rows_yield_empty_tails_and_are_dropped_by_the_converters():
    prompt, pos, neg = _split_shared_prefix("same text here", "same text here")
    assert prompt == "same text here" and pos == "" and neg == ""

    rows = [
        {
            "chosen": "\n\nHuman: hi\n\nAssistant: help",
            "rejected": "\n\nHuman: hi\n\nAssistant: help",
        }
    ]
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: rows)
    with pytest.raises(ValueError, match="no pairs"):
        loader.load("safe", "Anthropic/hh-rlhf")

    statements = [
        {"statement": "The city of X is in Y.", "label": 1, "city": "X"},
        {"statement": "The city of X is in Y.", "label": 0, "city": "X"},
    ]
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: statements)
    with pytest.raises(ValueError, match="no pairs"):
        loader.load("truthful", "geometry_of_truth/cities")


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


def test_register_rejects_records_missing_required_keys():
    with pytest.raises(ValueError, match="missing keys"):
        register_contrastive_pairs("probe", [{"prompt": "p", "positive": " a"}])


def test_hf_loader_respects_limit():
    rows = [
        {
            "question": f"Q{i}?",
            "mc1_targets": {"choices": [f"right{i}", f"wrong{i}"], "labels": [1, 0]},
        }
        for i in range(10)
    ]
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: rows)
    assert len(loader.load("truthful", "truthful_qa", limit=3)) == 3


def test_hh_rlhf_caps_the_streaming_split_by_default():
    from contrastive_sources import DEFAULT_HF_STREAM_LIMIT

    def endless_loader(*_a, **_k):
        i = 0
        while True:
            yield {
                "chosen": f"\n\nHuman: hi\n\nAssistant: safe {i}",
                "rejected": f"\n\nHuman: hi\n\nAssistant: harm {i}",
            }
            i += 1

    loader = HFContrastiveLoader(load_dataset=endless_loader)
    pair_set = loader.load("safe", "Anthropic/hh-rlhf")
    assert len(pair_set) == DEFAULT_HF_STREAM_LIMIT


# --- multiple-choice (CAA letter) format -----------------------------------


def _content_pairs(count: int, tier: str = "easy") -> list[ContrastivePair]:
    return [
        ContrastivePair(
            prompt=f"Q: question {i}?\nA:",
            positive_completion=f" right{i}",
            negative_completion=f" wrong{i}",
            tier=tier,
        )
        for i in range(count)
    ]


def test_multiple_choice_moves_answers_into_the_prompt_and_reads_the_letter():
    pair = ContrastivePair(
        prompt="Q: Where is the Great Wall?\nA: It is located in",
        positive_completion=" China.",
        negative_completion=" Japan.",
        tier="medium",
    )
    (mc,) = to_multiple_choice([pair], seed=0)
    assert mc.pair_format == MULTIPLE_CHOICE_FORMAT
    assert mc.read_position == -1
    assert mc.tier == "medium"
    assert mc.prompt.startswith("Q: Where is the Great Wall?\n(")
    assert mc.prompt.endswith(f"\n{MC_ANSWER_CUE}")
    assert "It is located in China." in mc.prompt and "It is located in Japan." in mc.prompt
    correct = mc.correct_letter
    other = "B" if correct == "A" else "A"
    assert mc.positive_completion == f" {correct}" and mc.negative_completion == f" {other}"
    # The correct option sits under the recorded letter.
    assert f"({correct}) It is located in China." in mc.prompt
    assert f"({other}) It is located in Japan." in mc.prompt


def test_multiple_choice_letters_are_balanced_within_every_tier():
    pairs = _content_pairs(20, "easy") + _content_pairs(20, "medium") + _content_pairs(21, "hard")
    mc = to_multiple_choice(pairs, seed=4)
    for tier in TIERS:
        counts = letter_counts(pair for pair in mc if pair.tier == tier)
        assert abs(counts["A"] - counts["B"]) <= 1, (tier, counts)
    assert {pair.correct_letter for pair in mc} == {"A", "B"}


@pytest.mark.parametrize("per_tier", [15, 5, 7])
def test_multiple_choice_odd_tiers_do_not_stack_their_spare_letter(per_tier):
    """Three odd tiers must leave the whole set within one, not three, of balance."""
    from contrastive_data import letter_imbalance

    pairs = [pair for tier in TIERS for pair in _content_pairs(per_tier, tier)]
    mc = to_multiple_choice(pairs, seed=1)
    assert letter_imbalance(mc) <= 1, letter_counts(mc)
    for tier in TIERS:
        assert letter_imbalance(pair for pair in mc if pair.tier == tier) <= 1


def test_multiple_choice_assignment_is_seeded_and_not_a_fixed_alternation():
    pairs = _content_pairs(40)
    first = [pair.correct_letter for pair in to_multiple_choice(pairs, seed=1)]
    again = [pair.correct_letter for pair in to_multiple_choice(pairs, seed=1)]
    other_seed = [pair.correct_letter for pair in to_multiple_choice(pairs, seed=2)]
    assert first == again
    assert first != other_seed
    assert first != ["A", "B"] * 20


def test_multiple_choice_passes_through_pairs_already_converted():
    mc = to_multiple_choice(_content_pairs(4), seed=0)
    assert to_multiple_choice(mc, seed=9) == mc


def test_multiple_choice_of_a_prompt_without_a_role_line_keeps_it_whole():
    bare = ContrastivePair(
        prompt="The city of Lodz is in",
        positive_completion=" Poland.",
        negative_completion=" Peru.",
    )
    (mc,) = to_multiple_choice([bare])
    assert mc.prompt.startswith("The city of Lodz is in\n(")
    assert "(A) Poland." in mc.prompt or "(B) Poland." in mc.prompt


def test_multiple_choice_pair_validates_letters():
    with pytest.raises(ValueError, match="correct_letter"):
        ContrastivePair(
            prompt="p\nAnswer:",
            positive_completion=" A",
            negative_completion=" B",
            pair_format=MULTIPLE_CHOICE_FORMAT,
        )
    with pytest.raises(ValueError, match="correct letter"):
        ContrastivePair(
            prompt="p\nAnswer:",
            positive_completion=" B",
            negative_completion=" A",
            pair_format=MULTIPLE_CHOICE_FORMAT,
            correct_letter="A",
        )
    with pytest.raises(ValueError, match="only meaningful"):
        ContrastivePair(
            prompt="p", positive_completion=" a", negative_completion=" b", correct_letter="A"
        )


def test_quality_report_flags_an_unbalanced_letter_assignment():
    mc = to_multiple_choice(_content_pairs(10), seed=0)
    skewed = [pair for pair in mc if pair.correct_letter == "A"] + [
        pair for pair in mc if pair.correct_letter == "B"
    ][:1]
    report = ContrastivePairSet.from_pairs("x", skewed).quality_report()
    assert not report.letters_balanced
    assert any("unbalanced" in warning for warning in report.warnings)
    balanced = ContrastivePairSet.from_pairs("x", mc).quality_report()
    assert balanced.letters_balanced


def test_default_pair_format_is_multiple_choice_for_the_repaired_goals_only():
    assert resolve_pair_format("truthful") == MULTIPLE_CHOICE_FORMAT
    assert resolve_pair_format("reasoning") == MULTIPLE_CHOICE_FORMAT
    assert resolve_pair_format("safe") == COMPLETION_FORMAT
    assert resolve_pair_format("safe", MULTIPLE_CHOICE_FORMAT) == MULTIPLE_CHOICE_FORMAT
    assert resolve_pair_format("unknown-goal") == COMPLETION_FORMAT
    with pytest.raises(ValueError, match="pair_format"):
        resolve_pair_format("truthful", "essay")


_MC_GOALS = sorted(
    goal for goal, cert in CERTIFIED.items() if cert.pair_format == MULTIPLE_CHOICE_FORMAT
)


@pytest.mark.parametrize("goal", _MC_GOALS)
def test_builtin_default_load_is_balanced_multiple_choice(goal):
    pair_set = load_contrastive_dataset(goal, source="builtin")
    assert pair_set.is_multiple_choice
    assert all(pair.pair_format == MULTIPLE_CHOICE_FORMAT for pair in pair_set)
    report = pair_set.quality_report()
    assert report.ok and report.letters_balanced
    options = {pair.prompt for pair in pair_set}
    assert len(options) == len(pair_set), "every MC prompt carries its own options"


def test_hf_truthful_qa_loads_in_the_letter_format_by_default():
    rows = [
        {
            "question": f"Q{i}?",
            "mc1_targets": {"choices": [f"right{i}", f"wrong{i}"], "labels": [1, 0]},
        }
        for i in range(8)
    ]
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: rows)
    mc = loader.load("truthful", "truthful_qa", pair_format=MULTIPLE_CHOICE_FORMAT)
    assert len(mc) == 8 and mc.is_multiple_choice
    counts = letter_counts(mc)
    assert counts == {"A": 4, "B": 4}
    assert all(pair.tier == "hard" for pair in mc)
    plain = loader.load("truthful", "truthful_qa")
    assert not plain.is_multiple_choice


# --- certified source with fallback -----------------------------------------


def test_certified_load_uses_the_goals_certified_source():
    from contrastive_data import load_certified_dataset

    rows = [
        {
            "question": f"Q{i}?",
            "mc1_targets": {"choices": [f"right{i}", f"wrong{i}"], "labels": [1, 0]},
        }
        for i in range(6)
    ]
    loaded = load_certified_dataset("truthful", load_dataset=lambda *a, **k: rows)
    assert loaded.certified and loaded.fallback_reason is None
    assert loaded.pair_set.source == "truthful_qa"
    assert loaded.pair_set.is_multiple_choice


def test_certified_load_falls_back_to_builtin_when_the_source_is_unavailable(caplog):
    from contrastive_data import load_certified_dataset

    def offline(*_a, **_k):
        raise OSError("no network, no cache")

    with caplog.at_level("WARNING"):
        loaded = load_certified_dataset("truthful", load_dataset=offline)
    assert not loaded.certified
    assert loaded.fallback_reason and "truthful_qa unavailable" in loaded.fallback_reason
    assert loaded.pair_set.source == "builtin"
    assert loaded.pair_set.is_multiple_choice, "fallback keeps the certified format"
    assert "UNCERTIFIED" in caplog.text


def test_certified_load_does_not_swallow_converter_bugs():
    from contrastive_data import load_certified_dataset

    def bad_rows(*_a, **_k):
        return [{"question": "Q?", "mc1_targets": {"choices": ["a"], "labels": [1]}}]

    # A row with no wrong option yields no pairs -> from_pairs raises ValueError,
    # which is a data bug and must surface, not degrade to the built-in set.
    with pytest.raises(ValueError):
        load_certified_dataset("truthful", load_dataset=bad_rows)


def test_certified_load_of_an_uncertified_goal_is_flagged_not_certified(caplog):
    """A goal absent from CERTIFIED (the deliberation proxy) must never read as certified."""
    from contrastive_data import certification_for, load_certified_dataset

    assert certification_for("deliberation") is None
    with caplog.at_level("WARNING"):
        loaded = load_certified_dataset("deliberation")
    assert not loaded.certified
    assert loaded.fallback_reason and "no certified" in loaded.fallback_reason
    assert loaded.pair_set.source == "builtin"
    assert "UNCERTIFIED" in caplog.text


def test_certified_load_falls_back_on_a_datasets_build_error():
    exceptions = pytest.importorskip("datasets.exceptions")

    def failing_build(*_a, **_k):
        raise exceptions.DatasetGenerationError("parquet conversion failed")

    from contrastive_data import load_certified_dataset

    loaded = load_certified_dataset("truthful", load_dataset=failing_build)
    assert not loaded.certified and loaded.pair_set.source == "builtin"


def test_geometry_of_truth_rejects_a_name_that_is_not_a_plain_file_stem():
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: [])
    with pytest.raises(ValueError, match="must match"):
        loader.load("truthful", "geometry_of_truth/../secrets")


def test_certified_load_of_a_builtin_goal_never_falls_back():
    from contrastive_data import load_certified_dataset

    loaded = load_certified_dataset("safe")
    assert loaded.certified and loaded.pair_set.source == "builtin"
    assert not loaded.pair_set.is_multiple_choice


# --- Geometry of Truth -------------------------------------------------------


def test_geometry_of_truth_cities_become_matched_pairs_read_at_the_full_stop():
    from contrastive_sources import GEOMETRY_OF_TRUTH_URL

    rows = [
        {"statement": "The city of Krasnodar is in Russia.", "label": 1, "city": "Krasnodar"},
        {"statement": "The city of Krasnodar is in South Africa.", "label": 0, "city": "Krasnodar"},
        {"statement": "The city of Lodz is in Poland.", "label": 1, "city": "Lodz"},
        {"statement": "The city of Lodz is in Peru.", "label": 0, "city": "Lodz"},
        # Lima has no false partner and must be skipped.
        {"statement": "The city of Lima is in Peru.", "label": 1, "city": "Lima"},
    ]
    seen = {}

    def fake_csv(builder, data_files=None, split=None):
        seen.update(builder=builder, data_files=data_files, split=split)
        return rows

    loader = HFContrastiveLoader(load_dataset=fake_csv)
    pair_set = loader.load("truthful", "geometry_of_truth/cities")
    assert seen == {
        "builder": "csv",
        "data_files": GEOMETRY_OF_TRUTH_URL.format(name="cities"),
        "split": "train",
    }
    assert len(pair_set) == 2
    first = pair_set.pairs[0]
    assert first.prompt == "The city of Krasnodar is in"
    assert first.positive_completion == " Russia."
    assert first.negative_completion == " South Africa."
    assert first.read_position == -1
    assert first.source == "geometry_of_truth/cities"
    assert not pair_set.is_multiple_choice


def test_geometry_of_truth_unpaired_statements_share_a_neutral_prefix():
    from contrastive_sources import GOT_STATEMENT_PREFIX

    rows = [
        {"statement": "The Spanish word 'uno' means 'one'.", "label": 1},
        {"statement": "The Spanish word 'con' means 'to speak'.", "label": 0},
        {"statement": "The Spanish word 'tener' means 'to have'.", "label": 1},
    ]
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: rows)
    pair_set = loader.load("truthful", "geometry_of_truth/sp_en_trans")
    assert len(pair_set) == 1
    pair = pair_set.pairs[0]
    assert pair.prompt == GOT_STATEMENT_PREFIX
    assert pair.positive_completion == " The Spanish word 'uno' means 'one'."
    assert pair.negative_completion == " The Spanish word 'con' means 'to speak'."


def test_geometry_of_truth_respects_limit_and_can_be_lettered():
    rows = []
    for i in range(10):
        rows.append({"statement": f"The city of C{i} is in Right{i}.", "label": 1, "city": f"C{i}"})
        rows.append({"statement": f"The city of C{i} is in Wrong{i}.", "label": 0, "city": f"C{i}"})
    loader = HFContrastiveLoader(load_dataset=lambda *a, **k: rows)
    assert len(loader.load("truthful", "geometry_of_truth/cities", limit=4)) == 4
    mc = loader.load("truthful", "geometry_of_truth/cities", pair_format=MULTIPLE_CHOICE_FORMAT)
    assert mc.is_multiple_choice and mc.quality_report().letters_balanced
