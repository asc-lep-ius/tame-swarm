"""Functional specialisation measures: what experts compute, not what they own."""

from dataclasses import replace

import pytest
import torch

from mob import ROUTER_SOFTMAX, MixtureOfBidders, apply_mob_to_model, get_mob_layers
from specialisation import (
    CATEGORY_NAMES,
    CATEGORY_SPECIAL,
    _ProbeCapture,
    _record_layer_stats,
    expert_output_divergence,
    probe_specialisation,
    report_decisiveness,
    routing_profiles,
    token_categories,
)


def _layer(config, **overrides):
    return MixtureOfBidders(replace(config, **overrides))


def test_upcycled_experts_start_identical(tiny_mob_config):
    """The upcycling guarantee, stated as a measurement.

    LoRA-B is zeroed at construction, so every expert is the shared base FFN and
    divergence must read as zero. This is the property that makes a day-zero
    comparison between arms meaningful, and it is also the sanity check that the
    metric is not manufacturing differences out of noise.
    """
    layer = _layer(tiny_mob_config)

    result = expert_output_divergence(layer, torch.randn(64, tiny_mob_config.hidden_dim))

    assert result.mean_cosine_distance == pytest.approx(0.0, abs=1e-6)
    assert result.mean_relative_l2 == pytest.approx(0.0, abs=1e-6)


def test_divergence_rises_when_experts_differ(tiny_mob_config):
    layer = _layer(tiny_mob_config)
    with torch.no_grad():
        for index, expert in enumerate(layer.experts):
            expert.down_adapter_B.weight.normal_(mean=0.0, std=0.5 * (index + 1))

    result = expert_output_divergence(layer, torch.randn(64, tiny_mob_config.hidden_dim))

    assert result.mean_cosine_distance > 1e-3
    assert result.max_cosine_distance >= result.min_cosine_distance


def test_divergence_needs_two_experts(tiny_mob_config):
    layer = _layer(tiny_mob_config, num_experts=1, top_k=1)

    with pytest.raises(ValueError, match="at least two experts"):
        expert_output_divergence(layer, torch.randn(8, tiny_mob_config.hidden_dim))


def test_token_categories_read_the_word_boundary(fake_tokenizer):
    categories = token_categories(fake_tokenizer, torch.arange(0, 14).view(2, 7))

    assert categories.shape == (2, 7)
    assert set(categories.flatten().tolist()) <= set(range(len(CATEGORY_NAMES)))


def test_uniform_routing_has_no_divergence_from_the_corpus():
    """Every expert seeing the corpus distribution is the null this must return 0 on."""
    categories = torch.tensor([0, 1, 2, 3] * 8)
    # Expert changes every four tokens, category every one, so each expert takes
    # one token of each category -- routing that is blind to the token.
    selected = torch.arange(32).div(4, rounding_mode="floor").remainder(4).view(-1, 1)

    profile = routing_profiles(selected, categories, num_experts=4)

    assert profile.mean_js_from_corpus == pytest.approx(0.0, abs=1e-9)


def test_a_specialised_expert_diverges_from_the_corpus():
    """Expert 0 takes every digit; the measure has to see it."""
    categories = torch.tensor([2, 1, 1, 1] * 8)
    selected = torch.tensor([0 if category == 2 else 1 for category in categories]).view(-1, 1)

    profile = routing_profiles(selected, categories, num_experts=2)

    assert profile.mean_js_from_corpus > 0.1
    assert profile.per_expert_category_share[0, 2] == pytest.approx(1.0)


def test_idle_experts_do_not_contribute_a_divergence_they_never_earned():
    categories = torch.tensor([0, 1] * 8)
    selected = torch.zeros(16, 1, dtype=torch.long)

    profile = routing_profiles(selected, categories, num_experts=4)

    assert profile.expert_token_share[0] == pytest.approx(1.0)
    assert profile.expert_token_share[1:].sum() == pytest.approx(0.0)
    assert profile.mean_js_from_corpus == pytest.approx(0.0, abs=1e-9)


def test_report_decisiveness_is_one_when_the_report_decides():
    confidences = torch.tensor([[[0.1, 0.9, 0.3], [0.7, 0.2, 0.1]]])
    selected = torch.tensor([[[1, 2], [0, 1]]])

    assert report_decisiveness(selected, confidences) == pytest.approx(1.0)


def test_report_decisiveness_falls_when_something_overturns_the_report():
    """The #12 measurement: wealth overturning the report is exactly this shortfall."""
    confidences = torch.tensor([[[0.1, 0.9, 0.3], [0.7, 0.2, 0.1]]])
    selected = torch.tensor([[[1, 2], [2, 1]]])

    assert report_decisiveness(selected, confidences) == pytest.approx(0.5)


def test_probe_reports_every_measure(
    tiny_causal_lm, tiny_mob_config, fake_tokenizer, held_out_split
):
    model = apply_mob_to_model(tiny_causal_lm, tiny_mob_config, layers_to_modify=[1, 2])

    report = probe_specialisation(
        model,
        held_out_split,
        fake_tokenizer,
        torch.device("cpu"),
        batch_size=4,
        probe_tokens=64,
        divergence_tokens=32,
    )

    assert report is not None
    assert report.probe_tokens == 64
    assert 0.0 <= report.report_decisiveness <= 1.0
    assert report.profile.per_expert_category_share.shape == (
        tiny_mob_config.num_experts,
        len(CATEGORY_NAMES),
    )
    assert set(report.as_metrics()) == {
        "spec/expert_cosine_distance",
        "spec/expert_relative_l2",
        "spec/routing_js_from_corpus",
        "spec/routing_kl_from_uniform",
        "spec/report_decisiveness",
        "spec/probe_tokens",
    }


def test_control_arm_is_fully_report_decided(
    tiny_causal_lm, tiny_mob_config, fake_tokenizer, held_out_split
):
    """The softmax control allocates on the report alone, so this reads exactly 1.0.

    That is what makes the number comparable rather than arm-specific: the same
    definition applied to the auction says how often wealth overturns the report.
    """
    model = apply_mob_to_model(
        tiny_causal_lm,
        replace(tiny_mob_config, router=ROUTER_SOFTMAX),
        layers_to_modify=[1, 2],
    )

    report = probe_specialisation(
        model,
        held_out_split,
        fake_tokenizer,
        torch.device("cpu"),
        batch_size=4,
        probe_tokens=64,
        divergence_tokens=32,
    )

    assert report is not None
    assert report.report_decisiveness == pytest.approx(1.0)


def test_probe_warns_below_the_stated_floor(
    tiny_causal_lm, tiny_mob_config, fake_tokenizer, held_out_split, caplog
):
    model = apply_mob_to_model(tiny_causal_lm, tiny_mob_config, layers_to_modify=[1])

    with caplog.at_level("WARNING"):
        probe_specialisation(
            model,
            held_out_split,
            fake_tokenizer,
            torch.device("cpu"),
            batch_size=4,
            probe_tokens=100_000,
            divergence_tokens=32,
        )

    assert "several points of noise" in caplog.text


def test_dense_arm_has_nothing_to_probe(tiny_causal_lm, fake_tokenizer, held_out_split):
    """No MoB layers is the dense arm, not a failure, so this returns None."""
    assert not get_mob_layers(tiny_causal_lm)

    report = probe_specialisation(
        tiny_causal_lm,
        held_out_split,
        fake_tokenizer,
        torch.device("cpu"),
        batch_size=4,
        probe_tokens=64,
    )

    assert report is None


def test_probe_counts_real_tokens_rather_than_padded_positions(
    tiny_causal_lm, tiny_mob_config, fake_tokenizer, padded_held_out_split
):
    """The >=4096-token floor has to be met with tokens, not with pad.

    A pad is not inert: it carries an id, takes a category and is routed like any
    other position, so counting it both inflates the reported probe size and drags
    every statistic towards whatever the gate does with a constant input.
    """
    split = padded_held_out_split
    padded_positions = split.input_ids.numel()
    assert split.num_tokens < padded_positions, "fixture must actually contain padding"

    model = apply_mob_to_model(tiny_causal_lm, tiny_mob_config, layers_to_modify=[1, 2])

    report = probe_specialisation(
        model,
        split,
        fake_tokenizer,
        torch.device("cpu"),
        batch_size=4,
        probe_tokens=padded_positions,
        divergence_tokens=32,
    )

    assert report is not None
    assert report.probe_tokens == split.num_tokens


def test_probe_never_profiles_a_pad_token(
    tiny_causal_lm, tiny_mob_config, fake_tokenizer, padded_held_out_split
):
    """Pads are the only special tokens this fixture produces, so their share is 0.

    Were they scored they would dominate: they are the single most common id in the
    split, and the corpus marginal is the null a specialisation claim is measured
    against.
    """
    model = apply_mob_to_model(tiny_causal_lm, tiny_mob_config, layers_to_modify=[1, 2])

    report = probe_specialisation(
        model,
        padded_held_out_split,
        fake_tokenizer,
        torch.device("cpu"),
        batch_size=4,
        probe_tokens=padded_held_out_split.input_ids.numel(),
        divergence_tokens=32,
    )

    assert report is not None
    assert report.profile.corpus_category_share[CATEGORY_SPECIAL].item() == pytest.approx(0.0)
    assert report.profile.per_expert_category_share[:, CATEGORY_SPECIAL].sum().item() == (
        pytest.approx(0.0)
    )


class _FakeStats:
    def __init__(self, selected_experts, confidences):
        self.selected_experts = selected_experts
        self.confidences = confidences


class _FakeLayer:
    def __init__(self, stats):
        self.last_stats = stats


def test_recorded_winners_are_exactly_the_unpadded_positions():
    """Pins the mask against the rows it indexes, which the report-level tests cannot.

    Each winner is set to its own flat position, so a mask applied to the wrong axis
    or after the wrong reshape surfaces as the wrong positions surviving rather than
    merely the wrong count -- and a count-only assertion would pass either way.
    """
    positions = torch.arange(12).reshape(2, 6, 1)
    layer = _FakeLayer(_FakeStats(positions, positions.to(torch.float32).repeat(1, 1, 3)))
    keep = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.bool)

    capture = _ProbeCapture.empty([layer])
    _record_layer_stats([layer], keep, wanted=6, capture=capture)

    recorded = capture.selected_per_layer[0][0]
    assert recorded[:, 0].tolist() == [0, 2, 4, 7, 9, 11]
    assert capture.confidences_per_layer[0][0][:, 0].tolist() == [0, 2, 4, 7, 9, 11]


def test_wanted_truncates_after_the_mask_not_before():
    """``wanted`` caps the probe at its token budget; it must not reintroduce pads."""
    positions = torch.arange(12).reshape(2, 6, 1)
    layer = _FakeLayer(_FakeStats(positions, positions.to(torch.float32).repeat(1, 1, 3)))
    keep = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.bool)

    capture = _ProbeCapture.empty([layer])
    _record_layer_stats([layer], keep, wanted=3, capture=capture)

    assert capture.selected_per_layer[0][0][:, 0].tolist() == [0, 2, 4]


def test_a_layer_without_statistics_raises_rather_than_misaligning():
    """Skipping the layer would advance the shared categories and not its winners."""
    capture = _ProbeCapture.empty([_FakeLayer(None)])

    with pytest.raises(RuntimeError, match="recorded no statistics"):
        _record_layer_stats([_FakeLayer(None)], torch.ones(4, dtype=torch.bool), 4, capture)
