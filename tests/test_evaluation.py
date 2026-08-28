"""The held-out split, and the promise that reading it changes nothing."""

from dataclasses import replace

import pytest
import torch

from evaluation import (
    HOLDOUT_STRIDE,
    SOURCE_TRAIN_HOLDOUT,
    SOURCE_VALIDATION_SPLIT,
    HeldOutSplit,
    build_held_out_split,
    collect_documents,
    evaluate,
    fingerprint_tokens,
    is_held_out_position,
    is_usable_document,
)
from mob import apply_mob_to_model, frozen_economy, get_mob_layers
from tests.conftest import TRAIN_ROWS, fake_load_dataset


def test_usable_document_rejects_blank_rows():
    assert is_usable_document("some text")
    assert not is_usable_document("   ")
    assert not is_usable_document("")
    assert not is_usable_document(None)


def test_holdout_positions_are_a_fixed_stride():
    assert is_held_out_position(0)
    assert is_held_out_position(HOLDOUT_STRIDE)
    assert not is_held_out_position(1)


def test_holdout_and_training_stream_are_disjoint():
    """The acceptance criterion, checked on the two code paths that must agree.

    The evaluation collector and the training-side filter are different call sites
    in different modules; this asserts they partition the same stream rather than
    that each looks correct on its own.
    """
    held_out = collect_documents(TRAIN_ROWS, "text", count=4, stride=HOLDOUT_STRIDE)
    trained_on = [
        row["text"]
        for index, row in enumerate(TRAIN_ROWS)
        if not is_held_out_position(index) and is_usable_document(row["text"])
    ]

    assert held_out
    assert set(held_out).isdisjoint(trained_on)


def test_blank_rows_do_not_shift_holdout_positions():
    """Dropping blanks before the position test would desynchronise the two sides.

    Every third row here is blank. If the collector counted only usable documents
    while the training filter counted raw rows, the two would disagree and the
    "held-out" documents would sit in the training set -- silently, and only for
    datasets with blank rows.
    """
    held_out = collect_documents(TRAIN_ROWS, "text", count=3, stride=HOLDOUT_STRIDE)

    for document in held_out:
        index = next(i for i, row in enumerate(TRAIN_ROWS) if row["text"] == document)
        assert is_held_out_position(index)


def test_validation_split_is_preferred(fake_tokenizer):
    split = build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )

    assert split.source == SOURCE_VALIDATION_SPLIT
    assert "article-level disjoint" in split.leakage_risk
    assert split.num_sequences == 8


def test_falls_back_to_train_holdout_and_says_so(fake_tokenizer, caplog):
    """A splitless dataset still works, and the weaker guarantee is not silent."""
    with caplog.at_level("WARNING"):
        split = build_held_out_split(
            "splitless", None, fake_tokenizer, 16, fake_load_dataset, num_sequences=4
        )

    assert split.source == SOURCE_TRAIN_HOLDOUT
    assert "may share articles" in split.leakage_risk
    assert "optimistic" in caplog.text


def test_split_is_identical_across_arms(fake_tokenizer):
    first = build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )
    second = build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )

    assert first.fingerprint == second.fingerprint
    assert torch.equal(first.input_ids, second.input_ids)


def test_fingerprint_separates_different_data(fake_tokenizer):
    split = build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )
    altered = split.input_ids.clone()
    altered[0, 0] += 1

    assert fingerprint_tokens(altered) != split.fingerprint


def test_round_trip_through_disk(tmp_path, fake_tokenizer):
    split = build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )
    path = tmp_path / "held_out_split.pt"
    split.save(path)

    restored = HeldOutSplit.load(path)

    assert restored.fingerprint == split.fingerprint
    assert restored.source == split.source
    assert torch.equal(restored.input_ids, split.input_ids)


def test_tampered_cache_is_refused(tmp_path, fake_tokenizer):
    """A cache file outlives the code that wrote it, so it is checked, not trusted."""
    split = build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )
    path = tmp_path / "held_out_split.pt"
    replace(split, fingerprint="0" * 16).save(path)

    with pytest.raises(ValueError, match="does not match its fingerprint"):
        HeldOutSplit.load(path)


def test_empty_document_set_is_an_error(fake_tokenizer):
    with pytest.raises(ValueError, match="No held-out documents"):
        HeldOutSplit.from_documents([], fake_tokenizer, 16, "test", "test")


def _mob_model(tiny_causal_lm, tiny_mob_config):
    return apply_mob_to_model(tiny_causal_lm, tiny_mob_config, layers_to_modify=[1, 2])


def test_evaluate_reports_loss_and_perplexity(tiny_causal_lm, tiny_mob_config, held_out_split):
    model = _mob_model(tiny_causal_lm, tiny_mob_config)

    result = evaluate(model, held_out_split, batch_size=4, device=torch.device("cpu"))

    assert result.num_batches == 2
    assert result.num_tokens > 0
    assert result.loss > 0
    assert result.perplexity == pytest.approx(float(torch.exp(torch.tensor(result.loss))), rel=1e-4)
    assert result.fingerprint == held_out_split.fingerprint
    assert set(result.as_metrics()) == {"eval/loss", "eval/perplexity", "eval/tokens"}


def test_evaluation_moves_no_economic_state(tiny_causal_lm, tiny_mob_config, held_out_split):
    """The acceptance criterion: no wealth updates, no state advanced, nothing lost.

    Usage counts are included deliberately -- they are not wealth, but the inference
    exploration bonus reads them, so an evaluation that advanced them would still be
    a training step wearing a different name.
    """
    model = _mob_model(tiny_causal_lm, tiny_mob_config)
    model.train()

    wealth_before = [mob.expert_wealth.clone() for mob in get_mob_layers(model)]
    usage_before = [mob.expert_usage_count.clone() for mob in get_mob_layers(model)]
    performance_before = [mob.expert_performance_ema.clone() for mob in get_mob_layers(model)]

    evaluate(model, held_out_split, batch_size=4, device=torch.device("cpu"))

    for mob, wealth, usage, performance in zip(
        get_mob_layers(model), wealth_before, usage_before, performance_before, strict=True
    ):
        assert torch.equal(mob.expert_wealth, wealth)
        assert torch.equal(mob.expert_usage_count, usage)
        assert torch.equal(mob.expert_performance_ema, performance)

    assert model.training, "evaluation must restore the trainer's mode"


def test_evaluation_preserves_training_statistics(tiny_causal_lm, tiny_mob_config, held_out_split):
    """An eval between a training step and its log line must not overwrite the line.

    ``_log_training_step`` reads ``last_stats``; without restoration the logged
    routing diagnostics would describe the held-out batches instead of the training
    batch they are printed beside.
    """
    model = _mob_model(tiny_causal_lm, tiny_mob_config)
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))
    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    training_stats = [mob.last_stats for mob in get_mob_layers(model)]

    evaluate(model, held_out_split, batch_size=4, device=torch.device("cpu"))

    for mob, stats in zip(get_mob_layers(model), training_stats, strict=True):
        assert mob.last_stats is stats


def test_wealth_history_records_no_evaluation_rows(tiny_causal_lm, tiny_mob_config, held_out_split):
    model = _mob_model(tiny_causal_lm, tiny_mob_config)
    model.train()
    for mob in get_mob_layers(model):
        mob.start_tracking()

    evaluate(model, held_out_split, batch_size=4, device=torch.device("cpu"))

    for mob in get_mob_layers(model):
        assert mob.get_wealth_history() == []


def test_freeze_is_restored_after_an_exception(tiny_causal_lm, tiny_mob_config):
    model = _mob_model(tiny_causal_lm, tiny_mob_config)

    with pytest.raises(RuntimeError), frozen_economy(model):
        raise RuntimeError("forward blew up")

    for mob in get_mob_layers(model):
        assert not mob._economy_frozen


def test_repeated_evaluation_returns_the_identical_loss(
    tiny_causal_lm, tiny_mob_config, held_out_split
):
    """ "Deterministic routing" in the criterion's own words, stated as a number.

    Bit-identical rather than approximate: anything that made the second pass differ
    -- a wealth update surviving the freeze, dropout left live, a shuffled split --
    would move the loss, and a tolerance would hide exactly the drift being ruled
    out.
    """
    model = _mob_model(tiny_causal_lm, tiny_mob_config)

    first = evaluate(model, held_out_split, batch_size=4, device=torch.device("cpu"))
    second = evaluate(model, held_out_split, batch_size=4, device=torch.device("cpu"))

    assert first.loss == second.loss
    assert first.num_tokens == second.num_tokens

