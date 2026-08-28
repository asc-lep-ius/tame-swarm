"""Parity between arms, asserted programmatically rather than assumed."""

from dataclasses import fields, replace

import pytest
import torch

from parity import (
    ArmFingerprint,
    ParityError,
    assert_parity,
    data_order_fingerprint,
    fingerprint_arm,
    unchecked_config_fields,
)
from train import TrainingConfig

BASE = ArmFingerprint(
    router="mob",
    seed=42,
    model_id="tiny",
    dtype="float32",
    dataset="wikitext/wikitext-2-raw-v1",
    max_steps=400,
    batch_size=2,
    gradient_accumulation_steps=8,
    max_seq_length=512,
    learning_rate=2e-5,
    warmup_steps=40,
    weight_decay=0.01,
    num_experts=4,
    top_k=2,
    adapter_rank=32,
    requested_layers=(5, 6, 7),
    use_lora=False,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    calibration_loss_weight=0.15,
    wealth_update_frequency=1,
    gradient_checkpointing=True,
    device="cpu",
    probe_tokens=4096,
    eval_split="abc123",
    data_order="def456",
    converted_layers=3,
)


def _batches(seed: int, count: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, 64, (2, 8), generator=generator)} for _ in range(count)]


def test_identical_streams_agree():
    assert data_order_fingerprint(iter(_batches(0))) == data_order_fingerprint(iter(_batches(0)))


def test_different_data_is_caught():
    """The reason the fingerprint hashes tokens and not the dataset name."""
    assert data_order_fingerprint(iter(_batches(0))) != data_order_fingerprint(iter(_batches(1)))


def test_reordered_data_is_caught():
    batches = _batches(0)
    assert data_order_fingerprint(iter(batches)) != data_order_fingerprint(iter(batches[::-1]))


def test_empty_stream_is_an_error():
    with pytest.raises(ValueError, match="No batches"):
        data_order_fingerprint(iter([]))


def test_three_arms_at_parity_pass():
    arms = [
        BASE,
        replace(BASE, router="softmax"),
        replace(BASE, router="dense", converted_layers=0),
    ]

    assert_parity(arms)


def test_dense_arm_may_convert_nothing():
    """``converted_layers`` differs by construction and must not fail the check."""
    assert_parity([BASE, replace(BASE, router="dense", converted_layers=0)])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 43),
        ("max_steps", 401),
        ("adapter_rank", 16),
        ("requested_layers", (5, 6)),
        ("eval_split", "different"),
        ("data_order", "different"),
        ("learning_rate", 3e-5),
        ("batch_size", 4),
    ],
)
def test_any_other_difference_is_a_confound(field, value):
    arms = [BASE, replace(BASE, router="softmax", **{field: value})]

    with pytest.raises(ParityError, match=field):
        assert_parity(arms)


def test_all_disagreements_are_reported_at_once():
    """One run per confound is an afternoon; the message carries every difference."""
    arms = [BASE, replace(BASE, router="softmax", seed=1, adapter_rank=8, max_steps=1)]

    with pytest.raises(ParityError) as error:
        assert_parity(arms)

    message = str(error.value)
    assert "seed" in message
    assert "adapter_rank" in message
    assert "max_steps" in message


def test_duplicate_routers_are_rejected():
    """Two arms with the same gate are not a comparison, whatever else matches."""
    with pytest.raises(ParityError, match="distinct routers"):
        assert_parity([BASE, BASE])


def test_a_single_arm_is_vacuously_at_parity():
    assert_parity([BASE])


def test_fingerprint_arm_reads_the_training_config():
    """The map from config to fingerprint is where a parity check goes vacuous.

    Every value is distinct, so a transposition -- ``num_experts`` read into
    ``top_k``, say -- fails here rather than passing every comparison while the arms
    it certifies differ.
    """
    config = TrainingConfig(
        router="softmax",
        seed=7,
        model_id="tiny-model",
        dtype="float32",
        dataset_name="wikitext",
        max_steps=11,
        batch_size=3,
        gradient_accumulation_steps=5,
        max_seq_length=64,
        learning_rate=1.5e-4,
        warmup_steps=2,
        weight_decay=0.02,
        num_experts=6,
        top_k=4,
        adapter_rank=9,
        mob_layers_start=5,
        mob_layers_end=8,
        use_lora=True,
        lora_rank=13,
        lora_alpha=17,
        lora_dropout=0.11,
        calibration_loss_weight=0.23,
        wealth_update_frequency=19,
        gradient_checkpointing=False,
        device="cpu",
        probe_tokens=8192,
    )

    arm = fingerprint_arm(
        config,
        eval_split_fingerprint="split-hash",
        data_order="order-hash",
        converted_layers=3,
        dataset_config="wikitext-2-raw-v1",
    )

    assert arm.router == "softmax"
    assert arm.seed == 7
    assert arm.model_id == "tiny-model"
    assert arm.dtype == "float32"
    assert arm.dataset == "wikitext/wikitext-2-raw-v1"
    assert arm.max_steps == 11
    assert arm.batch_size == 3
    assert arm.gradient_accumulation_steps == 5
    assert arm.max_seq_length == 64
    assert arm.learning_rate == 1.5e-4
    assert arm.warmup_steps == 2
    assert arm.weight_decay == 0.02
    assert arm.num_experts == 6
    assert arm.top_k == 4
    assert arm.adapter_rank == 9
    assert arm.requested_layers == (5, 6, 7)
    assert arm.use_lora is True
    assert arm.lora_rank == 13
    assert arm.lora_alpha == 17
    assert arm.lora_dropout == 0.11
    assert arm.calibration_loss_weight == 0.23
    assert arm.wealth_update_frequency == 19
    assert arm.gradient_checkpointing is False
    assert arm.device == "cpu"
    assert arm.probe_tokens == 8192
    assert arm.eval_split == "split-hash"
    assert arm.data_order == "order-hash"
    assert arm.converted_layers == 3


def test_dataset_config_is_omitted_when_the_dataset_has_none():
    arm = fingerprint_arm(
        TrainingConfig(dataset_name="openwebtext"),
        eval_split_fingerprint="s",
        data_order="d",
        converted_layers=0,
    )
    assert arm.dataset == "openwebtext"


def test_every_training_config_field_is_fingerprinted_or_declared():
    """A field added to TrainingConfig later must not silently become a confound.

    Parity is only as strong as its field list, and the failure mode of a missing
    field is a comparison that passes while the arms differ -- the same class of
    defect as a config field nothing reads.
    """
    unchecked = unchecked_config_fields(field.name for field in fields(TrainingConfig))
    assert unchecked == (), (
        f"TrainingConfig fields {unchecked} are neither in ArmFingerprint nor declared "
        "in parity.NOT_A_CONFOUND; add them to whichever is right and say why"
    )


def test_lora_settings_break_parity():
    """A different trainable-parameter budget is a confound, not a detail."""
    with pytest.raises(ParityError, match="lora_rank"):
        assert_parity([BASE, replace(BASE, router="softmax", lora_rank=64)])


def test_a_different_objective_breaks_parity():
    with pytest.raises(ParityError, match="calibration_loss_weight"):
        assert_parity([BASE, replace(BASE, router="softmax", calibration_loss_weight=0.9)])
