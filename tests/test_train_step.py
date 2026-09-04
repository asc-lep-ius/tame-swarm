"""Trainer-level checks that need a real model round the MoB layers.

Both defects here were invisible to the layer-level tests: one is a property of
what PEFT does to a model after MoB has been applied, the other of what gradient
checkpointing does to a forward pass that carries economic side effects.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from smoke_fixture import build_smoke_fixture  # noqa: E402

from mob import get_mob_layers  # noqa: E402
from train import TAMETrainer, TrainingConfig  # noqa: E402


@pytest.fixture(scope="module")
def smoke_fixture(tmp_path_factory) -> tuple[str, str]:
    return build_smoke_fixture(tmp_path_factory.mktemp("smoke"))


def _config(smoke_fixture: tuple[str, str], output_dir: Path, **overrides) -> TrainingConfig:
    model_id, dataset = smoke_fixture
    settings = dict(
        model_id=model_id,
        output_dir=str(output_dir),
        dataset_name=dataset,
        num_experts=4,
        top_k=2,
        adapter_rank=4,
        mob_layers_start=1,
        mob_layers_end=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=4,
        warmup_steps=1,
        max_seq_length=32,
        eval_steps=100,
        save_steps=100,
        log_frequency=100,
        held_out_sequences=8,
        probe_tokens=64,
        device="cpu",
        dtype="float32",
        gradient_checkpointing=False,
        seed=0,
        deterministic=True,
    )
    settings.update(overrides)
    return TrainingConfig(**settings)


def test_lora_leaves_the_mob_adapters_and_heads_trainable(smoke_fixture, tmp_path):
    """PEFT freezes everything it did not inject, and MoB is injected before PEFT.

    Measured before the fix: under --use_lora every expert adapter and every
    confidence head reported requires_grad=False, so a LoRA run trained the
    attention projections and nothing the economy reads. The shared base FFN is
    the one MoB parameter that should stay frozen under LoRA -- that is the
    memory-constrained intent of the flag.
    """
    pytest.importorskip("peft")
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "lora", use_lora=True))
    trainer.setup()

    mob_layers = get_mob_layers(trainer.model)
    assert mob_layers, "the smoke model must carry MoB layers for this to test anything"
    for mob in mob_layers:
        assert all(p.requires_grad for p in mob.confidence_heads.parameters())
        assert all(p.requires_grad for p in mob.experts.parameters())
        base = [p for name, p in mob.named_parameters() if name.startswith("base_")]
        assert base and not any(p.requires_grad for p in base)

    lora = [p for name, p in trainer.model.named_parameters() if "lora_" in name]
    assert lora and all(p.requires_grad for p in lora), "the attention LoRA must still train"
    attention = [
        p
        for name, p in trainer.model.named_parameters()
        if "q_proj" in name and "lora_" not in name
    ]
    assert attention and not any(p.requires_grad for p in attention)
