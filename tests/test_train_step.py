"""Trainer-level checks that need a real model round the MoB layers.

Both defects here were invisible to the layer-level tests: one is a property of
what PEFT does to a model after MoB has been applied, the other of what gradient
checkpointing does to a forward pass that carries economic side effects.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

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


def test_gradient_checkpointing_recompute_leaves_the_economy_untouched(smoke_fixture, tmp_path):
    """The recompute is a pure re-run; only the real forward moves the economy.

    Two defects measured before the fix, both on this fixture. The recompute
    doubled the usage counts -- 128 and 117 of 64 tokens, the second pass cut
    short mid-loop by the checkpoint's early stop. And with the settlement between
    forward and backward, the recompute re-ran the auction on wealth that had
    already moved, picked different winners, and raised a CheckpointError at step 0
    with 8 experts.
    """
    trainer = TAMETrainer(
        _config(smoke_fixture, tmp_path / "ckpt", num_experts=8, gradient_checkpointing=True)
    )
    trainer.setup()
    batch = next(iter(trainer.train_dataloader))
    tokens = batch["input_ids"].numel()

    for step in range(3):
        trainer.global_step = step
        metrics = trainer.train_step(batch)
        assert math.isfinite(metrics["loss"])

    for mob in get_mob_layers(trainer.model):
        assert mob.expert_usage_count.sum().item() == pytest.approx(3 * tokens * mob.config.top_k)
        assert mob.last_value_summary is not None
        assert any(
            head.proj.weight.grad is not None and head.proj.weight.grad.abs().sum() > 0
            for head in mob.confidence_heads
        ), "the value objective must still reach the heads under checkpointing"


def test_checkpointing_does_not_change_what_the_economy_sees(smoke_fixture, tmp_path):
    """Same seed, same batch: with and without checkpointing the step must agree."""

    def one_step(checkpointing: bool) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
        trainer = TAMETrainer(
            _config(
                smoke_fixture,
                tmp_path / f"ckpt_{checkpointing}",
                gradient_checkpointing=checkpointing,
            )
        )
        trainer.setup()
        trainer.global_step = 0
        trainer.train_step(next(iter(trainer.train_dataloader)))
        mobs = get_mob_layers(trainer.model)
        head_grads = [
            torch.cat([h.proj.weight.grad.flatten() for h in mob.confidence_heads]) for mob in mobs
        ]
        usage = [mob.expert_usage_count.clone() for mob in mobs]
        wealth = [mob.expert_wealth.clone() for mob in mobs]
        return head_grads, usage, [w.sum().item() for w in wealth]

    plain_grads, plain_usage, plain_wealth = one_step(False)
    ckpt_grads, ckpt_usage, ckpt_wealth = one_step(True)

    for plain, ckpt in zip(plain_grads, ckpt_grads, strict=True):
        assert torch.allclose(plain, ckpt, atol=1e-6)
    for plain, ckpt in zip(plain_usage, ckpt_usage, strict=True):
        assert torch.equal(plain, ckpt)
    assert plain_wealth == pytest.approx(ckpt_wealth, rel=1e-5)


@pytest.mark.parametrize("router", ["softmax", "mob"])
def test_the_auxiliary_objectives_backward_on_their_own_graph(smoke_fixture, tmp_path, router):
    """The z-loss and value objective are backwarded after the LM backward.

    Under the softmax control arm the LM loss trains the heads through the gate,
    so a z-loss built on the routing reports would share the graph the LM backward
    has already freed -- "backward through the graph a second time". The auxiliary
    objectives read their own pass over the heads instead.
    """
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / router, router=router))
    trainer.setup()
    trainer.global_step = 0

    metrics = trainer.train_step(next(iter(trainer.train_dataloader)))

    assert math.isfinite(metrics["total_loss"])
    for mob in get_mob_layers(trainer.model):
        assert all(
            head.proj.weight.grad is not None and torch.isfinite(head.proj.weight.grad).all()
            for head in mob.confidence_heads
        )


def test_confidence_heads_train_at_their_own_learning_rate(smoke_fixture, tmp_path):
    """Measured on Qwen3-1.7B at the backbone's 2e-5: 120 steps moved no report by 1e-3."""
    trainer = TAMETrainer(
        _config(smoke_fixture, tmp_path / "head_lr", confidence_head_learning_rate=0.0123)
    )
    trainer.setup()

    head_ids = {
        id(p) for mob in get_mob_layers(trainer.model) for p in mob.confidence_heads.parameters()
    }
    assert head_ids
    # The warmup scheduler has already rewritten every group's live ``lr`` for
    # step 0; the configured rate survives as ``initial_lr``.
    head_groups = [g for g in trainer.optimizer.param_groups if g["initial_lr"] == 0.0123]
    assert len(head_groups) == 1
    assert {id(p) for p in head_groups[0]["params"]} == head_ids
    for group in trainer.optimizer.param_groups:
        if group is not head_groups[0]:
            assert not head_ids & {id(p) for p in group["params"]}
