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


@pytest.mark.parametrize("accumulation_steps", [1, 4])
def test_the_loss_gradient_scale_is_the_valid_token_count_times_accumulation(
    smoke_fixture, tmp_path, accumulation_steps, monkeypatch
):
    """The one expression that sets the economy's magnitude, pinned against the batch.

    The trainer backwards ``mean over N valid tokens / accumulation steps``, so
    the gradient the value hook captures is ``N x accumulation`` times smaller
    than the gradient of the summed per-token loss the reward constants were
    derived on. Every layer-level test passes a hand-written scale, so nothing
    else would fail if this drifted -- a silently mis-scaled economy is the defect
    class #15 was opened over.
    """
    import train as train_module

    trainer = TAMETrainer(
        _config(
            smoke_fixture,
            tmp_path / f"scale_{accumulation_steps}",
            gradient_accumulation_steps=accumulation_steps,
        )
    )
    trainer.setup()
    batch = next(iter(trainer.train_dataloader))
    # The smoke corpus is short lines at max_seq_length 32, so the batch carries
    # padding; the scale must count valid *shifted* positions only.
    assert (batch["attention_mask"] == 0).any(), "fixture must carry padding"

    seen: list[float] = []
    original = train_module.update_all_mob_from_loss

    def spy(model, per_token_loss, token_mask=None, loss_gradient_scale=1.0):
        seen.append(loss_gradient_scale)
        original(model, per_token_loss, token_mask, loss_gradient_scale)

    monkeypatch.setattr(train_module, "update_all_mob_from_loss", spy)
    trainer.global_step = 0
    trainer.train_step(batch)

    shift_labels = batch["labels"][..., 1:]
    shift_mask = batch["attention_mask"][..., 1:]
    valid = int(((shift_labels != -100) & (shift_mask == 1)).sum())
    assert seen == [float(valid * accumulation_steps)]


def test_realised_values_do_not_depend_on_the_accumulation_setting(smoke_fixture, tmp_path):
    """The /accumulation in the backward and the x accumulation in the scale cancel exactly."""

    def realised(accumulation_steps: int) -> list[torch.Tensor]:
        trainer = TAMETrainer(
            _config(
                smoke_fixture,
                tmp_path / f"acc_{accumulation_steps}",
                gradient_accumulation_steps=accumulation_steps,
            )
        )
        trainer.setup()
        # Upcycled adapters are zero and contribute nothing; give every expert
        # the same small planted delta in both runs so there is a value to compare.
        generator = torch.Generator().manual_seed(9)
        with torch.no_grad():
            for mob in get_mob_layers(trainer.model):
                for name, param in mob.experts.named_parameters():
                    if name.endswith("_B.weight"):
                        param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
        trainer.global_step = 0
        trainer.train_step(next(iter(trainer.train_dataloader)))
        return [mob.last_realised_values for mob in get_mob_layers(trainer.model)]

    for one, four in zip(realised(1), realised(4), strict=True):
        assert one is not None and four is not None
        assert (one != 0).any(), "fixture experts must contribute, or the check is vacuous"
        assert torch.allclose(one, four, atol=1e-6)


# --- Coupling activation (#14) ---------------------------------------------


def _certify_smoke_goal(monkeypatch, model_id: str, layers: tuple[int, ...], certified=True):
    """Certify a goal for the smoke model and fake its extraction: no network, no HF cache.

    The subspace estimation and the seeding run for real; only the contrastive
    pairs -- which would pull TruthfulQA -- are replaced.
    """
    from smoke_fixture import SMOKE_HIDDEN_DIM

    import train as train_module
    from contrastive_data import CERTIFIED, MULTIPLE_CHOICE_FORMAT, Certification
    from steering import SteeringVector
    from steering_pipeline import SteeringExtraction

    monkeypatch.setitem(
        CERTIFIED,
        "smoke",
        Certification(
            "truthful_qa", MULTIPLE_CHOICE_FORMAT, layers=layers, strength=1.0, model=model_id
        ),
    )

    def fake_extract(model, tokenizer, goal, config):
        requested = list(config.steering_layers)
        generator = torch.Generator().manual_seed(1)
        vectors = {
            layer: SteeringVector(goal, torch.randn(SMOKE_HIDDEN_DIM, generator=generator), layer)
            for layer in requested
        }
        return SteeringExtraction(
            goal=goal,
            vectors=vectors,
            pair_count=4,
            source="truthful_qa" if certified else "builtin",
            layers=requested,
            tier_counts={},
            pair_format=MULTIPLE_CHOICE_FORMAT,
            certified=certified,
            fallback_reason=None if certified else "truthful_qa needs the train extra",
        )

    monkeypatch.setattr(train_module, "extract_steering_vectors", fake_extract)


def test_coupling_goal_seeds_the_certified_mob_layers_and_trains_the_receptor(
    smoke_fixture, tmp_path, monkeypatch
):
    """The flag couples the intersection of the certified layers and the MoB range, and it learns.

    MoB at layers 1-2, certification at 2-3: only layer 2 is coupled. The receptor
    sits in the heads' optimizer group, the training step advances the warmup, and
    one optimizer step moves the receptor off zero through the real path.
    """
    model_id, _ = smoke_fixture
    _certify_smoke_goal(monkeypatch, model_id, layers=(2, 3))
    config = _config(
        smoke_fixture,
        tmp_path / "coupled",
        coupling_goal="smoke",
        coupling_beta=0.3,
        coupling_warmup_steps=2,
    )
    trainer = TAMETrainer(config)
    trainer.setup()

    by_layer = dict(zip([1, 2], get_mob_layers(trainer.model), strict=True))
    assert not hasattr(by_layer[1], "coupling"), "layer 1 is uncertified and must stay uncoupled"
    coupling = by_layer[2].coupling
    assert coupling.config.coupling_beta == 0.3
    assert coupling.config.warmup_steps == 2

    # The scheduler has already rewritten every group's live ``lr``, so the head
    # group is found by a head, not by its rate.
    a_head = by_layer[2].confidence_heads[0].proj.weight
    groups_with = lambda param: [  # noqa: E731
        index
        for index, group in enumerate(trainer.optimizer.param_groups)
        if any(candidate is param for candidate in group["params"])
    ]
    assert groups_with(coupling.detector) == groups_with(a_head)
    assert trainer.optimizer.param_groups[groups_with(a_head)[0]]["weight_decay"] == 0.0

    batch = next(iter(trainer.train_dataloader))
    trainer.global_step = 0
    trainer.train_step(batch)
    assert int(coupling._coupling_step.item()) == 0
    assert coupling.detector.grad is not None
    assert float(coupling.detector.grad.abs().sum()) == 0.0, "no field yet at step 0: beta is 0"
    for step in (1, 2):
        trainer.global_step = step
        trainer.train_step(batch)
        assert int(coupling._coupling_step.item()) == step

    metrics = by_layer[2].last_stats.coupling_metrics
    assert metrics is not None
    assert float(metrics.beta_effective.item()) == pytest.approx(0.3)
    assert float(coupling.detector.grad.abs().sum()) > 0.0, "the value objective must reach it"
    measurements = trainer._coupling_measurements()
    assert measurements["coupling/beta_effective"] == pytest.approx(0.3)
    assert measurements["coupling/detector_norm_mean"] == 0.0, "nothing has stepped yet"


def test_setup_checks_that_the_seeded_state_survived_the_device_move(
    smoke_fixture, tmp_path, monkeypatch
):
    """A coupling that lost its direction on the device move would log as seeded and add nothing.

    The re-dispatch fallback used to reallocate every buffer with ``to_empty`` (#19
    removed it); the guard runs after every device move regardless and compares
    each coupling with the snapshot seeding took.
    """
    import train as train_module

    model_id, _ = smoke_fixture
    _certify_smoke_goal(monkeypatch, model_id, layers=(2,))
    calls: list[int] = []
    guard = train_module.TAMETrainer._assert_seeded_couplings_intact
    monkeypatch.setattr(
        train_module.TAMETrainer,
        "_assert_seeded_couplings_intact",
        lambda self: (calls.append(1), guard(self)),
    )
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "guard", coupling_goal="smoke"))
    trainer.setup()
    assert calls == [1], "setup must run the guard once, after the device move"

    coupling = get_mob_layers(trainer.model)[1].coupling
    kept = coupling.steering_direction.clone()
    with torch.no_grad():
        coupling.steering_direction.zero_()
    with pytest.raises(RuntimeError, match="layer 2 did not survive"):
        guard(trainer)

    with torch.no_grad():
        coupling.steering_direction.copy_(kept)
    guard(trainer)
    with torch.no_grad():
        coupling.detector.uniform_(-0.01, 0.01)
    with pytest.raises(RuntimeError, match="layer 2 did not survive"):
        guard(trainer)


def test_without_a_coupling_goal_routing_stays_uncoupled(smoke_fixture, tmp_path):
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "plain"))
    trainer.setup()
    trainer.global_step = 0
    trainer.train_step(next(iter(trainer.train_dataloader)))

    assert not any(hasattr(mob, "coupling") for mob in get_mob_layers(trainer.model))
    assert trainer._coupling_measurements() == {}


def test_setup_refuses_to_seed_from_a_fallback_vector(smoke_fixture, tmp_path, monkeypatch):
    from steering_pipeline import UncertifiedDirectionError

    model_id, _ = smoke_fixture
    _certify_smoke_goal(monkeypatch, model_id, layers=(2,), certified=False)
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "fallback", coupling_goal="smoke"))

    with pytest.raises(UncertifiedDirectionError, match="uncertified"):
        trainer.setup()


def test_setup_refuses_a_goal_whose_certified_layers_carry_no_mob_layer(
    smoke_fixture, tmp_path, monkeypatch
):
    """Seeding nothing under a coupling flag would be the silent no-op #14 exists to remove."""
    model_id, _ = smoke_fixture
    _certify_smoke_goal(monkeypatch, model_id, layers=(3,))
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "nomob", coupling_goal="smoke"))

    with pytest.raises(RuntimeError, match="seeded nothing"):
        trainer.setup()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"coupling_goal": "deliberation"}, "no certified layers"),
        ({"coupling_goal": "truthful", "model_id": "google/gemma-2-2b-it"}, "certified on"),
        ({"coupling_goal": "truthful", "router": "dense"}, "dense arm"),
        ({"coupling_goal": "truthful", "coupling_beta": 0.0}, "inert"),
        ({"coupling_beta": -0.1}, "coupling_beta"),
    ],
)
def test_an_uncertified_coupling_is_refused_at_construction(overrides, match):
    """The boundary, not minutes into a run with the model loaded."""
    with pytest.raises(ValueError, match=match):
        TrainingConfig(**overrides)


# --- The model that trains is the model that was loaded (#19) ---------------


def test_setup_refuses_a_model_left_on_the_meta_device(smoke_fixture, tmp_path, monkeypatch):
    """A tensor on meta means accelerate offloaded it; refuse, never 'materialise'.

    The path this replaced reallocated every tensor with ``to_empty``, re-initialised
    what happened to read as zero, reloaded the checkpoint keys it could match --
    none of MoB's -- and trained the result.
    """
    import train as train_module

    convert = train_module.TAMETrainer._apply_mob

    def convert_then_offload(self):
        convert(self)
        self.model.lm_head.to_empty(device="meta")

    monkeypatch.setattr(train_module.TAMETrainer, "_apply_mob", convert_then_offload)
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "meta"))

    with pytest.raises(RuntimeError) as refusal:
        trainer.setup()

    assert "lm_head.weight" in str(refusal.value)
    assert "offloaded part of the model" in str(refusal.value)


def test_a_model_loaded_with_an_offloaded_ffn_is_refused_before_conversion(
    smoke_fixture, tmp_path, monkeypatch
):
    """The load-time refusal fires before conversion would fail copying a meta FFN.

    Without it the user would see ``Cannot copy out of meta tensor`` from inside
    ``from_pretrained_ffn`` instead of the refusal and its ways out.
    """
    import train as train_module

    load = train_module.AutoModelForCausalLM.from_pretrained

    def load_then_offload(*args, **kwargs):
        model = load(*args, **kwargs)
        model.model.layers[1].mlp.gate_proj.to_empty(device="meta")
        return model

    monkeypatch.setattr(train_module.AutoModelForCausalLM, "from_pretrained", load_then_offload)
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "meta_ffn"))

    with pytest.raises(RuntimeError) as refusal:
        trainer.setup()

    assert "layers.1.mlp.gate_proj.weight" in str(refusal.value)
    assert "offloaded part of the model" in str(refusal.value)


def _first_adapter_b(mob):
    return next(
        param for name, param in mob.experts.named_parameters() if name.endswith("_B.weight")
    )


PERTURBATIONS = {
    "shared base": (lambda mob: mob.base_gate_proj.weight.add_(1.0), "shared base gate_proj"),
    "ledger": (lambda mob: mob.expert_wealth.fill_(1.0), "ledger expert_wealth"),
    "adapter": (lambda mob: _first_adapter_b(mob).fill_(0.1), "adapter .* is not zero"),
    "confidence bias": (lambda mob: mob.confidence_heads[0].proj.bias.zero_(), "confidence head 0"),
    "nan": (lambda mob: mob.base_up_proj.weight[0, 0].fill_(float("nan")), "NaN or inf"),
}


@pytest.mark.parametrize("what", list(PERTURBATIONS))
def test_setup_post_conditions_fail_when_the_converted_model_is_perturbed(
    smoke_fixture, tmp_path, what
):
    """Each invariant setup asserts has a state in which it fails, named in the error."""
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / what.replace(" ", "_")))
    trainer.setup()  # the unmodified path passes every post-condition
    perturb, expected = PERTURBATIONS[what]

    with torch.no_grad():
        perturb(get_mob_layers(trainer.model)[1])

    with pytest.raises(RuntimeError, match=expected):
        trainer._assert_setup_invariants()


def test_a_converted_layer_nobody_fingerprinted_is_not_trusted(smoke_fixture, tmp_path):
    """A shared base with no fingerprint is one nobody can vouch for: fail, do not skip."""
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "unfingerprinted"))
    trainer.setup()

    trainer._ffn_fingerprints.pop(1)

    with pytest.raises(RuntimeError, match="no FFN fingerprint recorded for layer 1"):
        trainer._assert_setup_invariants()


def test_the_shared_base_check_reads_the_layer_it_was_taken_from(smoke_fixture, tmp_path):
    """Fingerprints are keyed by transformer layer; MoB at layers 1-2 must match 1-2, not 0-1."""
    trainer = TAMETrainer(_config(smoke_fixture, tmp_path / "keyed"))
    trainer.setup()

    assert sorted(trainer._ffn_fingerprints) == [1, 2]
    with torch.no_grad():
        get_mob_layers(trainer.model)[0].base_down_proj.weight.mul_(2.0)
    with pytest.raises(RuntimeError, match="layer 1: shared base down_proj"):
        trainer._assert_setup_invariants()
