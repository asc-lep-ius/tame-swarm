"""Determinism (#13, #31): every RNG source seeded from one config field, provably.

``test_two_runs_produce_bitwise_identical_loss_traces`` is the CI gate #13 asks
for: a short training config, run twice, with the train/loss trace compared
step for step. It needs a GPU to be a meaningful check at all -- CPU ops are
already close to deterministic by default, and the nondeterminism this project
is exposed to (cuDNN algorithm selection, cuBLAS workspace reuse, atomic
scatter-adds) is CUDA-specific -- so it is marked ``gpu`` and runs in
``test-gpu``, which after this ships no longer sets ``allow_failure: true``.

#31 found that the smoke configuration is not the one the results are quoted
from: at the ablation's own configuration (Qwen3-1.7B, LoRA, sixteen converted
layers, 512 tokens) the memory-efficient attention backward has no deterministic
form under ``warn``, and two runs at identical fingerprints diverge.
``test_strict_mode_reproduces_train_loss_at_the_ablation_configuration`` is the
gate at that configuration, under ``strict``.
"""

import gc
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from smoke_fixture import build_smoke_fixture  # noqa: E402

import determinism  # noqa: E402
from config import MODEL_PROFILES  # noqa: E402
from determinism import configure_determinism, seed_worker  # noqa: E402
from train import TAMETrainer, TrainingConfig  # noqa: E402

# The configuration every result since #13 is quoted from (#25's ablation, #28's
# field-present arms): run_seeds.py's flags in that launch, at ~50 steps.
ABLATION_MODEL_ID = MODEL_PROFILES["qwen3-1.7b"]["model_id"]
ABLATION_STEPS = 50


@pytest.fixture(autouse=True)
def _restore_determinism_backend_state():
    """Snapshot and restore every piece of state ``configure_determinism`` touches.

    ``determinism._configured_mode`` and torch's own backend switches
    (``use_deterministic_algorithms`` and its ``warn_only``,
    ``cudnn.deterministic``, ``cudnn.benchmark``) have to change together or a
    later test -- or a later trainer in the same process -- reads a mode that no
    longer describes the real backend state. A test that only
    ``monkeypatch.setattr``s the latch leaves this exact desync behind it;
    restoring all five here, in one place, is what keeps every test in this
    file free to mutate them without policing its own cleanup.
    """
    import torch

    mode_before = determinism._configured_mode  # noqa: SLF001
    algos_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_det_before = torch.backends.cudnn.deterministic
    cudnn_bench_before = torch.backends.cudnn.benchmark
    yield
    determinism._configured_mode = mode_before  # noqa: SLF001
    torch.use_deterministic_algorithms(algos_before, warn_only=warn_only_before)
    torch.backends.cudnn.deterministic = cudnn_det_before
    torch.backends.cudnn.benchmark = cudnn_bench_before


def test_configure_determinism_reproduces_torch_numpy_and_random_streams():
    import random

    import numpy as np
    import torch

    configure_determinism(seed=123, deterministic="warn")
    torch_draw_1 = torch.rand(4)
    numpy_draw_1 = np.random.rand(4)
    random_draw_1 = [random.random() for _ in range(4)]

    configure_determinism(seed=123, deterministic="warn")
    torch_draw_2 = torch.rand(4)
    numpy_draw_2 = np.random.rand(4)
    random_draw_2 = [random.random() for _ in range(4)]

    assert torch.equal(torch_draw_1, torch_draw_2)
    assert list(numpy_draw_1) == list(numpy_draw_2)
    assert random_draw_1 == random_draw_2


def test_configure_determinism_with_different_seeds_diverges():
    """The check above can't fail if it always compares equal streams to themselves."""
    import torch

    configure_determinism(seed=1, deterministic="warn")
    first = torch.rand(4)

    configure_determinism(seed=2, deterministic="warn")
    second = torch.rand(4)

    assert not torch.equal(first, second)


def test_configure_determinism_sets_backend_switches(monkeypatch):
    """The bitwise-identical-trace test only runs on GPU (``test-gpu``); without
    this, deleting the line that sets ``CUBLAS_WORKSPACE_CONFIG`` -- the one the
    module docstring's ordering is built around -- would leave the CPU-only
    ``test`` job green on every merge request.
    """
    import torch

    monkeypatch.setattr(determinism, "_configured_mode", "off")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_determinism(seed=0, deterministic="warn")

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_strict_refuses_rather_than_warns_and_warn_after_strict_lets_the_kernels_through(
    monkeypatch,
):
    """#31: ``strict`` is ``warn_only=False``, the switch that makes torch's attention
    backward take its deterministic path instead of warning once and using atomics.

    A comparison harness runs several trainers in one process, so the latch has
    to move between the two modes in both directions, not only from off.
    """
    import torch

    monkeypatch.setattr(determinism, "_configured_mode", "off")

    configure_determinism(seed=0, deterministic="strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()

    configure_determinism(seed=0, deterministic="warn")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()

    configure_determinism(seed=0, deterministic="strict")
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_a_mode_outside_the_three_is_refused_and_a_bool_is_not_coerced():
    """The field was a bool before #31; ``True`` must not silently read as ``warn``."""
    with pytest.raises(ValueError, match="deterministic must be one of"):
        configure_determinism(seed=0, deterministic="always")
    with pytest.raises(ValueError, match="deterministic must be one of"):
        configure_determinism(seed=0, deterministic=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deterministic must be one of"):
        TrainingConfig(deterministic=True)  # type: ignore[arg-type]


def test_configure_determinism_off_undoes_a_prior_mode_in_the_same_process(monkeypatch):
    """A comparison harness builds several trainers in one process (see
    ``compare_routers.py``, ``run_seeds.py``); a later ``off`` trainer must not
    silently keep running under the earlier one's backend switches, or its
    ``ArmFingerprint.deterministic=False`` would describe a run it did not
    perform.
    """
    import torch

    monkeypatch.setattr(determinism, "_configured_mode", "off")
    configure_determinism(seed=0, deterministic="strict")
    assert torch.are_deterministic_algorithms_enabled()

    configure_determinism(seed=0, deterministic="off")

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    # A true undo restores torch's own default, not the inverse of the
    # enabled branch -- benchmark=True would trade one bias for another.
    assert torch.backends.cudnn.benchmark is False


def test_seed_worker_is_deterministic_given_the_same_initial_seed():
    import random

    import numpy as np
    import torch

    torch.manual_seed(7)
    seed_worker(worker_id=0)
    numpy_draw_1 = np.random.rand(4)
    random_draw_1 = random.random()

    torch.manual_seed(7)
    seed_worker(worker_id=0)
    numpy_draw_2 = np.random.rand(4)
    random_draw_2 = random.random()

    assert list(numpy_draw_1) == list(numpy_draw_2)
    assert random_draw_1 == random_draw_2


def _train_losses(output_dir: Path) -> list[float]:
    losses = []
    for line in (output_dir / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "train/loss" in record:
            losses.append(record["train/loss"])
    return losses


def _run_smoke_training(output_dir: Path, model_id: str, dataset: str, device: str) -> list[float]:
    """Train the smoke fixture for a handful of steps; return the train/loss trace."""
    config = TrainingConfig(
        model_id=model_id,
        output_dir=str(output_dir),
        dataset_name=dataset,
        num_experts=2,
        adapter_rank=4,
        mob_layers_start=1,
        mob_layers_end=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=8,
        warmup_steps=1,
        max_seq_length=32,
        eval_steps=100,  # past max_steps: this test is about train/loss, not eval
        save_steps=100,
        log_frequency=1,
        held_out_sequences=8,
        probe_tokens=64,
        device=device,
        dtype="float32",
        gradient_checkpointing=False,
        seed=0,
        deterministic="warn",
    )
    trainer = TAMETrainer(config)
    trainer.setup()
    trainer.train()
    return _train_losses(output_dir)


@pytest.mark.gpu
def test_two_runs_produce_bitwise_identical_loss_traces(tmp_path):
    """#13's gate at the smoke configuration, under ``warn``: nothing the smoke
    model exercises lacks a deterministic kernel, so the mode that lets such
    kernels through is already bitwise here. The ablation configuration is not
    -- see the test below."""
    model_id, dataset = build_smoke_fixture(tmp_path / "fixture")

    losses_a = _run_smoke_training(tmp_path / "run_a", model_id, dataset, device="cuda")
    losses_b = _run_smoke_training(tmp_path / "run_b", model_id, dataset, device="cuda")

    # >1 distinct value, not just >0 entries: a trace that is 8 copies of the
    # same number would pass a bitwise-identical check vacuously, without ever
    # exercising whether training actually progressed identically.
    assert len(set(losses_a)) > 1
    assert losses_a == losses_b, (
        "Two runs of an identical configuration diverged with determinism enabled "
        f"-- run_a={losses_a} vs run_b={losses_b}"
    )


def _require_cached(model_id: str) -> None:
    """Skip on a developer's machine without the model; fail under CI, as
    ``test_real_model.py`` does: a skipped gate is a green job that ran nothing."""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(model_id, local_files_only=True)
    except OSError as exc:  # pragma: no cover - depends on the runner's cache
        message = f"{model_id} is not in the local HuggingFace cache: {exc}"
        if os.environ.get("CI"):
            pytest.fail(message + " -- the GPU runner must mount the workstation's cache")
        pytest.skip(message)


def _run_ablation_training(
    output_dir: Path, deterministic: str, steps: int = ABLATION_STEPS
) -> tuple[list[float], dict[str, float]]:
    """#25's arm -- run_seeds.py's flags in that launch -- for ``steps`` micro-steps.

    Every field that shapes the computation is the ablation's; the cadences
    (log every step, one evaluation at the end, one checkpoint) are not part of
    what is computed and are set so the trace is dense and the run short. The
    model is freed and the caching allocator emptied before returning, for the
    reason ``run_seeds.run_seed`` gives: a second full-size trainer in the same
    process otherwise sees less free VRAM than there is and offloads to meta.
    """
    import torch

    config = TrainingConfig(
        model_id=ABLATION_MODEL_ID,
        output_dir=str(output_dir),
        dataset_name="wikitext",
        router="mob",
        num_experts=4,
        adapter_rank=32,
        mob_layers_start=6,
        mob_layers_end=22,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=steps,
        warmup_steps=max(1, steps // 10),
        max_seq_length=512,
        eval_steps=steps,
        save_steps=steps,
        log_frequency=1,
        held_out_sequences=320,
        probe_tokens=4096,
        device="cuda",
        dtype="bfloat16",
        gradient_checkpointing=False,
        use_lora=True,
        seed=0,
        deterministic=deterministic,
    )
    trainer = TAMETrainer(config)
    trainer.setup()
    trainer.train()
    final = dict(trainer.eval_history[-1]) if trainer.eval_history else {}
    losses = _train_losses(output_dir)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return losses, final


@pytest.mark.gpu
def test_strict_mode_reproduces_train_loss_at_the_ablation_configuration(tmp_path):
    """#31's gate: at the configuration the results are quoted from, ``strict``
    makes two runs one trajectory. Under ``warn`` this same pair diverges, and
    #25 measured how far; the README names the kernel."""
    _require_cached(ABLATION_MODEL_ID)

    losses_a, final_a = _run_ablation_training(tmp_path / "run_a", "strict")
    losses_b, final_b = _run_ablation_training(tmp_path / "run_b", "strict")

    assert len(losses_a) >= ABLATION_STEPS // 2
    assert len(set(losses_a)) > 1
    assert losses_a == losses_b, (
        "Two strict runs at the ablation configuration diverged -- "
        f"run_a={losses_a} vs run_b={losses_b}"
    )
    assert final_a["eval/loss"] == final_b["eval/loss"], (
        f"eval/loss differs between strict runs: {final_a['eval/loss']} vs {final_b['eval/loss']}"
    )
