"""Determinism (#13): every RNG source seeded from one config field, provably.

``test_two_runs_produce_bitwise_identical_loss_traces`` is the CI gate #13 asks
for: a short training config, run twice, with the train/loss trace compared
step for step. It needs a GPU to be a meaningful check at all -- CPU ops are
already close to deterministic by default, and the nondeterminism this project
is exposed to (cuDNN algorithm selection, cuBLAS workspace reuse, atomic
scatter-adds) is CUDA-specific -- so it is marked ``gpu`` and runs in
``test-gpu``, which after this ships no longer sets ``allow_failure: true``.
"""

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
from determinism import configure_determinism, seed_worker  # noqa: E402
from train import TAMETrainer, TrainingConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_determinism_backend_state():
    """Snapshot and restore every piece of state ``configure_determinism`` touches.

    ``determinism._backend_configured`` and torch's own backend switches
    (``use_deterministic_algorithms``, ``cudnn.deterministic``,
    ``cudnn.benchmark``) have to change together or a later test -- or a later
    trainer in the same process -- reads a flag that no longer describes the
    real backend state. A test that only ``monkeypatch.setattr``s the flag
    leaves this exact desync behind it; restoring all four here, in one place,
    is what keeps every test in this file free to mutate them without policing
    its own cleanup.
    """
    import torch

    flag_before = determinism._backend_configured  # noqa: SLF001
    algos_before = torch.are_deterministic_algorithms_enabled()
    cudnn_det_before = torch.backends.cudnn.deterministic
    cudnn_bench_before = torch.backends.cudnn.benchmark
    yield
    determinism._backend_configured = flag_before  # noqa: SLF001
    torch.use_deterministic_algorithms(algos_before)
    torch.backends.cudnn.deterministic = cudnn_det_before
    torch.backends.cudnn.benchmark = cudnn_bench_before


def test_configure_determinism_reproduces_torch_numpy_and_random_streams():
    import random

    import numpy as np
    import torch

    configure_determinism(seed=123, deterministic=True)
    torch_draw_1 = torch.rand(4)
    numpy_draw_1 = np.random.rand(4)
    random_draw_1 = [random.random() for _ in range(4)]

    configure_determinism(seed=123, deterministic=True)
    torch_draw_2 = torch.rand(4)
    numpy_draw_2 = np.random.rand(4)
    random_draw_2 = [random.random() for _ in range(4)]

    assert torch.equal(torch_draw_1, torch_draw_2)
    assert list(numpy_draw_1) == list(numpy_draw_2)
    assert random_draw_1 == random_draw_2


def test_configure_determinism_with_different_seeds_diverges():
    """The check above can't fail if it always compares equal streams to themselves."""
    import torch

    configure_determinism(seed=1, deterministic=True)
    first = torch.rand(4)

    configure_determinism(seed=2, deterministic=True)
    second = torch.rand(4)

    assert not torch.equal(first, second)


def test_configure_determinism_sets_backend_switches(monkeypatch):
    """The bitwise-identical-trace test only runs on GPU (``test-gpu``); without
    this, deleting the line that sets ``CUBLAS_WORKSPACE_CONFIG`` -- the one the
    module docstring's ordering is built around -- would leave the CPU-only
    ``test`` job green on every merge request.
    """
    import torch

    import determinism

    monkeypatch.setattr(determinism, "_backend_configured", False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_determinism(seed=0, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()


def test_configure_determinism_false_undoes_a_prior_true_in_the_same_process(monkeypatch):
    """A comparison harness builds several trainers in one process (see
    ``compare_routers.py``, ``run_seeds.py``); a later ``deterministic=False``
    trainer must not silently keep running under the earlier one's backend
    switches, or its ``ArmFingerprint.deterministic=False`` would describe a run
    it did not perform.
    """
    import torch

    import determinism

    monkeypatch.setattr(determinism, "_backend_configured", False)
    configure_determinism(seed=0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()

    configure_determinism(seed=0, deterministic=False)

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
        deterministic=True,
    )
    trainer = TAMETrainer(config)
    trainer.setup()
    trainer.train()

    metrics_path = output_dir / "metrics.jsonl"
    losses = []
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        if "train/loss" in record:
            losses.append(record["train/loss"])
    return losses


@pytest.mark.gpu
def test_two_runs_produce_bitwise_identical_loss_traces(tmp_path):
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
