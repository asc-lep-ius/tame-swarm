"""Determinism configuration for reproducible runs (#13).

Two runs of an identical configuration must produce bitwise-identical loss
traces (enforced by ``tests/test_determinism.py``, marked ``gpu``, in CI). This
module is the one place that seeds every RNG source and toggles the
deterministic-algorithm switches; everything else reads
``TrainingConfig.deterministic`` -- one of ``DETERMINISM_MODES`` -- and calls
``configure_determinism`` once, from ``TAMETrainer.__init__``, before any model
or data touches a device.

Three modes (#31). ``warn`` calls ``torch.use_deterministic_algorithms`` with
``warn_only=True``: a deterministic kernel wherever one exists, and the rest
logged as a known variance source rather than refused. At the ablation
configuration (Qwen3-1.7B, LoRA, 16 converted layers, 512 tokens) the kernel
that gets through is the memory-efficient scaled-dot-product attention backward
-- torch's ``attention_backward.cu`` warns "Memory Efficient attention defaults
to a non-deterministic algorithm" once per run and then accumulates its query
gradient with atomics -- and two runs at identical fingerprints diverge from the
first optimizer step. ``strict`` calls it with ``warn_only=False``: the same
backward takes torch's deterministic path, anything with no deterministic form
at all raises a ``RuntimeError`` naming itself, and ``train/loss`` reproduces
bitwise (``tests/test_determinism.py`` asserts this at the ablation
configuration) for about five percent per step. It is the default from #31 on:
every arm recorded before ran under ``warn``, and the README says so. ``off``
undoes every switch. Which mode an arm ran under is in its fingerprint, so a
strict arm compared against a warn one is declared drift, not read as parity.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

DETERMINISM_OFF = "off"
DETERMINISM_WARN = "warn"
DETERMINISM_STRICT = "strict"
DETERMINISM_MODES = (DETERMINISM_OFF, DETERMINISM_WARN, DETERMINISM_STRICT)
DETERMINISM_DEFAULT = DETERMINISM_STRICT

# Only the once-per-process env var and backend switches are guarded, by the
# mode they were last set for; RNG seeding itself repeats on every call, since a
# comparison harness builds several TAMETrainers in one process
# (compare_routers.py, run_seeds.py) and each one wants its own seed applied.
_configured_mode = DETERMINISM_OFF


def validate_determinism_mode(mode: object) -> str:
    """The mode as one of ``DETERMINISM_MODES``, or a ValueError naming them.

    A bool is refused rather than coerced: the field was a bool before #31, and a
    ``deterministic=True`` that quietly became ``warn`` would be a config that
    reads as strict to nobody and to everybody.
    """
    if isinstance(mode, str) and mode in DETERMINISM_MODES:
        return mode
    raise ValueError(f"deterministic must be one of {DETERMINISM_MODES}, got {mode!r}")


def configure_determinism(seed: int, deterministic: str) -> None:
    """Seed torch/numpy/random/CUDA, and force deterministic kernels per the mode.

    The backend switches below must run before the first CUDA context is
    created: ``CUBLAS_WORKSPACE_CONFIG`` only takes effect if it is set before
    cuBLAS picks its own workspace, and setting it after is silently inert.
    ``TAMETrainer.__init__`` calls this before ``setup()`` touches a device,
    which is the latest point that still works -- and within the function, the
    backend block runs before ``torch.cuda.manual_seed_all``, the first call
    here that can itself touch CUDA.

    A comparison harness builds several ``TAMETrainer``s in one process
    (``compare_routers.py``, ``run_seeds.py``), each with its own
    ``deterministic`` mode in principle, and the fingerprint asserts that mode
    must match across arms. Once the backend switches are latched on for one
    mode, a later call for another has to actually move them -- ``warn`` after
    ``strict`` must let the kernels through again, and ``off`` must turn every
    switch off -- or an arm's own fingerprint would describe a run it did not
    perform, which ``parity.py`` calls out as worse than recording nothing.
    """
    global _configured_mode

    mode = validate_determinism_mode(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode != DETERMINISM_OFF and mode != _configured_mode:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Routes torch's own "no deterministic implementation for X" warnings
        # through logging instead of bare stderr, so a run's known variance
        # sources land in the same log everything else does. Under strict there
        # is nothing to route: an op without a deterministic form raises.
        warn_only = mode == DETERMINISM_WARN
        logging.captureWarnings(True)
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        _configured_mode = mode
        logger.info(
            f"Determinism {mode}: "
            f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}, "
            f"cudnn.deterministic=True, use_deterministic_algorithms(warn_only={warn_only})"
        )
    elif mode == DETERMINISM_OFF and _configured_mode != DETERMINISM_OFF:
        # A true undo, back to torch's own defaults -- not the inverse of the
        # block above. ``benchmark=True`` would trade a different bias (cuDNN's
        # autotuner picking an algorithm based on runtime timing) for the one
        # just removed; only False matches what a process that never called
        # this function would have.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        _configured_mode = DETERMINISM_OFF
        logger.warning(
            "Determinism disabled for this run, undoing the previous run's "
            "backend switches (cudnn, use_deterministic_algorithms) in this "
            "process. CUBLAS_WORKSPACE_CONFIG stays set -- it only affects "
            "cuBLAS's workspace reuse strategy, not this run's own outcome."
        )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)``. Seeds numpy/random inside a worker process.

    Each worker is already given a distinct ``torch.initial_seed()`` by
    ``DataLoader`` (base seed + worker id), but numpy and the stdlib ``random``
    module don't inherit that: left unset, every worker's numpy calls draw from
    whatever state the fork inherited from the parent, identical across workers
    and identical across runs only by accident of process layout.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
