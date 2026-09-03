"""Determinism configuration for reproducible runs (#13).

Two runs of an identical configuration must produce bitwise-identical loss
traces (enforced by ``tests/test_determinism.py``, marked ``gpu``, in CI). This
module is the one place that seeds every RNG source and toggles the
deterministic-algorithm switches; everything else reads
``TrainingConfig.deterministic`` and calls ``configure_determinism`` once, from
``TAMETrainer.__init__``, before any model or data touches a device.

``torch.use_deterministic_algorithms`` is called with ``warn_only=True`` rather
than the strict default: several ops the HF model stack exercises (some
embedding-backward and interpolate paths on CUDA) have no deterministic kernel
at all, and refusing to run is worse than running with a known, logged variance
source. #12 already measured a ~46-point between-seed spread on report
decisiveness; knowing whether spread like that comes from seed variance or
kernel nondeterminism is exactly what changes how many seeds a result needs.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Only the once-per-process env var and backend switches are guarded; RNG
# seeding itself repeats on every call, since a comparison harness builds
# several TAMETrainers in one process (compare_routers.py, run_seeds.py) and
# each one wants its own seed applied.
_backend_configured = False


def configure_determinism(seed: int, deterministic: bool) -> None:
    """Seed torch/numpy/random/CUDA, and force deterministic kernels if asked.

    The backend switches below must run before the first CUDA context is
    created: ``CUBLAS_WORKSPACE_CONFIG`` only takes effect if it is set before
    cuBLAS picks its own workspace, and setting it after is silently inert.
    ``TAMETrainer.__init__`` calls this before ``setup()`` touches a device,
    which is the latest point that still works -- and within the function, the
    backend block runs before ``torch.cuda.manual_seed_all``, the first call
    here that can itself touch CUDA.

    A comparison harness builds several ``TAMETrainer``s in one process
    (``compare_routers.py``, ``run_seeds.py``), each with its own
    ``deterministic`` value in principle, and ``ArmFingerprint.deterministic``
    asserts that value must match across arms. Once the backend switches are
    latched on, a later call with ``deterministic=False`` has to actually turn
    them off -- leaving them on would make an arm's own fingerprint describe a
    run it did not perform, which ``parity.py`` calls out as worse than
    recording nothing at all.
    """
    global _backend_configured

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic and not _backend_configured:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Routes torch's own "no deterministic implementation for X" warnings
        # through logging instead of bare stderr, so a run's known variance
        # sources land in the same log everything else does.
        logging.captureWarnings(True)
        torch.use_deterministic_algorithms(True, warn_only=True)

        _backend_configured = True
        logger.info(
            "Determinism enabled: "
            f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}, "
            "cudnn.deterministic=True, use_deterministic_algorithms(warn_only=True)"
        )
    elif not deterministic and _backend_configured:
        # A true undo, back to torch's own defaults -- not the inverse of the
        # block above. ``benchmark=True`` would trade a different bias (cuDNN's
        # autotuner picking an algorithm based on runtime timing) for the one
        # just removed; only False matches what a process that never called
        # this function would have.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        _backend_configured = False
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
