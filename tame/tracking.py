"""MLflow experiment tracking, isolated behind one module (#7).

Every other module reaches MLflow through the five functions below, never
through ``import mlflow`` directly -- so switching backends later touches this
file and nothing else. When ``mlflow`` is not installed (the ``serve``/``chat``
extras don't include it), every function becomes a no-op and training proceeds
exactly as it would with tracking enabled.

Lazy by construction, matching ``MetricSink``: ``init_tracking`` only records
the config to use later. Nothing touches disk -- no tracking URI is set, no
run is opened -- until ``log_provenance`` actually opens one. A trainer built
in a test and never trained does not leave an ``mlruns/`` directory behind it,
the same guarantee ``MetricSink`` gives for ``metrics.jsonl``.

Only ``log_provenance`` may open the pending run; ``log_step`` and
``log_checkpoint`` act only on a run that is already active and are no-ops
otherwise. This is deliberate, not merely lazy: ``TAMETrainer.setup()``
guarantees ``log_provenance`` runs before the training loop ever calls
``log_step`` or ``log_checkpoint``, so tying the lazy start to it means a
module holding an unconsumed pending config -- e.g. a trainer built in a test
that never calls ``setup()`` -- can never be mistaken, by some unrelated later
caller of ``log_step``, for a run that should exist. A ``log_step``-triggered
lazy start could not make that distinction.

One run is tracked at a time, matching how a training process is actually
driven: one arm runs start-to-finish, or a comparison harness runs several
arms sequentially in one process. ``end_tracking`` closes the current run so
the next ``init_tracking`` starts clean.
"""

import dataclasses
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    # MLflow >=3 refuses to open a plain ``file:`` store unless this is set,
    # having put it in "maintenance mode" in favour of a database backend. The
    # issue's own design call is zero infrastructure -- no server, no database
    # to provision -- so this opts back into exactly the local-filesystem
    # tracking that call asked for, rather than switching to sqlite.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("'mlflow' not installed. Experiment tracking disabled (no-op).")

_pending_config: Any | None = None
# Whether *this module* considers a run open -- deliberately not answered by
# asking mlflow, so a failure inside mlflow's own machinery (``start_run``
# raising after ``set_experiment`` succeeded, ``end_run`` raising before it
# manages to pop its internal run stack) can never leave this module's idea of
# "is a run active" out of sync with the guard every log_* function trusts.
_run_open: bool = False


def init_tracking(config: Any) -> None:
    """Record the config a run will start from. Starts nothing yet.

    The actual ``mlflow.start_run()`` -- and the params dump this config
    produces -- is deferred to ``log_provenance``, called once ``setup()`` has
    built the MoB config and the arm fingerprint, so the run opens with
    everything known about it rather than a partial write.
    """
    global _pending_config
    _pending_config = config


def _active() -> bool:
    """Whether a run is already open. Never starts one."""
    return HAS_MLFLOW and _run_open


def _ensure_started_for_provenance() -> bool:
    """Open the pending run if there is one. Only ``log_provenance`` calls this."""
    global _pending_config
    if not HAS_MLFLOW:
        return False
    if _run_open:
        return True
    if _pending_config is None:
        return False
    config, _pending_config = _pending_config, None
    _start_run(config)
    return _run_open


def _start_run(config: Any) -> None:
    """Open the run. Any failure here degrades to "tracking never started"
    rather than aborting the caller: ``_run_open`` only becomes ``True`` on the
    line after a fully successful ``start_run`` + param dump, so every other
    function's no-op guard stays correct regardless of which mlflow call
    failed or how far it got. If mlflow's own run stack disagrees -- e.g. an
    earlier ``end_run`` failure left a run active there that this module has
    already forgotten about -- ``start_run`` itself raises "already active",
    caught here the same as any other failure: no crash, no attempt to log
    into a run this call didn't open.
    """
    global _run_open
    try:
        output_dir = getattr(config, "output_dir", ".")
        if "MLFLOW_TRACKING_URI" not in os.environ:
            # Local filesystem tracking, scoped to this run's own output
            # directory rather than the process's working directory -- a test
            # that merely constructs a trainer in a tmp_path never touches the
            # repo root. A comparison harness running several arms out of
            # several output dirs sets MLFLOW_TRACKING_URI itself to share one
            # store across them; see scripts/compare_routers.py.
            mlflow.set_tracking_uri(f"file:{Path(output_dir) / 'mlruns'}")

        model_id = getattr(config, "model_id", "unknown-model")
        # One experiment per model: this is what makes "cross-run comparison"
        # mean something in the MLflow UI. A fresh experiment per run would
        # isolate every run from its own replicates, which is the opposite of
        # the point.
        mlflow.set_experiment(model_id)

        router = getattr(config, "router", "unknown-arm")
        seed = getattr(config, "seed", "?")
        timestamp = _utc_timestamp()
        mlflow.start_run(run_name=f"{router}-seed{seed}-{timestamp}")

        mlflow.log_params(_flatten_params(dataclasses.asdict(config)))
        _run_open = True
        logger.info(f"MLflow tracking started: experiment={model_id!r}, run={router}-seed{seed}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow run failed to start, continuing without tracking: {exc}")


def log_provenance(mob_config: Any | None = None, fingerprint: Any | None = None) -> None:
    """Log everything needed to say which code and environment produced a run.

    A metric curve without this is not reproducible, it is anecdotal: two runs
    with identical params can still differ in git SHA, torch/CUDA build, or
    driver, and any of those can move a number as much as a hyperparameter
    does. Logged once, from ``TAMETrainer.setup()`` after the MoB config and
    arm fingerprint both exist.

    ``SteeringConfig`` params are not logged here: ``TAMETrainer`` builds one
    only to seed the routing coupling (#14), and everything that seeding depends
    on -- the goal, beta and warmup -- is a ``TrainingConfig`` field, logged with
    the rest of them when the run opens.
    """
    if not _ensure_started_for_provenance():
        return

    params: dict[str, Any] = {}
    if mob_config is not None:
        params.update(_prefixed(dataclasses.asdict(mob_config), "mob"))
    if fingerprint is not None:
        as_dict = fingerprint.as_dict() if hasattr(fingerprint, "as_dict") else fingerprint
        params.update(_prefixed(dict(as_dict), "fingerprint"))

    params.update(_prefixed(_git_provenance(), "git"))
    params.update(_prefixed(_environment_provenance(), "env"))

    try:
        mlflow.log_params(_flatten_params(params))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow provenance logging failed, continuing without it: {exc}")


def log_step(step: int, metrics: dict[str, float]) -> None:
    """Log one measurement group at a training or eval step.

    Non-finite values (NaN, inf) are dropped rather than sent to MLflow, which
    rejects them; the JSONL sink -- ``MetricSink``, which calls this -- is the
    log of record for those and keeps them. A no-op if no run is active yet --
    see the module docstring on why this never opens one itself.
    """
    if not _active():
        return

    finite = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and math.isfinite(value)
    }
    if not finite:
        return
    try:
        mlflow.log_metrics(finite, step=step)
    except Exception as exc:  # noqa: BLE001
        # A metric write failing must not cost the training step that produced
        # it -- this is called from inside the training loop's own logging path.
        logger.warning(f"MLflow metric logging failed at step {step}, continuing: {exc}")


def log_checkpoint(checkpoint_dir: Path) -> None:
    """Log a checkpoint directory as an MLflow artifact. A no-op if no run is active.

    A checkpoint of a multi-billion-parameter model can fail to upload for
    reasons that have nothing to do with training -- a full disk, a network
    blip against a remote store -- and the checkpoint is already safely on
    local disk by the time this is called. That must not cost the run.
    """
    if not _active():
        return
    try:
        mlflow.log_artifacts(str(checkpoint_dir), artifact_path=checkpoint_dir.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow artifact logging failed for {checkpoint_dir}, continuing: {exc}")


def end_tracking() -> None:
    """Close the current run, if tracking ever started one.

    Called from ``TAMETrainer.train()``'s ``finally`` block, after every
    checkpoint and metric for the run is already safely on disk -- a
    store-side failure while ending the run must not raise out of there.
    ``_run_open`` is cleared unconditionally regardless of whether
    ``mlflow.end_run()`` itself succeeded: this module's own idea of "is a run
    active" is what every no-op guard trusts, so it must never depend on
    mlflow's internal run stack having been popped correctly by a call that
    just failed. A next run that finds mlflow's own stack still non-empty (the
    consequence of ``end_run`` failing before it got there) surfaces that as
    ``start_run`` raising "already active", handled by ``_start_run`` the same
    as any other failure to open a run.
    """
    global _pending_config, _run_open
    if HAS_MLFLOW and _run_open:
        try:
            mlflow.end_run()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MLflow end_run failed, continuing: {exc}")
        finally:
            _run_open = False
    _pending_config = None


def _flatten_params(params: dict[str, Any]) -> dict[str, str]:
    """MLflow params are strings; a tuple or nested dict has to become one."""
    return {key: str(value) for key, value in params.items() if value is not None}


def _prefixed(fields: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}.{key}": value for key, value in fields.items()}


def _utc_timestamp() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_provenance() -> dict[str, Any]:
    """SHA, branch and dirty flag. A dirty tree invalidates the SHA as an
    identifier for what produced a run, so it is recorded rather than hidden."""
    sha = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git(["status", "--porcelain"])
    return {
        "sha": sha or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status) if status is not None else "unknown",
    }


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _environment_provenance() -> dict[str, Any]:
    """torch/CUDA/driver/GPU and the deployed image digest, best-effort.

    The image digest is only assigned by a registry after push, so it cannot
    be baked into the image at build time; it is read from an env var that
    deployment tooling is expected to set (e.g. from the registry manifest
    after push). Absent locally, which is the expected case for a dev run.
    """
    import torch

    provenance: dict[str, Any] = {
        "torch_version": torch.__version__,
        "image_digest": os.environ.get("TAME_IMAGE_DIGEST", "unknown"),
    }

    if torch.cuda.is_available():
        provenance["cuda_version"] = torch.version.cuda or "unknown"
        provenance["gpu_model"] = torch.cuda.get_device_name(0)
        try:
            # Private API, absent from stubs: not part of torch's public surface.
            driver_version = torch._C._cuda_getDriverVersion()  # pyright: ignore[reportAttributeAccessIssue] # noqa: SLF001
            provenance["driver_version"] = f"{driver_version // 1000}.{driver_version % 1000 // 10}"
        except (AttributeError, RuntimeError):
            provenance["driver_version"] = "unknown"
    else:
        provenance["cuda_version"] = "none"
        provenance["gpu_model"] = "none"
        provenance["driver_version"] = "none"

    return provenance
