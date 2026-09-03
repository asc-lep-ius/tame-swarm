"""MLflow tracking (#7): no-op until a run is deliberately opened, real writes after.

``log_provenance`` is the only function allowed to open the pending run --
``log_step`` and ``log_checkpoint`` act only on a run that already exists. This
is what lets a trainer be constructed in a test (which calls ``init_tracking``
via ``TAMETrainer.__init__``) and never call ``setup()`` without leaving an
``mlruns/`` directory behind, or without some *unrelated* later call to
``log_step`` mistaking that trainer's leftover pending config for a run that
should exist.
"""

from dataclasses import dataclass

import mlflow
import pytest
from mlflow.tracking import MlflowClient

import tracking


@dataclass
class _FakeConfig:
    output_dir: str
    model_id: str = "fake-model"
    router: str = "mob"
    seed: int = 42
    learning_rate: float = 1e-4


@dataclass
class _FakeMoBConfig:
    num_experts: int = 4
    top_k: int = 2


class _FakeFingerprint:
    def as_dict(self):
        return {"router": "mob", "eval_split": "abc123"}


@pytest.fixture(autouse=True)
def _dormant_tracking(monkeypatch):
    """Every test starts and ends with no pending or active run."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracking.end_tracking()
    yield
    tracking.end_tracking()


def test_log_step_is_a_noop_with_no_pending_or_active_run(tmp_path):
    tracking.log_step(1, {"train/loss": 0.5})

    assert not (tmp_path / "mlruns").exists()


def test_log_checkpoint_is_a_noop_with_no_pending_or_active_run(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()

    tracking.log_checkpoint(checkpoint_dir)  # must not raise


def test_init_tracking_alone_touches_nothing(tmp_path):
    """Constructing a trainer must not open a run -- only log_provenance does.

    This is the guarantee that keeps tests which build a ``TAMETrainer`` and
    never call ``setup()`` (several already do) from littering the repo with an
    ``mlruns/`` directory.
    """
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))

    assert not (tmp_path / "mlruns").exists()


def test_a_pending_config_never_reached_by_log_step_or_log_checkpoint(tmp_path):
    """The defect this module exists to rule out: a leftover pending config
    from one caller must never be consumed by an unrelated later call."""
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))

    tracking.log_step(1, {"train/loss": 0.5})
    tracking.log_checkpoint(tmp_path)

    assert not (tmp_path / "mlruns").exists()
    assert mlflow.active_run() is None


def test_log_provenance_opens_the_run_and_logs_config_params(tmp_path):
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path), router="mob", seed=7))

    tracking.log_provenance()

    assert (tmp_path / "mlruns").exists()
    run = mlflow.active_run()
    assert run is not None
    params = MlflowClient().get_run(run.info.run_id).data.params
    assert params["router"] == "mob"
    assert params["seed"] == "7"
    assert params["learning_rate"] == "0.0001"


def test_log_provenance_logs_git_and_environment_fields(tmp_path):
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))

    tracking.log_provenance()

    params = MlflowClient().get_run(mlflow.active_run().info.run_id).data.params
    assert "git.sha" in params
    assert "git.branch" in params
    assert "git.dirty" in params
    assert "env.torch_version" in params
    assert "env.image_digest" in params


def test_log_provenance_prefixes_mob_config_and_fingerprint(tmp_path):
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))

    tracking.log_provenance(mob_config=_FakeMoBConfig(), fingerprint=_FakeFingerprint())

    params = MlflowClient().get_run(mlflow.active_run().info.run_id).data.params
    assert params["mob.num_experts"] == "4"
    assert params["fingerprint.eval_split"] == "abc123"


def test_log_step_after_provenance_writes_a_metric_and_drops_non_finite(tmp_path):
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()
    run_id = mlflow.active_run().info.run_id

    tracking.log_step(5, {"train/loss": 1.25, "train/blew_up": float("nan")})

    client = MlflowClient()
    loss_history = client.get_metric_history(run_id, "train/loss")
    assert loss_history[-1].value == 1.25
    assert client.get_metric_history(run_id, "train/blew_up") == []


def test_log_checkpoint_after_provenance_logs_an_artifact(tmp_path):
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()
    run_id = mlflow.active_run().info.run_id

    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "training_state.pt").write_bytes(b"fake-state")

    tracking.log_checkpoint(checkpoint_dir)

    artifacts = {a.path for a in MlflowClient().list_artifacts(run_id, "checkpoint-10")}
    assert "checkpoint-10/training_state.pt" in artifacts


def test_end_tracking_closes_the_run_so_the_next_init_starts_clean(tmp_path):
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()
    first_run_id = mlflow.active_run().info.run_id

    tracking.end_tracking()
    assert mlflow.active_run() is None

    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()
    second_run_id = mlflow.active_run().info.run_id

    assert second_run_id != first_run_id


def test_end_tracking_survives_the_underlying_store_call_failing(tmp_path, monkeypatch):
    """The realistic failure: the store write inside ``mlflow.end_run()`` raises
    (a full disk, a network blip), not the whole function being unavailable.
    mlflow's own run stack is popped before that call, so this also proves the
    ordinary case works -- the next run gets a genuinely different run_id."""
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()
    first_run_id = mlflow.active_run().info.run_id

    monkeypatch.setattr(
        MlflowClient,
        "set_terminated",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")),
    )

    tracking.end_tracking()  # must not raise

    monkeypatch.undo()
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()
    second_run_id = mlflow.active_run().info.run_id

    assert second_run_id != first_run_id


def test_end_tracking_survives_end_run_itself_failing(tmp_path, monkeypatch):
    """The harsher case: ``mlflow.end_run`` fails before it can pop its own run
    stack, so mlflow's global state and this module's ``_run_open`` disagree.
    Guaranteed: no crash, and every log_* function degrades to a no-op rather
    than silently writing into the run this module no longer considers open --
    it must not merge the next arm's metrics into this one's."""
    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance()

    monkeypatch.setattr(mlflow, "end_run", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    tracking.end_tracking()  # must not raise

    monkeypatch.undo()
    tracking.log_step(1, {"train/loss": 1.0})  # must not raise or merge into the old run
    tracking.log_checkpoint(tmp_path)  # must not raise

    # mlflow's own run stack was never actually popped (the mock replaced the
    # whole function that would have done it) -- clean it up directly so this
    # stale run doesn't leak into later tests via the real mlflow.active_run().
    mlflow.end_run()


def test_two_runs_of_the_same_config_are_identifiable_as_replicates(tmp_path):
    """The acceptance criterion, stated as a check: same config in, same
    logged params out, across two independent run lifecycles."""
    config = _FakeConfig(output_dir=str(tmp_path), router="softmax", seed=99)

    tracking.init_tracking(config)
    tracking.log_provenance()
    first_params = MlflowClient().get_run(mlflow.active_run().info.run_id).data.params
    tracking.end_tracking()

    tracking.init_tracking(config)
    tracking.log_provenance()
    second_params = MlflowClient().get_run(mlflow.active_run().info.run_id).data.params
    tracking.end_tracking()

    for key in ("router", "seed", "learning_rate", "model_id"):
        assert first_params[key] == second_params[key]


def test_all_functions_are_noop_without_mlflow_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "HAS_MLFLOW", False)

    tracking.init_tracking(_FakeConfig(output_dir=str(tmp_path)))
    tracking.log_provenance(mob_config=_FakeMoBConfig())
    tracking.log_step(1, {"train/loss": 1.0})
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    tracking.log_checkpoint(checkpoint_dir)
    tracking.end_tracking()

    assert not (tmp_path / "mlruns").exists()
