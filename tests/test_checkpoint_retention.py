"""Checkpoint retention and best-loss tracking (#7).

Retention keeps first, best (by held-out eval/loss) and final; everything else
beyond ``checkpoint_keep_last`` most-recent checkpoints is pruned. Best-loss
tracking is what gives retention something real to protect -- before #7,
``TAMETrainer.best_loss`` was declared and never updated.
"""

import train
from evaluation import EvalResult
from train import TAMETrainer, TrainingConfig


def _trainer(tmp_path, **overrides):
    base = {"device": "cpu", "output_dir": str(tmp_path)}
    base.update(overrides)
    return TAMETrainer(TrainingConfig(**base))


def _make_checkpoint(tmp_path, step):
    checkpoint_dir = tmp_path / f"checkpoint-{step}"
    checkpoint_dir.mkdir()
    return checkpoint_dir


def _remaining_checkpoints(tmp_path):
    return {p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")}


def test_keeps_first_best_final_and_recent(tmp_path):
    trainer = _trainer(tmp_path, checkpoint_keep_last=2)
    for step in (10, 20, 30, 40, 50):
        _make_checkpoint(tmp_path, step)
    trainer._checkpoint_steps = [10, 20, 30, 40, 50]
    trainer.best_step = 20

    trainer._prune_checkpoints(final=True)

    assert _remaining_checkpoints(tmp_path) == {
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-40",
        "checkpoint-50",
    }
    assert trainer._checkpoint_steps == [10, 20, 40, 50]


def test_zero_keep_last_retains_only_first_best_final(tmp_path):
    trainer = _trainer(tmp_path, checkpoint_keep_last=0)
    for step in (10, 20, 30):
        _make_checkpoint(tmp_path, step)
    trainer._checkpoint_steps = [10, 20, 30]
    trainer.best_step = 20

    trainer._prune_checkpoints(final=True)

    assert _remaining_checkpoints(tmp_path) == {"checkpoint-10", "checkpoint-20", "checkpoint-30"}


def test_best_protects_the_nearest_saved_step_not_an_exact_match(tmp_path):
    """Eval and save cadences are independent, so best_step rarely lands on a
    saved checkpoint exactly -- retention protects the nearest one instead.

    400 also survives here even at ``checkpoint_keep_last=0``: it is the
    checkpoint this very prune call was triggered by, and a 0 budget means "do
    not keep extra older ones", not "delete the one just written".
    """
    trainer = _trainer(tmp_path, checkpoint_keep_last=0)
    for step in (100, 200, 300, 400):
        _make_checkpoint(tmp_path, step)
    trainer._checkpoint_steps = [100, 200, 300, 400]
    trainer.best_step = 260  # nearest saved step is 300

    trainer._prune_checkpoints(final=False)

    assert _remaining_checkpoints(tmp_path) == {
        "checkpoint-100",
        "checkpoint-300",
        "checkpoint-400",
    }


def test_intermediate_checkpoints_are_not_protected_without_final(tmp_path):
    trainer = _trainer(tmp_path, checkpoint_keep_last=1)
    for step in (10, 20, 30):
        _make_checkpoint(tmp_path, step)
    trainer._checkpoint_steps = [10, 20, 30]
    trainer.best_step = None

    trainer._prune_checkpoints(final=False)

    assert _remaining_checkpoints(tmp_path) == {"checkpoint-10", "checkpoint-30"}


def test_zero_keep_last_evicts_the_previous_transient_checkpoint_on_the_next_save(tmp_path):
    """0 protects only the checkpoint each prune call was triggered by, not a
    running window -- the previous transient survivor is gone by the next one."""
    trainer = _trainer(tmp_path, checkpoint_keep_last=0)
    for step in (10, 20):
        _make_checkpoint(tmp_path, step)
    trainer._checkpoint_steps = [10, 20]
    trainer.best_step = None
    trainer._prune_checkpoints(final=False)
    assert _remaining_checkpoints(tmp_path) == {"checkpoint-10", "checkpoint-20"}

    _make_checkpoint(tmp_path, 30)
    trainer._checkpoint_steps.append(30)
    trainer._prune_checkpoints(final=False)

    assert _remaining_checkpoints(tmp_path) == {"checkpoint-10", "checkpoint-30"}


def test_only_permanently_retained_checkpoints_are_archived_to_mlflow(tmp_path, monkeypatch):
    """Archiving every checkpoint would make the MLflow store grow without
    bound regardless of local retention -- only first/best/final should reach
    ``log_checkpoint``, never a transient recent-window checkpoint."""
    archived = []
    monkeypatch.setattr(
        train, "log_checkpoint", lambda checkpoint_dir: archived.append(checkpoint_dir.name)
    )

    trainer = _trainer(tmp_path, checkpoint_keep_last=1)
    for step in (10, 20, 30):
        _make_checkpoint(tmp_path, step)
    trainer._checkpoint_steps = [10, 20, 30]
    trainer.best_step = 20  # exact match: nearest saved step is 20

    trainer._prune_checkpoints(final=False)

    # 10 (first) and 20 (best) are permanent and get archived; 30 only survives
    # on disk as the transient "most recent" -- it must not be archived.
    assert set(archived) == {"checkpoint-10", "checkpoint-20"}


def test_a_permanently_retained_checkpoint_is_archived_only_once(tmp_path, monkeypatch):
    archived = []
    monkeypatch.setattr(
        train, "log_checkpoint", lambda checkpoint_dir: archived.append(checkpoint_dir.name)
    )

    trainer = _trainer(tmp_path, checkpoint_keep_last=1)
    _make_checkpoint(tmp_path, 10)
    trainer._checkpoint_steps = [10]
    trainer.best_step = 10

    trainer._prune_checkpoints(final=False)
    trainer._prune_checkpoints(final=False)  # nothing changed; must not re-archive

    assert archived == ["checkpoint-10"]


def _fake_result(loss: float) -> EvalResult:
    return EvalResult(
        loss=loss, perplexity=2.71**loss, num_tokens=100, num_batches=1, fingerprint="f"
    )


def test_a_new_low_held_out_loss_updates_best_step(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.model = object()
    trainer.tokenizer = object()
    trainer.held_out_split = object()

    results = iter([_fake_result(2.0), _fake_result(1.5), _fake_result(1.8)])
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: next(results))
    monkeypatch.setattr(train, "probe_specialisation", lambda *a, **k: None)

    trainer.evaluate_held_out(10)
    assert trainer.best_loss == 2.0
    assert trainer.best_step == 10

    trainer.evaluate_held_out(20)
    assert trainer.best_loss == 1.5
    assert trainer.best_step == 20

    # Worse than the best seen so far -- must not move it.
    trainer.evaluate_held_out(30)
    assert trainer.best_loss == 1.5
    assert trainer.best_step == 20
