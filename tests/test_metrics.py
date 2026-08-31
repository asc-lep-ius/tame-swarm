"""The metric sink: nothing on disk until there is something to write."""

import json

from metrics import MetricSink


def test_construction_touches_nothing(tmp_path):
    """A constructor that makes a directory leaves litter wherever it is imported.

    The trainer builds its sink in ``__init__``, so an eager ``mkdir`` here created
    an empty ``./tame_checkpoints`` for anything that merely instantiated a
    trainer -- including a packaging tool walking the repository, which is how this
    was found.
    """
    sink = MetricSink(tmp_path / "runs" / "metrics.jsonl")

    assert not (tmp_path / "runs").exists()

    sink.log(1, {"train/loss": 0.5})
    sink.close()

    assert (tmp_path / "runs" / "metrics.jsonl").exists()


def test_records_carry_step_and_run_tags(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricSink(path, run_tags={"router": "mob", "seed": 42}) as sink:
        sink.log(10, {"eval/loss": 1.5})
        sink.log(20, {"eval/loss": 1.25})

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [record["step"] for record in records] == [10, 20]
    assert all(record["router"] == "mob" and record["seed"] == 42 for record in records)
    assert [record["eval/loss"] for record in records] == [1.5, 1.25]


def test_non_finite_values_survive_the_round_trip(tmp_path):
    """A NaN loss is a fact about the run; the log must not disagree with the console."""
    path = tmp_path / "metrics.jsonl"
    with MetricSink(path) as sink:
        sink.log(1, {"train/loss": float("nan")})

    record = json.loads(path.read_text())

    assert record["train/loss"] != record["train/loss"]
