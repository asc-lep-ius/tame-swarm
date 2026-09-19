"""run_seeds.py's replicate floor (#31): the arithmetic and the table, no model.

The replicate itself is a full training run and is exercised by the smoke sweep;
what a unit test can pin is that the floor is the same estimator ``aggregate``
uses across seeds, at n = 2, so ``compare_runs.py`` can pool the two.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_seeds  # noqa: E402
from run_seeds import (  # noqa: E402
    aggregate,
    build_parser,
    format_table,
    measure_replication,
    replication_std,
)

from parity import arm_label  # noqa: E402
from train import TrainingConfig  # noqa: E402


def test_the_floor_is_the_sample_std_of_one_seed_run_twice():
    first = {"eval/loss": 2.7947, "spec/report_decisiveness": 0.923, "only/first": 1.0}
    second = {"eval/loss": 2.7965, "spec/report_decisiveness": 0.923, "only/second": 2.0}

    floor = replication_std(first, second)

    assert floor == {
        "eval/loss": math.sqrt(sum((v - 2.7956) ** 2 for v in (2.7947, 2.7965)) / 1),
        "spec/report_decisiveness": 0.0,
    }
    # The same estimator aggregate() applies across seeds, at n = 2.
    assert floor["eval/loss"] == aggregate({0: first, 1: second})["eval/loss"]["std"]


def test_the_table_carries_the_floor_only_when_one_was_measured():
    stats = aggregate({0: {"eval/loss": 2.79}, 1: {"eval/loss": 2.81}, 2: {"eval/loss": 2.80}})

    without = format_table(stats)
    with_floor = format_table(stats, {"eval/loss": 0.0013})
    unmeasured_metric = format_table(stats, {})

    assert "repl_std" not in without
    assert "repl_std" in with_floor.splitlines()[0]
    assert with_floor.splitlines()[2].endswith("0.00130")
    assert unmeasured_metric.splitlines()[2].endswith("n/a")


def test_a_replicate_that_fails_leaves_the_floor_unmeasured_not_the_sweep_lost(monkeypatch):
    """The replicate is the (N+1)th trainer in the process; if it dies, the N
    seeds already measured still get their summary."""

    def dies(seed, config, replicate=False):
        raise RuntimeError("Cannot copy out of meta tensor")

    monkeypatch.setattr(run_seeds, "run_seed", dies)

    metrics, floor, error = measure_replication(0, TrainingConfig(), {"eval/loss": 2.79}, {"a": 1})

    assert (metrics, floor) == (None, None)
    assert error == "the replicate of seed 0 failed: Cannot copy out of meta tensor"


def test_a_replicate_that_is_a_different_arm_is_not_a_floor(monkeypatch):
    """A commit during the sweep changes the SHA the replicate fingerprints under;
    the pair's spread is then code and arm, and is recorded as no floor at all."""
    monkeypatch.setattr(
        run_seeds, "run_seed", lambda seed, config, replicate=False: ({"eval/loss": 2.81}, {"a": 2})
    )

    metrics, floor, error = measure_replication(0, TrainingConfig(), {"eval/loss": 2.79}, {"a": 1})

    assert metrics == {"eval/loss": 2.81}
    assert floor is None
    assert error is not None and error.startswith("the replicate of seed 0 is not the same arm")


def test_the_floor_is_measured_when_the_replicate_is_the_same_arm(monkeypatch):
    monkeypatch.setattr(
        run_seeds, "run_seed", lambda seed, config, replicate=False: ({"eval/loss": 2.81}, {"a": 1})
    )

    metrics, floor, error = measure_replication(0, TrainingConfig(), {"eval/loss": 2.79}, {"a": 1})

    assert (metrics, error) == ({"eval/loss": 2.81}, None)
    assert floor == replication_std({"eval/loss": 2.79}, {"eval/loss": 2.81})


def test_the_sweep_accepts_the_uniform_draw_so_a_new_arm_can_meet_a_recorded_one():
    """Every recorded group ran under the uniform draw and the draw is in the
    fingerprint (#38), so without this flag no new sweep could be compared."""
    args = build_parser().parse_args(["--exploration_draw", "uniform"])

    assert args.exploration_draw == "uniform"
    assert build_parser().parse_args([]).exploration_draw == "staleness"


def test_the_sweep_takes_the_stakes_dial_and_names_the_arm_by_it():
    """#39's three arms are one flag apart, and the label says which one a summary is."""
    args = build_parser().parse_args(["--persistence_coupling", "decoupled"])

    assert args.persistence_coupling == "decoupled"
    assert build_parser().parse_args([]).persistence_coupling == "value"
    assert arm_label("mob", None, "truthful", "decoupled") == "mob@truthful~decoupled"
    assert arm_label("mob", None, None, "value") == "mob"
