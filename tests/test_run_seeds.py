"""run_seeds.py's replicate floor (#31): the arithmetic and the table, no model.

The replicate itself is a full training run and is exercised by the smoke sweep;
what a unit test can pin is that the floor is the same estimator ``aggregate``
uses across seeds, at n = 2, so ``compare_runs.py`` can pool the two.
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_seeds  # noqa: E402
from run_seeds import (  # noqa: E402
    aggregate,
    build_parser,
    format_table,
    goal_term_metrics,
    measure_replication,
    parse_sweep_args,
    replication_std,
)

from goal_field import parse_goal_doses  # noqa: E402
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


def test_the_goal_term_reading_is_lifted_out_of_the_run_s_own_metrics_file(tmp_path):
    """The share the preregistration reads before the primary has to reach the summary.

    It is a training-step measurement, so it is in ``metrics.jsonl`` and never in
    ``eval_history``, which is where every other headline metric comes from. Last
    line wins, because the summary describes the run as it ended.
    """
    lines = [
        {"step": 0, "train/loss": 3.1, "auction/mean_goal_term": 0.001, "auction/goal_share": 0.01},
        {"step": 10, "eval/loss": 2.9},
        {
            "step": 20,
            "train/loss": 2.8,
            "auction/mean_goal_term": 0.004,
            "auction/goal_share": 0.07,
        },
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    assert goal_term_metrics(tmp_path) == {
        "auction/mean_goal_term": 0.004,
        "auction/goal_share": 0.07,
    }


def test_an_arm_with_no_auction_contributes_no_goal_term_row(tmp_path):
    """Empty, not zero: the dense arm has no economy, which is not a term of size nothing."""
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"step": 0, "train/loss": 3.1}) + "\n")

    assert goal_term_metrics(tmp_path) == {}
    assert goal_term_metrics(tmp_path / "nowhere") == {}


def test_the_sweep_takes_a_dose_per_goal_and_reaches_the_fingerprint_with_it():
    """#54's dose axis, from the flag to the tuples the fingerprint carries.

    The arm label is deliberately *not* asked to change: the two dose groups of
    signature 1's primary are one arm read at two doses, and putting the dose in
    the label would rename every table row the fixture already recorded.
    """
    args = build_parser().parse_args(["--goal_dose", "truthful=0.068,safe=0.017"])
    goals, doses = parse_goal_doses(args.goal_dose)

    assert (goals, doses) == (("truthful", "safe"), (0.068, 0.017))
    config = TrainingConfig(
        mob_layers_start=6, mob_layers_end=22, goal_fields=goals, goal_doses=doses
    )
    assert (config.goal_fields, config.goal_doses) == (goals, doses)

    assert build_parser().parse_args([]).goal_dose is None
    assert parse_goal_doses(build_parser().parse_args([]).goal_dose or ()) == ((), ())


def test_borrowing_a_floor_and_measuring_one_are_not_asked_for_together(capsys):
    """#56: --floor_recorded_at is what a sweep does *instead* of a replicate."""
    with pytest.raises(SystemExit):
        parse_sweep_args(["--floor_recorded_at", "runs/value", "--replicate"])

    assert "--no-replicate" in capsys.readouterr().err
