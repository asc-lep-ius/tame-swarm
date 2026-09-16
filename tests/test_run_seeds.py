"""run_seeds.py's replicate floor (#31): the arithmetic and the table, no model.

The replicate itself is a full training run and is exercised by the smoke sweep;
what a unit test can pin is that the floor is the same estimator ``aggregate``
uses across seeds, at n = 2, so ``compare_runs.py`` can pool the two.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_seeds import aggregate, format_table, replication_std  # noqa: E402


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
