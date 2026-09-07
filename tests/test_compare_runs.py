"""compare_runs.compare(): the arithmetic behind "is this effect real" (#13).

No model, no GPU, no fixture -- these are hand-built ``seed_summary.json``-shaped
dicts, the same shape ``run_seeds.py`` writes to disk and ``load_group`` reads
back. That is deliberate: this is the function the project's noise-floor verdict
rests on, and it is pure arithmetic, so there is no reason its correctness should
depend on anything slower than this.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_runs import assert_groups_at_parity, compare  # noqa: E402

from parity import ParityError  # noqa: E402

from .arm_fingerprints import BASE  # noqa: E402


def _group(router: str, values: dict[str, list[float]]) -> dict:
    """Build a group dict shaped like run_seeds.py's seed_summary.json.

    ``values`` maps metric name to one value per seed; seeds are just enumerated.
    """
    per_seed = {}
    for seed_index in range(max(len(v) for v in values.values())):
        per_seed[seed_index] = {
            metric: values[metric][seed_index]
            for metric in values
            if seed_index < len(values[metric])
        }
    stats = {metric: {"n": len(v)} for metric, v in values.items()}
    return {"router": router, "per_seed": per_seed, "stats": stats}


def test_identical_zero_values_report_zero_not_infinite():
    """The regression case: a metric that measured nothing in either group must
    not read as the most significant one on the page."""
    group = _group("mob", {"spec/expert_cosine_distance": [0.0, 0.0, 0.0]})

    result = compare(group, group)

    assert result["spec/expert_cosine_distance"]["delta_over_std"] == 0.0


def test_nonzero_delta_over_zero_spread_is_signed_infinity():
    group_a = _group("mob", {"eval/loss": [1.0, 1.0, 1.0]})
    group_b_up = _group("softmax", {"eval/loss": [2.0, 2.0, 2.0]})
    group_b_down = _group("softmax", {"eval/loss": [0.5, 0.5, 0.5]})

    up = compare(group_a, group_b_up)["eval/loss"]["delta_over_std"]
    down = compare(group_a, group_b_down)["eval/loss"]["delta_over_std"]

    assert math.isinf(up) and up > 0
    assert math.isinf(down) and down < 0


def test_normal_path_matches_hand_computed_pooled_std():
    group_a = _group("mob", {"eval/loss": [1.0, 2.0, 3.0]})
    group_b = _group("softmax", {"eval/loss": [2.0, 4.0, 6.0]})

    result = compare(group_a, group_b)["eval/loss"]

    mean_a, mean_b = 2.0, 4.0
    var_a = sum((v - mean_a) ** 2 for v in (1.0, 2.0, 3.0)) / 2
    var_b = sum((v - mean_b) ** 2 for v in (2.0, 4.0, 6.0)) / 2
    expected_pooled_std = math.sqrt((2 * var_a + 2 * var_b) / 4)

    assert result["mean_a"] == mean_a
    assert result["mean_b"] == mean_b
    assert result["delta"] == mean_b - mean_a
    assert math.isclose(result["pooled_std"], expected_pooled_std)
    assert math.isclose(result["delta_over_std"], (mean_b - mean_a) / expected_pooled_std)


def test_a_metric_with_only_one_seed_in_either_group_is_omitted():
    group_a = _group("mob", {"eval/loss": [1.0, 2.0, 3.0]})
    group_b = _group("softmax", {"eval/loss": [1.0]})  # n=1: no spread to pool

    result = compare(group_a, group_b)

    assert "eval/loss" not in result


def _with_fingerprints(group: dict, **changes) -> dict:
    prints = {
        seed: replace(BASE, seed=int(seed), **changes).as_dict() for seed in group["per_seed"]
    }
    return {**group, "fingerprints": prints}


def test_groups_that_differ_only_in_the_coupling_goal_are_at_parity():
    """#6's ablation: the coupled and uncoupled auction arms, seed by seed."""
    group_a = _with_fingerprints(_group("mob", {"eval/loss": [2.79, 2.80, 2.79]}))
    group_b = _with_fingerprints(
        _group("mob", {"eval/loss": [2.78, 2.79, 2.79]}), coupling_goal="truthful"
    )

    assert assert_groups_at_parity(group_a, group_b) is True


def test_groups_that_differ_in_a_confound_are_refused():
    group_a = _with_fingerprints(_group("mob", {"eval/loss": [2.79, 2.80]}))
    group_b = _with_fingerprints(
        _group("mob", {"eval/loss": [2.78, 2.79]}), coupling_goal="truthful", adapter_rank=8
    )

    with pytest.raises(ParityError, match="adapter_rank"):
        assert_groups_at_parity(group_a, group_b)


def test_groups_that_share_no_seed_are_compared_unchecked(caplog):
    group_a = _with_fingerprints(_group("mob", {"eval/loss": [2.79, 2.80]}))
    group_b = _with_fingerprints(_group("mob", {"eval/loss": [2.78, 2.79]}))
    group_b["fingerprints"] = {
        seed + 10: prints for seed, prints in group_b["fingerprints"].items()
    }

    with caplog.at_level("WARNING", logger="compare_runs"):
        assert assert_groups_at_parity(group_a, group_b) is False

    assert any("share no seed" in record.message for record in caplog.records)


def test_a_summary_without_fingerprints_is_compared_unchecked(caplog):
    group = _group("mob", {"eval/loss": [2.79, 2.80]})

    with caplog.at_level("WARNING", logger="compare_runs"):
        assert assert_groups_at_parity(group, _with_fingerprints(group)) is False

    assert any("no arm fingerprints" in record.message for record in caplog.records)
