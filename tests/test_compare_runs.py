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

from compare_runs import (  # noqa: E402
    assert_groups_at_parity,
    assert_same_code,
    bootstrap_mean,
    compare,
    declared_primary,
    expected_largest_under_null,
    format_primary,
    format_table,
    multiplicity_line,
    paired_deltas,
    pooled_replication_std,
    replication_note,
)

from parity import CodeDriftError, ParityError  # noqa: E402

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


def test_a_field_on_group_is_at_parity_with_a_summary_recorded_before_the_field_existed():
    """#28 against #25: the older summary carries no steer keys and reads as field-off."""
    group_a = _with_fingerprints(_group("mob", {"eval/loss": [2.79, 2.80, 2.79]}))
    group_a["fingerprints"] = {
        seed: {key: value for key, value in prints.items() if not key.startswith("steer_")}
        for seed, prints in group_a["fingerprints"].items()
    }
    group_b = _with_fingerprints(
        _group("mob", {"eval/loss": [2.78, 2.79, 2.79]}),
        steer_goal="truthful",
        steer_strength=4.0,
        steer_layers=(13, 16),
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


def test_groups_measured_against_different_goals_are_refused():
    """#24: the routing columns are a contrast only against one direction."""
    from compare_runs import assert_same_measured_goal

    group_a = _group("mob", {"routing/win_share_e0": [0.5, 0.6]})
    group_b = _group("mob", {"routing/win_share_e0": [0.6, 0.7]})
    group_a["trace_goal"], group_b["trace_goal"] = "truthful", "safe"
    with pytest.raises(ParityError, match="different goals"):
        assert_same_measured_goal(group_a, group_b)

    group_b["trace_goal"] = "truthful"
    assert_same_measured_goal(group_a, group_b)
    # A summary written before the goal was recorded is compared as before.
    del group_b["trace_goal"]
    assert_same_measured_goal(group_a, group_b)


def test_paired_deltas_are_b_minus_a_on_the_seeds_both_groups_measured():
    group_a = _group("mob", {"eval/loss": [2.80, 2.75, 2.90]})
    group_b = _group("mob", {"eval/loss": [2.70, 2.70, 2.70]})
    # A seed only group B trained is not a pair; neither is a metric only it reported.
    group_b["per_seed"][3] = {"eval/loss": 2.5}
    del group_a["per_seed"][2]["eval/loss"]

    deltas = paired_deltas(group_a, group_b, "eval/loss")

    assert deltas == {0: pytest.approx(-0.10), 1: pytest.approx(-0.05)}
    assert paired_deltas(group_a, group_b, "eval/perplexity") == {}


def test_the_primary_interval_is_the_bootstrap_over_the_hand_computed_deltas():
    """Six pairs: enough for the interval to be named a 95% bootstrap."""
    group_a = _group("mob", {"eval/loss": [2.80, 2.75, 2.90, 2.85, 2.70, 2.95]})
    group_b = _group("mob", {"eval/loss": [2.70, 2.70, 2.75, 2.80, 2.65, 2.80]})

    deltas = paired_deltas(group_a, group_b, "eval/loss")
    report = format_primary("eval/loss", deltas, "mob", "mob@truthful", resamples=2000, seed=1)

    expected = [-0.10, -0.05, -0.15, -0.05, -0.05, -0.15]
    assert list(deltas.values()) == pytest.approx(expected)
    mean, low, high = bootstrap_mean(expected, resamples=2000, seed=1)
    assert f"mean {mean:+.5f}" in report
    assert f"[{low:+.5f}, {high:+.5f}]" in report
    assert "95% bootstrap" in report
    assert "no 95% coverage" not in report
    assert "s0=-0.10000" in report


def test_below_six_pairs_the_interval_is_named_a_range_not_a_95_percent_interval():
    """At n=3 the percentile interval *is* the sample range (#28 already said so)."""
    group_a = _group("mob", {"eval/loss": [2.80, 2.75, 2.90]})
    group_b = _group("mob", {"eval/loss": [2.70, 2.70, 2.75]})

    report = format_primary(
        "eval/loss", paired_deltas(group_a, group_b, "eval/loss"), "a", "b", resamples=2000
    )

    assert "resampled-mean range" in report
    assert "95% bootstrap" not in report
    assert "at n=3 the percentile interval is the sample range" in report


def test_under_three_pairs_prints_the_centre_and_says_there_is_no_interval():
    group_a = _group("mob", {"eval/loss": [2.80, 2.75]})
    group_b = _group("mob", {"eval/loss": [2.70, 2.70]})

    report = format_primary(
        "eval/loss", paired_deltas(group_a, group_b, "eval/loss"), "a", "b", resamples=2000
    )

    assert "mean -0.07500" in report
    assert "no interval at n=2" in report
    assert "[" not in report


def test_the_expected_largest_null_row_matches_the_half_normal_order_statistic():
    """Checked against a 200k-draw simulation: 0.798 at N=1, 2.051 at N=15."""
    assert expected_largest_under_null(1) == pytest.approx(math.sqrt(2 / math.pi), abs=1e-4)
    assert expected_largest_under_null(15) == pytest.approx(2.051, abs=1e-3)
    assert expected_largest_under_null(50) == pytest.approx(2.510, abs=1e-3)
    # More rows can only make the largest of them bigger.
    assert expected_largest_under_null(2) > expected_largest_under_null(1)
    with pytest.raises(ValueError, match="no rows"):
        expected_largest_under_null(0)


def test_the_table_prints_its_row_count_beside_the_null_maximum_for_that_count():
    """#25's reading: 1.6 spreads on one of fifteen rows is what fifteen null rows give."""
    values = {f"routing/goal_correlation_e{i}": [0.1 * i, 0.2 * i, 0.15 * i] for i in range(15)}
    group_a = _group("mob", values)
    group_b = _group("mob", {metric: [v + 0.01 for v in vs] for metric, vs in values.items()})

    line = multiplicity_line(compare(group_a, group_b))

    assert "15 row(s)" in line
    assert f"{expected_largest_under_null(15):.2f}" in line
    assert line in format_table(compare(group_a, group_b), "mob", "mob@truthful")


def test_an_undeclared_primary_prints_nothing_and_a_declared_one_travels_with_the_data():
    group_a = _group("mob", {"eval/loss": [2.79, 2.80]})
    group_b = _group("mob", {"eval/loss": [2.78, 2.79]})

    assert declared_primary(group_a, group_b) is None
    assert declared_primary(group_a, group_b, "eval/perplexity") == "eval/perplexity"

    # run_seeds.py --primary writes the declaration into one or both summaries.
    group_b["primary"] = "eval/loss"
    assert declared_primary(group_a, group_b) == "eval/loss"
    group_a["primary"] = "eval/loss"
    assert declared_primary(group_a, group_b) == "eval/loss"
    # The flag still overrides a declaration.
    assert declared_primary(group_a, group_b, "spec/report_decisiveness") == (
        "spec/report_decisiveness"
    )


def test_groups_declaring_different_primaries_adopt_neither(caplog):
    group_a = _group("mob", {"eval/loss": [2.79, 2.80]})
    group_b = _group("mob", {"eval/loss": [2.78, 2.79]})
    group_a["primary"], group_b["primary"] = "eval/loss", "spec/report_decisiveness"

    with caplog.at_level("WARNING", logger="compare_runs"):
        assert declared_primary(group_a, group_b) is None

    assert any("different primary metrics" in record.message for record in caplog.records)


def test_the_multiplicity_line_survives_an_infinite_row_and_an_empty_table():
    """A nonzero delta over zero spread is the biggest row there is, and prints as one."""
    group_a = _group("mob", {"eval/loss": [1.0, 1.0, 1.0]})
    group_b = _group("softmax", {"eval/loss": [2.0, 2.0, 2.0]})

    line = multiplicity_line(compare(group_a, group_b))

    assert "observed inf (eval/loss)" in line
    assert multiplicity_line({}) == "multiplicity: no rows compared"


SHA_A, SHA_B = "a" * 40, "b" * 40


def _at_code(group: dict, sha: str | None, dirty: bool | None = False) -> dict:
    return _with_fingerprints(group, code_sha=sha, code_dirty=dirty)


def test_two_groups_at_one_clean_sha_pass_the_code_check_and_say_so():
    group_a = _at_code(_group("mob", {"eval/loss": [2.79, 2.80]}), SHA_A)
    group_b = _at_code(_group("mob", {"eval/loss": [2.78, 2.79]}), SHA_A)

    line = assert_same_code(group_a, group_b)

    assert "one SHA across both groups (aaaaaaaaa, clean tree)" in line


def test_groups_at_different_shas_are_refused_by_default():
    """#25's two attempts: different code, equal fingerprints, read as a replication."""
    group_a = _at_code(_group("mob", {"eval/loss": [2.79, 2.80]}), SHA_A)
    group_b = _at_code(_group("mob", {"eval/loss": [2.78, 2.79]}), SHA_B)

    with pytest.raises(CodeDriftError, match="different code"):
        assert_same_code(group_a, group_b)


def test_a_summary_recorded_before_the_sha_counts_as_drift():
    """Every run dir under ~/tame-runs is legacy: no SHA is not the same SHA."""
    legacy = _with_fingerprints(_group("mob", {"eval/loss": [2.79, 2.80]}))
    legacy["fingerprints"] = {
        seed: {k: v for k, v in prints.items() if not k.startswith("code_")}
        for seed, prints in legacy["fingerprints"].items()
    }
    current = _at_code(_group("mob", {"eval/loss": [2.78, 2.79]}), SHA_A)

    with pytest.raises(CodeDriftError, match="no code SHA recorded"):
        assert_same_code(legacy, current)
    with pytest.raises(CodeDriftError, match="no code SHA recorded"):
        assert_same_code(legacy, legacy)
    # Older still: a summary with no fingerprints at all.
    with pytest.raises(CodeDriftError, match="no arm fingerprints"):
        assert_same_code(_group("mob", {"eval/loss": [2.79, 2.80]}), current)


def test_a_dirty_tree_counts_as_drift_even_at_one_sha():
    group_a = _at_code(_group("mob", {"eval/loss": [2.79, 2.80]}), SHA_A)
    group_b = _at_code(_group("mob", {"eval/loss": [2.78, 2.79]}), SHA_A, dirty=True)

    with pytest.raises(CodeDriftError, match="dirty tree"):
        assert_same_code(group_a, group_b)


def test_allow_code_drift_compares_and_prints_the_drift_beside_the_table(caplog):
    group_a = _at_code(_group("mob", {"eval/loss": [2.79, 2.80]}), SHA_A)
    group_b = _at_code(_group("mob", {"eval/loss": [2.78, 2.79]}), SHA_B)

    with caplog.at_level("WARNING", logger="compare_runs"):
        line = assert_same_code(group_a, group_b, allow_drift=True)

    assert line.startswith("code: DRIFT, allowed by --allow-code-drift")
    assert "different code: ['aaaaaaaaa', 'bbbbbbbbb']" in line
    assert any("code drift allowed" in record.message for record in caplog.records)


def _with_replicate(group: dict, seed: int, floors: dict[str, float]) -> dict:
    return {**group, "replicate_seed": seed, "replication_std": floors}


def test_the_table_quotes_the_run_to_run_floor_beside_every_delta():
    """#31: the floor pooled over the groups that measured it, root mean square."""
    group_a = _with_replicate(
        _group("mob", {"eval/loss": [2.79, 2.80, 2.81]}), 0, {"eval/loss": 0.0012}
    )
    group_b = _with_replicate(
        _group("mob", {"eval/loss": [2.78, 2.79, 2.80]}), 0, {"eval/loss": 0.0016}
    )

    comparison = compare(group_a, group_b)
    expected = math.sqrt((0.0012**2 + 0.0016**2) / 2)

    assert comparison["eval/loss"]["replication_std"] == pytest.approx(expected)
    assert pooled_replication_std(group_a, group_b, "eval/loss") == pytest.approx(expected)
    table = format_table(comparison, "mob", "mob+truthful")
    assert "repl_std" in table.splitlines()[0]
    assert f"{expected:.5f}" in table
    assert "2 group(s)" in replication_note(group_a, group_b)


def test_a_floor_from_one_group_is_quoted_alone_and_from_none_prints_n_a():
    group_a = _with_replicate(
        _group("mob", {"eval/loss": [2.79, 2.80, 2.81]}), 0, {"eval/loss": 0.0012}
    )
    group_b = _group("mob", {"eval/loss": [2.78, 2.79, 2.80]})

    assert pooled_replication_std(group_a, group_b, "eval/loss") == pytest.approx(0.0012)
    assert "1 group(s)" in replication_note(group_a, group_b)

    comparison = compare(group_b, group_b)
    assert math.isnan(comparison["eval/loss"]["replication_std"])
    assert "n/a" in format_table(comparison, "a", "b").splitlines()[2]
    assert "NOT measured" in replication_note(group_b, group_b)


def _with_borrowed_floor(group: dict, path: str, arm: str, floor: dict[str, float]) -> dict:
    """A group that spent its replicate budget on seeds (#56, section 8 rule 4)."""
    return {
        **group,
        "replicate_seed": None,
        "replication_std": None,
        "floor_recorded_at": {
            "path": path,
            "arm": arm,
            "replicate_seed": 0,
            "replication_std": floor,
            "is_zero": all(value == 0.0 for value in floor.values()),
        },
    }


def test_a_borrowed_floor_is_quoted_and_the_note_says_whose_runs_it_is():
    """A floor nobody surfaces is a floor nobody applies -- and one nobody labels
    is another sweep's runs wearing this one's authority."""
    group_a = _with_borrowed_floor(
        _group("mob", {"eval/loss": [2.79, 2.80, 2.81]}),
        "~/tame-runs/39-stakes-dial/body/r1/value",
        "mob",
        {"eval/loss": 0.0},
    )
    group_b = _with_borrowed_floor(
        _group("mob", {"eval/loss": [2.78, 2.79, 2.80]}),
        "~/tame-runs/39-stakes-dial/body/r1/value",
        "mob",
        {"eval/loss": 0.0},
    )

    assert pooled_replication_std(group_a, group_b, "eval/loss") == 0.0
    note = replication_note(group_a, group_b)
    assert "BORROWED by both groups and measured by neither" in note
    assert "39-stakes-dial/body/r1/value" in note

    measured = _with_replicate(
        _group("mob", {"eval/loss": [2.79, 2.80, 2.81]}), 0, {"eval/loss": 0.0012}
    )
    mixed = replication_note(measured, group_b)
    assert "1 group(s)" in mixed
    assert "The other group borrowed its floor" in mixed
