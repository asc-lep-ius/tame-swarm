"""Is an effect real, or is it inside the noise floor? One command (#13).

Takes two ``run_seeds.py`` output directories -- each a multi-seed replicate group
for one configuration -- and for every headline metric both groups share, reports:

    delta         group B's mean minus group A's
    pooled_std    the noise floor: replicate spread pooled across *both* groups
    delta / std   the delta in units of that spread

A delta smaller than its own pooled std is not distinguishable from what re-running
the same configuration already produces; #12's between-seed spread on report
decisiveness was ~46 points on its own, so a "the arms differ" claim that does not
clear its own noise floor is exactly the kind of number this project has already
been burned by publishing.

Deliberately reads ``seed_summary.json`` off disk rather than querying MLflow: the
summary is what ``run_seeds.py`` already computed the per-seed values from, it
needs no tracking backend to be installed or configured, and it is the same
log-of-record principle ``metrics.jsonl`` already follows (see ``metrics.py``) --
the number has value, MLflow is one of several places it is filed.

When both summaries carry the arm fingerprints ``run_seeds.py`` records, the
comparison first asserts parity between each seed's pair of arms -- the two groups
may differ in the router or the coupling goal and in nothing else -- and refuses
to print a delta whose arms disagree on anything more. A summary written before
fingerprints were recorded is compared unchecked, and says so.

    uv run python scripts/compare_runs.py \\
        --group_a runs/mob --group_b runs/softmax
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from parity import ArmFingerprint, assert_parity  # noqa: E402

logger = logging.getLogger("compare_runs")


def load_group(path: Path) -> dict[str, Any]:
    summary_path = path / "seed_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No seed_summary.json under {path} -- pass a --group_a/--group_b "
            "directory produced by scripts/run_seeds.py"
        )
    return json.loads(summary_path.read_text())


def assert_groups_at_parity(group_a: dict[str, Any], group_b: dict[str, Any]) -> bool:
    """Parity between the two groups' seed-matched arms; False when a group has no fingerprints.

    Seeds present in only one group are not compared -- a seed-mismatched pair is
    a different comparison, not a confound.
    """
    prints_a = group_a.get("fingerprints") or {}
    prints_b = group_b.get("fingerprints") or {}
    if not prints_a or not prints_b:
        logger.warning(
            "one of the groups carries no arm fingerprints (recorded by run_seeds.py since "
            "#6); parity between the arms is not asserted for this comparison"
        )
        return False
    shared_seeds = sorted(set(prints_a) & set(prints_b), key=str)
    if not shared_seeds:
        logger.warning(
            "the groups share no seed (%s vs %s); parity between the arms is not asserted",
            sorted(prints_a, key=str),
            sorted(prints_b, key=str),
        )
        return False
    try:
        pairs = [
            (ArmFingerprint(**prints_a[seed]), ArmFingerprint(**prints_b[seed]))
            for seed in shared_seeds
        ]
    except TypeError as exc:
        logger.warning(
            "a summary's fingerprints do not match this version's schema (%s); parity "
            "between the arms is not asserted",
            exc,
        )
        return False
    for arm_a, arm_b in pairs:
        assert_parity([arm_a, arm_b])
    return True


def _values(group: dict[str, Any], metric: str) -> list[float]:
    return [result[metric] for result in group["per_seed"].values() if metric in result]


def compare(group_a: dict[str, Any], group_b: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Delta and pooled std for every metric both groups measured on >=2 seeds each.

    A metric measured on only one seed in either group has no within-group spread
    to pool, and a delta reported without a noise floor to compare it against is
    exactly the unquotable single-sample number #13 exists to stop shipping --
    so it is skipped rather than reported with a missing denominator.
    """
    metrics_a = {metric for metric, stats in group_a["stats"].items() if stats["n"] >= 2}
    metrics_b = {metric for metric, stats in group_b["stats"].items() if stats["n"] >= 2}

    comparison: dict[str, dict[str, float]] = {}
    for metric in sorted(metrics_a & metrics_b):
        values_a = _values(group_a, metric)
        values_b = _values(group_b, metric)
        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)

        var_a = sum((v - mean_a) ** 2 for v in values_a) / (len(values_a) - 1)
        var_b = sum((v - mean_b) ** 2 for v in values_b) / (len(values_b) - 1)
        pooled_df = (len(values_a) - 1) + (len(values_b) - 1)
        pooled_std = math.sqrt(
            ((len(values_a) - 1) * var_a + (len(values_b) - 1) * var_b) / pooled_df
        )

        delta = mean_b - mean_a
        # A metric with zero pooled spread and zero delta (e.g. spec/expert_cosine_distance
        # before any specialisation has emerged) is "no difference measured", not an
        # infinitely significant one -- only a genuine nonzero delta over zero spread
        # is the unbounded case, and copysign keeps that case's direction (a group B
        # that fell rather than rose over zero spread must not print as +inf).
        if pooled_std > 0:
            delta_over_std = delta / pooled_std
        elif delta == 0:
            delta_over_std = 0.0
        else:
            delta_over_std = math.copysign(float("inf"), delta)
        comparison[metric] = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "pooled_std": pooled_std,
            "delta_over_std": delta_over_std,
        }
    return comparison


def format_table(comparison: dict[str, dict[str, float]], label_a: str, label_b: str) -> str:
    header = (
        f"{'metric':<32}{label_a:>14}{label_b:>14}{'delta':>12}{'pooled_std':>12}{'delta/std':>11}"
    )
    lines = [header, "-" * len(header)]
    for metric, values in comparison.items():
        lines.append(
            f"{metric:<32}{values['mean_a']:>14.5f}{values['mean_b']:>14.5f}"
            f"{values['delta']:>+12.5f}{values['pooled_std']:>12.5f}"
            f"{values['delta_over_std']:>11.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_a", type=str, required=True)
    parser.add_argument("--group_b", type=str, required=True)
    parser.add_argument(
        "--label_a", type=str, default=None, help="Default: group_a's arm, from its summary"
    )
    parser.add_argument("--label_b", type=str, default=None, help="Default: group_b's arm")
    args = parser.parse_args()

    group_a = load_group(Path(args.group_a))
    group_b = load_group(Path(args.group_b))
    label_a = args.label_a or str(group_a.get("arm") or group_a.get("router", "A"))
    label_b = args.label_b or str(group_b.get("arm") or group_b.get("router", "B"))

    checked = assert_groups_at_parity(group_a, group_b)
    comparison = compare(group_a, group_b)
    if not comparison:
        raise SystemExit(
            "No metric had >=2 seeds in both groups -- nothing to compare a noise "
            "floor against. Run scripts/run_seeds.py with --seeds covering at "
            "least two values for each group first."
        )

    print("\n" + format_table(comparison, label_a, label_b))
    print(
        "\ndelta/std is the delta in units of the pooled replicate spread -- "
        "well under 1 means the effect is not distinguishable from re-running "
        "the same configuration."
    )
    print(
        "parity between the arms: asserted per seed"
        if checked
        else "parity between the arms: NOT asserted (no fingerprints in a summary)"
    )


if __name__ == "__main__":
    main()
