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

One metric is the *primary* contrast and the rest are secondary (#35). With
fifteen rows on the table the largest one is read by eye, and the largest of
fifteen null rows sits near two pooled spreads on its own -- so ``--primary``
prints the paired per-seed deltas for one declared metric with a bootstrap
interval over them, and the table prints what its own row count is expected to
produce under the null beside the largest it actually found. The declaration
travels with the data: ``run_seeds.py --primary`` writes it into the summary,
and a comparison with no primary anywhere prints no primary block.

When both summaries carry the arm fingerprints ``run_seeds.py`` records, the
comparison first asserts parity between each seed's pair of arms -- the two groups
may differ in the router, the coupling goal or the goal field (#28) and in nothing
else -- and refuses to print a delta whose arms disagree on anything more. A
summary written before fingerprints were recorded is compared unchecked, and says
so; one written before the field existed reads as a field-off arm.

    uv run python scripts/compare_runs.py \\
        --group_a runs/mob --group_b runs/softmax
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from parity import (  # noqa: E402
    ArmFingerprint,
    CodeDriftError,
    ParityError,
    assert_parity,
    code_drift,
)

logger = logging.getLogger("compare_runs")

DEFAULT_RESAMPLES = 10_000
CONFIDENCE = 0.95
# Below this many paired values the percentile interval is exactly the sample
# range -- at n = 3 a resample repeats one value with probability 1/27, so the
# 2.5th and 97.5th percentiles of the resampled means are the sample minimum and
# maximum. It is labelled a range for that reason and carries no 95% coverage.
MIN_PAIRS_FOR_COVERAGE = 6
# Below this many, resampling says nothing a reader cannot already see in the
# values themselves; the centre is printed alone.
MIN_PAIRS_FOR_INTERVAL = 3


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


def load_fingerprints(group: dict[str, Any]) -> list[ArmFingerprint]:
    """Every fingerprint a summary carries; empty when it carries none, or another schema's."""
    prints = group.get("fingerprints") or {}
    try:
        return [ArmFingerprint(**value) for value in prints.values()]
    except TypeError as exc:
        logger.warning(
            "a summary's fingerprints do not match this version's schema (%s); its code "
            "identity cannot be read",
            exc,
        )
        return []


def assert_same_code(
    group_a: dict[str, Any], group_b: dict[str, Any], allow_drift: bool = False
) -> str:
    """Refuse two groups that cannot be shown to have run one code (#31); else say what it was.

    #25's two attempts ran different code, fingerprinted equal, and were read as a
    replication. A missing SHA is drift: every summary recorded before #31 has
    none, and comparing it unlabelled is exactly the reading that went wrong. With
    ``allow_drift`` the reasons are logged and returned as the label the table
    prints under itself, so the comparison is made and never made silently.
    """
    prints_a, prints_b = load_fingerprints(group_a), load_fingerprints(group_b)
    reasons: list[str] = []
    if not prints_a or not prints_b:
        reasons.append("  a summary carries no arm fingerprints, so no code SHA (before #6)")
    reasons.extend(code_drift(prints_a + prints_b))
    if not reasons:
        sha = next(arm.code_sha for arm in prints_a if arm.code_sha)
        return f"code: one SHA across both groups ({sha[:9]}, clean tree)"
    detail = "\n".join(reasons)
    if not allow_drift:
        raise CodeDriftError(
            "the groups cannot be shown to have run the same code, so every delta could be "
            "the code and not the arm:\n" + detail + "\n"
            "pass --allow-code-drift to compare anyway, with the drift printed beside the table"
        )
    logger.warning("code drift allowed by --allow-code-drift:\n%s", detail)
    return "code: DRIFT, allowed by --allow-code-drift:\n" + detail


def assert_same_measured_goal(group_a: dict[str, Any], group_b: dict[str, Any]) -> None:
    """The routing columns are a contrast only against one direction (#24).

    Two groups measured against different goals -- or one measured and one not --
    would difference correlations taken against different alignments, which is
    not a contrast at all. A summary written before the goal was recorded carries
    no key and is compared as before.
    """
    if "trace_goal" not in group_a or "trace_goal" not in group_b:
        return
    if group_a["trace_goal"] != group_b["trace_goal"]:
        raise ParityError(
            f"the groups measured routing against different goals "
            f"({group_a['trace_goal']!r} vs {group_b['trace_goal']!r}); their routing/ "
            "columns are not a contrast"
        )


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


def bootstrap_mean(
    values: list[float], resamples: int = DEFAULT_RESAMPLES, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and its percentile bootstrap interval over ``values``, resampled with replacement.

    At fewer than ``MIN_PAIRS_FOR_COVERAGE`` values the interval is exactly the
    sample range; callers label it as such.
    """
    if not values:
        raise ValueError("nothing to bootstrap")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples))
    tail = (1 - CONFIDENCE) / 2
    low = means[int(tail * (resamples - 1))]
    high = means[int((1 - tail) * (resamples - 1))]
    return sum(values) / len(values), low, high


def paired_deltas(
    group_a: dict[str, Any], group_b: dict[str, Any], metric: str
) -> dict[str, float]:
    """B minus A on ``metric``, per seed both groups measured it on.

    The pairing is the stratification: a seed fixes the data order and the
    initialisation, so its two runs differ only in the arm, and resampling these
    paired values resamples whole seeds rather than shuffling runs between the
    groups. Differencing the group means instead would throw that pairing away.
    """
    shared = set(group_a["per_seed"]) & set(group_b["per_seed"])
    seeds = sorted(
        (
            seed
            for seed in shared
            if metric in group_a["per_seed"][seed] and metric in group_b["per_seed"][seed]
        ),
        key=str,
    )
    return {
        seed: group_b["per_seed"][seed][metric] - group_a["per_seed"][seed][metric]
        for seed in seeds
    }


def expected_largest_under_null(rows: int, upper: float = 12.0, steps: int = 24_000) -> float:
    """E[max |delta/std|] over ``rows`` rows that measured nothing.

    Treating each row under the null as a standard normal draw, the largest of N
    of them is a half-normal order statistic with mean

        E[max] = integral_0^inf (1 - erf(x / sqrt(2))^N) dx

    which Simpson's rule evaluates to three decimals against a 200k-draw
    simulation (2.051 at N = 15), and unlike a simulation prints the same number
    on every machine. It is a reference point rather than a threshold, and a
    conservative one: the rows are not independent -- the per-expert win shares
    sum to ``top_k`` -- and ``delta/pooled_std`` at three seeds per group is not
    a standard normal but roughly sqrt(2/3) t_4, whose heavier tails put the
    real maximum well above this. Simulating the statistic ``compare`` actually
    computes, at three seeds per arm, gives 2.34 / 2.67 / 3.86 at 10 / 15 / 50
    rows against the 1.88 / 2.05 / 2.51 printed here. The half-normal is what
    #35 specified and what the number below is; a row that clears *it* has not
    yet cleared the null this project's own sample size produces.
    """
    if rows < 1:
        raise ValueError("no rows to take a maximum over")
    step = upper / steps

    def survival(x: float) -> float:
        return 1.0 - math.erf(x / math.sqrt(2.0)) ** rows

    total = survival(0.0) + survival(upper)
    for index in range(1, steps):
        total += (4 if index % 2 else 2) * survival(index * step)
    return total * step / 3


def multiplicity_line(comparison: dict[str, dict[str, float]]) -> str:
    """What this many rows produces under the null, beside the largest row found.

    The reader's eye goes to the biggest number on the table, and #25's write-up
    quoted one (1.6 pooled spreads on one expert's correlation) off fifteen rows
    -- which is what fifteen rows that measured nothing are expected to hand you
    anyway. Printing both makes the comparison available at the moment the
    largest row is read, rather than in a write-up afterwards.
    """
    rows = len(comparison)
    if not rows:
        return "multiplicity: no rows compared"
    largest_metric, largest = max(
        comparison.items(), key=lambda item: abs(item[1]["delta_over_std"])
    )
    observed = abs(largest["delta_over_std"])
    return (
        f"multiplicity: {rows} row(s); largest |delta/std| observed {observed:.2f} "
        f"({largest_metric}), expected largest under the null "
        f"{expected_largest_under_null(rows):.2f} -- the mean of the largest of {rows} "
        "standard normal draws in absolute value (half-normal order statistic, "
        "Simpson-integrated), for rows that measured nothing. Read the declared "
        "primary; the rest are secondary."
    )


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
    lines.append("")
    lines.append(multiplicity_line(comparison))
    return "\n".join(lines)


def declared_primary(
    group_a: dict[str, Any], group_b: dict[str, Any], override: str | None = None
) -> str | None:
    """The primary metric for this comparison: the flag, else what the summaries declare.

    A summary written before ``run_seeds.py --primary`` existed carries no key,
    and absence prints nothing -- the tool makes the declared reading the easy
    one, it does not refuse an undeclared comparison. Two groups that declare
    *different* primaries are not one contrast with one primary, so neither is
    adopted silently.
    """
    if override is not None:
        return override
    primary_a, primary_b = group_a.get("primary"), group_b.get("primary")
    if primary_a and primary_b and primary_a != primary_b:
        logger.warning(
            "the groups declare different primary metrics (%r vs %r); pass --primary to pick one",
            primary_a,
            primary_b,
        )
        return None
    return primary_a or primary_b or None


def format_primary(
    metric: str,
    deltas: dict[str, float],
    label_a: str,
    label_b: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> str:
    """The primary contrast: the paired per-seed deltas, their mean, and an interval.

    The interval is a percentile bootstrap over the paired values, and is named
    for what it is at the count in hand -- a 95% interval at
    ``MIN_PAIRS_FOR_COVERAGE`` pairs or more, the range of the resampled means
    below that, and no interval at all below ``MIN_PAIRS_FOR_INTERVAL``, where the
    centre is all there is to print.
    """
    pairs = len(deltas)
    per_seed = "  ".join(f"s{seed_key}={value:+.5f}" for seed_key, value in deltas.items())
    centre = sum(deltas.values()) / pairs
    header = f"primary: {metric}  ({label_b} minus {label_a}, paired by seed)"
    if pairs < MIN_PAIRS_FOR_INTERVAL:
        return (
            f"{header}\n  mean {centre:+.5f}   {per_seed}\n"
            f"  no interval at n={pairs}: below {MIN_PAIRS_FOR_INTERVAL} paired seeds a "
            "resample carries no information the per-seed values do not."
        )
    _, low, high = bootstrap_mean(list(deltas.values()), resamples, seed)
    label = (
        f"{CONFIDENCE:.0%} bootstrap" if pairs >= MIN_PAIRS_FOR_COVERAGE else "resampled-mean range"
    )
    footnote = f"  ({resamples} resamples over {pairs} paired seeds, {label}"
    if pairs < MIN_PAIRS_FOR_COVERAGE:
        footnote += (
            f"; at n={pairs} the percentile interval is the sample range and has no 95% coverage"
        )
    return f"{header}\n  mean {centre:+.5f}  [{low:+.5f}, {high:+.5f}]   {per_seed}\n{footnote})"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_a", type=str, required=True)
    parser.add_argument("--group_b", type=str, required=True)
    parser.add_argument(
        "--label_a", type=str, default=None, help="Default: group_a's arm, from its summary"
    )
    parser.add_argument("--label_b", type=str, default=None, help="Default: group_b's arm")
    parser.add_argument(
        "--primary",
        type=str,
        default=None,
        help=(
            "The one metric this comparison is read on, with an interval over its "
            "paired per-seed deltas (default: whatever the summaries declare, if any)"
        ),
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--bootstrap_seed", type=int, default=0, help="The bootstrap's own RNG seed"
    )
    parser.add_argument(
        "--allow-code-drift",
        action="store_true",
        help=(
            "Compare two groups that did not provably run the same code (different SHAs, a "
            "dirty tree, or a summary from before the SHA was recorded), with the drift "
            "printed beside the table (default: refused)"
        ),
    )
    args = parser.parse_args()

    group_a = load_group(Path(args.group_a))
    group_b = load_group(Path(args.group_b))
    label_a = args.label_a or str(group_a.get("arm") or group_a.get("router", "A"))
    label_b = args.label_b or str(group_b.get("arm") or group_b.get("router", "B"))

    checked = assert_groups_at_parity(group_a, group_b)
    code_line = assert_same_code(group_a, group_b, allow_drift=args.allow_code_drift)
    assert_same_measured_goal(group_a, group_b)
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

    primary = declared_primary(group_a, group_b, args.primary)
    if primary is not None:
        deltas = paired_deltas(group_a, group_b, primary)
        if deltas:
            print()
            print(
                format_primary(
                    primary, deltas, label_a, label_b, args.resamples, args.bootstrap_seed
                )
            )
        else:
            logger.warning(
                "the primary metric %r is not measured on any seed both groups share; no "
                "interval is printed for it",
                primary,
            )
    print(
        "parity between the arms: asserted per seed"
        if checked
        else "parity between the arms: NOT asserted (no fingerprints in a summary)"
    )
    print(code_line)


if __name__ == "__main__":
    main()
