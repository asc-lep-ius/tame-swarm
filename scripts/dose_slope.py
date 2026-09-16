"""#32's primary contrast: the slope of the allocation shift against the dose.

Each coupled arm is read against the one field-on uncoupled reference as the
total-variation shift of the slot allocation, paired by seed
(``allocation_shift.paired_shifts``); the excess over the re-running floor is
reported beside it. The primary is one number: the least-squares slope of that
shift against beta, per seed, with ``compare_runs``' percentile bootstrap over
the paired per-seed slopes -- an interval that includes zero reads "no dose
recruits". Two guardrails are printed with it, so a null is never read as one
when it is not: the held-out loss of each coupled arm minus the reference,
against the pooled spread, and the perceived shift's maximum and mean against
the norm cap, so a cap that binds is named rather than read as a null.

    uv run python scripts/dose_slope.py --reference runs/mob@truthful \\
        --arm 0.1=runs/mob+truthful@truthful --arm 0.3=runs/b0.3 --arm 1.0=runs/b1.0 \\
        --floor_a runs/mob --floor_b runs/mob-replication --cap 0.1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))

from allocation_shift import (  # noqa: E402
    assert_identical_fingerprints,
    excess_over_floor,
    load_group,
    paired_shifts,
)
from compare_runs import (  # noqa: E402
    DEFAULT_RESAMPLES,
    MIN_PAIRS_FOR_COVERAGE,
    assert_groups_at_parity,
    bootstrap_mean,
)

CAP_KEYS = ("coupling/delta_fraction_max", "coupling/delta_fraction_mean")


def slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y against x; the one number #32 decides on."""
    if len(points) < 2:
        raise ValueError("a slope needs at least two doses")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        raise ValueError("every dose is the same; there is no slope to read")
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def per_seed_slopes(shifts: dict[float, dict[str, float]]) -> dict[str, float]:
    """The slope of shift against dose at every seed all doses share."""
    seeds = sorted(set.intersection(*(set(s) for s in shifts.values())), key=str)
    if not seeds:
        raise ValueError("the doses share no seed")
    return {seed: slope([(beta, shifts[beta][seed]) for beta in sorted(shifts)]) for seed in seeds}


def loss_deltas(reference: dict[str, Any], arm: dict[str, Any]) -> dict[str, float]:
    seeds = sorted(set(reference["per_seed"]) & set(arm["per_seed"]), key=str)
    return {
        s: arm["per_seed"][s]["eval/loss"] - reference["per_seed"][s]["eval/loss"] for s in seeds
    }


def pooled_loss_spread(reference: dict[str, Any], arm: dict[str, Any]) -> float:
    """The pooled seed spread ``compare_runs`` reads a loss delta against."""
    spreads = [
        g["stats"]["eval/loss"]["std"]
        for g in (reference, arm)
        if "eval/loss" in g.get("stats", {})
    ]
    return (sum(s**2 for s in spreads) / len(spreads)) ** 0.5 if spreads else float("nan")


def cap_readings(group_dir: Path, seeds: list[str]) -> dict[str, dict[str, float]]:
    """The perceived shift's maximum and mean at the end of each seed's run."""
    readings: dict[str, dict[str, float]] = {}
    for seed in seeds:
        last: dict[str, float] = {}
        metrics = group_dir / "runs" / f"seed{seed}" / "metrics.jsonl"
        if not metrics.exists():
            continue
        for line in metrics.read_text().splitlines():
            last.update(json.loads(line))
        readings[seed] = {key: last[key] for key in CAP_KEYS if key in last}
    return readings


def interval_label(count: int) -> str:
    return "95% interval" if count >= MIN_PAIRS_FOR_COVERAGE else "resampled-mean range"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=str, required=True, help="the field-on uncoupled group")
    parser.add_argument(
        "--arm", action="append", required=True, help="<beta>=<group dir>, repeated"
    )
    parser.add_argument("--floor_a", type=str, default=None)
    parser.add_argument("--floor_b", type=str, default=None)
    parser.add_argument(
        "--cap", type=float, default=None, help="max_coupling_fraction the arms ran under"
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    if (args.floor_a is None) != (args.floor_b is None):
        parser.error("--floor_a and --floor_b go together")

    reference = load_group(Path(args.reference))
    arms: dict[float, tuple[Path, dict[str, Any]]] = {}
    for spec in args.arm:
        beta, _, path = spec.partition("=")
        arms[float(beta)] = (Path(path), load_group(Path(path)))
    for _, group in arms.values():
        assert_groups_at_parity(reference, group)

    shifts = {beta: paired_shifts(reference, group) for beta, (_, group) in arms.items()}
    floor = None
    if args.floor_a is not None:
        floor_a, floor_b = load_group(Path(args.floor_a)), load_group(Path(args.floor_b))
        assert_identical_fingerprints(floor_a, floor_b)
        floor = paired_shifts(floor_a, floor_b)

    slopes = per_seed_slopes(shifts)
    values = [slopes[s] for s in sorted(slopes, key=str)]
    centre, low, high = bootstrap_mean(values, args.resamples, args.seed)

    print(
        "dose  allocation shift (TV) per seed"
        + ("   excess over floor per seed" if floor else "")
        + "   eval/loss delta per seed (pooled spread)"
    )
    for beta in sorted(shifts):
        seeds = sorted(shifts[beta], key=str)
        row = f"{beta:<5} " + "  ".join(f"s{s}={shifts[beta][s]:.3f}" for s in seeds)
        if floor:
            excess = excess_over_floor(shifts[beta], floor)
            row += "   " + "  ".join(f"s{s}={excess[s]:+.3f}" for s in sorted(excess, key=str))
        deltas = loss_deltas(reference, arms[beta][1])
        row += "   " + "  ".join(f"s{s}={deltas[s]:+.4f}" for s in seeds)
        row += f" ({pooled_loss_spread(reference, arms[beta][1]):.4f})"
        print(row)
    print()
    per_seed = "  ".join(f"s{s}={slopes[s]:+.3f}" for s in sorted(slopes, key=str))
    print(f"primary: slope of the allocation shift against beta, per seed: {per_seed}")
    print(
        f"  mean {centre:+.3f}  [{low:+.3f}, {high:+.3f}]  ({args.resamples} resamples over "
        f"{len(values)} paired seeds, {interval_label(len(values))})"
    )
    print(
        "  an interval that includes zero reads 'no dose recruits'; the excess column says "
        "whether any dose moved the allocation further than re-running"
    )
    print()
    if args.cap is not None:
        print(f"cap: max_coupling_fraction = {args.cap}")
        for beta in sorted(arms):
            readings = cap_readings(arms[beta][0], sorted(shifts[beta], key=str))
            for seed, r in readings.items():
                at_cap = r.get("coupling/delta_fraction_max", 0.0) >= 0.98 * args.cap
                maximum = r.get("coupling/delta_fraction_max", float("nan"))
                mean = r.get("coupling/delta_fraction_mean", float("nan"))
                print(
                    f"  beta {beta} seed {seed}: delta_fraction_max {maximum:.4f} mean {mean:.4f}"
                    + ("  <- the cap binds" if at_cap else "")
                )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "shifts": {str(b): v for b, v in shifts.items()},
                    "floor": floor,
                    "slopes": slopes,
                    "slope_mean": centre,
                    "slope_range": [low, high],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
