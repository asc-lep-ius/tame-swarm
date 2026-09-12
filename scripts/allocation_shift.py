"""How far did the slot allocation move, against how far re-running moves it? (#28)

The per-expert win-share deltas ``compare_runs.py`` prints are fifteen rows read
one at a time, and the largest of fifteen null rows sits near two spreads on its
own. The allocation is one object -- a distribution of slots over experts -- so
its movement is one number: the total-variation shift

    TV(a, b) = sum_i |win_share_b,i - win_share_a,i| / 2

between two runs at the same seed, in units of the slots (``top_k``) each token
buys. TV is non-negative, so "the interval includes zero" is not a test it can
fail; the null is what *re-running* produces. Given a replication pair -- two
groups at identical fingerprints (``--floor_a``/``--floor_b``, #25's kept the
first ablation attempt for exactly this) -- the contrast is read as

    excess_s = TV(contrast, seed s) - TV(floor, seed s)

paired by seed, with a percentile bootstrap over the paired values. A range
that includes zero reads "the contrast moved the allocation no further than
re-running does". Below six paired values the 2.5th and 97.5th percentiles of
the resampled means are the sample minimum and maximum -- at n = 3 a resample
repeats one value with probability 1/27 -- so the interval is the per-seed
range and carries no 95% coverage; it is labelled as a range for that reason
(#35 owns the general form).

Parity is asserted between the two groups exactly as ``compare_runs.py`` does,
and the floor pair must be two groups at *identical* fingerprints: that is what
makes it a floor.

    uv run python scripts/allocation_shift.py \\
        --group_a runs/mob --group_b runs/mob@truthful \\
        --floor_a runs/mob --floor_b runs/mob-replication
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import assert_groups_at_parity, load_group  # noqa: E402

WIN_SHARE_PREFIX = "routing/win_share_e"
DEFAULT_RESAMPLES = 10_000
CONFIDENCE = 0.95
# Below this many paired values the percentile interval is the sample range.
MIN_PAIRS_FOR_COVERAGE = 6


def win_shares(result: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in result.items() if key.startswith(WIN_SHARE_PREFIX)}


def total_variation(result_a: dict[str, float], result_b: dict[str, float]) -> float:
    """Half the L1 distance between two runs' slot allocations.

    Both runs must report the same experts: a run missing a column would
    otherwise read as a smaller shift, and half-L1 over a subset of a
    ``top_k``-mass allocation is not a total variation at all.
    """
    shares_a, shares_b = win_shares(result_a), win_shares(result_b)
    if not shares_a or not shares_b:
        raise ValueError("no win-share column is shared by the two runs")
    if set(shares_a) != set(shares_b):
        raise ValueError(
            f"the two runs report different experts: {sorted(set(shares_a) ^ set(shares_b))}"
        )
    return sum(abs(shares_b[expert] - shares_a[expert]) for expert in shares_a) / 2


def paired_shifts(group_a: dict[str, Any], group_b: dict[str, Any]) -> dict[str, float]:
    """TV per seed the two groups share, keyed by seed."""
    seeds = sorted(set(group_a["per_seed"]) & set(group_b["per_seed"]), key=str)
    if not seeds:
        raise ValueError("the groups share no seed; the shift is paired by seed")
    return {
        seed: total_variation(group_a["per_seed"][seed], group_b["per_seed"][seed])
        for seed in seeds
    }


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


def excess_over_floor(contrast: dict[str, float], floor: dict[str, float]) -> dict[str, float]:
    """Contrast minus floor shift, per seed both report."""
    seeds = sorted(set(contrast) & set(floor), key=str)
    if not seeds:
        raise ValueError("the contrast and the floor share no seed")
    return {seed: contrast[seed] - floor[seed] for seed in seeds}


def assert_identical_fingerprints(group_a: dict[str, Any], group_b: dict[str, Any]) -> None:
    """A floor pair is two runs of one configuration: same fingerprint, seed by seed."""
    prints_a, prints_b = group_a.get("fingerprints") or {}, group_b.get("fingerprints") or {}
    if not prints_a or not prints_b:
        raise ValueError("a floor pair needs the arm fingerprints run_seeds.py records")
    for seed in sorted(set(prints_a) & set(prints_b), key=str):
        if prints_a[seed] != prints_b[seed]:
            differing = sorted(
                k for k in prints_a[seed] if prints_a[seed][k] != prints_b[seed].get(k)
            )
            raise ValueError(
                f"the floor pair is not a replication: seed {seed} differs on {differing}"
            )


def format_report(
    contrast: dict[str, float],
    floor: dict[str, float] | None,
    resamples: int,
    seed: int,
) -> str:
    def row(label: str, shifts: dict[str, float]) -> str:
        mean, low, high = bootstrap_mean(list(shifts.values()), resamples, seed)
        per_seed = "  ".join(f"s{s}={v:.3f}" for s, v in shifts.items())
        return f"{label:<24}{mean:>8.3f}  [{low:.3f}, {high:.3f}]   {per_seed}"

    interval = (
        f"{CONFIDENCE:.0%} bootstrap"
        if len(contrast) >= MIN_PAIRS_FOR_COVERAGE
        else "resampled-mean range"
    )
    lines = [
        f"{'allocation shift (TV)':<24}{'mean':>8}  {interval:<22}per seed",
        "-" * 78,
        row("contrast", contrast),
    ]
    if floor is not None:
        lines.append(row("floor (re-running)", floor))
        lines.append(row("excess over floor", excess_over_floor(contrast, floor)))
        lines.append(
            "\nexcess is TV(contrast) - TV(floor), paired by seed; a range that includes "
            "zero reads 'moved no further than re-running does'."
        )
    footer = f"({resamples} resamples over {len(contrast)} paired seeds; #35 owns the form"
    if len(contrast) < MIN_PAIRS_FOR_COVERAGE:
        footer += (
            f"; at n={len(contrast)} the percentile interval is the sample range and has "
            "no 95% coverage"
        )
    lines.append(footer + ")")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_a", type=str, required=True)
    parser.add_argument("--group_b", type=str, required=True)
    parser.add_argument("--floor_a", type=str, default=None, help="A replication pair at ...")
    parser.add_argument("--floor_b", type=str, default=None, help="... identical fingerprints")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0, help="The bootstrap's own RNG seed")
    parser.add_argument("--json", type=str, default=None, help="Also write the shifts here")
    args = parser.parse_args()
    if (args.floor_a is None) != (args.floor_b is None):
        parser.error("--floor_a and --floor_b go together")

    group_a, group_b = load_group(Path(args.group_a)), load_group(Path(args.group_b))
    assert_groups_at_parity(group_a, group_b)
    contrast = paired_shifts(group_a, group_b)
    floor = None
    if args.floor_a is not None:
        floor_a, floor_b = load_group(Path(args.floor_a)), load_group(Path(args.floor_b))
        assert_identical_fingerprints(floor_a, floor_b)
        floor = paired_shifts(floor_a, floor_b)
    print(format_report(contrast, floor, args.resamples, args.seed))
    if args.json:
        Path(args.json).write_text(json.dumps({"contrast": contrast, "floor": floor}, indent=2))


if __name__ == "__main__":
    main()
