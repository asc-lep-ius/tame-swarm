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
re-running does". The bootstrap itself, and the count below which its interval
is a range rather than a 95% interval, live in ``compare_runs`` (#35).

Parity is asserted between the two groups exactly as ``compare_runs.py`` does,
and the floor pair must be two groups at *identical* fingerprints: that is what
makes it a floor.

    uv run python scripts/allocation_shift.py \\
        --group_a runs/mob --group_b runs/mob@truthful \\
        --floor_a runs/mob --floor_b runs/mob-replication

**The readout is a mode of this script, never a second script beside it (#57).**
``--readout`` picks which number the shift *is*; ``total-variation`` is the
default and is the one every recorded row was read with, pinned bit for bit
against #39's fixture and body numbers in ``tests/test_allocation_shift.py``.
The three candidates #57 measures against it read the same runs differently:
over the tokens the swept goal is about rather than all of them
(``token-conditioned``), over every logged reading of the tail rather than the
end of training (``integrated``), and within one run's own before and after
rather than between two dose groups (``setpoint-step``). A readout changes
forward only (preregistration section 8, rule 2): an estimator chosen after a
run's sign was known is the next run's instrument, never that run's.
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import (  # noqa: E402
    CONFIDENCE,
    DEFAULT_RESAMPLES,
    MIN_PAIRS_FOR_COVERAGE,
    assert_groups_at_parity,
    bootstrap_mean,
    load_group,
)

WIN_SHARE_PREFIX = "routing/win_share_e"
# #57 candidate 2: the same slots, counted only on the tokens that call for the
# swept goal's type. Routing is load-bearing on a minority of tokens (about 7% on
# public MoEs, arXiv 2605.07260), so an average over all of them dilutes a
# cell-scale signature by whatever that minority's share is -- on this fixture,
# four types, a quarter.
SWEPT_TYPE_PREFIX = "routing/win_share_on_swept_type_e"
HELD_TYPE_PREFIX = "routing/win_share_on_held_type_e"
# #57 candidate 3: one windowed reading of the allocation per logged step,
# ``routing/win_share@<step>_e<i>``. The end of training is four numbers from a
# whole trajectory; these are the trajectory, after one wealth memory horizon.
READING_PREFIX = "routing/win_share@"
# #57 candidate 1: one run's allocation before the experimenter's setpoint step
# and after it has settled again. A within-run contrast, so its unit of pairing
# is the run rather than the dose group.
STEP_PRE_PREFIX = "routing/win_share_pre_step_e"
STEP_POST_PREFIX = "routing/win_share_post_step_e"


def win_shares(result: dict[str, float], prefix: str = WIN_SHARE_PREFIX) -> dict[str, float]:
    return {key: value for key, value in result.items() if key.startswith(prefix)}


def total_variation(
    result_a: dict[str, float], result_b: dict[str, float], prefix: str = WIN_SHARE_PREFIX
) -> float:
    """Half the L1 distance between two runs' slot allocations.

    Both runs must report the same experts: a run missing a column would
    otherwise read as a smaller shift, and half-L1 over a subset of a
    ``top_k``-mass allocation is not a total variation at all.
    """
    shares_a, shares_b = win_shares(result_a, prefix), win_shares(result_b, prefix)
    if not shares_a or not shares_b:
        raise ValueError(f"a run reports no {prefix!r} column")
    if set(shares_a) != set(shares_b):
        raise ValueError(
            f"the two runs report different experts: {sorted(set(shares_a) ^ set(shares_b))}"
        )
    return sum(abs(shares_b[expert] - shares_a[expert]) for expert in shares_a) / 2


def reading_steps(result: dict[str, float]) -> list[str]:
    """The logged steps a run recorded a windowed allocation at, in order."""
    steps = {
        key[len(READING_PREFIX) :].split("_e")[0] for key in win_shares(result, READING_PREFIX)
    }
    return sorted(steps, key=int)


def integrated_shift(result_a: dict[str, float], result_b: dict[str, float]) -> float:
    """The mean total variation over every reading the two runs share.

    Frison & Pocock: under compound symmetry the mean of ``k`` readings has
    variance ``sigma^2 (1 + (k-1) rho) / k``, so the gain over one reading is
    bounded by the correlation between successive readings and is nothing at
    ``rho = 1``. That correlation is measured on the fixture before the candidate
    is kept (``scripts/estimator_study.py``), not assumed.
    """
    steps = [step for step in reading_steps(result_a) if step in set(reading_steps(result_b))]
    if not steps:
        raise ValueError("the two runs share no logged reading; --readout integrated needs them")
    return sum(
        total_variation(result_a, result_b, f"{READING_PREFIX}{step}_e") for step in steps
    ) / len(steps)


def setpoint_step_shift(result: dict[str, float]) -> float:
    """One run's allocation shift across the experimenter's setpoint step.

    The thermostat's test: move the target, not the electricity price, and read
    the regulation curve. A within-run contrast, so the pairing is inside the run
    and the seed spread it has to clear is the spread of a *difference* rather
    than of two end-of-training allocations.
    """
    # Checked before the rename, or a run that has `routing/win_share_e*` and no
    # step columns is refused with "a run reports no 'routing/win_share_e'
    # column" -- naming a column it does have, from a traceback rather than a
    # message. Every one of #39's recorded groups takes that path.
    for prefix in (STEP_PRE_PREFIX, STEP_POST_PREFIX):
        if not any(key.startswith(prefix) for key in result):
            raise ValueError(
                f"a run reports no {prefix!r} column, so it did not take a setpoint step; "
                "--readout setpoint-step reads a run measure_stakes_dial.py stepped, not one "
                "of two dose groups"
            )
    return total_variation(
        {
            key.replace(STEP_PRE_PREFIX, WIN_SHARE_PREFIX): value
            for key, value in win_shares(result, STEP_PRE_PREFIX).items()
        },
        {
            key.replace(STEP_POST_PREFIX, WIN_SHARE_PREFIX): value
            for key, value in win_shares(result, STEP_POST_PREFIX).items()
        },
    )


@dataclass(frozen=True)
class Readout:
    """One way of turning runs into the number signature 1 is read on."""

    name: str
    within_run: bool
    describe: str
    reading: Callable[..., float]


READOUTS = {
    readout.name: readout
    for readout in (
        Readout(
            "total-variation",
            False,
            "the recorded readout: TV over the four end-of-training win shares",
            total_variation,
        ),
        Readout(
            "token-conditioned",
            False,
            "TV over the win shares on the tokens of the swept goal's own type",
            lambda a, b: total_variation(a, b, SWEPT_TYPE_PREFIX),
        ),
        Readout(
            "integrated",
            False,
            "the mean TV over every logged reading after one wealth memory horizon",
            integrated_shift,
        ),
        Readout(
            "setpoint-step",
            True,
            "one run's TV across the experimenter's setpoint step, read within the run",
            setpoint_step_shift,
        ),
    )
}
DEFAULT_READOUT = READOUTS["total-variation"]


def paired_shifts(
    group_a: dict[str, Any], group_b: dict[str, Any], readout: Readout = DEFAULT_READOUT
) -> dict[str, float]:
    """The shift between two groups per seed they share, keyed by seed."""
    if readout.within_run:
        raise ValueError(
            f"{readout.name} is read inside one run, not between two groups; call "
            "per_seed_readings(group, readout) and take the contrast across arms"
        )
    seeds = sorted(set(group_a["per_seed"]) & set(group_b["per_seed"]), key=str)
    if not seeds:
        raise ValueError("the groups share no seed; the shift is paired by seed")
    return {
        seed: readout.reading(group_a["per_seed"][seed], group_b["per_seed"][seed])
        for seed in seeds
    }


def per_seed_readings(group: dict[str, Any], readout: Readout) -> dict[str, float]:
    """One run's own reading per seed, for a readout whose contrast is inside the run."""
    if not readout.within_run:
        raise ValueError(
            f"{readout.name} is the distance between two groups; call paired_shifts(a, b)"
        )
    return {seed: readout.reading(result) for seed, result in sorted(group["per_seed"].items())}


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
    shared = sorted(set(prints_a) & set(prints_b), key=str)
    if not shared:
        raise ValueError("the floor pair shares no seed")
    for seed in shared:
        if prints_a[seed] != prints_b[seed]:
            differing = sorted(
                key
                for key in set(prints_a[seed]) | set(prints_b[seed])
                if prints_a[seed].get(key) != prints_b[seed].get(key)
            )
            raise ValueError(
                f"the floor pair is not a replication: seed {seed} differs on {differing}"
            )


def format_report(
    contrast: dict[str, float],
    floor: dict[str, float] | None,
    resamples: int,
    seed: int,
    readout: Readout = DEFAULT_READOUT,
) -> str:
    def row(label: str, shifts: dict[str, float]) -> str:
        mean, low, high = bootstrap_mean(list(shifts.values()), resamples, seed)
        per_seed = "  ".join(f"s{s}={v:.3f}" for s, v in shifts.items())
        return f"{label:<24}{mean:>8.3f}  [{low:.3f}, {high:.3f}]   {per_seed}"

    pairs = len(contrast) if floor is None else len(excess_over_floor(contrast, floor))
    interval = (
        f"{CONFIDENCE:.0%} bootstrap" if pairs >= MIN_PAIRS_FOR_COVERAGE else "resampled-mean range"
    )
    label = "allocation shift (TV)" if readout is DEFAULT_READOUT else readout.name
    lines = [
        f"{label:<24}{'mean':>8}  {interval:<22}per seed",
        "-" * 78,
        row("contrast" if not readout.within_run else "b - a, within each run", contrast),
    ]
    if readout is not DEFAULT_READOUT:
        lines.insert(1, f"readout: {readout.name} -- {readout.describe}")
    if floor is not None:
        lines.append(row("floor (re-running)", floor))
        lines.append(row("excess over floor", excess_over_floor(contrast, floor)))
        lines.append(
            "\nexcess is TV(contrast) - TV(floor), paired by seed; a range that includes "
            "zero reads 'moved no further than re-running does'."
        )
    footer = f"({resamples} resamples over {pairs} paired seeds; #35 owns the form"
    if pairs < MIN_PAIRS_FOR_COVERAGE:
        footer += (
            f"; at n={pairs} the percentile interval is the sample range and has no 95% coverage"
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
    parser.add_argument(
        "--readout",
        type=str,
        default=DEFAULT_READOUT.name,
        choices=sorted(READOUTS),
        help=(
            "Which number the shift is (#57). The default is the readout every recorded row "
            f"was read with (default: {DEFAULT_READOUT.name})"
        ),
    )
    args = parser.parse_args()
    if (args.floor_a is None) != (args.floor_b is None):
        parser.error("--floor_a and --floor_b go together")
    readout = READOUTS[args.readout]
    if readout.within_run and args.floor_a is not None:
        parser.error(
            f"--floor_a/--floor_b is the re-running floor of a *between-group* shift; "
            f"{readout.name} reads inside one run, and its floor is that run's own replicate"
        )

    # Every reader below raises ValueError on a group it cannot read with the
    # chosen readout -- a run with no step columns, two groups sharing no logged
    # reading. At the CLI boundary that is a usage error and prints as one, the
    # way dose_slope.py and compare_runs.py already do, rather than as a
    # traceback the operator reads past to find the sentence.
    try:
        group_a, group_b = load_group(Path(args.group_a)), load_group(Path(args.group_b))
        assert_groups_at_parity(group_a, group_b)
        if readout.within_run:
            # The two groups are two arms, and each run already carries its own
            # before and after: the contrast is the primary itself rather than a
            # distance between dose levels.
            readings_a = per_seed_readings(group_a, readout)
            readings_b = per_seed_readings(group_b, readout)
            contrast = {
                seed: readings_b[seed] - readings_a[seed]
                for seed in sorted(set(readings_a) & set(readings_b), key=str)
            }
        else:
            contrast = paired_shifts(group_a, group_b, readout)
        floor = None
        if args.floor_a is not None:
            floor_a, floor_b = load_group(Path(args.floor_a)), load_group(Path(args.floor_b))
            assert_identical_fingerprints(floor_a, floor_b)
            floor = paired_shifts(floor_a, floor_b, readout)
    except ValueError as exc:
        parser.error(str(exc))
    print(format_report(contrast, floor, args.resamples, args.seed, readout))
    if args.json:
        Path(args.json).write_text(json.dumps({"contrast": contrast, "floor": floor}, indent=2))


if __name__ == "__main__":
    main()
