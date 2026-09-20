"""What a contrast needs before it is worth GPU-hours (#56, preregistration section 8).

#39's body sweep cost 8.2 GPU-hours and resolved ``value`` minus ``decoupled`` at
a paired effect size of dz = 0.867 -- 13 paired seeds a side for 80% power under
a paired t, 26 runs an arm since a seed runs at both dose levels -- while the
contrast the stakes claim actually rests on, ``value`` minus ``shuffled``, read
dz = 0.151: 348 seeds, 696 runs an arm, about 700 GPU-hours. The fixture could
have said so for the price of a CPU-minute and was not asked. This is the asking.

Four questions, four flags, and they compose::

    uv run python scripts/power.py --dz 0.87                    # what it needs
    uv run python scripts/power.py --shifts a.json b.json       # what dz is
    uv run python scripts/power.py --dz 0.87 --plan             # how to spend it
    uv run python scripts/power.py --null-calibrate one_arm.json  # what it lies about

**The paired t, never the normal approximation.** ``(z_{1-a/2} + z_{power})^2 /
dz^2`` reads 10.4 where the exact calculation reads 13 at dz 0.867, because the
variance is estimated from the same handful of seeds -- a fifth of the sweep,
missing, at the count where it matters most. ``scipy.stats.nct`` is the
non-central t the alternative actually follows.

**The ceiling is a dead band, not a target.** 15 GPU-hours per confirmatory
sweep, the operator's decision of 2026-09-20: spend until the uncertainty about
the sign is inside it, then stop. A design whose power row exceeds the ceiling
is sent back to the fixture (#57, #48) -- so ``--plan`` refuses to schedule one
and says so, rather than printing a schedule nobody may run.

**Pairing is the cheapest lever and it is a property of the substrate, not of
the arithmetic.** #39's arms correlate at rho = 0.91 across seeds, which is why
dz reached 0.867 on an effect of 0.03 TV units; Sharma 2025 puts the gain at
1/(1 - rho), so those three paired seeds carried the precision of 34 unpaired
ones. ``strict`` determinism is what guarantees the pairing here (a seed fixes
the data order and the initialisation in both arms); a new substrate has to show
it rather than inherit it, so ``--shifts`` prints rho beside every dz.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import nct, norm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import MIN_PAIRS_FOR_COVERAGE, bootstrap_mean  # noqa: E402

ALPHA = 0.05
TARGET_POWER = 0.80
# The operator's decision of 2026-09-20, recorded in #56 and in section 8: what a
# confirmatory sweep may cost before its design goes back to the fixture.
CEILING_GPU_HOURS = 15.0
# Paired seeds per batch, the same decision. Three is the project's floor for a
# quoted number (#13) and the smallest batch that adds a seed to every arm.
BATCH_SEEDS = 3
# #39's body sweep: 24 runs in 8.2 GPU-hours, about 20.5 minutes a run at #25's
# budget. The unit every hour figure below is quoted in, and an assumption the
# moment a knob moves (section 8, rule 5).
HOURS_PER_RUN = 8.2 / 24
# A seed runs at both dose levels, so it costs two runs an arm; three arms is the
# stakes dial as #39 ran it. Both are flags because neither is a law.
RUNS_PER_SEED = 2
ARMS = 3
# Sequences drawn per look table. The boundary is a quantile of the maximum over
# looks, so its own Monte Carlo error is what this buys down: at 40k draws the
# 95th percentile of |t| lands within about 0.01 of itself across seeds, which is
# a hundredth of the boundary and nothing a decision turns on.
DEFAULT_DRAWS = 40_000
# Random splits per null calibration. Same reasoning, one order cheaper.
DEFAULT_SPLITS = 2_000


def paired_t_power(dz: float, pairs: int, alpha: float = ALPHA) -> float:
    """Power of a two-sided paired t-test at ``pairs`` pairs and effect size ``dz``.

    The alternative's statistic is non-central t with ``pairs - 1`` degrees of
    freedom and non-centrality ``dz * sqrt(pairs)``; the normal approximation
    drops the degrees of freedom and reads 10 where this reads 13.
    """
    if pairs < 2:
        return 0.0
    df = pairs - 1
    ncp = abs(dz) * math.sqrt(pairs)
    critical = float(student_t.ppf(1 - alpha / 2, df))
    value = float(1 - nct.cdf(critical, df, ncp) + nct.cdf(-critical, df, ncp))
    if math.isnan(value):
        # scipy's non-central t is a series that loses its footing at a large
        # non-centrality -- far from anywhere an answer lies, since ncp is about
        # 2.8 at any solution whatever dz is. A NaN there is the evaluation being
        # out of range, not a power of nothing, and the normal form is exact
        # enough at the degrees of freedom that produce it.
        return float(norm.cdf(ncp - critical) + norm.cdf(-ncp - critical))
    return value


def pairs_for_power(
    dz: float, power: float = TARGET_POWER, alpha: float = ALPHA, maximum: int = 100_000
) -> int:
    """The smallest pair count reaching ``power``; ``maximum`` when nothing under it does.

    Walked from the normal approximation rather than bisected: the approximation
    always under-reads (it drops the degrees of freedom), so it is a lower bound
    a few steps below the answer, and the walk keeps every evaluation in the
    range where the non-central t is well conditioned. A bisection reads the
    power at 50000 pairs on the way, where it is a NaN.
    """
    if dz == 0.0:
        return maximum
    pairs = max(2, int(normal_approximation(dz, power, alpha)))
    if pairs > maximum:
        return maximum
    while pairs > 2 and paired_t_power(dz, pairs - 1, alpha) >= power:
        pairs -= 1
    while pairs < maximum and paired_t_power(dz, pairs, alpha) < power:
        pairs += 1
    return pairs


def normal_approximation(dz: float, power: float = TARGET_POWER, alpha: float = ALPHA) -> float:
    """``((z_{1-a/2} + z_power) / dz)^2`` -- printed only to show what it costs."""
    if dz == 0.0:
        return float("inf")
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 / dz**2)


@dataclass(frozen=True)
class Effect:
    """A paired contrast as the power calculation sees it."""

    label: str
    deltas: dict[str, float]
    rho: float | None

    @property
    def mean(self) -> float:
        return statistics.fmean(self.deltas.values())

    @property
    def sd(self) -> float:
        return statistics.stdev(self.deltas.values())

    @property
    def dz(self) -> float:
        return self.mean / self.sd if self.sd > 0 else 0.0


def paired_effect(label: str, values_a: dict[str, float], values_b: dict[str, float]) -> Effect:
    """``a - b`` per shared seed, with the seed-level correlation between the arms.

    The correlation is the pairing's own measurement and not a diagnostic: under
    Sharma 2025 a paired design of ``r`` pairs carries the precision of
    ``r / (1 - rho)`` unpaired ones, so an arm pair that does not correlate has
    been paired for nothing and its dz will say so on its own.
    """
    seeds = sorted(set(values_a) & set(values_b), key=str)
    if len(seeds) < 2:
        raise ValueError(f"{label}: a paired effect size needs at least two shared seeds")
    deltas = {seed: values_a[seed] - values_b[seed] for seed in seeds}
    rho = None
    if len(seeds) >= 3:
        column_a = [values_a[seed] for seed in seeds]
        column_b = [values_b[seed] for seed in seeds]
        if statistics.stdev(column_a) > 0 and statistics.stdev(column_b) > 0:
            rho = statistics.correlation(column_a, column_b)
    return Effect(label, deltas, rho)


def load_shift(path: Path) -> dict[str, float]:
    """The per-seed values in a file ``allocation_shift.py --json`` wrote.

    Its ``contrast`` block is the shift per seed; a bare ``{seed: value}`` object
    is taken as-is, which is what a readout that is not the allocation shift
    writes while #57 is still choosing one.
    """
    record = json.loads(path.read_text())
    block = record.get("contrast", record) if isinstance(record, dict) else record
    if not isinstance(block, dict) or not block:
        raise ValueError(f"{path}: no per-seed values to read")
    return {str(seed): float(value) for seed, value in block.items()}


@dataclass(frozen=True)
class Budget:
    """What a seed count costs, and whether the ceiling allows it."""

    seeds: int
    runs_per_seed: int
    arms: int
    hours_per_run: float
    ceiling: float

    @property
    def runs_per_arm(self) -> int:
        return self.seeds * self.runs_per_seed

    @property
    def runs(self) -> int:
        return self.runs_per_arm * self.arms

    @property
    def hours(self) -> float:
        return self.runs * self.hours_per_run

    @property
    def fits(self) -> bool:
        return self.hours <= self.ceiling

    @property
    def seeds_affordable(self) -> int:
        per_seed = self.runs_per_seed * self.arms * self.hours_per_run
        return int(self.ceiling // per_seed) if per_seed > 0 else 0


def sequential_boundary(looks: list[int], alpha: float, draws: int, seed: int) -> float:
    """The constant |t| a group-sequential plan may stop on, family-wise error ``alpha``.

    AdaStop's shape (Mathieu et al. 2023) -- seeds in batches, a stopping rule at
    every batch, a preregistered maximum, and the error controlled over the whole
    sequence of looks rather than at each one -- with the boundary found for the
    paired t by simulation rather than by permutation. The reason is the seed
    count this project works at: a paired permutation test is a sign flip, so at
    three seeds there are eight arrangements and the smallest two-sided p it can
    produce is 0.25. Six is where a permutation test can first reject at all,
    which is the same count ``compare_runs.MIN_PAIRS_FOR_COVERAGE`` was set at
    for the bootstrap's coverage, arrived at from the other direction.

    Under the null the paired differences are exchangeable draws of mean zero;
    the statistic is scale-free, so standard normals carry no assumption beyond
    that. The boundary is the ``1 - alpha`` quantile of the maximum |t| over the
    looks, which is Pocock's constant form: one number every look is read
    against, at the price of a higher bar than a single final test.
    """
    if not looks:
        raise ValueError("a sequential plan needs at least one look")
    generator = np.random.default_rng(seed)
    sample = generator.standard_normal((draws, max(looks)))
    running = np.maximum.accumulate(np.abs(_look_statistics(sample, looks)), axis=1)
    return float(np.quantile(running[:, -1], 1 - alpha))


def _look_statistics(sample: np.ndarray, looks: list[int]) -> np.ndarray:
    """The paired t statistic of each drawn sequence at each look."""
    columns = []
    for count in looks:
        window = sample[:, :count]
        spread = window.std(axis=1, ddof=1)
        spread = np.where(spread == 0.0, np.inf, spread)
        columns.append(window.mean(axis=1) / (spread / math.sqrt(count)))
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class Plan:
    """A sequential design: where it looks, what it stops on, what it expects to spend."""

    looks: list[int]
    boundary: float
    stop_by: list[float]
    power: float
    expected_seeds: float
    draws: int
    seed: int


def sequential_plan(dz: float, looks: list[int], alpha: float, draws: int, seed: int) -> Plan:
    """Simulate the design at ``dz``: its power, where it stops, and what that costs.

    The same generator seed draws the null and the alternative, so a plan is
    reproducible from the numbers printed beside it and a reader can rerun it.
    """
    boundary = sequential_boundary(looks, alpha, draws, seed)
    generator = np.random.default_rng(seed + 1)
    sample = generator.standard_normal((draws, max(looks))) + dz
    statistics_by_look = np.abs(_look_statistics(sample, looks))
    crossed = statistics_by_look > boundary
    first = np.where(crossed.any(axis=1), crossed.argmax(axis=1), len(looks) - 1)
    stop_by = [float(crossed[:, : index + 1].any(axis=1).mean()) for index in range(len(looks))]
    spent = np.array(looks)[first]
    return Plan(
        looks=looks,
        boundary=boundary,
        stop_by=stop_by,
        power=float(crossed.any(axis=1).mean()),
        expected_seeds=float(spent.mean()),
        draws=draws,
        seed=seed,
    )


def batch_looks(maximum: int, batch: int) -> list[int]:
    """Cumulative seed counts at every batch boundary, the last one the maximum."""
    if maximum < 2:
        return []
    looks = [count for count in range(batch, maximum + 1, batch) if count >= 2]
    if not looks or looks[-1] != maximum:
        looks.append(maximum)
    return looks


def null_calibration(
    values: list[float], pairs: int, splits: int, alpha: float, seed: int, resamples: int
) -> dict[str, float]:
    """How often a readout calls a difference between two halves of one arm.

    Colas, Sigaud & Oudeyer 2018 compared an algorithm with itself at five seeds
    and got a false positive; this measures that rate for the readout actually in
    use rather than assuming the nominal one. ``values`` are 2N readings of one
    arm, split at random into two pseudo-arms of ``pairs`` each and paired in the
    drawn order -- there is no seed to pair on inside one arm, and under the null
    any pairing is exchangeable, so the type-I rate is the test's own. What the
    random pairing does not carry is the real design's ``rho``: it calibrates the
    error, never the power.

    Both readouts this project decides with are calibrated: the percentile
    bootstrap interval excluding zero (``compare_runs``, #35) and the paired t.
    """
    if len(values) < 2 * pairs:
        raise ValueError(f"a split into two halves of {pairs} needs {2 * pairs} readings")
    generator = np.random.default_rng(seed)
    critical = float(student_t.ppf(1 - alpha / 2, pairs - 1))
    bootstrap_calls = 0
    t_calls = 0
    for split in range(splits):
        drawn = generator.permutation(len(values))[: 2 * pairs]
        halves = zip(drawn[:pairs], drawn[pairs:], strict=True)
        deltas = [values[int(a)] - values[int(b)] for a, b in halves]
        _, low, high = bootstrap_mean(deltas, resamples, split)
        bootstrap_calls += int(low > 0 or high < 0)
        spread = statistics.stdev(deltas)
        if spread > 0:
            t_calls += int(abs(statistics.fmean(deltas)) / (spread / math.sqrt(pairs)) > critical)
    return {
        "pairs": float(pairs),
        "splits": float(splits),
        "bootstrap_false_positive_rate": bootstrap_calls / splits,
        "paired_t_false_positive_rate": t_calls / splits,
    }


def format_effect(effect: Effect) -> str:
    per_seed = "  ".join(f"s{seed}={value:+.4f}" for seed, value in effect.deltas.items())
    lines = [
        f"{effect.label}: {len(effect.deltas)} paired seeds",
        f"  mean {effect.mean:+.5f}   sample sd {effect.sd:.5f}   dz {effect.dz:+.4f}",
        f"  per seed  {per_seed}",
    ]
    if effect.rho is not None:
        gain = float("inf") if effect.rho >= 1 else len(effect.deltas) / (1 - effect.rho)
        lines.append(
            f"  rho {effect.rho:+.4f} between the arms across seeds; Sharma 2025 puts the "
            f"pairing's gain at 1/(1-rho), so these {len(effect.deltas)} paired seeds carry "
            f"the precision of {gain:.1f} unpaired ones"
        )
    else:
        lines.append("  rho: not read (fewer than three seeds, or an arm with no spread)")
    return "\n".join(lines)


def format_requirement(dz: float, budget: Budget, power: float, alpha: float) -> str:
    pairs = budget.seeds
    lines = [
        f"dz {dz:+.4f} at {power:.0%} power, alpha {alpha} two-sided, paired t (scipy.stats.nct)",
        f"  paired seeds a side      {pairs}"
        + (f"   (power {paired_t_power(dz, pairs, alpha):.3f}" if pairs < 100_000 else ""),
    ]
    if pairs < 100_000:
        lines[-1] += f", {paired_t_power(dz, pairs - 1, alpha):.3f} at {pairs - 1})"
        lines.append(
            f"  the normal approximation {normal_approximation(dz, power, alpha):.1f}"
            "   -- what it would have under-read by"
        )
    else:
        lines.append("  no pair count under 100000 reaches this power; the readout is the problem")
    lines += [
        f"  runs an arm              {budget.runs_per_arm}   ({budget.runs_per_seed} runs a seed)",
        f"  runs in the sweep        {budget.runs}   ({budget.arms} arms)",
        f"  GPU-hours                {budget.hours:.1f}"
        f"   (at {budget.hours_per_run * 60:.1f} min a run)",
    ]
    return "\n".join(lines)


def format_ceiling(budget: Budget, over_ceiling: str | None) -> str:
    if budget.fits:
        return (
            f"ceiling: {budget.hours:.1f} GPU-h against {budget.ceiling:.0f} -- inside it. "
            "The ceiling is a dead band: stop when the sign is settled, not at the ceiling."
        )
    lines = [
        f"ceiling: {budget.hours:.1f} GPU-h against {budget.ceiling:.0f} -- OVER, by "
        f"{budget.hours / budget.ceiling:.0f}x. The ceiling affords {budget.seeds_affordable} "
        f"paired seeds a side at these knobs.",
        "  A design over the ceiling is a design problem, sent back to the fixture (#57, "
        "#48) -- never a budget request, and never bought by shrinking a fingerprint knob "
        "(section 8, rule 5: that invalidates the dose unit and the recorded floor).",
    ]
    if over_ceiling is None:
        lines.append(
            "  REFUSED: no batch plan is printed. Pass --over-ceiling '<the preregistration "
            "row that raises the ceiling>' to plan anyway."
        )
    else:
        lines.append(f"  planned over the ceiling on: {over_ceiling}")
    return "\n".join(lines)


def format_plan(plan: Plan, budget: Budget, batch: int, alpha: float) -> str:
    single_look = float(student_t.ppf(1 - alpha / 2, plan.looks[-1] - 1))
    per_seed_hours = budget.runs_per_seed * budget.arms * budget.hours_per_run
    lines = [
        f"sequential plan: batches of {batch} paired seeds, maximum {plan.looks[-1]}, "
        f"family-wise alpha {alpha} over {len(plan.looks)} looks",
        f"  stop when |t| > {plan.boundary:.3f} at any look "
        f"({plan.draws} draws, generator seed {plan.seed}; a single final test would "
        f"read {single_look:.3f})",
        f"  {'look':>6}{'seeds':>7}{'runs':>7}{'GPU-h':>8}{'P(stopped by here)':>21}",
    ]
    for index, (count, cumulative) in enumerate(zip(plan.looks, plan.stop_by, strict=True), 1):
        lines.append(
            f"  {index:>6}{count:>7}{count * budget.runs_per_seed * budget.arms:>7}"
            f"{count * per_seed_hours:>8.1f}{cumulative:>21.3f}"
        )
    lines.append(
        f"  power over the whole sequence {plan.power:.3f}; expected spend "
        f"{plan.expected_seeds:.1f} seeds, {plan.expected_seeds * per_seed_hours:.1f} GPU-h"
    )
    if plan.power < TARGET_POWER:
        lines.append(
            f"  this plan does not reach {TARGET_POWER:.0%}: the maximum is the binding "
            "constraint, not the batch size. The readout or the substrate is what moves it."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dz", type=float, default=None, help="A paired effect size to price")
    parser.add_argument(
        "--shifts",
        type=Path,
        nargs=2,
        default=None,
        metavar=("A.json", "B.json"),
        help="Two allocation_shift.py --json files; the effect is A minus B, paired by seed",
    )
    parser.add_argument(
        "--null-calibrate",
        type=Path,
        default=None,
        metavar="ONE_ARM.json",
        help="2N readings of one arm; the readout's false-positive rate over random splits",
    )
    parser.add_argument(
        "--discount",
        type=float,
        nargs=2,
        default=None,
        metavar=("FIXTURE_DZ", "BODY_DZ"),
        help="One readout's dz on both substrates; prints the ratio as a stated assumption",
    )
    parser.add_argument("--plan", action="store_true", help="Also print the sequential design")
    parser.add_argument("--power", type=float, default=TARGET_POWER)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--batch", type=int, default=BATCH_SEEDS, help="Paired seeds per batch")
    parser.add_argument(
        "--max-seeds", type=int, default=None, help="Default: what fits the ceiling"
    )
    parser.add_argument("--arms", type=int, default=ARMS)
    parser.add_argument("--runs-per-seed", type=int, default=RUNS_PER_SEED)
    parser.add_argument("--hours-per-run", type=float, default=HOURS_PER_RUN)
    parser.add_argument("--ceiling", type=float, default=CEILING_GPU_HOURS, help="GPU-hours")
    parser.add_argument(
        "--over-ceiling",
        type=str,
        default=None,
        metavar="REASON",
        help="The preregistration row that raises the ceiling; without it a plan is refused",
    )
    parser.add_argument("--pairs", type=int, default=BATCH_SEEDS, help="--null-calibrate half size")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--resamples", type=int, default=2_000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="The simulation's own RNG seed")
    parser.add_argument("--json", type=Path, default=None, help="Also write the numbers here")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dz is None and args.shifts is None and args.null_calibrate is None:
        build_parser().error("nothing to compute: pass --dz, --shifts or --null-calibrate")
    record: dict[str, object] = {}
    blocks: list[str] = []

    dz = args.dz
    if args.shifts is not None:
        effect = paired_effect(
            f"{args.shifts[0].stem} minus {args.shifts[1].stem}",
            load_shift(args.shifts[0]),
            load_shift(args.shifts[1]),
        )
        blocks.append(format_effect(effect))
        record["effect"] = {
            "label": effect.label,
            "deltas": effect.deltas,
            "mean": effect.mean,
            "sd": effect.sd,
            "dz": effect.dz,
            "rho": effect.rho,
        }
        if len(effect.deltas) < MIN_PAIRS_FOR_COVERAGE:
            blocks.append(
                f"  measured on {len(effect.deltas)} seeds: dz is itself an estimate with a "
                f"wide interval below {MIN_PAIRS_FOR_COVERAGE} pairs, and the seed count it "
                "implies inherits that. Plan the first batch on it, never the whole sweep."
            )
        dz = dz if dz is not None else effect.dz

    if dz is not None:
        pairs = pairs_for_power(dz, args.power, args.alpha)
        budget = Budget(pairs, args.runs_per_seed, args.arms, args.hours_per_run, args.ceiling)
        blocks.append(format_requirement(dz, budget, args.power, args.alpha))
        blocks.append(format_ceiling(budget, args.over_ceiling))
        record["requirement"] = {
            "dz": dz,
            "power": args.power,
            "alpha": args.alpha,
            "paired_seeds": pairs,
            "runs_per_arm": budget.runs_per_arm,
            "runs": budget.runs,
            "gpu_hours": budget.hours,
            "normal_approximation": normal_approximation(dz, args.power, args.alpha),
            "ceiling_gpu_hours": args.ceiling,
            "fits_ceiling": budget.fits,
        }
        if args.plan and (budget.fits or args.over_ceiling is not None):
            maximum = args.max_seeds or min(pairs, max(budget.seeds_affordable, args.batch))
            looks = batch_looks(maximum, args.batch)
            if looks:
                plan = sequential_plan(dz, looks, args.alpha, args.draws, args.seed)
                blocks.append(format_plan(plan, budget, args.batch, args.alpha))
                record["plan"] = {
                    "looks": plan.looks,
                    "boundary": plan.boundary,
                    "stop_by": plan.stop_by,
                    "power": plan.power,
                    "expected_seeds": plan.expected_seeds,
                    "draws": plan.draws,
                    "seed": plan.seed,
                }

    if args.discount is not None:
        fixture_dz, body_dz = args.discount
        ratio = body_dz / fixture_dz if fixture_dz else float("inf")
        blocks.append(
            f"fixture-to-body discount: dz {body_dz:+.4f} on the body over {fixture_dz:+.4f} on "
            f"the fixture = {ratio:+.2f}x. **An assumption, not a measurement** (section 8, "
            "rule 3): it is one readout's ratio on one pair of substrates, and it is written "
            "beside a power row so the reader can see what was assumed, never folded into it."
        )
        record["discount"] = {"fixture_dz": fixture_dz, "body_dz": body_dz, "ratio": ratio}

    if args.null_calibrate is not None:
        values = list(load_shift(args.null_calibrate).values())
        calibration = null_calibration(
            values, args.pairs, args.splits, args.alpha, args.seed, args.resamples
        )
        lines = [
            f"null calibration on {len(values)} readings of one arm, split into two halves of "
            f"{args.pairs} over {args.splits} random splits:",
            "  percentile bootstrap excludes zero  "
            f"{calibration['bootstrap_false_positive_rate']:.3f}",
            f"  paired t at alpha {args.alpha}            "
            f"{calibration['paired_t_false_positive_rate']:.3f}",
            "  a rate above alpha is the readout's, not the arm's: the two halves are one arm.",
        ]
        if args.pairs < MIN_PAIRS_FOR_COVERAGE:
            lines.append(
                f"  at {args.pairs} pairs the percentile interval is the sample range "
                f"(compare_runs, #35), so it excludes zero exactly when all {args.pairs} "
                f"deltas share a sign: {2 * 0.5**args.pairs:.3f} under any symmetric null, "
                "whatever the readout. That is the number above, and it is arithmetic rather "
                "than a property of this arm."
            )
        blocks.append("\n".join(lines))
        record["null_calibration"] = calibration

    print("\n\n".join(blocks))
    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"\nrecord: {args.json}")


if __name__ == "__main__":
    main()
