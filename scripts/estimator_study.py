"""Which readout resolves signature 1 with fewer runs? The fixture study (#57).

#39's body sweep resolved ``value`` minus ``decoupled`` at dz 0.867 and the
contrast that isolates self-reference, ``value`` minus ``shuffled``, at dz 0.151
-- 348 paired seeds, about 700 GPU-hours, which is not a sweep anyone will run.
The readout is where that is fixed, if it can be fixed: four end-of-training win
shares from a 2000-step trajectory is a thin instrument, and this measures three
candidates against it on the fixture, where a run costs three CPU-seconds.

**Exploratory, and closed by a confirmatory stage.** The candidates may move as
they are tried; the chosen one is then validated on a planted effect it was not
developed on -- a different dose ratio and different seeds -- and only that
validation enters the preregistration. #39's 24 body directories are not read
here at all: an estimator chosen after a run's sign was known is the next run's
instrument, never that run's (section 8, rule 2).

**What "resolves" means, fixed here before the runs.** The planted effect is the
dose ratio change, whose allocation shift the fixture's closed form gives. A
readout's resolution is ``dz = mean / sample sd`` of its per-seed shift in the
``value`` arm, and the number it buys is the paired seeds a side that dz needs
for 80% power under a paired t (``scripts/power.py``). A candidate is adopted
only if it at least halves the count the recorded readout needs on the same
runs. The two arm contrasts are reported beside it, because the contrast that
has to become powerable is ``value`` minus ``shuffled`` and a readout that
sharpens the planted effect without sharpening that contrast has not bought #44
anything.

    uv run python scripts/estimator_study.py --stage all --out ~/tame-runs/57-estimator
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from allocation_shift import (  # noqa: E402
    DEFAULT_READOUT,
    READING_PREFIX,
    READOUTS,
    STEP_POST_PREFIX,
    STEP_PRE_PREFIX,
    Readout,
    reading_steps,
    total_variation,
)
from compare_runs import MIN_PAIRS_FOR_COVERAGE, bootstrap_mean  # noqa: E402
from measure_stakes_dial import (  # noqa: E402
    ARMS,
    GOAL_TYPES,
    READING_WINDOW,
    STEPS,
    WEALTH_HORIZON,
    build_differentiated,
    run_differentiated,
    share_columns,
)
from power import pairs_for_power  # noqa: E402

from mob import PERSISTENCE_DECOUPLED, PERSISTENCE_SHUFFLED, PERSISTENCE_VALUE  # noqa: E402
from parity import code_identity  # noqa: E402

STAGES = ("dose", "setpoint", "ablation", "null", "validate", "stability")
# Six, not three: below MIN_PAIRS_FOR_COVERAGE a dz is an estimate whose own
# interval is the sample range, and a readout chosen on three seeds would be
# chosen on the same thinness the readout is being replaced for. A fixture run
# is three CPU-seconds, so the seed count is free here in a way it never is on
# the body.
EXPLORATION_SEEDS = tuple(range(6))
EXPLORATION_RATIO = 4.0
# The confirmatory stage: a planted effect the candidates were not tried on.
VALIDATION_SEEDS = tuple(range(10, 16))
VALIDATION_RATIO = 2.0
BALANCED_RATIO = 1.0
# The between-group candidates, read on one set of runs. ``setpoint-step`` is a
# protocol rather than a statistic and has its own stage.
DOSE_READOUTS = ("total-variation", "token-conditioned", "integrated")
# The ledger mechanics the ablation switches off one at a time, each as the
# ``MoBConfig`` override that neutralises it without touching the others.
MECHANICS: dict[str, dict[str, float]] = {
    "as recorded": {},
    "no clamp": {"min_wealth": 1e-3, "max_wealth": 1e9},
    "no decay": {"wealth_decay": 1.0},
    "no exploration": {"exploration_rate": 0.0},
    "no price": {"payment_scale": 0.0},
}
# The step-and-recover protocol. One wealth memory horizon to settle, the step,
# one more to recover: the horizon is the washout an n-of-1 design needs, and
# the fixture's floor being zero is what lets a single run carry a contrast.
SETTLE_STEPS = WEALTH_HORIZON
RECOVER_STEPS = WEALTH_HORIZON
# Time constants the recovery curve is fitted over, in steps. The grid is coarse
# because the discriminator is about the asymptote and the fit's shape, not
# about the third digit of a time constant.
TAU_GRID = tuple(range(5, 305, 5))


def seed_key(seed: int) -> str:
    return str(seed)


@dataclass(frozen=True)
class Resolution:
    """What one readout costs on one contrast: the effect size and the seeds it needs."""

    readout: str
    contrast: str
    per_seed: dict[str, float]
    dz: float
    pairs: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "readout": self.readout,
            "contrast": self.contrast,
            "per_seed": self.per_seed,
            "dz": self.dz,
            "paired_seeds_at_80": self.pairs,
        }


def resolution(readout: str, contrast: str, per_seed: dict[str, float]) -> Resolution:
    values = list(per_seed.values())
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    dz = statistics.fmean(values) / spread if spread > 0 else 0.0
    return Resolution(readout, contrast, per_seed, dz, pairs_for_power(dz))


def format_resolutions(rows: list[Resolution], baseline: str) -> str:
    """One line a readout, with what it buys against the recorded one."""
    reference = {row.contrast: row for row in rows if row.readout == baseline}
    lines = [f"  {'readout':<20}{'contrast':<22}{'dz':>8}{'seeds@80%':>11}{'vs recorded':>13}"]
    for row in rows:
        against = reference.get(row.contrast)
        ratio = ""
        if against is not None and row.readout != baseline and against.pairs:
            ratio = f"{row.pairs / against.pairs:.2f}x"
        lines.append(
            f"  {row.readout:<20}{row.contrast:<22}{row.dz:>+8.3f}{row.pairs:>11}{ratio:>13}"
        )
    return "\n".join(lines)


def dose_groups(
    arms: tuple[str, ...],
    ratios: tuple[float, ...],
    seeds: tuple[int, ...],
    steps: int,
    **overrides,
) -> dict[str, dict[float, dict[str, dict[str, float]]]]:
    """Every arm at every ratio over every seed: the per-seed metric records."""
    groups: dict[str, dict[float, dict[str, dict[str, float]]]] = {}
    for arm in arms:
        groups[arm] = {}
        for ratio in ratios:
            groups[arm][ratio] = {}
            for seed in seeds:
                economy_metrics, _ = (
                    run_differentiated(arm, ratio, seed, steps)
                    if not overrides
                    else run_ablated(arm, ratio, seed, steps, **overrides)
                )
                groups[arm][ratio][seed_key(seed)] = economy_metrics
            print(f"  ran {arm:<10} ratio {ratio:<4} seeds {list(seeds)}", flush=True)
    return groups


def run_ablated(
    arm: str, ratio: float, seed: int, steps: int, **overrides
) -> tuple[dict[str, float], None]:
    """``run_differentiated`` with one ledger mechanic switched off (#57's ablation).

    A copy of the accumulation rather than a flag on the original: the recorded
    function stays the one that reproduces #39's rows bit for bit, and nothing
    an ablation needs can reach it.
    """
    economy = build_differentiated(arm, ratio, seed, **overrides)
    wins = torch.zeros(economy.config.num_experts)
    for step in range(steps):
        record = economy.step()
        if step >= steps - 100:
            wins += torch.bincount(
                record.selected_experts.flatten(), minlength=economy.config.num_experts
            ).float()
    return share_columns(wins, "routing/win_share_e"), None


def shifts_per_seed(
    groups: dict[float, dict[str, dict[str, float]]], readout: Readout, ratio: float
) -> dict[str, float]:
    """One arm's per-seed shift between the balanced ratio and ``ratio``."""
    balanced, dosed = groups[BALANCED_RATIO], groups[ratio]
    return {
        seed: readout.reading(balanced[seed], dosed[seed])
        for seed in sorted(set(balanced) & set(dosed), key=str)
    }


def stage_dose(seeds: tuple[int, ...], ratio: float, steps: int) -> dict[str, Any]:
    """The candidates that are statistics over the recorded protocol, on one set of runs."""
    print(f"\n== dose protocol: {len(ARMS)} arms x 2 ratios x {len(seeds)} seeds ==")
    groups = dose_groups(ARMS, (BALANCED_RATIO, ratio), seeds, steps)
    rows: list[Resolution] = []
    record: dict[str, Any] = {"seeds": list(seeds), "ratio": ratio, "shifts": {}}
    for name in DOSE_READOUTS:
        readout = READOUTS[name]
        shifts = {arm: shifts_per_seed(groups[arm], readout, ratio) for arm in ARMS}
        record["shifts"][name] = shifts
        rows.extend(contrast_rows(name, shifts))
    print(format_resolutions(rows, DEFAULT_READOUT.name))
    record["resolutions"] = [row.as_dict() for row in rows]
    record["reading_correlation"] = reading_correlation(groups, ratio)
    print(
        f"\n  successive readings correlate at rho = {record['reading_correlation']:.3f} within "
        "a run (Frison & Pocock: the mean of k readings has variance "
        "sigma^2 (1 + (k-1) rho) / k, so this is what the integrated candidate's gain is "
        "bounded by)"
    )
    return record


def reading_correlation(
    groups: dict[str, dict[float, dict[str, dict[str, float]]]], ratio: float
) -> float:
    """Correlation between successive logged readings of the shift, pooled over seeds.

    What bounds the integrated candidate: readings a run apart are not
    independent samples of the allocation, and at rho = 1 averaging six of them
    buys exactly nothing.
    """
    pairs: list[tuple[float, float]] = []
    for arm in groups:
        balanced, dosed = groups[arm][BALANCED_RATIO], groups[arm][ratio]
        for seed in sorted(set(balanced) & set(dosed), key=str):
            steps = [
                step
                for step in reading_steps(balanced[seed])
                if step in set(reading_steps(dosed[seed]))
            ]
            series = [
                total_variation(balanced[seed], dosed[seed], f"{READING_PREFIX}{step}_e")
                for step in steps
            ]
            pairs.extend(zip(series, series[1:], strict=False))
    if len(pairs) < 3:
        return float("nan")
    first, second = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    if statistics.stdev(first) == 0 or statistics.stdev(second) == 0:
        return float("nan")
    return statistics.correlation(first, second)


def run_setpoint_step(arm: str, seed: int, multiplier: float = 1.0) -> dict[str, float]:
    """One run, settled, stepped on the swept field's setpoint, and settled again.

    The step is one *resting spread* of the tissue's own reading of that field,
    measured in the settle window of this run rather than assumed: an error in
    the regulated variable, in the cells' units, the size the loop already lives
    with. Nothing about the dose moves, which is the point -- a thermostat is
    tested by moving the target, not the electricity price.
    """
    economy = build_differentiated(arm, BALANCED_RATIO, seed)
    swept = next(field for field in economy.goal_fields() if field.expert_type == GOAL_TYPES[0])
    pre_wins = torch.zeros(economy.config.num_experts)
    resting: list[float] = []
    for step in range(SETTLE_STEPS):
        record = economy.step()
        if step >= SETTLE_STEPS - READING_WINDOW:
            pre_wins += torch.bincount(
                record.selected_experts.flatten(), minlength=economy.config.num_experts
            ).float()
            resting.append(economy.goal_reading(swept, record.selected_experts))
    spread = statistics.stdev(resting)
    stepped = economy.step_goal_setpoint(GOAL_TYPES[0], multiplier * spread)

    post_wins = torch.zeros(economy.config.num_experts)
    errors: list[float] = []
    for step in range(RECOVER_STEPS):
        record = economy.step()
        errors.append(stepped.setpoint - economy.goal_reading(stepped, record.selected_experts))
        if step >= RECOVER_STEPS - READING_WINDOW:
            post_wins += torch.bincount(
                record.selected_experts.flatten(), minlength=economy.config.num_experts
            ).float()
    metrics = {
        **share_columns(pre_wins, STEP_PRE_PREFIX),
        **share_columns(post_wins, STEP_POST_PREFIX),
        "signature1/resting_spread": spread,
        "signature1/setpoint_step": multiplier * spread,
        "signature1/step_multiplier": multiplier,
        "signature1/setpoint_after": stepped.setpoint,
    }
    asymptote, tau, residual = fit_first_order_lag(errors)
    metrics.update(
        {
            "signature1/step_asymptote": asymptote,
            "signature1/step_tau": tau,
            "signature1/step_residual": residual,
        }
    )
    metrics["signature1/step_errors"] = float(len(errors))
    return metrics | {"_errors": errors}  # type: ignore[dict-item]


def fit_first_order_lag(curve: list[float]) -> tuple[float, float, float]:
    """Least-squares ``a + b exp(-t / tau)`` over a tau grid; asymptote, tau, RMS residual.

    The discriminator the pre-mortem asked for needs a *single* first-order lag
    fitted honestly, because the claim it guards against is "a step response any
    one-lag system would have produced". Linear in ``a`` and ``b`` once tau is
    fixed, so the grid is the whole search.
    """
    best = (float("nan"), float("nan"), float("inf"))
    times = list(range(len(curve)))
    for tau in TAU_GRID:
        basis = [math.exp(-t / tau) for t in times]
        mean_basis = statistics.fmean(basis)
        mean_curve = statistics.fmean(curve)
        denominator = sum((value - mean_basis) ** 2 for value in basis)
        if denominator == 0:
            continue
        slope = (
            sum((b - mean_basis) * (y - mean_curve) for b, y in zip(basis, curve, strict=True))
            / denominator
        )
        intercept = mean_curve - slope * mean_basis
        residual = math.sqrt(
            sum((y - (intercept + slope * b)) ** 2 for b, y in zip(basis, curve, strict=True))
            / len(curve)
        )
        if residual < best[2]:
            best = (intercept, float(tau), residual)
    return best


def imposed_lag_residual(curve: list[float], asymptote: float, tau: float) -> float:
    """RMS of a curve against another arm's fitted lag, amplitude refitted.

    The second half of the discriminator: if the treatment's response is only
    the control's lag at another amplitude, this is as small as its own fit's
    residual, and the step measured a transient rather than nesting.
    """
    basis = [math.exp(-t / tau) for t in range(len(curve))]
    centred = [value - asymptote for value in curve]
    denominator = sum(value**2 for value in basis)
    if denominator == 0:
        return float("nan")
    scale = sum(b * y for b, y in zip(basis, centred, strict=True)) / denominator
    return math.sqrt(
        sum((y - scale * b) ** 2 for b, y in zip(basis, centred, strict=True)) / len(curve)
    )


def stage_setpoint(
    seeds: tuple[int, ...], resamples: int, multiplier: float = 1.0
) -> dict[str, Any]:
    """Candidate 1: the same statistic, read across a designed step inside one run."""
    print(
        f"\n== setpoint-step protocol: {len(ARMS)} arms x {len(seeds)} seeds, "
        f"step {multiplier} resting spreads =="
    )
    runs: dict[str, dict[str, dict[str, float]]] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    for arm in ARMS:
        runs[arm], curves[arm] = {}, {}
        for seed in seeds:
            metrics = run_setpoint_step(arm, seed, multiplier)
            curves[arm][seed_key(seed)] = metrics.pop("_errors")  # type: ignore[arg-type]
            runs[arm][seed_key(seed)] = metrics
        print(f"  ran {arm:<10} seeds {list(seeds)}", flush=True)

    readout = READOUTS["setpoint-step"]
    shifts = {
        arm: {seed: readout.reading(metrics) for seed, metrics in runs[arm].items()} for arm in ARMS
    }
    rows = contrast_rows("setpoint-step", shifts)
    print(format_resolutions(rows, "setpoint-step"))

    discriminator = step_discriminator(runs, curves, resamples)
    print(
        "\n  discriminator (preregistered): the treatment's response must differ from the "
        "control's in a way one first-order lag fitted to the control cannot produce."
    )
    for line in discriminator["lines"]:
        print(f"  {line}")
    return {
        "seeds": list(seeds),
        "step_multiplier": multiplier,
        "shifts": shifts,
        "resolutions": [row.as_dict() for row in rows],
        "fits": {
            arm: {
                seed: {key: value for key, value in metrics.items() if key.startswith("signature1")}
                for seed, metrics in runs[arm].items()
            }
            for arm in ARMS
        },
        "discriminator": {key: value for key, value in discriminator.items() if key != "lines"},
    }


def step_discriminator(
    runs: dict[str, dict[str, dict[str, float]]],
    curves: dict[str, dict[str, list[float]]],
    resamples: int,
) -> dict[str, Any]:
    """Asymptote difference, and what the control's own lag leaves unexplained."""
    seeds = sorted(runs[PERSISTENCE_VALUE], key=str)
    asymptote_gap = {
        seed: runs[PERSISTENCE_VALUE][seed]["signature1/step_asymptote"]
        - runs[PERSISTENCE_DECOUPLED][seed]["signature1/step_asymptote"]
        for seed in seeds
    }
    own = {seed: runs[PERSISTENCE_VALUE][seed]["signature1/step_residual"] for seed in seeds}
    imposed = {
        seed: imposed_lag_residual(
            curves[PERSISTENCE_VALUE][seed],
            runs[PERSISTENCE_DECOUPLED][seed]["signature1/step_asymptote"],
            runs[PERSISTENCE_DECOUPLED][seed]["signature1/step_tau"],
        )
        for seed in seeds
    }
    excess = {seed: imposed[seed] - own[seed] for seed in seeds}
    lines = []
    for label, values in (
        ("asymptote, value - decoupled", asymptote_gap),
        ("residual under the control's lag, minus its own", excess),
    ):
        centre, low, high = bootstrap_mean(list(values.values()), resamples, 0)
        verdict = "excludes zero" if low > 0 or high < 0 else "includes zero"
        lines.append(f"{label:<48}{centre:>+9.4f}  [{low:+.4f}, {high:+.4f}]  {verdict}")
    lines.append(
        f"{'tau, value / decoupled (steps)':<48}"
        + "  ".join(
            f"s{seed}={runs[PERSISTENCE_VALUE][seed]['signature1/step_tau']:.0f}/"
            f"{runs[PERSISTENCE_DECOUPLED][seed]['signature1/step_tau']:.0f}"
            for seed in seeds
        )
    )
    return {
        "asymptote_gap": asymptote_gap,
        "own_residual": own,
        "imposed_residual": imposed,
        "excess_residual": excess,
        "lines": lines,
    }


def stage_ablation(
    seeds: tuple[int, ...], ratio: float, steps: int, resamples: int
) -> dict[str, Any]:
    """Which ledger mechanic carries the live-versus-pinned separation.

    Shao et al. 2025's warning read onto this economy: a separation with the
    shape of a mechanics effect is one the arms' *bookkeeping* produces, and the
    way to tell is to switch each mechanic off and see which switch removes it.
    """
    print(f"\n== mechanism ablation: {len(MECHANICS)} mechanics x 2 arms x {len(seeds)} seeds ==")
    arms = (PERSISTENCE_VALUE, PERSISTENCE_DECOUPLED)
    readout = DEFAULT_READOUT
    record: dict[str, Any] = {"seeds": list(seeds), "ratio": ratio, "contrasts": {}}
    lines = [f"  {'mechanic':<18}{'value - decoupled':>20}{'range':>26}{'verdict':>16}"]
    for mechanic, overrides in MECHANICS.items():
        groups = dose_groups(arms, (BALANCED_RATIO, ratio), seeds, steps, **overrides)
        shifts = {arm: shifts_per_seed(groups[arm], readout, ratio) for arm in arms}
        contrast = {
            seed: shifts[PERSISTENCE_VALUE][seed] - shifts[PERSISTENCE_DECOUPLED][seed]
            for seed in shifts[PERSISTENCE_VALUE]
        }
        centre, low, high = bootstrap_mean(list(contrast.values()), resamples, 0)
        verdict = "separates" if low > 0 or high < 0 else "no separation"
        lines.append(
            f"  {mechanic:<18}{centre:>+20.4f}{f'[{low:+.4f}, {high:+.4f}]':>26}{verdict:>16}"
        )
        record["contrasts"][mechanic] = {
            "per_seed": contrast,
            "mean": centre,
            "low": low,
            "high": high,
            "overrides": overrides,
        }
    print("\n".join(lines))
    return record


def stage_null(
    seeds: tuple[int, ...], ratio: float, steps: int, splits: int, resamples: int
) -> dict[str, Any]:
    """The planted null every readout must read as zero, and the readout's own error rate."""
    print(f"\n== null: the planted null, then {len(seeds)} same-arm seeds split at random ==")
    groups = dose_groups((PERSISTENCE_VALUE,), (BALANCED_RATIO, ratio), seeds, steps)
    record: dict[str, Any] = {"seeds": list(seeds), "planted_null": {}, "calibration": {}}
    for name in DOSE_READOUTS:
        readout = READOUTS[name]
        balanced = groups[PERSISTENCE_VALUE][BALANCED_RATIO]
        null = {seed: readout.reading(balanced[seed], balanced[seed]) for seed in balanced}
        record["planted_null"][name] = null
        print(f"  {name:<20} planted null (ratio 1 against itself): {max(null.values()):.12f}")
    from power import null_calibration  # noqa: PLC0415 -- the helper #56 built, used once

    for name in DOSE_READOUTS:
        readout = READOUTS[name]
        shifts = shifts_per_seed(groups[PERSISTENCE_VALUE], readout, ratio)
        half = len(shifts) // 2
        calibration = null_calibration(list(shifts.values()), half, splits, 0.05, 0, resamples)
        record["calibration"][name] = calibration
        print(
            f"  {name:<20} split into two halves of {half}: bootstrap calls a difference "
            f"{calibration['bootstrap_false_positive_rate']:.3f} of the time, paired t "
            f"{calibration['paired_t_false_positive_rate']:.3f}"
        )
    return record


def stage_validate(
    chosen: str,
    seeds: tuple[int, ...],
    ratio: float,
    steps: int,
    resamples: int,
    multiplier: float,
) -> dict[str, Any]:
    """The confirmatory close: the chosen readout on an effect it was not developed on.

    The recorded protocol runs at these seeds whatever the candidate is, because
    the adoption rule is a comparison and the baseline has to be measured on the
    same seeds rather than carried over from the exploration. A within-run
    candidate is a change of *protocol* as well as of statistic, so its planted
    effect is a setpoint step at a size the exploration did not use and its
    comparison against the recorded readout is labelled as the protocol
    comparison it is.
    """
    print(
        f"\n== validation: {chosen} against {DEFAULT_READOUT.name}, ratio {ratio}, "
        f"seeds {list(seeds)} -- neither the ratio nor the seeds the candidates were tried on =="
    )
    groups = dose_groups(ARMS, (BALANCED_RATIO, ratio), seeds, steps)
    readout = READOUTS[chosen]
    rows: list[Resolution] = []
    for name in dict.fromkeys((DEFAULT_READOUT.name, *(() if readout.within_run else (chosen,)))):
        shifts = {arm: shifts_per_seed(groups[arm], READOUTS[name], ratio) for arm in ARMS}
        rows.extend(contrast_rows(name, shifts))
    step: dict[str, Any] | None = None
    if readout.within_run:
        step = stage_setpoint(seeds, resamples, multiplier)
        rows.extend(Resolution(**_from_dict(row)) for row in step["resolutions"])
    print(format_resolutions(rows, DEFAULT_READOUT.name))

    chosen_planted, baseline_planted = (
        next(
            (
                row
                for row in rows
                if row.readout == name and row.contrast == "planted effect (value)"
            ),
            None,
        )
        for name in (chosen, DEFAULT_READOUT.name)
    )
    assert baseline_planted is not None and chosen_planted is not None
    halves = chosen_planted.pairs * 2 <= baseline_planted.pairs
    print(
        f"\n  preregistered adoption rule -- at least halves the recorded readout's count on "
        f"the planted effect: {chosen_planted.pairs} against {baseline_planted.pairs} -> "
        f"{'ADOPTED' if halves else 'NOT ADOPTED'}"
        + (
            "\n  (a within-run candidate's planted effect is a setpoint step and the recorded "
            "readout's is a dose change, so this row compares protocols, not statistics)"
            if readout.within_run
            else ""
        )
    )
    self_reference = {
        row.readout: row.pairs for row in rows if row.contrast == f"value - {PERSISTENCE_SHUFFLED}"
    }
    print(
        f"  what #44's primary needs -- value - shuffled at 13 paired seeds a side or fewer: "
        f"{chosen} {self_reference.get(chosen)}, {DEFAULT_READOUT.name} "
        f"{self_reference.get(DEFAULT_READOUT.name)}"
    )
    return {
        "chosen": chosen,
        "seeds": list(seeds),
        "ratio": ratio,
        "step_multiplier": multiplier if readout.within_run else None,
        "resolutions": [row.as_dict() for row in rows],
        "adopted": halves,
        "self_reference_pairs": self_reference,
        "setpoint": step,
    }


def contrast_rows(name: str, shifts: dict[str, dict[str, float]]) -> list[Resolution]:
    """The planted effect and the two arm contrasts, for one readout's per-arm shifts."""
    rows = [resolution(name, "planted effect (value)", shifts[PERSISTENCE_VALUE])]
    for other in (PERSISTENCE_DECOUPLED, PERSISTENCE_SHUFFLED):
        rows.append(
            resolution(
                name,
                f"value - {other}",
                {
                    seed: shifts[PERSISTENCE_VALUE][seed] - shifts[other][seed]
                    for seed in shifts[PERSISTENCE_VALUE]
                },
            )
        )
    return rows


def _from_dict(row: dict[str, Any]) -> dict[str, Any]:
    """A ``Resolution`` back from the shape ``as_dict`` writes."""
    return {
        "readout": row["readout"],
        "contrast": row["contrast"],
        "per_seed": row["per_seed"],
        "dz": row["dz"],
        "pairs": row["paired_seeds_at_80"],
    }


def stage_stability(seeds: tuple[int, ...], ratio: float, steps: int, resamples: int) -> Any:
    """How much of a readout's measured count is the readout, and how much is six seeds?

    Read after the confirmatory stage and reported as the diagnostic it is: it
    decides nothing, because the decision was taken on the design that was
    preregistered. What it answers is why the exploration and the confirmation
    disagreed by two orders of magnitude -- whether a count measured at six
    seeds is a property of the readout at all.
    """
    print(f"\n== stability: the same readouts at {len(seeds)} seeds, and what six seeds say ==")
    groups = dose_groups(ARMS, (BALANCED_RATIO, ratio), seeds, steps)
    generator = random.Random(0)
    record: dict[str, Any] = {"seeds": list(seeds), "ratio": ratio, "readouts": {}}
    lines = [f"  {'readout':<20}{'contrast':<22}{'dz':>8}{'seeds@80%':>11}{'six-seed range':>22}"]
    for name in DOSE_READOUTS:
        shifts = {arm: shifts_per_seed(groups[arm], READOUTS[name], ratio) for arm in ARMS}
        record["readouts"][name] = {}
        for row in contrast_rows(name, shifts):
            values = list(row.per_seed.items())
            counts = []
            for _ in range(resamples // 100):
                drawn = dict(generator.sample(values, 6))
                counts.append(resolution(name, row.contrast, drawn).pairs)
            counts.sort()
            low, high = counts[len(counts) // 20], counts[-max(1, len(counts) // 20)]
            span = f"{low} to {high}"
            lines.append(f"  {name:<20}{row.contrast:<22}{row.dz:>+8.3f}{row.pairs:>11}{span:>22}")
            record["readouts"][name][row.contrast] = {
                **row.as_dict(),
                "six_seed_counts_5th_to_95th": [low, high],
            }
    print("\n".join(lines))
    print(
        "\n  the range is the 5th to 95th percentile of the count a random six of these seeds "
        "produces, which is what the exploration and the confirmation each drew one of"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, default="all", choices=("all", *STAGES))
    parser.add_argument("--out", type=Path, default=Path.home() / "tame-runs" / "57-estimator")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in EXPLORATION_SEEDS))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--splits", type=int, default=2_000)
    parser.add_argument(
        "--step-multiplier",
        type=float,
        default=2.0,
        help="Resting spreads the validation's setpoint step moves, unused in exploration",
    )
    parser.add_argument(
        "--chosen",
        type=str,
        default="token-conditioned",
        choices=sorted(READOUTS),
        help="Which readout the validation stage confirms",
    )
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(","))
    code_sha, code_dirty = code_identity()
    print(f"code {code_sha} dirty={code_dirty}; exploratory (#57), {args.steps} steps a run")
    if len(seeds) < MIN_PAIRS_FOR_COVERAGE:
        print(
            f"  WARNING: {len(seeds)} seeds is below {MIN_PAIRS_FOR_COVERAGE}, so every dz here "
            "is an estimate whose interval is the sample range"
        )
    stages = STAGES if args.stage == "all" else (args.stage,)
    record: dict[str, Any] = {"code_sha": code_sha, "code_dirty": code_dirty, "steps": args.steps}
    if "dose" in stages:
        record["dose"] = stage_dose(seeds, EXPLORATION_RATIO, args.steps)
    if "setpoint" in stages:
        record["setpoint"] = stage_setpoint(seeds, args.resamples)
    if "ablation" in stages:
        record["ablation"] = stage_ablation(
            seeds[:3], EXPLORATION_RATIO, args.steps, args.resamples
        )
    if "null" in stages:
        null_seeds = tuple(range(2 * len(seeds)))
        record["null"] = stage_null(
            null_seeds, EXPLORATION_RATIO, args.steps, args.splits, args.resamples
        )
    if "stability" in stages:
        record["stability"] = stage_stability(
            tuple(range(24)), EXPLORATION_RATIO, args.steps, args.resamples
        )
    if "validate" in stages:
        record["validate"] = stage_validate(
            args.chosen,
            VALIDATION_SEEDS,
            VALIDATION_RATIO,
            args.steps,
            args.resamples,
            args.step_multiplier,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "estimator_study.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nrecord: {args.out / 'estimator_study.json'}")


if __name__ == "__main__":
    main()
