"""Does a cell's self-model become load-bearing? Lever 1, on the fixture (#60).

#39's body run found a cell paid in continuation for its own value behaves like
one paid for another cell's. The reading is that with four experts and two slots
the report barely decides who acts -- six possible sets per token -- and payment
is realised value whatever the report said, so a wrong self-model costs a cell
nothing. A constraint nothing depends on is outside the closure that makes a
self (Montévil & Mossio), which is what "decorative" means here.

**Lever 1, the two ratios.** Cells per slot -- `num_experts` in {4, 8, 16} at
`top_k` 2, which is 6, 28 and 120 possible sets per token -- against the cells'
share of the output, the planted correction's scale in {1, 2, 4}. The full grid,
so the table can say which ratio binds rather than which sweep was run.

**Lever 2 is not here.** It was built on this branch -- a scored self-prediction
each cell is paid for -- and stripped from it: the score's target was the value a
cell realised if it held the token and zero if it did not, so the payment moved
with the allocation and therefore with the bid. It is redesigned on dense
counterfactual targets under #66.

The contrast is `value` minus `shuffled` -- the one #39 could not
separate -- read with the readout #57 left in place, which is the recorded
total-variation shift between the two dose groups. **#57 read "no" on every
candidate readout**, and #60 is gated on that: what this script can say is
whether the *substrate* makes the contrast visible where the *estimator* could
not, which is the question the gate asks and the fixture can answer for the
price of CPU-seconds.

**The grid runs at the derived exchange rate (#73).** The scale multiplies
every realised value, reward and price against a wealth band that does not
move, so at the hand-set `reward_scale` the 2x and 4x rows measured the clamp
(README `#self-model`). Each scaled cell of the grid now derives its rate from
its own settlement first (`measure_ledger_stability.derive_reward_scale`) and
runs at it, with the hand-set rate run beside it as the pairing -- the
withdrawn grid, reproduced so the two can be read against each other -- and a
cell whose derivation is refused is recorded as refused rather than run.

    uv run python scripts/measure_self_model.py --out ~/tame-runs/60-self-model
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from allocation_shift import DEFAULT_READOUT  # noqa: E402
from measure_ledger_stability import (  # noqa: E402
    DEFAULT_STEPS as DERIVATION_STEPS,
)
from measure_ledger_stability import (
    DIFFERENTIATED,
    ClampedLedgerError,
    Derivation,
    derive_reward_scale,
)
from measure_stakes_dial import (  # noqa: E402
    FIXTURE,
    GOAL_TYPES,
    RATIOS,
    READING_WINDOW,
    REFERENCE_DOSE,
    SETPOINT,
    STEPS,
    TAIL,
    WEALTH_HORIZON,
    fixture_fingerprint,
    share_columns,
)
from power import pairs_for_power  # noqa: E402
from sweep_wealth_bounds import CEILING_TOLERANCE, FLOOR_TOLERANCE  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    competence_for,
    pearson,
    shuffled,
)

from mob import PERSISTENCE_SHUFFLED, PERSISTENCE_VALUE  # noqa: E402
from parity import code_identity  # noqa: E402

SEEDS = (0, 1, 2)
RECORDED_RATE = BASE_CONFIG.reward_scale
# The two rates a scaled cell runs at: the one derived for it, and the hand-set
# constant as the pairing. The recorded scale has one rate, the recorded one.
DERIVED, HAND_SET = "derived", "hand-set"
# Lever 1's grid. `top_k` stays at 2, so the cell counts are 6, 28 and 120
# possible winner sets per token; the scales are multiples of the recorded
# fixture's orthonormal correction, which is what the cells own of the output.
CELL_COUNTS = (4, 8, 16)
CONTRIBUTION_SCALES = (1.0, 2.0, 4.0)
DOSE_RATIO = RATIOS[-1]


def run_arm(
    arm: str,
    ratio: float,
    seed: int,
    cells: int = len(DEFAULT_COMPETENCE),
    scale: float = 1.0,
    steps: int = STEPS,
    reward_scale: float = RECORDED_RATE,
) -> dict[str, float]:
    """One arm at one dose, at a cell count, a contribution scale and an exchange rate.

    The guardrail columns are the ones #60's grid was published without:
    occupancy over the same tail the shares are read on, and the correlation on
    the ledger that tail ends at. The scale multiplies value,
    reward and price but not the wealth band, so the thing to watch is the band:
    occupancy at each bound says whether an arm is still an economy or a clamp,
    and ``r(wealth, competence)`` says whether the ledger still tracks who is
    good. At four times the recorded correction every ledger sits on the ceiling
    and that correlation is not a number -- which is the reading the grid needed
    and did not have.
    """
    config = replace(
        BASE_CONFIG, num_experts=cells, persistence_coupling=arm, reward_scale=reward_scale
    )
    competence = shuffled(competence_for(cells), seed)
    economy = DifferentiatedEconomy(
        competence,
        seed=seed,
        config=config,
        contribution_scale=scale,
    )
    economy.add_goal_field(GOAL_TYPES[0], SETPOINT, ratio * REFERENCE_DOSE)
    economy.add_goal_field(GOAL_TYPES[1], SETPOINT, REFERENCE_DOSE)
    wins = torch.zeros(cells)
    losses: list[float] = []
    on_type: list[float] = []
    # The two bounds take the two tolerances #16 measured, and that is not a
    # fudge: the ceiling is an attractor a clamped cell sits exactly on, while
    # the floor is escaped by a hair on every exploration win and decays back, so
    # one tolerance for both undercounts the floor badly.
    ceiling = BASE_CONFIG.max_wealth * (1 - CEILING_TOLERANCE)
    floor = BASE_CONFIG.min_wealth * (1 + FLOOR_TOLERANCE)
    at_ceiling = 0
    at_floor = 0
    # The reading window is the tail, or the whole run when it is shorter than
    # one: a smoke run at 40 steps that divides by 100 reports two and a half
    # times less occupancy than it saw, which is the direction that hides a
    # clamp rather than inventing one.
    window = min(steps, TAIL)
    for step in range(steps):
        record = economy.step()
        if step < steps - window:
            continue
        wins += torch.bincount(record.selected_experts.flatten(), minlength=cells).float()
        losses.append(record.loss)
        on_type.append(economy.on_type_share(record.selected_experts))
        wealth = economy.mob.expert_wealth
        at_ceiling += int((wealth >= ceiling).sum())
        at_floor += int((wealth <= floor).sum())
    metrics = share_columns(wins, "routing/win_share_e")
    metrics["eval/loss"] = statistics.fmean(losses)
    metrics["routing/on_type_share"] = statistics.fmean(on_type)
    metrics.update(
        band_occupancy(at_ceiling, at_floor, cells * window, economy.mob.expert_wealth, competence)
    )
    return metrics


def band_occupancy(
    at_ceiling: int,
    at_floor: int,
    cell_steps: int,
    wealth: torch.Tensor,
    competence: torch.Tensor,
) -> dict[str, float]:
    """Where in the wealth band one arm spent its reading window, and whether it tracked.

    ``cell_steps`` is recorded beside the rates it divides: a rate whose
    denominator is not in the record is a rate the next reader has to guess at,
    and guessing it wrong is exactly what hid a clamp here once. The interior is
    the column that says whether an arm is still an economy rather than a pair of
    bounds, which is the reading #60's grid was published without.

    The correlation is NaN when every ledger is on one bound, and is left that
    way: a clamped arm has no correlation to report, and a zero there reads as
    one that does.
    """
    return {
        CELL_STEPS_COLUMN: float(cell_steps),
        "guardrail/ceiling_occupancy": at_ceiling / cell_steps,
        "guardrail/floor_occupancy": at_floor / cell_steps,
        "guardrail/interior_occupancy": 1.0 - (at_ceiling + at_floor) / cell_steps,
        CORRELATION_COLUMN: pearson(wealth, competence),
    }


def contrast_dz(shifts_a: dict[str, float], shifts_b: dict[str, float]) -> tuple[float, int]:
    """``a - b`` per seed as an effect size, and the paired seeds it would need.

    #56's helper on #60's contrast, because #57 measured what a count read at
    three seeds is worth: the number below is what the fixture would have to
    spend to resolve this cell of the grid, not a claim that it has.
    """
    deltas = [shifts_a[seed] - shifts_b[seed] for seed in sorted(shifts_a, key=str)]
    spread = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    dz = statistics.fmean(deltas) / spread if spread > 0 else 0.0
    return dz, pairs_for_power(dz)


OCCUPANCY_COLUMNS = (
    "guardrail/ceiling_occupancy",
    "guardrail/floor_occupancy",
    "guardrail/interior_occupancy",
)
CORRELATION_COLUMN = "guardrail/wealth_vs_competence"
# The cell-steps the occupancies are a rate over, summed across the pool.
CELL_STEPS_COLUMN = "guardrail/cell_steps"
# How many of the pooled readings had no correlation to give, and out of how
# many. A mean with three of six seeds clamped is a different number from the
# same mean with none, and without the count they print identically.
CLAMPED_COLUMN = "guardrail/clamped_readings"
READINGS_COLUMN = "guardrail/readings"
# The two dose levels the readout differences. Guardrails are kept apart by
# dose because the readout *is* the difference between them: a band that moves
# with the dose is invisible in a number pooled across it.
DOSE_BALANCED = "balanced"
DOSE_DOSED = "dosed"


def pooled(readings: list[dict[str, float]]) -> dict[str, float]:
    """One arm at one dose, over the seeds, with the correlation NaN-aware.

    ``statistics.fmean`` propagates a single NaN over the whole pool, and one
    seed whose every ledger sits on a bound is exactly the case where the other
    five are worth reading. So the mean is taken over the readings that have a
    correlation and the count of the ones that did not travels beside it; NaN
    survives only where nothing was readable at all, which is the reading
    "a clamped cell is not read" asks for.
    """
    summary: dict[str, float] = {
        column: statistics.fmean(reading[column] for reading in readings)
        for column in OCCUPANCY_COLUMNS
    }
    summary[CELL_STEPS_COLUMN] = sum(reading[CELL_STEPS_COLUMN] for reading in readings)
    readable = [
        reading[CORRELATION_COLUMN]
        for reading in readings
        if not math.isnan(reading[CORRELATION_COLUMN])
    ]
    summary[CORRELATION_COLUMN] = statistics.fmean(readable) if readable else float("nan")
    summary[CLAMPED_COLUMN] = len(readings) - len(readable)
    summary[READINGS_COLUMN] = len(readings)
    return summary


def shifts_for(
    arm: str,
    seeds: tuple[int, ...],
    cells: int,
    scale: float,
    steps: int,
    reward_scale: float = RECORDED_RATE,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """One arm's allocation shift between the two dose levels, per seed, and its guardrails.

    The guardrails come back from the same runs rather than from a second pass:
    reading the band costs nothing on top of a run that has already happened, and
    a guardrail measured on a different run is a guardrail for a different run.
    They are returned per dose level for the reason the arms are returned
    separately -- the contrast is a difference between two things, and a
    guardrail pooled over the difference cannot say which side moved.
    """
    shifts: dict[str, float] = {}
    readings: dict[str, list[dict[str, float]]] = {DOSE_BALANCED: [], DOSE_DOSED: []}
    for seed in seeds:
        balanced = run_arm(arm, RATIOS[0], seed, cells, scale, steps, reward_scale)
        dosed = run_arm(arm, DOSE_RATIO, seed, cells, scale, steps, reward_scale)
        shifts[str(seed)] = DEFAULT_READOUT.reading(balanced, dosed)
        readings[DOSE_BALANCED].append(balanced)
        readings[DOSE_DOSED].append(dosed)
    return shifts, {dose: pooled(rows) for dose, rows in readings.items()}


def cell_fingerprints(
    cells: int,
    scale: float,
    seeds: tuple[int, ...],
    steps: int,
    reward_scale: float = RECORDED_RATE,
    reward_scale_derived: bool = False,
) -> dict[str, Any]:
    """One fingerprint per arm per seed, recorded per grid cell rather than per run.

    The two ratios are what this grid varies, so a cell whose fingerprint says it
    ran at the recorded fixture is a cell no later comparison can refuse.
    ``parity.assert_parity`` refuses two of these against each other, which is
    the whole point of writing them down.
    """
    return {
        arm: {
            str(seed): fixture_fingerprint(
                FIXTURE,
                seed,
                arm,
                (RATIOS[0] * REFERENCE_DOSE, REFERENCE_DOSE),
                steps,
                cells=cells,
                contribution_scale=scale,
                reward_scale=reward_scale,
                reward_scale_derived=reward_scale_derived,
            ).as_dict()
            for seed in seeds
        }
        for arm in (PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED)
    }


def band_columns(arm: str, dose: str, bands: dict[str, float]) -> str:
    """One arm at one dose, under the headings ``sweep_lever_one`` prints."""
    clamped = int(bands[CLAMPED_COLUMN])
    unreadable = f"  {clamped}/{int(bands[READINGS_COLUMN])} clamped" if clamped else ""
    return (
        f"{arm:>10}{dose:>10}"
        f"{100 * bands['guardrail/ceiling_occupancy']:>7.1f}%"
        f"{100 * bands['guardrail/floor_occupancy']:>7.1f}%"
        f"{bands[CORRELATION_COLUMN]:>+9.3f}{unreadable}"
    )


def rates_for(
    cells: int, scale: float, derivation_seeds: tuple[int, ...], derivation_steps: int
) -> tuple[dict[str, tuple[float, bool]], dict[str, Any]]:
    """The rates one grid cell runs at, and the derivation record that chose them.

    At the recorded scale the rate is the recorded constant and nothing is
    derived. Elsewhere the derived rate comes first and the hand-set constant
    is the pairing; a refused derivation leaves only the pairing, with the
    refusal in the record, because a cell read at the clamp is #60's grid and
    not a reading of the derivation.
    """
    if scale == 1.0:
        return {HAND_SET: (RECORDED_RATE, False)}, {"derived": None, "refused": None}
    rates: dict[str, tuple[float, bool]] = {}
    record: dict[str, Any]
    try:
        derivation: Derivation = derive_reward_scale(
            DIFFERENTIATED, scale, seeds=derivation_seeds, steps=derivation_steps, cells=cells
        )
    except ClampedLedgerError as refused:
        record = {"derived": None, "refused": str(refused)}
        print(f"  cells {cells} scale {scale:g}: derivation refused -- {refused}")
    else:
        rates[DERIVED] = (derivation.derived, True)
        record = {"derived": derivation.as_dict(), "refused": None}
        print(
            f"  cells {cells} scale {scale:g}: derived reward_scale {derivation.derived:.5f} "
            f"(seed spread {derivation.seed_spread:.5f}, {len(derivation.passes)} passes; "
            f"ceiling at the derived rate "
            f"{statistics.fmean(derivation.final.ceiling_occupancy.values()):.3f} against "
            f"{statistics.fmean(derivation.reference_ceiling_occupancy.values()):.3f} at 1x)"
        )
    rates[HAND_SET] = (RECORDED_RATE, False)
    return rates, record


def grid_row(
    cells: int,
    scale: float,
    reward_scale: float,
    derived: bool,
    seeds: tuple[int, ...],
    steps: int,
) -> dict[str, Any]:
    """One (cells, scale) cell of the grid at one exchange rate: the contrast and its guardrails."""
    value, value_bands = shifts_for(PERSISTENCE_VALUE, seeds, cells, scale, steps, reward_scale)
    other, other_bands = shifts_for(PERSISTENCE_SHUFFLED, seeds, cells, scale, steps, reward_scale)
    dz, pairs = contrast_dz(value, other)
    return {
        "cells": cells,
        "winner_sets": cells * (cells - 1) // 2,
        "scale": scale,
        "reward_scale": reward_scale,
        "reward_scale_derived": derived,
        "value": value,
        "shuffled": other,
        "dz": dz,
        "paired_seeds_at_80": pairs,
        # Per arm, because the confound that withdrew this grid was the
        # two arms clamping at *different* bounds: a mean over both would
        # have hidden exactly the asymmetry that produced the contrast.
        "guardrails": {PERSISTENCE_VALUE: value_bands, PERSISTENCE_SHUFFLED: other_bands},
        "fingerprints": cell_fingerprints(cells, scale, seeds, steps, reward_scale, derived),
    }


def print_row(row: dict[str, Any], rate: str) -> None:
    """Every side of the contrast on a line of its own rather than a mean over it.

    What withdrew this grid is that the two arms clamped at different bounds,
    and at different doses, inside what printed as one row.
    """
    value, other = row["value"], row["shuffled"]
    blocks = [
        band_columns(arm, dose, row["guardrails"][arm][dose])
        for arm in (PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED)
        for dose in (DOSE_BALANCED, DOSE_DOSED)
    ]
    contrast = (
        f"  {row['cells']:>6}{row['winner_sets']:>7}{row['scale']:>7.1f}"
        f"{rate:>9}{row['reward_scale']:>9.4f}"
        f"{statistics.fmean(value.values()):>9.4f}"
        f"{statistics.fmean(other.values()):>10.4f}{row['dz']:>+8.3f}{row['paired_seeds_at_80']:>8}"
    )
    print(contrast + blocks[0])
    for block in blocks[1:]:
        print(f"  {'':>73}{block}")


def sweep_lever_one(
    seeds: tuple[int, ...],
    steps: int,
    derivation_seeds: tuple[int, ...] = SEEDS,
    derivation_steps: int = DERIVATION_STEPS,
) -> dict[str, Any]:
    """The grid: cells per slot against the cells' share of the output, at the derived rate."""
    print("\n== lever 1: the two ratios, value minus shuffled at the recorded readout ==")
    print(
        f"  {'cells':>6}{'sets':>7}{'scale':>7}{'rate':>9}{'reward':>9}{'value':>9}"
        f"{'shuffled':>10}{'dz':>8}{'seeds':>8}"
        f"{'arm':>10}{'dose':>10}{'ceil%':>8}{'floor%':>8}{'r(w,c)':>9}"
    )
    grid: dict[str, Any] = {}
    for cells in CELL_COUNTS:
        for scale in CONTRIBUTION_SCALES:
            rates, derivation = rates_for(cells, scale, derivation_seeds, derivation_steps)
            rows = {
                rate: grid_row(cells, scale, reward_scale, derived, seeds, steps)
                for rate, (reward_scale, derived) in rates.items()
            }
            for rate, row in rows.items():
                print_row(row, rate)
            grid[f"cells{cells}-scale{scale}"] = {
                "cells": cells,
                "scale": scale,
                "derivation": derivation,
                "rates": rows,
            }
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.home() / "tame-runs" / "60-self-model")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--derivation-steps",
        type=int,
        default=DERIVATION_STEPS,
        help="the settle each scaled cell's exchange rate is derived on (#73)",
    )
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(","))
    code_sha, code_dirty = code_identity()
    print(
        f"code {code_sha} dirty={code_dirty}; the differentiated fixture, {len(seeds)} seeds, "
        f"{args.steps} steps, dose ratio {RATIOS[0]} against {DOSE_RATIO}"
    )
    print(
        "  the readout is the recorded total-variation shift: #57 adopted no candidate, so "
        "the instrument is unchanged and what varies here is the substrate"
    )
    record: dict[str, Any] = {
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "seeds": list(seeds),
        "steps": args.steps,
        "horizon": WEALTH_HORIZON,
        "reading_window": READING_WINDOW,
        "derivation_steps": args.derivation_steps,
        "recorded_reward_scale": RECORDED_RATE,
    }
    record["lever_one"] = sweep_lever_one(seeds, args.steps, seeds, args.derivation_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "self_model.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nrecord: {args.out / 'self_model.json'}")


if __name__ == "__main__":
    main()
