"""Does a cell's self-model become load-bearing? The two levers, on the fixture (#60).

#39's body run found a cell paid in continuation for its own value behaves like
one paid for another cell's. The reading is that with four experts and two slots
the report barely decides who acts -- six possible sets per token -- and payment
is realised value whatever the report said, so a wrong self-model costs a cell
nothing. A constraint nothing depends on is outside the closure that makes a
self (Montévil & Mossio), which is what "decorative" means here.

Two levers, measured separately, never together (a sweep that moves two knobs at
once is collinear, and the grid is the guard):

- **Lever 1, the two ratios.** Cells per slot -- `num_experts` in {4, 8, 16} at
  `top_k` 2, which is 6, 28 and 120 possible sets per token -- against the cells'
  share of the output, the planted correction's scale in {1, 2, 4}. The full
  grid, so the table can say which ratio binds rather than which sweep was run.
- **Lever 2, a self-prediction that costs something.** Each head emits a
  prediction of its own realised value beside its bid and is paid
  `mu x -(prediction - realised)^2`, a bounded strictly proper rule. The scored
  output is a separate projection from the bid, so the auction's price stays
  report-independent and the constitution is untouched by construction.

The contrast under both is `value` minus `shuffled` -- the one #39 could not
separate -- read with the readout #57 left in place, which is the recorded
total-variation shift between the two dose groups. **#57 read "no" on every
candidate readout**, and #60 is gated on that: what this script can say is
whether the *substrate* makes the contrast visible where the *estimator* could
not, which is the question the gate asks and the fixture can answer for the
price of CPU-seconds.

    uv run python scripts/measure_self_model.py --out ~/tame-runs/60-self-model
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from allocation_shift import DEFAULT_READOUT  # noqa: E402
from measure_stakes_dial import (  # noqa: E402
    GOAL_TYPES,
    RATIOS,
    READING_WINDOW,
    REFERENCE_DOSE,
    SETPOINT,
    STEPS,
    TAIL,
    WEALTH_HORIZON,
    share_columns,
    type_wins,
)
from power import pairs_for_power  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    shuffled,
)

from mob import PERSISTENCE_SHUFFLED, PERSISTENCE_VALUE  # noqa: E402
from parity import code_identity  # noqa: E402

SEEDS = (0, 1, 2)
# Lever 1's grid. `top_k` stays at 2, so the cell counts are 6, 28 and 120
# possible winner sets per token; the scales are multiples of the recorded
# fixture's orthonormal correction, which is what the cells own of the output.
CELL_COUNTS = (4, 8, 16)
CONTRIBUTION_SCALES = (1.0, 2.0, 4.0)
# Lever 2's prices. Zero is the recorded economy and is asserted to reproduce it
# bitwise; the rest are the sweep.
SELF_SCORE_MU = (0.0, 0.5, 2.0)
DOSE_RATIO = RATIOS[-1]


def competence_for(cells: int) -> torch.Tensor:
    """A competence vector of ``cells`` entries over the recorded fixture's range.

    Eight is the recorded vector itself, unchanged, so that cell of the grid is
    the fixture every other number in this project was measured on. Four takes
    every other entry and sixteen repeats each one, which keeps the range and
    the spacing rather than inventing a new distribution for each row -- a grid
    whose cells differ in two things at once cannot say which one binds.
    """
    if cells == len(DEFAULT_COMPETENCE):
        return DEFAULT_COMPETENCE
    if cells == len(DEFAULT_COMPETENCE) // 2:
        return DEFAULT_COMPETENCE[::2]
    if cells == 2 * len(DEFAULT_COMPETENCE):
        return DEFAULT_COMPETENCE.repeat_interleave(2)
    raise ValueError(f"no competence vector defined for {cells} cells")


def run_arm(
    arm: str,
    ratio: float,
    seed: int,
    cells: int = len(DEFAULT_COMPETENCE),
    scale: float = 1.0,
    mu: float = 0.0,
    steps: int = STEPS,
) -> dict[str, float]:
    """One arm at one dose, at a cell count, a contribution scale and a price."""
    config = replace(BASE_CONFIG, num_experts=cells, persistence_coupling=arm, self_score_mu=mu)
    economy = DifferentiatedEconomy(
        shuffled(competence_for(cells), seed),
        seed=seed,
        config=config,
        contribution_scale=scale,
    )
    economy.add_goal_field(GOAL_TYPES[0], SETPOINT, ratio * REFERENCE_DOSE)
    economy.add_goal_field(GOAL_TYPES[1], SETPOINT, REFERENCE_DOSE)
    wins = torch.zeros(cells)
    by_type = {expert_type: torch.zeros(cells) for expert_type in GOAL_TYPES}
    losses: list[float] = []
    on_type: list[float] = []
    for step in range(steps):
        record = economy.step()
        if step < steps - TAIL:
            continue
        wins += torch.bincount(record.selected_experts.flatten(), minlength=cells).float()
        for expert_type, counts in type_wins(economy, record.selected_experts).items():
            by_type[expert_type] += counts
        losses.append(record.loss)
        on_type.append(economy.on_type_share(record.selected_experts))
    metrics = share_columns(wins, "routing/win_share_e")
    metrics["eval/loss"] = statistics.fmean(losses)
    metrics["routing/on_type_share"] = statistics.fmean(on_type)
    return metrics


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


def shifts_for(
    arm: str, seeds: tuple[int, ...], cells: int, scale: float, mu: float, steps: int
) -> dict[str, float]:
    """One arm's allocation shift between the two dose levels, per seed."""
    shifts: dict[str, float] = {}
    for seed in seeds:
        balanced = run_arm(arm, RATIOS[0], seed, cells, scale, mu, steps)
        dosed = run_arm(arm, DOSE_RATIO, seed, cells, scale, mu, steps)
        shifts[str(seed)] = DEFAULT_READOUT.reading(balanced, dosed)
    return shifts


def sweep_lever_one(seeds: tuple[int, ...], steps: int) -> dict[str, Any]:
    """The grid: cells per slot against the cells' share of the output."""
    print("\n== lever 1: the two ratios, value minus shuffled at the recorded readout ==")
    print(f"  {'cells':>6}{'sets':>7}{'scale':>7}{'value':>9}{'shuffled':>10}{'dz':>8}{'seeds':>8}")
    grid: dict[str, Any] = {}
    for cells in CELL_COUNTS:
        sets = cells * (cells - 1) // 2
        for scale in CONTRIBUTION_SCALES:
            value = shifts_for(PERSISTENCE_VALUE, seeds, cells, scale, 0.0, steps)
            other = shifts_for(PERSISTENCE_SHUFFLED, seeds, cells, scale, 0.0, steps)
            dz, pairs = contrast_dz(value, other)
            grid[f"cells{cells}-scale{scale}"] = {
                "cells": cells,
                "winner_sets": sets,
                "scale": scale,
                "value": value,
                "shuffled": other,
                "dz": dz,
                "paired_seeds_at_80": pairs,
            }
            print(
                f"  {cells:>6}{sets:>7}{scale:>7.1f}"
                f"{statistics.fmean(value.values()):>9.4f}"
                f"{statistics.fmean(other.values()):>10.4f}{dz:>+8.3f}{pairs:>8}"
            )
    return grid


def sweep_lever_two(seeds: tuple[int, ...], steps: int) -> dict[str, Any]:
    """The price on the self-model, at the recorded configuration."""
    print("\n== lever 2: a self-prediction that costs something ==")
    print(f"  {'mu':>6}{'value':>9}{'shuffled':>10}{'dz':>8}{'seeds':>8}")
    prices: dict[str, Any] = {}
    for mu in SELF_SCORE_MU:
        value = shifts_for(PERSISTENCE_VALUE, seeds, len(DEFAULT_COMPETENCE), 1.0, mu, steps)
        other = shifts_for(PERSISTENCE_SHUFFLED, seeds, len(DEFAULT_COMPETENCE), 1.0, mu, steps)
        dz, pairs = contrast_dz(value, other)
        prices[f"mu{mu}"] = {
            "mu": mu,
            "value": value,
            "shuffled": other,
            "dz": dz,
            "paired_seeds_at_80": pairs,
        }
        print(
            f"  {mu:>6.1f}{statistics.fmean(value.values()):>9.4f}"
            f"{statistics.fmean(other.values()):>10.4f}{dz:>+8.3f}{pairs:>8}"
        )
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.home() / "tame-runs" / "60-self-model")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--skip-grid", action="store_true")
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
    }
    if not args.skip_grid:
        record["lever_one"] = sweep_lever_one(seeds, args.steps)
    record["lever_two"] = sweep_lever_two(seeds, args.steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "self_model.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nrecord: {args.out / 'self_model.json'}")


if __name__ == "__main__":
    main()
