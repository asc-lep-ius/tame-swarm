"""Is the auction a seniority system? The gate decomposed on the fixture (#62).

The bid is ``confidence x wealth``; in the log domain the gate is
``log(confidence) + log(wealth)``, and ``log(wealth)`` sits exactly where a
load-balancing bias would (#11, #26). On every token of a settled run this reads
how much of the gate's variation across cells the report carries and how often
the winner set is simply wealth's top-*k*. It reads **seniority** when that
fraction is above 0.90 on the ``value`` arm at the reference dose over the
settled tail -- the line fixed in the issue before anything ran -- because then
*which* cell acts is decided by the ledger's history, not by the token, and no
per-token readout can see allocation whatever the cells' loudness.

Two fixtures, three arms, three seeds, the recorded configuration (2667 steps,
eight memory horizons at the shipped decay; the last horizon is the tail, #16's
convention). The floor is ``seed0`` against ``seed0-replicate`` in every group:
the fixture is bitwise deterministic on CPU, so the floor is expected to read
exactly zero, and it is measured rather than assumed. The built-in control is
the ``decoupled`` arm, whose pinned ledger makes the wealth term a constant by
construction, so the confidence term is all that varies there -- the ceiling on
what the report can do on this substrate. Section 8 rule 8 binds: ceiling and
floor occupancy and ``r(wealth, competence)`` are reported beside every reading.

    uv run python scripts/gate_decomposition.py --fixture both --seeds 0,1,2 \\
        --out ~/tame-runs/gate-decomposition/

The body half -- #39's 24 checkpoints, forward only -- is
``scripts/gate_decomposition_body.py``; both write ``gate_decomposition.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from measure_ledger_stability import DEFAULT_STEPS, DEFAULT_TAIL  # noqa: E402
from measure_self_model import CORRELATION_COLUMN, band_occupancy  # noqa: E402
from measure_stakes_dial import (  # noqa: E402
    FIXTURE,
    GOAL_TYPES,
    REFERENCE_DOSE,
    SETPOINT,
    fixture_fingerprint,
)
from sweep_wealth_bounds import CEILING_TOLERANCE, FLOOR_TOLERANCE  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    SyntheticEconomy,
    shuffled,
)

from mob import (  # noqa: E402
    PERSISTENCE_DECOUPLED,
    PERSISTENCE_SHUFFLED,
    PERSISTENCE_VALUE,
)
from mob.gate_decomposition import GateDecomposition, decompose, merge  # noqa: E402
from parity import code_identity  # noqa: E402

# #62's line, written into the issue before the first read.
SENIORITY_LINE = 0.90
ARMS = (PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED, PERSISTENCE_DECOUPLED)
SEEDS = (0, 1, 2)
QUALITY = "quality"
DIFFERENTIATED = "differentiated"
FIXTURES = (QUALITY, DIFFERENTIATED)
FIXTURE_IDS = {QUALITY: "quality-fixture", DIFFERENTIATED: FIXTURE}
# One memory horizon at the shipped decay (1 / (1 - 0.997)): the unit the time
# course is read in, and the tail #16 settles over.
HORIZON = DEFAULT_TAIL
REPLICATE = "seed0-replicate"
# The tail's fields the floor is read on, and the table prints.
READ_FIELDS = (
    "seniority_fraction",
    "sold_seniority_fraction",
    "undecidable_fraction",
    "confidence_fraction",
    "wealth_fraction",
    "cross_fraction",
)


def script_identity() -> dict[str, Any]:
    """The code SHA beside a digest of this file, so a reading names the script that read it."""
    code_sha, code_dirty = code_identity()
    digest = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    return {"code_sha": code_sha, "code_dirty": code_dirty, "script_sha1": digest}


def doses_for(fixture: str) -> tuple[float, ...]:
    """Dose ratio 1 on the differentiated fixture; the quality fixture has no field."""
    return (REFERENCE_DOSE, REFERENCE_DOSE) if fixture == DIFFERENTIATED else ()


def build_economy(fixture: str, arm: str, seed: int) -> SyntheticEconomy:
    """The recorded configuration of either fixture, at one arm, competence shuffled by seed."""
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    if fixture == QUALITY:
        return SyntheticEconomy(competence, seed, config=config)
    if fixture != DIFFERENTIATED:
        raise ValueError(f"unknown fixture {fixture!r}; one of {FIXTURES}")
    economy = DifferentiatedEconomy(competence, seed, config=config)
    economy.add_goal_field(GOAL_TYPES[0], SETPOINT, REFERENCE_DOSE)
    economy.add_goal_field(GOAL_TYPES[1], SETPOINT, REFERENCE_DOSE)
    return economy


def _course(per_step: list[GateDecomposition], horizon: int) -> list[dict[str, Any]]:
    """The reading per memory horizon from the start: does seniority rise as the ledger settles?"""
    course: list[dict[str, Any]] = []
    for window in range(len(per_step) // horizon):
        start, stop = window * horizon, (window + 1) * horizon
        reading = merge(per_step[start:stop])
        course.append(
            {
                "steps": [start, stop],
                "seniority_fraction": reading.seniority_fraction,
                "sold_seniority_fraction": reading.sold_seniority_fraction,
                "undecidable_fraction": reading.undecidable_fraction,
                "confidence_fraction": reading.confidence_fraction,
                "wealth_fraction": reading.wealth_fraction,
            }
        )
    return course


def read_run(
    fixture: str,
    arm: str,
    seed: int,
    steps: int = DEFAULT_STEPS,
    tail: int = DEFAULT_TAIL,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """One arm of one fixture at one seed: the gate decomposed on every token, read over the tail.

    The wealth the gate read is taken *before* each step -- ``allocation_wealth``
    is the live ledger under ``value`` and ``shuffled`` and the pinned constant
    under ``decoupled`` -- because the ledger a step records is the one it left
    behind, not the one it bid with.
    """
    economy = build_economy(fixture, arm, seed)
    mob = economy.mob
    window = min(steps, tail)
    ceiling = BASE_CONFIG.max_wealth * (1 - CEILING_TOLERANCE)
    floor = BASE_CONFIG.min_wealth * (1 + FLOOR_TOLERANCE)
    per_step: list[GateDecomposition] = []
    at_ceiling = at_floor = 0
    for step in range(steps):
        wealth_read = mob.allocation_wealth().detach().clone()
        economy.step()
        stats = mob.last_stats
        assert stats is not None
        per_step.append(
            decompose(stats.confidences, wealth_read, stats.selected_experts, mob.config.top_k)
        )
        if step >= steps - window:
            at_ceiling += int((mob.expert_wealth >= ceiling).sum())
            at_floor += int((mob.expert_wealth <= floor).sum())
    cells = mob.config.num_experts
    return {
        "fixture": fixture,
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "tail_steps": window,
        "horizon": horizon,
        "tail": asdict(merge(per_step[-window:])),
        "course": _course(per_step, horizon),
        "guardrail": band_occupancy(
            at_ceiling, at_floor, cells * window, mob.expert_wealth, economy.competence
        ),
        "fingerprint": fixture_fingerprint(
            FIXTURE_IDS[fixture], seed, arm, doses_for(fixture), steps
        ).as_dict(),
        **script_identity(),
    }


def floor_between(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    """The read's own noise: the tail fields of one run against its replicate."""
    return {
        field: abs(float(first["tail"][field]) - float(second["tail"][field]))
        for field in READ_FIELDS
    }


def _fmt(value: float) -> str:
    return "   n/a" if value != value else f"{value:6.3f}"


def _row(label: str, tail: dict[str, Any], guardrail: dict[str, float]) -> str:
    return (
        f"  {label:<28}"
        + "".join(_fmt(float(tail[field])) for field in READ_FIELDS)
        + f"   ceil {100 * guardrail['guardrail/ceiling_occupancy']:5.1f}%"
        f"  floor {100 * guardrail['guardrail/floor_occupancy']:5.1f}%"
        f"  r {_fmt(guardrail[CORRELATION_COLUMN])}"
    )


def _verdict_lines(readings: list[dict[str, Any]]) -> list[str]:
    """The primary against the line: ``value`` over the tail, per fixture, three seeds as three."""
    lines: list[str] = []
    for fixture in FIXTURES:
        fractions = [
            reading["tail"]["seniority_fraction"]
            for reading in readings
            if reading["fixture"] == fixture
            and reading["arm"] == PERSISTENCE_VALUE
            and reading["label"] != REPLICATE
        ]
        if not fractions:
            continue
        above = [f for f in fractions if not math.isnan(f) and f > SENIORITY_LINE]
        lines.append(
            f"  {fixture:<15} value seniority "
            + ", ".join(_fmt(f) for f in fractions)
            + f"  -> {len(above)} of {len(fractions)} seeds above the {SENIORITY_LINE:.2f} line"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=(*FIXTURES, "both"), default="both")
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    parser.add_argument(
        "--out", type=Path, default=Path.home() / "tame-runs" / "gate-decomposition"
    )
    args = parser.parse_args()
    fixtures = FIXTURES if args.fixture == "both" else (args.fixture,)
    arms = tuple(args.arms.split(","))
    seeds = tuple(int(part) for part in args.seeds.split(","))
    identity = script_identity()
    print(
        f"code {identity['code_sha']} dirty={identity['code_dirty']} "
        f"script {identity['script_sha1']}"
    )
    print(f"  {args.steps} steps, tail {args.tail}, seniority line {SENIORITY_LINE}")
    header = "  " + " " * 28 + "".join(f"{field[:6]:>7}" for field in READ_FIELDS)
    readings: list[dict[str, Any]] = []
    floors: dict[str, dict[str, float]] = {}
    for fixture in fixtures:
        print(f"\n== {fixture} fixture ==\n{header}")
        for arm in arms:
            labels = [(f"seed{seed}", seed) for seed in seeds]
            if seeds and seeds[0] == 0:
                labels.append((REPLICATE, 0))
            group: dict[str, dict[str, Any]] = {}
            for label, seed in labels:
                reading = read_run(fixture, arm, seed, steps=args.steps, tail=args.tail)
                reading["label"] = label
                group[label] = reading
                readings.append(reading)
                run_dir = args.out / "fixture" / fixture / arm / label
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "gate_decomposition.json").write_text(
                    json.dumps(reading, indent=2, default=str)
                )
                print(_row(f"{arm} {label}", reading["tail"], reading["guardrail"]))
            if REPLICATE in group:
                floor = floor_between(group["seed0"], group[REPLICATE])
                floors[f"{fixture}/{arm}"] = floor
                print(
                    f"  {arm} floor (seed0 vs replicate)   "
                    + "".join(_fmt(v) for v in floor.values())
                )
    print("\n== primary: winner set == wealth's top-k, value arm, settled tail ==")
    verdict = _verdict_lines(readings)
    print("\n".join(verdict))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        **identity,
        "seniority_line": SENIORITY_LINE,
        "steps": args.steps,
        "tail": args.tail,
        "floors": floors,
        "verdict_lines": verdict,
        "readings": readings,
    }
    (args.out / "fixture" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nrecord: {args.out / 'fixture' / 'summary.json'}")


if __name__ == "__main__":
    main()
