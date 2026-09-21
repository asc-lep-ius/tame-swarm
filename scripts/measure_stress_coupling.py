"""Does a shared, unattributed stress bind cells better than an attributed payment? (#59)

Three arms at equal goal budget on the differentiated fixture, read through
#57's setpoint-step protocol because Pio-Lopez et al. found binding dispensable
under a static objective and selected under change:

- ``attributed`` — #33's goal term at the reference dose, no charge. Every arm
  ever recorded is this one.
- ``shared`` — no goal term; every cell at the layer pays ``lambda`` times the
  layer's transmitted stress, winner or not, so the only way to lower the charge
  is to act (``tame/mob/stress.py``).
- ``mixed`` — both at half budget.

**Equal budget is measured, not assumed.** ``lambda`` is set so that the wealth
the charge removes per token at initialisation equals the wealth the goal term
adds per token at the reference dose, on this fixture, before the run. Without
it the arms differ in how much wealth moves as well as in what moves it, and the
comparison is a dose sweep wearing an ontology.

**The primary, preregistered:** the residual stress over one wealth memory
horizon after the swept field's setpoint is stepped by one resting spread,
``shared`` minus ``attributed``, paired by seed, with the resampled-mean range;
inside the range reads "no". With the discriminator #57 built: the recovery
under ``shared`` must differ from ``attributed`` in a way a single first-order
lag fitted to ``attributed`` cannot produce, else the step measured a transient.

**The control is a replayed stress.** ``shuffled`` charges the same arm a
trajectory of charges recorded from another seed's ``shared`` run: the right
shape, the right size, and nothing this tissue does can lower it. If binding
survives that, the effect is the drain and not the stress.

    uv run python scripts/measure_stress_coupling.py --out ~/tame-runs/59-stress
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from compare_runs import DEFAULT_RESAMPLES, bootstrap_mean  # noqa: E402
from dose_slope import interval_label  # noqa: E402
from estimator_study import (  # noqa: E402
    fit_first_order_lag,
    imposed_lag_residual,
    within_control_excess,
)
from measure_stakes_dial import (  # noqa: E402
    GOAL_TYPES,
    READING_WINDOW,
    REFERENCE_DOSE,
    SETPOINT,
    WEALTH_HORIZON,
)
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    pearson,
    shuffled,
)

from mob.ledger import PERSISTENCE_DECOUPLED  # noqa: E402
from mob.stress import STRESS_ATTRIBUTED, STRESS_MIXED, STRESS_SHARED  # noqa: E402
from parity import code_identity  # noqa: E402

SEEDS = (0, 1, 2)
SETTLE_STEPS = WEALTH_HORIZON
RECOVER_STEPS = WEALTH_HORIZON
# The window #59 sweeps around the equal-budget price, and the paracrine
# fractions it sweeps at each. The primary sits at (1, 1/2).
LAMBDA_MULTIPLES = (0.25, 1.0, 4.0)
GAMMAS = (0.0, 0.5, 1.0)
PRIMARY_GAMMA = 0.5
# Where the fixture's setpoint sits, in resting spreads above the reading the
# tissue settles at on its own. The body's is `resting_mean + lift x strength`
# from `AlignmentCalibration` -- a target the tissue can reach, a little above
# where it rests -- and #39's fixture setpoint of 0.5 is not that: the tissue
# settles around 0.23 and sits eight resting spreads from it, so the gate is
# permanently open and the charge is a flat drain rather than an error signal.
# One spread puts the resting tissue exactly *at* the gate, so at rest nothing
# is transmitted and the experimenter's step is what opens it, which is the
# regime the mechanism is about.
SETPOINT_LIFT = 1.0
ARMS = (STRESS_ATTRIBUTED, STRESS_SHARED, STRESS_MIXED)
SHUFFLED_STRESS = "shuffled-stress"
STATE_GATED = "state-gated"


@dataclass(frozen=True)
class Calibration:
    """What the fixture has to measure before the arms can be at parity."""

    resting_sigma: float
    resting_reading: float
    setpoint: float
    mean_stress: float
    attributed_per_token: float
    equal_budget_lambda: float
    # What `mean_stress` was divided by, and what the control has to divide by
    # too: `last_stress_charge` is a per-step *total* over the layer's tokens,
    # while `stress_override` is a per-token magnitude the layer expands to all
    # of them. Replaying a total as a magnitude charged the control arm
    # `tokens` times the drain it was supposed to match.
    tokens: int = 1

    def as_dict(self) -> dict[str, float]:
        return {
            "resting_sigma": self.resting_sigma,
            "resting_reading": self.resting_reading,
            "setpoint": self.setpoint,
            "mean_stress": self.mean_stress,
            "attributed_per_token": self.attributed_per_token,
            "tokens": float(self.tokens),
            "equal_budget_lambda": self.equal_budget_lambda,
        }


def build(
    seed: int,
    *,
    coupling: str = STRESS_ATTRIBUTED,
    stress_lambda: float = 0.0,
    gamma: float = 0.0,
    gate_mode: str = "fixed",
    dose: float = REFERENCE_DOSE,
    resting_sigma: float | None = None,
    setpoint: float = SETPOINT,
) -> DifferentiatedEconomy:
    """One arm of the fixture: the dial at ``value``, the coupling as asked.

    The swept field is attached first, so it is the layer's own stress and the
    held field is its neighbourhood -- which is the fixture's analogue of the
    body's layers above and below, and is called one in ``mob.stress``.
    """
    config = replace(
        BASE_CONFIG,
        stress_coupling=coupling,
        stress_lambda=stress_lambda,
        stress_gamma=gamma,
        stress_gate_mode=gate_mode,
    )
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)
    economy.add_goal_field(GOAL_TYPES[0], setpoint, dose, resting_sigma=resting_sigma)
    economy.add_goal_field(GOAL_TYPES[1], setpoint, dose, resting_sigma=resting_sigma)
    return economy


def calibrate(seed: int, steps: int = SETTLE_STEPS) -> Calibration:
    """The resting spread, and the price that puts the two channels at one budget.

    Both are read on a run that charges nothing, because a spread the mechanism
    measured while the mechanism was running is not a resting spread -- the same
    reason the body's setpoints come from the pristine model before conversion.
    The attributed budget is measured by difference: one step of this fixture at
    the reference dose against one step at dose zero, from the same seed, which
    at step 0 are the same trajectory, so what separates their ledgers is the
    goal term and nothing else.
    """
    # Read with no goal payment at all: the tissue's own resting state, which
    # is what the body's calibration measures on the pristine model before any
    # field exists. Measured under the payment instead, the "resting" reading
    # is where the payment pulled it, and a setpoint one spread above *that* is
    # four spreads away from where an unpaid tissue actually sits -- which is
    # how the first version of this left the gate permanently open.
    economy = build(seed, dose=0.0)
    readings: list[float] = []
    field = economy.goal_fields()[0]
    for step in range(steps):
        record = economy.step()
        if step >= steps - READING_WINDOW:
            readings.append(economy.goal_reading(field, record.selected_experts))
    resting_sigma = statistics.stdev(readings)
    resting_reading = statistics.fmean(readings)
    setpoint = resting_reading + SETPOINT_LIFT * resting_sigma

    paid = build(seed, setpoint=setpoint)
    unpaid = build(seed, dose=0.0, setpoint=setpoint)
    paid.step()
    unpaid.step()
    attributed = float(paid.mob.expert_wealth.sum()) - float(unpaid.mob.expert_wealth.sum())
    tokens = paid.batch_size * paid.seq_len

    stressed = build(
        seed,
        coupling=STRESS_SHARED,
        stress_lambda=1.0,
        gamma=PRIMARY_GAMMA,
        resting_sigma=resting_sigma,
        setpoint=setpoint,
    )
    stressed.step()
    charge = stressed.mob.last_stress_charge
    cells = stressed.config.num_experts
    price = abs(attributed) / (cells * charge) if charge > 0 else 0.0
    return Calibration(
        resting_sigma=resting_sigma,
        resting_reading=resting_reading,
        setpoint=setpoint,
        mean_stress=charge / tokens,
        attributed_per_token=abs(attributed) / tokens,
        equal_budget_lambda=price,
        tokens=tokens,
    )


def run_stepped(
    seed: int,
    arm: str,
    calibration: Calibration,
    multiple: float = 1.0,
    gamma: float = PRIMARY_GAMMA,
    gate_mode: str = "fixed",
    replay: list[float] | None = None,
) -> dict[str, Any]:
    """Settle, step the swept field's setpoint by one resting spread, settle again.

    Returns the residual stress over the recovery horizon -- the primary -- with
    the guardrails read on the same run, because a guardrail read on another run
    is a guardrail for another run.
    """
    price = calibration.equal_budget_lambda * multiple
    if arm == STRESS_ATTRIBUTED:
        economy = build(
            seed,
            dose=REFERENCE_DOSE,
            resting_sigma=calibration.resting_sigma,
            setpoint=calibration.setpoint,
        )
    elif arm == STRESS_SHARED:
        economy = build(
            seed,
            coupling=STRESS_SHARED,
            stress_lambda=price,
            gamma=gamma,
            gate_mode=gate_mode,
            dose=0.0,
            resting_sigma=calibration.resting_sigma,
            setpoint=calibration.setpoint,
        )
    elif arm == STRESS_MIXED:
        economy = build(
            seed,
            coupling=STRESS_MIXED,
            stress_lambda=price / 2,
            gamma=gamma,
            gate_mode=gate_mode,
            dose=REFERENCE_DOSE / 2,
            resting_sigma=calibration.resting_sigma,
            setpoint=calibration.setpoint,
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")

    competence = shuffled(DEFAULT_COMPETENCE, seed)
    charges: list[float] = []
    for step in range(SETTLE_STEPS):
        if replay is not None:
            economy.mob.stress_override = torch.tensor(replay[step % len(replay)])
        economy.step()
        charges.append(economy.mob.last_stress_charge)

    stepped = economy.step_goal_setpoint(GOAL_TYPES[0], calibration.resting_sigma)
    residuals: list[float] = []
    losses: list[float] = []
    on_type: list[float] = []
    off_type_relief: list[float] = []
    for step in range(RECOVER_STEPS):
        if replay is not None:
            # `SETTLE_STEPS +`, or the recovery wraps back onto the donor's
            # settle phase and the arm never sees the donor's response to the
            # step -- which is the only phase the primary is read on.
            economy.mob.stress_override = torch.tensor(replay[(SETTLE_STEPS + step) % len(replay)])
        record = economy.step()
        reading = economy.goal_reading(stepped, record.selected_experts)
        residuals.append(abs(stepped.setpoint - reading) / calibration.resting_sigma)
        losses.append(record.loss)
        share = economy.on_type_share(record.selected_experts)
        on_type.append(share)
        off_type_relief.append(1.0 - share)
        charges.append(economy.mob.last_stress_charge)

    asymptote, tau, fit_residual = fit_first_order_lag(residuals)
    wealth = economy.mob.expert_wealth
    tail = slice(-READING_WINDOW, None)
    return {
        "residual_stress": statistics.fmean(residuals[tail]),
        "residual_curve": residuals,
        "asymptote": asymptote,
        "tau": tau,
        "fit_residual": fit_residual,
        "tail_loss": statistics.fmean(losses[tail]),
        "wealth_vs_competence": pearson(wealth, competence),
        "floor_occupancy": float((wealth <= economy.config.min_wealth + 1e-6).float().mean()),
        "least_used_share": float(
            (
                economy.mob.expert_usage_count / economy.mob.expert_usage_count.sum().clamp_min(1)
            ).min()
        ),
        "on_type_share": statistics.fmean(on_type[tail]),
        "charge_per_step": statistics.fmean(charges[-READING_WINDOW:]),
        "charges": charges,
    }


def contrast(label: str, a: dict[str, float], b: dict[str, float], resamples: int) -> str:
    """``b - a`` per shared seed, with the range #35's form gives at this count."""
    seeds = sorted(set(a) & set(b), key=str)
    deltas = [b[seed] - a[seed] for seed in seeds]
    centre, low, high = bootstrap_mean(deltas, resamples, 0)
    per_seed = "  ".join(f"s{seed}={delta:+.4f}" for seed, delta in zip(seeds, deltas, strict=True))
    return (
        f"{label:<44}{centre:>+9.4f}  [{low:+.4f}, {high:+.4f}]  "
        f"{interval_label(len(deltas))}  {per_seed}"
    )


def _as_fits(runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """One arm's fits under the key names `within_control_excess` reads them by."""
    return {
        seed: {
            "signature1/step_asymptote": run["asymptote"],
            "signature1/step_tau": run["tau"],
            "signature1/step_residual": run["fit_residual"],
        }
        for seed, run in runs.items()
    }


def read_arms(
    seeds: tuple[int, ...], calibrations: dict[int, Calibration], resamples: int
) -> dict[str, Any]:
    """The three arms at the equal-budget price, and the primary between two of them."""
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        runs[arm] = {}
        for seed in seeds:
            runs[arm][str(seed)] = run_stepped(seed, arm, calibrations[seed])
        print(f"  ran {arm:<12} seeds {list(seeds)}", flush=True)

    def column(arm: str, key: str) -> dict[str, float]:
        return {seed: float(run[key]) for seed, run in runs[arm].items()}

    print("\n== primary: residual stress after the setpoint step, shared minus attributed ==")
    print(
        contrast(
            "shared - attributed (residual stress)",
            column(STRESS_ATTRIBUTED, "residual_stress"),
            column(STRESS_SHARED, "residual_stress"),
            resamples,
        )
    )
    print("  a range that includes zero reads 'no'")
    print("\n== the discriminator: what one first-order lag fitted to attributed cannot do ==")
    print(
        contrast(
            "asymptote, shared - attributed",
            column(STRESS_ATTRIBUTED, "asymptote"),
            column(STRESS_SHARED, "asymptote"),
            resamples,
        )
    )
    # `imposed_lag_residual` refits only the amplitude over a strict subfamily of
    # what `fit_first_order_lag` already minimised, so `imposed - own >= 0` for
    # every curve and "excludes zero" on it is arithmetic rather than a reading.
    # #57 found that on its own setpoint step and built the null this reuses:
    # the same statistic on the control arm against its own neighbour, so the
    # difference can come out either way. Without it this row read
    # +0.0502 [+0.0143, +0.1002] "excludes zero" on twelve positive-by-
    # -construction values, and a review read that as the discriminator firing.
    imposed = {
        seed: imposed_lag_residual(
            runs[STRESS_SHARED][seed]["residual_curve"],
            runs[STRESS_ATTRIBUTED][seed]["asymptote"],
            runs[STRESS_ATTRIBUTED][seed]["tau"],
        )
        for seed in runs[STRESS_SHARED]
    }
    excess = {seed: imposed[seed] - runs[STRESS_SHARED][seed]["fit_residual"] for seed in imposed}
    null = within_control_excess(
        {PERSISTENCE_DECOUPLED: _as_fits(runs[STRESS_ATTRIBUTED])},
        {
            PERSISTENCE_DECOUPLED: {
                seed: run["residual_curve"] for seed, run in runs[STRESS_ATTRIBUTED].items()
            }
        },
    )
    print(
        contrast(
            "residual under a foreign lag, over what attributed does to itself",
            {seed: 0.0 for seed in excess},
            {seed: excess[seed] - null[seed] for seed in excess},
            resamples,
        )
    )
    print(
        f"    treatment excess / control excess  "
        f"{statistics.fmean(excess.values()):+.4f} / {statistics.fmean(null.values()):+.4f}"
        "   both positive by construction, so the row above is the reading"
    )

    print("\n== guardrails (must not move) and the secondaries ==")
    header = f"  {'arm':<14}{'residual':>10}{'tail loss':>11}{'r(w,c)':>9}{'floor':>8}"
    print(header + f"{'least share':>13}{'on-type':>9}{'charge/step':>13}")
    for arm in ARMS:
        row = {
            key: statistics.fmean(column(arm, key).values())
            for key in (
                "residual_stress",
                "tail_loss",
                "wealth_vs_competence",
                "floor_occupancy",
                "least_used_share",
                "on_type_share",
                "charge_per_step",
            )
        }
        print(
            f"  {arm:<14}{row['residual_stress']:>10.4f}{row['tail_loss']:>11.4f}"
            f"{row['wealth_vs_competence']:>9.3f}{row['floor_occupancy']:>8.2f}"
            f"{row['least_used_share']:>13.4f}{row['on_type_share']:>9.3f}"
            f"{row['charge_per_step']:>13.4f}"
        )
    return {
        arm: {
            seed: {
                key: value for key, value in run.items() if key not in ("residual_curve", "charges")
            }
            for seed, run in runs[arm].items()
        }
        for arm in ARMS
    } | {"_curves": {arm: {s: r["residual_curve"] for s, r in runs[arm].items()} for arm in ARMS}}


def read_control(
    seeds: tuple[int, ...], calibrations: dict[int, Calibration], resamples: int
) -> dict[str, Any]:
    """Shuffled stress: the same drain, from a tissue this one cannot affect."""
    print("\n== control: a replayed stress from another seed's shared run ==")
    shared: dict[str, float] = {}
    replayed: dict[str, float] = {}
    for index, seed in enumerate(seeds):
        donor = seeds[(index + 1) % len(seeds)]
        recorded = run_stepped(donor, STRESS_SHARED, calibrations[donor])["charges"]
        # The donor's charge is a per-step total over its tokens; the override
        # is a per-token magnitude. Undo the donor's own price *and* its token
        # count, so what is replayed is the donor's stress trajectory and what
        # prices it is the recipient's own budget.
        donated = calibrations[donor]
        scale = donated.equal_budget_lambda * donated.tokens
        charges = [charge / scale if scale > 0 else 0.0 for charge in recorded]
        shared[str(seed)] = run_stepped(seed, STRESS_SHARED, calibrations[seed])["residual_stress"]
        replayed[str(seed)] = run_stepped(seed, STRESS_SHARED, calibrations[seed], replay=charges)[
            "residual_stress"
        ]
        print(f"  seed {seed} replaying seed {donor}", flush=True)
    print(contrast("shuffled-stress - shared (residual stress)", shared, replayed, resamples))
    print("  binding that survives a replayed charge was the drain, not the stress")
    return {"shared": shared, "shuffled_stress": replayed}


def sweep_window(
    seeds: tuple[int, ...], calibrations: dict[int, Calibration], resamples: int
) -> dict[str, Any]:
    """The window: where binding shows and no guardrail moves, and where soup starts."""
    print("\n== the window: lambda x gamma, residual stress and what it cost ==")
    print(f"  {'lambda':>8}{'gamma':>7}{'residual':>10}{'tail loss':>11}{'floor':>8}{'on-type':>9}")
    window: dict[str, Any] = {}
    for multiple in LAMBDA_MULTIPLES:
        for gamma in GAMMAS:
            runs = {
                str(seed): run_stepped(
                    seed, STRESS_SHARED, calibrations[seed], multiple=multiple, gamma=gamma
                )
                for seed in seeds
            }
            keys = ("residual_stress", "tail_loss", "floor_occupancy", "on_type_share")
            row = {key: statistics.fmean(float(run[key]) for run in runs.values()) for key in keys}
            # The per-seed values beside the mean, because a window cell read as
            # a null needs the paired deltas `scripts/power.py` prices -- and
            # storing only the mean is how three nulls in this sweep came to be
            # stated as findings with no interval and nothing left to price them
            # with.
            window[f"lambda{multiple}-gamma{gamma}"] = row | {
                "per_seed": {
                    seed: {key: float(run[key]) for key in keys} for seed, run in runs.items()
                }
            }
            print(
                f"  {multiple:>8.2f}{gamma:>7.2f}{row['residual_stress']:>10.4f}"
                f"{row['tail_loss']:>11.4f}{row['floor_occupancy']:>8.2f}"
                f"{row['on_type_share']:>9.3f}"
            )
    print("  floor occupancy at 1.00 is soup: the ledger erased, not a tissue bound")
    return window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.home() / "tame-runs" / "59-stress")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--skip-window", action="store_true")
    # The control is the only stage whose numbers the #59 review moved, and the
    # arms it is read against are unaffected, so re-deriving it should not cost
    # the whole sweep again.
    parser.add_argument(
        "--only-control",
        action="store_true",
        help="Calibrate, then run the replayed-stress control alone and write it beside the rest",
    )
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(","))
    code_sha, code_dirty = code_identity()
    print(f"code {code_sha} dirty={code_dirty}; the differentiated fixture, {len(seeds)} seeds")

    print("\n== calibration: the resting spread, and the price that equalises the budgets ==")
    calibrations = {seed: calibrate(seed) for seed in seeds}
    for seed, calibration in calibrations.items():
        print(
            f"  seed {seed}: resting reading {calibration.resting_reading:.4f}, spread "
            f"{calibration.resting_sigma:.5f}, setpoint {calibration.setpoint:.4f}, mean stress "
            f"{calibration.mean_stress:.5f}, attributed {calibration.attributed_per_token:.5f} "
            f"per token, equal-budget lambda {calibration.equal_budget_lambda:.5f}"
        )

    record: dict[str, Any] = {
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "seeds": list(seeds),
        "calibration": {str(seed): value.as_dict() for seed, value in calibrations.items()},
    }
    if not args.only_control:
        record["arms"] = read_arms(seeds, calibrations, args.resamples)
    record["control"] = read_control(seeds, calibrations, args.resamples)
    if not (args.skip_window or args.only_control):
        record["window"] = sweep_window(seeds, calibrations, args.resamples)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stress_coupling.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nrecord: {args.out / 'stress_coupling.json'}")


if __name__ == "__main__":
    main()
