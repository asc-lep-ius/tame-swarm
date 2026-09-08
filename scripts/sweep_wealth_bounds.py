"""Re-derive the wealth band and its decay under the economy #9, #11 and #15 left (#16).

The four constants were fitted against a gate that no longer exists. Since then
#9 made payments quasi-linear transfers, #11 moved the gate to the log domain so
routing reads only *relative* wealth, and #15 redefined value as the counterfactual
against the shared base. What the band still shapes is three things:

1. **Selection.** Winners are ``argtopk(confidence x wealth)``, so the band's
   *ratio* is exactly how much report advantage a poor expert needs to overturn a
   rich one. At the shipped ``[15, 750]`` that is 50x.
2. **Price magnitude.** A winner pays ``b_(k+1) / w_j``, so a rich winner pays
   less per unit of externality than a poor one -- the band sets how strong that
   feedback is.
3. **Rebate size.** ``_compute_rebates`` divides by the harmonic mean of the *k*
   richest, so the band sets how much of the collection returns.

``wealth_decay`` is not a fourth knob of the same kind. Wealth moves by
``w <- decay * w + net``, so the ledger is a leaky integrator: it forgets with a
time constant of ``1 / (1 - decay)`` steps and settles where inflow balances the
leak, ``w* ~ net / (1 - decay)``. That makes decay the *memory horizon* -- how
long ago an expert's earnings still speak for it -- and it sets the equilibrium
scale the band has to contain. 0.997 is 333 steps, which is longer than any
damage episode the economy is asked to recover from.

So the sweep is over (ratio, decay), with the band centred geometrically on
``initial_wealth`` because the gate reads log-wealth; a separate pass varies the
absolute scale at fixed ratio, which the gate cannot see but prices and rebates
can. The shipped band is included as a literal row throughout.

What the aggregates cannot say is whether a band lets a damaged market re-form.
``--recovery`` runs the two damage protocols recorded as strict expected failures
in ``tests/test_homeostatic_recovery.py``, through the same
``scripts/economy_damage.py`` the suite asserts on, so a flip reported here is a
flip the suite sees.

Run:  uv run python scripts/sweep_wealth_bounds.py
      uv run python scripts/sweep_wealth_bounds.py --recovery
      uv run python scripts/sweep_wealth_bounds.py --share proportional
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from economy_damage import (  # noqa: E402
    LONG_FORCED_EPISODE,
    REGAINED_SHARE,
    SEEDS,
    TRACKING_AFTER_RELEASE,
    release_and_measure,
    ruin_and_measure,
)
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    MoBConfig,
    SyntheticEconomy,
    shuffled,
)

from mob.auction import ROUTING_SHARE_PROPORTIONAL  # noqa: E402
from specialisation import report_decisiveness  # noqa: E402

# Long enough that the ceiling transient is over: at the shipped band the leaders
# reach max_wealth within a few hundred steps, and a run that stops there reports
# the climb rather than what the economy settles into.
STEPS = 600
# Distinct winners are counted over the tail, where the reports are calibrated.
TAIL = 100
# Relative headroom for calling a float32 ledger "at" a bound.
CLAMP_TOLERANCE = 1e-4

# max_wealth / min_wealth: how many times better a poor expert's report must be to
# overturn a rich one's wealth. 1.0 makes wealth inert in selection, which is the
# limit worth having on the table -- it is what the mechanism reduces to if the
# ledger is not allowed to decide anything.
RATIOS = (1.0, 4.0, 10.0, 25.0, 50.0)
# Memory horizons 1/(1-decay) of 50, 100, 200, 333 steps, and 1.0: never forget.
DECAYS = (0.98, 0.99, 0.995, 0.997, 1.0)
# At fixed ratio and decay, what the absolute scale moves: nothing in the gate,
# which is scale invariant since #11, but prices go as 1/w and rebates with them.
SCALES = (7.5, 25.0, 75.0, 250.0, 750.0)


@dataclass(frozen=True)
class Band:
    """A candidate setting of the four constants, and how to name it in a table."""

    label: str
    initial_wealth: float
    min_wealth: float
    max_wealth: float
    wealth_decay: float

    @property
    def ratio(self) -> float:
        return self.max_wealth / self.min_wealth

    def config(self, base: MoBConfig = BASE_CONFIG, **overrides) -> MoBConfig:
        return replace(
            base,
            initial_wealth=self.initial_wealth,
            min_wealth=self.min_wealth,
            max_wealth=self.max_wealth,
            wealth_decay=self.wealth_decay,
            **overrides,
        )


SHIPPED = Band("shipped", 75.0, 15.0, 750.0, 0.997)


def centred(ratio: float, decay: float, scale: float = 75.0) -> Band:
    """A band of the given ratio centred geometrically on ``scale``.

    Geometric rather than arithmetic centring because the gate reads
    ``log(confidence) + log(wealth)``: the distance an expert has to travel to
    overturn another is a ratio, so the midpoint that leaves an expert equally far
    from both bounds is the geometric one.
    """
    half = math.sqrt(ratio)
    return Band(f"r={ratio:g} d={decay:g}", scale, scale / half, scale * half, decay)


@dataclass
class Reading:
    """What one run of one band settled into."""

    ceiling_occupancy: float
    floor_occupancy: float
    steps_to_ceiling: int | None
    gini: float
    mean_wealth: float
    floor_share: float
    overturn: float
    distinct_winners: float
    charge_per_step: float
    rebate_fraction: float
    top1: float
    effective_experts: float


def gini(wealth: torch.Tensor) -> float:
    ordered = torch.sort(wealth)[0]
    n = len(ordered)
    index = torch.arange(1, n + 1, dtype=ordered.dtype)
    return ((2 * (index * ordered).sum()) / (n * ordered.sum()) - (n + 1) / n).abs().item()


def _overturned(confidences: torch.Tensor, wealth: torch.Tensor, top_k: int) -> float:
    """Fraction of tokens where wealth, not the report, decides the top-1 winner.

    This is ``1 - spec/report_decisiveness`` and is computed by that very
    function, so the number the sweep chooses a band on and the number a training
    run logs cannot drift apart. It is the statistic #16 asks for -- how often
    wealth overturns the ranking the strategyproofness argument is about.

    The winner is recomputed from the reports and the wealth the gate saw rather
    than read off ``selected_experts``, so the exploration slot -- drawn before
    any report and displacing a winner on 2% of training tokens -- does not read
    as wealth overturning a report. A probe pass measures it the same way, in
    eval mode, where exploration is off.
    """
    weighted = torch.topk(confidences * wealth, top_k, dim=-1).indices
    return 1.0 - report_decisiveness(weighted, confidences)


def measure(band: Band, seed: int, steps: int, share: str) -> Reading:
    config = band.config(routing_share=share)
    economy = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)
    mob = economy.mob

    collected: list[float] = []
    returned: list[float] = []
    original = mob._vcg_charges

    def spy(payments, selected, num_tokens, reward_multiplier, rebates=None, valid_mask=None):
        net = original(payments, selected, num_tokens, reward_multiplier, rebates, valid_mask)
        gross = original(payments, selected, num_tokens, reward_multiplier, None, valid_mask)
        collected.append(gross.sum().item())
        returned.append((gross - net).sum().item())
        return net

    mob._vcg_charges = spy

    ceiling = config.max_wealth * (1 - CLAMP_TOLERANCE)
    floor = config.min_wealth * (1 + CLAMP_TOLERANCE)
    at_ceiling = at_floor = 0
    steps_to_ceiling: int | None = None
    overturns: list[float] = []
    top1s: list[float] = []
    effective: list[float] = []
    tail_wins = torch.zeros(config.num_experts)

    for step in range(steps):
        record = economy.step()
        stats = mob.last_stats
        assert stats is not None
        overturns.append(_overturned(stats.confidences, stats.expert_wealth, config.top_k))
        top1s.append(stats.routing.top1_mean.item())
        effective.append(stats.routing.effective_experts.item())

        wealth = mob.expert_wealth
        ceiling_now = int((wealth >= ceiling).sum())
        at_ceiling += ceiling_now
        at_floor += int((wealth <= floor).sum())
        if ceiling_now and steps_to_ceiling is None:
            steps_to_ceiling = step + 1
        if step >= steps - TAIL:
            tail_wins += torch.bincount(
                record.selected_experts.flatten(), minlength=config.num_experts
            ).float()

    wealth = mob.expert_wealth
    expert_steps = steps * config.num_experts
    floored = int((wealth <= floor).sum())
    return Reading(
        ceiling_occupancy=at_ceiling / expert_steps,
        floor_occupancy=at_floor / expert_steps,
        steps_to_ceiling=steps_to_ceiling,
        gini=gini(wealth),
        mean_wealth=wealth.mean().item(),
        # What share of the surviving wealth is the floor itself rather than
        # anything the economy paid: the direct read on "the floor holds up the
        # mean". It is a lower bound on the support, since an expert above the
        # floor may have been held up by it earlier in the run.
        floor_share=floored * config.min_wealth / wealth.sum().item(),
        overturn=statistics.mean(overturns),
        distinct_winners=float((tail_wins > 0).sum()),
        charge_per_step=statistics.mean(collected),
        rebate_fraction=sum(returned) / max(sum(collected), 1e-12),
        top1=statistics.mean(top1s),
        effective_experts=statistics.mean(effective),
    )


def _aggregate(band: Band, seeds: tuple[int, ...], steps: int, share: str) -> Reading:
    runs = [measure(band, seed, steps, share) for seed in seeds]
    reached = [r.steps_to_ceiling for r in runs if r.steps_to_ceiling is not None]
    return Reading(
        ceiling_occupancy=statistics.mean(r.ceiling_occupancy for r in runs),
        floor_occupancy=statistics.mean(r.floor_occupancy for r in runs),
        # The median of the seeds that reached it at all; a band where only some
        # seeds clamp is reported by the occupancy column beside it.
        steps_to_ceiling=int(statistics.median(reached)) if reached else None,
        gini=statistics.mean(r.gini for r in runs),
        mean_wealth=statistics.mean(r.mean_wealth for r in runs),
        floor_share=statistics.mean(r.floor_share for r in runs),
        overturn=statistics.mean(r.overturn for r in runs),
        distinct_winners=statistics.mean(r.distinct_winners for r in runs),
        charge_per_step=statistics.mean(r.charge_per_step for r in runs),
        rebate_fraction=statistics.mean(r.rebate_fraction for r in runs),
        top1=statistics.mean(r.top1 for r in runs),
        effective_experts=statistics.mean(r.effective_experts for r in runs),
    )


HEADER = (
    f"{'band':>14} {'min':>7} {'max':>8} {'decay':>6} "
    f"{'ceil%':>6} {'t_ceil':>7} {'floor%':>7} {'flr/W':>6} "
    f"{'gini':>6} {'meanW':>8} {'overturn':>9} {'wins':>5} "
    f"{'chg/step':>9} {'rebate':>7} {'top1':>6} {'n_eff':>6}"
)


def _row(band: Band, reading: Reading) -> str:
    reached = f"{reading.steps_to_ceiling}" if reading.steps_to_ceiling is not None else "--"
    # At ratio 1 the two bounds coincide, so every expert is at both of them on
    # every step by construction and the four clamp columns measure the band's
    # definition rather than the economy's behaviour.
    clamps = (
        f"{'n/a':>6} {'n/a':>7} {'n/a':>7} {'n/a':>6}"
        if band.ratio == 1.0
        else (
            f"{100 * reading.ceiling_occupancy:>5.1f}% {reached:>7} "
            f"{100 * reading.floor_occupancy:>6.1f}% {reading.floor_share:>6.2f}"
        )
    )
    return (
        f"{band.label:>14} {band.min_wealth:>7.1f} {band.max_wealth:>8.1f} "
        f"{band.wealth_decay:>6.3f} {clamps} "
        f"{reading.gini:>6.3f} {reading.mean_wealth:>8.1f} "
        f"{100 * reading.overturn:>8.1f}% {reading.distinct_winners:>5.1f} "
        f"{reading.charge_per_step:>9.3f} {100 * reading.rebate_fraction:>6.1f}% "
        f"{reading.top1:>6.3f} {reading.effective_experts:>6.3f}"
    )


def sweep_bands(seeds: tuple[int, ...], steps: int, share: str) -> None:
    print(f"\n=== band x decay, routing_share={share}, {steps} steps, seeds {seeds} ===\n")
    print(HEADER)
    print(_row(SHIPPED, _aggregate(SHIPPED, seeds, steps, share)))
    for ratio in RATIOS:
        for decay in DECAYS:
            band = centred(ratio, decay)
            print(_row(band, _aggregate(band, seeds, steps, share)))


def sweep_scale(seeds: tuple[int, ...], steps: int, share: str) -> None:
    print(f"\n=== absolute scale at ratio 10, decay 0.99, routing_share={share} ===\n")
    print(HEADER)
    for scale in SCALES:
        band = centred(10.0, 0.99, scale)
        print(_row(replace(band, label=f"w0={scale:g}"), _aggregate(band, seeds, steps, share)))


def sweep_recovery(candidates: list[Band], seeds: tuple[int, ...]) -> None:
    """The two strict expected failures, run against each candidate band.

    Both are pass/fail claims about whether a damaged market re-forms, which no
    aggregate above can stand in for. The thresholds are the suite's own.
    """
    print(f"\n=== the two expected failures, seeds {seeds} ===\n")
    print(
        f"{'band':>16} {'min':>7} {'max':>8} {'decay':>6} "
        f"{'r(share,comp)':>14} {'regained':>9} {'loss/steady':>12} {'forced':>7} "
        f"{'ruined share':>13} {'w/median':>9} {'ruin':>6}"
    )
    chance = 1.0 / DEFAULT_COMPETENCE.numel()
    for band in candidates:
        config = band.config()
        forced = [release_and_measure(seed, LONG_FORCED_EPISODE, config) for seed in seeds]
        ruined = [ruin_and_measure(seed, config) for seed in seeds]

        # Both gates are all-seeds, so the seed that decides them is the worst one
        # on each statistic; a mean over three seeds sits comfortably inside a
        # threshold that one of them misses.
        tracking = min(f[1] for f in forced)
        regained = min(f[2] for f in forced)
        loss_ratio = max(f[0] for f in forced)
        forced_passes = (
            tracking > TRACKING_AFTER_RELEASE and regained > REGAINED_SHARE and loss_ratio <= 1.0
        )

        share = min(r[0] for r in ruined)
        wealth_ratio = min(r[1] / max(r[2], 1e-9) for r in ruined)
        ruin_passes = all(r[0] > chance and r[1] > r[2] for r in ruined)

        print(
            f"{band.label:>16} {band.min_wealth:>7.1f} {band.max_wealth:>8.1f} "
            f"{band.wealth_decay:>6.3f} "
            f"{tracking:>14.2f} {regained:>9.2f} {loss_ratio:>12.2f} "
            f"{'PASS' if forced_passes else 'fail':>7} "
            f"{share:>13.3f} {wealth_ratio:>9.2f} "
            f"{'PASS' if ruin_passes else 'fail':>6}"
        )
    print(
        f"\nEvery column is the worst of the {len(seeds)} seeds, which is what the "
        f"all-seeds gate reads.\nThresholds are the suite's: forced -- "
        f"r > {TRACKING_AFTER_RELEASE}, "
        f"regained > {REGAINED_SHARE}, loss <= 1.0x steady, on every seed; "
        f"ruin -- share > 1/8 and wealth > median, on every seed.\n"
        "PASS means the strict xfail of that name would flip."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--share", default=BASE_CONFIG.routing_share)
    parser.add_argument("--recovery", action="store_true", help="run the two damage protocols")
    parser.add_argument("--scale", action="store_true", help="vary the absolute scale only")
    parser.add_argument(
        "--band",
        nargs=4,
        metavar=("W0", "MIN", "MAX", "DECAY"),
        type=float,
        action="append",
        help="an extra candidate band, repeatable",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    extra = [
        Band(f"w0={w0:g} r={mx / mn:g}", w0, mn, mx, decay)
        for w0, mn, mx, decay in (args.band or [])
    ]

    if args.recovery:
        sweep_recovery([SHIPPED, *extra], seeds)
        return
    if args.scale:
        sweep_scale(seeds, args.steps, args.share)
        return

    sweep_bands(seeds, args.steps, args.share)
    if args.share == ROUTING_SHARE_PROPORTIONAL:
        print(
            "\ntop1 and n_eff are the #11 gate diagnostics and only move under this "
            "share;\nthe uniform split the auction is strategyproof under fixes them "
            "at 1/top_k and top_k."
        )


if __name__ == "__main__":
    main()
