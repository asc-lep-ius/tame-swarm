"""Re-derive the wealth band and its decay under the economy #9, #11 and #15 left (#16).

The four constants were fitted against a gate that no longer exists. Since then
#9 made payments quasi-linear transfers, #11 moved the gate to the log domain so
routing reads only *relative* wealth, and #15 redefined value as the counterfactual
against the shared base. What the band still shapes is three things:

1. **Selection.** Winners are ``argtopk(confidence x wealth)``, so the band's
   *ratio* is exactly how much report advantage a poor expert needs to overturn a
   rich one. At the shipped ``[15, 750]`` that is 50x.
2. **Price magnitude, in relative terms only.** A winner pays ``b_(k+1) / w_j``
   and ``b_(k+1)`` is itself ``c_(k+1) x w_(k+1)``, so a price is a *ratio* of
   wealths and is exactly invariant to rescaling the whole ledger -- measured
   unchanged to 7 significant figures at 0.1x and 10x. So is the rebate, which
   divides a bid by a harmonic mean of wealths. What the band sets is the price a
   given *spread* produces, not a price level.
3. **The reward is the one quantity that is not scale-free.** It is a loss
   reduction and carries no wealth at all, so the inflow is an absolute number of
   credits per step while every outflow is a ratio. That asymmetry is the whole
   reason the band has to be *placed*, and it is what the scale pass measures.

``wealth_decay`` is not a fourth knob of the same kind. Wealth moves by
``w <- decay * w + net``, so the ledger is a leaky integrator: it forgets with a
time constant of ``1 / (1 - decay)`` steps and settles where inflow balances the
leak, at the fixed point of ``w = (decay * w) + net(w)``. Because ``net`` is an
absolute inflow, that equilibrium is an absolute wealth -- and the thing that has
to contain it is the **band**. ``--start`` and ``--scale`` separate the two:
holding the shipped band and moving only ``initial_wealth`` from 25 to 750 leaves
the settled Gini at 0.691-0.693 and the mean wealth at 199, while sliding the
whole band at a fixed ratio takes the settled Gini from 0.000 to 0.519. What
``initial_wealth`` does *not* leave alone is which experts end up rich: floor
occupancy and the overturn rate move with it, because that is a path-dependent
selection outcome and the leaky-integrator time constant bounds only the leak.

So the sweep is over (ratio, decay), with the band centred geometrically because
the gate reads log-wealth; a separate pass slides the whole band, keeping the
ratio fixed, which is what changes whether the equilibrium falls inside it. The
shipped band is included as a literal row throughout.

**Every steady-state statistic is read over the last ``TAIL`` steps**, and the
step budget scales with the decay, because a fixed budget is a different number
of memory horizons at each one: 600 steps is twelve horizons at 0.98 and 1.8 at
0.997, which made the slower decays report their transient rather than what they
settle into (shipped-band Gini 0.48 at 600 steps against 0.69 settled).

What the aggregates cannot say is whether a band lets a damaged market re-form.
``--recovery`` runs the two damage protocols recorded as strict expected failures
in ``tests/test_homeostatic_recovery.py``, through the same
``scripts/economy_damage.py`` the suite asserts on, so a flip reported here is a
flip the suite sees.

Run:  uv run python scripts/sweep_wealth_bounds.py
      uv run python scripts/sweep_wealth_bounds.py --share proportional
      uv run python scripts/sweep_wealth_bounds.py --scale
      uv run python scripts/sweep_wealth_bounds.py --recovery
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
    SyntheticEconomy,
    shuffled,
)

from mob import MoBConfig  # noqa: E402
from mob.auction import ROUTING_SHARE_PROPORTIONAL  # noqa: E402
from specialisation import report_decisiveness  # noqa: E402

# A fixed step budget is a different number of memory horizons at each decay --
# 600 steps is twelve at 0.98 and 1.8 at 0.997 -- so the slow decays would report
# their transient. Eight horizons leaves under 0.04% of the initial condition.
HORIZONS = 8
MIN_STEPS = 600
# Every steady-state statistic is read over this many final steps, so rows with
# different budgets are still compared over the same window.
TAIL = 500
# Relative headroom for calling a float32 ledger "at" a bound. The two bounds need
# different tolerances and it is not a fudge: the ceiling is an attractor from
# above, so a clamped expert sits *exactly* on it and float32 headroom is all that
# is wanted. The floor is escaped by a hair every time an exploration win lands and
# then decays back, so the same tolerance systematically undercounts it -- measured
# at the shipped band, 1e-4 reads 34% where 10% reads 74%, and no floor-bound expert
# ever exceeds 17.1 out of 750. A tenth of the floor is below any wealth an expert
# holds a slot on and above the exploration jitter.
CEILING_TOLERANCE = 1e-4
FLOOR_TOLERANCE = 0.10

# max_wealth / min_wealth: how many times better a poor expert's report must be to
# overturn a rich one. 1.0 makes wealth inert in selection, which is the limit
# worth having on the table -- it is what the mechanism reduces to if the ledger is
# not allowed to decide anything.
RATIOS = (1.0, 4.0, 10.0, 25.0, 50.0)
# Memory horizons 1/(1-decay) of 50, 100, 200, 333 steps, and 1.0: never forget,
# which has no equilibrium at all and is run at the longest finite budget.
DECAYS = (0.98, 0.99, 0.995, 0.997, 1.0)
# At a fixed ratio, sliding the whole band is what changes whether the economy's
# equilibrium falls inside it. The gate cannot see this -- and neither can prices
# or rebates, which are ratios of wealths and exactly scale-free. What is not
# scale-free is the reward, so the inflow is an absolute number of credits and the
# band is what has to be placed around the wealth it accumulates to.
SCALES = (7.5, 25.0, 75.0, 250.0, 750.0)


def horizon(decay: float) -> int:
    """Step budget for a decay: eight memory horizons.

    Only ``decay >= 1.0`` is clamped, and because it has no equilibrium to settle
    to rather than because 2667 steps is enough for it. A ``min(decay, slowest)``
    here would silently send 0.999 to 2.7 horizons while the legend claims eight
    -- republishing a transient under a settled label, which is the defect the
    horizon rule exists to remove.
    """
    if decay >= 1.0:
        decay = max(d for d in DECAYS if d < 1.0)
    return max(MIN_STEPS, math.ceil(HORIZONS / (1.0 - decay)))


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
    """What one band settled into. Every rate is over the final ``TAIL`` steps."""

    ceiling_occupancy: float
    floor_occupancy: float
    steps_to_ceiling: int | None
    seeds_reaching_ceiling: int
    seeds: int
    gini: float
    gini_spread: float
    mean_wealth: float
    floor_share: float
    interior_occupancy: float
    overturn: float
    overturn_spread: float
    distinct_winners: float
    charge_per_step: float
    rebate_fraction: float
    top1: float
    effective_experts: float
    steps: int


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


def _charge_spy(mob) -> tuple[list[float], list[float]]:
    """Record gross collection and gross rebate per settlement, in credits.

    ``_vcg_charges`` allocates its own accumulator and mutates no instance state,
    so calling it a second time with ``rebates=None`` is free of side effects and
    the difference is exactly what the redistribution returned. Shadowing the
    bound method is the pattern ``sweep_payment_scale.py`` already uses.
    """
    collected: list[float] = []
    returned: list[float] = []
    original = mob._vcg_charges

    def spy(
        payments: torch.Tensor | None,
        selected: torch.Tensor,
        num_tokens: int,
        reward_multiplier: float,
        rebates: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        net = original(payments, selected, num_tokens, reward_multiplier, rebates, valid_mask)
        gross = original(payments, selected, num_tokens, reward_multiplier, None, valid_mask)
        collected.append(gross.sum().item())
        returned.append((gross - net).sum().item())
        return net

    mob._vcg_charges = spy
    return collected, returned


@dataclass
class _Accumulator:
    """Per-step tallies for one run, so ``measure`` stays a description of the run."""

    num_experts: int
    ceiling: float
    floor: float
    at_ceiling: int = 0
    at_floor: int = 0
    in_interior: int = 0
    steps_to_ceiling: int | None = None

    def __post_init__(self) -> None:
        self.overturns: list[float] = []
        self.top1s: list[float] = []
        self.effective: list[float] = []
        self.wins = torch.zeros(self.num_experts)

    def see_clamps(self, wealth: torch.Tensor, step: int, in_tail: bool) -> None:
        """Time-to-first-clamp is a whole-run event; occupancy is a tail rate."""
        if self.steps_to_ceiling is None and int((wealth >= self.ceiling).sum()):
            self.steps_to_ceiling = step + 1
        if not in_tail:
            return
        self.at_ceiling += int((wealth >= self.ceiling).sum())
        self.at_floor += int((wealth <= self.floor).sum())
        self.in_interior += int(((wealth > self.floor) & (wealth < self.ceiling)).sum())

    def see_step(self, record, stats, top_k: int) -> None:
        self.overturns.append(_overturned(stats.confidences, stats.expert_wealth, top_k))
        self.top1s.append(stats.routing.top1_mean.item())
        self.effective.append(stats.routing.effective_experts.item())
        self.wins += torch.bincount(
            record.selected_experts.flatten(), minlength=self.num_experts
        ).float()


def measure(band: Band, seed: int, steps: int, share: str) -> Reading:
    config = band.config(routing_share=share)
    economy = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)
    mob = economy.mob
    collected, returned = _charge_spy(mob)
    tally = _Accumulator(
        num_experts=config.num_experts,
        ceiling=config.max_wealth * (1 - CEILING_TOLERANCE),
        floor=config.min_wealth * (1 + FLOOR_TOLERANCE),
    )
    tail_from = steps - TAIL

    for step in range(steps):
        record = economy.step()
        stats = mob.last_stats
        assert stats is not None
        in_tail = step >= tail_from
        tally.see_clamps(mob.expert_wealth, step, in_tail)
        if in_tail:
            tally.see_step(record, stats, config.top_k)

    wealth = mob.expert_wealth
    expert_steps = TAIL * config.num_experts
    return Reading(
        ceiling_occupancy=tally.at_ceiling / expert_steps,
        floor_occupancy=tally.at_floor / expert_steps,
        steps_to_ceiling=tally.steps_to_ceiling,
        seeds_reaching_ceiling=int(tally.steps_to_ceiling is not None),
        seeds=1,
        gini=gini(wealth),
        gini_spread=0.0,
        mean_wealth=wealth.mean().item(),
        # The share of surviving wealth that sits on the floor: the direct read on
        # "does the floor hold up the mean".
        floor_share=wealth[wealth <= tally.floor].sum().item() / wealth.sum().item(),
        interior_occupancy=tally.in_interior / expert_steps,
        overturn=statistics.mean(tally.overturns),
        overturn_spread=0.0,
        distinct_winners=float((tally.wins > 0).sum()),
        charge_per_step=statistics.mean(collected[tail_from:]),
        rebate_fraction=sum(returned[tail_from:]) / max(sum(collected[tail_from:]), 1e-12),
        top1=statistics.mean(tally.top1s),
        effective_experts=statistics.mean(tally.effective),
        steps=steps,
    )


def _aggregate(band: Band, seeds: tuple[int, ...], share: str) -> Reading:
    steps = horizon(band.wealth_decay)
    runs = [measure(band, seed, steps, share) for seed in seeds]
    reached = [r.steps_to_ceiling for r in runs if r.steps_to_ceiling is not None]
    ginis = [r.gini for r in runs]
    overturns = [r.overturn for r in runs]
    return Reading(
        ceiling_occupancy=statistics.mean(r.ceiling_occupancy for r in runs),
        floor_occupancy=statistics.mean(r.floor_occupancy for r in runs),
        # The median of the seeds that reached it at all; the count beside it says
        # how many did, so a band where one seed clamps cannot read like one where
        # all three do.
        steps_to_ceiling=int(statistics.median(reached)) if reached else None,
        seeds_reaching_ceiling=len(reached),
        seeds=len(runs),
        gini=statistics.mean(ginis),
        gini_spread=statistics.pstdev(ginis),
        mean_wealth=statistics.mean(r.mean_wealth for r in runs),
        floor_share=statistics.mean(r.floor_share for r in runs),
        interior_occupancy=statistics.mean(r.interior_occupancy for r in runs),
        overturn=statistics.mean(overturns),
        overturn_spread=statistics.pstdev(overturns),
        distinct_winners=statistics.mean(r.distinct_winners for r in runs),
        charge_per_step=statistics.mean(r.charge_per_step for r in runs),
        rebate_fraction=statistics.mean(r.rebate_fraction for r in runs),
        top1=statistics.mean(r.top1 for r in runs),
        effective_experts=statistics.mean(r.effective_experts for r in runs),
        steps=runs[0].steps,
    )


HEADER = (
    f"{'band':>14} {'min':>7} {'max':>8} {'decay':>6} {'steps':>6} "
    f"{'ceil%':>6} {'t_ceil':>10} {'floor%':>7} {'mid%':>6} {'flr/W':>6} "
    f"{'gini':>15} {'meanW':>8} {'overturn':>16} {'wins':>5} "
    f"{'chg/step':>9} {'rebate':>7} {'top1':>6} {'n_eff':>6}"
)


def _row(band: Band, reading: Reading) -> str:
    if reading.steps_to_ceiling is None:
        reached = "--"
    else:
        reached = f"{reading.steps_to_ceiling} ({reading.seeds_reaching_ceiling}/{reading.seeds})"
    # At ratio 1 the two bounds coincide, so every expert is at both of them on
    # every step by construction and the four clamp columns measure the band's
    # definition rather than the economy's behaviour.
    clamps = (
        f"{'n/a':>6} {'n/a':>10} {'n/a':>7} {'n/a':>6} {'n/a':>6}"
        if band.ratio == 1.0
        else (
            f"{100 * reading.ceiling_occupancy:>5.1f}% {reached:>10} "
            f"{100 * reading.floor_occupancy:>6.1f}% "
            f"{100 * reading.interior_occupancy:>5.1f}% {reading.floor_share:>6.2f}"
        )
    )
    return (
        f"{band.label:>14} {band.min_wealth:>7.1f} {band.max_wealth:>8.1f} "
        f"{band.wealth_decay:>6.3f} {reading.steps:>6} {clamps} "
        f"{reading.gini:>7.3f}+-{reading.gini_spread:<6.3f} {reading.mean_wealth:>8.1f} "
        f"{100 * reading.overturn:>8.1f}%+-{100 * reading.overturn_spread:<5.1f} "
        f"{reading.distinct_winners:>5.1f} "
        f"{reading.charge_per_step:>9.3f} {100 * reading.rebate_fraction:>6.1f}% "
        f"{reading.top1:>6.3f} {reading.effective_experts:>6.3f}"
    )


def _legend(seeds: tuple[int, ...]) -> str:
    return (
        f"\nRates are over the last {TAIL} steps; gini, meanW and flr/W are the final "
        f"state. `mid%` is occupancy of the\nband's interior -- near zero means the "
        f"ledger has saturated into a lattice and its Gini is then a closed-form\n"
        f"function of the ratio and top_k rather than a reading of the economy. The "
        f"floor is counted within {100 * FLOOR_TOLERANCE:.0f}% of\nitself and the "
        f"ceiling within {CEILING_TOLERANCE:g}; see CEILING_TOLERANCE for why they "
        f"differ. "
        f"+- is the population sd over the {len(seeds)} seeds.\nThe step budget is "
        f"{HORIZONS} memory horizons 1/(1-decay), floored at {MIN_STEPS}, so rows at "
        "different decays are compared\nat the same distance from their own equilibrium; "
        "decay 1.0 has no equilibrium and is run at the longest finite budget."
    )


def sweep_bands(seeds: tuple[int, ...], share: str) -> None:
    print(f"\n=== band x decay, routing_share={share}, seeds {seeds} ===\n")
    print(HEADER)
    print(_row(SHIPPED, _aggregate(SHIPPED, seeds, share)))
    for ratio in RATIOS:
        for decay in DECAYS:
            band = centred(ratio, decay)
            print(_row(band, _aggregate(band, seeds, share)))
    print(_legend(seeds))


def sweep_scale(seeds: tuple[int, ...], share: str, decay: float, ratio: float = 10.0) -> None:
    """Slide the whole band at a fixed ratio: what decides if the equilibrium fits.

    This moves ``min_wealth``, ``max_wealth`` and ``initial_wealth`` together, so
    it says what the band's *location* does and cannot attribute anything to
    ``initial_wealth`` alone -- for that, hold the band and move only the start.
    """
    print(f"\n=== whole band slid at ratio {ratio:g}, decay {decay:g}, share={share} ===\n")
    print(HEADER)
    for scale in SCALES:
        band = centred(ratio, decay, scale)
        print(_row(replace(band, label=f"w0={scale:g}"), _aggregate(band, seeds, share)))
    print(_legend(seeds))


def sweep_start(seeds: tuple[int, ...], share: str, band: Band = SHIPPED) -> None:
    """Hold the band and move only ``initial_wealth``: what the starting point buys.

    The companion to ``sweep_scale``, and the one that separates the two. Whatever
    moves here is the transient; whatever does not is the band's.
    """
    print(
        f"\n=== initial_wealth alone, band [{band.min_wealth:g}, {band.max_wealth:g}], "
        f"decay {band.wealth_decay:g}, share={share} ===\n"
    )
    print(HEADER)
    for start in SCALES:
        if not band.min_wealth <= start <= band.max_wealth:
            continue
        moved = replace(band, label=f"w0={start:g}", initial_wealth=start)
        print(_row(moved, _aggregate(moved, seeds, share)))
    print(_legend(seeds))


def sweep_advantage(seeds: tuple[int, ...], share: str) -> None:
    """What report advantage competence buys, and at which horizon it was read.

    The quantity ``test_the_band_ratio_bounds_the_report_advantage_demanded_of_the
    _poorest`` compares the band's ratio against: the most competent expert's mean
    report over the least competent one's, on the planted-competence fixture.

    It exists as a mode because it is **not a constant of the fixture** and a
    hardcoded value would be unsourced. The heads keep calibrating, so it grows
    with the training horizon -- roughly 2.3 at the 400 steps the damage protocols
    run, roughly 5.4 at the sweep's settled budget.

    The flat band is printed beside it as a control, and it reads *higher* at the
    settled budget rather than lower: a band that shuts six experts out also stops
    their heads seeing targets, so it suppresses the very advantage it is then
    compared against. Which of the two is the fair comparison is arguable -- the
    test asserts against both, so nothing rests on the choice.
    """
    print(f"\n=== report advantage competence buys, seeds {seeds}, share={share} ===\n")
    print(f"{'horizon':>9} {'band':>10} " + " ".join(f"{'seed' + str(s):>7}" for s in seeds))
    flat = replace(SHIPPED, min_wealth=SHIPPED.initial_wealth, max_wealth=SHIPPED.initial_wealth)
    for steps in (400, horizon(SHIPPED.wealth_decay)):
        for label, band in (("shipped", SHIPPED), ("flat", flat)):
            values = [_report_advantage(band, seed, steps, share) for seed in seeds]
            print(
                f"{steps:>9} {label:>10} "
                + " ".join(f"{v:>7.2f}" for v in values)
                + f"   mean {statistics.mean(values):>5.2f}"
            )
    print(
        "\nThe band's ratio is what this is compared against. The shipped 50x exceeds "
        "every reading\nhere -- by about 7x the largest under the band itself, and by "
        "about 3x the largest with the\nledger flat, which is the control and runs "
        "higher because a band that shuts six experts out\nalso stops their heads "
        "calibrating. The demand is never met either way."
    )


def _report_advantage(band: Band, seed: int, steps: int, share: str) -> float:
    """Most-competent over least-competent expert's mean report, over the tail."""
    config = band.config(routing_share=share)
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best, worst = int(competence.argmax()), int(competence.argmin())
    economy = SyntheticEconomy(competence, seed=seed, config=config)
    window = min(TAIL, steps // 2)
    reports = []
    for step in range(steps):
        economy.step()
        if step >= steps - window:
            stats = economy.mob.last_stats
            assert stats is not None
            reports.append(stats.confidences.reshape(-1, config.num_experts).mean(0))
    mean_report = torch.stack(reports).mean(0)
    return float(mean_report[best] / mean_report[worst])


def _recovery_row(band: Band, seeds: tuple[int, ...], share: str) -> str:
    """One band's line of the damage table, verdicts on ``seeds[0]`` with a seed count."""
    config = band.config(routing_share=share)
    forced = [release_and_measure(seed, LONG_FORCED_EPISODE, config) for seed in seeds]
    ruined = [ruin_and_measure(seed, config) for seed in seeds]
    chance = 1.0 / DEFAULT_COMPETENCE.numel()

    def verdict(passes: list[bool]) -> str:
        return f"{'PASS' if passes[0] else 'fail'} ({sum(passes)}/{len(passes)})"

    forced_pass = [
        f.tracking > TRACKING_AFTER_RELEASE and f.regained > REGAINED_SHARE and f.loss_ratio <= 1.0
        for f in forced
    ]
    # At ratio 1 the clamp restores a zeroed wealth before anything reads it, so the
    # ruin protocol never damages anything and its verdicts say nothing.
    degenerate = " (no damage)" if band.ratio == 1.0 else ""
    return (
        f"{band.label:>16} {band.min_wealth:>7.1f} {band.max_wealth:>8.1f} "
        f"{band.wealth_decay:>6.3f} "
        f"{forced[0].tracking:>14.2f} {forced[0].regained:>9.2f} "
        f"{forced[0].loss_ratio:>12.2f} {verdict(forced_pass):>10} "
        f"{ruined[0][0]:>11.3f} {verdict([r[0] > chance for r in ruined]):>11} "
        f"{ruined[0][1] / max(ruined[0][2], 1e-9):>9.2f} "
        f"{verdict([r[1] > r[2] for r in ruined]):>12}{degenerate}"
    )


def sweep_recovery(candidates: list[Band], seeds: tuple[int, ...], share: str) -> None:
    """The three strict expected failures, run against each candidate band.

    Pass/fail claims about whether a damaged market re-forms, which no aggregate
    above can stand in for. The protocols come from ``economy_damage`` and their
    horizons are the suite's fixed ones, *not* scaled with the decay, because the
    question is whether the tests as written would flip.

    The verdict columns are evaluated on ``seeds[0]`` alone, because that is the
    seed all three tests run; the count beside each says how many of the seeds
    swept agree, so a band that flips only the asserted seed is visible as such.
    """
    print(f"\n=== the three expected failures, seeds {seeds}, share={share} ===\n")
    print(
        f"{'band':>16} {'min':>7} {'max':>8} {'decay':>6} "
        f"{'r(share,comp)':>14} {'regained':>9} {'loss/steady':>12} {'forced':>10} "
        f"{'ruin share':>11} {'-> market':>11} {'w/median':>9} {'-> standing':>12}"
    )
    for band in candidates:
        print(_recovery_row(band, seeds, share))
    print(
        f"\nStatistics are seed {seeds[0]}'s, which is the seed all three tests run; "
        f"(n/{len(seeds)}) counts how many\nof the swept seeds agree with that verdict. "
        f"Thresholds are the suite's: forced -- r > {TRACKING_AFTER_RELEASE}, regained > "
        f"{REGAINED_SHARE},\nloss <= 1.0x steady, all three; market -- share > 1/8; "
        "standing -- wealth > median.\nPASS means the strict xfail of that name would flip "
        "and turn the suite red."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--share", default=BASE_CONFIG.routing_share)
    parser.add_argument("--recovery", action="store_true", help="run the damage protocols")
    parser.add_argument("--scale", action="store_true", help="slide the whole band")
    parser.add_argument("--start", action="store_true", help="move initial_wealth alone")
    parser.add_argument(
        "--advantage", action="store_true", help="what report advantage competence buys"
    )
    parser.add_argument(
        "--decay", type=float, default=SHIPPED.wealth_decay, help="for --scale only"
    )
    parser.add_argument(
        "--quick", action="store_true", help="with --recovery, skip the ratio x decay grid"
    )
    parser.add_argument(
        "--band",
        nargs=4,
        metavar=("W0", "MIN", "MAX", "DECAY"),
        type=float,
        action="append",
        help="an extra candidate band, repeatable",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    seeds = tuple(args.seeds)

    extra = [
        Band(f"w0={w0:g} r={mx / mn:g}", w0, mn, mx, decay)
        for w0, mn, mx, decay in (args.band or [])
    ]

    if args.recovery:
        grid = [] if args.quick else [centred(ratio, decay) for ratio in RATIOS for decay in DECAYS]
        return sweep_recovery([SHIPPED, *grid, *extra], seeds, args.share)
    if args.scale:
        return sweep_scale(seeds, args.share, args.decay)
    if args.start:
        return sweep_start(seeds, args.share)
    if args.advantage:
        return sweep_advantage(seeds, args.share)

    sweep_bands(seeds, args.share)
    if args.share == ROUTING_SHARE_PROPORTIONAL:
        print(
            "\ntop1 and n_eff are the #11 gate diagnostics and only move under this "
            "share;\nthe uniform split the auction is strategyproof under fixes them "
            "at 1/top_k and top_k."
        )


if __name__ == "__main__":
    main()
