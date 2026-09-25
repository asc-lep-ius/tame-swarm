"""The ledger's fixed point, solved from a run's own inflow and price (#40, #73).

README ``#ledger-stability`` derives one map for both ledger modes::

    w <- (1 - rho) w + rho S + R - kappa / w

and reads two roots off it: the wealth a cell settles at, and the ruin threshold
below which the map pushes it further down. This script measures ``R`` and
``kappa`` on the planted-competence fixtures and solves the quadratic, so
every number in that section is reproducible and #43 can re-derive its own
constants rather than tune them.

``R`` is the reward signal's per-step inflow and ``kappa`` is the charge times
the wealth it was charged against -- a winner pays ``b_(k+1) / w``, so the charge
carries exactly one power of ``1/w`` and ``kappa`` is what is left. Both are read
over a tail, because the quadratic describes a ledger whose inflow is stationary
over its own memory (``1 / rho`` = 333 steps) and says nothing about a transient.

**The exchange rate (#73).** ``R`` and ``kappa`` are both priced in realised-value
units and share one coefficient, ``reward_scale``, so a configuration whose
cells own more of the output (#60's ``contribution_scale``) moves both together
and its winners' fixed point into the ceiling: at twice the recorded correction
most of the ``value`` arm's cell-steps sit on the ceiling and at four times every
ledger is exactly ``max_wealth``, so the bid ``confidence x wealth`` carries no
wealth at all. ``derive_reward_scale`` sets the rate per configuration from the
settled economy's own ``R`` and ``kappa`` so that the winners' fixed point is the
recorded configuration's -- one settlement across arms, #40's closed forms kept,
and at the recorded scale the recorded constant to the bit.

    uv run python scripts/measure_ledger_stability.py
    uv run python scripts/measure_ledger_stability.py --mode setpoint --seeds 0
    uv run python scripts/measure_ledger_stability.py --derive-reward-scale \\
        --fixture differentiated-fixture --contribution-scale 2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from sweep_wealth_bounds import CEILING_TOLERANCE, FLOOR_TOLERANCE  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DifferentiatedEconomy,
    SyntheticEconomy,
    competence_for,
    pearson,
    shuffled,
)

from mob import MoBConfig  # noqa: E402
from mob.ledger import (  # noqa: E402
    LEDGER_DECAY,
    LEDGER_SETPOINT,
    PERSISTENCE_VALUE,
    SUPPORTED_LEDGER_MODES,
    SUPPORTED_PERSISTENCE_COUPLINGS,
    RewardSignal,
    Settlement,
)
from parity import code_identity  # noqa: E402

# Eight memory horizons at the shipped decay, which is the budget #16's sweep
# settled on: shorter and the ledger is still reading its own transient, which is
# the one regime the closed form is not about.
DEFAULT_STEPS = 2667
DEFAULT_TAIL = 333
# Above this share of the slots a cell is holding a market rather than the
# exploration gift; the column #16 reports as ``win>1%``.
MARKET_SHARE = 0.01

QUALITY = "quality-fixture"
DIFFERENTIATED = "differentiated-fixture"
FIXTURES = (QUALITY, DIFFERENTIATED)
DEFAULT_SEEDS = (0, 1, 2)
# A cell *rests* on a bound when it sits there for more than this fraction of
# the tail -- as against ``CellReading.clamped``, which is whether a bound held
# it at any step. The two answer different questions: the closed form is not
# about a cell a bound touched, and a ledger is not readable when more cells
# rest on the ceiling than there are slots.
RESTING = 0.5
# The derivation's own budget: passes at successive trial rates before it
# refuses, and how close two successive rates have to be to call it settled.
MAX_PASSES = 6
CONVERGENCE = 0.01
DEFAULT_DERIVATION_OUT = Path.home() / "tame-runs" / "73-exchange-rate"


class ClampedLedgerError(RuntimeError):
    """A rate cannot be read off a ledger the band is holding; #73's refusal."""


@dataclass
class _Recorded:
    """Wraps a layer's reward signal and keeps what it paid, against what ledger."""

    inner: RewardSignal
    paid: list[torch.Tensor] = field(default_factory=list)
    against: list[torch.Tensor] = field(default_factory=list)

    @property
    def multiplier(self) -> float:
        return self.inner.multiplier

    def __call__(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor:
        self.against.append(wealth.clone())
        paid = self.inner(wealth, settlement)
        self.paid.append(paid.clone())
        return paid

    def gift(self, wealth: torch.Tensor, settlement: Settlement) -> torch.Tensor | None:
        return self.inner.gift(wealth, settlement)


@dataclass(frozen=True)
class CellReading:
    """One cell's settled ledger, and the two roots the map predicts for it."""

    competence: float
    wealth: float
    share: float
    reward: float
    price_coefficient: float
    settles_at: float
    ruined_below: float
    # The bare closed form ``S + n / rho`` at the flat mean of the tail's inflow,
    # beside the quadratic solved from R and kappa. It is the right prediction
    # only where the inflow is stationary over the ledger's own memory, and the
    # gap between the two is what says whether it is -- see the ``decoupled`` row
    # in README #ledger-stability, where the inflow carries no wealth at all and
    # this is nonetheless the looser of the two.
    from_flat_inflow: float
    clamped: bool
    # How far the recorded inflow, run back through the map, lands from the
    # ledger it should reproduce. Not a property of the economy: a check that the
    # settlement was recorded whole, so that R and kappa below are readings of it
    # rather than of part of it. Meaningless for a clamped cell, whose ledger was
    # held by a bound the recursion knows nothing about.
    reconstruction_error: float
    # The fraction of the tail this cell spent on each bound, at the tolerance
    # that bound needs (#16: the ceiling is an attractor and is read at 1e-4, the
    # floor is escaped by a hair on every exploration win and is read at 10%).
    tail_at_ceiling: float = 0.0
    tail_at_floor: float = 0.0

    @property
    def relative_error(self) -> float:
        """How far the predicted equilibrium is from the wealth the cell rests at."""
        return abs(self.settles_at - self.wealth) / self.wealth

    @property
    def flat_inflow_error(self) -> float:
        """How far the bare ``S + n / rho`` is from it -- the inflow's own drift."""
        return abs(self.from_flat_inflow - self.wealth) / self.wealth

    @property
    def rests_on_ceiling(self) -> bool:
        return self.tail_at_ceiling > RESTING

    @property
    def rests_on_floor(self) -> bool:
        return self.tail_at_floor > RESTING


@dataclass(frozen=True)
class LedgerReading:
    """One arm of the fixture: every cell, and what the allocation did."""

    mode: str
    coupling: str
    seed: int
    cells: tuple[CellReading, ...]
    market_holders: int
    least_share: float
    wealth_vs_competence: float
    tail_loss: float
    fixture: str = QUALITY
    contribution_scale: float = 1.0
    reward_scale: float = BASE_CONFIG.reward_scale

    @property
    def ceiling_occupancy(self) -> float:
        """Cell-steps at the ceiling over the tail, as a fraction of all of them."""
        return statistics.fmean(cell.tail_at_ceiling for cell in self.cells)

    @property
    def floor_occupancy(self) -> float:
        return statistics.fmean(cell.tail_at_floor for cell in self.cells)

    @property
    def cells_on_ceiling(self) -> int:
        return sum(cell.rests_on_ceiling for cell in self.cells)

    def saturated(self, top_k: int) -> bool:
        """More cells rest on the ceiling than there are slots: the ledger cannot rank.

        At the recorded configuration exactly ``top_k`` cells rest there with
        their fixed points above it (README #ledger-stability), and that is a
        ledger the auction still reads -- the winners are ranked against the
        floor cells by wealth. Once a third cell joins them the bids of the
        clamped cells carry no wealth and the allocation among them is the
        reports' alone, which is the regime #60's 2x and 4x rows were read in.
        """
        return self.cells_on_ceiling > top_k

    def winners(self) -> list[CellReading]:
        """Market holders by share, largest first."""
        holders = [cell for cell in self.cells if cell.share > MARKET_SHARE]
        return sorted(holders, key=lambda cell: cell.share, reverse=True)


def build_economy(
    fixture: str,
    seed: int,
    config: MoBConfig,
    contribution_scale: float = 1.0,
    cells: int = BASE_CONFIG.num_experts,
) -> SyntheticEconomy:
    """The fixture at ``seed``, at a cell count and a contribution scale."""
    competence = shuffled(competence_for(cells), seed)
    config = replace(config, num_experts=cells)
    if fixture == QUALITY:
        return SyntheticEconomy(
            competence, seed=seed, config=config, contribution_scale=contribution_scale
        )
    if fixture == DIFFERENTIATED:
        return DifferentiatedEconomy(
            competence, seed=seed, config=config, contribution_scale=contribution_scale
        )
    raise ValueError(f"unknown fixture {fixture!r}; one of {FIXTURES}")


def measure(
    mode: str,
    seed: int,
    steps: int = DEFAULT_STEPS,
    tail: int = DEFAULT_TAIL,
    coupling: str = PERSISTENCE_VALUE,
    fixture: str = QUALITY,
    contribution_scale: float = 1.0,
    reward_scale: float = BASE_CONFIG.reward_scale,
    cells: int = BASE_CONFIG.num_experts,
    config: MoBConfig = BASE_CONFIG,
) -> LedgerReading:
    """Run the fixture under ``mode`` and solve each cell's quadratic.

    ``coupling`` is #39's stakes dial. Under ``decoupled`` every price is computed
    from the pinned wealth, so the inflow carries no wealth at all and the bare
    ``S + n / rho`` is the exact fixed point of the map -- which makes that arm
    the one place the *stationarity* of the inflow is the only thing left between
    the formula and the ledger.

    ``contribution_scale`` and ``reward_scale`` are #73's pair: the first is what
    the cells own of the output, the second what a unit of it is worth in wealth.
    ``config`` is the base every other field is written onto; a test hands one
    over with a narrower band, and nothing else does.
    """
    if tail > steps:
        raise ValueError(f"tail must fit inside the run, got {tail} of {steps} steps")

    config = replace(
        config, ledger_mode=mode, persistence_coupling=coupling, reward_scale=reward_scale
    )
    economy = build_economy(fixture, seed, config, contribution_scale, cells)
    config = economy.config
    layer = economy.mob
    recorded = _Recorded(layer.wealth_updater.reward)
    layer.wealth_updater = replace(layer.wealth_updater, reward=recorded)

    charges: list[torch.Tensor] = []
    priced = layer._vcg_charges

    def recording_charge(*args, **kwargs) -> torch.Tensor:
        charge = priced(*args, **kwargs)
        charges.append(charge.clone())
        return charge

    layer._vcg_charges = recording_charge  # type: ignore[method-assign]

    start = layer.expert_wealth.clone()
    wins = torch.zeros(config.num_experts)
    losses: list[float] = []
    # Whether a bound held the cell at any point in the tail, rather than whether
    # it happens to sit on one at the last step: a cell resting on the floor that
    # took an exploration gift on the final step is still a clamped cell, and the
    # closed form is not about it.
    clamped = torch.zeros(config.num_experts, dtype=torch.bool)
    at_ceiling = torch.zeros(config.num_experts)
    at_floor = torch.zeros(config.num_experts)
    ceiling = config.max_wealth * (1 - CEILING_TOLERANCE)
    floor = config.min_wealth * (1 + FLOOR_TOLERANCE)
    for step in range(steps):
        record = economy.step()
        if step >= steps - tail:
            wins += torch.bincount(
                record.selected_experts.flatten(), minlength=config.num_experts
            ).float()
            losses.append(record.loss)
            clamped |= (layer.expert_wealth <= config.min_wealth) | (
                layer.expert_wealth >= config.max_wealth
            )
            at_ceiling += (layer.expert_wealth >= ceiling).float()
            at_floor += (layer.expert_wealth <= floor).float()

    share = wins / wins.sum()
    rho = 1.0 - config.wealth_decay
    reconstructed = _replay(
        torch.stack(recorded.paid) - torch.stack(charges),
        start=start,
        decay=config.wealth_decay,
        setpoint=config.initial_wealth if mode == LEDGER_SETPOINT else 0.0,
    )
    setpoint = config.initial_wealth if mode == LEDGER_SETPOINT else 0.0
    reward = torch.stack(recorded.paid)[-tail:].mean(dim=0)
    charged = torch.stack(charges)[-tail:]
    price_coefficient = (charged * torch.stack(recorded.against)[-tail:]).mean(dim=0)
    flat_inflow = (torch.stack(recorded.paid) - torch.stack(charges))[-tail:].mean(dim=0)

    cells_read = []
    for index in range(config.num_experts):
        wealth = float(layer.expert_wealth[index])
        cells_read.append(
            CellReading(
                competence=float(economy.competence[index]),
                wealth=wealth,
                share=float(share[index]),
                reward=float(reward[index]),
                price_coefficient=float(price_coefficient[index]),
                clamped=bool(clamped[index]),
                reconstruction_error=abs(float(reconstructed[index]) - wealth) / wealth,
                from_flat_inflow=layer.wealth_updater.equilibrium(float(flat_inflow[index])),
                tail_at_ceiling=float(at_ceiling[index]) / tail,
                tail_at_floor=float(at_floor[index]) / tail,
                **_roots(rho, setpoint, float(reward[index]), float(price_coefficient[index])),
            )
        )

    return LedgerReading(
        mode=mode,
        coupling=coupling,
        seed=seed,
        cells=tuple(cells_read),
        market_holders=int((share > MARKET_SHARE).sum()),
        least_share=float(share.min()),
        wealth_vs_competence=pearson(layer.expert_wealth, economy.competence),
        tail_loss=sum(losses) / len(losses),
        fixture=fixture,
        contribution_scale=contribution_scale,
        reward_scale=reward_scale,
    )


def _replay(
    inflow: torch.Tensor, start: torch.Tensor, decay: float, setpoint: float
) -> torch.Tensor:
    """The ledger the recorded inflow implies: ``decay^T w_0 + sum_j decay^j n_(T-1-j)``.

    The map with no clamp in it, summed in float64 so what it reports is the
    recording's completeness rather than the replay's own rounding.
    """
    steps = inflow.size(0)
    weights = torch.tensor([decay**j for j in range(steps)], dtype=torch.float64)
    relaxed = start.double() * decay**steps
    toward = setpoint * (1.0 - decay) * float(weights.sum())
    return relaxed + toward + (inflow.double() * weights.flip(0).unsqueeze(-1)).sum(dim=0)


def _roots(
    rho: float, setpoint: float, reward: float, price_coefficient: float
) -> dict[str, float]:
    """The two fixed points of ``rho w^2 - (rho S + R) w + kappa = 0``.

    ``nan`` for both when the discriminant is negative: the outflow outruns the
    inflow at every wealth, there is no equilibrium, and the cell is on its way to
    the floor whatever it starts with.
    """
    linear = rho * setpoint + reward
    discriminant = linear**2 - 4.0 * rho * price_coefficient
    if discriminant < 0.0:
        return {"settles_at": math.nan, "ruined_below": math.nan}
    root = math.sqrt(discriminant)
    return {
        "settles_at": (linear + root) / (2.0 * rho),
        "ruined_below": (linear - root) / (2.0 * rho),
    }


# --- #73: the exchange rate, derived from the settlement ---------------------------------


def rate_placing(
    target: float, reward: float, price_coefficient: float, rate: float, rho: float, setpoint: float
) -> float:
    """The ``reward_scale`` that puts a cell's upper root at ``target``.

    ``reward`` and ``price_coefficient`` were read at ``rate``, and both are
    proportional to it -- reward and charge share the one coefficient -- so the
    raw inflow and charge are ``R / rate`` and ``kappa / rate``, and the quadratic
    ``rho w^2 - (rho S + s r) w + s k = 0`` at ``w = target`` solves for ``s``::

        s = rho x target x (target - S) / (r x target - k)

    Read against a cell's own root this is the identity, which is what makes the
    recorded scale return the recorded rate exactly. ``nan`` when no positive
    rate places the cell there: the raw inflow at that wealth does not cover the
    raw charge, so scaling both cannot help.
    """
    raw_reward, raw_charge = reward / rate, price_coefficient / rate
    denominator = raw_reward * target - raw_charge
    if not math.isfinite(target) or denominator <= 0.0:
        return math.nan
    return rho * target * (target - setpoint) / denominator


@dataclass(frozen=True)
class DerivationPass:
    """One trial rate: the readings it produced and the rate they derive."""

    rate: float
    # Per seed, keyed by cell: the reference winner's root at this rate, and
    # the rate that would place that cell at its reference root. The seed's
    # own derived rate is the mean over its winners.
    fixed_points: dict[int, dict[int, float]]
    per_cell: dict[int, dict[int, float]]
    derived: dict[int, float]
    cells_on_ceiling: dict[int, int]
    ceiling_occupancy: dict[int, float]
    floor_occupancy: dict[int, float]
    wealth_vs_competence: dict[int, float]
    saturated_seeds: tuple[int, ...]
    next_rate: float
    # How the next trial was chosen: from the derivation, or by halving because
    # the derivation gave no finite positive rate on a saturated ledger.
    next_rate_from: str

    @property
    def saturated(self) -> bool:
        return bool(self.saturated_seeds)


@dataclass(frozen=True)
class Derivation:
    """The exchange rate for one configuration, and every pass that led to it.

    ``derived`` is the rate the final pass *ran at*: every occupancy, root and
    correlation the final pass records was read at that rate, and it is within
    ``tolerance`` of the rate that pass would derive next. At the recorded
    scale the final pass is the reference run, so ``derived`` is the recorded
    constant exactly.
    """

    fixture: str
    contribution_scale: float
    cells: int
    seeds: tuple[int, ...]
    steps: int
    tail: int
    recorded_rate: float
    # Per seed, keyed by cell: the recorded configuration's winners' roots,
    # which the derivation places the same cells at.
    reference: dict[int, dict[int, float]]
    reference_ceiling_occupancy: dict[int, float]
    reference_floor_occupancy: dict[int, float]
    reference_wealth_vs_competence: dict[int, float]
    passes: tuple[DerivationPass, ...]
    derived: float
    # The spread across seeds of the final pass's per-seed derivation.
    seed_spread: float
    code_sha: str | None
    code_dirty: bool | None

    @property
    def final(self) -> DerivationPass:
        return self.passes[-1]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _winner_targets(reading: LedgerReading) -> dict[int, float]:
    """The reference's market holders with a real root, keyed by cell.

    Keyed by cell rather than by rank: the seed fixes each cell's identity
    across scales, so the scaled reading's cell ``i`` is paired with the
    reference's cell ``i`` whatever its share rank became. A winner with no real
    root has no fixed point to place and is left out.
    """
    return {
        index: cell.settles_at
        for index, cell in enumerate(reading.cells)
        if cell.share > MARKET_SHARE and math.isfinite(cell.settles_at)
    }


def _solve(
    reading: LedgerReading, targets: dict[int, float], rate: float, config: MoBConfig
) -> tuple[dict[int, float], dict[int, float], float]:
    """Each reference winner's root at this rate, the rate placing it, and the seed's mean."""
    rho = 1.0 - config.wealth_decay
    setpoint = config.initial_wealth if reading.mode == LEDGER_SETPOINT else 0.0
    fixed_points = {cell: reading.cells[cell].settles_at for cell in targets}
    per_cell = {
        cell: rate_placing(
            target,
            reading.cells[cell].reward,
            reading.cells[cell].price_coefficient,
            rate,
            rho,
            setpoint,
        )
        for cell, target in targets.items()
    }
    finite = [value for value in per_cell.values() if math.isfinite(value) and value > 0.0]
    derived = statistics.fmean(finite) if len(finite) == len(targets) else math.nan
    return fixed_points, per_cell, derived


def _run_pass(
    readings: dict[int, LedgerReading],
    targets: dict[int, dict[int, float]],
    rate: float,
    config: MoBConfig,
) -> DerivationPass:
    """Solve one trial rate on every seed and choose the next."""
    seeds = sorted(readings)
    solved = {seed: _solve(readings[seed], targets[seed], rate, config) for seed in seeds}
    derived = {seed: solved[seed][2] for seed in seeds}
    saturated = tuple(seed for seed in seeds if readings[seed].saturated(config.top_k))
    finite = [value for value in derived.values() if math.isfinite(value)]
    if len(finite) == len(seeds):
        next_rate, origin = statistics.fmean(finite), "derived"
    elif saturated:
        next_rate, origin = rate / 2.0, "halved"
    else:
        unplaced = [seed for seed in seeds if not math.isfinite(derived[seed])]
        raise ClampedLedgerError(
            f"at rate {rate:.6g} no positive rate places the reference winners of seeds "
            f"{unplaced} at their roots: the raw inflow at the target does not cover the "
            "raw charge"
        )
    return DerivationPass(
        rate=rate,
        fixed_points={seed: solved[seed][0] for seed in seeds},
        per_cell={seed: solved[seed][1] for seed in seeds},
        derived=derived,
        cells_on_ceiling={seed: readings[seed].cells_on_ceiling for seed in seeds},
        ceiling_occupancy={seed: readings[seed].ceiling_occupancy for seed in seeds},
        floor_occupancy={seed: readings[seed].floor_occupancy for seed in seeds},
        wealth_vs_competence={seed: readings[seed].wealth_vs_competence for seed in seeds},
        saturated_seeds=saturated,
        next_rate=next_rate,
        next_rate_from=origin,
    )


def _reference_targets(
    reference: dict[int, LedgerReading], config: MoBConfig, fixture: str, cells: int
) -> dict[int, dict[int, float]]:
    """The recorded configuration's winners' roots per seed, refused when read off a clamp."""
    clamped = [seed for seed, reading in reference.items() if reading.saturated(config.top_k)]
    if clamped:
        raise ClampedLedgerError(
            f"the recorded configuration of {fixture} at {cells} cells rests more than "
            f"{config.top_k} cells on the ceiling on seeds {clamped}; a target read off a "
            "clamp is a target for a clamp"
        )
    targets = {seed: _winner_targets(reading) for seed, reading in reference.items()}
    empty = [seed for seed, target in targets.items() if not target]
    if empty:
        raise ClampedLedgerError(
            f"no winner of the recorded configuration has a real fixed point on seeds {empty}"
        )
    return targets


def derive_reward_scale(
    fixture: str,
    contribution_scale: float,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
    tail: int = DEFAULT_TAIL,
    cells: int = BASE_CONFIG.num_experts,
    config: MoBConfig = BASE_CONFIG,
    max_passes: int = MAX_PASSES,
    tolerance: float = CONVERGENCE,
) -> Derivation:
    """The rate at which ``contribution_scale`` saturates the band as the recorded scale does.

    The recorded configuration -- the same fixture and cell count at scale one
    and the recorded rate -- is run first, and its winners' upper roots are the
    targets, keyed by cell. Then the scaled configuration is run at the
    recorded rate, the same cells' ``R`` and ``kappa`` are read, and
    ``rate_placing`` says what rate puts each of them at its target; the seeds'
    mean is the next trial rate, and the passes repeat until a pass would derive
    a rate within ``tolerance`` of the one it ran at. **The rate returned is the
    one that pass ran at**, so every guardrail the record carries was read at
    it. At scale one the first pass *is* the reference, every cell's rate is
    the identity, and the derivation returns the recorded constant exactly with
    the economy untouched.

    **A saturated pass is never read as a derivation.** Once more cells rest
    on the ceiling than there are slots the auction is deciding among clamped
    bids, and the ``R`` and ``kappa`` such a pass reads belong to that
    allocation and not to the economy the rate is for; the pass only sets the
    next trial (halving it when it cannot even solve), and the rate that is
    returned comes from a pass the band was not holding. A configuration whose
    every pass is saturated -- or whose reference is, since a target read off a
    clamp is a target for a clamp -- is refused with ``ClampedLedgerError``
    rather than given a number.

    The rate is derived once per configuration from the seed set, and the
    spread across seeds is recorded beside it; a rate re-derived per seed
    would be a confound between the seeds it was meant to pair.
    """
    if contribution_scale <= 0.0:
        raise ValueError(f"contribution_scale must be positive, got {contribution_scale}")
    if not seeds:
        raise ValueError("the derivation needs at least one seed")
    recorded = config.reward_scale

    def run(seed: int, scale: float, rate: float) -> LedgerReading:
        return measure(
            LEDGER_DECAY,
            seed,
            steps=steps,
            tail=tail,
            fixture=fixture,
            contribution_scale=scale,
            reward_scale=rate,
            cells=cells,
            config=config,
        )

    reference = {seed: run(seed, 1.0, recorded) for seed in seeds}
    targets = _reference_targets(reference, config, fixture, cells)
    passes: list[DerivationPass] = []
    rate = recorded
    for _ in range(max_passes):
        readings = (
            reference
            if contribution_scale == 1.0 and rate == recorded
            else {seed: run(seed, contribution_scale, rate) for seed in seeds}
        )
        passes.append(_run_pass(readings, targets, rate, config))
        final = passes[-1]
        if not final.saturated and abs(final.next_rate - rate) <= tolerance * rate:
            return _finish(
                fixture, contribution_scale, cells, steps, tail, reference, targets, passes
            )
        rate = final.next_rate
    trail = ", ".join(
        f"{p.rate:.4g} -> {p.next_rate:.4g}{' (saturated)' if p.saturated else ''}" for p in passes
    )
    raise ClampedLedgerError(
        f"no unsaturated, settled rate for {fixture} x{contribution_scale:g} at {cells} cells "
        f"within {max_passes} passes: {trail}"
    )


def _finish(
    fixture: str,
    contribution_scale: float,
    cells: int,
    steps: int,
    tail: int,
    reference: dict[int, LedgerReading],
    targets: dict[int, dict[int, float]],
    passes: list[DerivationPass],
) -> Derivation:
    final = passes[-1]
    finite = [value for value in final.derived.values() if math.isfinite(value)]
    code_sha, code_dirty = code_identity()
    seeds = tuple(sorted(reference))
    return Derivation(
        fixture=fixture,
        contribution_scale=contribution_scale,
        cells=cells,
        seeds=seeds,
        steps=steps,
        tail=tail,
        recorded_rate=reference[seeds[0]].reward_scale,
        reference=targets,
        reference_ceiling_occupancy={seed: reference[seed].ceiling_occupancy for seed in seeds},
        reference_floor_occupancy={seed: reference[seed].floor_occupancy for seed in seeds},
        reference_wealth_vs_competence={
            seed: reference[seed].wealth_vs_competence for seed in seeds
        },
        passes=tuple(passes),
        derived=final.rate,
        seed_spread=statistics.stdev(finite) if len(finite) > 1 else 0.0,
        code_sha=code_sha,
        code_dirty=code_dirty,
    )


def _report_derivation(derivation: Derivation) -> None:
    print(
        f"\n=== exchange rate: {derivation.fixture} x{derivation.contribution_scale:g}, "
        f"{derivation.cells} cells, seeds {list(derivation.seeds)}, "
        f"{derivation.steps} steps / {derivation.tail} tail ==="
    )
    for seed in derivation.seeds:
        roots = ", ".join(f"{cell}:{w:.0f}" for cell, w in derivation.reference[seed].items())
        print(
            f"  reference seed {seed}: winners settle at [{roots}], "
            f"ceiling {derivation.reference_ceiling_occupancy[seed]:.3f} "
            f"floor {derivation.reference_floor_occupancy[seed]:.3f} "
            f"r {derivation.reference_wealth_vs_competence[seed]:+.3f}"
        )
    print(
        f"  {'pass':>4} {'rate':>9} {'seed':>5} {'w+':>22} {'derived':>9} "
        f"{'ceil':>6} {'floor':>6} {'r':>7}"
    )
    for index, p in enumerate(derivation.passes, 1):
        for seed in derivation.seeds:
            roots = ", ".join(f"{w:.0f}" for w in p.fixed_points[seed].values())
            print(
                f"  {index:>4} {p.rate:>9.5f} {seed:>5} {roots:>22} {p.derived[seed]:>9.5f} "
                f"{p.ceiling_occupancy[seed]:>6.3f} {p.floor_occupancy[seed]:>6.3f} "
                f"{p.wealth_vs_competence[seed]:>+7.3f}"
                f"{'  saturated' if seed in p.saturated_seeds else ''}"
            )
        print(f"       -> next {p.next_rate:.5f} ({p.next_rate_from})")
    print(
        f"  derived reward_scale {derivation.derived:.6f} (recorded {derivation.recorded_rate}), "
        f"seed spread {derivation.seed_spread:.6f}, {len(derivation.passes)} pass(es)"
    )


def _report(reading: LedgerReading) -> None:
    print(
        f"\n--- {reading.fixture} x{reading.contribution_scale:g} at reward_scale "
        f"{reading.reward_scale:g}: {reading.mode} ledger, {reading.coupling} arm, "
        f"seed {reading.seed}: "
        f"win>{MARKET_SHARE:.0%} {reading.market_holders} of {len(reading.cells)}, "
        f"least share {reading.least_share:.4f}, "
        f"r(wealth, competence) {reading.wealth_vs_competence:.3f}, "
        f"tail loss {reading.tail_loss:.4f}, "
        f"ceiling {reading.ceiling_occupancy:.3f} floor {reading.floor_occupancy:.3f}"
    )
    print(
        f"{'c':>5} {'wealth':>9} {'share':>7} {'R':>8} {'kappa':>9} "
        f"{'settles at':>11} {'ruined below':>13} {'rel':>7} {'n/rho':>9} {'rel':>7} {'replay':>9}"
    )
    for cell in reading.cells:
        marker = " (clamped)" if cell.clamped else ""
        print(
            f"{cell.competence:>5.2f} {cell.wealth:>9.2f} {cell.share:>7.4f} {cell.reward:>8.4f} "
            f"{cell.price_coefficient:>9.2f} {cell.settles_at:>11.2f} {cell.ruined_below:>13.2f} "
            f"{cell.relative_error:>7.3f} {cell.from_flat_inflow:>9.2f} "
            f"{cell.flat_inflow_error:>7.3f} {cell.reconstruction_error:>9.1e}{marker}"
        )


def derivation_path(out: Path, fixture: str, cells: int, contribution_scale: float) -> Path:
    return out / f"derivation_{fixture}_cells{cells}_scale{contribution_scale:g}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="both", choices=[*sorted(SUPPORTED_LEDGER_MODES), "both"])
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS)), help="comma-separated"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    parser.add_argument(
        "--coupling",
        default=PERSISTENCE_VALUE,
        choices=sorted(SUPPORTED_PERSISTENCE_COUPLINGS),
        help="#39's stakes dial; decoupled is where the bare closed form is exact",
    )
    parser.add_argument("--fixture", default=QUALITY, choices=FIXTURES)
    parser.add_argument("--cells", type=int, default=BASE_CONFIG.num_experts)
    parser.add_argument(
        "--contribution-scale",
        type=float,
        default=1.0,
        help="#60's knob; 1.0 is the recorded fixture",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=BASE_CONFIG.reward_scale,
        help="a hand-set rate to read the ledger at; the derivation ignores it",
    )
    parser.add_argument(
        "--derive-reward-scale",
        action="store_true",
        help="#73: derive the rate for --contribution-scale from the settlement and record it",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DERIVATION_OUT)
    args = parser.parse_args()
    seeds = tuple(int(seed) for seed in args.seeds.split(","))

    if args.derive_reward_scale:
        derivation = derive_reward_scale(
            args.fixture,
            args.contribution_scale,
            seeds=seeds,
            steps=args.steps,
            tail=args.tail,
            cells=args.cells,
        )
        _report_derivation(derivation)
        path = derivation_path(args.out, args.fixture, args.cells, args.contribution_scale)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(derivation.as_dict(), indent=2, default=str))
        print(f"  record: {path}")
        return

    modes = [LEDGER_DECAY, LEDGER_SETPOINT] if args.mode == "both" else [args.mode]
    for mode in modes:
        for seed in seeds:
            _report(
                measure(
                    mode,
                    seed,
                    steps=args.steps,
                    tail=args.tail,
                    coupling=args.coupling,
                    fixture=args.fixture,
                    contribution_scale=args.contribution_scale,
                    reward_scale=args.reward_scale,
                    cells=args.cells,
                )
            )


if __name__ == "__main__":
    main()
