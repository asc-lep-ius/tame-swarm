"""The ledger's fixed point, solved from a run's own inflow and price (#40).

README ``#ledger-stability`` derives one map for both ledger modes::

    w <- (1 - rho) w + rho S + R - kappa / w

and reads two roots off it: the wealth a cell settles at, and the ruin threshold
below which the map pushes it further down. This script measures ``R`` and
``kappa`` on the planted-competence quality fixture and solves the quadratic, so
every number in that section is reproducible and #43 can re-derive its own
constants rather than tune them.

``R`` is the reward signal's per-step inflow and ``kappa`` is the charge times
the wealth it was charged against -- a winner pays ``b_(k+1) / w``, so the charge
carries exactly one power of ``1/w`` and ``kappa`` is what is left. Both are read
over a tail, because the quadratic describes a ledger whose inflow is stationary
over its own memory (``1 / rho`` = 333 steps) and says nothing about a transient.

    uv run python scripts/measure_ledger_stability.py
    uv run python scripts/measure_ledger_stability.py --mode setpoint --seeds 0
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from mob.ledger import (  # noqa: E402
    LEDGER_DECAY,
    LEDGER_SETPOINT,
    SUPPORTED_LEDGER_MODES,
    RewardSignal,
    Settlement,
)

# Eight memory horizons at the shipped decay, which is the budget #16's sweep
# settled on: shorter and the ledger is still reading its own transient, which is
# the one regime the closed form is not about.
DEFAULT_STEPS = 2667
DEFAULT_TAIL = 333
# Above this share of the slots a cell is holding a market rather than the
# exploration gift; the column #16 reports as ``win>1%``.
MARKET_SHARE = 0.01


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
    clamped: bool

    @property
    def relative_error(self) -> float:
        """How far the predicted equilibrium is from the wealth the cell rests at."""
        return abs(self.settles_at - self.wealth) / self.wealth


@dataclass(frozen=True)
class LedgerReading:
    """One arm of the fixture: every cell, and what the allocation did."""

    mode: str
    seed: int
    cells: tuple[CellReading, ...]
    market_holders: int
    least_share: float
    wealth_vs_competence: float
    tail_loss: float


def measure(
    mode: str, seed: int, steps: int = DEFAULT_STEPS, tail: int = DEFAULT_TAIL
) -> LedgerReading:
    """Run the quality fixture under ``mode`` and solve each cell's quadratic."""
    if tail > steps:
        raise ValueError(f"tail must fit inside the run, got {tail} of {steps} steps")

    config = replace(BASE_CONFIG, ledger_mode=mode)
    economy = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)
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

    wins = torch.zeros(config.num_experts)
    losses: list[float] = []
    # Whether a bound held the cell at any point in the tail, rather than whether
    # it happens to sit on one at the last step: a cell resting on the floor that
    # took an exploration gift on the final step is still a clamped cell, and the
    # closed form is not about it.
    clamped = torch.zeros(config.num_experts, dtype=torch.bool)
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

    share = wins / wins.sum()
    rho = 1.0 - config.wealth_decay
    setpoint = config.initial_wealth if mode == LEDGER_SETPOINT else 0.0
    reward = torch.stack(recorded.paid)[-tail:].mean(dim=0)
    charged = torch.stack(charges)[-tail:]
    price_coefficient = (charged * torch.stack(recorded.against)[-tail:]).mean(dim=0)

    cells = []
    for index in range(config.num_experts):
        wealth = float(layer.expert_wealth[index])
        cells.append(
            CellReading(
                competence=float(economy.competence[index]),
                wealth=wealth,
                share=float(share[index]),
                reward=float(reward[index]),
                price_coefficient=float(price_coefficient[index]),
                clamped=bool(clamped[index]),
                **_roots(rho, setpoint, float(reward[index]), float(price_coefficient[index])),
            )
        )

    return LedgerReading(
        mode=mode,
        seed=seed,
        cells=tuple(cells),
        market_holders=int((share > MARKET_SHARE).sum()),
        least_share=float(share.min()),
        wealth_vs_competence=pearson(layer.expert_wealth, economy.competence),
        tail_loss=sum(losses) / len(losses),
    )


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


def _report(reading: LedgerReading) -> None:
    print(
        f"\n--- {reading.mode}, seed {reading.seed}: "
        f"win>{MARKET_SHARE:.0%} {reading.market_holders} of {len(reading.cells)}, "
        f"least share {reading.least_share:.4f}, "
        f"r(wealth, competence) {reading.wealth_vs_competence:.3f}, "
        f"tail loss {reading.tail_loss:.4f}"
    )
    print(
        f"{'c':>5} {'wealth':>9} {'share':>7} {'R':>8} {'kappa':>9} "
        f"{'settles at':>11} {'ruined below':>13} {'rel':>7}"
    )
    for cell in reading.cells:
        marker = " (clamped)" if cell.clamped else ""
        print(
            f"{cell.competence:>5.2f} {cell.wealth:>9.2f} {cell.share:>7.4f} {cell.reward:>8.4f} "
            f"{cell.price_coefficient:>9.2f} {cell.settles_at:>11.2f} {cell.ruined_below:>13.2f} "
            f"{cell.relative_error:>7.3f}{marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="both", choices=[*sorted(SUPPORTED_LEDGER_MODES), "both"])
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated fixture seeds")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    args = parser.parse_args()

    modes = [LEDGER_DECAY, LEDGER_SETPOINT] if args.mode == "both" else [args.mode]
    for mode in modes:
        for seed in (int(seed) for seed in args.seeds.split(",")):
            _report(measure(mode, seed, steps=args.steps, tail=args.tail))


if __name__ == "__main__":
    main()
