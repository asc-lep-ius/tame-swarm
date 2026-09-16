"""Read the differentiated planted-competence fixture, beside the quality fixture (#25).

Every economy baseline in the repository was measured on the quality fixture
(``scripts/synthetic_economy.py::SyntheticEconomy``), on which competence is
token-independent and the experts' contributions are parallel. The
differentiated fixture (``DifferentiatedEconomy``) changes the loss *identity*,
not just its value, so its numbers are not the suite's numbers and are read here
rather than substituted for them: the suite's baselines stay on the quality
fixture, named as such, and this script is where the differentiated fixture's
are recorded.

Four passes:

``--fixture``      the #15 readings (surplus per win, r(wealth, win share),
                   r(wealth, competence)) and the specialisation readings the
                   quality fixture cannot give (on-type share against chance and
                   the oracle, the least-used expert's share, whether any type's
                   weaker expert outranks its stronger), per seed, on both fixtures.
``--legibility``   the same readings as the field's legibility (``type_signal``)
                   and the budget vary. This is the curve on which the fixture's
                   default was chosen, and where the ruin of the most competent
                   cell is observed.
``--recovery``     the damage protocols the suite asserts on the quality fixture
                   (``scripts/economy_damage.py``: senescence, the forced
                   episodes, ruin), run unchanged on the differentiated fixture.
``--warmup``       the same readings after a perception warmup of the given
                   lengths: the heads trained on the value objective with routing
                   forced uniform and the ledger frozen, before the economy runs
                   its recorded budget. This separates the two readings of the
                   ruin of the most competent cell (#34) -- a *developmental
                   order* in which selection precedes perception, or the *value
                   asymmetry* that punishes a competent cell hardest per off-type
                   token, which no ordering can remove.

Run:  uv run python scripts/measure_differentiated_economy.py
      uv run python scripts/measure_differentiated_economy.py --legibility
      uv run python scripts/measure_differentiated_economy.py --recovery
      uv run python scripts/measure_differentiated_economy.py --legibility --warmup 0,50,200
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from economy_damage import (  # noqa: E402
    DAMAGE_HORIZON,
    LONG_FORCED_EPISODE,
    SEEDS,
    SHORT_FORCED_EPISODE,
    WINDOW,
    Build,
    ForcedSubset,
    floor_without,
    quality_fixture,
    release_and_measure,
    ruin_and_measure,
    senesce,
    steady,
    survivors_track_competence,
    window,
)
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DEFAULT_TYPE_SIGNAL,
    DifferentiatedEconomy,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from mob import MoBConfig  # noqa: E402
from specialisation import expert_output_divergence  # noqa: E402

STEPS = 600
TAIL = 100
LEGIBILITY = (2.0, 4.0, 8.0)
BUDGETS = (600, 2000)
# The warmup is asked at the legibility where the inversion is observed and at the
# one where it is not, so a warmup that removes the inversion can be told apart
# from one that moves every row. 8.0 is left out: nothing inverts there to remove.
WARMUP_LEGIBILITY = (2.0, 4.0)
DEFAULT_WARMUPS = (0, 50, 200)


def differentiated_fixture(
    competence: torch.Tensor, seed: int, config: MoBConfig
) -> SyntheticEconomy:
    return DifferentiatedEconomy(competence, seed=seed, config=config)


@dataclass(frozen=True)
class Reading:
    """One fixture at one seed, read over the tail of a run."""

    fixture: str
    seed: int
    contribution_distance: float
    surplus: float
    wealth_vs_win_share: float
    wealth_vs_competence: float
    loss: float
    oracle_loss: float | None
    on_type: float | None
    min_share: float
    inverted_types: int | None


def _oracle_loss(
    economy: DifferentiatedEconomy, hidden: torch.Tensor, target: torch.Tensor
) -> float:
    """The loss with both slots on the token's own type: the efficient allocation's."""
    assert economy.last_types is not None
    members = torch.stack(
        [
            (economy.expert_types == expert_type).nonzero().flatten()[: economy.config.top_k]
            for expert_type in range(economy.num_types)
        ]
    )
    return float(economy.closed_form_loss(hidden, target, members[economy.last_types]).mean())


def _inverted_types(economy: DifferentiatedEconomy, share: torch.Tensor) -> int:
    """How many types have their weaker expert outranking their stronger one."""
    inverted = 0
    for expert_type in range(economy.num_types):
        members = (economy.expert_types == expert_type).nonzero().flatten().tolist()
        best = max(members, key=lambda index: float(economy.competence[index]))
        worst = min(members, key=lambda index: float(economy.competence[index]))
        inverted += int(share[best] < share[worst])
    return inverted


def read(economy: SyntheticEconomy, name: str, seed: int, steps: int = STEPS) -> Reading:
    hidden = torch.randn(256, economy.config.hidden_dim)
    divergence = expert_output_divergence(economy.mob, hidden)
    wins = torch.zeros(economy.config.num_experts)
    surplus: list[float] = []
    losses: list[float] = []
    oracle: list[float] = []
    on_type: list[float] = []
    for step in range(steps):
        record = economy.step()
        if step < steps - TAIL:
            continue
        wins += torch.bincount(
            record.selected_experts.flatten(), minlength=economy.config.num_experts
        ).float()
        surplus.append(record.mean_surplus)
        losses.append(record.loss)
        if isinstance(economy, DifferentiatedEconomy):
            on_type.append(economy.on_type_share(record.selected_experts))
            probe_hidden, probe_target = economy._draw()
            oracle.append(_oracle_loss(economy, probe_hidden, probe_target))
    share = wins / wins.sum()
    differentiated = isinstance(economy, DifferentiatedEconomy)
    return Reading(
        fixture=name,
        seed=seed,
        contribution_distance=divergence.mean_contribution_cosine_distance,
        surplus=sum(surplus) / len(surplus),
        wealth_vs_win_share=pearson(economy.mob.expert_wealth, share),
        wealth_vs_competence=pearson(economy.mob.expert_wealth, economy.competence),
        loss=sum(losses) / len(losses),
        oracle_loss=sum(oracle) / len(oracle) if oracle else None,
        on_type=sum(on_type) / len(on_type) if on_type else None,
        min_share=float(share.min()),
        inverted_types=_inverted_types(economy, share) if differentiated else None,  # type: ignore[arg-type]
    )


def _row(reading: Reading) -> str:
    def cell(value: float | int | None, width: int = 9, digits: int = 3) -> str:
        if value is None:
            return f"{'-':>{width}}"
        return f"{value:>{width}.{digits}f}" if isinstance(value, float) else f"{value:>{width}}"

    return (
        f"{reading.fixture:<15}{reading.seed:>5}"
        f"{cell(reading.contribution_distance)}{cell(reading.surplus, digits=4)}"
        f"{cell(reading.wealth_vs_win_share)}{cell(reading.wealth_vs_competence)}"
        f"{cell(reading.loss)}{cell(reading.oracle_loss)}{cell(reading.on_type)}"
        f"{cell(reading.min_share)}{cell(reading.inverted_types)}"
    )


HEADER = (
    f"{'fixture':<15}{'seed':>5}{'contrib':>9}{'surplus':>9}{'r(w,s)':>9}{'r(w,c)':>9}"
    f"{'loss':>9}{'oracle':>9}{'on-type':>9}{'min':>9}{'inverted':>9}"
)


def fixture_pass() -> None:
    print(HEADER)
    print("-" * len(HEADER))
    for seed in SEEDS:
        competence = shuffled(DEFAULT_COMPETENCE, seed)
        print(_row(read(SyntheticEconomy(competence, seed=seed), "quality", seed)))
        print(_row(read(DifferentiatedEconomy(competence, seed=seed), "differentiated", seed)))
    print(
        "\ncontrib: mean contribution cosine distance (0 on the quality fixture by construction,"
        " 24/28 on the differentiated one). on-type: share of slots held by an expert of the"
        f" token's own type; chance {BASE_CONFIG.top_k / DEFAULT_COMPETENCE.numel():.2f}, "
        "optimum 1.0. oracle: the loss with both slots on the token's type. inverted: types"
        " whose weaker expert outranks its stronger."
    )


def legibility_pass() -> None:
    print(HEADER.replace("fixture        ", "signal   steps "))
    print("-" * len(HEADER))
    for signal in LEGIBILITY:
        for steps in BUDGETS:
            for seed in SEEDS:
                economy = DifferentiatedEconomy(
                    shuffled(DEFAULT_COMPETENCE, seed), seed=seed, type_signal=signal
                )
                reading = read(economy, f"{signal:<8.1f} {steps:<5}", seed, steps)
                print(_row(reading))
    print(
        f"\nthe default is type_signal={DEFAULT_TYPE_SIGNAL}: legible enough that no type"
        " inverts and wealth tracks competence, with the optimum still unreached."
    )


def warm_up_heads(economy: SyntheticEconomy, steps: int, seed: int) -> None:
    """Let every head see every token type before any cell can be shut out (#34).

    Routing is forced uniform over *all* experts, so no report can concentrate the
    slots and nothing can be ruined while the heads are still learning to read the
    field; the heads still train on the value objective, which is the whole point.
    The wealth ledger is restored after every step, so what the warmup carries into the
    economy is perception and not a market position -- the economy that follows
    starts from the wealth every cell was born with, as the unwarmed arm does.

    The post-warmup budget is unchanged: these steps are *additional*, and a
    warmup arm is a longer run, not a reallocated one.
    """
    if steps <= 0:
        return
    everyone = list(range(economy.config.num_experts))
    gate = economy.mob.gate
    # Its own stream, keyed by length and seed as ``force_routing``'s is, so the
    # forcing never replicates the draws of the economy it precedes.
    economy.mob.gate = ForcedSubset(everyone, economy.config.top_k, seed=1000 * steps + seed)
    ledger = economy.mob.expert_wealth.detach().clone()
    try:
        for _ in range(steps):
            economy.step()
            with torch.no_grad():
                economy.mob.expert_wealth.copy_(ledger)
    finally:
        economy.mob.gate = gate


def warmup_pass(warmups: tuple[int, ...]) -> None:
    print(HEADER.replace("fixture        ", "signal   warm  "))
    print("-" * len(HEADER))
    for signal in WARMUP_LEGIBILITY:
        for steps in warmups:
            for seed in SEEDS:
                economy = DifferentiatedEconomy(
                    shuffled(DEFAULT_COMPETENCE, seed), seed=seed, type_signal=signal
                )
                warm_up_heads(economy, steps, seed)
                print(_row(read(economy, f"{signal:<8.1f} {steps:<5}", seed)))
    print(
        f"\nwarm: steps of forced-uniform routing with the ledger frozen before the"
        f" economy's {STEPS}; warm 0 is the recorded arm. The primary reading is"
        " inverted at type_signal 2.0: 0 of 3 seeds says the ruin of the most"
        " competent expert is an ordering effect, 2 of 3 says it is the value"
        " asymmetry and no warmup can remove it, 1 of 3 decides nothing."
    )


def recovery_pass() -> None:
    for name, build in (("quality", quality_fixture), ("differentiated", differentiated_fixture)):
        print(f"\n== {name} fixture ==")
        _senescence(build)
        for episode in (SHORT_FORCED_EPISODE, LONG_FORCED_EPISODE):
            for seed in SEEDS:
                released = release_and_measure(seed, episode, build=build)
                print(
                    f"forced {episode:>3} seed {seed}: loss ratio {released.loss_ratio:.2f}"
                    f" tracking {released.tracking:+.2f} regained {released.regained:.2f}"
                    f" (steady share {released.steady_share:.3f}, steady loss"
                    f" {released.steady_loss:.3f} -> {released.post_loss:.3f})"
                )
        share, wealth, median = ruin_and_measure(SEEDS[0], build=build)
        print(
            f"ruin seed {SEEDS[0]}: share {share:.4f} (chance"
            f" {1 / DEFAULT_COMPETENCE.numel():.3f}) wealth {wealth:.1f} median {median:.1f}"
        )


def _senescence(build: Build) -> None:
    for seed in SEEDS:
        competence = shuffled(DEFAULT_COMPETENCE, seed)
        best = int(competence.argmax())
        floor = floor_without(competence, best, seed, build=build)
        economy, _ = steady(competence, seed, build=build)
        senesce(economy, best)
        window(economy, DAMAGE_HORIZON - WINDOW)
        loss, share = window(economy, WINDOW)
        tracking = survivors_track_competence(share, competence, best)
        print(
            f"senescence seed {seed}: loss/floor {loss / floor:.2f}"
            f" dead share {float(share[best]):.3f} survivors track {tracking:+.2f}"
        )


def warmup_lengths(value: str) -> tuple[int, ...]:
    """``"0,50,200"`` -> ``(0, 50, 200)``: the warmup arms to run, in order."""
    lengths = []
    for part in value.split(","):
        try:
            steps = int(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"warmup lengths must be integers, got {part!r}"
            ) from error
        if steps < 0:
            raise argparse.ArgumentTypeError(f"warmup lengths must be >= 0, got {steps}")
        lengths.append(steps)
    return tuple(lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="The per-seed readings (default)")
    parser.add_argument("--legibility", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument(
        "--warmup",
        type=warmup_lengths,
        nargs="?",
        const=DEFAULT_WARMUPS,
        default=None,
        metavar="STEPS[,STEPS...]",
        help=(
            "Perception-warmup arms, comma separated, at type_signal "
            f"{' and '.join(str(signal) for signal in WARMUP_LEGIBILITY)}"
            f" (default {','.join(str(steps) for steps in DEFAULT_WARMUPS)})"
        ),
    )
    args = parser.parse_args()
    if not (args.legibility or args.recovery or args.warmup is not None):
        args.fixture = True
    if args.fixture:
        fixture_pass()
    if args.legibility:
        legibility_pass()
    if args.warmup is not None:
        warmup_pass(args.warmup)
    if args.recovery:
        recovery_pass()


if __name__ == "__main__":
    main()
