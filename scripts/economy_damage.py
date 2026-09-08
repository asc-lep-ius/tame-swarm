"""The damage protocols the planted-competence economy is judged by, in one place.

``tests/test_homeostatic_recovery.py`` asserts these; ``sweep_wealth_bounds.py``
runs them against candidate wealth bands. They have to be the *same* protocol:
#16's acceptance criterion is whether the two strict expected failures flip under
a candidate band, and a sweep measuring its own private variant of them could
report a flip the suite does not see.

Every protocol here takes the config it runs under, because that is the variable
#16 sweeps. The defaults reproduce the numbers recorded in the expected-failure
reasons, so a run at ``BASE_CONFIG`` is the baseline the sweep's rows are read
against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    MoBConfig,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from mob.auction import AuctionOutcome  # noqa: E402

STEADY_STEPS = 200
WINDOW = 50
DAMAGE_HORIZON = 200
RELEASE_HORIZON = 150
SHORT_FORCED_EPISODE = 50
LONG_FORCED_EPISODE = 150
SEEDS = (0, 1, 2)

# A market that has re-formed sits at the loss a collective born without the
# damaged expert reaches, within this factor, and no longer routes to it.
RE_FORMATION_FACTOR = 1.1
DEAD_SHARE_CEILING = 0.01
SURVIVOR_TRACKING_FLOOR = 0.5
TRACKING_AFTER_RELEASE = 0.7
# The best expert has regained its standing when it holds this much of the share
# it held before the episode (0.30-0.43 of the slots, by seed).
REGAINED_SHARE = 0.8


def window(economy: SyntheticEconomy, steps: int) -> tuple[float, torch.Tensor]:
    """Mean loss and per-expert win share over ``steps`` steps."""
    losses: list[float] = []
    wins = torch.zeros(economy.config.num_experts)
    for _ in range(steps):
        record = economy.step()
        losses.append(record.loss)
        wins += torch.bincount(
            record.selected_experts.flatten(), minlength=economy.config.num_experts
        ).float()
    return sum(losses) / len(losses), wins / wins.sum()


def steady(
    competence: torch.Tensor, seed: int, config: MoBConfig = BASE_CONFIG
) -> tuple[SyntheticEconomy, float]:
    """An economy run to its steady state, and the loss it settled at."""
    economy = SyntheticEconomy(competence, seed=seed, config=config)
    window(economy, STEADY_STEPS - WINDOW)
    loss, _ = window(economy, WINDOW)
    return economy, loss


def floor_without(
    competence: torch.Tensor, expert: int, seed: int, config: MoBConfig = BASE_CONFIG
) -> float:
    """The loss a collective born without ``expert`` settles at: the re-formation target."""
    born_without = competence.clone()
    born_without[expert] = 0.0
    return steady(born_without, seed, config)[1]


def survivors_track_competence(share: torch.Tensor, competence: torch.Tensor, dead: int) -> float:
    keep = torch.tensor([index != dead for index in range(competence.numel())])
    return pearson(share[keep], competence[keep])


def senesce(economy: SyntheticEconomy, expert: int) -> None:
    """The cell is still wired and still bids; it just stops contributing anything."""
    with torch.no_grad():
        economy.mob.experts[expert].down_adapter_B.weight.zero_()  # type: ignore[union-attr]


def ruin(economy: SyntheticEconomy, expert: int) -> None:
    with torch.no_grad():
        economy.mob.expert_wealth[expert] = 0.0


def freeze_heads(economy: SyntheticEconomy) -> None:
    for group in economy.optimizer.param_groups:
        group["lr"] = 0.0


class ForcedSubset(torch.nn.Module):
    """A gate that ignores every report and routes each token to a fixed subset of experts."""

    def __init__(self, subset: list[int], top_k: int, seed: int):
        super().__init__()
        self.subset = torch.tensor(subset)
        self.top_k = top_k
        # Its own stream, so the forcing does not move the economy's draws.
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, confidences: torch.Tensor, wealth: torch.Tensor) -> AuctionOutcome:
        batch, seq_len, num_experts = confidences.shape
        draws = torch.stack(
            [
                torch.randperm(len(self.subset), generator=self.generator)[: self.top_k]
                for _ in range(batch * seq_len)
            ]
        ).view(batch, seq_len, self.top_k)
        selected = self.subset[draws]
        weights = torch.full_like(confidences[..., : self.top_k], 1.0 / self.top_k)
        rebates = torch.zeros_like(confidences)
        return AuctionOutcome(selected, weights, torch.zeros_like(weights), rebates, None)


def force_routing(economy: SyntheticEconomy, subset: list[int], steps: int, seed: int) -> None:
    """Route by fiat for ``steps`` steps; the forcing stream is a replicate of the seed too."""
    gate = economy.mob.gate
    economy.mob.gate = ForcedSubset(subset, economy.config.top_k, seed=1000 * steps + seed)
    try:
        window(economy, steps)
    finally:
        economy.mob.gate = gate


def release_and_measure(
    seed: int, episode: int, config: MoBConfig = BASE_CONFIG
) -> tuple[float, float, float, float]:
    """``(loss / steady loss, routing-competence correlation, best share / its steady share)``."""
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    least_competent = competence.argsort()[:3].tolist()
    economy = SyntheticEconomy(competence, seed=seed, config=config)
    window(economy, STEADY_STEPS - WINDOW)
    steady_loss, steady_share = window(economy, WINDOW)

    force_routing(economy, least_competent, episode, seed)
    window(economy, RELEASE_HORIZON - WINDOW)
    loss, share = window(economy, WINDOW)
    return (
        loss / steady_loss,
        pearson(share, competence),
        float(share[best]) / float(steady_share[best]),
        float(steady_share[best]),
    )


def ruin_and_measure(seed: int, config: MoBConfig = BASE_CONFIG) -> tuple[float, float, float]:
    """``(best expert's win share, its wealth, the median wealth)`` after it is ruined.

    The two quantities ``test_a_ruined_competent_expert_returns_to_the_market``
    asserts on: a share above chance, and a wealth above the median.
    """
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    economy, _ = steady(competence, seed, config)

    ruin(economy, best)
    window(economy, DAMAGE_HORIZON - WINDOW)
    _, share = window(economy, WINDOW)
    wealth = economy.mob.expert_wealth
    return float(share[best]), float(wealth[best]), float(wealth.median())
