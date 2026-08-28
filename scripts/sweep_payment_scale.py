"""Measure what deviating from the balanced VCG transfer costs the wealth economy.

``payment_scale`` is not fitted. Reports, prices and rewards share one unit --
loss reduction -- and reward and charge share one coefficient, so 1.0 is the value
that makes ``reward - charge`` a quasi-linear utility and puts wealth's break-even
exactly at the auction's price. This script sweeps around that point to show what
over- and under-pricing does to the charge fraction and the wealth distribution,
so the choice is checkable rather than asserted in a comment nobody can re-run.

Run:  uv run python scripts/sweep_payment_scale.py
"""

import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from mob import MixtureOfBidders, MoBConfig  # noqa: E402

BASE = MoBConfig(
    num_experts=8,
    top_k=2,
    hidden_dim=64,
    intermediate_dim=128,
    adapter_rank=8,
    adapter_alpha=8.0,
)
# A nonconstant routing objective: without genuine competence differences there is
# nothing for the economy to specialise on and every scale looks the same.
COMPETENCE = torch.tensor([0.9, 0.7, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1])
# 1.0 is the quasi-linear point; the sweep brackets it to show what deviating costs.
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SEEDS = (0, 1, 2)
STEPS = 400


def gini(wealth: torch.Tensor) -> float:
    ordered = torch.sort(wealth)[0]
    n = len(ordered)
    index = torch.arange(1, n + 1, dtype=ordered.dtype)
    return ((2 * (index * ordered).sum()) / (n * ordered.sum()) - (n + 1) / n).abs().item()


def run(payment_scale: float, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    mob = MixtureOfBidders(replace(BASE, payment_scale=payment_scale))
    mob.train()

    charges: list[float] = []
    rewards: list[float] = []
    original = mob._vcg_charges

    def spy(payments, selected, num_tokens, reward_multiplier):
        charge = original(payments, selected, num_tokens, reward_multiplier)
        charges.append(charge.sum().item())
        return charge

    mob._vcg_charges = spy

    for _ in range(STEPS):
        mob(torch.randn(2, 16, 64))
        selected = mob._cached_selected_experts
        quality = COMPETENCE[selected].mean(dim=-1)
        loss = ((2.0 - quality) + 0.05 * torch.randn(2, 16)).abs()

        before = mob.expert_wealth.clone()
        mob.update_wealth_from_loss(loss)
        gross = (mob.expert_wealth - before * BASE.wealth_decay).sum().item() + charges[-1]
        rewards.append(gross)

    wealth = mob.expert_wealth
    return {
        "charge_pct": 100 * sum(charges) / max(sum(abs(r) for r in rewards), 1e-9),
        "gini": gini(wealth),
        "spread": (wealth.max() - wealth.min()).item(),
        "floored": float((wealth <= BASE.min_wealth + 1e-4).sum().item()),
    }


def main() -> None:
    print(f"{'scale':>6} {'chg%rew':>8} {'gini':>17} {'spread':>9} {'floored':>8}")
    for scale in SCALES:
        runs = [run(scale, seed) for seed in SEEDS]
        ginis = [r["gini"] for r in runs]
        print(
            f"{scale:>6.2f} "
            f"{statistics.mean(r['charge_pct'] for r in runs):>7.1f}% "
            f"{statistics.mean(ginis):>7.3f}+-{statistics.pstdev(ginis):<8.3f} "
            f"{statistics.mean(r['spread'] for r in runs):>9.2f} "
            f"{statistics.mean(r['floored'] for r in runs):>8.1f}"
        )

    print(
        "\npayment_scale is no longer fitted: 1.0 is the coefficient that makes\n"
        "reward - charge a single quasi-linear utility. This sweep is a check on\n"
        "what deviating from it costs, not a search for a value."
    )


if __name__ == "__main__":
    main()
