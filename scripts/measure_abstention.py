"""Is winning profitable for a competent expert? The #15 measurement, re-runnable.

The issue was opened on a synthetic routing objective -- six experts, top-2, 300
steps -- reading a surplus of -0.22 per win, r(wealth, win share) = -0.28 and a
wealth ordering that tracked expert index whatever the competence was. This is
that measurement on the planted-competence fixture, with the competence shuffled
away from index so the number can mean something, and with the market read at its
steady state as well as over the whole run.

Run:  uv run python scripts/measure_abstention.py [--steps 400] [--window 100] [--seeds 0,1,2]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_economy import (  # noqa: E402
    DEFAULT_COMPETENCE,
    RunSummary,
    SyntheticEconomy,
    pearson,
    shuffled,
)


def measure(competence: torch.Tensor, seed: int, steps: int, window: int) -> RunSummary:
    return SyntheticEconomy(competence, seed=seed).run(steps, window=window)


def format_row(label: str, seed: int, summary: RunSummary) -> str:
    return (
        f"{label:<9}{seed:>5}"
        f"{summary.mean_realised_value:>9.4f}{summary.mean_price:>9.4f}{summary.mean_surplus:>9.4f}"
        f"{summary.final_realised_value:>9.4f}{summary.final_report:>9.4f}"
        f"{summary.final_price:>9.4f}{summary.final_surplus:>9.4f}"
        f"{summary.wealth_vs_win_share:>9.3f}{summary.wealth_vs_competence:>9.3f}"
        f"{summary.wealth_vs_index:>9.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]

    header = (
        f"{'order':<9}{'seed':>5}"
        f"{'value':>9}{'price':>9}{'surplus':>9}"
        f"{'value':>9}{'report':>9}{'price':>9}{'surplus':>9}"
        f"{'r(w,win)':>9}{'r(w,cmp)':>9}{'r(w,idx)':>9}"
    )
    print(f"{'':<14}{'--- whole run ---':^27}{'--- final window ---':^36}")
    print(header)
    print("-" * len(header))
    for seed in seeds:
        sorted_summary = measure(DEFAULT_COMPETENCE, seed, args.steps, args.window)
        print(format_row("sorted", seed, sorted_summary))
        competence = shuffled(DEFAULT_COMPETENCE, seed)
        shuffled_summary = measure(competence, seed, args.steps, args.window)
        print(format_row("shuffled", seed, shuffled_summary))
        index_correlation = pearson(competence, torch.arange(competence.numel()))
        competence_text = ", ".join(f"{value:.2f}" for value in competence.tolist())
        print(f"{'':<14}shuffled competence [{competence_text}]")
        print(f"{'':<14}r(competence, index) = {index_correlation:+.2f}")

    print(
        "\nsurplus > 0 in the final window is the #15 criterion: winning pays once the reports\n"
        "are calibrated. r(w,cmp) must survive the shuffle; r(w,idx) is only meaningful on the\n"
        "shuffled rows, where a value near the sorted row's would be the initialisation artefact."
    )


if __name__ == "__main__":
    main()
