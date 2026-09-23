"""The arithmetic of a blockade read (#63), with no fixture in it.

The Operative Corollary's individuation operation looks for *means-substitution
under blockade*: an entity recruits new means toward the same end when the old
means are blocked, and a part does not. ``scripts/blockade.py`` drives the
fixture; everything here is what it does with the counts it brings back, kept
separate so the readouts can be checked on hand-built shares where the answer is
known before any economy runs.

Every share below is a share of the *slots on the tokens of one type* -- the
blocked cell's own type -- because that is the work the blocked cell was doing
and the only place a substitute can do it. On the quality fixture every token is
of the one type and these reduce to the plain win shares.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import torch
from scipy.stats import t as student_t


def dominant_cell(on_type_wins: torch.Tensor, competence: torch.Tensor) -> int:
    """The cell doing the most of the tissue's work: on-type slots held, weighted by competence.

    Not the wealthiest -- on the quality fixture both winners sit on the ceiling
    -- and not the most competent, which may be shut out on a legibility regime
    where it costs the most off-type. The cell whose removal takes the most
    delivered correction out of the output is the one whose blockade asks the
    question.
    """
    if on_type_wins.numel() != competence.numel():
        raise ValueError("on_type_wins and competence must have one entry per cell")
    return int((on_type_wins.float() * competence.float()).argmax())


def type_shares(type_wins: torch.Tensor) -> torch.Tensor:
    """Each cell's share of the slots on one type's tokens; zeros when the window saw none."""
    total = float(type_wins.sum())
    if total == 0.0:
        return torch.zeros_like(type_wins, dtype=torch.float32)
    return type_wins.float() / total


def uptake(pre_shares: torch.Tensor, window_shares: torch.Tensor, blocked: int) -> float:
    """Readout (a): the fraction of the blocked cell's pre-block share the others hold inside W.

    Shares on one type's tokens sum to one across cells in both windows, so what
    the blocked cell lost is exactly what the remaining cells took up; this is
    that loss as a fraction of what it held. Zero when nothing moved, one when
    the cell has been substituted for entirely, negative when it holds *more*
    inside the window than before -- which the ledger blockade on ``decoupled``
    can produce, since the gate there never reads the pinned ledger.
    """
    before = float(pre_shares[blocked])
    if before == 0.0:
        raise ValueError(
            "the blocked cell held nothing before the block; there is nothing to take up"
        )
    return (before - float(window_shares[blocked])) / before


def gains(pre_shares: torch.Tensor, window_shares: torch.Tensor) -> torch.Tensor:
    """Per cell, the share of the type's slots gained inside the window."""
    return window_shares - pre_shares


def winners(pre_shares: torch.Tensor, top_k: int) -> list[int]:
    """The ``top_k`` cells by pre-block share: those already holding a slot on the type's tokens."""
    return [int(index) for index in pre_shares.argsort(descending=True)[:top_k]]


def predicted_substitute(
    competence: torch.Tensor,
    on_type: torch.Tensor,
    pre_shares: torch.Tensor,
    blocked: int,
    top_k: int,
) -> int | None:
    """Op 2 step 5's prediction: the most competent on-type cell not already winning.

    A token routes ``top_k`` *distinct* cells, so the other pre-block winner
    already holds a slot on nearly every token of the type and cannot take the
    freed one; the auction's substitute, if the auction substitutes by
    competence, is the best of the rest. ``None`` when no on-type cell is left
    to predict, which the differentiated fixture reaches with two cells a type.
    """
    excluded = set(winners(pre_shares, top_k)) | {blocked}
    candidates = [
        index
        for index in range(competence.numel())
        if bool(on_type[index]) and index not in excluded
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda index: float(competence[index]))


def planted_statistic(
    cell_gains: torch.Tensor,
    on_type: torch.Tensor,
    pre_shares: torch.Tensor,
    blocked: int,
    predicted: int,
    top_k: int,
) -> float | None:
    """The predicted substitute's gain over the mean gain of the other eligible cells.

    Positive when the freed share went where competence says it should, about
    zero when the exploration draw scattered it, negative when it went to the
    wrong cell. It is the per-seed value the planted-effect calibration (section
    8 rule 7) reads under a paired t: the sign is known before the run. ``None``
    when no other on-type cell is eligible -- two cells a type, one blocked, one
    predicted -- so there is nothing to compare the prediction against.
    """
    excluded = set(winners(pre_shares, top_k)) | {blocked, predicted}
    others = [
        float(cell_gains[index])
        for index in range(cell_gains.numel())
        if bool(on_type[index]) and index not in excluded
    ]
    if not others:
        return None
    return float(cell_gains[predicted]) - statistics.fmean(others)


def returned(pre_shares: torch.Tensor, post_shares: torch.Tensor, blocked: int) -> float:
    """Readout (c): the blocked cell's share after release, as a fraction of what it held before."""
    before = float(pre_shares[blocked])
    if before == 0.0:
        raise ValueError("the blocked cell held nothing before the block")
    return float(post_shares[blocked]) / before


def half_life(shares_per_step: list[float], before: float, final: float) -> int | None:
    """The first step inside the window at which the blocked cell has lost half its eventual loss.

    ``None`` when the cell lost nothing over the window. A per-step share is
    noisy, so the step is read on the running mean rather than on the raw step.
    """
    drop = before - final
    if drop <= 0:
        return None
    running = 0.0
    for step, share in enumerate(shares_per_step, 1):
        running += share
        if before - running / step >= drop / 2:
            return step
    return None


def reconvergence_step(
    losses: list[float], target: float, factor: float, trailing: int
) -> int | None:
    """The first step whose trailing mean is within ``factor`` of ``target``; ``None`` if never.

    Counted from the first step of the blocked run. The trailing mean is what
    makes a single lucky step not count as convergence, and it is why the
    earliest possible answer is ``trailing`` itself.
    """
    if trailing <= 0:
        raise ValueError("trailing must be positive")
    for end in range(trailing, len(losses) + 1):
        if statistics.fmean(losses[end - trailing : end]) <= factor * target:
            return end
    return None


@dataclass(frozen=True)
class PairedTest:
    """A paired t on per-seed deltas, with the effect size #56's helper prices."""

    mean: float
    sd: float
    dz: float
    t: float
    p: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "sd": self.sd,
            "dz": self.dz,
            "t": self.t,
            "p": self.p,
            "n": float(self.n),
        }


def paired_t(deltas: list[float]) -> PairedTest:
    """Two-sided paired t against zero; identical deltas have no spread and an infinite t."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a paired t needs at least two deltas")
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0.0:
        infinite = math.inf if mean != 0.0 else 0.0
        return PairedTest(
            mean,
            0.0,
            math.copysign(infinite, mean),
            math.copysign(infinite, mean),
            0.0 if mean != 0.0 else 1.0,
            n,
        )
    dz = mean / sd
    t = dz * math.sqrt(n)
    p = float(2 * student_t.sf(abs(t), n - 1))
    return PairedTest(mean, sd, dz, t, p, n)
