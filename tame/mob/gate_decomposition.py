"""The auction's gate split into its wealth and confidence terms, per token (#62).

A bid is ``confidence x wealth``, so the gate a token is decided on is
``log(confidence) + log(wealth)``: the cell's report about *this* token plus the
ledger's memory of every token before it. #11 put ``log(wealth)`` exactly where
DeepSeek's load-balancing bias sits, and #16 found the ledger settling into a
two-valued lattice. Together those raise the question this module answers with
a number: how much of the per-token variation of the gate across cells is the
report, and how often the winner set is simply wealth's top-*k*. A market that
allocates by history is a seniority system, and in one no per-token readout can
see allocation whatever the cells' loudness -- which would make #58's chance
reading true by construction rather than by accident.

Everything here is arithmetic on tensors a step already produced. Nothing is
paid, nothing is settled, and the wealth read is the one the gate read: the
caller passes ``allocation_wealth()`` taken *before* the step, because the
ledger in ``MoBStats`` is the one the step left behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# A bid of exactly zero is unreachable in production (``auction.BID_LOG_FLOOR``
# says why); this keeps the log finite on a test that feeds one.
LOG_FLOOR = 1e-30

# Below this, the gate does not vary across cells on the token at all -- every
# bid equal -- and a fraction of nothing is not a number. Such tokens are counted
# rather than folded into a mean as zeros.
DEGENERATE_VARIANCE = 1e-24


@dataclass(frozen=True)
class GateDecomposition:
    """What the gate did on a set of tokens, and how much of it was the ledger.

    ``identity_gap`` is the largest ``|log(c) + log(w) - log(c x w)|`` over every
    cell on every token, with the product taken in the dtype the auction bid in
    -- the check that the two terms *are* the gate and not a model of it.

    ``seniority_fraction`` is the fraction of decidable tokens whose winner set
    equals wealth's top-*k*; ``sold_seniority_fraction`` is the same for the
    slots the auction *sold* (bid top-*k*), which differs from the winners only by
    the exploration gift. A token is undecidable when wealth's *k*-th and
    (*k*+1)-th cells tie -- on a pinned ledger every token is, and that is the
    ``decoupled`` control reading as it should.

    The three ``*_fraction`` variance terms average, over tokens with a
    non-degenerate gate, ``var(log c) / var(gate)``, ``var(log w) / var(gate)``
    and ``2 cov / var(gate)``; per token they sum to one exactly, so their means
    do too, and ``degenerate_fraction`` says how many tokens were left out.
    """

    tokens: int
    identity_gap: float
    seniority_fraction: float
    sold_seniority_fraction: float
    undecidable_fraction: float
    confidence_fraction: float
    wealth_fraction: float
    cross_fraction: float
    degenerate_fraction: float
    wealth_term_spread: float


def wealth_top_k(wealth: torch.Tensor, top_k: int) -> tuple[torch.Tensor, bool]:
    """Wealth's top-*k* cells, and whether that set is unique.

    The set is undecidable when the *k*-th and (*k*+1)-th wealths are equal: a
    ledger pinned at one value, or two cells on the same bound, gives the gate
    no ranking to hand the token to. Exact equality rather than a tolerance, so
    two cells one float apart on the ceiling still rank -- which is what the
    auction itself does with them.
    """
    if top_k >= wealth.numel():
        return torch.arange(wealth.numel()), True
    ranked = torch.sort(wealth.double(), descending=True)
    decidable = bool(ranked.values[top_k - 1] > ranked.values[top_k])
    return ranked.indices[:top_k], decidable


def _same_set(candidates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per row of ``candidates`` ``(T, k)``, whether it is ``reference`` ``(k,)`` as a set."""
    return torch.eq(
        torch.sort(candidates, dim=-1).values, torch.sort(reference).values.unsqueeze(0)
    ).all(dim=-1)


def decompose(
    confidences: torch.Tensor,
    wealth: torch.Tensor,
    selected: torch.Tensor,
    top_k: int,
    mask: torch.Tensor | None = None,
) -> GateDecomposition:
    """Split one step's gate on every token it saw.

    ``confidences`` is ``(..., n)`` as ``MoBStats.confidences`` records it,
    ``selected`` is ``(..., k)`` beside it, ``wealth`` is the ``(n,)`` vector the
    gate read for these tokens, and ``mask`` (``(...,)`` boolean) drops padding.
    """
    n = confidences.shape[-1]
    reports = confidences.reshape(-1, n)
    winners = selected.reshape(-1, selected.shape[-1])
    if mask is not None:
        keep = mask.reshape(-1).bool()
        reports, winners = reports[keep], winners[keep]
    if reports.shape[0] == 0:
        raise ValueError("no tokens to decompose")

    # The product in the bid's own dtype, as ``VCGAuctioneer.forward`` takes it;
    # the logs in float64, so the gap measures the auction's rounding and not ours.
    bids = reports * wealth.to(reports.dtype).unsqueeze(0)
    log_bid = bids.double().clamp_min(LOG_FLOOR).log()
    log_conf = reports.double().clamp_min(LOG_FLOOR).log()
    log_wealth = wealth.double().clamp_min(LOG_FLOOR).log()
    gate = log_conf + log_wealth.unsqueeze(0)
    identity_gap = float((gate - log_bid).abs().max())

    var_conf = log_conf.var(dim=-1, unbiased=False)
    var_wealth = log_wealth.var(unbiased=False)
    centred_wealth = log_wealth - log_wealth.mean()
    cov = ((log_conf - log_conf.mean(dim=-1, keepdim=True)) * centred_wealth).mean(dim=-1)
    var_gate = gate.var(dim=-1, unbiased=False)
    live = var_gate > DEGENERATE_VARIANCE
    if bool(live.any()):
        denominator = var_gate[live]
        confidence_fraction = float((var_conf[live] / denominator).mean())
        wealth_fraction = float((var_wealth / denominator).mean())
        cross_fraction = float((2.0 * cov[live] / denominator).mean())
    else:
        confidence_fraction = wealth_fraction = cross_fraction = float("nan")

    reference, decidable = wealth_top_k(wealth, top_k)
    tokens = reports.shape[0]
    if decidable:
        seniority = float(_same_set(winners, reference).double().mean())
        sold = torch.topk(bids, top_k, dim=-1).indices
        sold_seniority = float(_same_set(sold, reference).double().mean())
    else:
        seniority = sold_seniority = float("nan")

    return GateDecomposition(
        tokens=tokens,
        identity_gap=identity_gap,
        seniority_fraction=seniority,
        sold_seniority_fraction=sold_seniority,
        undecidable_fraction=0.0 if decidable else 1.0,
        confidence_fraction=confidence_fraction,
        wealth_fraction=wealth_fraction,
        cross_fraction=cross_fraction,
        degenerate_fraction=float(1.0 - live.double().mean()),
        wealth_term_spread=float(log_wealth.max() - log_wealth.min()),
    )


def _weighted(parts: list[GateDecomposition], field: str, weights: list[float]) -> float:
    """A token-weighted mean over the parts where the field is a number."""
    total = 0.0
    numerator = 0.0
    for part, weight in zip(parts, weights, strict=True):
        value = getattr(part, field)
        if value != value or weight == 0.0:  # NaN: the part had no such tokens
            continue
        numerator += value * weight
        total += weight
    return numerator / total if total > 0.0 else float("nan")


def merge(parts: list[GateDecomposition]) -> GateDecomposition:
    """Steps combined token-weighted, so a short last batch does not count as a long one.

    The seniority fractions weight by *decidable* tokens and the variance
    fractions by *non-degenerate* ones, since each was a mean over exactly that
    population in its part; the two exclusion fractions weight by all tokens.
    """
    if not parts:
        raise ValueError("nothing to merge")
    tokens = [float(part.tokens) for part in parts]
    decidable = [part.tokens * (1.0 - part.undecidable_fraction) for part in parts]
    live = [part.tokens * (1.0 - part.degenerate_fraction) for part in parts]
    return GateDecomposition(
        tokens=int(sum(tokens)),
        identity_gap=max(part.identity_gap for part in parts),
        seniority_fraction=_weighted(parts, "seniority_fraction", decidable),
        sold_seniority_fraction=_weighted(parts, "sold_seniority_fraction", decidable),
        undecidable_fraction=_weighted(parts, "undecidable_fraction", tokens),
        confidence_fraction=_weighted(parts, "confidence_fraction", live),
        wealth_fraction=_weighted(parts, "wealth_fraction", live),
        cross_fraction=_weighted(parts, "cross_fraction", live),
        degenerate_fraction=_weighted(parts, "degenerate_fraction", tokens),
        wealth_term_spread=max(part.wealth_term_spread for part in parts),
    )
