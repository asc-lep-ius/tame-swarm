"""Measures of what experts *do*, which is what "specialisation" is a claim about.

The project's headline number has been the Gini coefficient of the wealth vector.
It is a reasonable health statistic for the economy and it is not evidence of
specialisation: it measures dispersion of a quantity produced by an EMA with a
hand-tuned decay, a competitive bonus and a hard clamp, so its value is largely a
property of that update rule's fixed point. A Gini of 0.12-0.35 is entirely
consistent with every expert computing the same function.

Worse than unsupported, its direction is backwards. Wealth multiplies the report
inside the bid, so a *rising* Gini mechanically increases wealth's share of the
routing decision -- meaning less of the routing is decided by what an expert
reports about the token in front of it. Read as a specialisation curve it points
the wrong way.

The three measures here are about function rather than balance:

* **Pairwise expert output divergence** -- run every expert over the *same*
  held-out hidden states and compare the outputs. Experts computing near-identical
  functions score near zero however unequal their wealth.
* **Per-expert routing profiles** -- which token categories each expert actually
  wins, and how far that is from the corpus marginal. This is the measure that can
  say *what* an expert specialised on, not merely that something differed.
* **Report decisiveness** -- the fraction of tokens whose top-1 winner is the
  expert with the highest report, i.e. how much of the allocation the reports
  govern rather than the wealth scalar. On the synthetic economy this ran at
  31-33% for the auction at a wealth spread of only 2.0x, which is why it is
  reported per arm rather than assumed to be near 1.

Probe sizes follow the #12 measurement note: report decisiveness needs >=4096
tokens before a single arm's estimate is stable to about a point, and between-seed
spread of ~46 points survives every probe size from 32 to 16,384, so per-seed
values are reported rather than only a mean.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from evaluation import HeldOutSplit
from mob import (
    MIN_TOKENS_FOR_CORRELATION,
    LightweightExpert,
    MixtureOfBidders,
    RoutingTraceSummary,
    frozen_economy,
    get_mob_layers,
    mean_goal_correlation,
    mob_layers_by_index,
    pearson_by_expert,
)

logger = logging.getLogger(__name__)

# Tokens used for pairwise expert output divergence. Smaller than the routing
# probe on purpose: this one runs every expert over every probed token, so it costs
# num_experts FFN forwards per layer, while a mean pairwise distance over
# high-dimensional outputs is already tight at a few hundred tokens.
DIVERGENCE_PROBE_TOKENS = 512

# Guards a division by the norm of an expert output that is identically zero. Only
# reachable at initialisation, where LoRA-B is zeroed and every expert returns the
# shared base output -- which is exactly the state whose divergence should read as
# zero rather than as a NaN.
NORM_EPSILON = 1e-12

CATEGORY_WORD_START = 0
CATEGORY_SUBWORD = 1
CATEGORY_DIGIT = 2
CATEGORY_PUNCTUATION = 3
CATEGORY_WHITESPACE = 4
CATEGORY_SPECIAL = 5
CATEGORY_NAMES = ("word", "subword", "digit", "punct", "space", "special")

# The leading-space markers SentencePiece and byte-level BPE use. A token carrying
# one starts a word; one without continues the previous token, and that distinction
# is the most basic thing an expert could plausibly specialise on.
WORD_START_MARKERS = ("▁", "Ġ")


@dataclass(frozen=True)
class DivergenceResult:
    """How differently the experts of one layer compute, on identical inputs.

    Two readings of the same forward, and they answer different questions.

    The ``*_cosine_distance`` fields compare each expert's whole **output**. With a
    shared base that output is the base FFN plus the expert's adapters, so it is
    diluted by everything the experts have in common: the planted-competence
    fixture, whose experts scale one identical correction, reads 0.018 here
    because the correction sits on a shared base, and a real model at 0.00044 may
    be an undifferentiated tissue or a differentiated one whose adapters are one
    percent of the stream. It cannot tell those apart.

    The ``*_contribution_*`` fields compare what each expert **adds** to the shared
    base -- the same quantity the economy prices (#15) and the only thing a routing
    decision can choose between. Zero here means the cells are interchangeable
    however the outputs look; on the fixture above it reads 1e-8. This is the
    number #25's differentiation checkpoint gates on. ``contribution_norm_ratio``
    is the dilution itself: the contribution norm over the output norm, above one when
    the adapters oppose the base rather than shade it.
    """

    mean_cosine_distance: float
    mean_relative_l2: float
    min_cosine_distance: float
    max_cosine_distance: float
    mean_contribution_cosine_distance: float = 0.0
    min_contribution_cosine_distance: float = 0.0
    max_contribution_cosine_distance: float = 0.0
    contribution_norm_ratio: float = 0.0


@dataclass(frozen=True)
class RoutingProfile:
    """Which token categories each expert wins, and how unusual that is."""

    per_expert_category_share: torch.Tensor
    corpus_category_share: torch.Tensor
    mean_js_from_corpus: float
    mean_kl_from_uniform: float
    expert_token_share: torch.Tensor


@dataclass(frozen=True)
class SpecialisationReport:
    """One arm's held-out probe, reduced.

    ``goal_correlation`` and ``win_share`` are the two columns #24's routing
    contrast is taken over, measured on this arm's probe tokens against the goal
    direction the caller supplied (``probe_specialisation(goal_directions=...)``):
    per expert, the Pearson correlation of a token's alignment with whether the
    expert won a slot on it, and the expert's share of the slots -- the same two
    statistics the served trace reports, so ``compare_runs.py`` can difference a
    coupled arm against an uncoupled one the way the outcome probe differences
    served against unsteered. ``None`` when no direction reached a probed layer.
    """

    divergence: DivergenceResult
    profile: RoutingProfile
    report_decisiveness: float
    probe_tokens: int
    goal_correlation: list[float | None] | None = None
    win_share: list[float] | None = None

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics. A per-expert key is absent, not zero, where the value is ``None``."""
        metrics = {
            "spec/expert_cosine_distance": self.divergence.mean_cosine_distance,
            "spec/expert_relative_l2": self.divergence.mean_relative_l2,
            "spec/expert_contribution_cosine_distance": (
                self.divergence.mean_contribution_cosine_distance
            ),
            "spec/expert_contribution_norm_ratio": self.divergence.contribution_norm_ratio,
            "spec/routing_js_from_corpus": self.profile.mean_js_from_corpus,
            "spec/routing_kl_from_uniform": self.profile.mean_kl_from_uniform,
            "spec/report_decisiveness": self.report_decisiveness,
            "spec/probe_tokens": float(self.probe_tokens),
        }
        for expert, value in enumerate(self.goal_correlation or []):
            if value is not None:
                metrics[f"routing/goal_correlation_e{expert}"] = value
        for expert, value in enumerate(self.win_share or []):
            metrics[f"routing/win_share_e{expert}"] = value
        return metrics


def token_categories(tokenizer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Map token ids to coarse lexical categories.

    Coarse on purpose. A finer taxonomy (part of speech, topic) would need a parser
    whose own errors are not separable from the routing signal, while these six are
    read straight off the token string and mean the same thing for any tokenizer
    with a leading-space convention. They are enough to answer the question that
    matters first -- *does any expert's intake differ from the corpus at all* --
    and a richer taxonomy is only worth building once the answer is yes.
    """
    flat = input_ids.reshape(-1).tolist()
    pieces = tokenizer.convert_ids_to_tokens(flat)
    special = set(getattr(tokenizer, "all_special_tokens", []) or [])

    categories = []
    for piece in pieces:
        if piece in special:
            categories.append(CATEGORY_SPECIAL)
            continue

        starts_word = piece.startswith(WORD_START_MARKERS)
        core = piece.lstrip("".join(WORD_START_MARKERS)).strip()

        if not core:
            categories.append(CATEGORY_WHITESPACE)
        elif core.isdigit():
            categories.append(CATEGORY_DIGIT)
        elif not any(character.isalnum() for character in core):
            categories.append(CATEGORY_PUNCTUATION)
        elif starts_word:
            categories.append(CATEGORY_WORD_START)
        else:
            categories.append(CATEGORY_SUBWORD)

    return torch.tensor(categories, dtype=torch.long).view_as(input_ids)


def expert_output_divergence(
    mob: MixtureOfBidders, hidden_states: torch.Tensor
) -> DivergenceResult:
    """Compare every pair of experts on the same tokens.

    Cosine distance is the headline because it is invariant to the output scale --
    two experts differing only by a gain are not two functions -- and relative L2 is
    reported beside it because cosine alone cannot tell a small difference in
    direction from a large one.

    Every expert is run over *all* the probed tokens, ignoring who actually won
    them. That is the point: a routing-conditioned comparison measures the router,
    and this is meant to measure the experts.
    """
    outputs: list[torch.Tensor] = []
    contributions: list[torch.Tensor] = []
    with torch.no_grad():
        for expert in mob.experts:
            if mob.use_shared_base:
                # The reference is the shared base alone; the expert's contribution
                # is what the economy prices, and what routing can choose between.
                output, reference = cast(LightweightExpert, expert).forward_with_reference(
                    hidden_states, mob.base_gate_proj, mob.base_up_proj, mob.base_down_proj
                )
                contributions.append((output - reference).float())
            else:
                # A full expert shares nothing, so its whole output is its own.
                output = expert(hidden_states)
                contributions.append(output.float())
            outputs.append(output.float())

    if len(outputs) < 2:
        raise ValueError("Expert divergence needs at least two experts")

    cosine_distances: list[float] = []
    relative_l2: list[float] = []
    contribution_distances: list[float] = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            first, second = outputs[i], outputs[j]
            cosine_distances.append(_mean_cosine_distance(first, second))
            scale = 0.5 * (first.norm(dim=-1) + second.norm(dim=-1))
            relative_l2.append(
                float(((first - second).norm(dim=-1) / scale.clamp_min(NORM_EPSILON)).mean().item())
            )
            contribution_distances.append(_mean_cosine_distance(contributions[i], contributions[j]))

    output_norm = torch.stack([output.norm(dim=-1) for output in outputs]).mean()
    contribution_norm = torch.stack([added.norm(dim=-1) for added in contributions]).mean()

    return DivergenceResult(
        mean_cosine_distance=sum(cosine_distances) / len(cosine_distances),
        mean_relative_l2=sum(relative_l2) / len(relative_l2),
        min_cosine_distance=min(cosine_distances),
        max_cosine_distance=max(cosine_distances),
        mean_contribution_cosine_distance=sum(contribution_distances) / len(contribution_distances),
        min_contribution_cosine_distance=min(contribution_distances),
        max_contribution_cosine_distance=max(contribution_distances),
        contribution_norm_ratio=float(
            (contribution_norm / output_norm.clamp_min(NORM_EPSILON)).item()
        ),
    )


def _mean_cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Mean over tokens of ``1 - cos``, with two zero vectors counted as identical.

    ``cosine_similarity`` returns 0 for a pair of zero vectors, which would read as
    a distance of 1 -- maximally different -- exactly where the experts are most
    alike: at upcycling every contribution is zero. A zero against a nonzero
    vector stays at 1, which is the honest reading of one cell doing something
    and another nothing.
    """
    cosine = torch.nn.functional.cosine_similarity(first, second, dim=-1, eps=NORM_EPSILON)
    both_zero = (first.norm(dim=-1) < NORM_EPSILON) & (second.norm(dim=-1) < NORM_EPSILON)
    return float(torch.where(both_zero, torch.zeros_like(cosine), 1.0 - cosine).mean().item())


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p||q) in bits, over the last dimension, with empty bins contributing zero.

    ``xlogy`` rather than ``p * log(p/q)``: an expert that never wins a category
    gives ``0 * log 0``, which is 0 in the limit and NaN in floating point, and a
    category no expert wins would otherwise poison the whole row.
    """
    safe_q = q.clamp_min(torch.finfo(q.dtype).tiny)
    return (torch.xlogy(p, p) - torch.xlogy(p, safe_q)).sum(dim=-1) / torch.log(
        torch.tensor(2.0, dtype=p.dtype)
    )


def _winner_category_counts(
    selected_experts: torch.Tensor,
    categories: torch.Tensor,
    num_experts: int,
    num_categories: int,
) -> torch.Tensor:
    """Tokens each expert won, by category. Float64 so the shares divide exactly."""
    winners = selected_experts[..., 0].reshape(-1)
    flat_categories = categories.reshape(-1)[: winners.numel()]

    counts = torch.zeros(num_experts, num_categories, dtype=torch.float64)
    counts.index_put_(
        (winners.cpu().long(), flat_categories.cpu().long()),
        torch.ones(winners.numel(), dtype=torch.float64),
        accumulate=True,
    )
    return counts


def routing_profiles(
    selected_experts: torch.Tensor,
    categories: torch.Tensor,
    num_experts: int,
    num_categories: int = len(CATEGORY_NAMES),
) -> RoutingProfile:
    """What each expert's intake looks like, against the corpus and against uniform.

    Divergence from the **corpus marginal** is the load-bearing number: it asks
    whether an expert's intake differs from what any expert would see by routing at
    random, which is the null a specialisation claim has to beat. Divergence from
    **uniform over categories** is reported too because #12 asks for it, but it is
    largely a statement about English -- subwords are far more common than digits --
    and every expert scores high on it whether or not it specialised.

    Jensen-Shannon is used against the corpus because it is symmetric and finite
    even when an expert wins no token of some category, which is common at small
    probe sizes and is not the same event as a genuinely divergent profile.
    """
    counts = _winner_category_counts(selected_experts, categories, num_experts, num_categories)

    expert_totals = counts.sum(dim=-1, keepdim=True)
    per_expert = counts / expert_totals.clamp_min(1.0)
    corpus = counts.sum(dim=0) / counts.sum().clamp_min(1.0)
    uniform = torch.full((num_categories,), 1.0 / num_categories, dtype=torch.float64)

    mixture = 0.5 * (per_expert + corpus.unsqueeze(0))
    js = 0.5 * _kl(per_expert, mixture) + 0.5 * _kl(corpus.expand_as(per_expert), mixture)

    # An expert that won nothing has no profile; averaging its all-zero row in would
    # report a divergence it never earned.
    active = (expert_totals.squeeze(-1) > 0).to(torch.float64)
    active_count = active.sum().clamp_min(1.0)

    return RoutingProfile(
        per_expert_category_share=per_expert,
        corpus_category_share=corpus,
        mean_js_from_corpus=float(((js * active).sum() / active_count).item()),
        mean_kl_from_uniform=float(
            ((_kl(per_expert, uniform.expand_as(per_expert)) * active).sum() / active_count).item()
        ),
        expert_token_share=(expert_totals.squeeze(-1) / counts.sum().clamp_min(1.0)),
    )


def report_decisiveness(selected_experts: torch.Tensor, confidences: torch.Tensor) -> float:
    """Fraction of tokens whose top-1 winner is the expert with the highest report.

    Defined identically for both arms, which is what makes it comparable: under the
    auction the realised winner maximises ``report x wealth``, so this measures how
    often wealth overturns the report; under the softmax control the winner
    maximises the report alone, so it is 1.0 by construction and says so. All of
    #10's strategyproofness work applies to the report, and this is the number that
    says how much of the allocation the report actually decides.
    """
    top_by_report = confidences.argmax(dim=-1)
    realised = selected_experts[..., 0]
    return float((top_by_report == realised).float().mean().item())


@dataclass
class _ProbeCapture:
    """One pass over the probe split, before it is reduced to a single report.

    Padding is excluded from everything recorded here. The split is padded to
    ``max_seq_length``, so on a corpus of short documents the pads outnumber the
    real tokens -- and a pad is not inert: it carries an id, takes a category and is
    routed like any other position. Scoring pads drags every statistic towards
    whatever the gate does with a constant input, and lets the >=4096-token floor be
    met by counting padding.
    """

    selected_per_layer: list[list[torch.Tensor]]
    confidences_per_layer: list[list[torch.Tensor]]
    hidden_states: dict[int, list[torch.Tensor]]
    # The goal direction each layer's tokens are aligned against, unit norm, keyed
    # by ``id(mob)``; a layer no direction reaches records no alignment (#24).
    directions: dict[int, torch.Tensor]
    alignment_per_layer: list[list[torch.Tensor]]
    # The batch in flight: the pre-hook writes each layer's unpadded alignment
    # here and ``_record_layer_stats`` truncates it to the tokens still wanted.
    batch_alignment: dict[int, torch.Tensor] = field(default_factory=dict)
    category_chunks: list[torch.Tensor] = field(default_factory=list)
    seen_tokens: int = 0

    @classmethod
    def empty(
        cls,
        mob_layers: list[MixtureOfBidders],
        directions: dict[int, torch.Tensor] | None = None,
    ) -> "_ProbeCapture":
        return cls(
            selected_per_layer=[[] for _ in mob_layers],
            confidences_per_layer=[[] for _ in mob_layers],
            hidden_states={id(mob): [] for mob in mob_layers},
            directions=dict(directions or {}),
            alignment_per_layer=[[] for _ in mob_layers],
        )

    @property
    def categories(self) -> torch.Tensor:
        if not self.category_chunks:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(self.category_chunks)


def _register_divergence_hooks(
    mob_layers: list[MixtureOfBidders],
    capture: "_ProbeCapture",
    current_keep: list[torch.Tensor | None],
    divergence_tokens: int,
) -> list[Any]:
    """Capture the hidden states entering each MoB layer, minus the padded rows.

    A pre-hook sees a hidden-state tensor and nothing else, so the mask for the batch
    in flight is handed to it through ``current_keep`` rather than rederived. The
    same stream is what the confidence heads read, so a layer with a goal direction
    also records each token's alignment with it here -- every unpadded token, not
    only the divergence prefix, since a correlation needs the whole window.
    """

    def make_hook(mob: MixtureOfBidders) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            hidden = cast(torch.Tensor, inputs[0]).detach()
            rows = hidden.reshape(-1, hidden.shape[-1])
            keep = current_keep[0]
            if keep is not None:
                rows = rows[keep]
            direction = capture.directions.get(id(mob))
            if direction is not None:
                stream = rows.float()
                capture.batch_alignment[id(mob)] = (
                    (stream @ direction.to(stream.device)) / stream.norm(dim=-1).clamp_min(1e-12)
                ).cpu()
            store = capture.hidden_states[id(mob)]
            collected = sum(tensor.shape[0] for tensor in store)
            if collected >= divergence_tokens:
                return
            store.append(rows[: divergence_tokens - collected])

        return hook

    return [mob.register_forward_pre_hook(make_hook(mob)) for mob in mob_layers]


def _record_layer_stats(
    mob_layers: list[MixtureOfBidders],
    keep: torch.Tensor,
    wanted: int,
    capture: _ProbeCapture,
) -> None:
    """Append this batch's unpadded routing decisions, one entry per layer.

    A layer with no statistics raises rather than being skipped: skipping advances
    the shared category chunk without advancing that layer's winners, silently
    misaligning the two for every batch that follows -- wrong numbers rather than an
    error.
    """
    for index, mob in enumerate(mob_layers):
        stats = mob.last_stats
        if stats is None:
            raise RuntimeError(
                f"MoB layer {index} recorded no statistics during the specialisation "
                "probe; its routing decisions cannot be aligned with the token categories"
            )
        selected = stats.selected_experts.reshape(-1, stats.selected_experts.shape[-1])
        confidences = stats.confidences.reshape(-1, stats.confidences.shape[-1])
        capture.selected_per_layer[index].append(selected[keep][:wanted].cpu())
        capture.confidences_per_layer[index].append(confidences[keep][:wanted].cpu())
        alignment = capture.batch_alignment.pop(id(mob), None)
        if alignment is not None:
            capture.alignment_per_layer[index].append(alignment[:wanted])


def _collect_probe_data(
    model: torch.nn.Module,
    mob_layers: list[MixtureOfBidders],
    split: HeldOutSplit,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    probe_tokens: int,
    divergence_tokens: int,
    directions: dict[int, torch.Tensor] | None = None,
) -> _ProbeCapture:
    """Run the probe split, recording only the positions the attention mask keeps."""
    capture = _ProbeCapture.empty(mob_layers, directions)
    current_keep: list[torch.Tensor | None] = [None]
    handles = _register_divergence_hooks(mob_layers, capture, current_keep, divergence_tokens)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), frozen_economy(model):
            for batch in split.batches(batch_size):
                if capture.seen_tokens >= probe_tokens:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                keep = attention_mask.reshape(-1).bool()
                current_keep[0] = keep

                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                wanted = min(probe_tokens - capture.seen_tokens, int(keep.sum()))
                if wanted == 0:
                    continue

                categories = token_categories(tokenizer, input_ids.cpu()).reshape(-1)
                capture.category_chunks.append(categories[keep.cpu()][:wanted])
                _record_layer_stats(mob_layers, keep, wanted, capture)
                capture.seen_tokens += wanted
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return capture


def _reduce_capture(
    capture: _ProbeCapture,
    mob_layers: list[MixtureOfBidders],
    device: torch.device,
) -> SpecialisationReport | None:
    """Pool the per-layer measurements into one report.

    Every layer sees the same tokens, so a mean over layers of a per-layer mean is
    the pooled quantity.
    """
    num_experts = mob_layers[0].config.num_experts
    categories = capture.categories
    divergences = [
        expert_output_divergence(mob, torch.cat(capture.hidden_states[id(mob)]).to(device))
        for mob in mob_layers
        if capture.hidden_states[id(mob)]
    ]
    profiles = [
        routing_profiles(torch.cat(selected), categories, num_experts)
        for selected in capture.selected_per_layer
        if selected
    ]
    decisiveness = [
        report_decisiveness(torch.cat(selected), torch.cat(confidences))
        for selected, confidences in zip(
            capture.selected_per_layer, capture.confidences_per_layer, strict=True
        )
        if selected and confidences
    ]

    if not divergences or not profiles or not decisiveness:
        logger.warning("Specialisation probe captured no MoB activity; skipping report")
        return None

    return SpecialisationReport(
        divergence=_pool_divergence(divergences),
        profile=_pool_profiles(profiles),
        report_decisiveness=_mean(iter(decisiveness)),
        probe_tokens=capture.seen_tokens,
        goal_correlation=_goal_correlation(capture, num_experts),
        win_share=_win_share(capture, num_experts),
    )


def _wins(selected: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``(tokens, num_experts)`` of 0/1: whether each expert held a slot on the token."""
    wins = torch.zeros(selected.shape[0], num_experts)
    wins.scatter_(1, selected.long(), 1.0)
    return wins


def _goal_correlation(capture: _ProbeCapture, num_experts: int) -> list[float | None] | None:
    """The served trace's ``goal_correlation``, per layer over the probe tokens, then averaged.

    Only layers that carried a direction and at least
    ``MIN_TOKENS_FOR_CORRELATION`` tokens contribute, on the served trace's own
    rule; the layer average is the served surface's too (``mean_goal_correlation``).
    """
    layers: list[RoutingTraceSummary] = []
    for selected, alignment in zip(
        capture.selected_per_layer, capture.alignment_per_layer, strict=True
    ):
        if not selected or not alignment:
            continue
        chosen = torch.cat(selected)
        aligned = torch.cat(alignment)[: chosen.shape[0]]
        if aligned.shape[0] < MIN_TOKENS_FOR_CORRELATION:
            continue
        correlation = pearson_by_expert(aligned, _wins(chosen[: aligned.shape[0]], num_experts))
        layers.append(
            RoutingTraceSummary(
                tokens=int(aligned.shape[0]),
                top1_mean=0.0,
                top1_median=0.0,
                top1_saturated_fraction=0.0,
                effective_experts=0.0,
                win_share=[],
                goal_alignment_mean=float(aligned.mean()),
                goal_correlation=correlation,
            )
        )
    return mean_goal_correlation(layers, num_experts)


def _win_share(capture: _ProbeCapture, num_experts: int) -> list[float] | None:
    """Each expert's share of the slots over the probe tokens, averaged over the layers.

    Every slot, not the top-1 winner ``routing_profiles`` counts: this pairs with
    the served trace's ``win_share`` so the two can be read against each other,
    and sums to ``top_k`` for the same reason.
    """
    shares = [
        _wins(torch.cat(selected), num_experts).mean(dim=0)
        for selected in capture.selected_per_layer
        if selected
    ]
    if not shares:
        return None
    return torch.stack(shares).mean(dim=0).tolist()


def _pool_divergence(divergences: list[DivergenceResult]) -> DivergenceResult:
    return DivergenceResult(
        mean_cosine_distance=_mean(d.mean_cosine_distance for d in divergences),
        mean_relative_l2=_mean(d.mean_relative_l2 for d in divergences),
        min_cosine_distance=min(d.min_cosine_distance for d in divergences),
        max_cosine_distance=max(d.max_cosine_distance for d in divergences),
        mean_contribution_cosine_distance=_mean(
            d.mean_contribution_cosine_distance for d in divergences
        ),
        min_contribution_cosine_distance=min(
            d.min_contribution_cosine_distance for d in divergences
        ),
        max_contribution_cosine_distance=max(
            d.max_contribution_cosine_distance for d in divergences
        ),
        contribution_norm_ratio=_mean(d.contribution_norm_ratio for d in divergences),
    )


def _pool_profiles(profiles: list[RoutingProfile]) -> RoutingProfile:
    return RoutingProfile(
        per_expert_category_share=torch.stack([p.per_expert_category_share for p in profiles]).mean(
            dim=0
        ),
        corpus_category_share=profiles[0].corpus_category_share,
        mean_js_from_corpus=_mean(p.mean_js_from_corpus for p in profiles),
        mean_kl_from_uniform=_mean(p.mean_kl_from_uniform for p in profiles),
        expert_token_share=torch.stack([p.expert_token_share for p in profiles]).mean(dim=0),
    )


def probe_specialisation(
    model: torch.nn.Module,
    split: HeldOutSplit,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    probe_tokens: int,
    divergence_tokens: int = DIVERGENCE_PROBE_TOKENS,
    goal_directions: dict[int, torch.Tensor] | None = None,
) -> SpecialisationReport | None:
    """Run the held-out probe and reduce it to one report per model.

    ``probe_tokens`` counts real tokens: padded positions are excluded from every
    statistic, so the number reported is the number the >=4096 floor is met with.

    ``goal_directions`` maps a block index to the goal direction injected there
    -- ``homeostat.projected_direction``, never the raw vector -- and turns on the
    routing columns #24's contrast is taken over. A block that is not a MoB layer
    is ignored; a MoB layer with no direction records routing and no alignment.

    Returns ``None`` for a model with no MoB layers -- the ``dense`` arm has no
    experts to diverge and no gate to profile, which is not a failure.
    """
    mob_layers = get_mob_layers(model)
    if not mob_layers:
        return None

    by_index = mob_layers_by_index(model)
    directions = {
        id(by_index[layer]): direction.detach().reshape(-1).float()
        / direction.detach().reshape(-1).float().norm().clamp_min(1e-12)
        for layer, direction in (goal_directions or {}).items()
        if layer in by_index
    }
    capture = _collect_probe_data(
        model,
        mob_layers,
        split,
        tokenizer,
        device,
        batch_size,
        probe_tokens,
        divergence_tokens,
        directions,
    )

    if capture.seen_tokens == 0:
        logger.warning("Specialisation probe found no unpadded tokens; skipping report")
        return None

    if capture.seen_tokens < probe_tokens:
        logger.warning(
            f"Specialisation probe collected {capture.seen_tokens} of {probe_tokens} requested "
            "unpadded tokens; below ~1000 tokens a single arm carries several points of noise "
            "on report decisiveness, which is the same order as the effect between arms"
        )

    return _reduce_capture(capture, mob_layers, device)


def _mean(values: Iterable[float]) -> float:
    materialised = list(values)
    return sum(materialised) / len(materialised)
