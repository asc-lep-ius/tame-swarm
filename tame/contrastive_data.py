"""Behavioural contrastive pairs and the sets they group into.

The defect #3 exists to correct is not the *count* of contrastive examples but
their *shape*. The steering vectors shipped before this module were extracted
from instruction prefixes -- "Answer truthfully:" against "Make up a false
answer:" -- and a difference-in-means over those recovers the direction that
separates two English sentences *about* a behaviour, not the direction along
which the model *exhibits* it. CAA (Rimsky et al., 2024) and RepE (Zou et al.,
2023) contrast activations at the position where the model is producing the
behaviour: A/B completions of a shared prompt, read at the answer token.

A :class:`ContrastivePair` is therefore a *shared prompt* plus two contrasting
**completions**, with a recorded read position -- the token whose activation the
extractor reads. The instruction-prefix templates are retained (see
:data:`INSTRUCTION_PREFIX_CONTROL`) not because they are useful for steering but
because they are the artefact this design is trying to avoid extracting, which
makes them the negative control the behavioural validation measures against.

This module is the data and validation half; the extraction that reads these
pairs at their recorded positions lives in ``steering.py``, and the pipeline that
ties loading, extraction and normalisation together lives in
``steering_pipeline.py``. Loading published A/B datasets from HuggingFace is
optional and imports ``datasets`` lazily, so the ``serve`` extra never needs it.
"""

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

import torch

from contrastive_templates import (
    BUILTIN_PAIRS,
    INSTRUCTION_PREFIX_CONTROL,
    TIERS,
)

logger = logging.getLogger(__name__)

# Below this many pairs a difference-in-means estimate is dominated by
# example-specific artefacts rather than the shared behavioural direction; the
# activation-engineering literature puts the robust range at 50-200. Warned, not
# enforced: a caller registering a small custom set for a probe is legitimate.
MIN_PAIRS_PER_GOAL = 60
# Each difficulty tier must carry at least this many pairs, so "60 pairs" cannot
# be met by 58 easy ones and two hard ones -- the hard tier is what stress-tests
# the direction against adversarial, behaviour-preserving surface changes.
MIN_PAIRS_PER_TIER = 15
# Two completions whose representations sit this close carry almost no contrast,
# so their difference is noise added to the mean. A quality warning, not a
# rejection: the caller decides whether a near-duplicate pair earns its place.
MAX_COMPLETION_SIMILARITY = 0.95

VALID_TIERS = frozenset(TIERS)


@dataclass(frozen=True)
class ContrastivePair:
    """A shared prompt with two contrasting completions, read at one position.

    ``read_position`` indexes the *completion*, not the whole sequence: ``-1`` is
    the last completion token (the answer token, the CAA default), ``0`` the
    first. The extractor converts it to an absolute position once it knows where
    the prompt ends, which is the token boundary tokenisation actually produced --
    never a character offset.
    """

    prompt: str
    positive_completion: str
    negative_completion: str
    read_position: int = -1
    tier: str = "medium"
    source: str = "builtin"

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not self.positive_completion.strip() or not self.negative_completion.strip():
            raise ValueError("both completions must be non-empty")
        if self.tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {sorted(VALID_TIERS)}, got {self.tier!r}")

    @property
    def positive_text(self) -> str:
        return self.prompt + self.positive_completion

    @property
    def negative_text(self) -> str:
        return self.prompt + self.negative_completion


def _normalise(text: str) -> str:
    """Fold whitespace and case so trivially-restyled duplicates collide."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _token_set(text: str) -> set[str]:
    return set(_normalise(text).split())


def _jaccard(left: str, right: str) -> float:
    a, b = _token_set(left), _token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class QualityReport:
    """What a set's pairs look like, and every way they fall short.

    ``ok`` is the count/coverage verdict alone; the similarity findings are
    advisory because whether a near-duplicate pair earns its place is the caller's
    call, not this module's.
    """

    goal: str
    pair_count: int
    tier_counts: dict[str, int]
    duplicate_pairs: list[int]
    lexical_near_duplicates: list[tuple[int, int]]
    high_similarity_pairs: list[tuple[int, float]]
    warnings: list[str] = field(default_factory=list)

    @property
    def meets_count(self) -> bool:
        return self.pair_count >= MIN_PAIRS_PER_GOAL

    @property
    def meets_tier_coverage(self) -> bool:
        return all(self.tier_counts.get(tier, 0) >= MIN_PAIRS_PER_TIER for tier in VALID_TIERS)

    @property
    def ok(self) -> bool:
        return self.meets_count and self.meets_tier_coverage and not self.duplicate_pairs


@dataclass(frozen=True)
class ContrastivePairSet:
    """An immutable, deduplicated set of pairs for one goal.

    Construct through :meth:`from_pairs` rather than the bare constructor; it is
    the boundary at which exact duplicates are dropped and structure is checked,
    so nothing downstream has to re-validate.
    """

    goal: str
    pairs: tuple[ContrastivePair, ...]
    source: str = "builtin"

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    @classmethod
    def from_pairs(
        cls, goal: str, pairs: Iterable[ContrastivePair], source: str = "builtin"
    ) -> "ContrastivePairSet":
        deduped: list[ContrastivePair] = []
        seen: set[tuple[str, str, str]] = set()
        for pair in pairs:
            key = (pair.prompt, pair.positive_completion, pair.negative_completion)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(pair)
        if not deduped:
            raise ValueError(f"goal {goal!r}: no pairs after deduplication")
        return cls(goal=goal, pairs=tuple(deduped), source=source)

    def tier_counts(self) -> dict[str, int]:
        counts: dict[str, int] = dict.fromkeys(VALID_TIERS, 0)
        for pair in self.pairs:
            counts[pair.tier] += 1
        return counts

    def quality_report(
        self, embedder: Callable[[Sequence[str]], torch.Tensor] | None = None
    ) -> QualityReport:
        """Structural checks always; the representation-similarity check if given.

        ``embedder`` maps a batch of texts to a ``(len(texts), dim)`` tensor -- in
        practice the extraction model's own pooled activations, so the similarity
        is measured in the space the steering vector is read from rather than in a
        generic sentence-embedding space that the model never sees. Without one,
        the report falls back to a lexical near-duplicate scan, which catches the
        crude failure (two completions that are almost the same words) but not the
        semantic one (two different sentences the model represents identically).
        """
        warnings: list[str] = []
        duplicate_pairs = [
            index
            for index, pair in enumerate(self.pairs)
            if _normalise(pair.positive_completion) == _normalise(pair.negative_completion)
        ]
        if duplicate_pairs:
            warnings.append(
                f"{len(duplicate_pairs)} pair(s) have identical positive/negative completions"
            )

        lexical_near_duplicates = [
            (index, other)
            for index in range(len(self.pairs))
            for other in range(index + 1, len(self.pairs))
            if _jaccard(self.pairs[index].positive_text, self.pairs[other].positive_text) > 0.9
        ]

        high_similarity_pairs: list[tuple[int, float]] = []
        if embedder is not None:
            high_similarity_pairs = self._embedding_similarities(embedder)
            if high_similarity_pairs:
                warnings.append(
                    f"{len(high_similarity_pairs)} pair(s) exceed completion similarity "
                    f"{MAX_COMPLETION_SIMILARITY}: their contrast adds noise, not signal"
                )

        counts = self.tier_counts()
        if len(self.pairs) < MIN_PAIRS_PER_GOAL:
            warnings.append(f"{len(self.pairs)} pairs < recommended {MIN_PAIRS_PER_GOAL}")
        for tier in sorted(VALID_TIERS):
            if counts.get(tier, 0) < MIN_PAIRS_PER_TIER:
                warnings.append(f"tier {tier!r} has {counts.get(tier, 0)} < {MIN_PAIRS_PER_TIER}")

        return QualityReport(
            goal=self.goal,
            pair_count=len(self.pairs),
            tier_counts=counts,
            duplicate_pairs=duplicate_pairs,
            lexical_near_duplicates=lexical_near_duplicates,
            high_similarity_pairs=high_similarity_pairs,
            warnings=warnings,
        )

    def _embedding_similarities(
        self, embedder: Callable[[Sequence[str]], torch.Tensor]
    ) -> list[tuple[int, float]]:
        positives = embedder([pair.positive_text for pair in self.pairs])
        negatives = embedder([pair.negative_text for pair in self.pairs])
        cosines = torch.nn.functional.cosine_similarity(
            positives.float(), negatives.float(), dim=-1
        )
        return [
            (index, float(value))
            for index, value in enumerate(cosines.tolist())
            if value > MAX_COMPLETION_SIMILARITY
        ]


def _pairs_from_records(records: Iterable[dict], source: str) -> list[ContrastivePair]:
    return [
        ContrastivePair(
            prompt=record["prompt"],
            positive_completion=record["positive"],
            negative_completion=record["negative"],
            read_position=record.get("read_position", -1),
            tier=record.get("tier", "medium"),
            source=source,
        )
        for record in records
    ]


# Custom pairs registered at runtime, keyed by goal. Kept separate from the
# built-in templates so a caller cannot silently shadow or mutate them; a load
# with source="builtin" reads only the templates, source="custom" only these.
_CUSTOM_PAIRS: dict[str, list[ContrastivePair]] = {}


def register_contrastive_pairs(
    goal: str, pairs: Iterable[ContrastivePair | dict], replace_existing: bool = False
) -> ContrastivePairSet:
    """Register user-defined pairs for a goal, validated and deduplicated.

    Accepts either :class:`ContrastivePair` instances or the same dict shape the
    templates use. Returns the resulting set so the caller sees the effect of
    deduplication immediately. Warns rather than raises on a thin set: a probe
    with a handful of bespoke pairs is a legitimate use, distinct from a goal the
    system steers on by default.
    """
    materialised = [
        pair if isinstance(pair, ContrastivePair) else _pairs_from_records([pair], "custom")[0]
        for pair in pairs
    ]
    if not materialised:
        raise ValueError(f"goal {goal!r}: no pairs supplied")

    existing = [] if replace_existing else _CUSTOM_PAIRS.get(goal, [])
    combined = ContrastivePairSet.from_pairs(goal, [*existing, *materialised], source="custom")
    _CUSTOM_PAIRS[goal] = list(combined.pairs)

    report = combined.quality_report()
    for warning in report.warnings:
        logger.warning("Custom pairs for %r: %s", goal, warning)
    return combined


def clear_custom_pairs(goal: str | None = None) -> None:
    """Drop registered custom pairs -- one goal, or all. Mainly for tests."""
    if goal is None:
        _CUSTOM_PAIRS.clear()
    else:
        _CUSTOM_PAIRS.pop(goal, None)


def available_goals(source: str = "builtin") -> list[str]:
    if source == "custom":
        return sorted(_CUSTOM_PAIRS)
    return sorted(BUILTIN_PAIRS)


def load_contrastive_dataset(
    goal: str,
    source: str = "builtin",
    load_dataset: Callable[..., Iterable[dict]] | None = None,
) -> ContrastivePairSet:
    """Load one goal's pairs from the built-in templates, custom registrations, or HF.

    ``source`` is ``"builtin"`` (the hand-authored templates), ``"custom"`` (pairs
    registered via :func:`register_contrastive_pairs`), or a HuggingFace dataset
    name understood by :class:`HFContrastiveLoader`. The HF path imports
    ``datasets`` lazily and accepts an injected ``load_dataset`` so it is testable
    without the network.
    """
    if source == "builtin":
        if goal not in BUILTIN_PAIRS:
            raise ValueError(f"unknown goal {goal!r}; built-in goals: {available_goals()}")
        return ContrastivePairSet.from_pairs(
            goal, _pairs_from_records(BUILTIN_PAIRS[goal], "builtin"), source="builtin"
        )
    if source == "custom":
        if goal not in _CUSTOM_PAIRS:
            raise ValueError(f"no custom pairs registered for goal {goal!r}")
        return ContrastivePairSet.from_pairs(goal, _CUSTOM_PAIRS[goal], source="custom")

    return HFContrastiveLoader(load_dataset=load_dataset).load(goal, source)


def load_instruction_prefix_control(goal: str) -> ContrastivePairSet:
    """The retained instruction-prefix pairs, as the labelled negative control.

    These are the *old* extraction's inputs -- instruction prefixes, not
    behavioural completions -- rebuilt as pairs so the behavioural validation can
    extract a vector from them and show it scores measurably worse. If it does
    not, the validation metric is measuring prompt wording and is itself wrong.
    """
    if goal not in INSTRUCTION_PREFIX_CONTROL:
        raise ValueError(f"no instruction-prefix control for goal {goal!r}")
    return ContrastivePairSet.from_pairs(
        goal,
        _pairs_from_records(INSTRUCTION_PREFIX_CONTROL[goal], "instruction-prefix"),
        source="instruction-prefix-control",
    )


class HFContrastiveLoader:
    """Convert published A/B datasets from HuggingFace into the standard format.

    The datasets worth pulling are worth pulling *because* they are already A/B:
    ``truthful_qa`` pairs a question with a correct answer and distractors;
    ``Anthropic/hh-rlhf`` pairs a shared conversation with a chosen (harmless) and
    a rejected (harmful) continuation. Each converter knows one dataset's schema
    and emits :class:`ContrastivePair` objects with the answer token as the read
    position. Results are cached in-process so repeated goals in one run pay the
    conversion once.
    """

    def __init__(self, load_dataset: Callable[..., Iterable[dict]] | None = None):
        self._load_dataset = load_dataset
        self._cache: dict[tuple[str, str], ContrastivePairSet] = {}

    def load(self, goal: str, dataset: str, limit: int | None = None) -> ContrastivePairSet:
        key = (goal, dataset)
        if key in self._cache:
            return self._cache[key]

        converter = self._converters().get(dataset)
        if converter is None:
            raise ValueError(
                f"no HuggingFace converter for dataset {dataset!r}; "
                f"known: {sorted(self._converters())}"
            )
        pairs = converter(goal, limit)
        pair_set = ContrastivePairSet.from_pairs(goal, pairs, source=dataset)
        self._cache[key] = pair_set
        return pair_set

    def _resolve_loader(self) -> Callable[..., Iterable[dict]]:
        if self._load_dataset is not None:
            return self._load_dataset
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the train extra
            raise RuntimeError(
                "loading contrastive pairs from HuggingFace needs the 'train' extra "
                "(datasets); install it or pass source='builtin'"
            ) from exc
        return load_dataset

    def _converters(self) -> dict[str, Callable[[str, int | None], list[ContrastivePair]]]:
        return {
            "truthful_qa": self._from_truthful_qa,
            "Anthropic/hh-rlhf": self._from_hh_rlhf,
        }

    def _from_truthful_qa(self, goal: str, limit: int | None) -> list[ContrastivePair]:
        rows = self._resolve_loader()("truthful_qa", "multiple_choice", split="validation")
        pairs: list[ContrastivePair] = []
        for row in rows:
            targets = row["mc1_targets"]
            choices, labels = targets["choices"], targets["labels"]
            correct = [choice for choice, label in zip(choices, labels, strict=False) if label == 1]
            wrong = [choice for choice, label in zip(choices, labels, strict=False) if label == 0]
            if not correct or not wrong:
                continue
            pairs.append(
                ContrastivePair(
                    prompt=f"Q: {row['question']}\nA:",
                    positive_completion=" " + correct[0].strip(),
                    negative_completion=" " + wrong[0].strip(),
                    read_position=-1,
                    tier="hard",  # adversarial by construction: plausible falsehoods
                    source="truthful_qa",
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
        return pairs

    def _from_hh_rlhf(self, goal: str, limit: int | None) -> list[ContrastivePair]:
        rows = self._resolve_loader()(
            "Anthropic/hh-rlhf", data_dir="harmless-base", split="train", streaming=True
        )
        pairs: list[ContrastivePair] = []
        for row in rows:
            prompt, chosen, rejected = _split_shared_prefix(row["chosen"], row["rejected"])
            if not chosen.strip() or not rejected.strip() or not prompt.strip():
                continue
            pairs.append(
                ContrastivePair(
                    prompt=prompt,
                    positive_completion=chosen,
                    negative_completion=rejected,
                    read_position=-1,
                    tier="medium",
                    source="Anthropic/hh-rlhf",
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
        return pairs


def _split_shared_prefix(chosen: str, rejected: str) -> tuple[str, str, str]:
    """Shared prompt and the two divergent tails of a preference pair.

    hh-rlhf stores a whole conversation twice, identical up to the final assistant
    turn. The shared prefix is the prompt; the tails are the contrasting
    completions the behaviour is read from. Split at the last common character so
    both tails begin at the same point in the same turn.
    """
    limit = min(len(chosen), len(rejected))
    split = 0
    while split < limit and chosen[split] == rejected[split]:
        split += 1
    return chosen[:split], chosen[split:], rejected[split:]


def replace_read_position(pair: ContrastivePair, read_position: int) -> ContrastivePair:
    """A copy of ``pair`` reading at a different completion position."""
    return replace(pair, read_position=read_position)
