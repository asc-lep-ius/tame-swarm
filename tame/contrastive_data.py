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
extractor reads. It comes in two formats (#17). In the *completion* format the
arms are the answers themselves, read at the answer token; that read also carries
what the answer is *about*, which over heterogeneous facts swamps the behaviour
and is why the #3 ``truthful`` vector steered toward falsehood. In the
*multiple-choice* format -- CAA's actual one -- both answers sit in the prompt as
``(A)``/``(B)`` options, each arm is a single letter, and the read is at the
letter: the moment of commitment, identical content, only the choice differs.
Each goal has a certified (source, format) in :data:`CERTIFIED`; the gate in
``scripts/validate_steering.py`` is what puts an entry there.

The instruction-prefix templates are retained (see
:data:`INSTRUCTION_PREFIX_CONTROL`) not because they are useful for steering but
because they are the artefact this design is trying to avoid extracting, which
makes them the negative control the behavioural validation measures against.

This module is the data and validation half; the converters for published
datasets live in ``contrastive_sources.py``, the extraction that reads these
pairs at their recorded positions in ``steering.py``, and the pipeline that ties
loading, extraction and normalisation together in ``steering_pipeline.py``.
"""

import logging
import random
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
# How a pair presents its contrast to the model. ``completion`` is the #3 shape:
# the two arms are the answers themselves, read at the answer token. That reads
# *what the answer is about* along with whether it is right, and for
# heterogeneous facts the content dominates the diff-in-means -- the failure #17
# diagnosed. ``multiple_choice`` is CAA's actual format (Rimsky et al., 2024):
# both answers sit in the shared prompt as ``(A)``/``(B)`` options, each arm is a
# single letter, and the read is at the letter -- the moment of commitment, where
# the content is identical between arms and only the choice differs.
COMPLETION_FORMAT = "completion"
MULTIPLE_CHOICE_FORMAT = "multiple_choice"
PAIR_FORMATS = frozenset({COMPLETION_FORMAT, MULTIPLE_CHOICE_FORMAT})
MC_LETTERS: tuple[str, str] = ("A", "B")
# The role markers the built-in templates and HF converters end a prompt with;
# multiple-choice conversion splits the question from the answer stem here.
_ROLE_MARKERS = ("A:", "Assistant:")
MC_ANSWER_CUE = "Answer:"
# A letter assignment further from balance than this many pairs leaks the bare
# "A minus B" token direction into the diff-in-means. Conversion balances exactly;
# the tolerance exists for sets truncated after conversion.
MAX_LETTER_IMBALANCE = 1

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
    pair_format: str = COMPLETION_FORMAT
    # Multiple-choice only: the letter the *positive* option was placed under. It
    # is what makes the randomisation auditable -- a set's letter counts must be
    # balanced, or the diff-in-means carries the bare "A minus B" token direction.
    correct_letter: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not self.positive_completion.strip() or not self.negative_completion.strip():
            raise ValueError("both completions must be non-empty")
        if self.tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {sorted(VALID_TIERS)}, got {self.tier!r}")
        if self.pair_format not in PAIR_FORMATS:
            raise ValueError(f"pair_format must be one of {sorted(PAIR_FORMATS)}")
        if self.pair_format == MULTIPLE_CHOICE_FORMAT:
            self._check_multiple_choice()
        elif self.correct_letter is not None:
            raise ValueError("correct_letter is only meaningful for multiple_choice pairs")

    def _check_multiple_choice(self) -> None:
        if self.correct_letter not in MC_LETTERS:
            raise ValueError(f"multiple_choice pair needs correct_letter in {MC_LETTERS}")
        expected_wrong = _other_letter(self.correct_letter)
        if self.positive_completion.strip() != self.correct_letter:
            raise ValueError("multiple_choice positive completion must be the correct letter")
        if self.negative_completion.strip() != expected_wrong:
            raise ValueError("multiple_choice negative completion must be the other letter")

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


def _other_letter(letter: str) -> str:
    return MC_LETTERS[1] if letter == MC_LETTERS[0] else MC_LETTERS[0]


def _split_role_prompt(prompt: str) -> tuple[str, str]:
    """(question block, answer stem) of a completion-format prompt.

    The templates and QA converters end every prompt with a role line -- ``A:`` or
    ``Assistant:`` -- optionally followed by the start of the answer ("A: It is
    located in"). The stem belongs to the option text, not the question, so each
    multiple-choice option reads as a complete answer. A prompt with no role line
    (a declarative statement prefix) is all question and has no stem.
    """
    question, _, last_line = prompt.rstrip().rpartition("\n")
    for marker in _ROLE_MARKERS:
        if last_line.startswith(marker):
            return question, last_line[len(marker) :].strip()
    return prompt.rstrip(), ""


def _balanced_letters(count: int, spare: str, rng: random.Random) -> list[str]:
    """Half A, half B, an odd count's spare going to ``spare``, in seeded random order."""
    letters = list(MC_LETTERS) * (count // 2) + ([spare] if count % 2 else [])
    rng.shuffle(letters)
    return letters


def _as_multiple_choice(pair: ContrastivePair, correct_letter: str) -> ContrastivePair:
    question, stem = _split_role_prompt(pair.prompt)
    positive = f"{stem} {pair.positive_completion.strip()}".strip()
    negative = f"{stem} {pair.negative_completion.strip()}".strip()
    option_a, option_b = (positive, negative) if correct_letter == "A" else (negative, positive)
    prompt = f"{question}\n(A) {option_a}\n(B) {option_b}\n{MC_ANSWER_CUE}"
    return replace(
        pair,
        prompt=prompt,
        positive_completion=f" {correct_letter}",
        negative_completion=f" {_other_letter(correct_letter)}",
        read_position=-1,
        pair_format=MULTIPLE_CHOICE_FORMAT,
        correct_letter=correct_letter,
    )


def to_multiple_choice(pairs: Iterable[ContrastivePair], seed: int = 0) -> list[ContrastivePair]:
    """Re-express completion pairs in CAA's ``(A)``/``(B)`` letter format.

    The two answers move into the shared prompt as options and each arm becomes a
    single letter, read at that letter (``read_position=-1``). Which letter carries
    the correct option is assigned by a seeded shuffle that is *exactly balanced
    within each tier*, so averaging ``h(correct letter) - h(wrong letter)`` over the
    set cancels the letter-identity component and leaves the choice direction.

    Balance is a property of the set being averaged, so convert the extraction set
    and the held-out set separately rather than converting once and splitting, and
    check :func:`letter_imbalance` on whatever is finally averaged. Pairs already
    in this format pass through unchanged.
    """
    rng = random.Random(seed)
    materialised = list(pairs)
    by_tier: dict[str, list[int]] = {}
    for index, pair in enumerate(materialised):
        if pair.pair_format == COMPLETION_FORMAT:
            by_tier.setdefault(pair.tier, []).append(index)

    # Each odd tier leaves one spare letter; alternate which letter takes it across
    # tiers, in a fixed tier order, so the whole set is also within one of balance
    # rather than accumulating one spare A per tier.
    converted = list(materialised)
    spares = iter(MC_LETTERS * ((len(by_tier) + 1) // 2 + 1))
    for tier in sorted(by_tier, key=lambda name: TIERS.index(name) if name in TIERS else 99):
        indices = by_tier[tier]
        spare = next(spares) if len(indices) % 2 else MC_LETTERS[0]
        for index, letter in zip(indices, _balanced_letters(len(indices), spare, rng), strict=True):
            converted[index] = _as_multiple_choice(materialised[index], letter)
    return converted


def letter_counts(pairs: Iterable[ContrastivePair]) -> dict[str, int]:
    """How many multiple-choice pairs place the correct option under each letter."""
    counts = dict.fromkeys(MC_LETTERS, 0)
    for pair in pairs:
        if pair.correct_letter is not None:
            counts[pair.correct_letter] += 1
    return counts


def letter_imbalance(pairs: Iterable[ContrastivePair]) -> int:
    counts = letter_counts(pairs)
    return abs(counts["A"] - counts["B"])


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
    letter_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def meets_count(self) -> bool:
        return self.pair_count >= MIN_PAIRS_PER_GOAL

    @property
    def letters_balanced(self) -> bool:
        """Vacuously true for completion-format sets, which have no letters."""
        if not self.letter_counts:
            return True
        return abs(self.letter_counts["A"] - self.letter_counts["B"]) <= MAX_LETTER_IMBALANCE

    @property
    def meets_tier_coverage(self) -> bool:
        return all(self.tier_counts.get(tier, 0) >= MIN_PAIRS_PER_TIER for tier in VALID_TIERS)

    @property
    def ok(self) -> bool:
        return (
            self.meets_count
            and self.meets_tier_coverage
            and not self.duplicate_pairs
            and self.letters_balanced
        )


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

    @property
    def is_multiple_choice(self) -> bool:
        return any(pair.pair_format == MULTIPLE_CHOICE_FORMAT for pair in self.pairs)

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

        letters = letter_counts(self.pairs) if self.is_multiple_choice else {}
        if letters and abs(letters["A"] - letters["B"]) > MAX_LETTER_IMBALANCE:
            warnings.append(
                f"correct-letter counts {letters} are unbalanced: the diff-in-means would "
                "carry the bare A-minus-B token direction"
            )

        return QualityReport(
            goal=self.goal,
            pair_count=len(self.pairs),
            tier_counts=counts,
            duplicate_pairs=duplicate_pairs,
            lexical_near_duplicates=lexical_near_duplicates,
            high_similarity_pairs=high_similarity_pairs,
            letter_counts=letters,
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


_REQUIRED_RECORD_KEYS = ("prompt", "positive", "negative")


def _pairs_from_records(records: Iterable[dict], source: str) -> list[ContrastivePair]:
    pairs = []
    for record in records:
        missing = [key for key in _REQUIRED_RECORD_KEYS if key not in record]
        if missing:
            raise ValueError(f"contrastive pair record is missing keys {missing}: {record!r}")
        pairs.append(
            ContrastivePair(
                prompt=record["prompt"],
                positive_completion=record["positive"],
                negative_completion=record["negative"],
                read_position=record.get("read_position", -1),
                tier=record.get("tier", "medium"),
                source=source,
            )
        )
    return pairs


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


@dataclass(frozen=True)
class Certification:
    """What the behavioural gate certified for a goal: the pairs, and where they act.

    ``layers`` and ``strength`` are the injection the gate was measured at;
    ``strength_band`` is the range of strengths that still pass it, when that was
    swept (#4), and ``readout_layer`` is where the homeostat's sensor reads. A goal
    without a measured band is served at its certified strength, held constant:
    the loop may only move within a band the gate has actually passed.
    """

    source: str
    pair_format: str
    layers: tuple[int, ...] | None = None
    strength: float | None = None
    strength_band: tuple[float, float] | None = None
    readout_layer: int | None = None
    model: str | None = None


# What each goal is extracted from by default: the (source, format) the gate in
# ``scripts/validate_steering.py`` certified on the reference model, with the
# measurements in the README "Steering validation" section. ``safe`` keeps the
# completion format because that is its certified shape; moving a passing goal to
# a new format would spend the control the other goals are measured against.
# ``truthful`` is certified on TruthfulQA, not the built-in facts: the built-in
# held-out is saturated (the model prefers the true option by ~6 nats already), so
# no direction can be told from noise there, while TruthfulQA's misconceptions are
# adversarial by construction and 817 rows deep.
BUILTIN_SOURCE = "builtin"
# #17 measured every goal at layers 14/18/22, strength 4.0. #4 swept ``truthful``
# layer by layer on Qwen3-1.7B (``scripts/sweep_steering_layers.py``): the direction
# steers the *wrong* way below layer 13,
# reads prompt wording at 12 (the prefix control outscores it), passes alone at 13
# and 16-21, and is null from 22 up. As a set, 13 + 16-21 passes the gate at every
# strength from 2 to 8 with the prefix control below zero; the band stops at 6
# because the log-probability drift on natural continuations doubles again at 8.
# Dropping 13 halves the effect; adding 24 changes nothing.
TRUTHFUL_LAYERS = (13, 16, 17, 18, 19, 20, 21)
CERTIFIED_MODEL = "Qwen/Qwen3-1.7B"
CERTIFIED: dict[str, Certification] = {
    "truthful": Certification(
        "truthful_qa",
        MULTIPLE_CHOICE_FORMAT,
        layers=TRUTHFUL_LAYERS,
        strength=4.0,
        strength_band=(2.0, 6.0),
        readout_layer=22,
        model=CERTIFIED_MODEL,
    ),
    "reasoning": Certification(
        BUILTIN_SOURCE,
        MULTIPLE_CHOICE_FORMAT,
        layers=(14, 18, 22),
        strength=4.0,
        model=CERTIFIED_MODEL,
    ),
    "safe": Certification(
        BUILTIN_SOURCE, COMPLETION_FORMAT, layers=(14, 18, 22), strength=4.0, model=CERTIFIED_MODEL
    ),
}
_UNCERTIFIED = Certification(BUILTIN_SOURCE, COMPLETION_FORMAT)


def certification_for(goal: str) -> Certification | None:
    """The certified (source, format) for ``goal``, or ``None`` if the gate never passed it."""
    return CERTIFIED.get(goal)


def resolve_pair_format(goal: str, pair_format: str | None = None) -> str:
    """The explicit format if given, else the goal's certified default."""
    resolved = pair_format or (certification_for(goal) or _UNCERTIFIED).pair_format
    if resolved not in PAIR_FORMATS:
        raise ValueError(f"pair_format must be one of {sorted(PAIR_FORMATS)}, got {resolved!r}")
    return resolved


def pair_set_in_format(
    goal: str, pairs: Iterable[ContrastivePair], source: str, pair_format: str, seed: int
) -> ContrastivePairSet:
    materialised = list(pairs)
    if pair_format == MULTIPLE_CHOICE_FORMAT:
        materialised = to_multiple_choice(materialised, seed=seed)
    return ContrastivePairSet.from_pairs(goal, materialised, source=source)


def load_contrastive_dataset(
    goal: str,
    source: str = "builtin",
    load_dataset: Callable[..., Iterable[dict]] | None = None,
    limit: int | None = None,
    pair_format: str | None = None,
    seed: int = 0,
) -> ContrastivePairSet:
    """Load one goal's pairs from the built-in templates, custom registrations, or HF.

    ``source`` is ``"builtin"`` (the hand-authored templates), ``"custom"`` (pairs
    registered via :func:`register_contrastive_pairs`), or a HuggingFace dataset
    name understood by :class:`contrastive_sources.HFContrastiveLoader`. The HF path imports
    ``datasets`` lazily and accepts an injected ``load_dataset`` so it is testable
    without the network. ``limit`` caps how many pairs an HF source yields; it is
    ignored for the built-in and custom sources, which are already bounded.

    ``pair_format`` selects the shape the pairs are presented in (see
    :data:`PAIR_FORMATS`); ``None`` takes the goal's certified default. Every
    source is authored in the completion format and converted on load, so one
    content table serves both formats. ``seed`` fixes the letter randomisation.
    """
    resolved = resolve_pair_format(goal, pair_format)
    if source == "builtin":
        if goal not in BUILTIN_PAIRS:
            raise ValueError(f"unknown goal {goal!r}; built-in goals: {available_goals()}")
        pairs = _pairs_from_records(BUILTIN_PAIRS[goal], "builtin")
        return pair_set_in_format(goal, pairs, "builtin", resolved, seed)
    if source == "custom":
        if goal not in _CUSTOM_PAIRS:
            raise ValueError(f"no custom pairs registered for goal {goal!r}")
        return pair_set_in_format(goal, _CUSTOM_PAIRS[goal], "custom", resolved, seed)

    # Imported here, not at module level: the sources module depends on this one.
    from contrastive_sources import HFContrastiveLoader, default_loader  # noqa: PLC0415

    loader = default_loader() if load_dataset is None else HFContrastiveLoader(load_dataset)
    return loader.load(goal, source, limit=limit, pair_format=resolved, seed=seed)


@dataclass(frozen=True)
class CertifiedLoad:
    """A goal's pairs from its certified source, or the fallback with the reason."""

    pair_set: ContrastivePairSet
    certified: bool
    fallback_reason: str | None = None


def load_certified_dataset(
    goal: str,
    load_dataset: Callable[..., Iterable[dict]] | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> CertifiedLoad:
    """The goal's certified (source, format), falling back to the built-in set.

    The certified source may need the ``train`` extra (``datasets``) and a warm HF
    cache; the ``serve`` image has neither. Rather than fail to start, the loader
    drops to the built-in templates in the certified *format* and says so: the
    vector the server then steers on is uncertified, and the caller is expected to
    log that rather than present it as the measured one. Only the environmental
    failures fall back -- a converter bug (``ValueError``) still raises.
    """
    certification = certification_for(goal)
    if certification is None:
        reason = f"goal {goal!r} has no certified (source, format); using built-in pairs"
        logger.warning("Goal %r: %s -- the extracted vector is UNCERTIFIED", goal, reason)
        pair_set = load_contrastive_dataset(
            goal, source=BUILTIN_SOURCE, pair_format=_UNCERTIFIED.pair_format, seed=seed
        )
        return CertifiedLoad(pair_set=pair_set, certified=False, fallback_reason=reason)
    try:
        pair_set = load_contrastive_dataset(
            goal,
            source=certification.source,
            load_dataset=load_dataset,
            limit=limit,
            pair_format=certification.pair_format,
            seed=seed,
        )
    except (RuntimeError, OSError, *_dataset_build_errors()) as exc:
        if certification.source == BUILTIN_SOURCE:
            raise
        reason = f"{certification.source} unavailable ({exc}); using built-in pairs"
        logger.warning("Goal %r: %s -- the extracted vector is UNCERTIFIED", goal, reason)
        fallback = load_contrastive_dataset(
            goal, source=BUILTIN_SOURCE, pair_format=certification.pair_format, seed=seed
        )
        return CertifiedLoad(pair_set=fallback, certified=False, fallback_reason=reason)
    return CertifiedLoad(pair_set=pair_set, certified=True)


def _dataset_build_errors() -> tuple[type[Exception], ...]:
    """``datasets``' own error base, if the package is installed.

    ``DatasetNotFoundError`` is an ``OSError``, but a failed build or parquet
    conversion raises ``DatasetGenerationError``, which is not; both are
    environmental and both must fall back rather than leave the server unsteered.
    Resolved lazily so the ``serve`` extra never imports ``datasets``.
    """
    try:
        from datasets.exceptions import DatasetsError  # noqa: PLC0415
    except ImportError:
        return ()
    return (DatasetsError,)


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


def certified_source(goal: str) -> str:
    """The source a goal is certified on, or the built-in templates."""
    certification = certification_for(goal)
    return certification.source if certification else BUILTIN_SOURCE


def interleaved_split(
    pairs: Sequence[ContrastivePair], held_out: int
) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
    """Every k-th pair held out so topics interleave; the rest extract.

    The split the gate certifies on. Every script that measures against the
    certified held-out set takes it from here, so a change to the split cannot
    silently desynchronise a measurement from the certification.
    """
    k = max(2, len(pairs) // max(1, held_out))
    kept = [pair for index, pair in enumerate(pairs) if index % k == 0][:held_out]
    rest = [pair for index, pair in enumerate(pairs) if index % k != 0]
    return rest, kept


def replace_read_position(pair: ContrastivePair, read_position: int) -> ContrastivePair:
    """A copy of ``pair`` reading at a different completion position."""
    return replace(pair, read_position=read_position)
