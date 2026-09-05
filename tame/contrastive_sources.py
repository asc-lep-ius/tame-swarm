"""Converters from published datasets to :class:`ContrastivePair` sets.

The datasets worth pulling are worth pulling *because* they are already A/B or
already labelled true/false, so the conversion is schema knowledge, not
authoring: TruthfulQA pairs a question with a correct answer and distractors,
``Anthropic/hh-rlhf`` a shared conversation with a chosen and a rejected
continuation, and the Geometry-of-Truth files a subject with a true and a false
statement. Each converter emits completion-format pairs; the caller's format
(letter or completion) is applied on top, so one converter serves both.

``datasets`` is imported lazily, so the ``serve`` extra never needs it, and every
converter accepts an injected ``load_dataset`` so it is testable without the
network. ``contrastive_data`` imports this module lazily for the same reason it
must not be imported here at module level: this module depends on that one.
"""

import logging
from collections.abc import Callable, Iterable

from contrastive_data import (
    COMPLETION_FORMAT,
    ContrastivePair,
    ContrastivePairSet,
    pair_set_in_format,
)

logger = logging.getLogger(__name__)

# Default cap on pairs drawn from a *streaming* HF source (hh-rlhf is ~42k rows),
# so a bare `load_contrastive_dataset(..., source="Anthropic/hh-rlhf")` yields an
# extraction-sized set rather than materialising the whole split. Pass an explicit
# `limit` to override. truthful_qa is small enough (817 rows) to leave uncapped.
DEFAULT_HF_STREAM_LIMIT = 500

# Geometry of Truth (Marks & Tegmark, 2023): curated true/false *declarative*
# statements -- "The city of Paris is in France." -- on which a truth direction is
# demonstrably linear and transfers across topics. Loaded as
# ``source="geometry_of_truth/<name>"`` (``cities``, ``neg_cities``,
# ``sp_en_trans``, ``larger_than``, ...) straight from the paper's repository via
# the ``datasets`` CSV builder, which caches the file like any HF dataset. The
# read is at the statement's final token (the full stop), *after* the content, as
# the paper reads it.
GEOMETRY_OF_TRUTH_SOURCE = "geometry_of_truth"
GEOMETRY_OF_TRUTH_URL = (
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/{name}.csv"
)
# Statement sets without a matched true/false partner per subject are paired
# true-with-false under this shared prefix, so the diff-in-means is the paper's
# unpaired mu(true) - mu(false) while the pair still has a non-empty prompt.
GOT_STATEMENT_PREFIX = "Statement:"


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
        self._cache: dict[tuple, ContrastivePairSet] = {}

    def load(
        self,
        goal: str,
        dataset: str,
        limit: int | None = None,
        pair_format: str = COMPLETION_FORMAT,
        seed: int = 0,
    ) -> ContrastivePairSet:
        key = (goal, dataset, limit, pair_format, seed)
        if key in self._cache:
            return self._cache[key]

        converter = self._converter_for(dataset)
        if converter is None:
            raise ValueError(
                f"no HuggingFace converter for dataset {dataset!r}; "
                f"known: {sorted(self._converters())} and {GEOMETRY_OF_TRUTH_SOURCE}/<name>"
            )
        pair_set = pair_set_in_format(goal, converter(goal, limit), dataset, pair_format, seed)
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

    def _converter_for(
        self, dataset: str
    ) -> Callable[[str, int | None], list[ContrastivePair]] | None:
        converter = self._converters().get(dataset)
        if converter is None and dataset.startswith(GEOMETRY_OF_TRUTH_SOURCE + "/"):
            name = dataset.split("/", 1)[1]
            return lambda goal, limit: self._from_geometry_of_truth(name, dataset, limit)
        return converter

    def _from_geometry_of_truth(
        self, name: str, source: str, limit: int | None
    ) -> list[ContrastivePair]:
        rows = list(
            self._resolve_loader()(
                "csv", data_files=GEOMETRY_OF_TRUTH_URL.format(name=name), split="train"
            )
        )
        if rows and "city" in rows[0]:
            pairs = _matched_statement_pairs(rows, subject_key="city", source=source)
        else:
            pairs = _unpaired_statement_pairs(rows, source=source)
        return pairs if limit is None else pairs[:limit]

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
        # harmless-base streams ~42k rows; without a cap this materialises the whole
        # split into ContrastivePair objects. Default to a sane extraction-sized cap
        # so an unbounded call from the public loader is not a foot-gun.
        limit = DEFAULT_HF_STREAM_LIMIT if limit is None else limit
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
    completions the behaviour is read from. The split is backed up to the last
    whitespace so each tail starts at a word boundary *with its leading space*:
    a prompt that ends in a space tokenises differently from the same prefix
    inside the full text, which would misalign the completion span the extractor
    and the log-odds metric slice against.
    """
    limit = min(len(chosen), len(rejected))
    split = 0
    while split < limit and chosen[split] == rejected[split]:
        split += 1
    boundary = chosen.rfind(" ", 0, split)
    if boundary > 0:
        split = boundary
    return chosen[:split], chosen[split:], rejected[split:]


def _matched_statement_pairs(
    rows: Iterable[dict], subject_key: str, source: str
) -> list[ContrastivePair]:
    """One pair per subject that has both a true and a false statement.

    ``cities.csv`` carries "The city of X is in <right>." (1) and "The city of X
    is in <wrong>." (0) for every city, so the shared prefix is the prompt and the
    two countries plus full stop are the completions, read at the stop.
    """
    by_subject: dict[str, dict[int, str]] = {}
    for row in rows:
        by_subject.setdefault(row[subject_key], {})[int(row["label"])] = row["statement"]
    pairs = []
    for statements in by_subject.values():
        if 1 not in statements or 0 not in statements:
            continue
        prompt, positive, negative = _split_shared_prefix(statements[1], statements[0])
        if not prompt.strip() or not positive.strip() or not negative.strip():
            continue
        pairs.append(
            ContrastivePair(
                prompt=prompt,
                positive_completion=positive,
                negative_completion=negative,
                read_position=-1,
                tier="medium",
                source=source,
            )
        )
    return pairs


def _unpaired_statement_pairs(rows: Iterable[dict], source: str) -> list[ContrastivePair]:
    """True statements zipped with false ones under a shared neutral prefix."""
    true_rows = [row["statement"] for row in rows if int(row["label"]) == 1]
    false_rows = [row["statement"] for row in rows if int(row["label"]) == 0]
    return [
        ContrastivePair(
            prompt=GOT_STATEMENT_PREFIX,
            positive_completion=f" {true_statement.strip()}",
            negative_completion=f" {false_statement.strip()}",
            read_position=-1,
            tier="medium",
            source=source,
        )
        for true_statement, false_statement in zip(true_rows, false_rows, strict=False)
    ]
