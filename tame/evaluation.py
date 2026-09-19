"""A held-out split that is the same every run, and a loop that reads it for free.

Two defects #12 was opened over live here. The first is that ``eval_steps`` was
declared and never read, so nothing in the project was ever measured off the
training stream. The second is subtler and is why this module builds the split
rather than slicing one: the dataset is loaded with ``streaming=True``, which
gives no shuffle and no split, so the obvious "hold out the tail" produces a
number that is held out by index and not by content.

**Where the split comes from.** Preference order, and the two are not equivalent:

1. The dataset's own ``validation`` split. wikitext and c4 both ship one, and
   theirs is split by *article*, so a held-out document shares no sentences with a
   training document. Disjointness is then a property of the dataset, guaranteed
   upstream, rather than an argument about our own indexing.
2. Failing that, a fixed index-strided holdout of the training stream, with the
   training loader skipping exactly those indices. This is uniform across datasets
   and is what makes a splitless dataset usable at all, but it is *weaker*: strided
   lines are drawn from the same articles as their neighbours, so a model that has
   memorised the surrounding paragraph is credited for it. A number produced this
   way is disjoint in the index sense the acceptance criterion asks for and
   optimistic in the sense a reader cares about, which is why the split records
   which path produced it and every log line carries it.

**Why it is materialised and hashed.** Comparability across arms is the entire
point of #12: three arms whose held-out sets differ by so much as tokenisation
are three numbers that cannot be subtracted. The split is tokenised once, frozen
to disk, and fingerprinted; the parity check refuses to compare arms whose
fingerprints differ.

**The second held-out artefact (#41).** The split above is fixed so that arms can
be subtracted; a *viability margin* needs the opposite property. A margin read on
a corpus the organism has already trained on is a margin the organism can farm --
#33's "cell that farms the field" one scale up -- so the core of Phase 2 regulates
against a **rotating** stream instead: documents dated after the checkpoint's data
cutoff, drawn from a manifest that carries their dates, refreshed on a schedule
nothing in the system can move, and fingerprinted so that two numbers read on two
rotations cannot be compared as though they were one. Interleaved into it at
positions derived from the canary set's own fingerprint are **canaries** -- items
with known answers that do not rotate. The stream is what the organism cannot
memorise and the canaries are what it can, so farming reads as the two accuracies
parting company. What is computed from all of this lives in ``viability_margins``:
this module owns the held-out data, that one owns the metrics.
"""

import hashlib
import json
import logging
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch

from mob import frozen_economy

logger = logging.getLogger(__name__)

# Stride between held-out documents in the fallback path. Prime, so it cannot
# come into phase with any periodic structure in the source (wikitext's alternating
# heading/body/blank rows being the obvious one) and quietly hold out only headings.
HOLDOUT_STRIDE = 97

# Held-out sequences to materialise. At the default max_seq_length of 512 this is
# up to ~131k padded positions; the count that matters is ``num_tokens``, which
# excludes padding and stays comfortably above the >=4096-token probe floor the #12
# measurement note settled on, while remaining small enough to evaluate in-run.
DEFAULT_HELD_OUT_SEQUENCES = 256

SOURCE_VALIDATION_SPLIT = "validation-split"
SOURCE_TRAIN_HOLDOUT = "train-index-holdout"

# Matches the cap in the training loop's perplexity, so the two numbers are
# comparable at the top of the range instead of one saturating before the other.
MAX_LOG_PERPLEXITY = 20.0


def is_usable_document(text: object) -> bool:
    """Whether a raw dataset row counts as a document, for *both* consumers.

    The fallback holdout is defined by position in the stream of usable documents,
    so the eval builder and the training-side skip must agree on what a position
    is. wikitext is roughly one third blank rows; if one side counted them and the
    other did not, the indices would drift apart and the "held out" documents would
    be in the training set. One predicate, imported by both, is what makes the
    disjointness test meaningful rather than a test of a coincidence.
    """
    return isinstance(text, str) and bool(text.strip())


def is_held_out_position(position: int, stride: int = HOLDOUT_STRIDE) -> bool:
    """Whether a raw row index of the training stream is reserved for evaluation.

    Deliberately a predicate on the **raw** row index rather than on a position in
    some filtered stream. Both sides of the disjointness -- the collector below and
    the training loader's ``filter(..., with_indices=True)`` -- see the same raw
    index from the same stream, so "held out" and "skipped in training" are the
    same arithmetic on the same number. Any definition that counted only usable
    documents would need the two sides to filter identically before they could
    agree, which is a coincidence to be tested rather than an invariant.
    """
    return position % stride == 0


def fingerprint_tokens(input_ids: torch.Tensor) -> str:
    """A stable name for one materialised split.

    Over the token ids alone: two arms that tokenise the same documents with the
    same tokenizer and the same length agree here, and any difference in dataset,
    ordering, truncation or vocabulary shows up as a different string. Shape is
    folded in so that a reshape cannot collide with the tensor it came from.
    """
    digest = hashlib.sha256()
    digest.update(str(tuple(input_ids.shape)).encode())
    digest.update(input_ids.to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class HeldOutSplit:
    """Tokenised held-out data, fixed at construction and identical across arms."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    fingerprint: str
    source: str
    dataset: str

    @property
    def num_sequences(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.attention_mask.sum().item())

    @property
    def leakage_risk(self) -> str:
        """What disjointness this split actually provides, in one line.

        Carried on the object rather than left to the caller's memory, because the
        two sources differ in exactly the way a reader of the resulting perplexity
        needs to know about.
        """
        if self.source == SOURCE_VALIDATION_SPLIT:
            return "article-level disjoint (dataset's own split)"
        return "index-disjoint only; strided from train, may share articles"

    def batches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        """Fixed-order batches. No shuffling: the order is part of the artefact."""
        for start in range(0, self.num_sequences, batch_size):
            stop = start + batch_size
            yield {
                "input_ids": self.input_ids[start:stop],
                "attention_mask": self.attention_mask[start:stop],
            }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "fingerprint": self.fingerprint,
                "source": self.source,
                "dataset": self.dataset,
            },
            path,
        )
        logger.info(
            f"Held-out split frozen to {path} "
            f"({self.num_sequences} sequences, fingerprint {self.fingerprint})"
        )

    @classmethod
    def load(cls, path: str | Path) -> "HeldOutSplit":
        """Restore a frozen split, refusing one that is not what it says it is.

        The fingerprint is recomputed rather than trusted. A cache file is the one
        input to a comparison that lives outside the process and outlives the code
        that wrote it, so "the same split as last run" has to be checked once
        rather than assumed for every run that reads it afterwards.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        input_ids = payload["input_ids"]
        recomputed = fingerprint_tokens(input_ids)
        if recomputed != payload["fingerprint"]:
            raise ValueError(
                f"Held-out split at {path} does not match its fingerprint "
                f"(recorded {payload['fingerprint']}, recomputed {recomputed}); "
                "delete it and let the run rebuild it"
            )
        return cls(
            input_ids=input_ids,
            attention_mask=payload["attention_mask"],
            fingerprint=recomputed,
            source=payload["source"],
            dataset=payload["dataset"],
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[str],
        tokenizer: Callable[..., Any],
        max_seq_length: int,
        source: str,
        dataset: str,
    ) -> "HeldOutSplit":
        if not documents:
            raise ValueError(f"No held-out documents were collected for dataset '{dataset}'")

        encoded = tokenizer(
            documents,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fingerprint=fingerprint_tokens(input_ids),
            source=source,
            dataset=dataset,
        )


def collect_documents(
    rows: Iterable[dict[str, Any]],
    text_column: str,
    count: int,
    stride: int | None = None,
) -> list[str]:
    """Pull ``count`` usable documents out of a (possibly streaming) row iterable.

    With ``stride`` set this is the fallback holdout, and only rows at held-out raw
    positions are eligible. Blank rows -- about a third of wikitext -- are dropped
    *after* the position test, never before it, so dropping them cannot shift the
    positions the training loader is skipping.
    """
    documents: list[str] = []

    for position, row in enumerate(rows):
        if stride is not None and not is_held_out_position(position, stride):
            continue
        text = row.get(text_column)
        if not is_usable_document(text):
            continue
        documents.append(str(text))
        if len(documents) >= count:
            break

    if len(documents) < count:
        logger.warning(
            f"Collected {len(documents)} of {count} requested held-out documents; the source "
            "ran out of usable rows. Deterministic, so arms stay comparable, but the split is "
            "smaller than configured -- lower the stride or raise the source's row budget"
        )
    return documents


def build_held_out_split(
    dataset_name: str,
    dataset_config: str | None,
    tokenizer: Callable[..., Any],
    max_seq_length: int,
    load_dataset: Callable[..., Any],
    num_sequences: int = DEFAULT_HELD_OUT_SEQUENCES,
    text_column: str = "text",
) -> HeldOutSplit:
    """Build the split, preferring the dataset's own validation shard.

    ``load_dataset`` is injected rather than imported so this is testable without
    a network round trip; the trainer passes ``datasets.load_dataset``.
    """
    args = [dataset_name] if dataset_config is None else [dataset_name, dataset_config]

    try:
        validation = load_dataset(*args, split="validation")
    except Exception as error:  # noqa: BLE001 - datasets raises several unrelated types
        logger.warning(
            f"No usable 'validation' split for '{dataset_name}' ({type(error).__name__}: {error}). "
            f"Falling back to a stride-{HOLDOUT_STRIDE} holdout of the training stream. "
            "This is disjoint by index but may share articles with training data, so "
            "held-out perplexity from this path is optimistic -- prefer a dataset "
            "that ships a validation split for any published number."
        )
        train_stream = load_dataset(*args, split="train", streaming=True)
        documents = collect_documents(
            train_stream, text_column, num_sequences, stride=HOLDOUT_STRIDE
        )
        source = SOURCE_TRAIN_HOLDOUT
    else:
        documents = collect_documents(validation, text_column, num_sequences)
        source = SOURCE_VALIDATION_SPLIT

    split = HeldOutSplit.from_documents(
        documents,
        tokenizer,
        max_seq_length,
        source=source,
        dataset=f"{dataset_name}/{dataset_config}" if dataset_config else dataset_name,
    )
    logger.info(
        f"Held-out split: {split.num_sequences} sequences, {split.num_tokens} tokens, "
        f"source={split.source}, {split.leakage_risk}, fingerprint={split.fingerprint}"
    )
    return split


@dataclass(frozen=True)
class EvalResult:
    loss: float
    perplexity: float
    num_tokens: int
    num_batches: int
    fingerprint: str

    def as_metrics(self) -> dict[str, float]:
        """Named so nothing can be mistaken for the training-batch numbers.

        ``train.py`` reports ``perplexity`` from ``exp(main_loss)`` on the batch it
        just trained on. That is a training loss, and #12 exists partly because it
        was being read as though it were not, so these carry an ``eval/`` prefix
        and the training metrics carry ``train/``.
        """
        return {
            "eval/loss": self.loss,
            "eval/perplexity": self.perplexity,
            "eval/tokens": float(self.num_tokens),
        }


def _batch_loss(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[float, int]:
    """Loss summed over the scoreable tokens of one batch, and how many there were.

    Same shift, same mask, same ignored positions as ``train_step``: an eval loss not
    computed identically to the training loss is not comparable to it, and the gap
    between the two is the thing being read.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous() == 1

    per_token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)

    return float((per_token_loss * shift_mask).sum().item()), int(shift_mask.sum().item())


def evaluate(
    model: torch.nn.Module,
    split: HeldOutSplit,
    batch_size: int,
    device: torch.device,
) -> EvalResult:
    """Held-out loss and perplexity, with the economy frozen and nothing adapting.

    Token-weighted rather than batch-weighted: the last batch is usually short, and
    a mean of per-batch means would silently overweight its tokens. With
    ``padding="max_length"`` the difference is small, but it is free to be right and
    the alternative is a number whose definition depends on ``batch_size`` -- which
    the parity check would then have to pin for no reason.

    ``model.eval()`` and the frozen economy are restored on the way out, including
    when the forward raises, so an evaluation cannot leave the trainer in eval mode.
    """
    was_training = model.training
    total_loss = 0.0
    total_tokens = 0
    batches = 0

    model.eval()
    try:
        with torch.no_grad(), frozen_economy(model):
            for batch in split.batches(batch_size):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                loss, tokens = _batch_loss(model, input_ids, attention_mask)
                total_loss += loss
                total_tokens += tokens
                batches += 1
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Held-out split contained no scoreable tokens")

    mean_loss = total_loss / total_tokens
    return EvalResult(
        loss=mean_loss,
        perplexity=float(torch.exp(torch.tensor(min(mean_loss, MAX_LOG_PERPLEXITY)))),
        num_tokens=total_tokens,
        num_batches=batches,
        fingerprint=split.fingerprint,
    )


# --- The rotating held-out stream and its canaries (#41) ---------------------------------

# What a manifest file has to say it is. A manifest whose shape changed under the
# same filename is not the artefact the recorded fingerprint names, and a version
# that is checked at the boundary is the cheapest way to be told so.
MANIFEST_VERSION = 1

# The cadence a manifest gets when it does not name one. Here rather than at a
# call site on purpose: the whole claim of this stream is that the refresh is not
# the system's to choose, and a default a caller could pass would be a schedule a
# caller could move.
DEFAULT_REFRESH_DAYS = 30

STREAM_MANIFEST = "rotating-stream"
CANARY_MANIFEST = "canary-set"


class StaleStreamError(ValueError):
    """Raised when the rotating stream was not refreshed on the schedule it declares."""


@dataclass(frozen=True)
class DatedItem:
    """One manifest row: a prompt, the continuation it is scored against, and its date.

    ``published`` is the document's own date and not the date it entered the
    manifest. The stream's only claim is that the organism has not trained on this
    text, and only the former can support it: a manifest assembled today out of
    articles from two years ago is newer than the last update in no sense that
    matters.
    """

    item_id: str
    published: date
    prompt: str
    answer: str


def _manifest_digest(
    name: str, items: Sequence[DatedItem], refreshed: date, refresh_days: int
) -> str:
    """A stable name for one rotation, over the rows rather than over their tokens.

    ``fingerprint_tokens`` hashes a materialised split because two arms whose
    tokenisation differs must not be subtracted. This hashes the manifest itself,
    for a different reason: what a reader checks beside a margin is a *date* and a
    name they can look up, and an arm's tokenizer is already pinned by ``model_id``
    in the same fingerprint. The refresh schedule is folded in, so a manifest whose
    cadence was quietly widened is a different rotation and reads as one.
    """
    digest = hashlib.sha256()
    digest.update(f"{MANIFEST_VERSION}|{name}|{refreshed.isoformat()}|{refresh_days}".encode())
    for item in items:
        digest.update(
            f"\x00{item.item_id}\x01{item.published.isoformat()}\x01"
            f"{item.prompt}\x01{item.answer}".encode()
        )
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class DatedManifest:
    """A dated item set and the schedule it is refreshed on.

    Used twice, deliberately. Once for the **rotating stream**, whose items must be
    newer than the checkpoint's data cutoff and whose whole point is that they turn
    over. Once for the **canary set**, whose items are static by construction -- a
    canary that rotated could not show memorisation, which is the one thing it is
    there to show. Each carries its own ``refreshed`` date and its own
    ``refresh_days``, and each lands in ``ArmFingerprint`` under its own field,
    because two margins read on the same stream and different canaries are as
    incomparable as two read the other way round.
    """

    name: str
    items: tuple[DatedItem, ...]
    refreshed: date
    refresh_days: int = DEFAULT_REFRESH_DAYS

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError(f"Manifest '{self.name}' carries no items")
        if self.refresh_days <= 0:
            raise ValueError(
                f"Manifest '{self.name}' declares a refresh cadence of {self.refresh_days} days; "
                "a schedule that never comes due is not a schedule"
            )

    @property
    def fingerprint(self) -> str:
        return _manifest_digest(self.name, self.items, self.refreshed, self.refresh_days)

    @property
    def oldest(self) -> date:
        return min(item.published for item in self.items)

    def days_overdue(self, today: date) -> int:
        """Days past the refresh this manifest promised; zero while it is in date."""
        return max(0, (today - self.refreshed).days - self.refresh_days)

    def newer_than(self, cutoff: date) -> "DatedManifest":
        """The rows published strictly after ``cutoff``, which is the checkpoint's.

        Dropped rather than refused, and said loudly: a manifest ages into the
        training set one row at a time as checkpoints move forward, and a stream
        that refused outright on the first stale row would be unusable for exactly
        as long as it took somebody to delete it by hand. Empty *is* refused --
        a margin read on no items is not a small margin, it is no measurement.
        """
        fresh = tuple(item for item in self.items if item.published > cutoff)
        dropped = len(self.items) - len(fresh)
        if dropped:
            logger.warning(
                f"Manifest '{self.name}': {dropped} of {len(self.items)} items are dated on or "
                f"before the checkpoint cutoff {cutoff.isoformat()} and were dropped. They are "
                "not held out from a model trained to that cutoff -- refresh the manifest"
            )
        if not fresh:
            raise ValueError(
                f"Manifest '{self.name}' has no item published after the checkpoint cutoff "
                f"{cutoff.isoformat()}; there is no stream newer than the last update to read"
            )
        return DatedManifest(
            name=self.name, items=fresh, refreshed=self.refreshed, refresh_days=self.refresh_days
        )


def load_manifest(path: str | Path, name: str | None = None) -> DatedManifest:
    """Read a manifest file, validating its shape before anything reads its rows."""
    payload = json.loads(Path(path).read_text())
    version = payload.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"Manifest at {path} declares version {version!r}, not {MANIFEST_VERSION}; "
            "the fingerprint of a manifest read under the wrong schema names nothing"
        )
    rows = payload.get("items")
    if not isinstance(rows, list):
        raise ValueError(f"Manifest at {path} has no 'items' list")

    items = []
    for index, row in enumerate(rows):
        missing = sorted({"item_id", "published", "prompt", "answer"} - set(row))
        if missing:
            raise ValueError(f"Manifest at {path}, item {index}: missing keys {missing}")
        items.append(
            DatedItem(
                item_id=str(row["item_id"]),
                published=date.fromisoformat(str(row["published"])),
                prompt=str(row["prompt"]),
                answer=str(row["answer"]),
            )
        )

    return DatedManifest(
        name=name or str(payload.get("name", Path(path).stem)),
        items=tuple(items),
        refreshed=date.fromisoformat(str(payload["refreshed"])),
        refresh_days=int(payload.get("refresh_days", DEFAULT_REFRESH_DAYS)),
    )


@dataclass(frozen=True)
class RotationRecord:
    """What an arm records about the rotation its margins were read on.

    Six strings rather than an object reference, because this is what survives into
    a summary on disk and is read back a month later by ``parity.manifest_drift``.
    All default to ``None``: every run recorded before #41 read no rotating stream
    at all, and a *missing* rotation has to count as drift rather than as agreement,
    which is the same rule -- and the same reason -- as ``code_sha``'s.
    """

    stream_fingerprint: str | None = None
    stream_refreshed: str | None = None
    stream_refresh_days: int | None = None
    canary_fingerprint: str | None = None
    canary_refreshed: str | None = None
    canary_refresh_days: int | None = None

    @classmethod
    def of(cls, stream: DatedManifest, canaries: DatedManifest) -> "RotationRecord":
        return cls(
            stream_fingerprint=stream.fingerprint,
            stream_refreshed=stream.refreshed.isoformat(),
            stream_refresh_days=stream.refresh_days,
            canary_fingerprint=canaries.fingerprint,
            canary_refreshed=canaries.refreshed.isoformat(),
            canary_refresh_days=canaries.refresh_days,
        )

    @property
    def label(self) -> str:
        """The date and the fingerprint, in the form every recorded number carries."""
        return (
            f"stream={self.stream_fingerprint}@{self.stream_refreshed} "
            f"canaries={self.canary_fingerprint}@{self.canary_refreshed}"
        )


ROTATION_ABSENT = RotationRecord()


@dataclass(frozen=True)
class RotatingStream:
    """The tokenised rotation: prompts the organism sees, answers it does not.

    ``input_ids`` carries the prompt alone -- the answer's tokens are never attended
    and never scored -- and ``answer_positions`` is the index whose logits predict
    ``answer_ids``. That is one prediction per item rather than one per token, which
    is what the margins below are defined over: a per-token accuracy on a document
    is dominated by the tokens no one is asking about.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    answer_ids: torch.Tensor
    answer_positions: torch.Tensor
    is_canary: torch.Tensor
    item_ids: tuple[str, ...]
    rotation: RotationRecord

    @property
    def num_items(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_canaries(self) -> int:
        return int(self.is_canary.sum().item())

    def batches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        """Fixed-order batches. The order is the rotation's, not the reader's."""
        for start in range(0, self.num_items, batch_size):
            stop = start + batch_size
            yield {
                "input_ids": self.input_ids[start:stop],
                "attention_mask": self.attention_mask[start:stop],
                "answer_ids": self.answer_ids[start:stop],
                "answer_positions": self.answer_positions[start:stop],
            }


def place_canaries(
    stream_items: Sequence[DatedItem], canaries: DatedManifest
) -> tuple[list[DatedItem], list[bool]]:
    """Interleave the canaries at positions nothing in the system chooses.

    The seed is the canary manifest's own fingerprint. That is the point rather
    than a convenience: a placement seeded from a config is a placement somebody
    can hold fixed, and a canary whose position is fixed and knowable is an item
    the organism can learn to treat differently from the stream around it. Derived
    this way the positions turn over exactly when the canary set does, and no
    caller can ask for a particular arrangement without changing the canaries --
    which is recorded, fingerprinted and compared.
    """
    rng = random.Random(int(canaries.fingerprint, 16))
    items = list(stream_items)
    flags = [False] * len(items)
    for canary in canaries.items:
        position = rng.randrange(len(items) + 1)
        items.insert(position, canary)
        flags.insert(position, True)
    return items, flags


def _encode_items(
    items: Sequence[DatedItem], tokenizer: Callable[..., Any], max_seq_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise prompts, and find each answer's first token by extending the prompt.

    The answer is tokenised *in place* -- ``prompt + answer`` encoded once and read
    at the prompt's length -- rather than on its own. Encoding an answer alone gives
    a different first token whenever the tokenizer prepends a sentence marker or
    merges across the boundary, and a target token that is silently the wrong one
    produces an accuracy that is wrong by a constant nobody can see. The prefix
    property is asserted rather than assumed, so a tokenizer that does merge across
    the boundary refuses here instead of being measured.
    """
    prompts = [item.prompt for item in items]
    continued = [item.prompt + item.answer for item in items]
    encode = {
        "truncation": True,
        "max_length": max_seq_length,
        "padding": "max_length",
        "return_tensors": "pt",
    }
    prompt_encoded = tokenizer(prompts, **encode)
    full_encoded = tokenizer(continued, **encode)

    prompt_ids = prompt_encoded["input_ids"]
    prompt_mask = prompt_encoded["attention_mask"]
    full_ids = full_encoded["input_ids"]
    lengths = prompt_mask.sum(dim=1)

    answer_ids = torch.zeros(len(items), dtype=torch.long)
    for row, item in enumerate(items):
        length = int(lengths[row].item())
        if length < 1:
            raise ValueError(f"Item {item.item_id!r} tokenises to an empty prompt")
        if length >= max_seq_length:
            raise ValueError(
                f"Item {item.item_id!r} fills the whole {max_seq_length}-token window with its "
                "prompt, leaving no position for its answer"
            )
        if not bool(torch.equal(full_ids[row, :length], prompt_ids[row, :length])):
            raise ValueError(
                f"Item {item.item_id!r}: the tokenizer does not encode its prompt as a prefix of "
                "prompt+answer, so the answer's first token cannot be located. Start the answer "
                "with a space, or use a tokenizer that does not merge across the boundary"
            )
        answer_ids[row] = full_ids[row, length]

    return prompt_ids, prompt_mask, answer_ids, lengths - 1


def build_rotating_stream(
    stream: DatedManifest,
    canaries: DatedManifest,
    tokenizer: Callable[..., Any],
    max_seq_length: int,
    cutoff: date,
    today: date | None = None,
    allow_stale: bool = False,
) -> RotatingStream:
    """The rotation as the core reads it: fresh documents, canaries hidden among them.

    Two refusals, and neither is a formality. Items dated on or before ``cutoff``
    are dropped, because they are not held out from a checkpoint trained to that
    date. A manifest past its own refresh date is refused outright unless
    ``allow_stale``: a stream that stopped rotating is a fixed corpus wearing a
    rotating name, and every day it is late is a day the organism had to catch up
    with it. ``today`` is injected so that the refusal is a property of the
    manifest and not of the clock the test happens to run on.
    """
    today = today or date.today()
    overdue = stream.days_overdue(today)
    if overdue and not allow_stale:
        raise StaleStreamError(
            f"Rotating stream '{stream.name}' was refreshed {stream.refreshed.isoformat()} on a "
            f"{stream.refresh_days}-day schedule and is {overdue} days overdue at "
            f"{today.isoformat()}. The schedule is not the system's to move: refresh the manifest, "
            "or pass allow_stale to read a margin that says so"
        )

    fresh = stream.newer_than(cutoff)
    items, flags = place_canaries(fresh.items, canaries)
    input_ids, attention_mask, answer_ids, answer_positions = _encode_items(
        items, tokenizer, max_seq_length
    )
    built = RotatingStream(
        input_ids=input_ids,
        attention_mask=attention_mask,
        answer_ids=answer_ids,
        answer_positions=answer_positions,
        is_canary=torch.tensor(flags, dtype=torch.bool),
        item_ids=tuple(item.item_id for item in items),
        rotation=RotationRecord.of(fresh, canaries),
    )
    logger.info(
        f"Rotating stream: {built.num_items} items ({built.num_canaries} canaries), "
        f"published after {cutoff.isoformat()}, oldest {fresh.oldest.isoformat()}, "
        f"{built.rotation.label}"
    )
    return built
