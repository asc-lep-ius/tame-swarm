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
"""

import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
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
