"""The held-out stream that rotates, the canaries hidden in it, and their schedules.

``evaluation`` owns the split that is **fixed** so that arms can be subtracted.
This module owns the artefact with the opposite property, and the two are
separate files because they fail differently and because one of them had run out
of room. A *viability margin* read on a corpus the organism has already trained
on is a margin the organism can farm -- #33's "cell that farms the field" one
scale up -- so the core of Phase 2 (#42) regulates against documents dated after
the checkpoint's data cutoff, drawn from a manifest that carries their dates,
refreshed on a schedule nothing in the system can move, and fingerprinted so that
two numbers read on two rotations cannot be compared as though they were one.

Interleaved into that stream at positions derived from the canary set's own
fingerprint are **canaries**: items with known answers that do not rotate. The
stream is what the organism cannot memorise and the canaries are what it can, so
farming reads as the two accuracies parting company -- which ``viability_margins``
computes, and this module only makes measurable.

Nothing here imports ``mob`` or any training code. That is deliberate rather than
incidental: ``parity`` records a ``RotationRecord`` on every arm, and a parity
check that had to import the model layer to name a rotation would be a config
module with a torch dependency on the far side of it.
"""

import hashlib
import json
import logging
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

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

    Flat fields rather than an object reference, because this is what survives into
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
    # How late the stream was when it was read, and the cutoff it was filtered
    # against. Neither is derivable from the fingerprint: a manifest read forty
    # days past its refresh hashes exactly as it did on the day it was fresh, so
    # without this a stale reading is indistinguishable downstream from a current
    # one. The cutoff is already implied -- filtering changes the item set and so
    # the fingerprint -- and is carried anyway because "after which date" is the
    # first question a reader of a margin asks and should not have to derive.
    stream_days_overdue: int | None = None
    stream_cutoff: str | None = None

    @classmethod
    def of(
        cls,
        stream: DatedManifest,
        canaries: DatedManifest,
        cutoff: date,
        days_overdue: int = 0,
    ) -> "RotationRecord":
        return cls(
            stream_fingerprint=stream.fingerprint,
            stream_refreshed=stream.refreshed.isoformat(),
            stream_refresh_days=stream.refresh_days,
            canary_fingerprint=canaries.fingerprint,
            canary_refreshed=canaries.refreshed.isoformat(),
            canary_refresh_days=canaries.refresh_days,
            stream_days_overdue=days_overdue,
            stream_cutoff=cutoff.isoformat(),
        )

    @property
    def label(self) -> str:
        """The date and the fingerprint, in the form every recorded number carries.

        A stale reading says so here rather than in a log line somebody has to go
        and find, because this string is what sits beside the number.
        """
        stale = f" STALE+{self.stream_days_overdue}d" if self.stream_days_overdue else ""
        return (
            f"stream={self.stream_fingerprint}@{self.stream_refreshed}{stale} "
            f"after={self.stream_cutoff} "
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

    Three refusals guard that reading, and two of them exist because the prefix
    assertion alone is not enough. Under a **left-padding** tokenizer the first
    ``length`` positions of both encodings are padding, so the prefix check compares
    pad against pad, passes, and the target becomes the pad token -- the vacuous
    version of the very check meant to prevent it. And an **answer that tokenises to
    nothing** -- a blank or whitespace-only manifest row, which is what a
    hand-assembled first manifest carries -- leaves the position after the prompt
    unattended, so the item is scored "predict pad from pad" and contributes a
    silently unanswerable row to the accuracy the core regulates on.
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
    full_mask = full_encoded["attention_mask"]
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
        if not bool(prompt_mask[row, :length].all()):
            raise ValueError(
                f"Item {item.item_id!r}: the tokenizer pads on the left, so the prompt's real "
                "tokens do not lead and the answer cannot be located by the prompt's length -- "
                "the prefix check below would compare padding against padding and pass. Use "
                "padding_side='right', as the trainer's tokenizer does"
            )
        if int(full_mask[row].sum().item()) <= length:
            raise ValueError(
                f"Item {item.item_id!r}: its answer adds no token to its prompt, so there is "
                "nothing to score. A blank answer is a row that cannot be answered, not a row "
                "that is always wrong"
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
    if overdue:
        # Said, and then carried: the warning is for whoever is watching the run,
        # and ``stream_days_overdue`` is for whoever reads the number afterwards.
        # An override that only logged would be an override that vanished.
        logger.warning(
            f"Rotating stream '{stream.name}' read {overdue} days past its refresh "
            f"(refreshed {stream.refreshed.isoformat()}, {stream.refresh_days}-day schedule, read "
            f"{today.isoformat()}) because allow_stale was passed. Every margin off this rotation "
            "is labelled STALE and is a margin on a corpus that stopped rotating"
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
        rotation=RotationRecord.of(fresh, canaries, cutoff=cutoff, days_overdue=overdue),
    )
    logger.info(
        f"Rotating stream: {built.num_items} items ({built.num_canaries} canaries), "
        f"published after {cutoff.isoformat()}, oldest {fresh.oldest.isoformat()}, "
        f"{built.rotation.label}"
    )
    return built
