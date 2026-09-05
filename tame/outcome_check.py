"""Does the vector change what the model *generates*, not only the next token?

The behavioural gate (``behavioural_validation``) reads one teacher-forced token:
does the model now prefer the positive completion? For a goal like deliberation
that is necessary but not sufficient -- reasoning is a generation-length
behaviour, and a direction can shift the A/B letter without changing a single
free-running token. This module is the lightweight outcome check #17 asks for
alongside the log-odds shift: greedy-decode the held-out questions with and
without the vector, and report two outcome deltas.

- **Length** -- mean generated tokens before the model stops or starts a new
  question. The deliberation proxy predicts more of them; a "prefer the longer
  option" artefact would too, which is why accuracy is reported beside it.
- **Accuracy** -- the fraction of generations that contain the reference answer.
  On a small base model over a handful of questions this is coarse, and it is
  reported, not gated: it is the falsifier for "more tokens, no more reasoning".

Greedy decoding is deterministic, so the check runs without a judge or sampling.
The decode loop is written out rather than delegated to ``generate`` so the fake
models in the tests and the real one take the same path through the same hooks.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from behavioural_validation import attach_steering_hooks
from contrastive_data import ContrastivePair
from steering import SteeringConfig

logger = logging.getLogger(__name__)

# A raw (non-chat) model continuing a "Q: ...\nA:" prompt tends to invent the next
# question; generation is counted up to that point, or to end-of-text.
DEFAULT_STOP_STRINGS = ("\nQ:", "\n\n")
DEFAULT_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class OutcomeCheck:
    """Length and accuracy of greedy generations, unsteered versus steered."""

    label: str
    num_questions: int
    baseline_length: float
    steered_length: float
    baseline_accuracy: float
    steered_accuracy: float

    @property
    def length_delta(self) -> float:
        return self.steered_length - self.baseline_length

    @property
    def accuracy_delta(self) -> float:
        return self.steered_accuracy - self.baseline_accuracy


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def contains_answer(generation: str, answer: str) -> bool:
    """Whole-word match of the normalised reference answer inside the generation."""
    needle = _normalise(answer)
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, _normalise(generation)) is not None


def _eos_ids(tokenizer) -> set[int]:
    """End-of-text ids; ``eos_token_id`` may be an int or a list (Llama-3 style)."""
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {int(item) for item in value}
    return set()


def greedy_continue(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
    max_length: int = 256,
) -> tuple[str, int]:
    """Greedy continuation of ``prompt`` and the number of tokens it took.

    Stops at end-of-text, at the first stop string, or at ``max_new_tokens``. The
    returned text excludes the stop string; the count is the tokens generated
    before stopping. No KV cache: the whole sequence is re-run each step, which is
    fine for the tens of questions this is meant for and keeps the fakes simple.
    """
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)[
        "input_ids"
    ].to(device)
    eos = _eos_ids(tokenizer)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=input_ids).logits
            next_id = int(logits[0, -1].argmax().item())
            if next_id in eos:
                break
            generated.append(next_id)
            text = tokenizer.decode(generated)
            cut = _first_stop(text, stop_strings)
            if cut is not None:
                return text[:cut], _token_count(tokenizer, text[:cut], len(generated))
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(generated), len(generated)


def _first_stop(text: str, stop_strings: Sequence[str]) -> int | None:
    positions = [text.find(stop) for stop in stop_strings if stop in text]
    return min(positions) if positions else None


def _token_count(tokenizer, text: str, upper_bound: int) -> int:
    """Tokens in the kept text; a stop string emitted first leaves none."""
    if not text:
        return 0
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return min(int(ids.shape[1]), upper_bound)


def _run(
    model: nn.Module,
    tokenizer,
    pairs: Sequence[ContrastivePair],
    device: torch.device,
    max_new_tokens: int,
    stop_strings: Sequence[str],
) -> tuple[float, float, list[str]]:
    lengths, hits, texts = [], 0, []
    for pair in pairs:
        text, length = greedy_continue(
            model, tokenizer, pair.prompt, device, max_new_tokens, stop_strings
        )
        lengths.append(length)
        hits += int(contains_answer(text, pair.positive_completion))
        texts.append(text)
    return sum(lengths) / len(lengths), hits / len(pairs), texts


def measure_outcome(
    model: nn.Module,
    tokenizer,
    directions: dict[int, torch.Tensor],
    pairs: Sequence[ContrastivePair],
    config: SteeringConfig,
    device: torch.device,
    label: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
) -> OutcomeCheck:
    """Greedy generations for ``pairs`` unsteered and with ``directions`` injected.

    ``pairs`` are *completion-format* pairs: the prompt is the question and the
    positive completion is the reference answer the accuracy check looks for.
    """
    if not pairs:
        raise ValueError("outcome check needs at least one question")
    base_len, base_acc, base_texts = _run(
        model, tokenizer, pairs, device, max_new_tokens, stop_strings
    )
    handles = attach_steering_hooks(model, directions, config)
    try:
        steer_len, steer_acc, steer_texts = _run(
            model, tokenizer, pairs, device, max_new_tokens, stop_strings
        )
    finally:
        for handle in handles:
            handle.remove()

    for pair, before, after in zip(pairs, base_texts, steer_texts, strict=True):
        logger.debug(
            "Outcome %s | %s\n  base:    %r\n  steered: %r", label, pair.prompt, before, after
        )

    return OutcomeCheck(
        label=label,
        num_questions=len(pairs),
        baseline_length=base_len,
        steered_length=steer_len,
        baseline_accuracy=base_acc,
        steered_accuracy=steer_acc,
    )
