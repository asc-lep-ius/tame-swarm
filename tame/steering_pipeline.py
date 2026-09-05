"""Load contrastive pairs, extract a behaviour direction, normalise, report.

The steering-vector default the server uses. It replaces the old
``create_default_steering_vectors`` -- which read instruction prefixes and
mean-pooled every position -- with the behavioural path #3 introduces: A/B
completions of a shared prompt, read at the answer token, differenced in means
and L2-normalised so magnitudes are comparable across goals.

Kept out of ``steering.py`` so the core module does not depend on the data layer;
this is the seam where the two meet.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.nn import functional as F

from contrastive_data import (
    MAX_LETTER_IMBALANCE,
    ContrastivePairSet,
    certification_for,
    letter_counts,
    letter_imbalance,
    load_certified_dataset,
    load_contrastive_dataset,
    resolve_pair_format,
)
from steering import SteeringConfig, SteeringVector, SteeringVectorExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SteeringExtraction:
    """Extracted vectors for one goal, with the metadata a reader needs to trust them.

    ``vectors`` is L2-normalised per layer (``SteeringVector`` guarantees unit
    norm), so ``pair_count``, ``source`` and ``tier_counts`` are what distinguish
    a direction estimated from sixty behavioural pairs from one estimated from
    four instruction prefixes -- the numbers that belong beside the vector, not
    inside it.
    """

    goal: str
    vectors: dict[int, SteeringVector]
    pair_count: int
    source: str
    layers: list[int]
    tier_counts: dict[str, int]
    pair_format: str
    # True only when (source, pair_format) is the pair the behavioural gate
    # certified for this goal (``contrastive_data.CERTIFIED``); a fallback or an
    # explicit override is measured by nobody and is labelled as such.
    certified: bool
    fallback_reason: str | None = None


def _load_pairs(
    goal: str,
    source: str | None,
    pair_format: str | None,
    load_dataset: Callable[..., Iterable[dict]] | None,
) -> tuple[ContrastivePairSet, bool, str | None]:
    if source is None:
        loaded = load_certified_dataset(goal, load_dataset=load_dataset)
        return loaded.pair_set, loaded.certified, loaded.fallback_reason
    resolved_format = resolve_pair_format(goal, pair_format)
    pair_set = load_contrastive_dataset(
        goal, source=source, load_dataset=load_dataset, pair_format=resolved_format
    )
    certification = certification_for(goal)
    certified = certification is not None and (certification.source, certification.pair_format) == (
        source,
        resolved_format,
    )
    return pair_set, certified, None


def _resolve_layers(model: nn.Module, layers: Sequence[int] | None) -> list[int]:
    if layers is not None:
        return list(layers)
    inner = getattr(model, "model", model)
    model_layers = getattr(inner, "layers", None)
    if model_layers is None:
        raise ValueError("cannot locate transformer layers to choose steering layers")
    num_layers = len(model_layers)
    return list(range(num_layers // 3, 2 * num_layers // 3))


def extract_steering_vectors(
    model: nn.Module,
    tokenizer,
    goal: str = "truthful",
    config: SteeringConfig | None = None,
    source: str | None = None,
    layers: Sequence[int] | None = None,
    load_dataset: Callable[..., Iterable[dict]] | None = None,
    max_pairs: int | None = None,
    pair_format: str | None = None,
) -> SteeringExtraction:
    """Extract an L2-normalised behaviour direction for ``goal`` at each layer.

    With ``source=None`` the pairs come from the goal's *certified* source and
    format (``contrastive_data.CERTIFIED``), falling back to the built-in set --
    flagged uncertified -- when that source needs a package or cache this
    environment lacks. An explicit ``source``/``pair_format`` overrides both, and
    counts as certified only if it happens to name the certified pair. Each pair
    is read at its recorded position, the means differenced and normalised.
    ``config.steering_layers`` (plus ``config.readout_layer``) selects the layers
    when ``layers`` is not given and a config is supplied; otherwise the middle
    third of the model is used.
    """
    if config is not None and layers is None and config.steering_layers:
        layers = list(config.steering_layers)
        # The sensor reads a layer above the actuators; it needs the goal direction
        # there too, or the homeostat would have to fall back to the top actuator.
        if config.readout_layer is not None and config.readout_layer not in layers:
            layers.append(config.readout_layer)
    resolved = _resolve_layers(model, layers)

    pair_set, certified, fallback_reason = _load_pairs(goal, source, pair_format, load_dataset)
    pairs = pair_set.pairs if max_pairs is None else pair_set.pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"goal {goal!r}: no pairs to extract from (max_pairs={max_pairs})")
    if pair_set.is_multiple_choice and letter_imbalance(pairs) > MAX_LETTER_IMBALANCE:
        logger.warning(
            "Goal %r: correct letters are unbalanced over the %d pairs being averaged "
            "(%s); the vector carries some of the bare A-minus-B direction",
            goal,
            len(pairs),
            letter_counts(pairs),
        )

    extractor = SteeringVectorExtractor(model, tokenizer, resolved)
    vectors = extractor.extract_from_pairs(pairs)
    resolved_format = pairs[0].pair_format
    status = "certified" if certified else "UNCERTIFIED"
    for vector in vectors.values():
        vector.name = goal
        vector.description = (
            f"Behavioural steering toward '{goal}' ({len(pairs)} {resolved_format} pairs, "
            f"{pair_set.source}, {status})"
        )

    return SteeringExtraction(
        goal=goal,
        vectors=vectors,
        pair_count=len(pairs),
        source=pair_set.source,
        layers=resolved,
        tier_counts=pair_set.tier_counts(),
        pair_format=resolved_format,
        certified=certified,
        fallback_reason=fallback_reason,
    )


def goal_similarity_matrix(
    vectors_by_goal: dict[str, dict[int, SteeringVector]], layer: int
) -> tuple[list[str], torch.Tensor]:
    """Cosine similarity between goal directions at one layer.

    The number the orthogonalisation decision (#3, feeding #4) must be made
    against: if ``cos(truthful, safe)`` is high, three PID loops would be
    regulating substantially the same direction and fighting each other. This
    reports it; it does not orthogonalise -- goal interaction stays dynamic
    through the economy, and only the decision about the measurement basis is
    downstream of this number.
    """
    goals = sorted(goal for goal, vectors in vectors_by_goal.items() if layer in vectors)
    if not goals:
        raise ValueError(f"no goal has a vector at layer {layer}")
    stacked = torch.stack([vectors_by_goal[goal][layer].vector.float() for goal in goals])
    normed = F.normalize(stacked, dim=-1)
    return goals, normed @ normed.T


def log_goal_similarity(
    vectors_by_goal: dict[str, dict[int, SteeringVector]], layer: int
) -> dict[tuple[str, str], float]:
    """Log the goal-similarity matrix at ``layer`` and return its off-diagonal pairs."""
    goals, matrix = goal_similarity_matrix(vectors_by_goal, layer)
    pairwise: dict[tuple[str, str], float] = {}
    logger.info("Goal cosine similarity at layer %d over goals %s:", layer, goals)
    for i, row_goal in enumerate(goals):
        row = "  ".join(f"{matrix[i, j].item():+.3f}" for j in range(len(goals)))
        logger.info("  %-10s %s", row_goal, row)
        for j in range(i + 1, len(goals)):
            pairwise[(row_goal, goals[j])] = float(matrix[i, j].item())
    return pairwise


def serving_config(
    goal: str, template: SteeringConfig, model_id: str | None = None
) -> SteeringConfig:
    """The config a server should steer ``goal`` with: certified layers and band, else the template.

    The layers, reference strength and strength band come from the goal's
    certification record, so what is served is what the gate passed. A goal whose
    band was never swept is held at its certified strength (``min == max``); a goal
    with no certification at all keeps the template's layers and band and is
    already reported uncertified by :func:`extract_steering_vectors`. The record
    names the model it was measured on; serving another model gets the same
    numbers and a warning, because the alternative -- the template's layers --
    was measured on nothing at all.
    """
    certification = certification_for(goal)
    if certification is None or certification.layers is None:
        return replace(template)
    if model_id is not None and certification.model is not None and model_id != certification.model:
        logger.warning(
            "Goal %r: layers %s and strength band were certified on %s, not %s; "
            "serving them unverified",
            goal,
            certification.layers,
            certification.model,
            model_id,
        )
    strength = (
        certification.strength if certification.strength is not None else template.base_strength
    )
    low, high = certification.strength_band or (strength, strength)
    return replace(
        template,
        steering_layers=list(certification.layers),
        readout_layer=certification.readout_layer,
        base_strength=strength,
        min_strength=low,
        max_strength=high,
    )


def calibration_texts(
    model: nn.Module,
    tokenizer,
    goal: str,
    num_prompts: int = 24,
    new_tokens: int = 32,
    chat_kwargs: dict | None = None,
    load_dataset: Callable[..., Iterable[dict]] | None = None,
) -> list[str]:
    """Texts that sample the *served* regime for ``goal``: its own prompts, answered.

    The resting distribution the homeostat is calibrated against must be the one it
    will read at inference. General prose without the chat template is not: #4
    measured chat-formatted answers sitting about two sigma above it along the
    truthful direction, which pinned the loop at the floor of its band. So the
    calibration corpus is the goal's contrastive prompts, wrapped in the chat
    template when the tokenizer has one, each followed by the model's own greedy
    continuation. Prompts are spread over the certified set so no tier dominates.
    """
    pair_set, _, _ = _load_pairs(goal, None, None, load_dataset)
    prompts = [pair.prompt for pair in pair_set.pairs]
    stride = max(1, len(prompts) // num_prompts)
    chosen = prompts[::stride][:num_prompts]
    template = getattr(tokenizer, "chat_template", None)
    texts = []
    for prompt in chosen:
        if template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(chat_kwargs or {}),
            )
        else:
            text = prompt
        if new_tokens > 0 and hasattr(model, "generate"):
            inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                generated = model.generate(  # pyright: ignore[reportCallIssue] # HF stubs
                    **inputs, max_new_tokens=new_tokens, do_sample=False
                )
            text = tokenizer.decode(generated[0], skip_special_tokens=False)
        texts.append(text)
    return texts
