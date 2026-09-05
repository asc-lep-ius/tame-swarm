"""Does injecting this vector change behaviour more than a random one would?

The quality gate #3 puts in place of PCA separability. The original acceptance
criterion -- "PCA: inter-goal distance > 2x intra-goal variance" -- would pass
*more easily* on the failure mode, because prompt-surface features separate
especially cleanly in activation space; it was anti-correlated with the property
it was meant to certify. This module measures the thing that actually matters: a
vector is accepted only if injecting it moves the model's preference on held-out
prompts, in the intended direction, by more than a matched random direction at
equal norm does.

**The metric.** For a held-out pair -- shared prompt, positive and negative
completion -- the log-odds is the length-normalised log-probability of the
positive completion minus that of the negative, under teacher forcing. It is
CAA's A/B probability read generalised to multi-token completions: a push on the
residual stream, then a read of which attractor the model now prefers. The effect
of a direction is the mean rise in that log-odds with the direction injected
versus unsteered.

**The controls.** Two, and both are load-bearing:

- *Random directions* at equal norm and layers set the floor. Steering at
  sufficient strength moves outputs in *some* direction; the question is whether
  it moves them in the *intended* one more than an arbitrary one does. Without
  this, "steering works" is unfalsifiable.
- *The instruction-prefix set* is the ceiling that must not be reached. A vector
  extracted from the retained instruction prefixes is measured the same way; if
  it scores as well as the completion-based vector, the metric is reading prompt
  wording and the metric itself is wrong.

No judge, no sampling: deterministic teacher-forced log-probs, so it runs in CI on
a tiny model and locally on Qwen3-1.7B.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from contrastive_data import ContrastivePair
from steering import SteeringConfig, SteeringHook, SteeringVector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectionEffect:
    """The mean held-out log-odds shift a single direction produced."""

    label: str
    effect: float
    baseline_log_odds: float
    steered_log_odds: float
    num_pairs: int


@dataclass(frozen=True)
class ValidationResult:
    """A vector's effect, the random-control distribution, and the two verdicts."""

    goal: str
    vector_effect: DirectionEffect
    random_effects: list[float]
    control_effect: DirectionEffect | None
    margin: float

    @property
    def random_mean(self) -> float:
        return (
            float(sum(self.random_effects) / len(self.random_effects))
            if self.random_effects
            else 0.0
        )

    @property
    def random_max(self) -> float:
        return max(self.random_effects) if self.random_effects else 0.0

    @property
    def beats_random(self) -> bool:
        """The effect clears the strongest random direction by the stated margin."""
        return self.vector_effect.effect > self.random_max + self.margin

    @property
    def beats_control(self) -> bool:
        """The completion vector outscores the instruction-prefix vector.

        Vacuously true when no control was measured; the caller decides whether to
        require a control, and the offline validation always supplies one.
        """
        if self.control_effect is None:
            return True
        return self.vector_effect.effect > self.control_effect.effect

    @property
    def passed(self) -> bool:
        return self.beats_random and self.beats_control


def _completion_log_prob(
    model: nn.Module, tokenizer, prompt: str, completion: str, device: torch.device, max_length: int
) -> float:
    """Length-normalised log p(completion | prompt) under teacher forcing.

    Length normalisation keeps a long correct answer comparable with a short wrong
    one: the quantity is per-token average log-probability over the completion
    span, scored only at the completion positions so the prompt's own likelihood
    never enters.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)[
        "input_ids"
    ]
    inputs = tokenizer(
        prompt + completion, return_tensors="pt", max_length=max_length, truncation=True
    ).to(device)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])
    start = min(int(prompt_ids.shape[1]), seq_len)
    if start >= seq_len:
        return float("nan")

    with torch.no_grad():
        logits = model(**inputs).logits.float()

    # Token at position t is predicted from logits at t-1.
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    targets = input_ids[0, 1:]
    token_lp = log_probs[torch.arange(seq_len - 1), targets]
    completion_lp = token_lp[start - 1 :]
    if completion_lp.numel() == 0:
        return float("nan")
    return float(completion_lp.mean().item())


def _pair_log_odds(
    model: nn.Module, tokenizer, pair: ContrastivePair, device: torch.device, max_length: int
) -> float:
    pos = _completion_log_prob(
        model, tokenizer, pair.prompt, pair.positive_completion, device, max_length
    )
    neg = _completion_log_prob(
        model, tokenizer, pair.prompt, pair.negative_completion, device, max_length
    )
    return pos - neg


def mean_log_odds(
    model: nn.Module,
    tokenizer,
    pairs: Sequence[ContrastivePair],
    device: torch.device,
    max_length: int = 128,
) -> float:
    """Mean held-out log-odds over pairs, skipping any that tokenise degenerately."""
    values = [_pair_log_odds(model, tokenizer, pair, device, max_length) for pair in pairs]
    finite = [value for value in values if value == value]  # drop NaN
    if not finite:
        raise ValueError("no held-out pair produced a finite log-odds")
    return float(sum(finite) / len(finite))


def attach_steering_hooks(
    model: nn.Module, directions: dict[int, torch.Tensor], config: SteeringConfig
) -> list:
    """Attach non-adaptive steering hooks injecting fixed directions, return handles."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers")  # noqa: B009  # model internals absent on nn.Module stubs
    handles = []
    fixed = SteeringConfig(
        steering_layers=config.steering_layers,
        base_strength=config.base_strength,
        adaptive=False,
        max_strength=config.max_strength,
        min_strength=config.min_strength,
        orthogonal_projection=False,
    )
    for layer_idx, direction in directions.items():
        if layer_idx >= len(layers):
            continue
        hook = SteeringHook(SteeringVector("probe", direction, layer_idx), fixed)
        handles.append(layers[layer_idx].register_forward_hook(hook))
    return handles


def measure_direction_effect(
    model: nn.Module,
    tokenizer,
    directions: dict[int, torch.Tensor],
    pairs: Sequence[ContrastivePair],
    config: SteeringConfig,
    device: torch.device,
    label: str,
    baseline_log_odds: float | None = None,
    max_length: int = 128,
) -> DirectionEffect:
    """Held-out log-odds shift from injecting ``directions`` at the configured strength.

    ``baseline_log_odds`` is the unsteered value; pass it once and reuse it across
    directions so the shared baseline is computed a single time.
    """
    if baseline_log_odds is None:
        baseline_log_odds = mean_log_odds(model, tokenizer, pairs, device, max_length)

    handles = attach_steering_hooks(model, directions, config)
    try:
        steered = mean_log_odds(model, tokenizer, pairs, device, max_length)
    finally:
        for handle in handles:
            handle.remove()

    return DirectionEffect(
        label=label,
        effect=steered - baseline_log_odds,
        baseline_log_odds=baseline_log_odds,
        steered_log_odds=steered,
        num_pairs=len(pairs),
    )


def _random_directions(
    reference: dict[int, torch.Tensor], generator: torch.Generator
) -> dict[int, torch.Tensor]:
    """One random unit direction per layer, matched to each reference vector's norm."""
    out: dict[int, torch.Tensor] = {}
    for layer_idx, vector in reference.items():
        noise = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
        noise = noise / noise.norm()
        out[layer_idx] = (noise * vector.norm()).to(vector.dtype)
    return out


def validate_steering_vector(
    model: nn.Module,
    tokenizer,
    goal: str,
    vectors: dict[int, SteeringVector],
    held_out: Sequence[ContrastivePair],
    config: SteeringConfig,
    device: torch.device,
    control_vectors: dict[int, SteeringVector] | None = None,
    num_random: int = 8,
    margin: float = 0.0,
    seed: int = 0,
    max_length: int = 128,
) -> ValidationResult:
    """Measure a goal vector against random directions and the instruction-prefix control.

    ``held_out`` must be disjoint from the pairs the vector was extracted from --
    this is a generalisation test, not a fit. Returns the effect, the random-control
    distribution and the two verdicts; the caller reads ``passed``.
    """
    directions = {layer: sv.vector for layer, sv in vectors.items()}
    baseline = mean_log_odds(model, tokenizer, held_out, device, max_length)

    vector_effect = measure_direction_effect(
        model,
        tokenizer,
        directions,
        held_out,
        config,
        device,
        f"{goal}:completion",
        baseline,
        max_length,
    )

    generator = torch.Generator().manual_seed(seed)
    random_effects: list[float] = []
    for index in range(num_random):
        random_dirs = _random_directions(directions, generator)
        effect = measure_direction_effect(
            model,
            tokenizer,
            random_dirs,
            held_out,
            config,
            device,
            f"random-{index}",
            baseline,
            max_length,
        )
        random_effects.append(effect.effect)

    control_effect = None
    if control_vectors is not None:
        control_dirs = {layer: sv.vector for layer, sv in control_vectors.items()}
        control_effect = measure_direction_effect(
            model,
            tokenizer,
            control_dirs,
            held_out,
            config,
            device,
            f"{goal}:instruction-prefix",
            baseline,
            max_length,
        )

    result = ValidationResult(
        goal=goal,
        vector_effect=vector_effect,
        random_effects=random_effects,
        control_effect=control_effect,
        margin=margin,
    )
    logger.info(
        "Validation %s: effect %+.4f vs random max %+.4f (mean %+.4f)%s -> %s",
        goal,
        result.vector_effect.effect,
        result.random_max,
        result.random_mean,
        f", control {control_effect.effect:+.4f}" if control_effect else "",
        "PASS" if result.passed else "FAIL",
    )
    return result
