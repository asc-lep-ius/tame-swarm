"""Did regulating the variable change what the organism does? (#5)

Everything else on the metrics surface is internal telemetry. A routing
correlation says the goal reaches the gate; a receptor norm says the coupling has
learned something; neither can say that any of it helped. #5's last acceptance
criterion asks for one number on this surface with a value that would mean
"steering did not help", and this is it: the served goal's own held-out pairs, run
on arms that differ only in the intervention, as the mean log-odds shift the
behavioural gate itself is defined on.

Three arms, in the order that isolates one thing each:

- ``served`` -- the model exactly as configured, hooks attached;
- ``unsteered`` -- the same pairs with the hooks off. ``served - unsteered`` is
  what steering does at all, against the certification's strongest matched random
  direction as the floor. **At or below that floor is the falsifying value.**
- ``constant`` -- only when the loop is adaptive: the same reference strength held
  fixed, loop off. ``served - constant`` is #4's value test, which found no
  significant difference on 100 held-out choices and is why the served system
  ships at a constant strength.

This is a probe and not a route's ordinary work: it runs forward passes, so it is
a ``POST`` that caches into ``TAMEApplication.outcome`` and ``GET /metrics/outcome``
reads the cache. The economy is frozen throughout -- a probe that paid the experts
would be a training step in disguise, and would move the wealth the next served
request routes on.

Twenty pairs is a smoke number. The certification measured 200, and at 20 the
standard error on a log-odds difference is wide enough that only a large effect or
a floor-level one is readable. That is the honest use: a change of regime that
this cannot see is not one the surface is claiming to see.
"""

import logging
import os
from dataclasses import replace

import torch

from app import TAMEApplication
from behavioural_validation import held_out_log_odds
from contrastive_data import ContrastivePair, certification_for, interleaved_split
from contrastive_data import load_contrastive_dataset as load_pairs
from evaluation import HeldOutSplit, evaluate, fingerprint_tokens
from mob.utils import frozen_economy, frozen_traces
from models import OutcomeArm, OutcomeMetrics
from observability import probed_at

logger = logging.getLogger(__name__)

DEFAULT_PROBE_PAIRS = 20
# A request may not turn into an evaluation run. The gate's own certification uses
# 200 and takes minutes; past this the caller wants ``scripts/validate_steering.py``.
MAX_PROBE_PAIRS = 200
# The certified held-out set is the last 200 of the interleaved split, so the
# probe's pairs are a prefix of the gate's -- never a fresh split, which would
# make ``beats_random`` a comparison between two different held-out sets.
CERTIFIED_HELD_OUT = 200
# Sequences of a configured held-out split to score per arm. Perplexity is a
# corpus statistic and MLflow owns the run-level one (#7); this is a spot check
# bounded to fit in a request.
PERPLEXITY_SEQUENCES = 32
PERPLEXITY_BATCH_SIZE = 4
HELD_OUT_SPLIT_ENV = "TAME_HELD_OUT_SPLIT"


class ProbeUnavailable(RuntimeError):
    """The probe cannot run here: no steering, or the goal's held-out set is not reachable."""


def certified_held_out(goal: str, count: int) -> list[ContrastivePair]:
    """The first ``count`` of the goal's certified held-out pairs, in the certified format.

    Source *and* format come from the certification record rather than from a
    caller's preference: ``certified_random_max`` was measured on those, and a
    delta measured on anything else is not comparable to it.
    """
    certification = certification_for(goal)
    if certification is None:
        raise ProbeUnavailable(
            f"goal {goal!r} has no certification record, so there is no held-out set "
            "the gate measured and no random floor to compare a delta against"
        )
    try:
        pairs = list(
            load_pairs(goal, source=certification.source, pair_format=certification.pair_format)
        )
    except Exception as exc:  # noqa: BLE001 - any loader failure is the same answer to the caller
        raise ProbeUnavailable(
            f"goal {goal!r}: the certified pair source {certification.source!r} is not "
            f"available on this host ({exc})"
        ) from exc
    _, held_out = interleaved_split(pairs, CERTIFIED_HELD_OUT)
    if not held_out:
        raise ProbeUnavailable(f"goal {goal!r}: the certified split is empty")
    # A stride over the gate's set, not its first N: the source is ordered by tier
    # and topic, so a prefix is one block of it. Either is a subset of the certified
    # held-out -- which is all the comparability with `random_max` requires -- but
    # only the stride is representative of it.
    stride = max(1, len(held_out) // count)
    return held_out[::stride][:count]


def _arm(
    app: TAMEApplication, pairs: list[ContrastivePair], device: torch.device
) -> tuple[OutcomeArm, list[float]]:
    """One arm's statistics, and its per-pair log-odds so the arms can be paired.

    The per-pair values are what make a 20-pair delta honest: paired against the
    same pair in another arm, the between-pair variance -- which is most of the
    variance, since the pairs differ in topic and difficulty -- cancels, and what
    is left is the intervention. Taking a difference of two arms' means would keep
    all of it, and would silently compare different subsets whenever a pair
    tokenises degenerately in one arm and not the other.
    """
    # Each arm is measured on a fresh loop, so an arm cannot inherit the previous
    # one's integral. The visible cost is that a probe leaves ``/metrics/pid``'s
    # rolling window empty -- ``window_passes`` reads 0 until the next served request.
    tissue = app.homeostat.homeostat if app.homeostat else None
    if tissue is not None:
        tissue.reset()
    values = held_out_log_odds(app.model, app.tokenizer, pairs, device)  # pyright: ignore[reportArgumentType] # AutoModelForCausalLM is an nn.Module at runtime
    mean = float(sum(values) / len(values))
    accuracy = sum(value > 0 for value in values) / len(values)
    return (
        OutcomeArm(mean_log_odds=mean, accuracy=accuracy, mean_strength=_mean_strength(app)),
        values,
    )


def _paired_delta(served: list[float], other: list[float]) -> tuple[float, float | None]:
    """Mean paired difference and its standard error, or ``(unpaired mean gap, None)``.

    ``held_out_log_odds`` drops a pair that tokenises degenerately, so two arms can
    return different lengths. When they do the pairing is broken and the honest
    answer is the difference of means with **no** standard error rather than a
    number computed over mismatched pairs.
    """
    if len(served) != len(other):
        return float(sum(served) / len(served) - sum(other) / len(other)), None
    deltas = [a - b for a, b in zip(served, other, strict=True)]
    mean = sum(deltas) / len(deltas)
    if len(deltas) < 2:
        return float(mean), None
    variance = sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1)
    return float(mean), float((variance / len(deltas)) ** 0.5)


def _floor_applies(app: TAMEApplication, certification) -> str | None:
    """Why the certification's random floor does not describe this process, if it does not.

    ``certified_random_max`` was measured at one model, one set of layers and one
    strength. A served process can differ on all three -- ``install_goal`` takes a
    strength, and nothing pins the model to the certified one -- and comparing a
    delta against a floor from another configuration is the same error as
    comparing it against another held-out set.
    """
    if certification.model is not None and app.model_id != certification.model:
        return f"served on {app.model_id}, certified on {certification.model}"
    layers = tuple(sorted(app.steering_config.steering_layers))
    if certification.layers is not None and layers != tuple(sorted(certification.layers)):
        return f"served at layers {list(layers)}, certified at {list(certification.layers)}"
    strength = app.steering_config.base_strength
    if certification.strength is not None and abs(strength - certification.strength) > 1e-6:
        return f"served at strength {strength}, certified at {certification.strength}"
    return None


def _mean_strength(app: TAMEApplication) -> float | None:
    """What the injection was held at over the arm; ``None`` when nothing was injected."""
    if app.homeostat is None or not app.homeostat.hooks:
        return None
    tissue = app.homeostat.homeostat
    history = list(tissue.strength_history)
    if app.steering_config.adaptive and history:
        return float(sum(history) / len(history))
    return app.steering_config.base_strength


def _perplexity(app: TAMEApplication, device: torch.device) -> float | None:
    """Held-out perplexity on a bounded prefix of a configured split, or ``None``.

    Opt-in through ``TAME_HELD_OUT_SPLIT``: a serving process has no held-out
    corpus of its own, and inventing one on the spot would produce a number whose
    disjointness from anything is unknown. The prefix is re-fingerprinted so it
    cannot be mistaken for the whole split's number in a run comparison.
    """
    path = os.environ.get(HELD_OUT_SPLIT_ENV)
    if not path:
        return None
    try:
        split = HeldOutSplit.load(path)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("[OUTCOME] %s=%s could not be loaded: %s", HELD_OUT_SPLIT_ENV, path, exc)
        return None
    input_ids = split.input_ids[:PERPLEXITY_SEQUENCES]
    prefix = replace(
        split,
        input_ids=input_ids,
        attention_mask=split.attention_mask[:PERPLEXITY_SEQUENCES],
        fingerprint=fingerprint_tokens(input_ids),
    )
    return evaluate(app.model, prefix, PERPLEXITY_BATCH_SIZE, device).perplexity  # pyright: ignore[reportArgumentType] # as above


def probe_outcome(app: TAMEApplication, num_pairs: int = DEFAULT_PROBE_PAIRS) -> OutcomeMetrics:
    """Run the arms, cache the result on ``app``, and return it.

    The hooks come off for the unsteered arm and go straight back in a ``finally``,
    for the same reason ``install_goal`` does it: a probe that raised half way
    through must not leave the server unsteered.
    """
    if app.homeostat is None:
        raise ProbeUnavailable("steering is not active, so there is no intervention to measure")
    if not 1 <= num_pairs <= MAX_PROBE_PAIRS:
        raise ValueError(f"num_pairs must be in [1, {MAX_PROBE_PAIRS}], got {num_pairs}")

    homeostat = app.homeostat
    goal = homeostat.goal
    pairs = certified_held_out(goal, num_pairs)
    device = next(app.model.parameters()).device  # pyright: ignore[reportAttributeAccessIssue] # HF stubs lack .parameters()
    adaptive = app.steering_config.adaptive

    # The probe's forwards are not served traffic: they must not pay the economy,
    # and they must not enter the routing window ``/metrics/coupling`` reports as
    # the served goal's effect -- the unsteered arm runs through the same layers.
    with frozen_economy(app.model), frozen_traces(app.model):  # pyright: ignore[reportArgumentType] # as above
        served, served_values = _arm(app, pairs, device)
        perplexity_steered = _perplexity(app, device)

        try:
            homeostat.detach_from_model()
            unsteered, unsteered_values = _arm(app, pairs, device)
            perplexity_unsteered = _perplexity(app, device)
        finally:
            homeostat.attach_to_model(app.model)  # pyright: ignore[reportArgumentType] # as above

        constant: OutcomeArm | None = None
        constant_values: list[float] | None = None
        if adaptive:
            app.steering_config.adaptive = False
            try:
                constant, constant_values = _arm(app, pairs, device)
            finally:
                app.steering_config.adaptive = adaptive

    certification = certification_for(goal)
    random_max = certification.random_max if certification else None
    floor_mismatch = _floor_applies(app, certification) if certification else None
    delta, delta_standard_error = _paired_delta(served_values, unsteered_values)
    arms = {"served": served, "unsteered": unsteered}
    if constant is not None:
        arms["constant"] = constant

    outcome = OutcomeMetrics(
        goal=goal,
        probed_at=probed_at(),
        num_pairs=len(served_values),
        arms=arms,
        served_minus_unsteered_log_odds=delta,
        served_minus_unsteered_standard_error=delta_standard_error,
        served_minus_unsteered_accuracy=served.accuracy - unsteered.accuracy,
        adaptive_minus_constant_log_odds=(
            _paired_delta(served_values, constant_values)[0]
            if constant_values is not None
            else None
        ),
        certified_random_max=random_max,
        # None, not False, when the floor was measured on a different configuration:
        # "the comparison does not apply here" is a third answer.
        beats_random=(
            (delta > random_max) if random_max is not None and floor_mismatch is None else None
        ),
        floor_not_applicable=floor_mismatch,
        held_out_perplexity_steered=perplexity_steered,
        held_out_perplexity_unsteered=perplexity_unsteered,
        stale=False,
    )
    logger.info(
        "[OUTCOME] %s: served %.4f, unsteered %.4f, delta %+.4f against random max %s on %d pairs",
        goal,
        served.mean_log_odds,
        unsteered.mean_log_odds,
        delta,
        f"{random_max:+.4f}" if random_max is not None else "unrecorded",
        len(pairs),
    )
    app.outcome = outcome
    return outcome
