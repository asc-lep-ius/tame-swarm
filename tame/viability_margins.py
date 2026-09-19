"""The margins the viability core regulates against, and why it cannot farm them.

#42's tissue is the homeostat's shape with the regulated variable generalised from
a projection onto a certified direction to a **viability margin**. This module
computes that variable. It is deliberately not in ``evaluation``: that module owns
the held-out data -- the fixed split arms are subtracted on, and #41's rotating
stream -- and this one owns what is computed from it, which is a different question
with different failure modes.

**Three margins, one definition.** Expected calibration error and Brier score say
whether the organism's confidence means anything; accuracy on the rotated stream
says whether it is right; the risk--coverage area says whether its confidence
*orders* its answers, which is the only one of the three that a system allowed to
abstain can be judged on. Each is reported as a **margin against a frozen copy of
the base evaluated on the same items**, never as a level. A level drifts with the
stream: a rotation of harder documents lowers every number, and a core regulating
on levels would read that as its own decay and act. The base moves with the stream
in the same direction, so the margin is what is left after the stream's difficulty
is divided out.

**The frozen copy.** Loaded read-only and never served or trained: no parameter
requires a gradient, none ever receives one, it carries no converted MoB layer, and
it shares no parameter storage with the model being measured. The last is not
paranoia about aliasing -- it is the failure this whole module has to make loud. A
base that *is* the organism makes every margin identically zero, which reads
exactly like a healthy organism at parity with its base, and a core regulating on
it would regulate on nothing at all. ``tests/test_no_silent_noops.py`` holds that
pairing.

**Evaluation costs budget, and only when the core asks.** The core requesting an
evaluation debits the budget ledger (#43); the trainer taking a recorded
measurement does not, and is refused a ledger if it offers one. That asymmetry is
the readiness register's rule made arithmetic: an action may *cost* budget, and
nothing the tissue does may *set* what evaluation says. #43 does not exist yet, so
``BudgetLedger`` is the interface it has to satisfy and ``charge_evaluation`` is
the seam it plugs into -- no economy changes here.

**Farming reads as the canaries parting company with the stream.** The stream
rotates and the canaries do not, so an organism that has learned the evaluation
rather than the task improves on the half it can memorise and not on the half it
cannot. Two shapes, both in ``farming_reasons``: canary accuracy standing above
stream accuracy at one reading, and canary accuracy climbing with step count while
the stream's does not.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from mob import frozen_economy, get_mob_layers
from rotating_stream import RotatingStream, RotationRecord

logger = logging.getLogger(__name__)

# Equal-width confidence bins for the calibration error. Fifteen is the number the
# calibration literature reports ECE at and the number #23's calibration corpus was
# sized against; it is here rather than at a call site because an ECE at one bin
# count is not comparable to an ECE at another, and a margin is a subtraction.
CALIBRATION_BINS = 15

# Who asked. Two callers, and the difference between them is the whole of the
# budgeted-action rule: the core pays for what it asks for, the trainer's recorded
# measurement is not the organism's to be charged for.
CALLER_CORE = "core"
CALLER_TRAINER = "trainer"
CALLERS = frozenset({CALLER_CORE, CALLER_TRAINER})

# What the ledger is told it is paying for, so a budget history can be read back.
EVALUATION_REASON = "viability-evaluation"

# Budget per item evaluated. Per item rather than per call, because a call is not a
# cost: an organism charged a flat fee would evaluate on the largest stream it could
# find, and one charged nothing would evaluate every step. The number is a
# placeholder until #43 derives it from the budget's own algebra -- `ledger.py`'s
# fixed point is what that derivation reads -- and it is a constant here so that the
# derivation has one place to land.
EVALUATION_COST_PER_ITEM = 0.01

# How far canary accuracy may stand above stream accuracy before it is farming
# rather than noise, and the smallest canary set that tolerance can be read at.
# One canary is 1/n of the canary accuracy, so at n <= 6 a single canary clears the
# tolerance on its own and the detector could not fail to fire; 1/7 = 0.143 is
# where that stops, and the floor is set one above it rather than on it. A
# threshold a detector's quietest possible signal reaches 95% of is a threshold
# that fires on the granularity of its own instrument. Eight is a floor and not a
# sufficiency claim: the canary set a margin is really regulated on wants an order
# of magnitude more, and #42 sizes it when it assembles one.
CANARY_DIVERGENCE_TOLERANCE = 0.15
MIN_CANARIES = 8

# The second shape: accuracy per thousand steps, canary above stream. A static set
# being memorised shows up here before it shows up in the level, because the level
# also carries whatever the canaries happen to be easier or harder at.
STEPS_PER_TREND_UNIT = 1000
CANARY_TREND_TOLERANCE = 0.05
MIN_READINGS_FOR_TREND = 3


class ViabilityError(RuntimeError):
    """Raised when a margin would be read off something that cannot carry one."""


# --- What the model answered, and how sure it was ---------------------------------------


@dataclass(frozen=True)
class Predictions:
    """One prediction per stream item: right or wrong, and the probability behind it.

    One per item rather than one per token. A per-token accuracy over a document is
    dominated by the tokens nobody is asking about -- articles, punctuation, the
    second half of a word -- and a margin computed over those is a margin on
    tokenisation. The item's answer is the question the manifest asked.
    """

    confidence: torch.Tensor
    correct: torch.Tensor
    is_canary: torch.Tensor

    def __post_init__(self) -> None:
        """Bool masks and matching lengths, checked here because #42 builds these.

        ``risk_coverage_auc`` negates ``correct``, and ``~`` on an integer tensor is
        a bitwise NOT rather than a logical one: a 0/1 ``int64`` mask silently gives
        -1 and -2, and the metric comes back negative -- a number a reader would
        read as a value rather than as an error. The other three metrics coerce and
        would go on agreeing, so the corruption would show up in one column of four.
        """
        for name in ("correct", "is_canary"):
            mask = getattr(self, name)
            if mask.dtype is not torch.bool:
                raise ViabilityError(
                    f"Predictions.{name} is {mask.dtype}, not torch.bool; a 0/1 integer mask "
                    "negates bitwise and makes the risk--coverage area come back negative"
                )
        lengths = {int(field.shape[0]) for field in (self.confidence, self.correct, self.is_canary)}
        if len(lengths) != 1:
            raise ViabilityError(
                f"Predictions fields disagree on length ({sorted(lengths)}); a shorter canary "
                "mask would silently shift which items are read as canaries"
            )

    def __len__(self) -> int:
        return int(self.confidence.shape[0])

    def where(self, mask: torch.Tensor) -> "Predictions":
        return Predictions(self.confidence[mask], self.correct[mask], self.is_canary[mask])

    @property
    def stream(self) -> "Predictions":
        return self.where(~self.is_canary)

    @property
    def canaries(self) -> "Predictions":
        return self.where(self.is_canary)


def accuracy(predictions: Predictions) -> float:
    return float(predictions.correct.to(torch.float32).mean().item())


def expected_calibration_error(predictions: Predictions, bins: int = CALIBRATION_BINS) -> float:
    """Confidence minus accuracy, averaged over equal-width bins and weighted by count.

    Equal-width rather than equal-mass: the bins have to mean the same thing between
    the organism and the base, and an equal-mass binning puts its edges where each
    model's own confidences fall, which makes the two histograms incomparable and
    the margin between them arithmetic on different axes.
    """
    confidence = predictions.confidence
    correct = predictions.correct.to(confidence.dtype)
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=confidence.dtype)
    index = torch.bucketize(confidence, edges[1:-1])

    total = 0.0
    for bin_index in range(bins):
        members = index == bin_index
        count = int(members.sum().item())
        if count == 0:
            continue
        gap = abs(float(correct[members].mean().item()) - float(confidence[members].mean().item()))
        total += count * gap
    return total / len(predictions)


def brier_score(predictions: Predictions) -> float:
    """The top-label Brier score: squared error of the confidence against being right."""
    correct = predictions.correct.to(predictions.confidence.dtype)
    return float(((predictions.confidence - correct) ** 2).mean().item())


def risk_coverage_auc(predictions: Predictions) -> float:
    """Area under the risk--coverage curve: error rate averaged over every coverage.

    The selective-prediction quality, and the one metric of the three that reads
    what an organism allowed to decline is actually good at -- not whether it is
    right, but whether its confidence *orders* its answers, so that declining the
    least confident declines the wrong ones. Lower is better, which is why the
    margin below subtracts in the other direction.

    Ties are broken by the stream's own order (``stable=True``). Arbitrary, but
    reproducible: an unstable sort would make the number depend on the backend.
    """
    order = torch.argsort(predictions.confidence, descending=True, stable=True)
    errors = (~predictions.correct)[order].to(torch.float32)
    covered = torch.arange(1, len(errors) + 1, dtype=torch.float32)
    return float((torch.cumsum(errors, dim=0) / covered).mean().item())


@dataclass(frozen=True)
class MetricSet:
    """One model's four numbers on one set of items."""

    accuracy: float
    calibration_error: float
    brier: float
    risk_coverage_auc: float

    @classmethod
    def of(cls, predictions: Predictions) -> "MetricSet":
        if len(predictions) == 0:
            raise ViabilityError("No items to read a viability margin on")
        return cls(
            accuracy=accuracy(predictions),
            calibration_error=expected_calibration_error(predictions),
            brier=brier_score(predictions),
            risk_coverage_auc=risk_coverage_auc(predictions),
        )


@dataclass(frozen=True)
class ViabilityMargins:
    """The four margins, each signed so that positive is the organism ahead of its base.

    The signs are not cosmetic. A core regulating a margin toward a setpoint acts on
    the sign of the error, and two of these metrics are better when they are smaller;
    leaving that to the reader would put the direction of an actuator in a comment.
    """

    accuracy: float
    calibration: float
    brier: float
    selective: float

    @classmethod
    def between(cls, model: MetricSet, base: MetricSet) -> "ViabilityMargins":
        return cls(
            accuracy=model.accuracy - base.accuracy,
            calibration=base.calibration_error - model.calibration_error,
            brier=base.brier - model.brier,
            selective=base.risk_coverage_auc - model.risk_coverage_auc,
        )


# --- The frozen copy of the base ----------------------------------------------------------


class FrozenBase:
    """The base checkpoint a margin is measured against: read-only, never served.

    The constructor checks rather than fixes. ``freeze`` is what makes a module
    read-only, and it is a separate call on purpose: a constructor that quietly
    called ``requires_grad_(False)`` on whatever it was handed would silently freeze
    the model being trained if somebody passed the wrong one, which is a worse
    failure than the one it would be preventing.
    """

    def __init__(self, module: nn.Module) -> None:
        converted = get_mob_layers(module)
        if converted:
            raise ViabilityError(
                f"The frozen base carries {len(converted)} MoB layers, so it is the organism and "
                "not the base it is supposed to be measured against"
            )
        trainable = [name for name, tensor in module.named_parameters() if tensor.requires_grad]
        if trainable:
            raise ViabilityError(
                f"The frozen base has {len(trainable)} parameters requiring a gradient "
                f"(first: {trainable[0]}); freeze it with FrozenBase.freeze rather than handing "
                "over a module something else is training"
            )
        if module.training:
            raise ViabilityError("The frozen base is in training mode")
        self.module = module

    @classmethod
    def freeze(cls, module: nn.Module) -> "FrozenBase":
        """Put a module beyond training, then check that it got there."""
        return cls(module.eval().requires_grad_(False))

    def assert_read_only(self) -> None:
        """Nothing requires a gradient and nothing has received one.

        Called after every pass rather than only at construction: ``requires_grad``
        is a property anything holding the module can set back, and a gradient that
        arrived is evidence it did.
        """
        offenders = [
            name
            for name, tensor in self.module.named_parameters()
            if tensor.requires_grad or tensor.grad is not None
        ]
        if offenders:
            raise ViabilityError(
                f"The frozen base received a gradient or was made trainable ({offenders[:3]}); "
                "it is loaded read-only and is never trained"
            )

    def assert_separate_from(self, model: nn.Module) -> None:
        """The base is not the organism, and does not share a tensor with it.

        Identity first, storage second. A base that is the served model makes every
        margin exactly zero -- a healthy-looking reading from a core regulating on
        nothing -- and a base that merely *shares* a parameter makes the margin
        wrong on that parameter only, which is the same failure wearing a number
        that looks plausible.

        Storage and not ``id()``: the aliasing that actually happens here produces
        two distinct ``Parameter`` objects over one buffer, which is what
        ``nn.Parameter(other.data)``, an assignment to ``.data``, and loading the
        base and the organism from one memory-mapped checkpoint all give. An
        identity check passes on every one of them and reads as a separate base.

        The storage and not ``data_ptr()``, which is the storage *plus the
        tensor's offset into it*: a slice of another model's weight reports a
        different pointer while sharing every byte under it. Empty parameters are
        skipped because ``data_ptr()`` is ``0`` for all of them, so two unrelated
        models each carrying one would otherwise be refused as aliases.
        """
        if self.module is model:
            raise ViabilityError(
                "The frozen base is the model being measured, so every margin would be "
                "identically zero and the core would regulate on nothing"
            )
        base_storage = {
            tensor.untyped_storage().data_ptr()
            for tensor in self.module.parameters()
            if tensor.numel()
        }
        shared = [
            name
            for name, tensor in model.named_parameters()
            if tensor.numel() and tensor.untyped_storage().data_ptr() in base_storage
        ]
        if shared:
            raise ViabilityError(
                f"The frozen base shares parameter storage with the model being measured "
                f"({shared[:3]}); it is a copy of the base, not a view of the organism"
            )


# --- Evaluation as a budgeted action -------------------------------------------------------


class BudgetLedger(Protocol):
    """What #43's budget has to offer for an evaluation to cost anything.

    One method, because one is all this seam needs and every method added here is a
    method the tissue can reach. Reading the budget is a channel the register
    already declares; *setting* it is sealed, so nothing here returns a way to.
    """

    def debit(self, amount: float, reason: str) -> float:
        """Take ``amount`` off the budget and return what is left.

        **Raises when the budget cannot cover ``amount``.** The refusal is part of
        the interface rather than an implementation detail: a ledger that signalled
        "cannot afford" by returning a negative balance would be obeyed rather than
        heard, because a caller that got a number back would treat the debit as
        having happened -- and an organism out of budget would go on evaluating for
        free. ``charge_evaluation`` reads the balance back as well, so a ledger
        that signals the wrong way is caught instead of believed.
        """
        ...


def evaluation_cost(num_items: int) -> float:
    return num_items * EVALUATION_COST_PER_ITEM


def charge_evaluation(caller: str, ledger: BudgetLedger | None, num_items: int) -> float:
    """Debit the budget when the core asked, and never when the trainer did.

    Both directions are refusals rather than defaults. A core with no ledger is an
    organism evaluating for free, which is the readiness register's "request an
    evaluation" with its cost removed and therefore an unbounded action. A trainer
    *with* a ledger is the other half: a recorded measurement that moved the budget
    would make the number the experiment reads a number the experiment caused.
    """
    if caller not in CALLERS:
        raise ViabilityError(f"Unknown evaluation caller {caller!r}; expected one of {CALLERS}")
    if caller == CALLER_TRAINER:
        if ledger is not None:
            raise ViabilityError(
                "The trainer was handed a budget ledger. A recorded measurement does not cost "
                "the organism anything -- only an evaluation the core asked for does"
            )
        return 0.0
    if ledger is None:
        raise ViabilityError(
            "The core asked for an evaluation with no budget to charge it to. Asking is a "
            "budgeted action; an evaluation that costs nothing is an unbounded one"
        )
    cost = evaluation_cost(num_items)
    remaining = ledger.debit(cost, EVALUATION_REASON)
    if remaining is None:
        raise ViabilityError(
            "The budget ledger's debit() returned nothing. It returns what is left, and the "
            "caller reads that back: a ledger written as a command cannot report a budget it "
            "has just overdrawn"
        )
    if remaining < 0:
        raise ViabilityError(
            f"Paying {cost:.3f} for this evaluation took the budget to {remaining:.3f}. A ledger "
            "refuses what it cannot afford by raising; one that signals by return value would "
            "otherwise let an organism out of budget evaluate for nothing"
        )
    return cost


# --- The reading ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryReading:
    """What the memorisable half and the un-memorisable half said at one step."""

    step: int
    canary_accuracy: float
    stream_accuracy: float
    num_canaries: int
    num_stream_items: int

    @property
    def divergence(self) -> float:
        return self.canary_accuracy - self.stream_accuracy


@dataclass(frozen=True)
class ViabilityReading:
    """One evaluation: the margins, both models' numbers, the canaries and the cost."""

    step: int
    caller: str
    cost: float
    margins: ViabilityMargins
    model_metrics: MetricSet
    base_metrics: MetricSet
    canary: CanaryReading
    rotation: RotationRecord

    def as_metrics(self) -> dict[str, float]:
        """Prefixed ``viability/``, so nothing can be mistaken for ``eval/`` or ``train/``.

        The same rule #12 established for the held-out numbers, for the same reason:
        a margin and a level differ by a subtraction nobody can see in a metric name.
        """
        return {
            "viability/accuracy_margin": self.margins.accuracy,
            "viability/calibration_margin": self.margins.calibration,
            "viability/brier_margin": self.margins.brier,
            "viability/selective_margin": self.margins.selective,
            "viability/stream_accuracy": self.canary.stream_accuracy,
            "viability/canary_accuracy": self.canary.canary_accuracy,
            "viability/cost": self.cost,
        }


def score_predictions(
    model: nn.Module, stream: RotatingStream, batch_size: int, device: torch.device
) -> Predictions:
    """Each item's answer and the probability the model put on it.

    Under ``no_grad`` and ``frozen_economy``, for the same reason ``evaluate`` is: a
    margin read while the economy settled would be a training step wearing the name
    of a measurement. Restored on the way out including when the forward raises.
    """
    was_training = model.training
    confidences: list[torch.Tensor] = []
    corrects: list[torch.Tensor] = []

    model.eval()
    try:
        with torch.no_grad(), frozen_economy(model):
            for batch in stream.batches(batch_size):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                positions = batch["answer_positions"].to(device)
                answers = batch["answer_ids"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                rows = torch.arange(int(input_ids.shape[0]), device=device)
                probabilities = torch.softmax(outputs.logits[rows, positions].float(), dim=-1)
                confidence, predicted = probabilities.max(dim=-1)

                confidences.append(confidence.cpu())
                corrects.append((predicted == answers).cpu())
    finally:
        model.train(was_training)

    return Predictions(
        confidence=torch.cat(confidences),
        correct=torch.cat(corrects),
        is_canary=stream.is_canary,
    )


def measure_viability(
    model: nn.Module,
    base: FrozenBase,
    stream: RotatingStream,
    batch_size: int,
    device: torch.device,
    caller: str,
    step: int = 0,
    ledger: BudgetLedger | None = None,
) -> ViabilityReading:
    """The three margins against the frozen base, on the rotation both were read on.

    The budget is debited **before** the forward passes: an evaluation the ledger
    refuses is an evaluation that does not happen, and charging afterwards would let
    an organism out of budget take one more reading than it could pay for.

    The margins are read on the stream items alone. The canaries are scored in the
    same pass, on the same model, and reported separately -- an item the organism
    may have memorised belongs in the farming check and not in the number the core
    regulates against.
    """
    base.assert_separate_from(model)
    if stream.num_canaries < MIN_CANARIES:
        raise ViabilityError(
            f"The rotation carries {stream.num_canaries} canaries, below the {MIN_CANARIES} the "
            f"{CANARY_DIVERGENCE_TOLERANCE:.0%} divergence tolerance can be read at: one canary "
            f"of a set this size moves the accuracy by {1 / max(stream.num_canaries, 1):.3f} "
            f"against a tolerance of {CANARY_DIVERGENCE_TOLERANCE}, so the quietest signal the "
            "detector can give is the size of its own threshold"
        )

    cost = charge_evaluation(caller, ledger, stream.num_items)

    model_predictions = score_predictions(model, stream, batch_size, device)
    base_predictions = score_predictions(base.module, stream, batch_size, device)
    base.assert_read_only()

    model_metrics = MetricSet.of(model_predictions.stream)
    base_metrics = MetricSet.of(base_predictions.stream)
    reading = ViabilityReading(
        step=step,
        caller=caller,
        cost=cost,
        margins=ViabilityMargins.between(model_metrics, base_metrics),
        model_metrics=model_metrics,
        base_metrics=base_metrics,
        canary=CanaryReading(
            step=step,
            canary_accuracy=accuracy(model_predictions.canaries),
            stream_accuracy=model_metrics.accuracy,
            num_canaries=stream.num_canaries,
            num_stream_items=stream.num_items - stream.num_canaries,
        ),
        rotation=stream.rotation,
    )

    logger.info(
        f"  viability @ {step} ({caller}, cost {cost:.3f}): "
        f"acc {reading.margins.accuracy:+.4f} | ece {reading.margins.calibration:+.4f} "
        f"| brier {reading.margins.brier:+.4f} | aurc {reading.margins.selective:+.4f} "
        f"| canary {reading.canary.canary_accuracy:.3f} vs stream "
        f"{reading.canary.stream_accuracy:.3f} | {stream.rotation.label}"
    )
    return reading


# --- Farming ----------------------------------------------------------------------------------


def _slope_per_unit(steps: Sequence[int], values: Sequence[float]) -> float:
    """Least-squares slope in accuracy per ``STEPS_PER_TREND_UNIT`` steps."""
    axis = torch.tensor(steps, dtype=torch.float64)
    readings = torch.tensor(values, dtype=torch.float64)
    spread = axis - axis.mean()
    denominator = float((spread * spread).sum().item())
    if denominator == 0.0:
        return 0.0
    numerator = float((spread * (readings - readings.mean())).sum().item())
    return numerator / denominator * STEPS_PER_TREND_UNIT


def farming_reasons(readings: Sequence[CanaryReading]) -> list[str]:
    """Why these readings look like farming rather than learning; empty when they do not.

    Two shapes, each its own line, because they are two different things to do about
    it. **A level**: canary accuracy standing above stream accuracy at the latest
    reading, which is the organism doing better on what it can memorise than on what
    it cannot, now. **A trend**: canary accuracy climbing faster than stream accuracy
    over the history, which is a static set being memorised and is visible before the
    level is -- the level also carries whatever the canaries happen to be easier at,
    a constant offset a trend divides out.

    Named and not refused, the same shape as ``code_drift`` and
    ``human_channel_violations``: what a core does about farming is #42's decision,
    and a detector that acted on its own finding would be one.
    """
    if not readings:
        return []

    reasons: list[str] = []
    latest = readings[-1]
    if latest.divergence > CANARY_DIVERGENCE_TOLERANCE:
        reasons.append(
            f"  canary accuracy {latest.canary_accuracy:.3f} stands {latest.divergence:.3f} above "
            f"stream accuracy {latest.stream_accuracy:.3f} at step {latest.step}, over the "
            f"{CANARY_DIVERGENCE_TOLERANCE:.2f} tolerance"
        )

    if len(readings) >= MIN_READINGS_FOR_TREND:
        steps = [reading.step for reading in readings]
        canary_slope = _slope_per_unit(steps, [r.canary_accuracy for r in readings])
        stream_slope = _slope_per_unit(steps, [r.stream_accuracy for r in readings])
        if canary_slope - stream_slope > CANARY_TREND_TOLERANCE:
            reasons.append(
                f"  canary accuracy is rising {canary_slope:.3f}/1k steps against the stream's "
                f"{stream_slope:.3f} over {len(readings)} readings: a static set being memorised"
            )
    return reasons
