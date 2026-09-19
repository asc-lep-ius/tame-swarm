"""The margins #42 regulates against, and the three ways #41 stops them being farmed.

The metric tests carry their arithmetic in the assertion rather than in a golden
number: a calibration error that agrees with a reimplementation of itself agrees
with its own bug too.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from evaluation import build_rotating_stream
from mob import apply_mob_to_model
from viability_margins import (
    CALLER_CORE,
    CALLER_TRAINER,
    CANARY_DIVERGENCE_TOLERANCE,
    EVALUATION_COST_PER_ITEM,
    EVALUATION_REASON,
    MIN_CANARIES,
    CanaryReading,
    FrozenBase,
    MetricSet,
    Predictions,
    ViabilityError,
    ViabilityMargins,
    accuracy,
    brier_score,
    charge_evaluation,
    expected_calibration_error,
    farming_reasons,
    measure_viability,
    risk_coverage_auc,
    score_predictions,
)

from .conftest import TINY_VOCAB_SIZE, build_tiny_causal_lm
from .rotating_fixtures import (
    CANARY_COUNT,
    CUTOFF,
    MAX_SEQ_LENGTH,
    TODAY,
    canary_manifest,
    stream_manifest,
)

CPU = torch.device("cpu")


def _predictions(confidence, correct, canary=None):
    return Predictions(
        confidence=torch.tensor(confidence, dtype=torch.float32),
        correct=torch.tensor(correct, dtype=torch.bool),
        is_canary=torch.tensor(
            canary if canary is not None else [False] * len(confidence), dtype=torch.bool
        ),
    )


# --- The metrics ---------------------------------------------------------------------------

# Two confident predictions, one of them wrong, and two diffident ones that are also
# wrong: a model that is overconfident where it is sure and underconfident where it
# is not, which is the shape every number below is readable against.
MIXED = _predictions([0.9, 0.9, 0.1, 0.1], [True, False, False, False])


def test_accuracy_is_the_share_of_items_answered():
    assert accuracy(MIXED) == pytest.approx(0.25)


def test_calibration_error_weights_each_bin_by_how_many_items_fall_in_it():
    """0.9 and 0.1 land in different fifteenths; the gaps are 0.4 and 0.1, two items each."""
    assert expected_calibration_error(MIXED) == pytest.approx((2 * 0.4 + 2 * 0.1) / 4)


def test_a_perfectly_calibrated_model_has_no_calibration_error():
    """The pairing: half of the items at 0.5 confidence, half of them right."""
    calibrated = _predictions([0.5] * 4, [True, False, True, False])

    assert expected_calibration_error(calibrated) == pytest.approx(0.0)


def test_brier_is_the_squared_error_of_the_confidence():
    assert brier_score(MIXED) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)


def test_risk_coverage_separates_two_models_of_identical_accuracy():
    """What selective prediction reads: not whether it is right, but whether it knows.

    Both models get half their items right. The first is confident exactly where it
    is correct, so declining the least confident declines the wrong ones; the second
    is confident exactly where it is wrong. Accuracy cannot tell them apart and this
    is the metric that exists to.
    """
    ordered = _predictions([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    inverted = _predictions([0.9, 0.8, 0.2, 0.1], [False, False, True, True])

    assert accuracy(ordered) == accuracy(inverted)
    assert risk_coverage_auc(ordered) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert risk_coverage_auc(inverted) == pytest.approx((1 + 1 + 2 / 3 + 0.5) / 4)
    assert risk_coverage_auc(ordered) < risk_coverage_auc(inverted)


def test_no_items_is_refused_rather_than_read_as_a_zero_margin():
    with pytest.raises(ViabilityError, match="No items"):
        MetricSet.of(_predictions([], []))


def test_the_margins_are_signed_so_positive_is_the_organism_ahead():
    """Two of the four metrics are better when smaller; the sign is not left to a comment."""
    model = MetricSet(accuracy=0.7, calibration_error=0.10, brier=0.20, risk_coverage_auc=0.30)
    base = MetricSet(accuracy=0.5, calibration_error=0.15, brier=0.25, risk_coverage_auc=0.40)

    margins = ViabilityMargins.between(model, base)

    assert margins.accuracy == pytest.approx(0.2)
    assert margins.calibration == pytest.approx(0.05)
    assert margins.brier == pytest.approx(0.05)
    assert margins.selective == pytest.approx(0.10)


def test_predictions_split_into_the_half_that_rotates_and_the_half_that_does_not():
    predictions = _predictions([0.9, 0.8, 0.7], [True, False, True], canary=[False, True, False])

    assert len(predictions.stream) == 2
    assert len(predictions.canaries) == 1
    assert bool(predictions.canaries.correct[0]) is False


# --- The frozen copy of the base --------------------------------------------------------


def test_freezing_puts_a_module_beyond_training():
    base = FrozenBase.freeze(build_tiny_causal_lm())

    assert not base.module.training
    assert all(not tensor.requires_grad for tensor in base.module.parameters())
    base.assert_read_only()


def test_a_module_something_else_is_training_is_refused():
    """The constructor checks rather than fixes: freezing the wrong module is worse."""
    with pytest.raises(ViabilityError, match="requiring a gradient"):
        FrozenBase(build_tiny_causal_lm())


def test_a_module_left_in_training_mode_is_refused():
    module = build_tiny_causal_lm().requires_grad_(False)
    module.train()

    with pytest.raises(ViabilityError, match="training mode"):
        FrozenBase(module)


def test_the_organism_is_refused_as_its_own_base(tiny_mob_config):
    """A converted model is the organism; the base is what it is measured against."""
    organism = apply_mob_to_model(build_tiny_causal_lm(), tiny_mob_config, layers_to_modify=[1, 2])

    with pytest.raises(ViabilityError, match="MoB layers"):
        FrozenBase.freeze(organism)


def test_a_base_that_shares_a_parameter_with_the_model_is_refused():
    """A view of the organism wearing a copy's name: the margin is wrong only there."""
    model = build_tiny_causal_lm()
    base = FrozenBase.freeze(deepcopy(model))
    base.module.lm_head.weight = model.lm_head.weight

    with pytest.raises(ViabilityError, match="shares parameter storage"):
        base.assert_separate_from(model)


def test_a_base_made_trainable_again_is_caught_after_the_pass():
    """``requires_grad`` is a property anything holding the module can set back."""
    base = FrozenBase.freeze(build_tiny_causal_lm())
    base.module.requires_grad_(True)

    with pytest.raises(ViabilityError, match="received a gradient or was made trainable"):
        base.assert_read_only()


# --- Evaluation as a budgeted action ----------------------------------------------------


class RecordingLedger:
    """The one method #43 has to offer, with a note of everything it was asked for."""

    def __init__(self, balance: float = 100.0) -> None:
        self.balance = balance
        self.debits: list[tuple[float, str]] = []

    def debit(self, amount: float, reason: str) -> float:
        self.debits.append((amount, reason))
        self.balance -= amount
        return self.balance


class RefusingLedger(RecordingLedger):
    def debit(self, amount: float, reason: str) -> float:
        raise ViabilityError("out of budget")


def test_the_core_pays_per_item_it_asked_to_have_evaluated():
    ledger = RecordingLedger()

    cost = charge_evaluation(CALLER_CORE, ledger, num_items=20)

    assert cost == pytest.approx(20 * EVALUATION_COST_PER_ITEM)
    assert ledger.debits == [(cost, EVALUATION_REASON)]
    assert ledger.balance == pytest.approx(100.0 - cost)


def test_the_trainers_recorded_measurement_costs_the_organism_nothing():
    assert charge_evaluation(CALLER_TRAINER, None, num_items=20) == 0.0


def test_a_trainer_handed_a_ledger_is_refused():
    """Otherwise the number the experiment reads is a number the experiment caused."""
    with pytest.raises(ViabilityError, match="recorded measurement does not cost"):
        charge_evaluation(CALLER_TRAINER, RecordingLedger(), num_items=20)


def test_a_core_with_no_ledger_is_refused():
    """An evaluation that costs nothing is an unbounded action."""
    with pytest.raises(ViabilityError, match="no budget to charge it to"):
        charge_evaluation(CALLER_CORE, None, num_items=20)


def test_an_unknown_caller_is_refused():
    with pytest.raises(ViabilityError, match="Unknown evaluation caller"):
        charge_evaluation("operator", None, num_items=20)


# --- The reading, end to end ---------------------------------------------------------------


@pytest.fixture
def rotation(fake_tokenizer):
    return build_rotating_stream(
        stream_manifest(),
        canary_manifest(),
        fake_tokenizer,
        MAX_SEQ_LENGTH,
        cutoff=CUTOFF,
        today=TODAY,
    )


@pytest.fixture
def organism_and_base(tiny_mob_config):
    torch.manual_seed(0)
    pristine = build_tiny_causal_lm()
    base = FrozenBase.freeze(deepcopy(pristine))
    return apply_mob_to_model(pristine, tiny_mob_config, layers_to_modify=[1, 2]), base


def test_a_trainers_reading_carries_every_margin_and_costs_nothing(organism_and_base, rotation):
    model, base = organism_and_base

    reading = measure_viability(
        model, base, rotation, batch_size=8, device=CPU, caller=CALLER_TRAINER, step=100
    )

    assert reading.cost == 0.0
    assert reading.rotation == rotation.rotation
    assert reading.canary.num_canaries == CANARY_COUNT
    assert reading.canary.num_stream_items == rotation.num_items - CANARY_COUNT
    assert set(reading.as_metrics()) == {
        "viability/accuracy_margin",
        "viability/calibration_margin",
        "viability/brier_margin",
        "viability/selective_margin",
        "viability/stream_accuracy",
        "viability/canary_accuracy",
        "viability/cost",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in reading.as_metrics().values())


def test_the_cores_reading_debits_the_budget_for_every_item(organism_and_base, rotation):
    model, base = organism_and_base
    ledger = RecordingLedger()

    reading = measure_viability(
        model,
        base,
        rotation,
        batch_size=8,
        device=CPU,
        caller=CALLER_CORE,
        ledger=ledger,
    )

    assert reading.cost == pytest.approx(rotation.num_items * EVALUATION_COST_PER_ITEM)
    assert ledger.debits == [(reading.cost, EVALUATION_REASON)]


def test_the_margins_are_read_on_the_stream_and_not_on_the_canaries(organism_and_base, rotation):
    """An item the organism may have memorised belongs in the farming check, not the margin."""
    model, base = organism_and_base

    reading = measure_viability(
        model, base, rotation, batch_size=8, device=CPU, caller=CALLER_TRAINER
    )
    model_predictions = score_predictions(model, rotation, 8, CPU)

    assert reading.model_metrics == MetricSet.of(model_predictions.stream)
    assert reading.canary.stream_accuracy == accuracy(model_predictions.stream)
    assert reading.canary.canary_accuracy == accuracy(model_predictions.canaries)


def test_an_evaluation_the_ledger_refuses_does_not_happen(organism_and_base, rotation):
    """Charged before the forward: an organism out of budget takes no extra reading."""
    model, base = organism_and_base
    counted = CountingModel(model)

    with pytest.raises(ViabilityError, match="out of budget"):
        measure_viability(
            counted,
            base,
            rotation,
            batch_size=8,
            device=CPU,
            caller=CALLER_CORE,
            ledger=RefusingLedger(),
        )

    assert counted.calls == 0


def test_too_few_canaries_to_read_the_tolerance_is_refused(fake_tokenizer, organism_and_base):
    model, base = organism_and_base
    thin = build_rotating_stream(
        stream_manifest(),
        canary_manifest(count=MIN_CANARIES - 1),
        fake_tokenizer,
        MAX_SEQ_LENGTH,
        cutoff=CUTOFF,
        today=TODAY,
    )

    with pytest.raises(ViabilityError, match="below the 8"):
        measure_viability(model, base, thin, batch_size=8, device=CPU, caller=CALLER_TRAINER)


def test_scoring_leaves_the_trainer_in_the_mode_it_found_it(organism_and_base, rotation):
    model, base = organism_and_base
    model.train()

    score_predictions(model, rotation, 8, CPU)

    assert model.training


# --- A planted farmer ------------------------------------------------------------------------


class CountingModel(nn.Module):
    """Wraps a model and counts the forwards, so "did not happen" is checkable."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return self.inner(**kwargs)


class ScriptedModel(nn.Module):
    """A model whose answer to every item is decided in advance.

    The planted farmer of the acceptance criterion: it answers the first
    ``right_on_canaries`` canaries correctly and the first ``right_on_stream``
    stream items, and nothing else. Raising the first while holding the second is
    memorisation of a static set, which is the thing the detector has to see.
    """

    PEAK = 8.0

    def __init__(self, stream, right_on_canaries: int, right_on_stream: int) -> None:
        super().__init__()
        self.answers: dict[tuple[int, ...], int] = {}
        canary_rows = [row for row in range(stream.num_items) if bool(stream.is_canary[row])]
        stream_rows = [row for row in range(stream.num_items) if not bool(stream.is_canary[row])]
        for rows, budget in ((canary_rows, right_on_canaries), (stream_rows, right_on_stream)):
            for rank, row in enumerate(rows):
                key = tuple(int(token) for token in stream.input_ids[row])
                # 0 is the pad id and never an answer, so it is reliably wrong.
                self.answers[key] = int(stream.answer_ids[row]) if rank < budget else 0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, TINY_VOCAB_SIZE)
        for row in range(batch):
            answer = self.answers[tuple(int(token) for token in input_ids[row])]
            logits[row, :, answer] = self.PEAK
        return SimpleNamespace(logits=logits)


def _readings(rotation, base, schedule):
    return [
        measure_viability(
            ScriptedModel(rotation, canaries, stream_items),
            base,
            rotation,
            batch_size=8,
            device=CPU,
            caller=CALLER_TRAINER,
            step=step,
        ).canary
        for step, canaries, stream_items in schedule
    ]


def test_a_planted_farmer_is_read_as_farming(rotation, organism_and_base):
    """Every canary right and no stream item: the level, at one reading."""
    _, base = organism_and_base

    readings = _readings(rotation, base, [(1000, CANARY_COUNT, 0)])

    assert readings[0].canary_accuracy == 1.0
    assert readings[0].stream_accuracy == 0.0
    assert readings[0].divergence > CANARY_DIVERGENCE_TOLERANCE
    assert "stands 1.000 above stream accuracy" in "\n".join(farming_reasons(readings))


def test_a_farmer_memorising_a_static_set_is_read_from_the_trend(rotation, organism_and_base):
    """The second shape: canary accuracy climbing with step count, the stream's flat."""
    _, base = organism_and_base

    readings = _readings(rotation, base, [(0, 2, 0), (500, 5, 0), (1000, CANARY_COUNT, 0)])

    assert [reading.canary_accuracy for reading in readings] == [0.25, 0.625, 1.0]
    reasons = farming_reasons(readings)
    assert any("a static set being memorised" in reason for reason in reasons)


def test_an_organism_learning_the_task_is_not_read_as_farming(rotation, organism_and_base):
    """The pairing. Both halves improve together, so neither shape fires."""
    _, base = organism_and_base
    third = rotation.num_items - CANARY_COUNT

    readings = _readings(
        rotation,
        base,
        [(0, 2, third // 4), (500, 5, third // 2), (1000, CANARY_COUNT, third)],
    )

    assert readings[-1].stream_accuracy == 1.0
    assert farming_reasons(readings) == []


def test_a_single_reading_cannot_fire_the_trend():
    """Two readings through a rising line is a line through two points."""
    rising = [
        CanaryReading(
            step=0, canary_accuracy=0.1, stream_accuracy=0.1, num_canaries=8, num_stream_items=12
        ),
        CanaryReading(
            step=1000, canary_accuracy=0.2, stream_accuracy=0.1, num_canaries=8, num_stream_items=12
        ),
    ]

    assert farming_reasons(rising[:1]) == []
    assert farming_reasons(rising) == []
    assert farming_reasons([]) == []
