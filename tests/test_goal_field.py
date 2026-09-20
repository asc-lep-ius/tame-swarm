"""The goal field on the body (#54): the setpoint, where it attaches, and what it refuses.

``tests/test_stakes_dial.py`` pins what the goal term computes; this pins the
wiring between the homeostat's calibration and the layers that get paid. Every
failure it names is a run that trains to completion and reports a goal term of
exactly zero, which is the answer #39 was launched to test.

The dose's own no-op -- a positive dose that changes nothing, a zero dose that
changes something -- is in ``tests/test_no_silent_noops.py`` with the rest of
that class.
"""

import pytest
import torch

from goal_field import (
    GOAL_ERROR_UNIT_COST,
    attach_goal_fields,
    parse_goal_doses,
    paying_layers,
)
from mob import mob_layers_by_index
from steering_pipeline import UncertifiedDirectionError, certified_coupling_layers
from train import TrainingConfig

from .goal_field_fixtures import (
    BODY_LAYERS,
    CALIBRATION_PASSAGES,
    REFERENCE_STRENGTH,
    body_calibration,
    paid_body,
    settle_body,
)

# The preregistered pair and the doses section 7 fixes: ``safe`` held at the
# reference, ``truthful`` swept, here at the top of the two levels.
PREREGISTERED = dict(
    goal_fields=("truthful", "safe"),
    goal_doses=(4 * GOAL_ERROR_UNIT_COST, GOAL_ERROR_UNIT_COST),
)
CONVERTED_RANGE = dict(mob_layers_start=6, mob_layers_end=22)


# --- The setpoint is the tissue's own, in the units the goal term reads ------------------


def test_the_setpoint_in_projection_units_is_the_inverse_of_the_z_score():
    """``setpoint_projection`` and ``z`` are one mapping read in two directions.

    The homeostat regulates in sigma and the goal term prices a raw projection, so
    a conversion that drifted from ``z``'s would give the cells a target their own
    controller does not hold -- silently, since both numbers look plausible.
    """
    calibration = body_calibration()

    for layer in BODY_LAYERS:
        projection = calibration.setpoint_projection(layer)
        assert calibration.z(layer, projection) == pytest.approx(calibration.setpoint_z(layer))
        # And the closed form the record is described by: where the cell sits with
        # every actuator injecting at the reference strength.
        cell = calibration.layers[layer]
        assert projection == pytest.approx(cell.resting_mean + cell.lift * REFERENCE_STRENGTH)


def test_a_cell_the_effort_does_not_move_is_asked_to_hold_where_it_already_rests():
    """Zero lift is the bottom actuator's measured state, not a missing measurement."""
    calibration = body_calibration(lift=0.0)

    for layer in BODY_LAYERS:
        assert calibration.setpoint_projection(layer) == pytest.approx(
            calibration.layers[layer].resting_mean
        )


# --- Where a goal is paid ----------------------------------------------------------------


def test_a_goal_is_paid_only_where_it_is_both_calibrated_and_converted():
    calibration = body_calibration(layers=(1, 2, 3))

    assert paying_layers(calibration, [1, 2]) == (1, 2)
    assert paying_layers(calibration, [2, 3, 9]) == (2, 3)
    assert paying_layers(calibration, [0, 9]) == ()


def test_attaching_puts_one_field_per_paying_layer_at_that_layer_s_own_setpoint():
    calibration = body_calibration()
    model = paid_body(("truthful",), (GOAL_ERROR_UNIT_COST,), {"truthful": calibration})

    records = {
        record.layer: record
        for record in attach_goal_fields(
            model, ("truthful",), (GOAL_ERROR_UNIT_COST,), {"truthful": calibration}
        )
    }
    assert sorted(records) == list(BODY_LAYERS)
    for layer, mob in mob_layers_by_index(model).items():
        assert len(mob.goal_fields) == 1
        field = mob.goal_fields[0]
        assert field.dose == GOAL_ERROR_UNIT_COST
        assert field.setpoint == pytest.approx(calibration.setpoint_projection(layer))
        assert torch.equal(field.vector, calibration.directions[layer])
        assert records[layer].num_passages == CALIBRATION_PASSAGES
        assert records[layer].reference_strength == REFERENCE_STRENGTH


def test_two_goals_certified_at_one_layer_both_pay_there():
    """``truthful`` and ``safe`` share layer 18 on the real model; the term sums them."""
    calibrations = {"truthful": body_calibration(), "safe": body_calibration(layers=(2,), seed=11)}
    model = paid_body(("truthful", "safe"), (0.068, 0.017), calibrations)

    layers = mob_layers_by_index(model)
    assert [field.dose for field in layers[1].goal_fields] == [0.068]
    assert [field.dose for field in layers[2].goal_fields] == [0.068, 0.017]


def test_attaching_twice_replaces_rather_than_doubles_the_dose():
    """Append-only on the layer, so a second setup would otherwise pay twice over."""
    calibration = body_calibration()
    model = paid_body(("truthful",), (0.017,), {"truthful": calibration})

    attach_goal_fields(model, ("truthful",), (0.017,), {"truthful": calibration})

    for mob in mob_layers_by_index(model).values():
        assert [field.dose for field in mob.goal_fields] == [0.017]


def test_a_goal_calibrated_at_no_converted_layer_is_refused_rather_than_attached_to_nothing():
    calibration = body_calibration(layers=(3,))

    with pytest.raises(ValueError, match="no cell could be paid"):
        paid_body(("truthful",), (0.017,), {"truthful": calibration})


def test_the_preregistered_pair_pays_at_the_layers_the_preregistration_names():
    """Section 7 claims ``truthful`` and ``safe`` are both certified inside 6:22 and share 18.

    That claim decides the whole run -- the second field, the shared layer, the
    dose ratio being a ratio of two fields the same cells feel -- and it is a fact
    about ``contrastive_data.CERTIFIED``, which a later certification could change
    without anyone rereading the document.
    """
    converted = list(range(6, 22))
    paid = {
        goal: paying_layers(body_calibration(layers=certified_coupling_layers(goal)), converted)
        for goal in ("truthful", "safe")
    }

    assert paid["truthful"] == (13, 16, 17, 18, 19, 20, 21)
    assert paid["safe"] == (14, 18)
    assert set(paid["truthful"]) & set(paid["safe"]) == {18}


@pytest.mark.gpu
def test_a_direction_calibrated_on_the_cpu_reaches_the_device_its_layer_is_on():
    """The failure this catches costs an arm, not a test.

    The calibration is measured before the model is placed and its directions are
    CPU tensors; the contributions the term multiplies them by are wherever the
    layer ended up. Nothing on CPU can see the mismatch, and on the GPU it is a
    ``RuntimeError`` raised in the first settlement of a nine-GPU-hour run.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    calibration = body_calibration()

    model = paid_body(
        ("truthful",), (GOAL_ERROR_UNIT_COST,), {"truthful": calibration}, device="cuda"
    )

    for mob in mob_layers_by_index(model).values():
        assert mob.goal_fields[0].vector.device.type == "cuda"
    values = settle_body(model)
    assert all(value.isfinite().all() for value in values)


# --- The flag's own grammar ---------------------------------------------------------------


def test_the_doses_parse_per_goal_repeated_or_comma_separated():
    assert parse_goal_doses(["truthful=0.068,safe=0.017"]) == (
        ("truthful", "safe"),
        (0.068, 0.017),
    )
    assert parse_goal_doses(["truthful=0.068", "safe=0.017"]) == (
        ("truthful", "safe"),
        (0.068, 0.017),
    )
    assert parse_goal_doses([]) == ((), ())


def test_a_bare_dose_and_a_dose_that_is_not_a_number_are_both_refused():
    with pytest.raises(ValueError, match="<goal>=<dose>"):
        parse_goal_doses(["0.017"])
    with pytest.raises(ValueError, match="not a number"):
        parse_goal_doses(["truthful=a lot"])


# --- What the config refuses, at construction ---------------------------------------------


def test_the_preregistered_pair_is_accepted_over_the_converted_range():
    """The pairing for the refusals below: the run this issue wires is a valid config."""
    config = TrainingConfig(**CONVERTED_RANGE, **PREREGISTERED)

    assert config.goal_fields == ("truthful", "safe")
    assert config.goal_doses == (4 * GOAL_ERROR_UNIT_COST, GOAL_ERROR_UNIT_COST)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(goal_fields=("truthful",), goal_doses=()), "same length"),
        (dict(goal_fields=("truthful", "truthful"), goal_doses=(0.1, 0.1)), "names a goal twice"),
        (dict(goal_fields=("truthful",), goal_doses=(-0.1,)), "must be >= 0"),
        (dict(goal_fields=("truthful",), goal_doses=(0.1,), router="dense"), "dense arm has none"),
    ],
)
def test_a_goal_field_that_could_not_pay_a_cell_is_refused_at_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**{**CONVERTED_RANGE, **kwargs})


def test_an_uncertified_goal_is_refused_the_way_the_coupling_refuses_it():
    with pytest.raises(UncertifiedDirectionError, match="no certified layers"):
        TrainingConfig(**CONVERTED_RANGE, goal_fields=("deliberation",), goal_doses=(0.017,))


def test_a_certified_goal_outside_the_converted_range_is_refused():
    """``safe`` is certified at 14, 18 and 22; a range that covers none of them pays nobody."""
    with pytest.raises(ValueError, match="none inside the converted range"):
        TrainingConfig(
            mob_layers_start=0, mob_layers_end=6, goal_fields=("safe",), goal_doses=(0.017,)
        )
