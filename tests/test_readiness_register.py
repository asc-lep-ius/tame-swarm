"""#46's register, checked from the module rather than from the prose.

Two halves. The **human-channel rule**: no action available to the tissue can
change its budget through a human's decision, which is asserted by enumerating
the action space out of ``tame/readiness.py`` -- never a list copied out of
``docs/readiness-register.md``, because a rule checked against a copy of itself
checks nothing. And the **register's own integrity**: evidence that cites
something re-readable, autonomies that are not yet granted carrying no flag to
flip, and the document naming exactly what the module does.

The gate flags' own two properties -- off by default, and in ``ArmFingerprint``
under the same name -- are in ``tests/test_no_silent_noops.py`` with the rest of
the silent-no-op family, which is the defect class a gate that is on by default
belongs to.
"""

import re
from dataclasses import fields, replace
from pathlib import Path

import pytest

from parity import ArmFingerprint, fingerprint_arm
from readiness import (
    AUTONOMIES,
    CORE_ACTIONS,
    OPERATOR,
    PHASE_NOT_YET_GRANTED,
    PHASE_SCHEDULED,
    PREREGISTERED_SIGNATURES,
    READABLE_CHANNELS,
    CoreAction,
    ReadinessConfig,
    granted_flags,
    human_channel_violations,
)
from train import TrainingConfig

REPOSITORY = Path(__file__).resolve().parent.parent
REGISTER = REPOSITORY / "docs" / "readiness-register.md"

SCHEDULED = tuple(a for a in AUTONOMIES if a.phase == PHASE_SCHEDULED)
NOT_YET_GRANTED = tuple(a for a in AUTONOMIES if a.phase == PHASE_NOT_YET_GRANTED)

# --- The human channel ----------------------------------------------------------------


@pytest.mark.parametrize("action", CORE_ACTIONS, ids=lambda action: action.name)
def test_no_action_has_the_operator_as_an_input(action):
    """The rule in its plainest form, over every action the tissue has.

    Not "the operator is unlikely to be persuaded" -- there is no channel to
    persuade down. Every input is a reading of the organism or its ledger.
    """
    assert OPERATOR not in action.inputs, (
        f"{action.name!r} takes the operator as an input: a human's decision would reach the "
        "budget through it, which is the one channel the register closes"
    )
    assert set(action.inputs) <= READABLE_CHANNELS


def test_the_whole_action_space_satisfies_the_rule():
    assert human_channel_violations() == []


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        pytest.param({"inputs": (OPERATOR,)}, "reads the operator", id="operator-as-input"),
        pytest.param({"inputs": ("mood",)}, "undeclared channels", id="undeclared-channel"),
        pytest.param({"writes": "budget"}, "writes 'budget'", id="writes-the-budget"),
        pytest.param({"gated_by": "autonomy_spawn"}, "not a granted", id="ungated-action"),
    ],
)
def test_the_check_fails_on_an_action_that_breaks_the_rule(corruption, expected):
    """The pairing every test in this repository owes: the inert state it fails in.

    A check that passes over the action space as built and cannot fail is a
    check that would pass over an action space with a human in it.
    """
    violations = human_channel_violations((replace(CORE_ACTIONS[0], **corruption),))

    assert len(violations) == 1 and expected in violations[0], violations


def test_an_action_reading_the_budget_is_still_allowed_to_be_charged_for_it():
    """Reading the ledger is not writing it; asking costs budget because the ledger charges."""
    asking = next(action for action in CORE_ACTIONS if action.name == "request an evaluation")

    assert "budget_reading" in asking.inputs
    assert asking.writes == "evaluation_queue"
    assert human_channel_violations((asking,)) == []


@pytest.mark.parametrize("flag", granted_flags())
def test_every_granted_autonomy_has_an_action_that_exercises_it(flag):
    """An autonomy in the register that no action carries is a grant nobody can use."""
    assert any(action.gated_by == flag for action in CORE_ACTIONS)


def test_no_action_writes_a_flag():
    """The tissue does not grant itself anything; the operator does, in the issue."""
    assert not set(granted_flags()) & {action.writes for action in CORE_ACTIONS}


# --- The register's own integrity --------------------------------------------------------


@pytest.mark.parametrize("autonomy", SCHEDULED, ids=lambda autonomy: autonomy.name)
def test_the_evidence_is_a_citation_and_the_citation_resolves(autonomy):
    """A preregistered signature or a named constitution test -- and the file must exist.

    Evidence nobody can re-read is not evidence. The signature IDs are
    ``docs/preregistration.md`` section 1 rows 1-4; the tests are the constitution
    group #38 built, and a citation that has been renamed away fails here rather
    than at the moment somebody reads the register to decide a grant.
    """
    assert autonomy.evidence, f"{autonomy.name} is scheduled with no evidence required"

    for citation in autonomy.evidence:
        if citation in PREREGISTERED_SIGNATURES:
            continue
        assert citation.startswith("tests/constitution/"), (
            f"{autonomy.name} cites {citation!r}, which is neither a preregistration "
            "signature nor a test under tests/constitution/"
        )
        assert (REPOSITORY / citation).exists(), f"{autonomy.name} cites a missing {citation}"


@pytest.mark.parametrize("autonomy", NOT_YET_GRANTED, ids=lambda autonomy: autonomy.name)
def test_a_phase_three_autonomy_carries_no_flag_to_flip(autonomy):
    """Listed, not granted: a flag that exists is a flag somebody can turn on."""
    assert autonomy.flag is None

    declared = {spec.name for spec in fields(ReadinessConfig)}
    fingerprinted = {
        spec.name for spec in fields(ArmFingerprint) if spec.name.startswith("autonomy_")
    }
    assert declared == fingerprinted == set(granted_flags()), (
        "a flag exists for an autonomy the register does not grant; the config, the fingerprint "
        "and the granted rows must name the same set"
    )


def test_the_document_names_what_the_module_declares():
    """The prose is the register a person reads; drift between it and the module is the defect.

    Names, flags and citations only -- the thresholds beside them are prose, and
    the module deliberately does not carry a copy to compare against.
    """
    text = REGISTER.read_text(encoding="utf-8")

    for autonomy in AUTONOMIES:
        assert autonomy.name.lower() in text.lower(), f"{autonomy.name} is not in the register"
        for citation in autonomy.evidence:
            assert citation in text, (
                f"{autonomy.name}'s evidence {citation!r} is not in the register"
            )

    assert set(re.findall(r"autonomy_[a-z_]+", text)) == set(granted_flags())


# --- What a run records ---------------------------------------------------------------------


def test_a_run_that_grants_nothing_fingerprints_as_granting_nothing():
    arm = fingerprint_arm(
        TrainingConfig(), eval_split_fingerprint="s", data_order="d", converted_layers=1
    )

    assert [getattr(arm, flag) for flag in granted_flags()] == [False] * len(granted_flags())


@pytest.mark.parametrize("flag", granted_flags())
def test_a_granted_autonomy_reaches_the_fingerprint(flag):
    """The wiring #42 threads: what the config says is what the arm is recorded as."""
    arm = fingerprint_arm(
        TrainingConfig(),
        eval_split_fingerprint="s",
        data_order="d",
        converted_layers=1,
        readiness=ReadinessConfig(**{flag: True}),
    )

    assert getattr(arm, flag) is True
    assert sum(getattr(arm, other) for other in granted_flags()) == 1


def test_the_action_space_is_a_tuple_of_actions_and_not_a_list_from_the_markdown():
    """#42 imports this; the shape it imports is pinned so a later edit cannot loosen it."""
    assert isinstance(CORE_ACTIONS, tuple) and CORE_ACTIONS
    assert all(isinstance(action, CoreAction) for action in CORE_ACTIONS)
