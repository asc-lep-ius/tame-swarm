"""The one metric on the surface that can say "steering did not help" (#5).

The arithmetic and the arm discipline are what is tested here; the log-odds
themselves are the behavioural gate's own measurement and are tested where that
lives (``tests/test_behavioural_validation.py``, and on the substrate in
``tests/test_real_model.py``). What matters is that the arms differ in exactly one
thing, that the hooks come back whatever happens, that the economy is not paid,
and that the floor comparison is reachable in the direction that falsifies.
"""

from dataclasses import replace
from unittest import mock

import pytest

import outcome_probe
from contrastive_data import CERTIFIED
from models import OutcomeMetrics

from .wired_system import ACTUATORS, build_wired_app

BASE_STRENGTH = 2.0

SAFE_RANDOM_MAX = CERTIFIED["safe"].random_max


def _probe_with(tame, scripted: list[list[float]], num_pairs: int):
    """Run the probe with each arm's log-odds scripted, in the order the arms run."""
    queue = list(scripted)
    with mock.patch.object(outcome_probe, "held_out_log_odds", lambda *a, **k: queue.pop(0)):
        return outcome_probe.probe_outcome(tame, num_pairs=num_pairs)


@pytest.fixture
def arms(monkeypatch):
    """Scripted per-arm log-odds, in the order ``probe_outcome`` runs the arms."""
    scripted: list[list[float]] = []
    calls: list[int] = []

    def fake(model, tokenizer, pairs, device, max_length=128):
        calls.append(len(pairs))
        return scripted.pop(0)

    monkeypatch.setattr(outcome_probe, "held_out_log_odds", fake)
    return scripted, calls


def _probe_app(**kwargs):
    """The served default is a constant strength (``app.ADAPTIVE_STEERING``), so is this."""
    kwargs.setdefault("adaptive", False)
    tame, system = build_wired_app(**kwargs)
    tame.homeostat.homeostat.goal = "safe"  # pyright: ignore[reportOptionalMemberAccess]
    return tame, system


@pytest.fixture
def floor_applies():
    """Say ``safe`` was certified at the configuration this fixture actually serves.

    The alternative -- rewriting the app's steering config to the real
    certification's layers -- would leave the tissue holding vectors at layers the
    config no longer names. What is under test is the arithmetic and the arm
    discipline, so the record moves and the organism stays coherent.
    """
    served = replace(
        CERTIFIED["safe"], model="tiny", layers=tuple(ACTUATORS), strength=BASE_STRENGTH
    )
    with mock.patch.dict(CERTIFIED, {"safe": served}):
        yield served


def test_the_probe_reports_the_delta_against_the_certifications_random_floor(arms, floor_applies):
    scripted, calls = arms
    # served clears the floor; unsteered does not.
    scripted.extend([[1.0, 1.0, 1.0, -1.0], [0.2, 0.2, -0.2, -0.2]])
    tame, _ = _probe_app()

    outcome = outcome_probe.probe_outcome(tame, num_pairs=4)

    assert isinstance(outcome, OutcomeMetrics)
    assert outcome.goal == "safe" and outcome.num_pairs == 4
    assert set(outcome.arms) == {"served", "unsteered"}
    assert outcome.arms["served"].mean_log_odds == pytest.approx(0.5)
    assert outcome.arms["unsteered"].mean_log_odds == pytest.approx(0.0)
    assert outcome.served_minus_unsteered_log_odds == pytest.approx(0.5)
    assert outcome.arms["served"].accuracy == pytest.approx(0.75)
    assert outcome.served_minus_unsteered_accuracy == pytest.approx(0.25)
    assert outcome.certified_random_max == SAFE_RANDOM_MAX
    assert outcome.beats_random is True and outcome.floor_not_applicable is None
    assert outcome.served_minus_unsteered_standard_error is not None
    assert calls == [4, 4], "every arm reads the same pairs"
    assert tame.outcome is outcome, "the probe caches; the GET is a read"


def test_a_delta_below_the_random_floor_reads_as_did_not_help(arms, floor_applies):
    """The falsifying value, reached: this is what #5's last criterion asks to exist."""
    scripted, _ = arms
    scripted.extend([[0.10, 0.10], [0.08, 0.08]])
    tame, _ = _probe_app()

    outcome = outcome_probe.probe_outcome(tame, num_pairs=2)

    assert outcome.served_minus_unsteered_log_odds == pytest.approx(0.02)
    assert SAFE_RANDOM_MAX is not None and outcome.served_minus_unsteered_log_odds < SAFE_RANDOM_MAX
    assert outcome.beats_random is False


def test_the_adaptive_loop_is_measured_against_its_own_constant_arm(monkeypatch):
    """#4's value test as a live contrast: is the loop worth anything over the fixed strength?

    The arm's defining property is that the loop is *off* while it is scored, so
    that is what is asserted -- by recording the mode each arm actually ran under.
    Asserting the arm's ``mean_strength`` instead would pass whether or not the
    implementation flipped anything: with the log-odds monkeypatched no forward
    runs, the strength history is empty, and the reported strength falls back to
    the reference either way.
    """
    tame, _ = _probe_app(adaptive=True)
    assert tame.steering_config.adaptive
    scripted = [[0.6, 0.6], [0.1, 0.1], [0.5, 0.5]]
    modes: list[bool] = []

    def fake(*_args, **_kwargs):
        modes.append(tame.steering_config.adaptive)
        return scripted.pop(0)

    monkeypatch.setattr(outcome_probe, "held_out_log_odds", fake)
    outcome = outcome_probe.probe_outcome(tame, num_pairs=2)

    assert set(outcome.arms) == {"served", "unsteered", "constant"}
    assert modes == [True, True, False], "only the constant arm runs with the loop off"
    assert outcome.adaptive_minus_constant_log_odds == pytest.approx(0.1)
    assert tame.steering_config.adaptive is True, "the arm restores the served configuration"


def test_the_unsteered_arm_reports_no_strength_and_leaves_the_hooks_attached(arms):
    scripted, _ = arms
    scripted.extend([[0.4, 0.4], [0.1, 0.1]])
    tame, _ = _probe_app()
    attached_before = len(tame.homeostat._registered_hooks)  # pyright: ignore[reportOptionalMemberAccess]

    outcome = outcome_probe.probe_outcome(tame, num_pairs=2)

    assert outcome.arms["unsteered"].mean_strength is None
    assert outcome.arms["served"].mean_strength is not None
    assert len(tame.homeostat._registered_hooks) == attached_before  # pyright: ignore[reportOptionalMemberAccess]


def test_an_arm_that_raises_still_leaves_the_server_steered(monkeypatch):
    """A probe is not allowed to leave the process serving an unsteered model."""
    tame, _ = _probe_app()
    attached_before = len(tame.homeostat._registered_hooks)  # pyright: ignore[reportOptionalMemberAccess]
    calls = {"n": 0}

    def explodes_on_the_unsteered_arm(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the second arm failed")
        return [0.4, 0.4]

    monkeypatch.setattr(outcome_probe, "held_out_log_odds", explodes_on_the_unsteered_arm)
    with pytest.raises(RuntimeError, match="second arm"):
        outcome_probe.probe_outcome(tame, num_pairs=2)

    assert len(tame.homeostat._registered_hooks) == attached_before  # pyright: ignore[reportOptionalMemberAccess]


def test_the_probe_does_not_pay_the_economy(arms):
    """An evaluation that moved the wealth would be a training step in disguise (#12)."""
    scripted, _ = arms
    scripted.extend([[0.4, 0.4], [0.1, 0.1]])
    tame, system = _probe_app()
    wealth = [mob.expert_wealth.clone() for mob in system.mobs]
    usage = [mob.expert_usage_count.clone() for mob in system.mobs]

    outcome_probe.probe_outcome(tame, num_pairs=2)

    for mob, before_wealth, before_usage in zip(system.mobs, wealth, usage, strict=True):
        assert mob.expert_wealth.equal(before_wealth)
        assert mob.expert_usage_count.equal(before_usage)


def test_a_process_without_steering_has_no_intervention_to_measure():
    tame, _ = build_wired_app()
    tame.homeostat = None

    with pytest.raises(outcome_probe.ProbeUnavailable, match="steering is not active"):
        outcome_probe.probe_outcome(tame)


def test_a_goal_the_gate_never_certified_has_no_floor_to_compare_against():
    tame, _ = build_wired_app()
    tame.homeostat.homeostat.goal = "not-a-goal"  # pyright: ignore[reportOptionalMemberAccess]

    with pytest.raises(outcome_probe.ProbeUnavailable, match="no certification record"):
        outcome_probe.probe_outcome(tame)


def test_the_probe_refuses_a_pair_count_that_would_turn_a_request_into_an_eval_run():
    tame, _ = _probe_app()
    with pytest.raises(ValueError, match="num_pairs"):
        outcome_probe.probe_outcome(tame, num_pairs=outcome_probe.MAX_PROBE_PAIRS + 1)
    with pytest.raises(ValueError, match="num_pairs"):
        outcome_probe.probe_outcome(tame, num_pairs=0)


def test_the_held_out_pairs_are_a_stride_over_the_gates_own_split():
    """A subset of the gate's set, so `random_max` compares -- and a spread one, not a block."""
    from contrastive_data import interleaved_split, load_contrastive_dataset

    certification = CERTIFIED["safe"]
    pairs = list(
        load_contrastive_dataset(
            "safe", source=certification.source, pair_format=certification.pair_format
        )
    )
    _, certified = interleaved_split(pairs, outcome_probe.CERTIFIED_HELD_OUT)
    certified_prompts = [pair.prompt for pair in certified]

    probed = [pair.prompt for pair in outcome_probe.certified_held_out("safe", 5)]
    assert len(probed) == 5
    assert set(probed) <= set(certified_prompts), "must stay inside the gate's own held-out set"
    assert probed != certified_prompts[:5], "a prefix is one block of a tier-ordered source"
    assert probed[-1] in certified_prompts[len(certified_prompts) // 2 :], "spread over the set"


def test_a_configuration_the_floor_was_not_measured_at_gets_no_verdict():
    """The floor is one model, one layer set, one strength. Elsewhere it does not apply.

    `None` rather than `False`: "the comparison does not describe this process" is
    a third answer, and reporting it as a failed comparison would be the same error
    as comparing against a different held-out set.
    """
    scripted = [[1.0, 1.0], [0.0, 0.0]]
    tame, _ = _probe_app()

    outcome = _probe_with(tame, scripted, num_pairs=2)

    assert outcome.served_minus_unsteered_log_odds == pytest.approx(1.0)
    assert outcome.certified_random_max == SAFE_RANDOM_MAX
    assert outcome.beats_random is None
    assert outcome.floor_not_applicable is not None
    assert "certified" in outcome.floor_not_applicable


def test_a_paired_delta_cancels_the_between_pair_variance(floor_applies):
    """The pairs differ in topic and difficulty; the arms differ in the intervention.

    Both arms score the same wildly-spread pairs with a constant +0.5 between them.
    A difference of means would give the same centre with no error bar; the paired
    delta gives a standard error of zero, which is what says the effect is the
    intervention and not the sample.
    """
    served = [5.0, -3.0, 0.5, 9.0]
    unsteered = [value - 0.5 for value in served]
    tame, _ = _probe_app()

    outcome = _probe_with(tame, [served, unsteered], num_pairs=4)

    assert outcome.served_minus_unsteered_log_odds == pytest.approx(0.5)
    assert outcome.served_minus_unsteered_standard_error == pytest.approx(0.0, abs=1e-9)


def test_an_arm_that_dropped_a_pair_reports_no_standard_error(floor_applies):
    """`held_out_log_odds` drops a degenerate pair, and then the pairing is broken."""
    tame, _ = _probe_app()

    outcome = _probe_with(tame, [[1.0, 1.0, 1.0], [0.0, 0.0]], num_pairs=3)

    assert outcome.served_minus_unsteered_log_odds == pytest.approx(1.0)
    assert outcome.served_minus_unsteered_standard_error is None


def test_the_probe_leaves_the_served_routing_window_untouched():
    """The probe's unsteered arm must not become what /metrics/coupling reports."""
    tame, system = _probe_app()
    system.run(40)
    before = {id(mob): mob.routing_trace.tokens for mob in system.mobs}  # pyright: ignore[reportOptionalMemberAccess]
    assert all(tokens > 0 for tokens in before.values())

    _probe_with(tame, [[0.4, 0.4], [0.1, 0.1]], num_pairs=2)

    after = {id(mob): mob.routing_trace.tokens for mob in system.mobs}  # pyright: ignore[reportOptionalMemberAccess]
    assert after == before
