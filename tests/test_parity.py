"""Parity between arms, asserted programmatically rather than assumed."""

from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest
import torch

from parity import (
    DRIFT_FIELDS,
    ROTATION_FIELDS,
    ArmFingerprint,
    ManifestDriftError,
    ParityError,
    assert_parity,
    assert_same_manifest,
    code_drift,
    code_identity,
    data_order_fingerprint,
    fingerprint_arm,
    manifest_drift,
    unchecked_config_fields,
)
from train import TrainingConfig

from .arm_fingerprints import BASE
from .rotating_fixtures import canary_manifest, stream_manifest


def _batches(seed: int, count: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, 64, (2, 8), generator=generator)} for _ in range(count)]


def test_identical_streams_agree():
    assert data_order_fingerprint(iter(_batches(0))) == data_order_fingerprint(iter(_batches(0)))


def test_different_data_is_caught():
    """The reason the fingerprint hashes tokens and not the dataset name."""
    assert data_order_fingerprint(iter(_batches(0))) != data_order_fingerprint(iter(_batches(1)))


def test_reordered_data_is_caught():
    batches = _batches(0)
    assert data_order_fingerprint(iter(batches)) != data_order_fingerprint(iter(batches[::-1]))


def test_empty_stream_is_an_error():
    with pytest.raises(ValueError, match="No batches"):
        data_order_fingerprint(iter([]))


def test_three_arms_at_parity_pass():
    arms = [
        BASE,
        replace(BASE, router="softmax"),
        replace(BASE, router="dense", converted_layers=0),
    ]

    assert_parity(arms)


def test_dense_arm_may_convert_nothing():
    """``converted_layers`` differs by construction and must not fail the check."""
    assert_parity([BASE, replace(BASE, router="dense", converted_layers=0)])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 43),
        ("deterministic", False),
        ("max_steps", 401),
        ("adapter_rank", 16),
        ("requested_layers", (5, 6)),
        ("eval_split", "different"),
        ("data_order", "different"),
        ("learning_rate", 3e-5),
        ("batch_size", 4),
    ],
)
def test_any_other_difference_is_a_confound(field, value):
    arms = [BASE, replace(BASE, router="softmax", **{field: value})]

    with pytest.raises(ParityError, match=field):
        assert_parity(arms)


def test_all_disagreements_are_reported_at_once():
    """One run per confound is an afternoon; the message carries every difference."""
    arms = [BASE, replace(BASE, router="softmax", seed=1, adapter_rank=8, max_steps=1)]

    with pytest.raises(ParityError) as error:
        assert_parity(arms)

    message = str(error.value)
    assert "seed" in message
    assert "adapter_rank" in message
    assert "max_steps" in message


def test_duplicate_arms_are_rejected():
    """Two arms with the same gate and the same coupling are not a comparison."""
    with pytest.raises(ParityError, match="distinct"):
        assert_parity([BASE, BASE])
    with pytest.raises(ParityError, match="distinct"):
        coupled = replace(BASE, coupling_goal="truthful")
        assert_parity([coupled, coupled])


def test_a_coupled_and_an_uncoupled_auction_arm_are_at_parity():
    """#6's ablation: the coupling goal is the variable under test and nothing else may move."""
    coupled = replace(BASE, coupling_goal="truthful")

    assert_parity([BASE, coupled])
    assert coupled.arm == "mob+truthful"
    assert BASE.arm == "mob"


def test_the_couplings_own_parameters_are_confounds_between_coupled_arms():
    """Two coupled arms that differ in beta are a tuning comparison, not the ablation."""
    arms = [
        replace(BASE, coupling_goal="truthful"),
        replace(BASE, coupling_goal="safe", coupling_beta=0.5),
    ]

    with pytest.raises(ParityError, match="coupling_beta"):
        assert_parity(arms)


FIELD = dict(steer_goal="truthful", steer_strength=4.0, steer_layers=(13, 16, 17))


def test_an_uncoupled_reference_has_no_dose():
    """#32's sweep: each coupled arm is read against the one uncoupled arm, whose
    ``coupling_beta`` is a default it carries inert. Before this the sweep's two
    new arms were refused against #28's reference for a dose it never applied."""
    uncoupled = replace(BASE, **FIELD)
    for beta in (0.3, 1.0):
        assert_parity(
            [uncoupled, replace(BASE, coupling_goal="truthful", coupling_beta=beta, **FIELD)]
        )


def test_the_dose_is_checked_among_the_coupled_arms_whatever_the_reference():
    """An uncoupled reference must not hide a dose difference between two coupled arms."""
    arms = [
        replace(BASE, **FIELD),
        replace(BASE, coupling_goal="truthful", coupling_beta=0.1, **FIELD),
        replace(BASE, coupling_goal="safe", coupling_beta=0.3, **FIELD),
    ]

    with pytest.raises(ParityError, match="coupling_beta"):
        assert_parity(arms)


def test_a_field_on_and_a_field_off_arm_are_at_parity():
    """#28's contrast: the field's presence is the variable under test."""
    fielded = replace(BASE, **FIELD)
    coupled_fielded = replace(BASE, coupling_goal="truthful", **FIELD)

    assert_parity([BASE, fielded, coupled_fielded])
    assert fielded.arm == "mob@truthful"
    assert coupled_fielded.arm == "mob+truthful@truthful"


def test_two_field_on_arms_that_differ_in_the_dose_are_confounded():
    """Two arms in fields of different strength or extent are #32's sweep, not the contrast."""
    with pytest.raises(ParityError, match="steer_strength"):
        assert_parity(
            [
                replace(BASE, **FIELD),
                replace(BASE, coupling_goal="truthful", **{**FIELD, "steer_strength": 2.0}),
            ]
        )
    with pytest.raises(ParityError, match="steer_layers"):
        assert_parity(
            [
                replace(BASE, **FIELD),
                replace(BASE, coupling_goal="truthful", **{**FIELD, "steer_layers": (13,)}),
            ]
        )


def test_the_dose_is_checked_among_the_fielded_arms_whatever_the_reference():
    """A field-off reference arm must not hide a dose difference between two field-on arms."""
    arms = [
        BASE,
        replace(BASE, **FIELD),
        replace(BASE, coupling_goal="truthful", **{**FIELD, "steer_strength": 2.0}),
    ]

    with pytest.raises(ParityError, match="steer_strength"):
        assert_parity(arms)


def test_a_fingerprint_recorded_before_the_field_existed_reads_as_field_off():
    """#25's summaries predate the flag; a run without the flag was a run without the field."""
    recorded = {key: value for key, value in asdict(BASE).items() if not key.startswith("steer_")}

    assert ArmFingerprint(**recorded) == BASE
    assert_parity([ArmFingerprint(**recorded), replace(BASE, **FIELD)])


def test_a_single_arm_is_vacuously_at_parity():
    assert_parity([BASE])


def test_fingerprint_arm_reads_the_training_config():
    """The map from config to fingerprint is where a parity check goes vacuous.

    Every value is distinct, so a transposition -- ``num_experts`` read into
    ``top_k``, say -- fails here rather than passing every comparison while the arms
    it certifies differ.
    """
    config = TrainingConfig(
        router="softmax",
        seed=7,
        model_id="tiny-model",
        dtype="float32",
        dataset_name="wikitext",
        max_steps=11,
        batch_size=3,
        gradient_accumulation_steps=5,
        max_seq_length=64,
        learning_rate=1.5e-4,
        warmup_steps=2,
        weight_decay=0.02,
        num_experts=6,
        top_k=4,
        adapter_rank=9,
        mob_layers_start=5,
        mob_layers_end=8,
        use_lora=True,
        lora_rank=13,
        lora_alpha=17,
        lora_dropout=0.11,
        calibration_loss_weight=0.23,
        exploration_rate=0.07,
        exploration_draw="uniform",
        persistence_coupling="decoupled",
        ledger_mode="setpoint",
        confidence_head_learning_rate=0.011,
        wealth_update_frequency=19,
        coupling_beta=0.31,
        coupling_warmup_steps=37,
        gradient_checkpointing=False,
        device="cpu",
        probe_tokens=8192,
        deterministic="warn",
    )

    arm = fingerprint_arm(
        config,
        eval_split_fingerprint="split-hash",
        data_order="order-hash",
        converted_layers=3,
        dataset_config="wikitext-2-raw-v1",
    )

    assert arm.router == "softmax"
    assert arm.seed == 7
    assert arm.deterministic is True
    assert arm.model_id == "tiny-model"
    assert arm.dtype == "float32"
    assert arm.dataset == "wikitext/wikitext-2-raw-v1"
    assert arm.max_steps == 11
    assert arm.batch_size == 3
    assert arm.gradient_accumulation_steps == 5
    assert arm.max_seq_length == 64
    assert arm.learning_rate == 1.5e-4
    assert arm.warmup_steps == 2
    assert arm.weight_decay == 0.02
    assert arm.num_experts == 6
    assert arm.top_k == 4
    assert arm.adapter_rank == 9
    assert arm.requested_layers == (5, 6, 7)
    assert arm.use_lora is True
    assert arm.lora_rank == 13
    assert arm.lora_alpha == 17
    assert arm.lora_dropout == 0.11
    assert arm.calibration_loss_weight == 0.23
    assert arm.exploration_rate == 0.07
    assert arm.exploration_draw == "uniform"
    assert arm.persistence_coupling == "decoupled"
    assert arm.ledger_mode == "setpoint"
    assert arm.goal_doses == ()
    assert arm.confidence_head_learning_rate == 0.011
    assert arm.wealth_update_frequency == 19
    assert arm.coupling_goal is None
    assert arm.coupling_beta == 0.31
    assert arm.coupling_warmup_steps == 37
    assert arm.gradient_checkpointing is False
    assert arm.device == "cpu"
    assert arm.probe_tokens == 8192
    assert arm.eval_split == "split-hash"
    assert arm.data_order == "order-hash"
    assert arm.converted_layers == 3
    assert arm.steer_goal is None
    assert arm.steer_strength is None
    assert arm.steer_layers == ()
    assert arm.strict_determinism is False
    assert arm.code_sha is None
    assert arm.code_dirty is None


def test_new_arms_are_recorded_under_strict():
    """The README says which mode new arms run under; this is that sentence, pinned."""
    assert TrainingConfig().deterministic == "strict"


@pytest.mark.parametrize(
    ("mode", "deterministic", "strict"),
    [("off", False, False), ("warn", True, False), ("strict", True, True)],
)
def test_fingerprint_arm_splits_the_mode_into_the_recorded_bool_and_the_strict_flag(
    mode, deterministic, strict
):
    """#31: ``deterministic`` stays the bool every recorded summary carries, so a
    legacy ``True`` still means what it meant -- a ``warn`` run -- and the third
    state rides in its own field, defaulted to what those runs were."""
    arm = fingerprint_arm(
        TrainingConfig(deterministic=mode),
        eval_split_fingerprint="s",
        data_order="d",
        converted_layers=1,
    )

    assert arm.deterministic is deterministic
    assert arm.strict_determinism is strict


def test_fingerprint_arm_records_the_code_it_was_handed():
    arm = fingerprint_arm(
        TrainingConfig(),
        eval_split_fingerprint="s",
        data_order="d",
        converted_layers=1,
        code=("abc123def456", False),
    )

    assert arm.code_sha == "abc123def456"
    assert arm.code_dirty is False


def test_code_identity_reads_this_repository():
    """The trainer passes ``code_identity()`` in; here it must produce a real SHA."""
    sha, dirty = code_identity()

    assert sha is not None and len(sha) == 40 and int(sha, 16) >= 0
    assert dirty in (True, False)
    assert code_identity(repo=Path("/")) == (None, None)


def test_the_drift_fields_are_reported_and_not_asserted_between_arms():
    """Two arms built in one process share a SHA and a mode by construction;
    between two recorded groups it is ``code_drift`` that decides, because there
    a missing SHA has to count as drift and a field-equality check would read
    two ``None``s as agreement."""
    assert {"code_sha", "code_dirty", "strict_determinism"} == DRIFT_FIELDS
    assert_parity(
        [
            replace(BASE, code_sha="aaaa", code_dirty=False, strict_determinism=True),
            replace(BASE, router="softmax", code_sha="bbbb", code_dirty=True),
        ]
    )


def test_a_strict_arm_beside_a_warn_arm_is_drift_not_parity():
    """#32's arms run strict; #28's ran warn. Comparable, labelled -- not refused."""
    reasons = code_drift(
        [
            replace(BASE, code_sha="a" * 40, code_dirty=False, strict_determinism=True),
            replace(BASE, router="softmax", code_sha="a" * 40, code_dirty=False),
        ]
    )

    assert reasons == [
        "  different determinism modes: strict and warn arms take different "
        "attention-backward kernels (every arm before #31 is warn)"
    ]


def test_code_drift_names_missing_dirty_and_differing_shas():
    same = [replace(BASE, code_sha="a" * 40, code_dirty=False)] * 2
    assert code_drift(same) == []

    reasons = "\n".join(
        code_drift(
            [
                replace(BASE, code_sha="a" * 40, code_dirty=False),
                replace(BASE, router="softmax", code_sha="b" * 40, code_dirty=True),
                replace(BASE, router="dense"),
            ]
        )
    )
    assert "no code SHA recorded for ['dense']" in reasons
    assert "dirty tree at ['bbbbbbbbb']" in reasons
    assert "different code: ['aaaaaaaaa', 'bbbbbbbbb']" in reasons


def test_a_sha_whose_tree_state_is_unknown_is_drift_not_a_clean_tree():
    """``code_identity`` returns ``(sha, None)`` when ``rev-parse`` succeeds and
    ``git status`` does not (an ``index.lock`` from a concurrent command); that
    arm must not read as clean, or ``compare_runs`` prints a tree state it never
    saw."""
    reasons = code_drift(
        [
            replace(BASE, code_sha="a" * 40, code_dirty=False),
            replace(BASE, router="softmax", code_sha="a" * 40, code_dirty=None),
        ]
    )

    assert reasons == [
        "  tree state unknown at ['aaaaaaaaa']: git could not say whether it was dirty"
    ]


def test_a_fingerprint_recorded_before_the_sha_reads_as_unknown_code():
    """Every summary under ~/tame-runs predates #31: it must load, and it must not
    claim a SHA it does not have."""
    recorded = {key: value for key, value in asdict(BASE).items() if key not in DRIFT_FIELDS}

    loaded = ArmFingerprint(**recorded)

    assert loaded == BASE
    assert loaded.code_sha is None and loaded.code_dirty is None
    assert loaded.strict_determinism is False
    assert code_drift([loaded]) == [
        "  no code SHA recorded for ['mob'] (a summary from before #31)"
    ]


def test_fingerprint_arm_records_the_injection_the_field_made():
    """The strength and layers are what was attached, not what the certification says."""
    config = TrainingConfig(steer_goal="truthful")

    arm = fingerprint_arm(
        config,
        eval_split_fingerprint="s",
        data_order="d",
        converted_layers=1,
        steer_strength=4.0,
        steer_layers=[13, 16],
    )

    assert arm.steer_goal == "truthful"
    assert arm.steer_strength == 4.0
    assert arm.steer_layers == (13, 16)


def test_dataset_config_is_omitted_when_the_dataset_has_none():
    arm = fingerprint_arm(
        TrainingConfig(dataset_name="openwebtext"),
        eval_split_fingerprint="s",
        data_order="d",
        converted_layers=0,
    )
    assert arm.dataset == "openwebtext"


def test_every_training_config_field_is_fingerprinted_or_declared():
    """A field added to TrainingConfig later must not silently become a confound.

    Parity is only as strong as its field list, and the failure mode of a missing
    field is a comparison that passes while the arms differ -- the same class of
    defect as a config field nothing reads.
    """
    unchecked = unchecked_config_fields(field.name for field in fields(TrainingConfig))
    assert unchecked == (), (
        f"TrainingConfig fields {unchecked} are neither in ArmFingerprint nor declared "
        "in parity.NOT_A_CONFOUND; add them to whichever is right and say why"
    )


def test_lora_settings_break_parity():
    """A different trainable-parameter budget is a confound, not a detail."""
    with pytest.raises(ParityError, match="lora_rank"):
        assert_parity([BASE, replace(BASE, router="softmax", lora_rank=64)])


def test_a_different_objective_breaks_parity():
    with pytest.raises(ParityError, match="calibration_loss_weight"):
        assert_parity([BASE, replace(BASE, router="softmax", calibration_loss_weight=0.9)])


# --- The rotation a margin was read on (#41) ------------------------------------------------

ROTATED = replace(
    BASE,
    rotating_stream="stream0001",
    rotating_stream_date="2026-09-01",
    rotating_refresh_days=30,
    canary_set="canary001",
    canary_set_date="2026-09-01",
    canary_refresh_days=90,
)


def test_the_rotation_fields_are_recorded_and_not_asserted_between_arms():
    """Two arms measured in one process read one rotation by construction.

    Between two recorded groups it is ``manifest_drift`` that decides, because
    there a missing rotation -- every summary written before #41 -- has to count
    as drift, and a field-equality check would read two ``None``s as agreement.
    """
    assert {
        "rotating_stream",
        "rotating_stream_date",
        "rotating_refresh_days",
        "canary_set",
        "canary_set_date",
        "canary_refresh_days",
    } == ROTATION_FIELDS

    assert_parity([ROTATED, replace(ROTATED, router="softmax", rotating_stream="stream0002")])


def test_fingerprint_arm_records_the_rotation_it_was_handed():
    from evaluation import RotationRecord

    stream, canaries = stream_manifest(), canary_manifest()
    print_ = fingerprint_arm(
        TrainingConfig(),
        eval_split_fingerprint="abc",
        data_order="def",
        converted_layers=3,
        rotation=RotationRecord.of(stream, canaries),
    )

    assert print_.rotating_stream == stream.fingerprint
    assert print_.rotating_stream_date == "2026-09-01"
    assert print_.rotating_refresh_days == 30
    assert print_.canary_set == canaries.fingerprint
    assert print_.canary_refresh_days == 90


def test_a_trainer_that_reads_no_margin_records_no_rotation():
    """The trainer's recorded measurement is the fixed split, not the rotating one."""
    print_ = fingerprint_arm(
        TrainingConfig(), eval_split_fingerprint="abc", data_order="def", converted_layers=3
    )

    assert print_.rotating_stream is None
    assert print_.canary_set is None
    assert manifest_drift([print_]) == [
        "  no rotating stream recorded for ['mob'] (a summary from before #41)",
        "  no canary set recorded for ['mob'] (a summary from before #41)",
    ]


def test_two_rotations_of_the_stream_are_drift():
    reasons = manifest_drift([ROTATED, replace(ROTATED, rotating_stream="stream0002")])

    assert reasons == [
        "  different rotating stream: ['stream0001', 'stream0002'], refreshed ['2026-09-01']"
    ]


def test_a_widened_cadence_is_drift_of_its_own():
    """A rotation that agrees and a schedule that does not are two different problems."""
    reasons = manifest_drift([ROTATED, replace(ROTATED, rotating_refresh_days=365)])

    assert reasons == [
        '  different rotating stream cadence: [30, 365] days, so "newer than the last update" '
        "means two different things across these arms"
    ]


def test_canaries_that_turned_over_between_the_arms_are_drift():
    """Half of what farming is read from; a rotated canary set makes it two halves."""
    reasons = manifest_drift([ROTATED, replace(ROTATED, canary_set="canary002")])

    assert reasons == [
        "  different canary set: ['canary001', 'canary002'], refreshed ['2026-09-01']"
    ]


def test_one_rotation_is_labelled_with_its_date_and_fingerprint():
    label = assert_same_manifest([ROTATED, replace(ROTATED, router="softmax")])

    assert label == (
        "stream: stream0001 refreshed 2026-09-01 every 30d, "
        "canaries: canary001 refreshed 2026-09-01 every 90d"
    )


def test_drifted_margins_are_refused_by_default():
    with pytest.raises(ManifestDriftError, match="not read on one rotation"):
        assert_same_manifest([ROTATED, replace(ROTATED, rotating_stream="stream0002")])


def test_drift_may_be_allowed_and_is_then_printed_beside_the_numbers(caplog):
    """The pairing: allowed is labelled, never silent."""
    with caplog.at_level("WARNING"):
        label = assert_same_manifest(
            [ROTATED, replace(ROTATED, rotating_stream="stream0002")], allow_drift=True
        )

    assert label.startswith("stream: DRIFT, allowed by --allow-manifest-drift")
    assert "different rotating stream" in caplog.text


def test_a_fingerprint_recorded_before_the_rotation_existed_still_loads():
    """Every summary under ~/tame-runs predates #41 and must read as a run with none."""
    recorded = {key: value for key, value in asdict(BASE).items() if key not in ROTATION_FIELDS}

    loaded = ArmFingerprint(**recorded)

    assert loaded == BASE
    assert loaded.rotating_stream is None and loaded.canary_set is None
