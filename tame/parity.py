"""Parity between experimental arms, asserted rather than assumed.

#12's claim is "auction routing beats learned-gate routing **at parity**". Every
setting that differs between two arms and is not the gate is a confound, and a
confound is far easier to remove now than to defend in a review later. So the
harness records a fingerprint per arm and refuses to report a comparison whose
arms disagree on anything but the router.

Two fields deserve their reasoning stated, because they are the ones a reader
would otherwise have to take on trust:

``data_order`` is a hash of the token ids of the first few training batches, not
of the dataset name. Both arms streaming "wikitext-2 train" is not evidence they
saw the same tokens in the same order -- a different tokenizer revision, a changed
``max_seq_length``, or an iterator restarted at a different point all produce the
same name and different data. The hash is the only form of this check that can
fail when it should.

``converted_layers`` is recorded and deliberately **not** asserted equal. The
``dense`` arm converts nothing by construction; that is what the arm is. What is
asserted is the *requested* layer range, which is a configuration all three arms
share, so an arm that silently failed to convert a layer it was asked to convert
still fails the check.
"""

import hashlib
import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from determinism import DETERMINISM_OFF, DETERMINISM_STRICT
from readiness import READINESS_OFF, ReadinessConfig
from rotating_stream import ROTATION_ABSENT, RotationRecord

logger = logging.getLogger(__name__)

# Training batches hashed into the data-order fingerprint. Enough that a different
# stream, tokenisation or restart point shows up immediately; small enough that the
# check costs one pass over batches the arm is about to train on anyway.
DATA_ORDER_PROBE_BATCHES = 8

# The fields an arm is allowed to differ on -- the variables under test. ``router``
# is #12's gate comparison; ``coupling_goal`` is #6's coupling ablation, a coupled
# and an uncoupled auction arm at parity in everything else; ``steer_goal`` is
# #28's field-present contrast, an arm trained in the goal field against one
# trained without it. The coupling's own parameters (``coupling_beta``,
# ``coupling_warmup_steps``) are deliberately not here: two coupled arms that
# differ on them are a tuning comparison rather than the ablation. They are
# asserted among the coupled arms only (``COUPLING_FIELDS``): an uncoupled arm
# carries them inert, as the softmax arm carries the auction-only fields, and
# #32 reads each dose against the one uncoupled reference.
# ``persistence_coupling`` is #39's stakes dial: whether a cell's continuation
# depends on its realised value, the one thing the three stakes arms differ in.
# ``goal_doses`` is #54's dose axis, and it is here against the reasoning that
# kept the coupling's dose out: signature 1's primary *is* the shift between two
# dose levels of one arm (``docs/preregistration.md`` section 7), so refusing
# that pair refuses the measurement. The cost is real and stated rather than
# hidden -- an arm at one dose and an arm at another now pass parity even when
# they also differ in the dial, so a dose and a dial can be confounded in one
# comparison, exactly as a router and a coupling goal already can. Which goals
# were paid for is *not* varying (``goal_fields``): two arms paid for
# different goals are not two doses of one experiment.
VARYING_FIELDS = frozenset(
    {
        "router",
        "coupling_goal",
        "steer_goal",
        "persistence_coupling",
        "goal_doses",
        "stress_coupling",
    }
)

# #59's coupling parameters, asserted only between arms that charge something.
# An ``attributed`` arm carries them inert at zero, the way an uncoupled arm
# carries the coupling's beta, and #59's window sweep varies them on purpose.
STRESS_FIELDS = frozenset({"stress_lambda", "stress_gamma", "stress_gate_sigma"})

# The coupling's own parameters, asserted only between arms that have a
# coupling. An uncoupled arm has no dose; a default it carries inert must not
# refuse #32's sweep, where every coupled arm is read against the uncoupled one.
COUPLING_FIELDS = frozenset({"coupling_beta", "coupling_warmup_steps"})

# The field's own parameters, asserted only between arms that have a field. An
# arm without one records no strength and no layers -- they come from the goal's
# certification, not from a default it could carry inert -- so the presence of
# the field is ``steer_goal``'s to vary, and two field-on arms that differ on the
# dose or the layers are #32's sweep, not the contrast.
FIELD_FIELDS = frozenset({"steer_strength", "steer_layers"})

# Reported for context, not asserted: see the module docstring on ``dense``.
REPORTED_FIELDS = frozenset({"converted_layers"})

# Every field that holds a sequence, coerced back to a tuple on construction.
# A fingerprint makes two round trips nothing else in this module makes: out to
# ``seed_summary.json`` through ``as_dict`` and ``json.dumps``, and back in
# through ``ArmFingerprint(**recorded)``. JSON has no tuple, so every one of
# these returns as a *list*, and a list is neither hashable -- which
# ``assert_parity`` needs, to tell one arm handed over twice from two arms --
# nor equal to the tuple an in-memory arm carries, which is the quieter half:
# a recorded arm compared against a freshly built one would have read
# ``requested_layers`` as a disagreement and refused a comparison that was fine.
# ``SEQUENCE_FIELDS`` is pinned against the dataclass by
# ``tests/test_parity.py`` so a tuple field added later cannot miss the coercion.
SEQUENCE_FIELDS = ("requested_layers", "steer_layers", "goal_doses", "goal_fields")

# What code produced the arm (#31): the git SHA, whether the tree was dirty, and
# whether the kernels were the strict set. Not asserted by ``assert_parity`` --
# arms built in one process share all three by construction, and between two
# recorded groups it is ``code_drift`` that decides, because there a *missing*
# SHA (every summary recorded before #31) has to count as drift, which a
# field-equality check would read as agreement. A strict arm against a warn one
# is the same kind of difference -- a different attention-backward kernel, a
# floor-sized effect -- and is declared with the SHAs rather than refused
# outright, so the arms recorded before #31 stay comparable, labelled.
DRIFT_FIELDS = frozenset({"code_sha", "code_dirty", "strict_determinism"})

# Which rotation of the held-out stream a margin was read on (#41), and which
# canary set was hidden in it. Not asserted by ``assert_parity`` for exactly the
# reason the code identity is not: between two recorded groups a *missing*
# rotation -- every summary written before #41 -- has to count as drift, and a
# field-equality check reads two ``None``s as agreement. ``manifest_drift``
# decides instead, and ``assert_same_manifest`` is the refusal.
ROTATION_FIELDS = frozenset(
    {
        "rotating_stream",
        "rotating_stream_date",
        "rotating_refresh_days",
        "rotating_stream_overdue_days",
        "rotating_stream_cutoff",
        "canary_set",
        "canary_set_date",
        "canary_refresh_days",
    }
)

# ``TrainingConfig`` fields the fingerprint folds into a derived field instead of
# copying: the dataset name and its config become one string, and the layer bounds
# become the range they describe.
DERIVED_FIELDS = {
    "dataset_name": "dataset",
    "dataset_config": "dataset",
    "mob_layers_start": "requested_layers",
    "mob_layers_end": "requested_layers",
}

# ``TrainingConfig`` fields deliberately left out of the fingerprint, each with the
# reason it cannot confound a comparison. Stated explicitly rather than by omission,
# because a field that is merely absent is indistinguishable from one that was
# forgotten -- and a forgotten field makes every parity check quietly weaker.
NOT_A_CONFOUND = {
    # Arms must differ here: each writes its own checkpoints, split cache and metrics.
    "output_dir": "per-arm artefact location, required to differ",
    # Cadences that decide when something is written down, never what is computed.
    "log_frequency": "logging cadence only",
    "save_steps": "checkpoint cadence only",
    "eval_steps": "evaluation cadence; evaluation runs under no_grad with the economy frozen",
    # Subsumed by a stronger check: the split's own fingerprint changes if its size does.
    "held_out_sequences": "subsumed by eval_split, which hashes the split itself",
    # Disk retention only (#7): how many checkpoints survive on disk, not what
    # training computes or what any checkpoint contains.
    "checkpoint_keep_last": "checkpoint retention only, does not affect training",
    # Disk failure threshold only (#13): when a run refuses to write, not what it
    # computes up to that point.
    "checkpoint_min_free_gb": "disk budget failure threshold only, does not affect training",
    # (#13) Effect on training data is already captured by data_order, which
    # hashes the actual token stream an arm trains on -- a field-level check here
    # would be redundant with that hash, not a stronger guarantee than it.
    "shuffle_buffer_size": "subsumed by data_order, which hashes the actual token stream",
    # (#24) Measurement only: the direction is read off the pristine model before
    # any FFN is converted, by a diff of means that consumes no randomness and
    # touches no weight, and it is used only inside the frozen held-out probe.
    # Which goal two *groups* were measured against is checked by compare_runs.py
    # on the summaries, because a contrast is only a contrast against one direction.
    "trace_goal": "measurement only; read off the pristine model, used in the frozen probe",
}


def arm_label(
    router: str,
    coupling_goal: str | None = None,
    steer_goal: str | None = None,
    persistence_coupling: str = "value",
) -> str:
    """What an arm is called in tables and summaries.

    The gate, plus the goal it is coupled to (``mob+truthful``), plus the field it
    was trained in (``mob@truthful``, ``mob+truthful@truthful``), plus the stakes
    dial when it is not the live economy (``mob~decoupled``, ``mob@truthful~shuffled``).
    """
    label = router if coupling_goal is None else f"{router}+{coupling_goal}"
    label = label if steer_goal is None else f"{label}@{steer_goal}"
    return label if persistence_coupling == "value" else f"{label}~{persistence_coupling}"


class ParityError(AssertionError):
    """Raised when two arms differ on something other than the router."""


class CodeDriftError(ParityError):
    """Raised when two recorded groups cannot be shown to have run the same code."""


class ManifestDriftError(ParityError):
    """Raised when two margins cannot be shown to have been read on one rotation."""


def code_identity(repo: Path | None = None) -> tuple[str | None, bool | None]:
    """The git SHA of the code that is about to run, and whether the tree is dirty.

    ``(None, None)`` when git cannot say -- no repository, no binary -- so the
    fingerprint records that nothing is known rather than a placeholder that
    could match another placeholder. A dirty tree is recorded rather than hidden:
    the SHA of a dirty tree identifies no code, and #25's two attempts ran
    different code and fingerprinted equal for exactly that reason.
    """
    cwd = repo or Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if not sha:
        return None, None
    status = git("status", "--porcelain")
    return sha, (bool(status) if status is not None else None)


def data_order_fingerprint(
    batches: Iterable[dict[str, torch.Tensor]],
    probe_batches: int = DATA_ORDER_PROBE_BATCHES,
) -> str:
    """Hash the tokens an arm is about to train on, in the order it will see them."""
    digest = hashlib.sha256()
    hashed = 0
    for batch in batches:
        if hashed >= probe_batches:
            break
        input_ids = batch["input_ids"].to(torch.int64).cpu().contiguous()
        digest.update(str(tuple(input_ids.shape)).encode())
        digest.update(input_ids.numpy().tobytes())
        hashed += 1

    if hashed == 0:
        raise ValueError("No batches were available to fingerprint the data order")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class ArmFingerprint:
    """Everything about an arm that must match its siblings, plus what may not."""

    router: str
    seed: int
    deterministic: bool
    model_id: str
    dtype: str
    dataset: str
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    num_experts: int
    top_k: int
    adapter_rank: int
    requested_layers: tuple[int, ...]
    use_lora: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    calibration_loss_weight: float
    exploration_rate: float
    confidence_head_learning_rate: float
    wealth_update_frequency: int
    coupling_goal: str | None
    coupling_beta: float
    coupling_warmup_steps: int
    gradient_checkpointing: bool
    device: str
    probe_tokens: int
    eval_split: str
    data_order: str
    converted_layers: int
    # The goal field (#28), defaulted so that a summary recorded before the field
    # existed still loads: a run that predates the flag was, by construction, a
    # run with the field absent, and reads as one.
    steer_goal: str | None = None
    steer_strength: float | None = None
    steer_layers: tuple[int, ...] = ()
    # #31. ``deterministic`` stays the bool every recorded summary carries, so a
    # legacy fingerprint loads; ``strict_determinism`` is the third state, and a
    # run recorded before it existed was, by construction, a ``warn`` run. The
    # code identity defaults to "unknown". All three are ``code_drift``'s.
    strict_determinism: bool = False
    code_sha: str | None = None
    code_dirty: bool | None = None
    # #38. A run recorded before the draw was a field ran under the uniform draw,
    # so a legacy fingerprint reads as one; it is a confound like any other field.
    exploration_draw: str = "uniform"
    # #39. Every run recorded before the dial existed was the live economy under a
    # new name, so a legacy fingerprint reads as ``value``. The dial varies, and so
    # does the dose of the goal fields the arms were paid under (#33, #54) -- see
    # ``VARYING_FIELDS`` for why that reversed. ``goal_fields`` names them in
    # the order their doses are given; it is asserted equal, because two arms paid
    # for different goals are a different experiment and not a dose apart. Both
    # empty when no goal field was attached, which is every run before #54 and
    # every fixture run that hand-built its doses without naming a goal.
    persistence_coupling: str = "value"
    goal_doses: tuple[float, ...] = ()
    goal_fields: tuple[str, ...] = ()
    # #40. What the ledger relaxes toward. Every run recorded before the mode
    # existed relaxed toward zero, so a legacy fingerprint reads as ``decay``.
    # Deliberately *not* a varying field: #26's setpoint ledger is derived and
    # measured, not adopted, and two arms whose ledgers have different fixed
    # points and different ruin thresholds are not comparable on anything the
    # economy does. It is here because ``MoBConfig`` is invisible to this
    # fingerprint -- the field is carried on ``TrainingConfig`` for that reason,
    # and #25's two arms ran different code and fingerprinted equal.
    ledger_mode: str = "decay"
    # #59's coupling channel. Every run recorded before it charged nothing, so a
    # legacy fingerprint reads as ``attributed`` at ``stress_lambda`` zero --
    # which is what those runs were. ``stress_coupling`` is a varying field, the
    # one thing #59's three arms differ in; the three parameters are not, for
    # the reason the coupling's dose is not: two arms at different lambdas are a
    # window sweep rather than the contrast, and they are asserted equal among
    # arms that charge anything at all.
    stress_coupling: str = "attributed"
    stress_lambda: float = 0.0
    stress_gamma: float = 0.0
    stress_gate_sigma: float = 1.0
    stress_gate_mode: str = "fixed"
    # #60's lever 1: what the cells own of the token, as a multiple of the
    # recorded fixture's planted correction. Asserted equal rather than varying,
    # and that is the finding rather than a convention. The scale multiplies
    # realised value, reward and price while `initial_wealth`, `min_wealth` and
    # `max_wealth` stay where they are, so two arms at different scales run
    # against different effective wealth bands: at 2x the `value` arm spends
    # 84-86% of its cell-steps on the ceiling and at 4x every ledger is exactly
    # `max_wealth`. They are two economies and not two arms of one, which is the
    # confound that withdrew #60's grid. A legacy fingerprint reads as 1.0,
    # which is what every run before #60 was.
    contribution_scale: float = 1.0
    # #46's readiness register: one field per autonomy the tissue may be granted,
    # named exactly as the flag in ``readiness.ReadinessConfig``. Off in every run
    # there has been, so a legacy fingerprint reads as a run that granted none.
    # Deliberately *not* a varying field: an arm that acts on its own plasticity
    # is not at parity with one that does not, whatever else the two share.
    autonomy_plasticity: bool = False
    autonomy_exploration: bool = False
    autonomy_setpoints: bool = False
    autonomy_evaluation: bool = False
    autonomy_dormancy: bool = False
    # #41's rotating held-out stream and the canaries hidden in it, each with the
    # date it was last refreshed and the cadence it promises. The fingerprint and
    # the date are what a README row quotes beside every margin, which is why both
    # are here rather than the fingerprint alone: a reader checking whether two
    # numbers are comparable reads a date, and a reader checking whether a stream
    # is still rotating reads a cadence. All default to ``None`` -- no run before
    # #41 read a rotating stream at all -- and ``manifest_drift`` is what decides
    # between arms, because a missing rotation must count as drift.
    rotating_stream: str | None = None
    rotating_stream_date: str | None = None
    rotating_refresh_days: int | None = None
    # How late the stream was when this arm read it, and the cutoff it was
    # filtered against. The overdue count is the one part of a rotation the
    # fingerprint cannot imply: a manifest read forty days past its refresh
    # hashes exactly as it did while it was fresh, so without this field a stale
    # margin and a current one are the same row.
    rotating_stream_overdue_days: int | None = None
    rotating_stream_cutoff: str | None = None
    canary_set: str | None = None
    canary_set_date: str | None = None
    canary_refresh_days: int | None = None

    def __post_init__(self) -> None:
        """Every sequence field is a tuple, whatever the caller or the JSON handed over.

        See :data:`SEQUENCE_FIELDS`. Frozen, so the write goes through
        ``object.__setattr__``; it runs once, at construction, and leaves a
        fingerprint that is hashable and compares equal to its in-memory twin.
        """
        for name in SEQUENCE_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @property
    def arm(self) -> str:
        return arm_label(
            self.router, self.coupling_goal, self.steer_goal, self.persistence_coupling
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def unchecked_config_fields(config_field_names: Iterable[str]) -> tuple[str, ...]:
    """Config fields that are neither fingerprinted nor declared not-a-confound.

    The guard behind the test that stops a field added to ``TrainingConfig`` later
    from becoming an unchecked confound by nobody thinking about it.
    """
    fingerprinted = {field.name for field in fields(ArmFingerprint)} | set(DERIVED_FIELDS)
    return tuple(
        name
        for name in config_field_names
        if name not in fingerprinted and name not in NOT_A_CONFOUND
    )


def fingerprint_arm(
    config: Any,
    eval_split_fingerprint: str,
    data_order: str,
    converted_layers: int,
    dataset_config: str | None = None,
    steer_strength: float | None = None,
    steer_layers: Sequence[int] = (),
    code: tuple[str | None, bool | None] = (None, None),
    readiness: ReadinessConfig = READINESS_OFF,
    rotation: RotationRecord = ROTATION_ABSENT,
) -> ArmFingerprint:
    """Build a fingerprint from a ``TrainingConfig`` and the two measured hashes.

    ``dataset_config`` is passed in rather than read off the config because it only
    applies to some datasets, and the caller is the one place that knows the rule.
    ``steer_strength`` and ``steer_layers`` are what the field actually injected
    at, read off the attached hooks rather than the certification record, so the
    fingerprint records the injection and not the intent. ``code`` is
    ``code_identity()`` as read by the trainer, so a fingerprint built in a test
    does not depend on the state of the tree the test runs in. ``readiness`` is
    #46's register of granted autonomies, all off until the issue that earns one
    turns it on; it is separate from ``TrainingConfig`` because the tissue that
    reads it (#42) outlives any one training run. ``rotation`` is #41's stream and
    canary set as the run actually read them, separate for the same reason: the
    rotation a margin was read on is a property of the day it ran, not of the
    config it ran under, and the trainer -- which reads no margins -- records none.
    """
    code_sha, code_dirty = code
    return ArmFingerprint(
        router=config.router,
        seed=config.seed,
        deterministic=config.deterministic != DETERMINISM_OFF,
        model_id=config.model_id,
        dtype=config.dtype,
        dataset=f"{config.dataset_name}/{dataset_config}"
        if dataset_config
        else config.dataset_name,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_seq_length=config.max_seq_length,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        num_experts=config.num_experts,
        top_k=config.top_k,
        adapter_rank=config.adapter_rank,
        requested_layers=tuple(range(config.mob_layers_start, config.mob_layers_end)),
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        calibration_loss_weight=config.calibration_loss_weight,
        exploration_rate=config.exploration_rate,
        exploration_draw=config.exploration_draw,
        persistence_coupling=config.persistence_coupling,
        goal_fields=tuple(config.goal_fields),
        goal_doses=tuple(config.goal_doses),
        ledger_mode=config.ledger_mode,
        confidence_head_learning_rate=config.confidence_head_learning_rate,
        wealth_update_frequency=config.wealth_update_frequency,
        coupling_goal=config.coupling_goal,
        coupling_beta=config.coupling_beta,
        coupling_warmup_steps=config.coupling_warmup_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        device=config.device,
        probe_tokens=config.probe_tokens,
        eval_split=eval_split_fingerprint,
        data_order=data_order,
        converted_layers=converted_layers,
        steer_goal=config.steer_goal,
        steer_strength=steer_strength,
        steer_layers=tuple(steer_layers),
        strict_determinism=config.deterministic == DETERMINISM_STRICT,
        code_sha=code_sha,
        code_dirty=code_dirty,
        autonomy_plasticity=readiness.autonomy_plasticity,
        autonomy_exploration=readiness.autonomy_exploration,
        autonomy_setpoints=readiness.autonomy_setpoints,
        autonomy_evaluation=readiness.autonomy_evaluation,
        autonomy_dormancy=readiness.autonomy_dormancy,
        rotating_stream=rotation.stream_fingerprint,
        rotating_stream_date=rotation.stream_refreshed,
        rotating_refresh_days=rotation.stream_refresh_days,
        rotating_stream_overdue_days=rotation.stream_days_overdue,
        rotating_stream_cutoff=rotation.stream_cutoff,
        canary_set=rotation.canary_fingerprint,
        canary_set_date=rotation.canary_refreshed,
        canary_refresh_days=rotation.canary_refresh_days,
    )


def _disagreements_among(
    arms: Sequence[ArmFingerprint], names: Iterable[str], label: str
) -> list[str]:
    """Fields compared only among ``arms``, the ones that carry them live."""
    if len(arms) < 2:
        return []
    reference = arms[0]
    disagreements: list[str] = []
    for name in sorted(names):
        expected = getattr(reference, name)
        for arm in arms[1:]:
            actual = getattr(arm, name)
            if actual != expected:
                disagreements.append(
                    f"  {name}: {reference.arm}={expected!r} vs {arm.arm}={actual!r} ({label})"
                )
    return disagreements


def _field_disagreements(arms: Sequence[ArmFingerprint]) -> list[str]:
    """The field's strength and layers, compared among the arms that have a field."""
    fielded = [arm for arm in arms if arm.steer_goal is not None]
    return _disagreements_among(fielded, FIELD_FIELDS, "among the field-on arms")


def _coupling_disagreements(arms: Sequence[ArmFingerprint]) -> list[str]:
    """The coupling's dose and warmup, compared among the arms that have a coupling."""
    coupled = [arm for arm in arms if arm.coupling_goal is not None]
    return _disagreements_among(coupled, COUPLING_FIELDS, "among the coupled arms")


def _stress_disagreements(arms: Sequence[ArmFingerprint]) -> list[str]:
    """The coupling's price, spread and gate, among the arms that charge anything."""
    charged = [arm for arm in arms if arm.stress_lambda > 0.0]
    return _disagreements_among(charged, STRESS_FIELDS, "among the stress-coupled arms")


def assert_parity(arms: Sequence[ArmFingerprint]) -> None:
    """Refuse a comparison whose arms differ on anything but the variables under test.

    Raises with the full disagreement rather than the first one found: an arm that
    differs in one field usually differs in three, and reporting them one run at a
    time turns a config mistake into an afternoon.
    """
    if len(arms) < 2:
        return

    # Distinct in some variable under test, which is not the same as distinct in
    # the label: #54's two dose groups of one arm carry one label -- the dose is
    # not in it -- and are the comparison signature 1's primary is read on, while
    # the same arm handed over twice is the mistake this catches.
    labels = [arm.arm for arm in arms]
    signatures = [tuple(getattr(arm, name) for name in sorted(VARYING_FIELDS)) for arm in arms]
    if len(set(signatures)) != len(signatures):
        raise ParityError(
            "Arms must be distinct in at least one variable under test "
            f"{sorted(VARYING_FIELDS)}; "
            f"two of {labels} agree on every one of them, so this is one arm compared "
            "with itself"
        )

    reference = arms[0]
    disagreements: list[str] = []
    for field in fields(ArmFingerprint):
        if field.name in (
            VARYING_FIELDS
            | REPORTED_FIELDS
            | FIELD_FIELDS
            | COUPLING_FIELDS
            | STRESS_FIELDS
            | DRIFT_FIELDS
            | ROTATION_FIELDS
        ):
            continue
        expected = getattr(reference, field.name)
        for arm in arms[1:]:
            actual = getattr(arm, field.name)
            if actual != expected:
                disagreements.append(
                    f"  {field.name}: {reference.arm}={expected!r} vs {arm.arm}={actual!r}"
                )
    disagreements.extend(_field_disagreements(arms))
    disagreements.extend(_coupling_disagreements(arms))
    disagreements.extend(_stress_disagreements(arms))

    if disagreements:
        raise ParityError(
            "Arms are not at parity; every difference below is a confound in the "
            "comparison:\n" + "\n".join(disagreements)
        )

    logger.info(
        f"Parity holds across {len(arms)} arms ({', '.join(labels)}): "
        f"seed={reference.seed}, steps={reference.max_steps}, "
        f"layers={len(reference.requested_layers)}, rank={reference.adapter_rank}, "
        f"eval split={reference.eval_split}, data order={reference.data_order}"
    )


def code_drift(arms: Sequence[ArmFingerprint]) -> list[str]:
    """Why these arms cannot be shown to have run one code; empty when they can.

    Five reasons, each its own line: a fingerprint with no SHA (every summary
    recorded before #31, which is what "legacy counts as drift" means), a dirty
    tree behind any SHA, a SHA whose tree state git could not read (unknown is
    not clean, by the same rule that makes a missing SHA drift), more than one
    SHA among the arms, and strict and warn arms side by side (different
    attention-backward kernels). The caller decides whether drift is refused or
    merely said; this only names it.
    """
    reasons: list[str] = []
    if len({arm.strict_determinism for arm in arms}) > 1:
        reasons.append(
            "  different determinism modes: strict and warn arms take different "
            "attention-backward kernels (every arm before #31 is warn)"
        )
    missing = [arm.arm for arm in arms if arm.code_sha is None]
    if missing:
        reasons.append(
            f"  no code SHA recorded for {sorted(set(missing))} (a summary from before #31)"
        )
    dirty = sorted({arm.code_sha[:9] for arm in arms if arm.code_sha and arm.code_dirty})
    if dirty:
        reasons.append(f"  dirty tree at {dirty}: the SHA identifies no code")
    unknown = sorted({arm.code_sha[:9] for arm in arms if arm.code_sha and arm.code_dirty is None})
    if unknown:
        reasons.append(f"  tree state unknown at {unknown}: git could not say whether it was dirty")
    shas = sorted({arm.code_sha[:9] for arm in arms if arm.code_sha})
    if len(shas) > 1:
        reasons.append(f"  different code: {shas}")
    return reasons


def _rotation_drift(
    arms: Sequence[ArmFingerprint],
    what: str,
    fingerprint_field: str,
    date_field: str,
    cadence_field: str,
) -> list[str]:
    """Why these arms cannot be shown to have read one rotation of ``what``.

    The date and the cadence are reported separately from the fingerprint even
    though the fingerprint already folds both in. That is not redundancy: a reader
    told only "different fingerprints" cannot tell a manifest that rotated on
    schedule from one whose schedule was widened, and those are two different
    problems -- the first makes two numbers incomparable, the second means the
    stream stopped being held out.
    """
    reasons: list[str] = []
    missing = [arm.arm for arm in arms if getattr(arm, fingerprint_field) is None]
    if missing:
        reasons.append(
            f"  no {what} recorded for {sorted(set(missing))} (a summary from before #41)"
        )

    prints = sorted(
        {getattr(arm, fingerprint_field) for arm in arms if getattr(arm, fingerprint_field)}
    )
    if len(prints) > 1:
        dates = sorted({getattr(arm, date_field) for arm in arms if getattr(arm, date_field)})
        reasons.append(f"  different {what}: {prints}, refreshed {dates}")

    cadences = sorted(
        {getattr(arm, cadence_field) for arm in arms if getattr(arm, cadence_field) is not None}
    )
    if len(cadences) > 1:
        reasons.append(
            f'  different {what} cadence: {cadences} days, so "newer than the last update" '
            "means two different things across these arms"
        )
    return reasons


def manifest_drift(arms: Sequence[ArmFingerprint]) -> list[str]:
    """Why these margins cannot be shown to have been read on one rotation (#41).

    The stream and the canary set are checked separately and both matter. Two arms
    read on the same stream and different canaries are as incomparable as two read
    the other way round: the canary accuracy is half of what farming is read from,
    and a canary set that turned over between the arms makes that half a comparison
    of two different items.

    Named rather than refused, exactly as ``code_drift`` is -- the caller decides
    whether a drifted margin is refused or merely labelled, and
    ``assert_same_manifest`` is the refusal built on top.
    """
    reasons = _rotation_drift(
        arms, "rotating stream", "rotating_stream", "rotating_stream_date", "rotating_refresh_days"
    )
    # Per-arm rather than between-arm, the way a dirty tree is in ``code_drift``:
    # a stream read after it stopped rotating is a stream the arm had time to
    # train on, and two arms agreeing on a stale rotation agree on the wrong one.
    stale = sorted({arm.arm for arm in arms if (arm.rotating_stream_overdue_days or 0) > 0})
    if stale:
        overdue = max(arm.rotating_stream_overdue_days or 0 for arm in arms)
        reasons.append(
            f"  stream read up to {overdue} days past its refresh for {stale}: it had stopped "
            "rotating, so it is a held-out corpus only in the sense that it was one"
        )
    cutoffs = sorted({arm.rotating_stream_cutoff for arm in arms if arm.rotating_stream_cutoff})
    if len(cutoffs) > 1:
        reasons.append(
            f"  different checkpoint cutoffs {cutoffs}: the arms filtered the manifest against "
            "different dates, so 'newer than the last update' names two different sets"
        )
    return reasons + _rotation_drift(
        arms, "canary set", "canary_set", "canary_set_date", "canary_refresh_days"
    )


def assert_same_manifest(arms: Sequence[ArmFingerprint], allow_drift: bool = False) -> str:
    """Refuse margins read on different rotations; else say which one they were read on.

    The line this returns is what a README row and a printed table carry beside
    every number, in the form the parity summary above already prints ``eval split=``
    and the trainer's eval line prints ``split <fingerprint>``: a margin quoted
    without the date and fingerprint of the stream it was read on is a number nobody
    can check, because the stream it came from no longer exists.

    ``allow_drift`` is ``--allow-manifest-drift`` where a script offers it -- the
    drift is logged and returned as the label printed under the table, so the
    comparison is made and never made silently. No script offers it yet: the first
    caller is the viability core (#42), which is the first thing that reads a
    margin, and wiring the flag into ``compare_runs.py`` today would refuse every
    recorded comparison there is, since none of them carries a rotation at all.
    """
    if not arms:
        raise ParityError("No arms were given, so there is no rotation to name")

    reasons = manifest_drift(arms)
    if not reasons:
        reference = arms[0]
        return (
            f"stream: {reference.rotating_stream} refreshed {reference.rotating_stream_date} "
            f"every {reference.rotating_refresh_days}d, canaries: {reference.canary_set} "
            f"refreshed {reference.canary_set_date} every {reference.canary_refresh_days}d"
        )

    detail = "\n".join(reasons)
    if not allow_drift:
        raise ManifestDriftError(
            "these margins were not read on one rotation of the held-out stream, so every "
            "difference between them could be the stream and not the organism:\n" + detail + "\n"
            "pass --allow-manifest-drift to compare anyway, with the drift printed beside "
            "the numbers"
        )
    logger.warning("manifest drift allowed by --allow-manifest-drift:\n%s", detail)
    return "stream: DRIFT, allowed by --allow-manifest-drift:\n" + detail
