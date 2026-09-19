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
VARYING_FIELDS = frozenset({"router", "coupling_goal", "steer_goal", "persistence_coupling"})

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
    # new name, so a legacy fingerprint reads as ``value``. The dial varies; the
    # doses of the goal fields the arms were paid under (#33) are asserted equal,
    # as the coupling's dose is: two arms at different doses are a sweep, not the
    # contrast. Empty when no goal field was attached.
    persistence_coupling: str = "value"
    goal_doses: tuple[float, ...] = ()
    # #40. What the ledger relaxes toward. Every run recorded before the mode
    # existed relaxed toward zero, so a legacy fingerprint reads as ``decay``.
    # Deliberately *not* a varying field: #26's setpoint ledger is derived and
    # measured, not adopted, and two arms whose ledgers have different fixed
    # points and different ruin thresholds are not comparable on anything the
    # economy does. It is here because ``MoBConfig`` is invisible to this
    # fingerprint -- the field is carried on ``TrainingConfig`` for that reason,
    # and #25's two arms ran different code and fingerprinted equal.
    ledger_mode: str = "decay"
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
    reads it (#42) outlives any one training run.
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


def assert_parity(arms: Sequence[ArmFingerprint]) -> None:
    """Refuse a comparison whose arms differ on anything but the variables under test.

    Raises with the full disagreement rather than the first one found: an arm that
    differs in one field usually differs in three, and reporting them one run at a
    time turns a config mistake into an afternoon.
    """
    if len(arms) < 2:
        return

    labels = [arm.arm for arm in arms]
    if len(set(labels)) != len(labels):
        raise ParityError(
            f"Arms must be distinct in router, coupling goal or steer goal, got {labels}"
        )

    reference = arms[0]
    disagreements: list[str] = []
    for field in fields(ArmFingerprint):
        if field.name in (
            VARYING_FIELDS | REPORTED_FIELDS | FIELD_FIELDS | COUPLING_FIELDS | DRIFT_FIELDS
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
