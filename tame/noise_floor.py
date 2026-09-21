"""The run-to-run floor: what it is a property of, and when it may be borrowed (#31, #56).

A replicate is one seed run twice. It measures the run-to-run floor and nothing
else, and it costs a quarter of a four-run group. #31 closed the nondeterminism
that made the floor worth measuring every time -- it named the kernel (the
memory-efficient attention backward), fixed it with ``--deterministic strict``,
and put the code SHA in the fingerprint -- and #39's body sweep then spent six
replicates, about two GPU-hours, to record a floor of exactly zero in all six
groups. Preregistration section 8, rule 4 is what came of that: replicates are
spent where the floor is *unmeasured*, and a sweep at a configuration whose floor
is already recorded names the floor it borrows instead of re-measuring it.

The clause that makes the rule safe is "at a configuration". A floor is a
property of the kernels a configuration selects, so it travels between two arms
of one sweep -- they differ in the ledger's arithmetic, not in which kernels run
-- and does not travel to a run at another shape, precision or device.
:data:`FLOOR_KNOBS` is that boundary, written as a list rather than left to
judgement, and :data:`NOT_A_FLOOR_KNOB` is its complement with the reason each
field is in it: a field that is merely absent is indistinguishable from one that
was forgotten, which is the argument ``parity.NOT_A_CONFOUND`` already makes for
the parity check.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from parity import ArmFingerprint, code_drift

logger = logging.getLogger(__name__)

# What a floor is a property of: the shape, the precision, the device and the
# kernel set. Two runs that agree on all of these run the same kernels over the
# same tensors, which is what makes one run's measured spread the other's floor.
FLOOR_KNOBS = frozenset(
    {
        "router",
        "model_id",
        "dtype",
        "device",
        "dataset",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "num_experts",
        "top_k",
        "adapter_rank",
        "requested_layers",
        "converted_layers",
        "use_lora",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "gradient_checkpointing",
        "deterministic",
        "strict_determinism",
    }
)

_VARIES = "varies between the sweep that measured the floor and the sweep that borrows it; \
requiring equality here would let a floor be borrowed only by the seed that measured it"
_ARM = "the arm's own variable: a floor is borrowed across arms of one sweep, or it buys nothing"
_OPTIMISER = "decides what the weights become, never which kernels compute them"
_MEASUREMENT = "read under no_grad with the economy frozen, after the trajectory it would have \
to move"
_CODE = "decided by code_drift, which borrow_floor runs on the lender and each borrower: a \
missing SHA has to count as drift, and a field-equality check would read two absences as agreement"
_REGISTER = "a register of what the tissue was allowed to do, not of what the arithmetic did"

# Every remaining fingerprint field, with the reason a floor does not depend on
# it. Pinned against the dataclass by ``tests/test_noise_floor.py``, so a field
# added later cannot join neither list by nobody thinking about it.
NOT_A_FLOOR_KNOB: dict[str, str] = {
    **dict.fromkeys(("seed", "data_order"), _VARIES),
    **dict.fromkeys(
        (
            "coupling_goal",
            "coupling_beta",
            "coupling_warmup_steps",
            "steer_goal",
            "steer_strength",
            "steer_layers",
            "persistence_coupling",
            "goal_doses",
            "goal_fields",
            "ledger_mode",
            "exploration_rate",
            "exploration_draw",
            "wealth_update_frequency",
        ),
        _ARM,
    ),
    **dict.fromkeys(
        (
            "learning_rate",
            "warmup_steps",
            "weight_decay",
            "calibration_loss_weight",
            "confidence_head_learning_rate",
        ),
        _OPTIMISER,
    ),
    **dict.fromkeys(("probe_tokens", "eval_split"), _MEASUREMENT),
    **dict.fromkeys(("code_sha", "code_dirty"), _CODE),
    **dict.fromkeys(
        (
            "autonomy_plasticity",
            "autonomy_exploration",
            "autonomy_setpoints",
            "autonomy_evaluation",
            "autonomy_dormancy",
            "rotating_stream",
            "rotating_stream_date",
            "rotating_refresh_days",
            "rotating_stream_overdue_days",
            "rotating_stream_cutoff",
            "canary_set",
            "canary_set_date",
            "canary_refresh_days",
        ),
        _REGISTER,
    ),
}


class BorrowedFloorError(ValueError):
    """Raised when a recorded floor may not stand in for the one a sweep did not measure."""


def unclassified_floor_fields() -> tuple[str, ...]:
    """Fingerprint fields that are neither a floor knob nor declared not to be one."""
    classified = FLOOR_KNOBS | set(NOT_A_FLOOR_KNOB)
    return tuple(field.name for field in fields(ArmFingerprint) if field.name not in classified)


def floor_knobs(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """The knob values a floor is a property of, as JSON round-trips them.

    Sequences come back from ``seed_summary.json`` as lists and go out as
    tuples, so both sides are coerced before they are compared -- the same trap
    ``parity.SEQUENCE_FIELDS`` exists for, met here on a different path.
    """
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in sorted(fingerprint.items())
        if key in FLOOR_KNOBS
    }


def differing_floor_knobs(recorded: dict[str, Any], own: dict[str, Any]) -> tuple[str, ...]:
    """Which floor knobs two fingerprints disagree on; empty when the floor travels."""
    left, right = floor_knobs(recorded), floor_knobs(own)
    return tuple(sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key)))


@dataclass(frozen=True)
class BorrowedFloor:
    """A floor measured by another sweep, and the evidence it applies to this one."""

    path: str
    arm: str
    replicate_seed: int | None
    replication_std: dict[str, float]
    knobs: dict[str, Any]
    code_sha: str | None = None
    code_dirty: bool | None = None
    # What `--allow-code-drift` waived, empty when it waived nothing. Recorded
    # for `compare_runs.assert_same_code`'s reason: a comparison made across
    # drift is made, and never made silently. Without this the escape leaves no
    # trace at all -- the printed line reads as unqualified agreement and the
    # summary says only what SHA the lender had, not that it disagreed.
    code_drift_allowed: list[str] = field(default_factory=list)

    @property
    def is_zero(self) -> bool:
        return all(value == 0.0 for value in self.replication_std.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "arm": self.arm,
            "replicate_seed": self.replicate_seed,
            "replication_std": self.replication_std,
            "knobs": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.knobs.items()
            },
            # Outside `knobs` because it is not a floor knob: the code identity
            # is decided by `code_drift` rather than by field equality. It is
            # recorded all the same, or a reader of a borrowed floor has no way
            # to ask what kernels measured it.
            "code_sha": self.code_sha,
            "code_dirty": self.code_dirty,
            "code_drift_allowed": list(self.code_drift_allowed),
            "is_zero": self.is_zero,
        }


def _drift_between(lender: dict[str, Any], own: dict[str, Any]) -> list[str]:
    """``parity.code_drift`` on a recorded lender and one borrower; empty when they agree.

    A schema this version cannot build is drift too, and said as such: a
    fingerprint nobody can reconstruct is a fingerprint nobody can check, and
    the alternative is a silent pass.
    """
    try:
        arms = [ArmFingerprint(**lender), ArmFingerprint(**own)]
    except TypeError as exc:
        return [f"  a fingerprint does not match this version's schema ({exc})"]
    return code_drift(arms)


def borrow_floor(
    path: Path,
    fingerprints: dict[Any, dict[str, Any]],
    allow_code_drift: bool = False,
) -> BorrowedFloor:
    """The floor recorded under ``path``, checked against the fingerprints borrowing it.

    Refuses four ways, because each is a different mistake. A summary with no
    recorded floor has nothing to lend -- including one that borrowed its own,
    since a floor relayed twice is a floor nobody measured at the configuration
    that quotes it. A summary whose replicate failed recorded the failure, not a
    floor. A summary at different floor knobs measured a different
    configuration's kernels, which is rule 4's whole clause.

    And the fourth is the one the floor knobs cannot state. A floor is a property
    of the kernels a configuration selects, and the code is what selects them --
    #31's whole finding. ``code_sha`` and ``code_dirty`` are deliberately not
    floor knobs, because a *missing* SHA has to count as drift and a
    field-equality check reads two absences as agreement; so the check is
    ``parity.code_drift``, run here on the lender against each borrower.
    ``allow_code_drift`` is the operator saying it anyway, and it mirrors
    ``compare_runs.py --allow-code-drift`` so the runs recorded before #31 stay
    borrowable when somebody names the decision -- including that escape's other
    half, which is that the waived reasons are logged and carried on the
    borrowed floor. A waiver nothing records is the silence the check replaced.
    """
    summary_path = path / "seed_summary.json"
    if not summary_path.exists():
        raise BorrowedFloorError(
            f"no seed_summary.json under {path} -- --floor_recorded_at names a directory "
            "written by scripts/run_seeds.py whose replicate ran"
        )
    summary = json.loads(summary_path.read_text())
    recorded = summary.get("replication_std")
    if not recorded:
        borrowed = summary.get("floor_recorded_at")
        why = "it borrowed its own floor" if borrowed else "it recorded none"
        raise BorrowedFloorError(
            f"{summary_path} has no measured floor to lend: {why}"
            + (f" ({summary['replication_error']})" if summary.get("replication_error") else "")
        )
    prints = summary.get("fingerprints") or {}
    if not prints:
        raise BorrowedFloorError(
            f"{summary_path} carries no arm fingerprints, so the configuration its floor was "
            "measured at cannot be read (every summary recorded before #6)"
        )
    lender = next(iter(prints.values()))
    waived: list[str] = []
    for seed, fingerprint in fingerprints.items():
        differing = differing_floor_knobs(lender, fingerprint)
        if differing:
            theirs, ours = floor_knobs(lender), floor_knobs(fingerprint)
            raise BorrowedFloorError(
                f"the floor recorded at {path} was measured at other knobs than this sweep's "
                f"seed {seed} runs, so it is not this configuration's floor: "
                + ", ".join(f"{key} {theirs.get(key)!r} vs {ours.get(key)!r}" for key in differing)
            )
        # After the knobs, because a sweep at another shape is the coarser
        # mismatch and its message is the more useful one when both apply.
        drift = _drift_between(lender, fingerprint)
        if drift and not allow_code_drift:
            raise BorrowedFloorError(
                f"the floor recorded at {path} was measured by other code than this sweep's "
                f"seed {seed} runs, and a floor is a property of the kernels the code selects "
                "(#31). Pass --allow-code-drift to borrow it anyway:\n" + "\n".join(drift)
            )
        for reason in drift:
            if reason not in waived:
                waived.append(reason)
    if waived:
        logger.warning(
            "code drift allowed by --allow-code-drift; the floor borrowed from %s was not "
            "measured by this sweep's code:\n%s",
            path,
            "\n".join(waived),
        )
    return BorrowedFloor(
        path=str(path),
        arm=summary.get("arm", "unknown"),
        replicate_seed=summary.get("replicate_seed"),
        replication_std=dict(recorded),
        knobs=floor_knobs(lender),
        code_sha=lender.get("code_sha"),
        code_dirty=lender.get("code_dirty"),
        code_drift_allowed=waived,
    )
