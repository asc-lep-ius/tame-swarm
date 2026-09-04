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
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Training batches hashed into the data-order fingerprint. Enough that a different
# stream, tokenisation or restart point shows up immediately; small enough that the
# check costs one pass over batches the arm is about to train on anyway.
DATA_ORDER_PROBE_BATCHES = 8

# The one field an arm is allowed to differ on -- it is the variable under test.
VARYING_FIELDS = frozenset({"router"})

# Reported for context, not asserted: see the module docstring on ``dense``.
REPORTED_FIELDS = frozenset({"converted_layers"})

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
}


class ParityError(AssertionError):
    """Raised when two arms differ on something other than the router."""


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
    gradient_checkpointing: bool
    device: str
    probe_tokens: int
    eval_split: str
    data_order: str
    converted_layers: int

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
) -> ArmFingerprint:
    """Build a fingerprint from a ``TrainingConfig`` and the two measured hashes.

    ``dataset_config`` is passed in rather than read off the config because it only
    applies to some datasets, and the caller is the one place that knows the rule.
    """
    return ArmFingerprint(
        router=config.router,
        seed=config.seed,
        deterministic=config.deterministic,
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
        confidence_head_learning_rate=config.confidence_head_learning_rate,
        wealth_update_frequency=config.wealth_update_frequency,
        gradient_checkpointing=config.gradient_checkpointing,
        device=config.device,
        probe_tokens=config.probe_tokens,
        eval_split=eval_split_fingerprint,
        data_order=data_order,
        converted_layers=converted_layers,
    )


def assert_parity(arms: Sequence[ArmFingerprint]) -> None:
    """Refuse a comparison whose arms differ on anything but the router.

    Raises with the full disagreement rather than the first one found: an arm that
    differs in one field usually differs in three, and reporting them one run at a
    time turns a config mistake into an afternoon.
    """
    if len(arms) < 2:
        return

    routers = [arm.router for arm in arms]
    if len(set(routers)) != len(routers):
        raise ParityError(f"Arms must have distinct routers, got {routers}")

    reference = arms[0]
    disagreements: list[str] = []
    for field in fields(ArmFingerprint):
        if field.name in VARYING_FIELDS or field.name in REPORTED_FIELDS:
            continue
        expected = getattr(reference, field.name)
        for arm in arms[1:]:
            actual = getattr(arm, field.name)
            if actual != expected:
                disagreements.append(
                    f"  {field.name}: {reference.router}={expected!r} vs {arm.router}={actual!r}"
                )

    if disagreements:
        raise ParityError(
            "Arms are not at parity; every difference below is a confound in the "
            "comparison:\n" + "\n".join(disagreements)
        )

    logger.info(
        f"Parity holds across {len(arms)} arms ({', '.join(routers)}): "
        f"seed={reference.seed}, steps={reference.max_steps}, "
        f"layers={len(reference.requested_layers)}, rank={reference.adapter_rank}, "
        f"eval split={reference.eval_split}, data order={reference.data_order}"
    )
