"""#62's gate decomposition on #39's body checkpoints: one forward, nothing paid.

The fixture half (``scripts/gate_decomposition.py``) reads the gate on a market
it drives itself. This reads it on the 24 frozen checkpoints #39's body run left
at ``79c1b17`` -- three arms x two dose ratios x four runs -- by restoring each
arm as #29 taught (``mob_modules.pt`` beside the LoRA adapters), reproducing its
recorded ``eval/loss`` first as #58 did, and then running its own held-out probe
once with the economy frozen, recording per MoB layer the confidences beside the
wealth the gate read. The wealth term is constant per cell on ``decoupled`` by
construction (a pinned ledger), which is the built-in control; ``seed0`` against
``seed0-replicate`` in every group is the floor.

    uv run python scripts/gate_decomposition_body.py \\
        --body-root ~/tame-runs/39-stakes-dial/body --out ~/tame-runs/gate-decomposition/

    uv run python scripts/gate_decomposition_body.py --checkpoint <run>/checkpoint-2000

Forward only, under a stated GPU budget: the run stops, and says so, at the first
checkpoint that would carry it past ``--gpu-minutes``. Nothing is written into
#39's run directories; the readings mirror their layout under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from gate_decomposition import READ_FIELDS, SENIORITY_LINE, _fmt, script_identity  # noqa: E402

from evaluation import HeldOutSplit, evaluate  # noqa: E402
from mob import frozen_economy, mob_layers_by_index  # noqa: E402
from mob.gate_decomposition import GateDecomposition, decompose, merge  # noqa: E402

DEFAULT_PROBE_TOKENS = 4096
# #58's reproduction tolerance: the recorded eval/loss is a bfloat16 cross-entropy
# and the reproduction differs from it in the fourth decimal on this body.
DEFAULT_LOSS_TOLERANCE = 2e-3
DEFAULT_GPU_MINUTES = 15.0
REFERENCE_RATIO = "r1"
REPLICATE = "seed0-replicate"


def read_checkpoint(
    checkpoint: Path, device_name: str
) -> tuple[torch.nn.Module, HeldOutSplit, dict[str, Any]]:
    """Rebuild the arm from what it recorded, restore it whole, and load its own probe.

    The config travels in ``training_state.pt``; the goal fields are not
    re-calibrated because a field prices realised value and never reaches the
    forward -- the reproduced ``eval/loss`` is what shows the arm is the one that
    trained. The MLflow store is pointed at scratch first, so a read leaves #39's
    run directory exactly as it found it.
    """
    from train import TAMETrainer, TrainingConfig, restore_checkpoint  # noqa: PLC0415

    # Set, not defaulted: a read must never log into whatever store the shell
    # happens to point at.
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{Path(tempfile.gettempdir()) / 'tame-gate-mlruns'}"
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    recorded = dict(state["config"])
    run_dir = checkpoint.parent
    recorded.update(
        output_dir=str(run_dir),
        device=device_name,
        goal_fields=(),
        goal_doses=(),
        gradient_checkpointing=False,
    )
    known = set(TrainingConfig.__dataclass_fields__)
    trainer = TAMETrainer(TrainingConfig(**{k: v for k, v in recorded.items() if k in known}))
    trainer.setup()
    assert trainer.model is not None
    restore_checkpoint(trainer.model, checkpoint)
    split = HeldOutSplit.load(run_dir / "held_out_split.pt")
    provenance = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "global_step": state.get("global_step"),
        "arm_fingerprint": state.get("arm_fingerprint"),
        "eval_split": split.fingerprint,
        "recorded_eval_loss": next(
            (
                entry["eval/loss"]
                for entry in reversed(state.get("eval_history") or [])
                if "eval/loss" in entry
            ),
            None,
        ),
    }
    return trainer.model, split, provenance


def verify_recorded_loss(
    model: torch.nn.Module,
    split: HeldOutSplit,
    batch_size: int,
    device: torch.device,
    provenance: dict[str, Any],
    tolerance: float,
) -> None:
    """The restored arm reproduces the loss the run recorded, or nothing is read from it."""
    recorded = provenance.get("recorded_eval_loss")
    reproduced = evaluate(model, split, batch_size, device).loss
    provenance["reproduced_eval_loss"] = reproduced
    if recorded is None:
        raise ValueError(f"{provenance['checkpoint']} records no eval/loss to reproduce")
    if abs(reproduced - recorded) > tolerance:
        raise ValueError(
            f"the restored arm reads {reproduced:.6f} where the run recorded {recorded:.6f} "
            f"(tolerance {tolerance}); it is not the arm that trained"
        )


def probe_batches(
    split: HeldOutSplit, batch_size: int, probe_tokens: int
) -> list[dict[str, torch.Tensor]]:
    """The run's own held-out split, cut by whole batches at the probe's token budget."""
    batches: list[dict[str, torch.Tensor]] = []
    seen = 0
    for batch in split.batches(batch_size):
        batches.append(batch)
        seen += int(batch["attention_mask"].sum())
        if seen >= probe_tokens:
            break
    return batches


def decompose_checkpoint(
    model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], device: torch.device
) -> dict[str, Any]:
    """One forward over the probe; per MoB layer, the gate split on every unpadded token.

    Read inside ``frozen_economy`` because it restores ``last_stats`` on exit,
    and in ``eval()`` so no exploration slot is handed out -- the sold set and the
    winner set coincide here, which the fixture's training forwards do not give.
    """
    layers = mob_layers_by_index(model)
    parts: dict[int, list[GateDecomposition]] = {index: [] for index in layers}
    wealth_read: dict[int, list[float]] = {}
    bid_dtype = ""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), frozen_economy(model):
            for batch in batches:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                for index, layer in layers.items():
                    stats = layer.last_stats
                    assert stats is not None, f"layer {index} recorded no stats"
                    wealth = layer.allocation_wealth().detach().cpu()
                    wealth_read[index] = wealth.tolist()
                    bid_dtype = str(stats.confidences.dtype)
                    parts[index].append(
                        decompose(
                            stats.confidences.cpu(),
                            wealth,
                            stats.selected_experts.cpu(),
                            layer.config.top_k,
                            mask=attention_mask.cpu(),
                        )
                    )
    finally:
        model.train(was_training)
    per_layer = {str(index): asdict(merge(layer_parts)) for index, layer_parts in parts.items()}
    return {
        "layers": per_layer,
        "overall": asdict(merge([part for layer_parts in parts.values() for part in layer_parts])),
        "wealth_read": {str(index): wealth for index, wealth in wealth_read.items()},
        "bid_dtype": bid_dtype,
        "top_k": next(iter(layers.values())).config.top_k,
    }


def read_one(checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    """One checkpoint, with the forward passes timed apart from the load.

    The budget is GPU-minutes, and the load -- ``from_pretrained``, the
    conversion, the LoRA wrap -- is CPU time the card spends idle; only the
    reproduction and the probe are what the budget prices.
    """
    device = torch.device(args.device)
    started = time.monotonic()
    model, split, provenance = read_checkpoint(checkpoint, args.device)
    loaded = time.monotonic()
    verify_recorded_loss(model, split, args.batch_size, device, provenance, args.loss_tolerance)
    batches = probe_batches(split, args.batch_size, args.probe_tokens)
    reading = decompose_checkpoint(model, batches, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reading.update(
        provenance=provenance,
        probe_batches=len(batches),
        gpu_seconds=time.monotonic() - loaded,
        wall_seconds=time.monotonic() - started,
        **script_identity(Path(__file__)),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reading


def _label(checkpoint: Path) -> tuple[str, str, str]:
    """``.../r1/value/runs/seed0/checkpoint-2000`` -> (r1, value, seed0)."""
    run = checkpoint.parent
    return run.parent.parent.parent.name, run.parent.parent.name, run.name


def _floor(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    return {
        field: abs(float(first["overall"][field]) - float(second["overall"][field]))
        for field in READ_FIELDS
    }


def _row(label: str, reading: dict[str, Any]) -> str:
    overall = reading["overall"]
    layers = " ".join(
        f"{index}:{_fmt(layer['seniority_fraction']).strip()}"
        for index, layer in reading["layers"].items()
    )
    return (
        f"  {label:<26}"
        + "".join(_fmt(float(overall[field])) for field in READ_FIELDS)
        + f"  loss {reading['provenance']['reproduced_eval_loss']:.4f}  layers {layers}"
    )


def run_body(args: argparse.Namespace) -> None:
    checkpoints = sorted(args.body_root.glob("r*/*/runs/seed*/checkpoint-*"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {args.body_root}")
    identity = script_identity(Path(__file__))
    print(
        f"code {identity['code_sha']} dirty={identity['code_dirty']} "
        f"script {identity['script_sha1']}"
    )
    print(
        f"  {len(checkpoints)} checkpoints, probe {args.probe_tokens} tokens, "
        f"budget {args.gpu_minutes} min"
    )
    print("  " + " " * 26 + "".join(f"{field[:6]:>7}" for field in READ_FIELDS))
    started = time.monotonic()
    readings: dict[tuple[str, str, str], dict[str, Any]] = {}
    unread: list[str] = []
    gpu_minutes = 0.0
    for checkpoint in checkpoints:
        per_checkpoint = gpu_minutes / len(readings) if readings else 0.0
        if readings and gpu_minutes + per_checkpoint > args.gpu_minutes:
            unread.append(str(checkpoint))
            continue
        ratio, arm, seed = _label(checkpoint)
        reading = read_one(checkpoint, args)
        readings[(ratio, arm, seed)] = reading
        gpu_minutes += reading["gpu_seconds"] / 60.0
        out_dir = args.out / "body" / ratio / arm / seed
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate_decomposition.json").write_text(json.dumps(reading, indent=2, default=str))
        print(_row(f"{ratio} {arm} {seed}", reading))
    floors = {
        f"{ratio}/{arm}": _floor(readings[(ratio, arm, "seed0")], readings[(ratio, arm, REPLICATE)])
        for ratio, arm, seed in list(readings)
        if seed == "seed0" and (ratio, arm, REPLICATE) in readings
    }
    for group, floor in floors.items():
        print(f"  floor {group:<20}" + "".join(_fmt(v) for v in floor.values()))
    primary = [
        reading["overall"]["seniority_fraction"]
        for (ratio, arm, seed), reading in readings.items()
        if ratio == REFERENCE_RATIO and arm == "value" and seed != REPLICATE
    ]
    print(
        f"\n== primary: value {REFERENCE_RATIO} seniority "
        + ", ".join(_fmt(f) for f in primary)
        + f" against the {SENIORITY_LINE:.2f} line =="
    )
    if unread:
        print(
            f"\n{len(unread)} checkpoints NOT read: "
            f"the {args.gpu_minutes} GPU-minute budget was reached"
        )
    summary = {
        **identity,
        "seniority_line": SENIORITY_LINE,
        "probe_tokens": args.probe_tokens,
        "gpu_minutes": gpu_minutes,
        "wall_minutes": (time.monotonic() - started) / 60.0,
        "floors": floors,
        "unread": unread,
        "readings": {"/".join(key): value for key, value in readings.items()},
    }
    (args.out / "body" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"  GPU {gpu_minutes:.1f} min of the {args.gpu_minutes} budget; "
        f"wall {summary['wall_minutes']:.1f} min"
    )
    print(f"record: {args.out / 'body' / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="one checkpoint directory")
    parser.add_argument("--body-root", type=Path, help="#39's body run directory, all 24")
    parser.add_argument(
        "--out", type=Path, default=Path.home() / "tame-runs" / "gate-decomposition"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-tokens", type=int, default=DEFAULT_PROBE_TOKENS)
    parser.add_argument("--loss-tolerance", type=float, default=DEFAULT_LOSS_TOLERANCE)
    parser.add_argument("--gpu-minutes", type=float, default=DEFAULT_GPU_MINUTES)
    args = parser.parse_args()
    if args.body_root is not None:
        run_body(args)
        return
    if args.checkpoint is None:
        parser.error("one of --checkpoint or --body-root is required")
    reading = read_one(args.checkpoint, args)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "gate_decomposition.json"
    target.write_text(json.dumps(reading, indent=2, default=str))
    print(_row(args.checkpoint.parent.name, reading))
    print(f"record: {target}")


if __name__ == "__main__":
    main()
