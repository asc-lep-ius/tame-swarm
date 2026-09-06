"""Run the three #12 arms at parity and print the comparison.

    mob      the auction
    softmax  the same confidence heads, softmaxed, with the economy switched off
    dense    the original FFN, no routing -- the capability-preservation floor

With ``--coupling_goal`` a fourth arm joins them: the auction with the routing
coupling seeded from that goal's certified direction (#14), #6's ablation arm,
labelled ``mob+<goal>``.

The arms differ in the gate, or in the coupling, and in nothing else, and that is
asserted rather than asserted-in-prose: each arm produces a fingerprint covering
seed, data order, converted layer range, adapter rank, step budget and held-out
split, and the comparison refuses to print if any two disagree.

By default this runs a **smoke** comparison on a randomly initialised ~45k
parameter Llama over a synthetic corpus, both built locally so the run needs no
network and no GPU. That configuration proves the harness end to end; it proves
nothing about routing, because an untrained tiny model on synthetic text has no
capability to preserve. For a real comparison pass a model and dataset::

    uv run python scripts/compare_routers.py \\
        --model_id google/gemma-2-2b-it --dataset wikitext \\
        --steps 1000 --device cuda --use_lora

One seed is one seed. The #12 measurement note found a between-seed spread of ~46
points on report decisiveness that was stable across every probe size tried, so a
single-seed table is a shakedown of the harness rather than a result. Repeat over
seeds -- and put an interval on them with #13's noise floor -- before any number
here is quoted.
"""

import argparse
import gc
import logging
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_fixture import build_smoke_fixture  # noqa: E402

from parity import arm_label, assert_parity  # noqa: E402
from train import ARM_MOB, ARMS, TAMETrainer, TrainingConfig  # noqa: E402

logger = logging.getLogger("compare_routers")


def run_arm(
    router: str, config: TrainingConfig, coupling_goal: str | None = None
) -> dict[str, object]:
    """Train one arm to completion and return its fingerprint and final metrics."""
    arm = arm_label(router, coupling_goal)
    logger.info("=" * 80)
    logger.info(f"Arm: {arm}")
    logger.info("=" * 80)

    trainer = TAMETrainer(
        replace(
            config,
            router=router,
            coupling_goal=coupling_goal,
            output_dir=f"{config.output_dir}/{arm}",
        )
    )
    trainer.setup()
    trainer.train()

    assert trainer.fingerprint is not None
    result = {
        "arm": arm,
        "fingerprint": trainer.fingerprint,
        "metrics": trainer.eval_history[-1] if trainer.eval_history else {},
    }

    # See run_seeds.run_seed: several full-size trainers share this process, and
    # without this a real model's checkpoint save on the *next* arm can fail with
    # "Cannot copy out of meta tensor" once PyTorch's cache makes VRAM look
    # scarcer than it is.
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def format_table(results: list[dict[str, object]]) -> str:
    """One row per arm, one column per metric, dashes where an arm has none.

    The dense arm has no experts to diverge and no gate to profile, so its
    specialisation cells are empty rather than zero -- a zero would read as
    "measured, and they compute the same function".
    """
    columns = [
        ("heldout_loss", "eval/loss"),
        ("heldout_ppl", "eval/perplexity"),
        ("cos_dist", "spec/expert_cosine_distance"),
        ("routing_JS", "spec/routing_js_from_corpus"),
        ("report_dec", "spec/report_decisiveness"),
    ]
    header = f"{'arm':<14}" + "".join(f"{title:>13}" for title, _ in columns)
    lines = [header, "-" * len(header)]

    for result in results:
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        row = f"{result['arm']:<14}"
        for _title, key in columns:
            value = metrics.get(key)
            row += f"{value:>13.5f}" if isinstance(value, float) else f"{'-':>13}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default=None, help="Default: a local smoke model")
    parser.add_argument("--dataset", type=str, default=None, help="Default: a local smoke corpus")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--adapter_rank", type=int, default=4)
    # 320 sequences of ~17 real tokens each clears the 4096-token probe floor the
    # #12 measurement note settled on, counting only unpadded positions. Defaults
    # that cannot satisfy their own floor warn on every run and train the reader to
    # ignore the warning.
    parser.add_argument("--held_out_sequences", type=int, default=320)
    parser.add_argument("--probe_tokens", type=int, default=4096)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--layers", type=str, default="1:3", help="MoB layer range as start:end (exclusive)"
    )
    parser.add_argument(
        "--coupling_goal",
        type=str,
        default=None,
        help=(
            "Also run the auction with the routing coupling seeded from this goal's "
            "certified direction, as a fourth arm (default: the three #12 arms only)"
        ),
    )
    args = parser.parse_args()

    workspace = (
        Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="tame-arms-"))
    )
    workspace.mkdir(parents=True, exist_ok=True)

    if args.model_id is None or args.dataset is None:
        logger.info("No model/dataset given: building the local smoke fixture")
        model_id, dataset = build_smoke_fixture(workspace)
        model_id = args.model_id or model_id
        dataset = args.dataset or dataset
    else:
        model_id, dataset = args.model_id, args.dataset

    start, end = (int(part) for part in args.layers.split(":"))
    config = TrainingConfig(
        model_id=model_id,
        output_dir=str(workspace / "runs"),
        dataset_name=dataset,
        num_experts=args.num_experts,
        adapter_rank=args.adapter_rank,
        mob_layers_start=start,
        mob_layers_end=end,
        batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.steps,
        # Proportional to the budget: at the 500-step default a short run would be
        # entirely warmup, and an arm compared at a different point on the schedule
        # is not compared at parity.
        warmup_steps=max(1, args.steps // 10),
        max_seq_length=args.max_seq_length,
        eval_steps=max(1, args.steps // 2),
        save_steps=args.steps,
        log_frequency=max(1, args.steps // 4),
        held_out_sequences=args.held_out_sequences,
        probe_tokens=args.probe_tokens,
        device=args.device,
        dtype="float32" if args.device == "cpu" else "bfloat16",
        gradient_checkpointing=False,
        use_lora=args.use_lora,
        seed=args.seed,
    )

    # Each arm gets its own output_dir (run_arm appends /<arm>), and #7's tracking
    # store defaults to output_dir-relative -- so without this, the three arms
    # write into three separate MLflow stores and can never be compared in one
    # `mlflow ui`, which defeats the point of a comparison harness. One shared
    # store, keyed off the workspace all three arms already write under.
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{workspace / 'mlruns'}")

    arms: list[tuple[str, str | None]] = [(router, None) for router in ARMS]
    if args.coupling_goal:
        arms.append((ARM_MOB, args.coupling_goal))
    results = [run_arm(router, config, coupling_goal) for router, coupling_goal in arms]
    assert_parity([result["fingerprint"] for result in results])  # pyright: ignore[reportArgumentType]

    print("\n" + format_table(results))
    print(f"\nartefacts: {workspace}")
    print(
        "\nOne seed, and the smoke fixture is an untrained model on synthetic text. "
        "This shows the harness runs at parity, not that any arm is better; repeat "
        "over seeds against a real model and dataset before quoting a number."
    )


if __name__ == "__main__":
    main()
