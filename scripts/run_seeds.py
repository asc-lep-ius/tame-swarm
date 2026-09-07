"""Run one configuration over several seeds and report mean +/- std (#13).

One seed is not a result: #12's measurement note found a ~46-point between-seed
spread on report decisiveness. This is the harness that makes "how many seeds
does an effect need to clear" answerable -- run a config N times, varying only the
seed, and report the spread on every headline metric rather than a single number
wearing the authority of one.

By default this runs a smoke sweep on the same local, no-network, no-GPU fixture
``compare_routers.py`` uses -- useful for exercising the harness itself, not for
measuring anything. For a real noise-floor measurement::

    uv run python scripts/run_seeds.py \\
        --model_id Qwen/Qwen3-1.7B --dataset wikitext \\
        --steps 1000 --device cuda --use_lora --seeds 0,1,2

Every seed shares one config except the seed itself -- unlike ``compare_routers.py``,
there is nothing here to assert parity *between*: the point of this harness is the
spread a fixed configuration produces on its own, which is exactly what a fixed
seed across arms is supposed to remove.
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_fixture import build_smoke_fixture  # noqa: E402

from coupling import DEFAULT_COUPLING_BETA, DEFAULT_WARMUP_STEPS  # noqa: E402
from parity import arm_label  # noqa: E402
from train import TAMETrainer, TrainingConfig  # noqa: E402

logger = logging.getLogger("run_seeds")

# Every eval-namespace metric a run might report; not every run reports every one
# (the dense arm has no spec/ metrics -- see compare_routers.format_table), so
# aggregation is over whichever of these each seed actually produced.
HEADLINE_METRICS = (
    "eval/loss",
    "eval/perplexity",
    "spec/expert_cosine_distance",
    "spec/routing_js_from_corpus",
    "spec/report_decisiveness",
)


def run_seed(seed: int, config: TrainingConfig) -> tuple[dict[str, float], dict[str, object]]:
    """Train one replicate to completion; its final headline metrics and its arm fingerprint.

    The fingerprint travels with the summary so that ``compare_runs.py`` can refuse
    a comparison between two groups that differ in anything but the variable under
    test -- the same guard ``compare_routers.py`` applies within one process.

    Explicitly frees the model and empties CUDA's caching allocator before
    returning: this loops several full-size trainers through one process, and a
    real (non-smoke) model leaves enough of PyTorch's cache reserved-but-unused
    that the *next* trainer's ``device_map="auto"`` balanced-memory calculation
    sees less free VRAM than is actually available and offloads part of the
    model to the meta device -- which then fails at checkpoint time with
    "Cannot copy out of meta tensor". Reproduced empirically: seed 0 of a real
    Qwen3-1.7B sweep saved cleanly, seed 1 (same process, no cleanup) did not.
    """
    logger.info("=" * 80)
    logger.info(f"Seed: {seed}")
    logger.info("=" * 80)

    trainer = TAMETrainer(replace(config, seed=seed, output_dir=f"{config.output_dir}/seed{seed}"))
    trainer.setup()
    trainer.train()

    final = trainer.eval_history[-1] if trainer.eval_history else {}
    result = {key: final[key] for key in HEADLINE_METRICS if key in final}
    assert trainer.fingerprint is not None
    fingerprint = trainer.fingerprint.as_dict()

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, fingerprint


def aggregate(per_seed: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and (sample) standard deviation per metric, over whichever seeds have it.

    Sample std (n-1 denominator) rather than population std (n): three replicates
    is the minimum this project ever quotes, and the population denominator would
    understate the spread exactly where the estimate is shakiest. A metric present
    in only one seed has no std to report -- it prints as a single point, not a 0.0
    that would misread as "measured, no spread".
    """
    stats: dict[str, dict[str, float]] = {}
    for metric in HEADLINE_METRICS:
        values = [result[metric] for result in per_seed.values() if metric in result]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = math.sqrt(variance)
        else:
            std = float("nan")
        stats[metric] = {"mean": mean, "std": std, "n": len(values)}
    return stats


def format_table(stats: dict[str, dict[str, float]]) -> str:
    header = f"{'metric':<32}{'mean':>12}{'std':>12}{'n':>4}"
    lines = [header, "-" * len(header)]
    for metric, values in stats.items():
        std_str = f"{values['std']:>12.5f}" if not math.isnan(values["std"]) else f"{'n=1':>12}"
        lines.append(f"{metric:<32}{values['mean']:>12.5f}{std_str}{values['n']:>4.0f}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default=None, help="Default: a local smoke model")
    parser.add_argument("--dataset", type=str, default=None, help="Default: a local smoke corpus")
    parser.add_argument("--router", type=str, default="mob", choices=["mob", "softmax", "dense"])
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seed list")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--adapter_rank", type=int, default=4)
    parser.add_argument("--held_out_sequences", type=int, default=320)
    parser.add_argument("--probe_tokens", type=int, default=4096)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--layers", type=str, default="1:3", help="MoB layer range as start:end (exclusive)"
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force deterministic kernels where one exists (default: on)",
    )
    # The coupled arm of #6's ablation: the same auction, with the routing
    # coupling seeded from a certified direction (#14). Everything else is shared
    # with the uncoupled arm, which is what makes the two summaries comparable.
    parser.add_argument(
        "--coupling_goal",
        type=str,
        default=None,
        help=(
            "Seed the routing coupling from this goal's certified direction at its "
            "certified layers only (default: routing stays uncoupled)"
        ),
    )
    parser.add_argument("--coupling_beta", type=float, default=DEFAULT_COUPLING_BETA)
    parser.add_argument("--coupling_warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    args = parser.parse_args()

    seeds = [int(part) for part in args.seeds.split(",")]
    if len(seeds) < 2:
        logger.warning(
            f"Only {len(seeds)} seed(s) given -- a spread needs at least two, and the "
            "project's own floor for a published number is three (#13)"
        )

    workspace = (
        Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="tame-seeds-"))
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
        router=args.router,
        num_experts=args.num_experts,
        adapter_rank=args.adapter_rank,
        mob_layers_start=start,
        mob_layers_end=end,
        batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.steps,
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
        deterministic=args.deterministic,
        coupling_goal=args.coupling_goal,
        coupling_beta=args.coupling_beta,
        coupling_warmup_steps=args.coupling_warmup_steps,
    )

    # One shared MLflow store across seeds, same reasoning as compare_routers.py:
    # otherwise each seed's run lands in its own file store and `mlflow ui` can
    # never show them side by side.
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{workspace / 'mlruns'}")

    runs = {seed: run_seed(seed, config) for seed in seeds}
    per_seed = {seed: metrics for seed, (metrics, _) in runs.items()}
    fingerprints = {seed: fingerprint for seed, (_, fingerprint) in runs.items()}
    stats = aggregate(per_seed)

    arm = arm_label(args.router, args.coupling_goal)
    print("\n" + format_table(stats))
    print(f"\narm: {arm} | seeds: {seeds} | steps: {args.steps}")
    print(f"artefacts: {workspace}")

    summary_path = workspace / "seed_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "router": args.router,
                "coupling_goal": args.coupling_goal,
                "seeds": seeds,
                "steps": args.steps,
                "per_seed": per_seed,
                "fingerprints": fingerprints,
                "stats": stats,
            },
            indent=2,
        )
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
