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

It did not remove all of it (#31). Under ``--deterministic warn``, the mode every
arm before #31 ran under, the attention backward is non-deterministic, so one
seed run twice is two trajectories; the between-seed spread above already
contains that, but nothing said how much of it was run-to-run. ``--replicate``
(on by default) runs the first seed a second time and records the pair's spread
as ``replication_std``, which ``compare_runs.py`` quotes beside every delta.
Under ``--deterministic strict``, the default since #31, the pair is bitwise
identical and the floor reads zero -- which is then the check that it still does.
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
from determinism import DETERMINISM_DEFAULT, DETERMINISM_MODES  # noqa: E402
from mob.auction import EXPLORATION_DRAW_STALENESS, SUPPORTED_EXPLORATION_DRAWS  # noqa: E402
from mob.ledger import PERSISTENCE_VALUE, SUPPORTED_PERSISTENCE_COUPLINGS  # noqa: E402
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
    "spec/expert_contribution_cosine_distance",
    "spec/expert_contribution_norm_ratio",
    "spec/routing_js_from_corpus",
    "spec/report_decisiveness",
)
# The per-expert routing columns #24's contrast is differenced over
# (``routing/goal_correlation_e<i>``, ``routing/win_share_e<i>``); how many there
# are depends on ``--num_experts``, so they are matched by prefix.
HEADLINE_PREFIXES = ("routing/",)


def headline_metrics(final: dict[str, float]) -> dict[str, float]:
    """The metrics a summary carries: the fixed headline set plus every routing column."""
    return {
        key: value
        for key, value in final.items()
        if key in HEADLINE_METRICS or key.startswith(HEADLINE_PREFIXES)
    }


def run_seed(
    seed: int, config: TrainingConfig, replicate: bool = False
) -> tuple[dict[str, float], dict[str, object]]:
    """Train one seed to completion; its final headline metrics and its arm fingerprint.

    A ``replicate`` is the same seed again in its own output directory, so it
    rebuilds its held-out split and re-hashes its data order rather than reading
    the first run's -- the fingerprint equality the caller asserts is then a
    measurement and not a copy.

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
    run_name = f"seed{seed}-replicate" if replicate else f"seed{seed}"
    logger.info("=" * 80)
    logger.info(f"Seed: {seed}" + (" (replicate: the run-to-run floor)" if replicate else ""))
    logger.info("=" * 80)

    trainer = TAMETrainer(replace(config, seed=seed, output_dir=f"{config.output_dir}/{run_name}"))
    trainer.setup()
    trainer.train()

    final = trainer.eval_history[-1] if trainer.eval_history else {}
    result = headline_metrics(final)
    assert trainer.fingerprint is not None
    fingerprint = trainer.fingerprint.as_dict()

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, fingerprint


def measure_replication(
    seed: int, config: TrainingConfig, first_metrics: dict[str, float], first_fingerprint: dict
) -> tuple[dict[str, float] | None, dict[str, float] | None, str | None]:
    """Run ``seed`` a second time; its metrics, the floor, and why there is none.

    The replicate is the last full-size trainer in a process that has already
    run every seed, so it is the one most exposed to the meta-device failure
    ``run_seed`` describes, and its fingerprint is re-read from git, so a commit
    made during a multi-hour sweep makes it a different arm. Neither may cost the
    seeds already measured: a floor that could not be measured is recorded as
    unmeasured (``compare_runs.py`` prints ``n/a``) and the summary is still
    written.
    """
    try:
        metrics, fingerprint = run_seed(seed, config, replicate=True)
    except Exception as exc:
        logger.error(f"the replicate of seed {seed} failed; the floor is unmeasured: {exc}")
        return None, None, f"the replicate of seed {seed} failed: {exc}"
    if fingerprint != first_fingerprint:
        why = (
            f"the replicate of seed {seed} is not the same arm as its first run, so its "
            f"spread is not a run-to-run floor: {first_fingerprint} vs {fingerprint}"
        )
        logger.error(why)
        return metrics, None, why
    return metrics, replication_std(first_metrics, metrics), None


def aggregate(per_seed: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and (sample) standard deviation per metric, over whichever seeds have it.

    Sample std (n-1 denominator) rather than population std (n): three replicates
    is the minimum this project ever quotes, and the population denominator would
    understate the spread exactly where the estimate is shakiest. A metric present
    in only one seed has no std to report -- it prints as a single point, not a 0.0
    that would misread as "measured, no spread".
    """
    stats: dict[str, dict[str, float]] = {}
    metrics = sorted({metric for result in per_seed.values() for metric in result})
    for metric in metrics:
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


def replication_std(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    """The run-to-run floor per metric: the sample std of one seed's two runs.

    Two values have one degree of freedom, so the sample std is |a - b| / sqrt(2)
    -- the same estimator ``aggregate`` uses across seeds, at n = 2, which is what
    lets ``compare_runs.py`` pool it with the between-seed spread on one footing.
    """
    return {
        metric: abs(first[metric] - second[metric]) / math.sqrt(2)
        for metric in sorted(first)
        if metric in second
    }


def format_table(
    stats: dict[str, dict[str, float]], replication: dict[str, float] | None = None
) -> str:
    header = f"{'metric':<32}{'mean':>12}{'std':>12}{'n':>4}"
    if replication is not None:
        header += f"{'repl_std':>12}"
    lines = [header, "-" * len(header)]
    for metric, values in stats.items():
        std_str = f"{values['std']:>12.5f}" if not math.isnan(values["std"]) else f"{'n=1':>12}"
        line = f"{metric:<32}{values['mean']:>12.5f}{std_str}{values['n']:>4.0f}"
        if replication is not None:
            line += f"{replication[metric]:>12.5f}" if metric in replication else f"{'n/a':>12}"
        lines.append(line)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The sweep's flags; a function so a test can parse them without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default=None, help="Default: a local smoke model")
    parser.add_argument("--dataset", type=str, default=None, help="Default: a local smoke corpus")
    parser.add_argument("--router", type=str, default="mob", choices=["mob", "softmax", "dense"])
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seed list")
    parser.add_argument("--steps", type=int, default=60)
    # #25's differentiation checkpoint reads the *step* at which the experts leave
    # the upcycling floor, which a two-point curve can only bracket. Default keeps
    # the two evaluations every run has always had.
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluate the held-out split every this many steps (default: steps // 2)",
    )
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
    # #31: strict reproduces bitwise; warn lets the attention backward through
    # non-deterministic, and the replicate below measures what that costs.
    parser.add_argument(
        "--deterministic",
        type=str,
        choices=DETERMINISM_MODES,
        default=DETERMINISM_DEFAULT,
        help=(
            "strict: every kernel deterministic or the run refuses, bitwise reproducible at "
            "about five percent per step; warn: the mode every arm before #31 ran under, the "
            "attention backward left non-deterministic and logged; off: torch's defaults "
            f"(default: {DETERMINISM_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--exploration_draw",
        type=str,
        choices=sorted(SUPPORTED_EXPLORATION_DRAWS),
        default=EXPLORATION_DRAW_STALENESS,
        help=(
            "Which loser the explored slot goes to: weighted by steps since it last held a "
            "token (#38, the default) or uniform, which every recorded arm ran under and "
            "which a new arm compared against one must pass, the draw being in the fingerprint"
        ),
    )
    # #39's stakes dial: the one field the three stakes arms differ in.
    parser.add_argument(
        "--persistence_coupling",
        type=str,
        choices=sorted(SUPPORTED_PERSISTENCE_COUPLINGS),
        default=PERSISTENCE_VALUE,
        help=(
            "Whether a cell's continuation depends on its realised value: value (the "
            "economy as recorded), decoupled (pinned wealth, uniform re-entry, a shadow "
            "ledger) or shuffled (regression targets permuted across experts each step)"
        ),
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
    # #24's routing contrast needs both arms measured against one direction, so
    # the uncoupled arm names the goal it is *measured* against without coupling.
    parser.add_argument(
        "--trace_goal",
        type=str,
        default=None,
        help=(
            "Measure held-out routing against this goal's certified direction without "
            "coupling to it (default: the coupling goal, or nothing)"
        ),
    )
    # #28's field-present arms: the goal injected during training, as served.
    # Field presence changes what is trained, so it is fingerprinted; the two
    # field-on arms (with and without --coupling_goal) inject identically.
    parser.add_argument(
        "--steer_goal",
        type=str,
        default=None,
        help=(
            "Inject this goal's certified direction during training, as served: constant "
            "loop, certified strength and layers (#28; default: the field is absent)"
        ),
    )
    # #31: the run-to-run floor, measured rather than assumed absent.
    parser.add_argument(
        "--replicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the first seed a second time and record the pair's spread as "
            "replication_std, the run-to-run floor compare_runs.py quotes (default: on)"
        ),
    )
    # #35: the metric this sweep is to be read on, declared before it is read.
    # It travels in the summary so compare_runs.py puts the interval on the
    # contrast that was chosen in advance rather than the largest row found.
    parser.add_argument(
        "--primary",
        type=str,
        default=None,
        help=(
            "Declare the one metric a comparison against this group is read on "
            "(default: none declared, and compare_runs.py prints no primary block)"
        ),
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    args = build_parser().parse_args()

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
        eval_steps=args.eval_steps or max(1, args.steps // 2),
        save_steps=args.steps,
        log_frequency=max(1, args.steps // 4),
        held_out_sequences=args.held_out_sequences,
        probe_tokens=args.probe_tokens,
        device=args.device,
        dtype="float32" if args.device == "cpu" else "bfloat16",
        gradient_checkpointing=False,
        use_lora=args.use_lora,
        deterministic=args.deterministic,
        exploration_draw=args.exploration_draw,
        persistence_coupling=args.persistence_coupling,
        coupling_goal=args.coupling_goal,
        coupling_beta=args.coupling_beta,
        coupling_warmup_steps=args.coupling_warmup_steps,
        trace_goal=args.trace_goal,
        steer_goal=args.steer_goal,
    )

    # One shared MLflow store across seeds, same reasoning as compare_routers.py:
    # otherwise each seed's run lands in its own file store and `mlflow ui` can
    # never show them side by side.
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{workspace / 'mlruns'}")

    runs = {seed: run_seed(seed, config) for seed in seeds}
    per_seed = {seed: metrics for seed, (metrics, _) in runs.items()}
    fingerprints = {seed: fingerprint for seed, (_, fingerprint) in runs.items()}
    stats = aggregate(per_seed)

    replicate_seed = seeds[0] if args.replicate else None
    replicate_metrics: dict[str, float] | None = None
    replication: dict[str, float] | None = None
    replication_error: str | None = None
    if replicate_seed is not None:
        replicate_metrics, replication, replication_error = measure_replication(
            replicate_seed, config, per_seed[replicate_seed], fingerprints[replicate_seed]
        )

    arm = arm_label(args.router, args.coupling_goal, args.steer_goal, args.persistence_coupling)
    if args.primary is not None and args.primary not in stats:
        logger.warning(
            f"the declared primary metric {args.primary!r} is not among the metrics this "
            f"sweep measured ({sorted(stats)}); it is recorded as declared, but no "
            "comparison will find it"
        )
    print("\n" + format_table(stats, replication))
    print(f"\narm: {arm} | seeds: {seeds} | steps: {args.steps} | primary: {args.primary}")
    if replicate_seed is None:
        print("replicate: none (--no-replicate); the run-to-run floor is not measured")
    elif replication_error is not None:
        print(f"replicate: seed {replicate_seed} unmeasured -- {replication_error}")
    else:
        print(
            f"replicate: seed {replicate_seed} run twice under --deterministic "
            f"{args.deterministic}; repl_std is the pair's sample std, the run-to-run floor"
        )
    print(f"artefacts: {workspace}")

    summary_path = workspace / "seed_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "router": args.router,
                "coupling_goal": args.coupling_goal,
                "steer_goal": args.steer_goal,
                "persistence_coupling": args.persistence_coupling,
                "trace_goal": config.trace_goal or config.coupling_goal or config.steer_goal,
                "seeds": seeds,
                "steps": args.steps,
                "primary": args.primary,
                "per_seed": per_seed,
                "fingerprints": fingerprints,
                "stats": stats,
                "replicate_seed": replicate_seed,
                # Raw provenance, read by nothing: it lets the floor be recomputed.
                "replicate": replicate_metrics,
                "replication_std": replication,
                "replication_error": replication_error,
            },
            indent=2,
        )
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
