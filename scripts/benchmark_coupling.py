"""What the serve-time machinery costs, on the real substrate (#5).

#5 budgets the coupling at under 5% of the forward pass and asks for end-to-end
throughput beside it: the percentage is the number that catches a regression in
the hook, throughput is the one a reader notices. Both are measured here, and so
are the two other things a served process carries that an uncoupled base model
does not -- the tissue's residual-stream hooks, and the per-token routing trace
(#5's own telemetry, whose cost has to be paid for by its usefulness).

Not a test. A CPU fake would say nothing about the real overhead -- the coupling
is one dot product and one axpy per token against an FFN, and on a CPU the
arithmetic intensity that makes it negligible is absent -- and the GPU job has no
room for a benchmark. This runs on the workstation, on demand, and its numbers go
into the README.

Method: teacher-forced forwards at a fixed shape, CUDA-synchronised either side,
warmup discarded, the **median** of the remaining repetitions (a mean over a run
that shares a GPU with a desktop compositor measures the compositor). Each
configuration is measured in the same process against the same weights, so the
comparison is of the hooks and not of two loads of the model.

Run:  HF_HUB_OFFLINE=1 uv run python scripts/benchmark_coupling.py [--out bench.json]
"""

import argparse
import json
import logging
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tame"))
sys.path.insert(0, str(ROOT))

from config import MODEL_PROFILES  # noqa: E402
from contrastive_data import CERTIFIED_MODEL  # noqa: E402
from homeostat import CognitiveHomeostat  # noqa: E402
from mob import (  # noqa: E402
    MoBConfig,
    SteeringCouplingConfig,
    apply_mob_to_model,
    mob_layers_by_index,
)
from mob.utils import frozen_economy  # noqa: E402
from steering import SteeringConfig  # noqa: E402
from steering_pipeline import (  # noqa: E402
    certified_coupling_layers,
    extract_steering_vectors,
    seed_coupling,
    serving_config,
)
from tracking import log_step  # noqa: E402

logger = logging.getLogger(__name__)

GOAL = "truthful"
EXTRACTION_PAIRS = 96
BATCH = 1
SEQUENCE = 256
WARMUP = 5
REPETITIONS = 20
GENERATE_TOKENS = 128
# #5's acceptance criterion for the coupling's share of the forward pass.
OVERHEAD_BUDGET_PCT = 5.0
PROMPTS = [
    "Explain, in a few sentences, why the sky is blue and what colour it turns at sunset.",
    "Summarise the difference between a virus and a bacterium for a curious ten-year-old.",
    "What are the main trade-offs between renting and buying a home?",
]


def _synchronise() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median_forward_seconds(model, input_ids: torch.Tensor) -> float:
    """Median wall time of one teacher-forced forward, warmup discarded."""
    with torch.no_grad(), frozen_economy(model):
        for _ in range(WARMUP):
            model(input_ids=input_ids, use_cache=False)
        _synchronise()

        samples = []
        for _ in range(REPETITIONS):
            started = time.perf_counter()
            model(input_ids=input_ids, use_cache=False)
            _synchronise()
            samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def tokens_per_second(model, tokenizer, device) -> float:
    """End-to-end greedy decode over the prompts: the number a user notices."""
    total_tokens, total_seconds = 0, 0.0
    with torch.no_grad(), frozen_economy(model):
        for prompt in PROMPTS:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            _synchronise()
            started = time.perf_counter()
            generated = model.generate(
                **inputs,
                max_new_tokens=GENERATE_TOKENS,
                min_new_tokens=GENERATE_TOKENS,
                do_sample=False,
            )
            _synchronise()
            total_seconds += time.perf_counter() - started
            total_tokens += int(generated.shape[1] - inputs["input_ids"].shape[1])
    return total_tokens / total_seconds


@contextmanager
def coupling_attached(model, homeostat, extraction, model_id: str, hidden_dim: int):
    """The certified coupling, receptor seeded to the direction so it is not inert."""
    coupled = seed_coupling(
        model,
        homeostat,
        extraction,
        certified_coupling_layers(GOAL, model_id),
        SteeringCouplingConfig(hidden_dim=hidden_dim),
    )
    for coupling in coupled.values():
        with torch.no_grad():
            coupling.detector.copy_(coupling.steering_direction)
        coupling.set_coupling_step(coupling.config.warmup_steps)
    try:
        yield
    finally:
        for layer, mob in mob_layers_by_index(model).items():
            if layer in coupled:
                mob.detach_coupling()


@contextmanager
def tissue_attached(model, homeostat):
    homeostat.attach_to_model(model)
    try:
        yield
    finally:
        homeostat.detach_from_model()


@contextmanager
def trace_enabled(model, homeostat):
    layers = mob_layers_by_index(model)
    for layer, mob in layers.items():
        trace = mob.enable_routing_trace()
        if layer in homeostat.steering_vectors:
            trace.set_direction(homeostat.projected_direction(layer)[0])
    try:
        yield
    finally:
        for mob in layers.values():
            mob.disable_routing_trace()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CERTIFIED_MODEL)
    parser.add_argument("--profile", default="qwen3-1.7b")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device: a CPU number says nothing about this overhead")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    profile = MODEL_PROFILES[args.profile]
    device = torch.device("cuda")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    )
    mob_layers = list(range(profile["mob_layers_start"], profile["mob_layers_end"]))
    model = apply_mob_to_model(
        model,
        MoBConfig(
            num_experts=4,
            top_k=2,
            hidden_dim=profile["hidden_dim"],
            intermediate_dim=profile["intermediate_dim"],
            adapter_rank=32,
            adapter_alpha=16.0,
            use_loss_feedback=False,
            use_local_quality=True,
            use_differentiable_routing=False,
        ),
        mob_layers,
    )
    model.to(device)
    model.eval()

    config = serving_config(GOAL, SteeringConfig(steering_layers=mob_layers), model_id=args.model)
    extraction = extract_steering_vectors(
        model, tokenizer, goal=GOAL, config=config, max_pairs=EXTRACTION_PAIRS
    )
    homeostat = CognitiveHomeostat(config)
    homeostat.add_steering_vectors(extraction.vectors)
    homeostat.estimate_capability_subspaces(model, tokenizer)

    input_ids = torch.randint(0, tokenizer.vocab_size, (BATCH, SEQUENCE), device=device)

    def measure(label: str) -> dict[str, float]:
        row = {
            "forward_seconds": median_forward_seconds(model, input_ids),
            "tokens_per_second": tokens_per_second(model, tokenizer, device),
        }
        logger.warning(
            "%-22s forward %.4f s   %.1f tok/s",
            label,
            row["forward_seconds"],
            row["tokens_per_second"],
        )
        return row

    results = {"base": measure("base")}
    with coupling_attached(model, homeostat, extraction, args.model, profile["hidden_dim"]):
        results["coupling"] = measure("+ coupling")
    with trace_enabled(model, homeostat):
        results["routing_trace"] = measure("+ routing trace")
    with tissue_attached(model, homeostat):
        results["tissue"] = measure("+ tissue hooks")
    with (
        coupling_attached(model, homeostat, extraction, args.model, profile["hidden_dim"]),
        trace_enabled(model, homeostat),
        tissue_attached(model, homeostat),
    ):
        results["served"] = measure("+ all three")

    base = results["base"]
    for row in results.values():
        row["forward_overhead_pct"] = 100.0 * (row["forward_seconds"] / base["forward_seconds"] - 1)
        row["throughput_delta_pct"] = 100.0 * (
            row["tokens_per_second"] / base["tokens_per_second"] - 1
        )

    overhead = results["coupling"]["forward_overhead_pct"]
    print(f"\n{'configuration':<16} {'forward (ms)':>13} {'overhead':>9} {'tok/s':>8} {'delta':>8}")
    for label, row in results.items():
        print(
            f"{label:<16} {1000 * row['forward_seconds']:>13.2f} "
            f"{row['forward_overhead_pct']:>+8.1f}% {row['tokens_per_second']:>8.1f} "
            f"{row['throughput_delta_pct']:>+7.1f}%"
        )
    print(
        f"\ncoupling_overhead_pct = {overhead:+.2f}% against a budget of {OVERHEAD_BUDGET_PCT}%: "
        f"{'PASS' if overhead < OVERHEAD_BUDGET_PCT else 'OVER BUDGET'}"
    )

    # A no-op with no active MLflow run, which is every invocation that has not
    # opened one: the number is what has value, the backend is where it is filed.
    log_step(
        0,
        {
            "bench/coupling_overhead_pct": overhead,
            "bench/routing_trace_overhead_pct": results["routing_trace"]["forward_overhead_pct"],
            "bench/tissue_overhead_pct": results["tissue"]["forward_overhead_pct"],
            "bench/tokens_per_second_uncoupled": base["tokens_per_second"],
            "bench/tokens_per_second_coupled": results["coupling"]["tokens_per_second"],
            "bench/tokens_per_second_served": results["served"]["tokens_per_second"],
        },
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "batch": BATCH,
                    "sequence": SEQUENCE,
                    "repetitions": REPETITIONS,
                    "generate_tokens": GENERATE_TOKENS,
                    "results": results,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
