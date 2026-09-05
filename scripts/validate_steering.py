"""Certify the behavioural steering vectors once, and record the numbers.

The #3 quality gate is offline by design: a steering vector's validity is a
property of the (model, goal, pairs, format) tuple, so it is measured once like
calibrating an instrument, not re-derived on every server start. This script is
that measurement, extended by #17 to the multiple-choice letter format, the
certified per-goal sources, and the second truthfulness dataset. For each goal it

1. loads the pairs from the goal's source and splits them into an extraction set
   and a disjoint held-out set -- per tier for the built-in templates, every k-th
   row for a HuggingFace source -- *then* converts each split to the goal's
   format, so a multiple-choice set is letter-balanced on both sides,
2. extracts the behavioural vector and the instruction-prefix control vector,
3. measures, on the held-out set, the log-odds shift each produces against a
   distribution of matched random directions (**the gate**),
4. measures the same vector on the *other* format and on the goal's transfer
   sources, and a vector extracted from each transfer source on the certified
   held-out set (**transfer diagnostics**, reported, not gated),
5. for the reasoning goals, greedy-decodes held-out questions with and without
   the vector and reports the length and accuracy deltas (**outcome check**),
6. prints the goal-similarity matrix for the orthogonalisation decision.

The numbers it prints are what belong in the issue and the README. It needs a
GPU-resident model; the default is the ungated Qwen3-1.7B the rest of the repo
measures on, and the HuggingFace sources need the ``train`` extra and a warm cache
(``scripts/warm_hf_datasets.py``).

Run:  uv run python scripts/validate_steering.py [--model Qwen/Qwen3-1.7B]
                                                 [--layers 14,18,22] [--strength 4.0]
                                                 [--goals truthful,reasoning,deliberation,safe]
                                                 [--held-out-per-tier 5] [--held-out 200]
                                                 [--no-transfer] [--no-outcome]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from behavioural_validation import ValidationResult, validate_steering_vector  # noqa: E402
from contrastive_data import (  # noqa: E402
    BUILTIN_SOURCE,
    COMPLETION_FORMAT,
    MAX_LETTER_IMBALANCE,
    MULTIPLE_CHOICE_FORMAT,
    Certification,
    ContrastivePair,
    certification_for,
    letter_counts,
    letter_imbalance,
    load_contrastive_dataset,
    load_instruction_prefix_control,
    to_multiple_choice,
)
from contrastive_templates import TIERS  # noqa: E402
from outcome_check import measure_outcome  # noqa: E402
from steering import SteeringConfig, SteeringVector, SteeringVectorExtractor  # noqa: E402
from steering_pipeline import log_goal_similarity  # noqa: E402

DEFAULT_GOALS = ("truthful", "reasoning", "deliberation", "safe")
# Goals under evaluation that are not (yet) in CERTIFIED still need a format to be
# measured in; the deliberation proxy is designed for the letter format.
CANDIDATE_FORMATS = {"deliberation": MULTIPLE_CHOICE_FORMAT}
# Sources a certified vector is additionally measured against, and from which a
# second vector is extracted and measured on the certified held-out set. Answers
# "is this the truth direction, or a TruthfulQA-shaped one?" -- Geometry of Truth
# is the dataset family on which truth is shown to be linear and to transfer.
TRANSFER_SOURCES = {
    "truthful": ("geometry_of_truth/cities", "geometry_of_truth/sp_en_trans", BUILTIN_SOURCE),
}
# The outcome check needs questions with reference answers; the reasoning
# content pairs carry both, and both reasoning goals are scored on the same ones.
OUTCOME_QUESTION_GOAL = "reasoning"
OUTCOME_GOALS = ("reasoning", "deliberation")
SEED_EXTRACT, SEED_HELD_OUT = 1, 2


def split_by_tier(pairs, held_out_per_tier):
    """Disjoint extraction/held-out split of a built-in set, stratified by tier."""
    held_out, extract = [], []
    seen = dict.fromkeys(TIERS, 0)
    for pair in reversed(pairs):
        if seen[pair.tier] < held_out_per_tier:
            held_out.append(pair)
            seen[pair.tier] += 1
        else:
            extract.append(pair)
    return list(reversed(extract)), list(reversed(held_out))


def split_interleaved(pairs, held_out_n):
    """Every k-th pair held out so topics interleave; the rest extract."""
    k = max(2, len(pairs) // max(1, held_out_n))
    held_out = [pair for index, pair in enumerate(pairs) if index % k == 0][:held_out_n]
    extract = [pair for index, pair in enumerate(pairs) if index % k != 0]
    return extract, held_out


def load_split(goal, source, args):
    """Content-format extraction and held-out sets for ``goal`` from ``source``."""
    pairs = list(load_contrastive_dataset(goal, source=source, pair_format=COMPLETION_FORMAT))
    if source == BUILTIN_SOURCE:
        return split_by_tier(pairs, args.held_out_per_tier)
    return split_interleaved(pairs, args.held_out)


def in_format(pairs, pair_format, seed):
    """Convert one split to ``pair_format`` and refuse a letter set the module would flag.

    The gate must not run on a set whose own quality report calls it unbalanced: a
    skewed letter assignment leaks the bare A-minus-B direction into the vector
    and, if the held-out set skews the same way, into the effect as well.
    """
    if pair_format != MULTIPLE_CHOICE_FORMAT:
        return pairs
    converted = to_multiple_choice(pairs, seed=seed)
    if letter_imbalance(converted) > MAX_LETTER_IMBALANCE:
        raise ValueError(f"letter assignment unbalanced: {letter_counts(converted)}")
    return converted


def other_format(pair_format):
    return COMPLETION_FORMAT if pair_format == MULTIPLE_CHOICE_FORMAT else MULTIPLE_CHOICE_FORMAT


def certification_under_test(goal) -> Certification:
    if goal in CANDIDATE_FORMATS:
        return Certification(BUILTIN_SOURCE, CANDIDATE_FORMATS[goal])
    return certification_for(goal) or Certification(BUILTIN_SOURCE, COMPLETION_FORMAT)


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # pyright: ignore[reportArgumentType] # HF stubs
    return model, tokenizer, device


class Bench:
    """One model, one config; measures vectors against held-out sets."""

    def __init__(self, model, tokenizer, device, config, layers, num_random):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.config, self.num_random = config, num_random
        self.extractor = SteeringVectorExtractor(model, tokenizer, layers)

    def extract(self, pairs) -> dict[int, SteeringVector]:
        return self.extractor.extract_from_pairs(pairs)

    def validate(self, goal, vectors, held_out, control) -> ValidationResult:
        return validate_steering_vector(
            self.model,
            self.tokenizer,
            goal=goal,
            vectors=vectors,
            held_out=held_out,
            config=self.config,
            device=self.device,
            control_vectors=control,
            num_random=self.num_random,
        )


ROW = "{label:<44}{n:>5}{effect:>9}{rand_max:>10}{control:>10}{verdict:>9}"


def print_header(title):
    print(f"\n{title}")
    header = ROW.format(
        label="vector -> held-out",
        n="n",
        effect="effect",
        rand_max="rand_max",
        control="control",
        verdict="verdict",
    )
    print(header)
    print("-" * len(header))


def print_row(label, result: ValidationResult, gate=True):
    control = result.control_effect.effect if result.control_effect else float("nan")
    verdict = ("PASS" if result.passed else "FAIL") if gate else ("+" if result.passed else "-")
    print(
        ROW.format(
            label=label,
            n=result.vector_effect.num_pairs,
            effect=f"{result.vector_effect.effect:+.4f}",
            rand_max=f"{result.random_max:+.4f}",
            control=f"{control:+.4f}",
            verdict=verdict,
        )
    )


def measure_goal(goal, bench, args, gate_rows, vectors_by_goal):
    cert = certification_under_test(goal)
    extract_c, held_out_c = load_split(goal, cert.source, args)
    extract = in_format(extract_c, cert.pair_format, SEED_EXTRACT)
    held_out = in_format(held_out_c, cert.pair_format, SEED_HELD_OUT)

    vectors = bench.extract(extract)
    vectors_by_goal[goal] = vectors
    control = bench.extract(list(load_instruction_prefix_control(goal)))

    result = bench.validate(goal, vectors, held_out, control)
    gate_rows.append((goal, cert, len(extract), result))
    print_header(f"== {goal}: {cert.source}, {cert.pair_format} (gate)")
    print_row(f"{goal}[{cert.pair_format}] -> {cert.source}[{cert.pair_format}]", result)

    if args.no_transfer:
        return held_out_c, vectors

    alt = other_format(cert.pair_format)
    print_row(
        f"{goal}[{cert.pair_format}] -> {cert.source}[{alt}]",
        bench.validate(goal, vectors, in_format(held_out_c, alt, SEED_HELD_OUT), control),
        gate=False,
    )
    for source in TRANSFER_SOURCES.get(goal, ()):
        transfer_source(goal, source, cert, bench, args, vectors, held_out, control)
    return held_out_c, vectors


def transfer_source(goal, source, cert, bench, args, vectors, certified_held_out, control):
    t_extract_c, t_held_out_c = load_split(goal, source, args)
    for pair_format in (cert.pair_format, other_format(cert.pair_format)):
        held = in_format(t_held_out_c, pair_format, SEED_HELD_OUT)
        print_row(
            f"{goal}[{cert.pair_format}] -> {source}[{pair_format}]",
            bench.validate(goal, vectors, held, control),
            gate=False,
        )
    t_vectors = bench.extract(in_format(t_extract_c, cert.pair_format, SEED_EXTRACT))
    print_row(
        f"{source}[{cert.pair_format}] -> {cert.source}[{cert.pair_format}]",
        bench.validate(goal, t_vectors, certified_held_out, control),
        gate=False,
    )


def outcome_checks(bench, vectors_by_goal, questions, args):
    print("\nOutcome check: greedy generation on held-out reasoning questions")
    print(f"{'goal':<14}{'n':>4}{'len base':>10}{'len steer':>11}{'acc base':>10}{'acc steer':>11}")
    for goal in OUTCOME_GOALS:
        if goal not in vectors_by_goal:
            continue
        directions = {layer: sv.vector for layer, sv in vectors_by_goal[goal].items()}
        outcome = measure_outcome(
            bench.model,
            bench.tokenizer,
            directions,
            questions,
            bench.config,
            bench.device,
            goal,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"{goal:<14}{outcome.num_questions:>4}{outcome.baseline_length:>10.1f}"
            f"{outcome.steered_length:>11.1f}{outcome.baseline_accuracy:>10.2f}"
            f"{outcome.steered_accuracy:>11.2f}"
            f"   (len {outcome.length_delta:+.1f}, acc {outcome.accuracy_delta:+.2f})"
        )


def print_summary(gate_rows):
    print("\nGate summary")
    print(
        f"{'goal':<14}{'source':<28}{'format':<17}{'extract':>8}{'held':>6}"
        f"{'effect':>9}{'rand_max':>10}{'control':>10}{'verdict':>9}"
    )
    for goal, cert, n_extract, result in gate_rows:
        control = result.control_effect.effect if result.control_effect else float("nan")
        print(
            f"{goal:<14}{cert.source:<28}{cert.pair_format:<17}{n_extract:>8}"
            f"{result.vector_effect.num_pairs:>6}{result.vector_effect.effect:>+9.4f}"
            f"{result.random_max:>+10.4f}{control:>+10.4f}"
            f"{'PASS' if result.passed else 'FAIL':>9}"
        )
    print(
        "\neffect  = mean held-out log-odds shift the behavioural vector produced\n"
        "control = the same for a vector extracted from the retained instruction prefixes;\n"
        "          it must score below the behavioural vector, or the metric is measuring\n"
        "          prompt wording rather than behaviour.\n"
        "Transfer rows (+/-) are diagnostics, not gates."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layers", default="14,18,22")
    parser.add_argument("--strength", type=float, default=4.0)
    parser.add_argument("--goals", default=",".join(DEFAULT_GOALS))
    parser.add_argument("--held-out-per-tier", type=int, default=5)
    parser.add_argument("--held-out", type=int, default=200, help="held-out cap for HF sources")
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-transfer", action="store_true")
    parser.add_argument("--no-outcome", action="store_true")
    args = parser.parse_args()

    layers = [int(layer) for layer in args.layers.split(",")]
    goals = [goal for goal in args.goals.split(",") if goal]
    config = SteeringConfig(
        steering_layers=layers,
        base_strength=args.strength,
        adaptive=False,
        orthogonal_projection=False,
    )

    model, tokenizer, device = load_model(args.model)
    print(f"model={args.model} device={device} layers={layers} strength={args.strength}")
    bench = Bench(model, tokenizer, device, config, layers, args.num_random)

    gate_rows: list = []
    vectors_by_goal: dict[str, dict[int, SteeringVector]] = {}
    held_out_content: dict[str, list[ContrastivePair]] = {}
    for goal in goals:
        held_out_content[goal], _ = measure_goal(goal, bench, args, gate_rows, vectors_by_goal)

    if not args.no_outcome and OUTCOME_QUESTION_GOAL in held_out_content:
        outcome_checks(bench, vectors_by_goal, held_out_content[OUTCOME_QUESTION_GOAL], args)

    print_summary(gate_rows)

    if len(vectors_by_goal) < 2:
        return
    mid_layer = layers[len(layers) // 2]
    print(f"\nGoal cosine similarity at layer {mid_layer} (for the #4 orthogonalisation decision):")
    pairwise = log_goal_similarity(vectors_by_goal, mid_layer)
    for (a, b), value in sorted(pairwise.items()):
        print(f"  cos({a}, {b}) = {value:+.3f}")
    worst = max(abs(value) for value in pairwise.values())
    print(
        f"\nLargest off-diagonal |cos| = {worst:.3f}. Orthogonalise the measurement basis only if\n"
        "this is high enough that independent per-goal PID loops would regulate one shared\n"
        "direction; goal interaction itself stays dynamic through the economy either way."
    )


if __name__ == "__main__":
    main()
