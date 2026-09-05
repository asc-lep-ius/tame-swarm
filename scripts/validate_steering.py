"""Certify the behavioural steering vectors once, and record the numbers.

The #3 quality gate is offline by design: a steering vector's validity is a
property of the (model, goal, pairs) triple, so it is measured once like
calibrating an instrument, not re-derived on every server start. This script is
that measurement. For each goal it

1. splits the built-in pairs into an extraction set and a disjoint held-out set,
2. extracts the behavioural vector from the extraction set,
3. extracts the instruction-prefix control vector (the retained old inputs),
4. measures, on the held-out set, the log-odds shift each produces against a
   distribution of matched random directions, and
5. prints the goal-similarity matrix and the orthogonalisation decision.

The numbers it prints are what belong in the issue and the README. It needs a
GPU-resident model; the default is the ungated Qwen3-1.7B the rest of the repo
measures on.

Run:  uv run python scripts/validate_steering.py [--model Qwen/Qwen3-1.7B]
                                                 [--layers 10,13,16]
                                                 [--strength 6.0] [--held-out-per-tier 5]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from behavioural_validation import validate_steering_vector  # noqa: E402
from contrastive_data import load_contrastive_dataset, load_instruction_prefix_control  # noqa: E402
from contrastive_templates import BUILTIN_PAIRS, TIERS  # noqa: E402
from steering import SteeringConfig, SteeringVectorExtractor  # noqa: E402
from steering_pipeline import log_goal_similarity  # noqa: E402


def split_pairs(pairs, held_out_per_tier):
    """Disjoint extraction/held-out split, stratified by tier.

    The last ``held_out_per_tier`` pairs of each tier are held out; validation on
    them is a generalisation test, not a re-fit of the extraction set.
    """
    held_out, extract = [], []
    seen = dict.fromkeys(TIERS, 0)
    for pair in reversed(pairs):
        if seen[pair.tier] < held_out_per_tier:
            held_out.append(pair)
            seen[pair.tier] += 1
        else:
            extract.append(pair)
    return list(reversed(extract)), list(reversed(held_out))


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layers", default="10,13,16")
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument("--held-out-per-tier", type=int, default=5)
    parser.add_argument("--num-random", type=int, default=8)
    args = parser.parse_args()

    layers = [int(layer) for layer in args.layers.split(",")]
    config = SteeringConfig(
        steering_layers=layers,
        base_strength=args.strength,
        adaptive=False,
        orthogonal_projection=False,
    )

    model, tokenizer, device = load_model(args.model)
    print(f"model={args.model} device={device} layers={layers} strength={args.strength}\n")

    extractor = SteeringVectorExtractor(model, tokenizer, layers)
    vectors_by_goal: dict[str, dict[int, object]] = {}

    header = (
        f"{'goal':<10}{'effect':>10}{'rand_max':>10}{'rand_mean':>11}{'control':>10}{'verdict':>9}"
    )
    print(header)
    print("-" * len(header))

    for goal in sorted(BUILTIN_PAIRS):
        pairs = list(load_contrastive_dataset(goal, source="builtin"))
        extract_pairs, held_out = split_pairs(pairs, args.held_out_per_tier)

        vectors = extractor.extract_from_pairs(extract_pairs)
        vectors_by_goal[goal] = vectors
        control_pairs = list(load_instruction_prefix_control(goal))
        control_vectors = extractor.extract_from_pairs(control_pairs)

        result = validate_steering_vector(
            model,
            tokenizer,
            goal=goal,
            vectors=vectors,
            held_out=held_out,
            config=config,
            device=device,
            control_vectors=control_vectors,
            num_random=args.num_random,
        )
        control = result.control_effect.effect if result.control_effect else float("nan")
        print(
            f"{goal:<10}{result.vector_effect.effect:>10.4f}{result.random_max:>10.4f}"
            f"{result.random_mean:>11.4f}{control:>10.4f}"
            f"{'PASS' if result.passed else 'FAIL':>9}"
        )

    print(
        "\neffect  = mean held-out log-odds shift the completion vector produced\n"
        "control = the same for a vector extracted from the retained instruction prefixes;\n"
        "          it must score below the completion vector, or the metric is measuring\n"
        "          prompt wording rather than behaviour.\n"
    )

    mid_layer = layers[len(layers) // 2]
    print(f"Goal cosine similarity at layer {mid_layer} (for the #4 orthogonalisation decision):")
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
