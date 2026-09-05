"""Sweep the layers a goal direction is injected at: efficacy, disruption, observability.

#4 found the server steering at layers it had never gated (the MoB range), where
the ``truthful`` direction fails the behavioural gate and flips answers toward
falsehood. Which layers to serve is an empirical question with three parts, and
this script measures all three so the choice is reproducible:

- **Efficacy** -- the #3/#17 behavioural gate (held-out log-odds shift against
  matched random directions and the instruction-prefix control), with the
  held-out letter accuracy beside it so a flip toward wrong answers is visible
  as a discrete outcome.
- **Disruption** -- how much the steered model's log-probability of its *own*
  unsteered continuations drops, next to the drop a random direction of equal
  norm causes at the same layers. Steering that disrupts far more than a random
  direction is damage, not steering.
- **Observability** -- at readouts above the injection, the lift minus the known
  additive passthrough of the injection itself, in units of the slow resting
  variability: the network's own response, which is all a homeostat can regulate.

Phase A measures every layer alone at one strength. Phase B builds candidate sets
from that profile (the certified set, the top layers by margin, contiguous windows
around the peak, every passing layer, and any explicit sets) and gates each at
several strengths to find its band. The numbers belong in ``contrastive_data.CERTIFIED``.

Run:  uv run python scripts/sweep_steering_layers.py [--model Qwen/Qwen3-1.7B] [--goal truthful]
          [--layers 4-27] [--strength 4] [--set-strengths 2,4,6,8]
          [--sets certified,top3,top5,window3,window5,passing,13+16-21]
          [--pairs 100] [--num-random 4] [--prompts 24] [--tokens 48] [--out sweep.json]
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from behavioural_validation import held_out_accuracy, validate_steering_vector  # noqa: E402
from contrastive_data import (  # noqa: E402
    COMPLETION_FORMAT,
    certification_for,
    certified_source,
    interleaved_split,
    load_contrastive_dataset,
    load_instruction_prefix_control,
    to_multiple_choice,
)
from steering import SteeringConfig, SteeringVectorExtractor  # noqa: E402
from steering_probe import ProjectionProbe, greedy_forced_sequences, step_schedule  # noqa: E402

SEED_EXTRACT, SEED_HELD_OUT = 1, 2
DEFAULT_SETS = "certified,top3,top5,window3,window5,passing"
RANDOM_DRIFT_SEEDS = (11, 12)
READOUT_OFFSETS = (1, 2, 4)


def parse_layers(spec: str) -> list[int]:
    """``"4-27"``, ``"13+16-21"`` or ``"14,18,22"`` to a sorted list of layers."""
    layers: set[int] = set()
    for part in spec.replace("+", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            layers.update(range(int(low), int(high) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def candidate_sets(
    profile: dict[int, dict], requested: Sequence[str], certified: Sequence[int] | None
) -> dict[str, list[int]]:
    """Named layer sets built from the single-layer profile, plus any explicit ones."""
    layers = sorted(profile)
    ranked = sorted(layers, key=lambda layer: profile[layer]["margin"], reverse=True)
    passing = [layer for layer in layers if profile[layer]["passed"]]
    peak = ranked[0]
    sets: dict[str, list[int]] = {}
    for name in requested:
        if name == "certified":
            if certified:
                sets[name] = sorted(certified)
        elif name.startswith("top"):
            sets[name] = sorted(ranked[: int(name[3:])])
        elif name.startswith("window"):
            half = int(name[6:]) // 2
            sets[name] = [
                layer for layer in range(peak - half, peak + half + 1) if layer in profile
            ]
        elif name == "passing":
            sets[name] = passing or [peak]
        else:
            sets[name] = parse_layers(name)
    return sets


def split_pairs(goal: str, held_out: int):
    """The certified source split as the gate splits it, both sides in letter format."""
    content = list(
        load_contrastive_dataset(goal, source=certified_source(goal), pair_format=COMPLETION_FORMAT)
    )
    extract_content, held_content = interleaved_split(content, 200)
    extract = to_multiple_choice(extract_content, seed=SEED_EXTRACT)
    held = to_multiple_choice(held_content, seed=SEED_HELD_OUT)[:held_out]
    return extract, held, held_content


class Sweep:
    """One model, one goal; measures single layers and layer sets."""

    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
            .eval()
            .to(self.device)  # pyright: ignore[reportArgumentType] # HF stubs
        )
        self.layers = parse_layers(args.layers)
        extract, self.held, held_content = split_pairs(args.goal, args.pairs)
        extractor = SteeringVectorExtractor(self.model, self.tokenizer, self.layers)
        self.vectors = extractor.extract_from_pairs(extract)
        self.control = extractor.extract_from_pairs(
            list(load_instruction_prefix_control(args.goal))
        )
        self.directions = {layer: vector.vector for layer, vector in self.vectors.items()}
        stride = max(1, len(held_content) // args.prompts)
        questions = [pair.prompt for pair in held_content[::stride]][: args.prompts]
        self.forced = greedy_forced_sequences(
            self.model, self.tokenizer, self.device, questions, args.tokens, enable_thinking=False
        )
        self.probe = ProjectionProbe(self.model, self.directions)
        self.resting, self.sigma_slow, self.baseline_log_prob = self._resting_state()
        self.accuracy_base = held_out_accuracy(self.model, self.tokenizer, self.held, self.device)

    def _resting_state(self):
        projections: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
        log_probs = []
        for ids, prompt_length in self.forced:
            records, token_log_probs = self.probe.forward(ids)
            log_probs.append(token_log_probs[prompt_length - 1 :].mean())
            for layer in self.layers:
                projections[layer].append(records[layer][prompt_length:])
        resting = {layer: torch.cat(values).mean().item() for layer, values in projections.items()}
        sigma = {
            layer: torch.stack([value.mean() for value in values]).std().item()
            for layer, values in projections.items()
        }
        return resting, sigma, torch.stack(log_probs).mean().item()

    def _config(self, layers: Sequence[int], strength: float) -> SteeringConfig:
        return SteeringConfig(
            steering_layers=list(layers),
            base_strength=strength,
            adaptive=False,
            orthogonal_projection=False,
        )

    def gate(self, layers: Sequence[int], strength: float) -> dict:
        result = validate_steering_vector(
            self.model,
            self.tokenizer,
            self.args.goal,
            {layer: self.vectors[layer] for layer in layers},
            self.held,
            self._config(layers, strength),
            self.device,
            control_vectors={layer: self.control[layer] for layer in layers},
            num_random=self.args.num_random,
        )
        control = result.control_effect.effect if result.control_effect else float("nan")
        return dict(
            effect=result.vector_effect.effect,
            rand_max=result.random_max,
            control=control,
            passed=result.passed,
            margin=result.vector_effect.effect - result.random_max,
        )

    def accuracy(self, layers: Sequence[int], strength: float) -> float:
        from behavioural_validation import attach_steering_hooks

        handles = attach_steering_hooks(
            self.model,
            {layer: self.directions[layer] for layer in layers},
            self._config(layers, strength),
        )
        try:
            return held_out_accuracy(self.model, self.tokenizer, self.held, self.device)
        finally:
            for handle in handles:
                handle.remove()

    def drift(self, layers: Sequence[int], strength: float, seed: int | None = None) -> float:
        """Mean log-probability change of the unsteered continuations under steering."""
        directions = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            directions = {
                layer: F.normalize(
                    torch.randn(self.directions[layer].shape, generator=generator), dim=0
                )
                for layer in layers
            }
        values = []
        for ids, prompt_length in self.forced:
            schedule = step_schedule(ids.shape[1], prompt_length, strength)
            _, token_log_probs = self.probe.forward(
                ids, inject=dict.fromkeys(layers, schedule), inject_directions=directions
            )
            values.append(token_log_probs[prompt_length - 1 :].mean())
        return torch.stack(values).mean().item() - self.baseline_log_prob

    def observability(self, layers: Sequence[int], strength: float, readouts: Sequence[int]):
        lifts: dict[int, list[torch.Tensor]] = {readout: [] for readout in readouts}
        for ids, prompt_length in self.forced:
            schedule = step_schedule(ids.shape[1], prompt_length, strength)
            records, _ = self.probe.forward(ids, inject=dict.fromkeys(layers, schedule))
            for readout in readouts:
                lifts[readout].append(records[readout][prompt_length:].mean())
        rows = {}
        for readout in readouts:
            lift = torch.stack(lifts[readout]).mean().item() - self.resting[readout]
            passthrough = strength * sum(
                F.cosine_similarity(self.directions[layer], self.directions[readout], dim=0).item()
                for layer in layers
                if layer < readout
            )
            sigma = max(self.sigma_slow[readout], 1e-6)
            rows[readout] = dict(
                lift=lift,
                passthrough=passthrough,
                response=lift - passthrough,
                lift_sigma=lift / sigma,
                response_sigma=(lift - passthrough) / sigma,
            )
        return rows

    def evaluate(self, name: str, layers: Sequence[int], strength: float) -> dict:
        started = time.time()
        top = max(layers)
        readouts = sorted(
            set(layers)
            | {top + offset for offset in READOUT_OFFSETS if top + offset in self.layers}
        )
        row: dict = dict(layers=list(layers), strength=strength, **self.gate(layers, strength))
        row["accuracy_base"] = self.accuracy_base
        row["accuracy"] = self.accuracy(layers, strength)
        row["drift"] = self.drift(layers, strength)
        row["drift_random"] = sum(
            self.drift(layers, strength, seed) for seed in RANDOM_DRIFT_SEEDS
        ) / len(RANDOM_DRIFT_SEEDS)
        row["observability"] = self.observability(layers, strength, readouts)
        readout_text = " ".join(
            f"r{readout}:{obs['lift_sigma']:+.2f}s/{obs['response_sigma']:+.2f}s"
            for readout, obs in row["observability"].items()
            if readout not in layers
        )
        print(
            f"{name:<16} S={strength:<4} effect {row['effect']:+.4f} "
            f"rand_max {row['rand_max']:+.4f} "
            f"ctrl {row['control']:+.4f} {'PASS' if row['passed'] else 'FAIL'} | "
            f"acc {row['accuracy_base']:.2f}->{row['accuracy']:.2f} | drift {row['drift']:+.4f} "
            f"(random {row['drift_random']:+.4f}) | lift/response above: {readout_text} | "
            f"{time.time() - started:.0f}s",
            flush=True,
        )
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--goal", default="truthful")
    parser.add_argument("--layers", default="4-27")
    parser.add_argument("--strength", type=float, default=4.0)
    parser.add_argument("--set-strengths", default="2,4,6,8")
    parser.add_argument("--sets", default=DEFAULT_SETS)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--num-random", type=int, default=4)
    parser.add_argument("--prompts", type=int, default=24)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--skip-profile", action="store_true", help="only run the explicit --sets")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    sweep = Sweep(args)
    print(
        f"model={args.model} goal={args.goal} layers={sweep.layers} held-out {len(sweep.held)} "
        f"prompts {len(sweep.forced)} base accuracy {sweep.accuracy_base:.2f}",
        flush=True,
    )
    report: dict = dict(args=vars(args), resting=sweep.resting, sigma_slow=sweep.sigma_slow)

    profile: dict[int, dict] = {}
    if not args.skip_profile:
        print(f"\n== Phase A: single layers at strength {args.strength}", flush=True)
        for layer in sweep.layers:
            profile[layer] = sweep.evaluate(f"L{layer}", [layer], args.strength)
        report["profile"] = profile
        ranked = sorted(profile, key=lambda layer: profile[layer]["margin"], reverse=True)
        print(f"\nranked by margin: {ranked}", flush=True)
        print(f"passing alone: {[layer for layer in sweep.layers if profile[layer]['passed']]}")

    certification = certification_for(args.goal)
    requested = [name for name in args.sets.split(",") if name]
    if profile:
        sets = candidate_sets(profile, requested, certification.layers if certification else None)
    else:
        sets = {name: parse_layers(name) for name in requested if name[0].isdigit()}
    strengths = [float(value) for value in args.set_strengths.split(",")]
    print("\n== Phase B: candidate sets by strength", flush=True)
    report["sets"] = {}
    for name, layers in sets.items():
        for strength in strengths:
            report["sets"][f"{name}@{strength}"] = sweep.evaluate(name, layers, strength)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
