"""Characterise the homeostat's plant, record the before/after, and test the loop's value.

#4 asks for the plant to be measured before any gain is chosen. This script is that
measurement, and every number it prints belongs in the issue and the README:

1. **Calibration** -- the resting projection of the stream onto the goal direction
   at every served layer (mean, per-token and slow sigma) and the actuator's gain
   at the readout; the setpoint and the SIMC-derived gains follow from these.
2. **Open-loop step response** -- teacher-forced, a step in strength at the first
   generated token: process gain, dead time and time-to-63% at the readout, at
   several strengths so linearity is checked rather than assumed.
3. **The P-only loop as served before #4** -- layers 6-21, ``kp=0.5``, setpoint 0.7 on
   the cosine: its steady-state error and strength trace under the sampling the
   server uses. The "before" half of every claim.
4. **The calibrated PI loop** on the same prompts and sampling: error, strength
   trace, saturation, settling. The "after" half.
5. **Cross-goal disturbance** -- steer one goal, read the others at the readout.
6. **Value test** -- adaptive versus constant strength on the held-out letter
   choice after a generated rationale, split by the prompt's resting alignment.
   This is the falsifier that decides ``app.ADAPTIVE_STEERING``.

Run:  uv run python scripts/characterise_plant.py [--model Qwen/Qwen3-1.7B] [--goal truthful]
                                                  [--prompts 24] [--tokens 48] [--seeds 3]
                                                  [--strengths 2,4,8] [--pairs 100]
                                                  [--rationale-tokens 32] [--no-value-test]
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from contrastive_data import (  # noqa: E402
    COMPLETION_FORMAT,
    certification_for,
    load_contrastive_dataset,
    to_multiple_choice,
)
from homeostat import CognitiveHomeostat  # noqa: E402
from steering import SteeringConfig, SteeringVectorExtractor  # noqa: E402
from steering_pipeline import (  # noqa: E402
    calibration_texts,
    extract_steering_vectors,
    serving_config,
)

LEGACY_LAYERS = list(range(6, 22))  # the MoB range the server reused for steering before #4
LEGACY = dict(base_strength=0.3, kp=0.5, target_alignment=0.7, max_strength=1.5, min_strength=0.0)
SAMPLING = dict(do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
OTHER_GOALS = ("reasoning", "safe")
RATIONALE_CUE = "Let's think step by step."


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device  # pyright: ignore[reportArgumentType] # HF stubs


def held_out_pairs(goal: str, count: int):
    """The certified held-out split, content format, interleaved as validate_steering does."""
    certification = certification_for(goal)
    source = certification.source if certification else "builtin"
    pairs = list(load_contrastive_dataset(goal, source=source, pair_format=COMPLETION_FORMAT))
    k = max(2, len(pairs) // 200)
    return [pair for index, pair in enumerate(pairs) if index % k == 0][:count]


def chat_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class Recorder:
    """Per-position projections onto a direction per layer, with a per-position injection."""

    def __init__(self, model, directions: dict[int, torch.Tensor]):
        self.model = model
        self.directions = directions
        self.records: dict[int, torch.Tensor] = {}
        self.inject: dict[int, torch.Tensor] = {}
        self.inject_dirs: dict[int, torch.Tensor] = {}

    def _hook(self, layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer in self.inject:
                schedule = self.inject[layer].to(hidden.device, hidden.dtype)
                hidden = hidden + schedule[None, :, None] * self.inject_dirs[layer].to(
                    hidden.device, hidden.dtype
                )
            if layer in self.directions:
                direction = self.directions[layer].to(hidden.device, hidden.dtype)
                self.records[layer] = (hidden[0] @ direction).detach().float().cpu()
            if layer in self.inject:
                return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
            return output

        return hook

    def run(self, input_ids: torch.Tensor):
        self.records = {}
        layers = self.model.model.layers
        handles = [
            layers[layer].register_forward_hook(self._hook(layer))
            for layer in sorted(set(self.directions) | set(self.inject))
        ]
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits
        finally:
            for handle in handles:
                handle.remove()
        return dict(self.records), logits


def forced_sequences(model, tokenizer, device, questions, tokens):
    out = []
    for question in questions:
        ids = tokenizer(chat_prompt(tokenizer, question), return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**ids, max_new_tokens=tokens, do_sample=False)
        out.append((generated, int(ids.input_ids.shape[1])))
    return out


def first_index(mask: torch.Tensor, default: int) -> int:
    hits = mask.nonzero()
    return int(hits.min().item()) if len(hits) else default


def step_response(model, config, directions, forced, strengths, readout):
    """Teacher-forced step at the first generated token: gain, dead time, tau63 at the readout."""
    recorder = Recorder(model, {readout: directions[readout]})
    rows = []
    for strength in strengths:
        deltas = []
        for ids, plen in forced:
            recorder.inject, recorder.inject_dirs = {}, {}
            base, _ = recorder.run(ids)
            schedule = torch.zeros(ids.shape[1])
            schedule[plen:] = strength
            recorder.inject = dict.fromkeys(config.steering_layers, schedule)
            recorder.inject_dirs = {layer: directions[layer] for layer in config.steering_layers}
            stepped, _ = recorder.run(ids)
            deltas.append(stepped[readout][plen:] - base[readout][plen:])
        # A continuation that hit end-of-text early is shorter; compare on the common span.
        horizon = min(len(delta) for delta in deltas)
        trajectory = torch.stack([delta[:horizon] for delta in deltas]).mean(dim=0)
        plateau = trajectory[len(trajectory) // 2 :].mean().item()
        dead = first_index(trajectory.abs() > 0.1 * abs(plateau), default=horizon)
        tau63 = first_index(trajectory >= 0.63 * plateau, default=-1) if plateau > 0 else -1
        rows.append(
            dict(
                strength=strength,
                gain=plateau / strength,
                dead_time=dead,
                tau63=tau63,
                first=trajectory[0].item(),
                last=trajectory[-1].item(),
            )
        )
        print(
            f"  S={strength:<4} gain {plateau / strength:+.3f}/unit  dead time {dead} tok  "
            f"tau63 {tau63} tok  first {trajectory[0]:+.2f} last {trajectory[-1]:+.2f}"
        )
    return rows


class LegacyPHook:
    """The pre-#4 rule, verbatim: per-layer P on the mean cosine, no memory, no readout."""

    def __init__(self, direction: torch.Tensor):
        self.direction = direction
        self.alignments: list[float] = []
        self.strengths: list[float] = []

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        direction = self.direction.to(hidden.device, hidden.dtype)
        alignment = F.cosine_similarity(hidden.mean(dim=(0, 1)), direction, dim=0).item()
        strength = LEGACY["base_strength"] + LEGACY["kp"] * (LEGACY["target_alignment"] - alignment)
        strength = max(LEGACY["min_strength"], min(LEGACY["max_strength"], strength))
        self.alignments.append(alignment)
        self.strengths.append(strength)
        modified = hidden + strength * direction
        return (modified,) + tuple(output[1:]) if isinstance(output, tuple) else modified


def trace_stats(values: Sequence[float], low: float, high: float) -> dict:
    tensor = torch.tensor(list(values))
    saturated = ((tensor >= high - 1e-6) | (tensor <= low + 1e-6)).float().mean().item()
    return dict(
        mean=tensor.mean().item(),
        sd=tensor.std().item() if len(tensor) > 1 else 0.0,
        min=tensor.min().item(),
        max=tensor.max().item(),
        saturated=saturated,
    )


def sample(model, tokenizer, device, question, tokens, seed):
    torch.manual_seed(seed)
    ids = tokenizer(chat_prompt(tokenizer, question), return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=tokens, **SAMPLING)


def legacy_baseline(model, tokenizer, device, directions, questions, tokens, seeds):
    layers = model.model.layers
    hooks = {layer: LegacyPHook(directions[layer]) for layer in LEGACY_LAYERS}
    handles = [layers[layer].register_forward_hook(hooks[layer]) for layer in LEGACY_LAYERS]
    try:
        for seed in range(seeds):
            for question in questions:
                sample(model, tokenizer, device, question, tokens, seed)
    finally:
        for handle in handles:
            handle.remove()
    rows = {}
    for layer in (LEGACY_LAYERS[0], LEGACY_LAYERS[len(LEGACY_LAYERS) // 2], LEGACY_LAYERS[-1]):
        hook = hooks[layer]
        alignment = torch.tensor(hook.alignments)
        strength = trace_stats(hook.strengths, LEGACY["min_strength"], LEGACY["max_strength"])
        rows[layer] = dict(
            alignment_mean=alignment.mean().item(),
            alignment_sd=alignment.std().item(),
            error_mean=LEGACY["target_alignment"] - alignment.mean().item(),
            strength=strength,
        )
        print(
            f"  L{layer:2d}: alignment {alignment.mean():+.3f} sd {alignment.std():.3f} | error "
            f"{LEGACY['target_alignment'] - alignment.mean():+.3f} | strength mean "
            f"{strength['mean']:.3f} sd {strength['sd']:.3f} [{strength['min']:.2f}, "
            f"{strength['max']:.2f}] saturated {strength['saturated']:.0%}"
        )
    return rows


def closed_loop(model, tokenizer, device, homeostat, config, questions, tokens, seeds):
    """The served loop under the served sampling; per-generation error and strength traces."""
    errors, strengths, settle, saturated = [], [], [], []
    for seed in range(seeds):
        for question in questions:
            homeostat.reset()
            sample(model, tokenizer, device, question, tokens, seed)
            loop = homeostat.homeostat
            pv = torch.tensor(list(loop.alignment_history))
            error = loop.setpoint - pv
            strength = torch.tensor(list(loop.strength_history))
            errors.append(error)
            strengths.append(strength)
            band = 0.25 * abs(error[0].item()) + 1e-6
            settle.append(first_index(error.abs() <= band, default=len(error)))
            saturated.append(
                (
                    (strength >= config.max_strength - 1e-6)
                    | (strength <= config.min_strength + 1e-6)
                )
                .float()
                .mean()
                .item()
            )
    error_all = torch.cat(errors)
    tails = torch.stack([e[-8:].mean() for e in errors])
    strength_all = torch.cat(strengths)
    strength_stats = trace_stats(strength_all.tolist(), config.min_strength, config.max_strength)
    row: dict = dict(
        error_mean=error_all.mean().item(),
        error_tail_mean=tails.mean().item(),
        error_tail_abs=tails.abs().mean().item(),
        error_sd=error_all.std().item(),
        settle_tokens=float(torch.tensor(settle, dtype=torch.float32).mean().item()),
        strength=strength_stats,
        saturated_fraction=float(torch.tensor(saturated).mean().item()),
    )
    print(
        f"  error mean {row['error_mean']:+.3f} sigma (tail {row['error_tail_mean']:+.3f}, |tail| "
        f"{row['error_tail_abs']:.3f}) sd {row['error_sd']:.3f} | settles in "
        f"{row['settle_tokens']:.1f} tok | "
        f"strength mean {strength_stats['mean']:.2f} sd {strength_stats['sd']:.2f} "
        f"[{strength_stats['min']:.2f}, {strength_stats['max']:.2f}] saturated "
        f"{row['saturated_fraction']:.0%} of tokens"
    )
    return row


def cross_goal(model, tokenizer, device, config, forced, goals, strength):
    """Steer goal A at its certified layers, read every goal's direction at the readout."""
    readout = config.readout_layer
    vectors = {}
    for goal in goals:
        goal_config = serving_config(goal, config)
        extraction = extract_steering_vectors(
            model, tokenizer, goal=goal, config=goal_config, layers=None
        )
        vectors[goal] = (goal_config, extraction.vectors)
    readouts = {}
    extractor = SteeringVectorExtractor(model, tokenizer, [readout])
    for goal in goals:
        pairs = list(load_contrastive_dataset(goal, source=serving_source(goal)))
        readouts[goal] = extractor.extract_from_pairs(pairs[:200])[readout].vector
    recorder = Recorder(model, {})
    # resting sigma of each goal at the readout (slow), unsteered
    sigma, resting = {}, {}
    for goal in goals:
        recorder.directions = {readout: readouts[goal]}
        recorder.inject, recorder.inject_dirs = {}, {}
        means = []
        for ids, plen in forced:
            rec, _ = recorder.run(ids)
            means.append(rec[readout][plen:].mean())
        resting[goal] = torch.stack(means)
        sigma[goal] = resting[goal].std().item()
    matrix = {}
    print(
        f"  rows: steered goal at strength {strength}; columns: lift at readout {readout} in sigma"
    )
    print("  " + " " * 12 + "".join(f"{goal:>12}" for goal in goals))
    for source_goal in goals:
        goal_config, goal_vectors = vectors[source_goal]
        matrix[source_goal] = {}
        for target_goal in goals:
            recorder.directions = {readout: readouts[target_goal]}
            lifts = []
            for index, (ids, plen) in enumerate(forced):
                schedule = torch.zeros(ids.shape[1])
                schedule[plen:] = strength
                recorder.inject = dict.fromkeys(goal_config.steering_layers, schedule)
                recorder.inject_dirs = {
                    layer: goal_vectors[layer].vector for layer in goal_config.steering_layers
                }
                rec, _ = recorder.run(ids)
                lifts.append(rec[readout][plen:].mean() - resting[target_goal][index])
            matrix[source_goal][target_goal] = torch.stack(lifts).mean().item() / sigma[target_goal]
        print(
            f"  {source_goal:>12}"
            + "".join(f"{matrix[source_goal][goal]:>+12.2f}" for goal in goals)
        )
    return matrix


def serving_source(goal: str) -> str:
    certification = certification_for(goal)
    return certification.source if certification else "builtin"


def letter_log_odds(model, tokenizer, device, homeostat, pair, rationale_tokens):
    """Generate a rationale with the loop running, then read the letter choice."""
    prompt = pair.prompt.rsplit("Answer:", 1)[0] + RATIONALE_CUE
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    homeostat.reset()
    with torch.no_grad():
        generated = model.generate(**ids, max_new_tokens=rationale_tokens, do_sample=False)
    text = tokenizer.decode(generated[0], skip_special_tokens=True) + "\nAnswer:"
    full = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**full).logits[0, -1].float()
    letters = {
        letter: int(tokenizer(f" {letter}", add_special_tokens=False)["input_ids"][-1])
        for letter in ("A", "B")
    }
    correct = pair.correct_letter
    wrong = "B" if correct == "A" else "A"
    log_probs = F.log_softmax(logits, dim=-1)
    return (log_probs[letters[correct]] - log_probs[letters[wrong]]).item(), homeostat.homeostat


def value_test(model, tokenizer, device, homeostat, config, pairs, rationale_tokens):
    """Adaptive vs constant strength on the letter choice, split by resting alignment."""
    readout = config.readout_layer
    direction = homeostat.steering_vectors[readout].vector
    recorder = Recorder(model, {readout: direction})
    resting = []
    for pair in pairs:
        prompt = pair.prompt.rsplit("Answer:", 1)[0] + RATIONALE_CUE
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        rec, _ = recorder.run(ids.input_ids)
        resting.append(homeostat.calibration.z(readout, rec[readout][-1].item()))
    resting_t = torch.tensor(resting)
    low = resting_t <= resting_t.median()

    results = {}
    for mode in ("constant", "adaptive"):
        config.adaptive = mode == "adaptive"
        odds, strengths = [], []
        for pair in pairs:
            value, loop = letter_log_odds(
                model, tokenizer, device, homeostat, pair, rationale_tokens
            )
            odds.append(value)
            strengths.append(loop.current_strength)
        results[mode] = dict(odds=torch.tensor(odds), strength=torch.tensor(strengths))
    config.adaptive = True

    delta = results["adaptive"]["odds"] - results["constant"]["odds"]
    summary = {}
    for label, mask in (
        ("all", torch.ones_like(low)),
        ("low_resting", low),
        ("high_resting", ~low),
    ):
        summary[label] = dict(
            n=int(mask.sum().item()),
            constant_log_odds=results["constant"]["odds"][mask].mean().item(),
            adaptive_log_odds=results["adaptive"]["odds"][mask].mean().item(),
            delta=delta[mask].mean().item(),
            delta_se=(delta[mask].std() / mask.sum().sqrt()).item() if mask.sum() > 1 else 0.0,
            constant_accuracy=(results["constant"]["odds"][mask] > 0).float().mean().item(),
            adaptive_accuracy=(results["adaptive"]["odds"][mask] > 0).float().mean().item(),
            adaptive_strength=results["adaptive"]["strength"][mask].mean().item(),
        )
        row = summary[label]
        print(
            f"  {label:<13} n={row['n']:>3}  log-odds constant {row['constant_log_odds']:+.3f} "
            f"adaptive {row['adaptive_log_odds']:+.3f}  delta {row['delta']:+.3f} "
            f"+/- {row['delta_se']:.3f} "
            f"| acc {row['constant_accuracy']:.2f} -> {row['adaptive_accuracy']:.2f} "
            f"| adaptive strength {row['adaptive_strength']:.2f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--goal", default="truthful")
    parser.add_argument("--prompts", type=int, default=24)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--strengths", default="2,4,8")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--rationale-tokens", type=int, default=32)
    parser.add_argument("--no-value-test", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="", help="comma-separated section numbers to run")
    args = parser.parse_args()
    sections = {int(number) for number in args.only.split(",") if number}
    wanted = lambda number: not sections or number in sections  # noqa: E731
    logging.basicConfig(level=logging.WARNING)

    model, tokenizer, device = load_model(args.model)
    template = SteeringConfig(adaptive=True, orthogonal_projection=False)
    config = serving_config(args.goal, template, model_id=args.model)
    print(
        f"model={args.model} goal={args.goal} layers={config.steering_layers} "
        f"readout={config.readout_layer} strength={config.base_strength} "
        f"band=[{config.min_strength}, {config.max_strength}]"
    )
    report: dict = dict(config=dict(vars(config)))

    extraction = extract_steering_vectors(model, tokenizer, goal=args.goal, config=config)
    homeostat = CognitiveHomeostat(config)
    homeostat.add_steering_vectors(extraction.vectors)
    # The served regime: the goal's own prompts through the chat template, answered.
    # (Thinking mode off throughout this script, so rationales and readings match.)
    texts = calibration_texts(model, tokenizer, args.goal, chat_kwargs={"enable_thinking": False})
    calibration = homeostat.calibrate(model, tokenizer, texts=texts)
    loop = homeostat.homeostat
    kp, ki = loop.gains()
    print("\n== 1. Calibration (goal prompts + greedy answers, unsteered; position 0 excluded)")
    for layer, stats in sorted(calibration.layers.items()):
        print(
            f"  L{layer:2d}: resting {stats.resting_mean:+8.2f}  "
            f"sigma_tok {stats.token_sigma:6.2f}  sigma_slow {stats.resting_sigma:6.2f}"
        )
    print(
        f"  readout {calibration.readout_layer}: gain {calibration.gain:+.3f} proj/unit = "
        f"{calibration.gain_z:+.3f} sigma/unit; setpoint {calibration.setpoint_z:+.3f} sigma at "
        f"strength {calibration.reference_strength}"
    )
    print(
        f"  filter tau {loop.filter_time_constant:.1f} tok, closed-loop tau "
        f"{config.closed_loop_tau or max(loop.filter_time_constant, 1.0):.1f} tok -> "
        f"SIMC kp {kp:.4f}, ki {ki:.4f}; stability bound ki < {loop.max_stable_ki():.4f}"
    )
    report["calibration"] = dict(
        layers={layer: vars(stats) for layer, stats in calibration.layers.items()},
        readout=calibration.readout_layer,
        gain=calibration.gain,
        gain_z=calibration.gain_z,
        setpoint_z=calibration.setpoint_z,
        kp=kp,
        ki=ki,
        max_stable_ki=loop.max_stable_ki(),
    )

    pairs = held_out_pairs(args.goal, 200)
    questions = [pair.prompt for pair in pairs[:: max(1, 200 // args.prompts)]][: args.prompts]
    forced = forced_sequences(model, tokenizer, device, questions, args.tokens)
    directions = {layer: vector.vector for layer, vector in extraction.vectors.items()}
    strengths = [float(value) for value in args.strengths.split(",")]

    if wanted(2):
        print(
            "\n== 2. Open-loop step response at the readout "
            "(teacher-forced, step at the first generated token)"
        )
        report["step_response"] = step_response(
            model, config, directions, forced, strengths, config.readout_layer
        )

    if wanted(3):
        print(
            "\n== 3. Before: P-only loop as served (layers 6-21, kp 0.5, setpoint 0.7 cosine), "
            f"temp 0.7 x {args.seeds} seeds"
        )
        legacy_extraction = extract_steering_vectors(
            model, tokenizer, goal=args.goal, config=config, layers=LEGACY_LAYERS
        )
        report["legacy"] = legacy_baseline(
            model,
            tokenizer,
            device,
            {layer: vector.vector for layer, vector in legacy_extraction.vectors.items()},
            questions,
            args.tokens,
            args.seeds,
        )

    if wanted(4):
        print(
            f"\n== 4. After: calibrated PI loop as served, temp 0.7 x {args.seeds} seeds "
            "(error in sigma)"
        )
        homeostat.attach_to_model(model)
        report["closed_loop"] = closed_loop(
            model, tokenizer, device, homeostat, config, questions, args.tokens, args.seeds
        )
        homeostat.detach_from_model()

    if wanted(5):
        print(
            "\n== 5. Cross-goal disturbance (each goal steered at its certified layers, strength 4)"
        )
        goals = (args.goal, *[goal for goal in OTHER_GOALS if goal != args.goal])
        report["cross_goal"] = cross_goal(model, tokenizer, device, config, forced, goals, 4.0)

    if wanted(6) and not args.no_value_test:
        print(
            f"\n== 6. Value test: adaptive vs constant on {args.pairs} held-out letter choices "
            f"after a {args.rationale_tokens}-token rationale"
        )
        letter_pairs = to_multiple_choice(pairs, seed=2)[: args.pairs]
        homeostat.attach_to_model(model)
        report["value_test"] = value_test(
            model, tokenizer, device, homeostat, config, letter_pairs, args.rationale_tokens
        )
        homeostat.detach_from_model()

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
