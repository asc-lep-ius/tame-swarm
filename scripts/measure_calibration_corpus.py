"""How much of the cells' disagreement is the calibration corpus, and how much the tissue (#23).

``tests/test_real_model.py`` calibrates the served tissue on 8 prompts answered with
16 tokens; the server uses 24 and 32. On the fixture's calibration the survivors'
gains run from 0.48 to 1.68 sigma per unit and the cells read one replayed
continuation several sigma apart. Either the eight-passage calibration under-samples
each cell's resting distribution -- the slow sigma is the standard deviation of
eight passage means, so its relative standard error is about ``1 / sqrt(2 (N - 1))``,
27% at eight, and every gain and setpoint carries it -- or the disagreement is the
substrate's and no corpus removes it.

This script builds the fixture's system once (same seed, extraction, subspaces and
continuation), re-calibrates the tissue on each corpus in turn, and replays the
same continuation under each regime the fixture measures, recording per cell the
resting sigma, lift, gain, setpoint and tail error. What moves with the corpus is
the fixture's; what does not is the tissue's.

Run:  uv run python scripts/measure_calibration_corpus.py [--corpora 8x16,8x32,24x32,48x32]
                                                          [--build-corpus 24x32] [--out report.json]

The fixture is built on ``--build-corpus`` and then re-calibrated on each corpus in
``--corpora``. Generating a calibration corpus runs the inference economy, so a
corpus measured after another one sits on an economy that has moved; ``--corpora ""``
measures the fixture exactly as built, which is what the fixture's own recorded
numbers have to come from.
"""

import argparse
import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tame"))
sys.path.insert(0, str(ROOT))

from tests.test_real_model import (  # noqa: E402
    CALIBRATION_PROMPTS,
    CALIBRATION_TOKENS,
    CONTENT_PUSH,
    DAMAGE_AT,
    DAMAGED_TAIL,
    GOAL,
    TAIL_TOKENS,
    ServedSystem,
    attached,
    build_served,
    content_push,
)

from steering_pipeline import calibration_texts  # noqa: E402

DEFAULT_CORPORA = "8x16,8x32,24x32,48x32"


def parse_corpus(item: str) -> tuple[int, int]:
    prompts, tokens = item.lower().split("x")
    return int(prompts), int(tokens)


def parse_corpora(spec: str) -> list[tuple[int, int]]:
    return [parse_corpus(item) for item in spec.split(",") if item.strip()]


def cell_tail_errors(system: ServedSystem, tokens: int) -> dict[int, float]:
    """Each cell's setpoint less its mean reading over the last ``tokens`` passes it fired in."""
    tissue = system.tissue
    errors = {}
    for layer, history in tissue.cell_history.items():
        tail = list(history)[-tokens:]
        errors[layer] = tissue.cell_setpoint(layer) - sum(tail) / len(tail)
    return errors


def regime(system: ServedSystem, inert: bool, push: bool) -> dict:
    """One replay: the consensus tail error and strength, and every cell's tail error."""
    pushing = content_push(system, CONTENT_PUSH) if push else nullcontext()
    with attached(system, inert=inert), pushing:
        system.replay()
        return dict(
            error=system.tail_error(),
            strength=system.tail_strength(),
            setpoint=system.tissue.setpoint,
            dispersion=system.tissue.dispersion,
            cells=cell_tail_errors(system, TAIL_TOKENS),
        )


def top_removed(system: ServedSystem) -> dict:
    """The fixture's undesigned damage: the top actuator removed halfway, live loop, pushed."""
    top = max(system.homeostat.actuator_layers)
    before: dict[str, float] = {}

    def remove_top_actuator() -> None:
        before["strength"] = system.tail_strength(DAMAGED_TAIL)
        system.homeostat._registered_hooks[list(system.homeostat.hooks).index(top)].remove()

    with attached(system), content_push(system, CONTENT_PUSH):
        system.replay(damage_at=system.prompt_length + DAMAGE_AT, damage=remove_top_actuator)
        return dict(
            error=system.tail_error(DAMAGED_TAIL),
            strength_before=before["strength"],
            strength_after=system.tail_strength(DAMAGED_TAIL),
            setpoint=system.tissue.setpoint,
            nominal_setpoint=system.tissue.nominal_setpoint,
            cells=cell_tail_errors(system, DAMAGED_TAIL),
        )


def calibrate_on(system: ServedSystem, prompts: int, tokens: int) -> tuple[ServedSystem, dict]:
    """Re-calibrate the fixture's tissue on a corpus; the cost of doing so is part of the answer."""
    started = time.perf_counter()
    texts = calibration_texts(
        system.model, system.tokenizer, GOAL, num_prompts=prompts, new_tokens=tokens
    )
    generated = time.perf_counter() - started
    system.homeostat.calibrate(system.model, system.tokenizer, texts=texts)
    calibrated = time.perf_counter() - started - generated
    # ``attached()`` pins kp/ki on the way out and the config object is shared with
    # the tissue, so without this every corpus after the first would be replayed
    # under the 8 x 16 loop rather than under its own SIMC pair.
    system.tissue.config.kp = None
    system.tissue.config.ki = None
    system = replace(system, derived_gains=system.tissue.gains())
    return system, summarise(system, prompts, tokens, generated, calibrated)


def summarise(
    system: ServedSystem, prompts: int, tokens: int, generated: float, calibrated: float
) -> dict:
    """The tissue's calibration as it stands, per cell and in the aggregate."""
    calibration = system.tissue.calibration
    assert calibration is not None
    kp, ki = system.derived_gains
    cells = {
        layer: dict(
            role="actuator" if layer in calibration.actuators else "readout",
            resting_mean=stats.resting_mean,
            resting_sigma=stats.sigma,
            token_sigma=stats.token_sigma,
            lift=stats.lift,
            gain=stats.gain_z,
            setpoint=calibration.setpoint_z(layer),
            weight=calibration.weight(layer),
        )
        for layer, stats in sorted(calibration.layers.items())
    }
    return dict(
        prompts=prompts,
        tokens=tokens,
        passages=calibration.num_passages,
        sigma_relative_se=1.0 / math.sqrt(2.0 * (calibration.num_passages - 1)),
        seconds_generating=generated,
        seconds_calibrating=calibrated,
        tissue_gain=calibration.gain_z,
        setpoint=system.tissue.setpoint,
        kp=kp,
        ki=ki,
        max_stable_ki=system.tissue.max_stable_ki(),
        cells=cells,
    )


def print_corpus(summary: dict, regimes: dict[str, dict], removal: dict) -> None:
    survivors = [
        cell["gain"] for layer, cell in summary["cells"].items() if cell["role"] == "actuator"
    ][1:]
    print(
        f"\n== {summary['prompts']} prompts x {summary['tokens']} tokens "
        f"({summary['passages']} passages, sigma rel. SE {summary['sigma_relative_se']:.0%}; "
        f"generate {summary['seconds_generating']:.1f} s + calibrate "
        f"{summary['seconds_calibrating']:.1f} s)"
    )
    print(
        f"  tissue gain {summary['tissue_gain']:.3f} sigma/unit, setpoint "
        f"{summary['setpoint']:.2f} sigma, kp {summary['kp']:.3f} ki {summary['ki']:.4f} "
        f"(bound {summary['max_stable_ki']:.3f}); survivors' gains "
        f"{min(survivors):.2f}-{max(survivors):.2f}"
    )
    header = (
        f"  {'cell':>4} {'role':<8} {'sig_slow':>8} {'sig_tok':>8} {'lift':>7} {'gain':>6} "
        f"{'setpt':>6} | " + " ".join(f"{name:>10}" for name in regimes) + f" {'top-rm':>8}"
    )
    print(header)
    for layer, cell in summary["cells"].items():
        tails = " ".join(f"{regimes[name]['cells'][layer]:>+10.2f}" for name in regimes)
        print(
            f"  {layer:>4} {cell['role']:<8} {cell['resting_sigma']:>8.2f} "
            f"{cell['token_sigma']:>8.2f} {cell['lift']:>+7.2f} {cell['gain']:>6.2f} "
            f"{cell['setpoint']:>6.2f} | {tails} {removal['cells'][layer]:>+8.2f}"
        )
    consensus = " ".join(f"{regimes[name]['error']:>+10.2f}" for name in regimes)
    print(f"  {'consensus':<50} | {consensus} {removal['error']:>+8.2f}")
    strengths = " ".join(f"{regimes[name]['strength']:>10.2f}" for name in regimes)
    print(f"  {'strength':<50} | {strengths} {removal['strength_after']:>8.2f}")
    dispersion = " ".join(f"{regimes[name]['dispersion']:>10.2f}" for name in regimes)
    print(f"  {'dispersion':<50} | {dispersion}")
    print(
        f"  top removed: setpoint {removal['setpoint']:.2f} (nominal "
        f"{removal['nominal_setpoint']:.2f}), strength {removal['strength_before']:.2f} -> "
        f"{removal['strength_after']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", default=DEFAULT_CORPORA)
    parser.add_argument("--build-corpus", default=f"{CALIBRATION_PROMPTS}x{CALIBRATION_TOKENS}")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    build_prompts, build_tokens = parse_corpus(args.build_corpus)
    started = time.perf_counter()
    system = build_served(build_prompts, build_tokens)
    print(
        f"fixture built in {time.perf_counter() - started:.1f} s "
        f"({build_prompts} x {build_tokens} calibration included)"
    )

    report = []
    for corpus in [None, *parse_corpora(args.corpora)]:
        if corpus is None:
            summary = summarise(system, build_prompts, build_tokens, 0.0, 0.0)
        else:
            system, summary = calibrate_on(system, *corpus)
        regimes = {
            "inert+push": regime(system, inert=True, push=True),
            "live+push": regime(system, inert=False, push=True),
            "inert": regime(system, inert=True, push=False),
            "live": regime(system, inert=False, push=False),
        }
        removal = top_removed(system)
        print_corpus(summary, regimes, removal)
        report.append(dict(calibration=summary, regimes=regimes, top_removed=removal))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
