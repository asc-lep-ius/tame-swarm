"""The coupled system on the certified substrate: Qwen3-1.7B, the truthful direction (#6).

Everything in ``tests/test_homeostatic_recovery.py`` regulates a random direction
on a random model; it proves the wiring, not the claim. These tests build the
served system for real -- MoB in the blocks, the routing coupling seeded at the
layers #4 certified, the tissue calibrated on the served regime -- and ask the
same questions of the direction the behavioural gate passed:

- a designed perturbation the loop has the authority to correct: content pushed
  against the truthful direction above the bottom actuator, with the loop inert
  (``kp = ki = 0``, still sensing) and with it live;
- the undesigned one: the top actuator removed mid-generation;
- the behavioural gate re-run for ``safe`` on the built-in split, against eight
  matched random directions and the instruction-prefix control -- the
  certification row in the README, as a test.

Resolution is the plant's, not the fixture's. #4 measured per-token content
moving the reading by about a sigma, and the reading carries a 9-token filter, so
the mean over a 60-token tail has roughly seven independent samples and a
standard error near 0.4 sigma. Recovery on the real model is therefore stated
against the inert loop on the same tokens -- the live loop halves the error and
pays for it in strength -- with an absolute band no tighter than the tissue's own
authority, not the 5% band the wired fixture meets.

The tissue error here is the consensus #21 defined: each cell's error weighted by
its calibrated gain, so the blind bottom cell (13, gain exactly zero) is reported
and not regulated. On this fixture's eight-prompt calibration the survivors' gains
spread from 0.48 (cell 16) to 1.68 (cell 19) and the cells disagree by several
sigma on the same tokens, so the weighting moves the regulated number by about a
sigma against the plain mean the tissue used to regulate; cell 13's own reading
(-0.24 sigma over the tail) was never what moved it. The dilution #21 removes is
a seventh of the blind error, below this tail's resolution, so the fixture and
the measured plant carry that claim and this module records the per-cell picture.

The same tokens, literally: the unsteered greedy continuation is generated once
and every regime replays it token by token through the cache, as the plant probe
(``steering_probe``) does. Greedy decoding under bf16 kernels is not bitwise
stable, one flipped token rewrites the tail, and the inference economy drifts
between generations, so a measurement on freshly generated text compares
regimes on different content; a replay compares them on the same. The economy
is frozen for every replay for the same reason. The pairing is also what makes
the removal test's 40-token windows enough: the content variation that sets the
0.4 sigma standard error above is shared between the regimes and cancels in
their difference, so the naive figure overstates the noise on the comparison.

``gpu``-marked. Loads the model from the local HuggingFace cache once per module
and never downloads. Without the cache it skips on a developer's machine and
**fails** under CI: #13's rule is that the job validating the science may not be
advisory, and a skipped real-model suite is a green job that ran nothing.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import cast

import pytest
import torch
import torch.nn as nn

from behavioural_validation import validate_steering_vector
from config import MODEL_PROFILES
from contrastive_data import (
    BUILTIN_SOURCE,
    COMPLETION_FORMAT,
    load_contrastive_dataset,
    load_instruction_prefix_control,
)
from contrastive_templates import TIERS
from homeostat import CognitiveHomeostat
from mob import MoBConfig, SteeringCouplingConfig, apply_mob_to_model
from mob.utils import frozen_economy
from steering import SteeringConfig, SteeringVectorExtractor
from steering_pipeline import (
    calibration_texts,
    certified_coupling_layers,
    extract_steering_vectors,
    seed_coupling,
    serving_config,
)

pytestmark = pytest.mark.gpu

PROFILE = MODEL_PROFILES["qwen3-1.7b"]
MODEL_ID = PROFILE["model_id"]
GOAL = "truthful"
# Enough pairs for a stable direction at a fraction of the certification's 612;
# the extraction is still the certified source and format, so it is certified.
EXTRACTION_PAIRS = 96
CALIBRATION_PROMPTS = 8
CALIBRATION_TOKENS = 16
PROMPT = "Explain, in a few sentences, why the sky is blue and what colour it turns at sunset."
GENERATED_TOKENS = 120
TAIL_TOKENS = 60
# The actuator is removed halfway through the replay; the two halves are compared
# on their last tokens, after each has had time to settle.
DAMAGE_AT = GENERATED_TOKENS // 2
DAMAGED_TAIL = 40
# A push against the direction, applied at every actuator's block the way the
# injection itself is, so each cell reads the pushes below it; -1.2 units against
# a reference of 4 is inside the band's authority. A push at one block alone reads
# as a seventh of this at the readout (measured: 0.15 sigma), below the prompt's
# own content deficit.
CONTENT_PUSH = -1.2
# Measured on the seeded replay (identical across processes on the RTX 5070 Ti),
# as the gain-weighted consensus over the tail: the inert loop +2.16 sigma, the
# live loop +0.61 at a strength raised by 0.82, so the live loop removes 72% of
# the inert error; the same tokens unpushed read +0.70 inert and +0.22 live. (The
# plain mean over all eight cells read +1.10 and +0.15, -0.01 and -0.15 before
# #21, an 86% removal: the middle cells' deficit and the top cells' surplus
# cancelled in it.) The absolute band is a fifth of the tissue setpoint: four
# times the fixture's 5% band, on a substrate whose tail standard error is
# already 8% of that setpoint. Measured 0.61 of 5.05 sigma, 12% (the plain mean:
# 0.15 of 3.88, 4%).
RECOVERY_FRACTION = 0.2
# The push must bite for the recovery to mean anything; measured 2.16 against this
# floor (1.10 as the plain mean), so a failure here says the push weakened on this
# substrate, not that the loop improved.
INERT_ERROR_SIGMA = 0.8
STRENGTH_RISE = 0.3
# After the top actuator is removed the survivors' strength must rise; by how much
# is the tissue's call (measured +1.21, from 4.00 to 5.21, with the error at 1.23
# sigma against 2.16 inert -- measured against the survivors' own setpoint of 5.24
# since #22, 1.03 against the calibrated 5.05), so only the direction is asserted.
SURVIVOR_STRENGTH_RISE = 0.1
SAFE_LAYERS = [14, 18, 22]
SAFE_STRENGTH = 4.0
HELD_OUT_PER_TIER = 5


@dataclass
class ServedSystem:
    model: torch.nn.Module
    tokenizer: object
    homeostat: CognitiveHomeostat
    config: SteeringConfig
    device: torch.device
    derived_gains: tuple[float, float]
    # The unsteered greedy continuation of PROMPT, prompt included, and where it ends.
    tokens: torch.Tensor
    prompt_length: int

    def replay(
        self, damage_at: int | None = None, damage: Callable[[], None] | None = None
    ) -> None:
        """Teacher-force the fixed continuation through the cache, one token per pass.

        ``damage`` runs once, after the pass at position ``damage_at``, so a cell can
        be removed while the tissue is carrying a load.
        """
        with torch.no_grad():
            out = self.model(input_ids=self.tokens[:, : self.prompt_length], use_cache=True)
            cache = out.past_key_values
            for position in range(self.prompt_length, self.tokens.shape[1]):
                out = self.model(
                    input_ids=self.tokens[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                if damage is not None and position == damage_at:
                    damage()

    @property
    def tissue(self):
        return self.homeostat.homeostat

    def tail_error(self, tokens: int = TAIL_TOKENS) -> float:
        history = list(self.tissue.alignment_history)[-tokens:]
        return self.tissue.setpoint - sum(history) / len(history)

    def tail_strength(self, tokens: int = TAIL_TOKENS) -> float:
        history = list(self.tissue.strength_history)[-tokens:]
        return sum(history) / len(history)


def _load_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, local_files_only=True, dtype=torch.bfloat16
        )
    except OSError as exc:  # pragma: no cover - depends on the runner's cache
        message = f"{MODEL_ID} is not in the local HuggingFace cache: {exc}"
        if os.environ.get("CI"):
            pytest.fail(
                message + " -- the GPU runner must mount the workstation's cache into the "
                "job (docker volume ~/.cache/huggingface:/root/.cache/huggingface:ro) so the "
                "real-model tests run rather than skip"
            )
        pytest.skip(message)
    return model, tokenizer


def build_served() -> ServedSystem:
    """The served system as ``app.build_homeostat`` builds it, plus the seeded coupling.

    Seeded: the MoB conversion jitters the adapters and initialises the heads at
    random, and an unseeded fixture generates a different continuation in every
    process, so its numbers would move between runs even though each run's regimes
    compare like with like.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, tokenizer = _load_model()
    device = torch.device("cuda")
    model.to(device)  # pyright: ignore[reportArgumentType] # HF stubs
    model.eval()

    mob_config = MoBConfig(
        num_experts=4,
        top_k=2,
        hidden_dim=PROFILE["hidden_dim"],
        intermediate_dim=PROFILE["intermediate_dim"],
        adapter_rank=32,
        adapter_alpha=16.0,
        use_loss_feedback=False,
        use_local_quality=True,
        use_differentiable_routing=False,
    )
    layers = list(range(PROFILE["mob_layers_start"], PROFILE["mob_layers_end"]))
    model = apply_mob_to_model(model, mob_config, layers)
    model.eval()

    template = SteeringConfig(steering_layers=layers, adaptive=True)
    config = serving_config(GOAL, template, model_id=MODEL_ID)
    extraction = extract_steering_vectors(
        model, tokenizer, goal=GOAL, config=config, max_pairs=EXTRACTION_PAIRS
    )
    assert extraction.certified, extraction.fallback_reason
    homeostat = CognitiveHomeostat(config)
    homeostat.add_steering_vectors(extraction.vectors)
    homeostat.estimate_capability_subspaces(model, tokenizer)
    texts = calibration_texts(
        model, tokenizer, GOAL, num_prompts=CALIBRATION_PROMPTS, new_tokens=CALIBRATION_TOKENS
    )
    homeostat.calibrate(model, tokenizer, texts=texts)

    coupled = seed_coupling(
        model,
        homeostat,
        extraction,
        certified_coupling_layers(GOAL, MODEL_ID),
        SteeringCouplingConfig(hidden_dim=PROFILE["hidden_dim"]),
    )
    assert sorted(coupled) == [13, 16, 17, 18, 19, 20, 21]
    for coupling in coupled.values():
        with torch.no_grad():
            coupling.detector.copy_(coupling.steering_direction)
        coupling.set_coupling_step(coupling.config.warmup_steps)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad(), frozen_economy(model):
        tokens = model.generate(
            **inputs,
            max_new_tokens=GENERATED_TOKENS,
            min_new_tokens=GENERATED_TOKENS,
            do_sample=False,
        )
    prompt_length = int(inputs["input_ids"].shape[1])

    return ServedSystem(
        model,
        tokenizer,
        homeostat,
        config,
        device,
        homeostat.homeostat.gains(),
        tokens,
        prompt_length,
    )


@pytest.fixture(scope="module")
def served() -> ServedSystem:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return build_served()


@contextmanager
def attached(system: ServedSystem, inert: bool = False) -> Iterator[None]:
    """The tissue on the model for one generation, live or sensing-only, then off again."""
    system.homeostat.reset()
    kp, ki = (0.0, 0.0) if inert else system.derived_gains
    system.tissue.set_gains(kp=kp, ki=ki)
    system.homeostat.attach_to_model(system.model)
    try:
        with frozen_economy(system.model):
            yield
    finally:
        system.homeostat.detach_from_model()
        system.tissue.set_gains(kp=system.derived_gains[0], ki=system.derived_gains[1])


@contextmanager
def content_push(system: ServedSystem, amount: float) -> Iterator[None]:
    """A constant offset along the injected direction at every actuator block, after its cell."""
    layers = cast(nn.ModuleList, system.model.model.layers)  # type: ignore[union-attr]

    def hook_for(layer: int):
        direction = system.homeostat.projected_direction(layer)[0]

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pushed = hidden + amount * direction.to(hidden.device, hidden.dtype)
            return (pushed,) + tuple(output[1:]) if isinstance(output, tuple) else pushed

        return hook

    handles = [
        layers[layer].register_forward_hook(hook_for(layer))
        for layer in system.homeostat.actuator_layers
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def test_the_served_tissue_is_the_measured_one(served: ServedSystem):
    """The calibration that runs here is #4's in shape: cells 13, 16-22, gains inside the bound.

    The blind cell's weight in the consensus is zero and every other cell's is its
    gain, so the tissue gain the shared gains derive from is the gain-weighted mean
    of the lifts -- above the plain mean, which counted the zero (#21).
    """
    calibration = served.homeostat.calibration
    assert calibration is not None
    assert calibration.sensors == (13, 16, 17, 18, 19, 20, 21, 22)
    assert calibration.layers[13].lift == pytest.approx(0.0, abs=1e-6)
    assert all(calibration.layers[layer].lift > 0 for layer in (16, 17, 18, 19, 20, 21, 22))
    kp, ki = served.derived_gains
    limit = served.tissue.max_stable_ki()
    assert limit is not None and 0 < ki < limit and kp > 0

    gains = {layer: calibration.layers[layer].gain_z for layer in calibration.sensors}
    assert served.tissue.cell_weight(13) == 0.0
    assert all(served.tissue.cell_weight(layer) == gains[layer] for layer in gains if layer != 13)
    weighted = sum(gain * gain for gain in gains.values()) / sum(gains.values())
    assert calibration.gain_z == pytest.approx(weighted)
    assert calibration.gain_z > sum(gains.values()) / len(gains)
    assert served.tissue.setpoint == pytest.approx(calibration.gain_z * served.config.base_strength)


@pytest.fixture(scope="module")
def inert_under_push(served: ServedSystem) -> tuple[float, float]:
    """The constant-strength loop's tail error and strength under the push: the reference."""
    with attached(served, inert=True), content_push(served, CONTENT_PUSH):
        served.replay()
        return served.tail_error(), served.tail_strength()


def test_the_live_tissue_recovers_from_content_pushed_against_the_direction(
    served: ServedSystem, inert_under_push: tuple[float, float]
):
    """Designed perturbation on the real plant, against the inert loop on the same tokens.

    Per cell over the pushed tail (sigma; inert -> live): 13 -0.24 -> -0.24, 16 +1.2
    -> +0.8, 17 +3.5 -> +2.0, 18 +5.9 -> +4.1, 19 +4.4 -> +2.0, 20 0.0 -> -1.8, 21
    -2.9 -> -3.7, 22 -3.1 -> -3.4: the live loop moves the high-gain middle cells
    toward setpoint and pushes the already-over-aligned top cells further, which is
    the least-squares compromise a single common strength can make. The blind cell
    reads the same either way and is reported with a weight of zero.
    """
    inert_error, inert_strength = inert_under_push
    with attached(served), content_push(served, CONTENT_PUSH):
        served.replay()
        live_error = served.tail_error()
        live_strength = served.tail_strength()
        blind = next(cell for cell in served.tissue.status()["cells"] if cell["layer"] == 13)

    assert inert_error > INERT_ERROR_SIGMA, "the push must bite for the recovery to mean anything"
    assert abs(live_error) < 0.5 * abs(inert_error), (live_error, inert_error)
    assert abs(live_error) < RECOVERY_FRACTION * served.tissue.setpoint, (live_error, inert_error)
    assert live_strength > inert_strength + STRENGTH_RISE
    assert blind["weight"] == 0.0 and blind["alive"] and abs(blind["error"]) > 0.0


def test_the_tissue_carries_on_after_its_top_actuator_is_removed_mid_replay(
    served: ServedSystem, inert_under_push: tuple[float, float]
):
    """Undesigned perturbation on the real plant: the top cell stops firing under load.

    Halfway through the replay the top actuator's hook is removed. The cell leaves
    the consensus after one pass, and with it its own reading -- on these tokens
    about -3 sigma -- so the consensus the survivors regulate is a different
    number from before, and rises mechanically -- as does the setpoint it is
    measured against, the survivors' own since #22 (5.24 sigma against the
    calibrated 5.05: cell 21's gain, 0.88, is below the consensus 1.26). What the
    damaged tissue must still do is act on it: the survivors raise their strength
    over the second half, and six cells live still hold the error below what the
    intact constant-strength loop leaves (measured: +0.79 sigma at 4.00 before the
    removal, +1.23 at 5.21 after it, against +2.16 inert; +1.03 against the
    calibrated setpoint). The removal is at the top, so every survivor keeps its
    calibrated weight: the rule #22 added fires only below a cell.
    """
    inert_error, _ = inert_under_push
    top = max(served.homeostat.actuator_layers)
    before: dict[str, float] = {}

    def remove_top_actuator() -> None:
        before["strength"] = served.tail_strength(DAMAGED_TAIL)
        served.homeostat._registered_hooks[list(served.homeostat.hooks).index(top)].remove()

    with attached(served), content_push(served, CONTENT_PUSH):
        served.replay(damage_at=served.prompt_length + DAMAGE_AT, damage=remove_top_actuator)
        cells = {cell["layer"]: cell for cell in served.tissue.status()["cells"]}
        after_error = served.tail_error(DAMAGED_TAIL)
        after_strength = served.tail_strength(DAMAGED_TAIL)
        after_setpoint = served.tissue.setpoint

    calibration = served.tissue.calibration
    assert calibration is not None
    survivors = {layer: cell for layer, cell in cells.items() if layer != top}
    assert cells[top]["alive"] is False and all(cell["alive"] for cell in survivors.values())
    assert all(cell["weight"] == calibration.weight(layer) for layer, cell in survivors.items())
    assert after_setpoint != pytest.approx(served.tissue.nominal_setpoint)
    assert after_strength > before["strength"] + SURVIVOR_STRENGTH_RISE
    assert abs(after_error) < abs(inert_error), (after_error, inert_error)


def _split_by_tier(pairs, held_out_per_tier):
    held_out, extract = [], []
    seen = dict.fromkeys(TIERS, 0)
    for pair in reversed(pairs):
        if seen[pair.tier] < held_out_per_tier:
            held_out.append(pair)
            seen[pair.tier] += 1
        else:
            extract.append(pair)
    return list(reversed(extract)), list(reversed(held_out))


def test_the_safe_direction_passes_the_behavioural_gate_on_the_certified_split(served):
    """The README's certification row for ``safe``, re-measured: +0.149 vs random max +0.030."""
    pairs = list(
        load_contrastive_dataset("safe", source=BUILTIN_SOURCE, pair_format=COMPLETION_FORMAT)
    )
    extract, held_out = _split_by_tier(pairs, HELD_OUT_PER_TIER)
    extractor = SteeringVectorExtractor(served.model, served.tokenizer, SAFE_LAYERS)
    vectors = extractor.extract_from_pairs(extract)
    control = extractor.extract_from_pairs(list(load_instruction_prefix_control("safe")))
    config = replace(
        served.config,
        steering_layers=SAFE_LAYERS,
        base_strength=SAFE_STRENGTH,
        adaptive=False,
        orthogonal_projection=False,
    )

    result = validate_steering_vector(
        served.model,
        served.tokenizer,
        goal="safe",
        vectors=vectors,
        held_out=held_out,
        config=config,
        device=served.device,
        control_vectors=control,
        num_random=8,
        seed=0,
    )

    assert result.vector_effect.effect > 0
    assert result.beats_random, (result.vector_effect.effect, result.random_effects)
    assert result.beats_control, (result.vector_effect.effect, result.control_effect)
