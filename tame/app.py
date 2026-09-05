from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_active_profile
from homeostat import CognitiveHomeostat
from mob import MixtureOfBidders, MoBConfig, apply_mob_to_model, load_mob_state
from steering import SteeringConfig
from steering_pipeline import (
    SteeringExtraction,
    calibration_texts,
    extract_steering_vectors,
    serving_config,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# The goal a server starts on. Its layers, reference strength and strength band come
# from the certification record (``contrastive_data.CERTIFIED``), not from here.
DEFAULT_GOAL = "truthful"
# Whether the served tissue adapts the strength or holds the certified constant.
# #4's value test (``scripts/characterise_plant.py``, adaptive vs constant on 100
# held-out letter choices after a generated rationale) found no significant
# difference (-0.18 +/- 0.29 log-odds; accuracy 0.71 -> 0.69), so the loop ships
# calibrated and switchable (``PUT /steering/gains`` with ``adaptive``) but off by
# default: the served system is exactly the constant-strength configuration the
# gate certified.
ADAPTIVE_STEERING = False


def build_homeostat(
    model: nn.Module,
    tokenizer,
    template: SteeringConfig,
    goal: str,
    model_id: str | None = None,
    strength: float | None = None,
) -> tuple[CognitiveHomeostat, SteeringExtraction, SteeringConfig]:
    """Extract, calibrate and attach the loop for ``goal``; startup and the API share this path.

    The config is derived from the goal's certification (layers, reference
    strength, band); ``strength`` overrides the reference strength -- and so the
    setpoint the calibration measures -- but only inside the certified band.
    ``template`` must be the pristine template, never a previously served config:
    gains pinned through the API and a certified goal's layers would otherwise
    leak into the next goal. Calibration failures degrade to the legacy cosine
    loop with a warning rather than leaving the server unsteered.
    """
    config = serving_config(goal, template, model_id=model_id)
    if strength is not None:
        if strength <= 0:
            raise ValueError(f"strength must be positive, got {strength}")
        if not config.min_strength <= strength <= config.max_strength:
            raise ValueError(
                f"strength {strength} is outside the certified band "
                f"[{config.min_strength}, {config.max_strength}] for goal {goal!r}"
            )
        config.base_strength = strength

    extraction = extract_steering_vectors(model, tokenizer, goal=goal, config=config)
    logger.info(
        "[HOMEOSTASIS] Extracted %r from %d %s pairs (%s, %s), tiers %s; layers %s, readout %s, "
        "strength %.2f in [%.2f, %.2f]",
        goal,
        extraction.pair_count,
        extraction.pair_format,
        extraction.source,
        "certified" if extraction.certified else "UNCERTIFIED",
        extraction.tier_counts,
        config.steering_layers,
        config.readout_layer,
        config.base_strength,
        config.min_strength,
        config.max_strength,
    )
    if not extraction.certified:
        logger.warning(
            "[HOMEOSTASIS] Steering on an uncertified '%s' vector: %s",
            extraction.goal,
            extraction.fallback_reason or "source/format is not the certified pair",
        )

    homeostat = CognitiveHomeostat(config)
    homeostat.add_steering_vectors(extraction.vectors)

    if config.orthogonal_projection:
        homeostat.estimate_capability_subspaces(model, tokenizer)
        retention = homeostat.get_capability_retention()
        if retention:
            logger.info(
                "[HOMEOSTASIS] Capability projection retains %.0f%%-%.0f%% "
                "of the steering vector across layers",
                100 * min(retention.values()),
                100 * max(retention.values()),
            )

    try:
        texts = calibration_texts(model, tokenizer, goal)
        homeostat.calibrate(model, tokenizer, texts=texts)
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "[HOMEOSTASIS] Alignment calibration failed (%s); the loop runs uncalibrated "
            "on the legacy cosine setpoint",
            exc,
        )

    homeostat.attach_to_model(model)
    logger.info(
        "[HOMEOSTASIS] Steering attached: actuators %s, readout %d, %s",
        homeostat.actuator_layers,
        homeostat.readout_layer,
        "adaptive" if config.adaptive else "constant strength",
    )
    return homeostat, extraction, config


@dataclass
class TAMEApplication:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    homeostat: CognitiveHomeostat | None
    mob_config: MoBConfig
    steering_config: SteeringConfig
    model_id: str
    # The pristine loop settings every goal install starts from; ``steering_config``
    # is the config of the goal currently served, and carries its pinned gains.
    steering_template: SteeringConfig | None = None

    @classmethod
    def from_profile(cls) -> TAMEApplication:
        profile = get_active_profile()
        model_id = profile["model_id"]

        mob_config = MoBConfig(
            num_experts=4,
            top_k=2,
            hidden_dim=profile["hidden_dim"],
            intermediate_dim=profile["intermediate_dim"],
            initial_wealth=75.0,
            wealth_decay=0.997,
            min_wealth=15.0,
            max_wealth=750.0,
            jitter_std=0.08,
            reward_scale=2.0,
            use_vcg_payments=True,
            use_shared_base=True,
            adapter_rank=32,
            adapter_alpha=16.0,
            use_loss_feedback=False,
            use_local_quality=True,
            use_differentiable_routing=False,
            inference_wealth_decay=0.98,
            inference_exploration_bonus=0.03,
            inference_wealth_compression=0.4,
        )

        # A template: the served goal's layers, reference strength and band come
        # from its certification record (see build_homeostat). The MoB layer range
        # here is only what an uncertified goal falls back to.
        steering_template = SteeringConfig(
            steering_layers=list(range(profile["mob_layers_start"], profile["mob_layers_end"])),
            adaptive=ADAPTIVE_STEERING,
        )
        steering_config = steering_template

        logger.info("=" * 60)
        logger.info("TAME SWARM: Initializing Agential Architecture")
        logger.info("=" * 60)

        logger.info("[GESTATIONAL] Loading base model: %s", model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if hasattr(model, "hf_device_map"):
            devices_used = set(model.hf_device_map.values())  # pyright: ignore[reportCallIssue] # hf_device_map is a dict at runtime, stubs see Tensor
            logger.info("[GESTATIONAL] Model distributed across devices: %s", devices_used)

        logger.info("[GESTATIONAL] Base model loaded")

        logger.info("[MORPHOGENESIS] Applying Mixture of Bidders transformation...")

        layers_to_modify = list(range(profile["mob_layers_start"], profile["mob_layers_end"]))

        logger.info(
            "[MORPHOGENESIS] Targeting %d layers for MoB transformation", len(layers_to_modify)
        )
        model = apply_mob_to_model(model, mob_config, layers_to_modify)

        logger.info(
            "[MORPHOGENESIS] MoB applied to layers %d-%d",
            layers_to_modify[0],
            layers_to_modify[-1],
        )

        model.eval()

        logger.info("[DIAGNOSTIC] Testing MoB output validity...")
        try:
            test_input = tokenizer("Test", return_tensors="pt").to(model.device)  # pyright: ignore[reportCallIssue] # AutoTokenizer stubs lack __call__
            with torch.inference_mode():
                test_output = model(**test_input, output_hidden_states=True)
                last_hidden = test_output.hidden_states[-1]
                has_nan = torch.isnan(last_hidden).any().item()
                has_inf = torch.isinf(last_hidden).any().item()
                mean_val = last_hidden.abs().mean().item()
                std_val = last_hidden.std().item()
                logger.info(
                    "[DIAGNOSTIC] Hidden states: mean_abs=%.4f, std=%.4f, has_nan=%s, has_inf=%s",
                    mean_val,
                    std_val,
                    has_nan,
                    has_inf,
                )
                if has_nan or has_inf:
                    logger.error(
                        "[DIAGNOSTIC] WARNING: Model producing NaN/Inf! Check MoB configuration."
                    )
                elif mean_val < 0.01 or std_val < 0.01:
                    logger.warning(
                        "[DIAGNOSTIC] WARNING: Hidden states may be collapsed (very low variance)"
                    )
                else:
                    logger.info("[DIAGNOSTIC] MoB output looks valid")
        except Exception as e:
            logger.warning("[DIAGNOSTIC] Test failed: %s", e)

        mob_state_paths = [
            "./tame_inference/mob_state.pt",
            "./mob_state.pt",
        ]

        compression = mob_config.inference_wealth_compression
        for state_path in mob_state_paths:
            if os.path.exists(state_path):
                try:
                    loaded = load_mob_state(model, state_path, compress_wealth=compression)
                    if loaded > 0:
                        logger.info(
                            "[MORPHOGENESIS] Restored trained expert specialization from %s",
                            state_path,
                        )
                    break
                except Exception as e:
                    logger.warning(
                        "[MORPHOGENESIS] Failed to load mob_state from %s: %s", state_path, e
                    )
        else:
            logger.info(
                "[MORPHOGENESIS] No trained mob_state found - experts start with default wealth"
            )

        logger.info("[HOMEOSTASIS] Extracting steering vectors for goal persistence...")

        homeostat: CognitiveHomeostat | None = None
        try:
            homeostat, _, steering_config = build_homeostat(
                model, tokenizer, steering_template, DEFAULT_GOAL, model_id=model_id
            )
        except Exception as e:
            logger.warning("[HOMEOSTASIS] Steering extraction failed: %s", e)
            logger.warning("[HOMEOSTASIS] Continuing without steering (degraded mode)")
            homeostat = None

        logger.info("=" * 60)
        logger.info("TAME SWARM: Online and Self-Regulating")
        logger.info("=" * 60)

        return cls(
            model=model,  # pyright: ignore[reportArgumentType] # apply_mob_to_model returns Module but is still AutoModelForCausalLM at runtime
            tokenizer=tokenizer,
            homeostat=homeostat,
            mob_config=mob_config,
            steering_config=steering_config,
            model_id=model_id,
            steering_template=steering_template,
        )

    def install_goal(self, goal: str, strength: float | None = None) -> SteeringExtraction:
        """Swap the served goal at runtime: detach, re-extract, re-calibrate, re-attach.

        The old hooks come off first because a calibration measured under them
        would not be a resting stream; if the new goal cannot be built they go
        straight back, so the server never silently serves an unsteered model.
        """
        template = self.steering_template or self.steering_config
        previous = self.homeostat
        if previous is not None:
            previous.detach_from_model()
        try:
            homeostat, extraction, config = build_homeostat(
                self.model,  # pyright: ignore[reportArgumentType] # AutoModelForCausalLM is an nn.Module at runtime
                self.tokenizer,
                template,
                goal,
                model_id=self.model_id,
                strength=strength,
            )
        except Exception:
            if previous is not None:
                previous.attach_to_model(self.model)  # pyright: ignore[reportArgumentType] # as above
            raise
        self.homeostat, self.steering_config = homeostat, config
        return extraction

    def start_mob_tracking(self) -> None:
        for layer in self.model.model.layers:  # pyright: ignore[reportAttributeAccessIssue] # HuggingFace Auto* stubs lack runtime model internals
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MixtureOfBidders):
                layer.mlp.start_tracking()

    def stop_mob_tracking(self) -> None:
        for layer in self.model.model.layers:  # pyright: ignore[reportAttributeAccessIssue] # HuggingFace Auto* stubs lack runtime model internals
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MixtureOfBidders):
                layer.mlp.stop_tracking()

    def get_mob_wealth_traces(self) -> dict[str, list[list[float]]]:
        traces: dict[str, list[list[float]]] = {}
        for idx, layer in enumerate(self.model.model.layers):  # pyright: ignore[reportAttributeAccessIssue] # HuggingFace Auto* stubs lack runtime model internals
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MixtureOfBidders):
                history = layer.mlp.get_wealth_history()
                if history:
                    traces[str(idx)] = history
        return traces

    def get_aggregated_wealth_trace(self) -> dict[str, Any]:
        traces = self.get_mob_wealth_traces()
        if not traces:
            return {"steps": [], "expert_wealth": []}

        all_histories = list(traces.values())
        if not all_histories:
            return {"steps": [], "expert_wealth": []}

        num_experts = len(all_histories[0][0]) if all_histories[0] else 0
        max_steps = max(len(h) for h in all_histories)

        aggregated = []
        for step in range(max_steps):
            step_wealth = [0.0] * num_experts
            count = 0
            for history in all_histories:
                if step < len(history):
                    for e in range(num_experts):
                        step_wealth[e] += history[step][e]
                    count += 1
            if count > 0:
                step_wealth = [w / count for w in step_wealth]
            aggregated.append(step_wealth)

        return {
            "steps": list(range(max_steps)),
            "expert_wealth": aggregated,
            "num_experts": num_experts,
            "num_layers": len(traces),
        }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    tame_app = TAMEApplication.from_profile()
    app.state.tame = tame_app
    yield
    if tame_app.homeostat:
        tame_app.homeostat.detach_from_model()
    logger.info("TAME Swarm shutting down")


def create_app() -> FastAPI:
    from routes import router

    application = FastAPI(
        title="TAME Swarm: Agential Swarm Node",
        description=(
            "A bio-inspired LLM inference server implementing the TAME architecture. "
            "Features Mixture of Bidders for emergent specialization and "
            "Activation Steering for cognitive homeostasis."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application
