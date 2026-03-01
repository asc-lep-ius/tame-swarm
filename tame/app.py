from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_active_profile
from mob import MixtureOfBidders, MoBConfig, apply_mob_to_model, load_mob_state
from steering import (
    CognitiveHomeostat,
    SteeringConfig,
    create_default_steering_vectors,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class TAMEApplication:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    homeostat: CognitiveHomeostat | None
    mob_config: MoBConfig
    steering_config: SteeringConfig
    model_id: str

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

        steering_config = SteeringConfig(
            steering_layers=list(range(profile["mob_layers_start"], profile["mob_layers_end"])),
            base_strength=0.3,
            adaptive=True,
            target_alignment=0.7,
            kp=0.5,
        )

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
            devices_used = set(model.hf_device_map.values())
            logger.info("[GESTATIONAL] Model distributed across devices: %s", devices_used)

        logger.info("[GESTATIONAL] Base model loaded")

        logger.info("[MORPHOGENESIS] Applying Mixture of Bidders transformation...")

        layers_to_modify = list(range(profile["mob_layers_start"], profile["mob_layers_end"]))

        logger.info("[MORPHOGENESIS] Targeting %d layers for MoB transformation", len(layers_to_modify))
        model = apply_mob_to_model(model, mob_config, layers_to_modify)

        logger.info(
            "[MORPHOGENESIS] MoB applied to layers %d-%d",
            layers_to_modify[0],
            layers_to_modify[-1],
        )

        model.eval()

        logger.info("[DIAGNOSTIC] Testing MoB output validity...")
        try:
            test_input = tokenizer("Test", return_tensors="pt").to(model.device)
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
                    logger.error("[DIAGNOSTIC] WARNING: Model producing NaN/Inf! Check MoB configuration.")
                elif mean_val < 0.01 or std_val < 0.01:
                    logger.warning("[DIAGNOSTIC] WARNING: Hidden states may be collapsed (very low variance)")
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
                    logger.warning("[MORPHOGENESIS] Failed to load mob_state from %s: %s", state_path, e)
        else:
            logger.info("[MORPHOGENESIS] No trained mob_state found - experts start with default wealth")

        logger.info("[HOMEOSTASIS] Extracting steering vectors for goal persistence...")

        homeostat: CognitiveHomeostat | None = None
        try:
            steering_vectors = create_default_steering_vectors(
                model,
                tokenizer,
                goal="truthful",
                layers=steering_config.steering_layers,
            )

            homeostat = CognitiveHomeostat(steering_config)
            homeostat.add_steering_vectors(steering_vectors)
            homeostat.attach_to_model(model)

            logger.info("[HOMEOSTASIS] Steering attached to %d layers", len(steering_vectors))
        except Exception as e:
            logger.warning("[HOMEOSTASIS] Steering extraction failed: %s", e)
            logger.warning("[HOMEOSTASIS] Continuing without steering (degraded mode)")
            homeostat = None

        logger.info("=" * 60)
        logger.info("TAME SWARM: Online and Self-Regulating")
        logger.info("=" * 60)

        return cls(
            model=model,
            tokenizer=tokenizer,
            homeostat=homeostat,
            mob_config=mob_config,
            steering_config=steering_config,
            model_id=model_id,
        )

    def start_mob_tracking(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MixtureOfBidders):
                layer.mlp.start_tracking()

    def stop_mob_tracking(self) -> None:
        for layer in self.model.model.layers:
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MixtureOfBidders):
                layer.mlp.stop_tracking()

    def get_mob_wealth_traces(self) -> dict[str, list[list[float]]]:
        traces: dict[str, list[list[float]]] = {}
        for idx, layer in enumerate(self.model.model.layers):
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
