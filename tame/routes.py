import asyncio
import json
import logging
from threading import Thread

import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from app import TAMEApplication
from dependencies import get_tame_app
from models import GenerateRequest, GenerateResponse, HealthResponse, SwarmStatus
from steering import create_default_steering_vectors

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(tame: TAMEApplication = Depends(get_tame_app)):
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        gpu_name = "Unknown"

    return HealthResponse(
        status="alive",
        gpu=gpu_name,
        model_id=tame.model_id,
        architecture="TAME (Mixture of Bidders + Cognitive Homeostasis)",
        mob_active=True,
        steering_active=tame.homeostat is not None,
    )


@router.get("/swarm/status", response_model=SwarmStatus)
def get_swarm_status(tame: TAMEApplication = Depends(get_tame_app)):
    from mob import MixtureOfBidders

    total_wealth = torch.zeros(tame.mob_config.num_experts)
    total_usage = torch.zeros(tame.mob_config.num_experts)
    num_mob_layers = 0

    for layer in tame.model.model.layers:
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MixtureOfBidders):
            mob = layer.mlp
            total_wealth += mob.expert_wealth.cpu()
            total_usage += mob.expert_usage_count.cpu()
            num_mob_layers += 1

    if num_mob_layers > 0:
        avg_wealth = (total_wealth / num_mob_layers).tolist()
        avg_usage = total_usage.tolist()
    else:
        avg_wealth = [0.0] * tame.mob_config.num_experts
        avg_usage = [0.0] * tame.mob_config.num_experts

    return SwarmStatus(
        num_experts=tame.mob_config.num_experts,
        expert_wealth=avg_wealth,
        expert_usage=avg_usage,
        layers_modified=num_mob_layers,
    )


@router.get("/homeostasis/status")
def get_homeostasis_status(tame: TAMEApplication = Depends(get_tame_app)):
    if tame.homeostat is None:
        return {"status": "disabled", "message": "Steering not active"}

    stats = tame.homeostat.get_alignment_stats()
    return {
        "status": "active",
        "config": {
            "base_strength": tame.steering_config.base_strength,
            "adaptive": tame.steering_config.adaptive,
            "target_alignment": tame.steering_config.target_alignment,
        },
        "current_stats": stats,
    }


@router.get("/traces/wealth")
def get_wealth_traces(tame: TAMEApplication = Depends(get_tame_app)):
    return tame.get_aggregated_wealth_trace()


@router.get("/traces/steering")
def get_steering_traces(tame: TAMEApplication = Depends(get_tame_app)):
    if tame.homeostat is None:
        return {"status": "disabled", "alignment_history": [], "strength_history": []}

    stats = tame.homeostat.get_alignment_stats()
    return {
        "status": "active",
        "alignment_history": stats.get("alignment_history", []),
        "strength_history": stats.get("strength_history", []),
        "target_alignment": tame.steering_config.target_alignment,
        "base_strength": tame.steering_config.base_strength,
        "mean_alignment": stats.get("mean_alignment"),
        "mean_strength": stats.get("mean_strength"),
    }


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, tame: TAMEApplication = Depends(get_tame_app)):
    original_strength = tame.homeostat.config.base_strength if tame.homeostat else None
    original_adaptive = tame.homeostat.config.adaptive if tame.homeostat else None

    try:
        if tame.homeostat:
            tame.homeostat.reset()
            if req.steering_strength is not None:
                tame.homeostat.config.base_strength = req.steering_strength
                tame.homeostat.config.adaptive = False

        messages = [{"role": "user", "content": req.prompt}]
        text = tame.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tame.tokenizer(text, return_tensors="pt").to(tame.model.device)

        with torch.inference_mode():
            outputs = tame.model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                top_k=50,
                top_p=0.95,
                pad_token_id=tame.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        response_text = tame.tokenizer.decode(generated_ids, skip_special_tokens=True)

        usage = {
            "input_tokens": inputs.input_ids.shape[1],
            "output_tokens": len(generated_ids),
        }

        homeostasis_stats = None
        if tame.homeostat:
            homeostasis_stats = tame.homeostat.get_alignment_stats()

        mob_stats = None
        if req.return_stats:
            swarm_status = get_swarm_status(tame)
            mob_stats = {
                "expert_wealth": swarm_status.expert_wealth,
                "expert_usage": swarm_status.expert_usage,
            }

        return GenerateResponse(
            response=response_text,
            usage=usage,
            homeostasis=homeostasis_stats,
            mob_stats=mob_stats,
        )

    except Exception as e:
        logger.error("Generation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed") from e

    finally:
        if tame.homeostat and original_strength is not None:
            tame.homeostat.config.base_strength = original_strength
            tame.homeostat.config.adaptive = original_adaptive


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest, tame: TAMEApplication = Depends(get_tame_app)):
    original_strength = tame.homeostat.config.base_strength if tame.homeostat else None
    original_adaptive = tame.homeostat.config.adaptive if tame.homeostat else None

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Preparing generation...'})}\n\n"

            if tame.homeostat:
                tame.homeostat.reset()
                if req.steering_strength is not None:
                    tame.homeostat.config.base_strength = req.steering_strength
                    tame.homeostat.config.adaptive = False

            tame.start_mob_tracking()

            messages = [{"role": "user", "content": req.prompt}]
            text = tame.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tame.tokenizer(text, return_tensors="pt").to(tame.model.device)
            input_length = inputs.input_ids.shape[1]

            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {input_length} input tokens...'})}\n\n"

            streamer = TextIteratorStreamer(
                tame.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=2400,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": req.max_tokens,
                "do_sample": req.temperature > 0,
                "temperature": req.temperature if req.temperature > 0 else 1.0,
                "top_k": 50,
                "top_p": 0.95,
                "pad_token_id": tame.tokenizer.pad_token_id,
                "streamer": streamer,
            }

            def generate_in_thread():
                try:
                    with torch.inference_mode():
                        tame.model.generate(**generation_kwargs)
                except Exception as e:
                    logger.error("Generation thread error: %s", e)

            thread = Thread(target=generate_in_thread)
            thread.start()

            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"

            full_response = ""
            token_count = 0
            last_status_update = 0
            last_trace_update = 0

            for token_text in streamer:
                full_response += token_text
                token_count += 1

                yield f"data: {json.dumps({'type': 'token', 'content': token_text})}\n\n"

                if token_count - last_status_update >= 10:
                    last_status_update = token_count
                    status_msg = f"Generated {token_count} tokens"

                    try:
                        for layer in tame.model.model.layers:
                            if hasattr(layer, "mlp") and hasattr(layer.mlp, "last_stats"):
                                stats = layer.mlp.last_stats
                                if stats and "expert_wealth" in stats:
                                    wealth = stats["expert_wealth"]
                                    top_expert = wealth.argmax().item()
                                    status_msg += f" | Expert {top_expert} leading"
                                    break
                    except Exception:
                        pass

                    yield f"data: {json.dumps({'type': 'progress', 'message': status_msg, 'tokens': token_count})}\n\n"

                if token_count - last_trace_update >= 25:
                    last_trace_update = token_count
                    try:
                        trace_update: dict = {"type": "trace_update", "tokens": token_count}

                        wealth_data = tame.get_aggregated_wealth_trace()
                        if wealth_data.get("expert_wealth"):
                            expert_wealth = wealth_data["expert_wealth"]
                            if len(expert_wealth) > 100:
                                step = len(expert_wealth) // 100
                                expert_wealth = expert_wealth[::step]
                            trace_update["wealth_trace"] = {
                                "steps": list(range(len(expert_wealth))),
                                "expert_wealth": expert_wealth,
                                "num_experts": wealth_data.get("num_experts", 0),
                                "num_layers": wealth_data.get("num_layers", 0),
                            }

                        if tame.homeostat:
                            homeo_stats = tame.homeostat.get_alignment_stats()
                            strength_history = homeo_stats.get("strength_history", [])
                            alignment_history = homeo_stats.get("alignment_history", [])

                            if len(strength_history) > 100:
                                subsample_step = len(strength_history) // 100
                                strength_history = strength_history[::subsample_step]
                                alignment_history = (
                                    alignment_history[::subsample_step] if alignment_history else []
                                )

                            if strength_history:
                                trace_update["steering_trace"] = {
                                    "strength_history": strength_history,
                                    "alignment_history": alignment_history,
                                    "target_alignment": tame.steering_config.target_alignment,
                                }

                        yield f"data: {json.dumps(trace_update)}\n\n"
                    except Exception as e:
                        logger.warning("Error sending trace update: %s", e)
                    yield f"data: {json.dumps({'type': 'progress', 'message': status_msg, 'tokens': token_count})}\n\n"

                await asyncio.sleep(0.01)

            thread.join(timeout=10)

            tame.stop_mob_tracking()

            final_stats: dict = {
                "type": "complete",
                "usage": {
                    "input_tokens": input_length,
                    "output_tokens": token_count,
                },
            }

            if tame.homeostat:
                homeo_stats = tame.homeostat.get_alignment_stats()
                final_stats["homeostasis"] = {
                    "mean_alignment": homeo_stats.get("mean_alignment"),
                    "current_strength": homeo_stats.get("current_strength"),
                    "mean_strength": homeo_stats.get("mean_strength"),
                }
                final_stats["steering_trace"] = {
                    "alignment_history": homeo_stats.get("alignment_history", []),
                    "strength_history": homeo_stats.get("strength_history", []),
                    "target_alignment": tame.steering_config.target_alignment,
                }

            if req.return_stats:
                try:
                    swarm_status = get_swarm_status(tame)
                    wealth_trace = tame.get_aggregated_wealth_trace()
                    final_stats["mob_stats"] = {
                        "expert_wealth": swarm_status.expert_wealth,
                        "expert_usage": swarm_status.expert_usage,
                    }
                    final_stats["wealth_trace"] = wealth_trace
                except Exception:
                    pass

            yield f"data: {json.dumps(final_stats)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            tame.stop_mob_tracking()
            yield f"data: {json.dumps({'type': 'error', 'message': 'Streaming generation failed'})}\n\n"

        finally:
            if tame.homeostat and original_strength is not None:
                tame.homeostat.config.base_strength = original_strength
                tame.homeostat.config.adaptive = original_adaptive

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/steering/update")
async def update_steering(
    tame: TAMEApplication = Depends(get_tame_app),
    goal: str = "truthful",
    strength: float = 0.3,
):
    if tame.homeostat is None:
        raise HTTPException(status_code=400, detail="Steering not initialized")

    try:
        tame.homeostat.detach_from_model()

        steering_vectors = create_default_steering_vectors(
            tame.model,
            tame.tokenizer,
            goal=goal,
            layers=tame.steering_config.steering_layers,
        )

        tame.homeostat.config.base_strength = strength

        tame.homeostat.add_steering_vectors(steering_vectors)
        tame.homeostat.attach_to_model(tame.model)

        return {
            "status": "updated",
            "goal": goal,
            "strength": strength,
            "layers": tame.steering_config.steering_layers,
        }

    except Exception as e:
        logger.error("Steering update error: %s", e)
        raise HTTPException(status_code=500, detail="Steering update failed") from e
