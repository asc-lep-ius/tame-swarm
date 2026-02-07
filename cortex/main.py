"""
TAME Cortex: Agential Swarm Node

This is the main API server for the TAME (Technological Approach to
Mind Everywhere) architecture. It implements:

1. Mixture of Bidders (MoB) - Economic auction-based expert routing
2. Cognitive Homeostasis - Activation steering for goal persistence
3. Agential generation - Goal-directed, drift-resistant inference

Based on Michael Levin's TAME framework for biological cognition.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import torch
import json
import asyncio
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import logging

from mob import MoBConfig, MixtureOfBidders, apply_mob_to_model
from steering import (
    SteeringConfig, 
    CognitiveHomeostat, 
    create_default_steering_vectors,
    SteeringVector
)

# --- CONFIGURATION ---
# Using Mistral-7B-Instruct - well-supported and efficient
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# =============================================================================
# MoB Configuration (Module 1: Agential Swarm)
# =============================================================================
# 
# MEMORY-PERFORMANCE TRADEOFFS FOR MIXTURE OF BIDDERS
# 
# This configuration uses "Balanced" profile optimized for 16GB GPU:
#   - 4 experts × 16 layers × rank 64 ≈ 14.5GB total
#   - ~1.5GB headroom for activations during inference
#
# Parameter Impact Analysis:
# ┌───────────────┬─────────────────────────────┬─────────────────┬───────────────────┐
# │ Parameter     │ Impact on Performance       │ Memory Cost     │ Diminishing Returns│
# ├───────────────┼─────────────────────────────┼─────────────────┼───────────────────┤
# │ Experts       │ More specialization diversity│ ~7MB/expert/layer│ After 4-8 experts │
# │ Layers        │ Depth of reasoning coverage │ ~28MB/layer     │ Early/late layers │
# │ Adapter Rank  │ Expert expressiveness       │ ~0.1MB/rank/exp │ After rank 64-128 │
# └───────────────┴─────────────────────────────┴─────────────────┴───────────────────┘
#
# Expert Count Rationale:
#   - 4 experts with top_k=2 captures 80-90% benefit of 8 experts
#   - VCG auction needs ≥3 experts for meaningful competition
#   - (winner + runner-up + losers creates proper incentive dynamics)
#
# Layer Selection Rationale (Mistral-7B has 32 layers):
#   - Layers 0-7:   Syntax, tokenization, basic features → DON'T MODIFY
#   - Layers 8-23:  Abstract reasoning, knowledge retrieval → PRIORITY ZONE ✓
#   - Layers 24-31: Output formatting, generation → DON'T MODIFY
#
# Adapter Rank Rationale:
#   - LoRA research shows rank 64 captures most adaptation capacity
#   - Rank 128+ is overkill for most tasks with minimal quality gain
#
# Alternative Profiles (adjust based on available GPU memory):
# ┌──────────────┬─────────┬────────┬──────┬─────────┬─────────────┐
# │ Profile      │ Experts │ Layers │ Rank │ Memory  │ Performance │
# ├──────────────┼─────────┼────────┼──────┼─────────┼─────────────┤
# │ Conservative │ 4       │ 12     │ 64   │ ~14.3GB │ Good        │
# │ Balanced ✓   │ 4       │ 16     │ 64   │ ~14.5GB │ Better      │
# │ Aggressive   │ 6       │ 20     │ 64   │ ~14.9GB │ Best        │
# │ Max Headroom │ 4       │ 20     │ 32   │ ~14.3GB │ Good+       │
# └──────────────┴─────────┴────────┴──────┴─────────┴─────────────┘
#
# =============================================================================
MOB_CONFIG = MoBConfig(
    num_experts=4,          # 4 experts: minimum for meaningful VCG auction dynamics
    top_k=2,                # 2 experts per token (sparse routing)
    hidden_dim=4096,        # Mistral hidden dimension
    intermediate_dim=14336, # Mistral FFN intermediate
    initial_wealth=100.0,   # Starting credits for each expert
    wealth_decay=0.99,      # Prevents runaway wealth accumulation
    jitter_std=0.01,        # Symmetry breaking noise
    # Memory-efficient mode: shared base FFN + lightweight LoRA-style adapters
    use_shared_base=True,   # Reduces memory from O(experts×FFN) to O(FFN + experts×adapters)
    adapter_rank=64,        # Sweet spot: captures most adaptation capacity
    adapter_alpha=16.0      # Adapter output scaling factor
)

# Steering Configuration (Module 2: Cognitive Homeostasis)
STEERING_CONFIG = SteeringConfig(
    steering_layers=list(range(10, 22)),  # Middle layers
    base_strength=0.3,
    adaptive=True,
    target_alignment=0.7,
    kp=0.5
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- GLOBAL STATE ---
model = None
tokenizer = None
homeostat = None
mob_stats_history = []


def start_mob_tracking():
    """Enable wealth tracking on all MoB layers for VCG auction analysis."""
    global model
    if model is None:
        return
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, MixtureOfBidders):
            layer.mlp.start_tracking()


def stop_mob_tracking():
    """Disable wealth tracking on all MoB layers."""
    global model
    if model is None:
        return
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, MixtureOfBidders):
            layer.mlp.stop_tracking()


def get_mob_wealth_traces() -> Dict[str, List[List[float]]]:
    """
    Collect wealth history from all MoB layers.
    
    Returns:
        Dictionary mapping layer index to wealth history.
        Each history is [num_forward_passes, num_experts].
    """
    global model
    traces = {}
    if model is None:
        return traces
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, MixtureOfBidders):
            history = layer.mlp.get_wealth_history()
            if history:
                traces[str(idx)] = history
    return traces


def get_aggregated_wealth_trace() -> Dict[str, Any]:
    """
    Get aggregated wealth trace across all MoB layers.
    
    Returns averaged wealth per step across all layers for simpler visualization.
    """
    traces = get_mob_wealth_traces()
    if not traces:
        return {"steps": [], "expert_wealth": []}
    
    # Find max length and number of experts
    all_histories = list(traces.values())
    if not all_histories:
        return {"steps": [], "expert_wealth": []}
    
    num_experts = len(all_histories[0][0]) if all_histories[0] else 0
    max_steps = max(len(h) for h in all_histories)
    
    # Aggregate wealth across layers
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
        "num_layers": len(traces)
    }


def initialize_tame_architecture():
    """
    Initialize the full TAME architecture:
    1. Load base model (Mistral-7B)
    2. Apply Mixture of Bidders transformation
    3. Extract and attach steering vectors
    """
    global model, tokenizer, homeostat
    
    logger.info("=" * 60)
    logger.info("TAME CORTEX: Initializing Agential Architecture")
    logger.info("=" * 60)
    
    # Phase 1: Gestational - Load base model
    logger.info(f"[GESTATIONAL] Loading base model: {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use device_map="auto" for efficient memory usage across GPU/CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Log device distribution
    if hasattr(model, 'hf_device_map'):
        devices_used = set(model.hf_device_map.values())
        logger.info(f"[GESTATIONAL] Model distributed across devices: {devices_used}")
    
    logger.info("[GESTATIONAL] Base model loaded")
    
    # Phase 2: Morphogenesis - Transform to Agential Swarm (MoB)
    logger.info("[MORPHOGENESIS] Applying Mixture of Bidders transformation...")
    
    # Apply MoB to the "reasoning core" layers (8-23)
    # See MOB_CONFIG docstring for full rationale on layer selection
    num_layers = len(model.model.layers)  # Mistral-7B has 32 layers
    layers_to_modify = list(range(8, 24))  # Balanced: 16 layers in reasoning zone
    
    logger.info(f"[MORPHOGENESIS] Targeting {len(layers_to_modify)} layers for MoB transformation")
    model = apply_mob_to_model(model, MOB_CONFIG, layers_to_modify)
    
    logger.info(f"[MORPHOGENESIS] MoB applied to layers {layers_to_modify[0]}-{layers_to_modify[-1]}")
    
    # Phase 3: Homeostatic Calibration - Extract steering vectors
    logger.info("[HOMEOSTASIS] Extracting steering vectors for goal persistence...")
    
    try:
        # Create steering vectors for truthful behavior
        steering_vectors = create_default_steering_vectors(
            model, tokenizer, 
            goal="truthful",
            layers=STEERING_CONFIG.steering_layers
        )
        
        # Initialize and attach homeostat
        homeostat = CognitiveHomeostat(STEERING_CONFIG)
        homeostat.add_steering_vectors(steering_vectors)
        homeostat.attach_to_model(model)
        
        logger.info(f"[HOMEOSTASIS] Steering attached to {len(steering_vectors)} layers")
    except Exception as e:
        logger.warning(f"[HOMEOSTASIS] Steering extraction failed: {e}")
        logger.warning("[HOMEOSTASIS] Continuing without steering (degraded mode)")
        homeostat = None
    
    logger.info("=" * 60)
    logger.info("TAME CORTEX: Online and Self-Regulating")
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI."""
    initialize_tame_architecture()
    yield
    # Cleanup
    if homeostat:
        homeostat.detach_from_model()
    logger.info("TAME Cortex shutting down")


app = FastAPI(
    title="TAME Cortex: Agential Swarm Node",
    description=(
        "A bio-inspired LLM inference server implementing the TAME architecture. "
        "Features Mixture of Bidders for emergent specialization and "
        "Activation Steering for cognitive homeostasis."
    ),
    version="0.1.0",
    lifespan=lifespan
)


# --- REQUEST/RESPONSE MODELS ---

class GenerateRequest(BaseModel):
    """Request for agential generation."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # TAME-specific parameters
    steering_strength: Optional[float] = Field(
        default=None, 
        description="Override steering strength (0.0-1.5). None = adaptive."
    )
    goal: Optional[str] = Field(
        default="truthful",
        description="Behavioral goal: truthful, reasoning, safe"
    )
    return_stats: bool = Field(
        default=False,
        description="Include MoB routing statistics in response"
    )


class GenerateResponse(BaseModel):
    """Response from agential generation."""
    response: str
    usage: Dict[str, int]
    homeostasis: Optional[Dict[str, float]] = None
    mob_stats: Optional[Dict[str, Any]] = None


class SwarmStatus(BaseModel):
    """Status of the expert swarm."""
    num_experts: int
    expert_wealth: List[float]
    expert_usage: List[float]
    layers_modified: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu: str
    model_id: str
    architecture: str
    mob_active: bool
    steering_active: bool


# --- API ENDPOINTS ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check system health and architecture status.
    """
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except:
        gpu_name = "Unknown"
        
    return HealthResponse(
        status="alive",
        gpu=gpu_name,
        model_id=MODEL_ID,
        architecture="TAME (Mixture of Bidders + Cognitive Homeostasis)",
        mob_active=True,
        steering_active=homeostat is not None
    )


@app.get("/swarm/status", response_model=SwarmStatus)
def get_swarm_status():
    """
    Get current status of the expert swarm economy.
    
    Returns wealth distribution and usage statistics for each expert,
    showing emergent specialization patterns.
    """
    # Collect stats from all MoB layers
    total_wealth = torch.zeros(MOB_CONFIG.num_experts)
    total_usage = torch.zeros(MOB_CONFIG.num_experts)
    num_mob_layers = 0
    
    for layer in model.model.layers:
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, MixtureOfBidders):
            mob = layer.mlp
            total_wealth += mob.expert_wealth.cpu()
            total_usage += mob.expert_usage_count.cpu()
            num_mob_layers += 1
            
    # Average across layers
    if num_mob_layers > 0:
        avg_wealth = (total_wealth / num_mob_layers).tolist()
        avg_usage = total_usage.tolist()
    else:
        avg_wealth = [0.0] * MOB_CONFIG.num_experts
        avg_usage = [0.0] * MOB_CONFIG.num_experts
        
    return SwarmStatus(
        num_experts=MOB_CONFIG.num_experts,
        expert_wealth=avg_wealth,
        expert_usage=avg_usage,
        layers_modified=num_mob_layers
    )


@app.get("/homeostasis/status")
def get_homeostasis_status():
    """
    Get current homeostatic alignment status.
    
    Shows how well the model is maintaining its goal-directed behavior.
    """
    if homeostat is None:
        return {"status": "disabled", "message": "Steering not active"}
        
    stats = homeostat.get_alignment_stats()
    return {
        "status": "active",
        "config": {
            "base_strength": STEERING_CONFIG.base_strength,
            "adaptive": STEERING_CONFIG.adaptive,
            "target_alignment": STEERING_CONFIG.target_alignment
        },
        "current_stats": stats
    }


@app.get("/traces/wealth")
def get_wealth_traces():
    """
    Get VCG auction wealth traces for visualization.
    
    Returns aggregated wealth distribution over forward passes.
    Good for analyzing: Is the auction forcing specialization?
    (Inequality = good, equal wealth = not specializing)
    """
    return get_aggregated_wealth_trace()


@app.get("/traces/steering")
def get_steering_traces():
    """
    Get homeostatic steering traces for visualization.
    
    Returns alignment and steering strength (α_t) history.
    Good for analyzing: Is steering adaptive?
    (Should spike when model drifts from goal)
    """
    if homeostat is None:
        return {"status": "disabled", "alignment_history": [], "strength_history": []}
    
    stats = homeostat.get_alignment_stats()
    return {
        "status": "active",
        "alignment_history": stats.get("alignment_history", []),
        "strength_history": stats.get("strength_history", []),
        "target_alignment": STEERING_CONFIG.target_alignment,
        "base_strength": STEERING_CONFIG.base_strength,
        "mean_alignment": stats.get("mean_alignment"),
        "mean_strength": stats.get("mean_strength"),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Agential generation endpoint.
    
    This is the primary inference endpoint. Unlike standard LLM generation,
    this uses:
    - MoB routing: Tokens are auctioned to specialized experts
    - Steering: Hidden states are pushed toward goal-aligned directions
    - Adaptive control: Steering strength adjusts based on drift
    
    The result is more goal-persistent, less drift-prone generation.
    """
    try:
        # Reset homeostatic tracking for this generation
        if homeostat:
            homeostat.reset()
            
            # Override steering strength if specified
            if req.steering_strength is not None:
                homeostat.config.base_strength = req.steering_strength
                homeostat.config.adaptive = False
        
        # 1. Tokenize with chat template
        messages = [{"role": "user", "content": req.prompt}]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 2. Generate (The Agential Thought Process)
        # During generation, MoB routes tokens through expert auctions
        # and steering maintains goal alignment
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 3. Decode
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 4. Collect statistics
        usage = {
            "input_tokens": inputs.input_ids.shape[1],
            "output_tokens": len(generated_ids)
        }
        
        # Get homeostasis stats if available
        homeostasis_stats = None
        if homeostat:
            homeostasis_stats = homeostat.get_alignment_stats()
            
        # Get MoB stats if requested
        mob_stats = None
        if req.return_stats:
            swarm_status = get_swarm_status()
            mob_stats = {
                "expert_wealth": swarm_status.expert_wealth,
                "expert_usage": swarm_status.expert_usage
            }
        
        return GenerateResponse(
            response=response_text,
            usage=usage,
            homeostasis=homeostasis_stats,
            mob_stats=mob_stats
        )

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STREAMING GENERATION (with token-by-token feedback)
# =============================================================================

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """
    Streaming generation endpoint.
    
    Returns Server-Sent Events (SSE) with:
    - Token-by-token output
    - Periodic status updates (expert routing, alignment)
    - Final statistics
    """
    
    async def event_generator():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Preparing generation...'})}\n\n"
            
            # Reset homeostatic tracking
            if homeostat:
                homeostat.reset()
                if req.steering_strength is not None:
                    homeostat.config.base_strength = req.steering_strength
                    homeostat.config.adaptive = False
            
            # Enable MoB wealth tracking for VCG auction visualization
            start_mob_tracking()
            
            # Tokenize
            messages = [{"role": "user", "content": req.prompt}]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {input_length} input tokens...'})}\n\n"
            
            # Create streamer using transformers' built-in TextIteratorStreamer
            streamer = TextIteratorStreamer(
                tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True,
                timeout=2400  # 40 minute timeout
            )
            
            # Generation kwargs
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer
            )
            
            # Run generation in background thread
            def generate_in_thread():
                try:
                    with torch.inference_mode():
                        model.generate(**generation_kwargs)
                except Exception as e:
                    logger.error(f"Generation thread error: {e}")
            
            thread = Thread(target=generate_in_thread)
            thread.start()
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            
            # Stream tokens as they arrive
            full_response = ""
            token_count = 0
            last_status_update = 0
            
            for token_text in streamer:
                full_response += token_text
                token_count += 1
                
                # Send token
                yield f"data: {json.dumps({'type': 'token', 'content': token_text})}\n\n"
                
                # Every 10 tokens, send a status update with routing info
                if token_count - last_status_update >= 10:
                    last_status_update = token_count
                    status_msg = f"Generated {token_count} tokens"
                    
                    # Add MoB info if available
                    try:
                        for layer in model.model.layers:
                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'last_stats'):
                                stats = layer.mlp.last_stats
                                if stats and 'expert_wealth' in stats:
                                    wealth = stats['expert_wealth']
                                    top_expert = wealth.argmax().item()
                                    status_msg += f" | Expert {top_expert} leading"
                                    break
                    except:
                        pass
                    
                    yield f"data: {json.dumps({'type': 'progress', 'message': status_msg, 'tokens': token_count})}\n\n"
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Wait for thread to complete
            thread.join(timeout=10)
            
            # Stop MoB tracking
            stop_mob_tracking()
            
            # Send final stats
            final_stats = {
                'type': 'complete',
                'usage': {
                    'input_tokens': input_length,
                    'output_tokens': token_count
                }
            }
            
            # Add homeostasis stats with traces
            if homeostat:
                homeo_stats = homeostat.get_alignment_stats()
                final_stats['homeostasis'] = {
                    'mean_alignment': homeo_stats.get('mean_alignment'),
                    'current_strength': homeo_stats.get('current_strength'),
                    'mean_strength': homeo_stats.get('mean_strength'),
                }
                # Include full traces for visualization
                final_stats['steering_trace'] = {
                    'alignment_history': homeo_stats.get('alignment_history', []),
                    'strength_history': homeo_stats.get('strength_history', []),
                    'target_alignment': STEERING_CONFIG.target_alignment,
                }
            
            # Add MoB stats with wealth trace
            if req.return_stats:
                try:
                    swarm_status = get_swarm_status()
                    wealth_trace = get_aggregated_wealth_trace()
                    final_stats['mob_stats'] = {
                        'expert_wealth': swarm_status.expert_wealth,
                        'expert_usage': swarm_status.expert_usage
                    }
                    final_stats['wealth_trace'] = wealth_trace
                except:
                    pass
            
            yield f"data: {json.dumps(final_stats)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            stop_mob_tracking()  # Ensure tracking is stopped on error
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/steering/update")
async def update_steering(goal: str = "truthful", strength: float = 0.3):
    """
    Update the active steering vector.
    
    Allows runtime modification of the agent's homeostatic goals.
    """
    global homeostat
    
    if homeostat is None:
        raise HTTPException(
            status_code=400, 
            detail="Steering not initialized"
        )
    
    try:
        # Remove old hooks
        homeostat.detach_from_model()
        
        # Create new steering vectors for the requested goal
        steering_vectors = create_default_steering_vectors(
            model, tokenizer,
            goal=goal,
            layers=STEERING_CONFIG.steering_layers
        )
        
        # Update config
        homeostat.config.base_strength = strength
        
        # Attach new vectors
        homeostat.add_steering_vectors(steering_vectors)
        homeostat.attach_to_model(model)
        
        return {
            "status": "updated",
            "goal": goal,
            "strength": strength,
            "layers": STEERING_CONFIG.steering_layers
        }
        
    except Exception as e:
        logger.error(f"Steering update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))