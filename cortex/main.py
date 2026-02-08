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

# =============================================================================
# MODEL PROFILES - Change ACTIVE_MODEL to switch between models
# =============================================================================
# 
# To switch models, change ACTIVE_MODEL below. All dimensions and layer ranges
# are automatically configured. Keep synchronized with train.py!
#
# Available profiles:
# ┌───────────────┬─────────┬──────────────┬────────────┬───────────────────────────┐
# │ Profile       │ Params  │ Train Speed  │ Quality    │ Access                    │
# ├───────────────┼─────────┼──────────────┼────────────┼───────────────────────────┤
# │ gemma-2-2b ✓  │ 2B      │ ~3.5x faster │ Medium     │ Open (no approval needed) │
# │ llama-3.2-3b  │ 3B      │ ~2.5x faster │ Good       │ Requires Meta approval    │
# │ mistral-7b    │ 7B      │ 1x (baseline)│ Best       │ Open                      │
# └───────────────┴─────────┴──────────────┴────────────┴───────────────────────────┘

MODEL_PROFILES = {
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b-it",
        "hidden_dim": 2304,
        "intermediate_dim": 9216,
        "num_layers": 26,
        "mob_layers_start": 5,   # Skip early syntax layers
        "mob_layers_end": 18,    # Skip late output layers
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_dim": 3072,
        "intermediate_dim": 8192,
        "num_layers": 28,
        "mob_layers_start": 6,
        "mob_layers_end": 20,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_layers": 32,
        "mob_layers_start": 8,
        "mob_layers_end": 24,
    },
}

# =============================================================================
# >>> CHANGE THIS TO SWITCH MODELS <<<
# =============================================================================
ACTIVE_MODEL = "gemma-2-2b"  # Options: "gemma-2-2b", "llama-3.2-3b", "mistral-7b"
# =============================================================================

# Get active profile
_profile = MODEL_PROFILES[ACTIVE_MODEL]
MODEL_ID = _profile["model_id"]

# =============================================================================
# MoB Configuration (Module 1: Agential Swarm)
# =============================================================================
# 
# IMPORTANT: Keep ACTIVE_MODEL synchronized with train.py!
# If you train with different model, the mob_state.pt will not load correctly.
#
# TUNING GUIDE (applies to all models):
# ┌───────────────┬─────────────────────────────────────┬───────────────────────────┐
# │ Parameter     │ What it controls                    │ Tuning advice             │
# ├───────────────┼─────────────────────────────────────┼───────────────────────────┤
# │ num_experts   │ Specialization diversity            │ 4-8 optimal, >8 diminishes│
# │ top_k         │ Experts per token (sparsity)        │ 2 is sweet spot           │
# │ adapter_rank  │ Expert expressiveness (LoRA rank)   │ 32-64 sufficient          │
# │ wealth_decay  │ How fast losers decay               │ 0.999=slow, 0.99=fast     │
# │ min_wealth    │ Floor prevents expert death         │ 10.0 safe default         │
# │ max_wealth    │ Cap prevents monopoly               │ 500.0 balanced            │
# │ jitter_std    │ Symmetry breaking noise             │ 0.05 for differentiation  │
# └───────────────┴─────────────────────────────────────┴───────────────────────────┘
#
# LAYER SELECTION (auto-configured per model):
#   - Early layers (0-20%):  Tokenization, syntax → DON'T MODIFY
#   - Middle layers (20-70%): Reasoning, knowledge → PRIORITY ZONE ✓
#   - Late layers (70-100%): Output format → DON'T MODIFY
#
# Memory estimate: ~3MB per expert per layer at rank 64
#
# =============================================================================
MOB_CONFIG = MoBConfig(
    num_experts=4,          # 4 experts: minimum for meaningful VCG auction dynamics
    top_k=2,                # 2 experts per token (sparse routing)
    hidden_dim=_profile["hidden_dim"],
    intermediate_dim=_profile["intermediate_dim"],
    initial_wealth=100.0,   # Starting credits for each expert
    wealth_decay=0.999,     # Mild decay for specialization (updated from 0.99)
    min_wealth=10.0,        # Minimum wealth (prevents death spiral)
    max_wealth=500.0,       # Maximum wealth (prevents monopoly)
    jitter_std=0.05,        # Symmetry breaking noise (increased for differentiation)
    reward_scale=1.0,       # Base reward multiplier
    use_vcg_payments=True,  # Enable VCG payment mechanism
    # Memory-efficient mode: shared base FFN + lightweight LoRA-style adapters
    use_shared_base=True,   # Reduces memory from O(experts×FFN) to O(FFN + experts×adapters)
    adapter_rank=32,        # Reduced from 64 for 16GB GPUs (32 still effective)
    adapter_alpha=16.0,     # Adapter output scaling factor
    # Specialization mechanisms (inference mode)
    use_loss_feedback=False,  # Disabled for inference (no training loop)
    use_local_quality=True,   # Use output quality signals for wealth updates
    use_differentiable_routing=False,  # Not needed for inference
    # Inference dynamics: more responsive wealth changes for visible VCG auction
    inference_wealth_decay=0.99,         # Faster decay (vs 0.999 training) - 1% per token
    inference_exploration_bonus=0.02,    # 2% bonus for underused experts
    inference_wealth_compression=0.5,    # Compress 50% toward mean on load
)

# Steering Configuration (Module 2: Cognitive Homeostasis)
# Steering layers align with MoB layers (reasoning core)
STEERING_CONFIG = SteeringConfig(
    steering_layers=list(range(_profile["mob_layers_start"], _profile["mob_layers_end"])),
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
    
    # Apply MoB to the "reasoning core" layers
    # Layer range is determined by the active model profile
    num_layers = len(model.model.layers)
    layers_to_modify = list(range(_profile["mob_layers_start"], _profile["mob_layers_end"]))
    
    logger.info(f"[MORPHOGENESIS] Targeting {len(layers_to_modify)} layers for MoB transformation")
    model = apply_mob_to_model(model, MOB_CONFIG, layers_to_modify)
    
    logger.info(f"[MORPHOGENESIS] MoB applied to layers {layers_to_modify[0]}-{layers_to_modify[-1]}")
    
    # Set model to eval mode - critical for MoB to use inference path
    model.eval()
    
    # Diagnostic: Verify MoB is producing valid output
    logger.info("[DIAGNOSTIC] Testing MoB output validity...")
    try:
        test_input = tokenizer("Test", return_tensors="pt").to(model.device)
        with torch.inference_mode():
            test_output = model(**test_input, output_hidden_states=True)
            # Check for NaN/Inf in hidden states
            last_hidden = test_output.hidden_states[-1]
            has_nan = torch.isnan(last_hidden).any().item()
            has_inf = torch.isinf(last_hidden).any().item()
            mean_val = last_hidden.abs().mean().item()
            std_val = last_hidden.std().item()
            logger.info(f"[DIAGNOSTIC] Hidden states: mean_abs={mean_val:.4f}, std={std_val:.4f}, has_nan={has_nan}, has_inf={has_inf}")
            if has_nan or has_inf:
                logger.error("[DIAGNOSTIC] WARNING: Model producing NaN/Inf! Check MoB configuration.")
            elif mean_val < 0.01 or std_val < 0.01:
                logger.warning("[DIAGNOSTIC] WARNING: Hidden states may be collapsed (very low variance)")
            else:
                logger.info("[DIAGNOSTIC] MoB output looks valid")
    except Exception as e:
        logger.warning(f"[DIAGNOSTIC] Test failed: {e}")
    
    # Phase 2b: Load trained MoB state if available
    # This restores expert specialization from training (wealth, performance EMA)
    # We use compress_wealth to reduce inequality for more dynamic inference
    mob_state_paths = [
        "./tame_inference/mob_state.pt",  # Exported from training
        "./mob_state.pt",                  # Local file
    ]
    
    from mob import load_mob_state
    # Use the inference_wealth_compression setting from config
    compression = MOB_CONFIG.inference_wealth_compression
    for state_path in mob_state_paths:
        import os
        if os.path.exists(state_path):
            try:
                loaded = load_mob_state(model, state_path, compress_wealth=compression)
                if loaded > 0:
                    logger.info(f"[MORPHOGENESIS] Restored trained expert specialization from {state_path}")
                break
            except Exception as e:
                logger.warning(f"[MORPHOGENESIS] Failed to load mob_state from {state_path}: {e}")
    else:
        logger.info("[MORPHOGENESIS] No trained mob_state found - experts start with default wealth")
    
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
    max_tokens: int = Field(default=512, ge=1, le=4096)
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
            last_trace_update = 0
            
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
                
                # Every 25 tokens, send intermediate trace data for live chart updates
                if token_count - last_trace_update >= 25:
                    last_trace_update = token_count
                    try:
                        trace_update = {'type': 'trace_update', 'tokens': token_count}
                        
                        # Get current wealth trace (subsampled for performance)
                        wealth_data = get_aggregated_wealth_trace()
                        if wealth_data.get('expert_wealth'):
                            # Subsample to max 100 points for performance
                            expert_wealth = wealth_data['expert_wealth']
                            if len(expert_wealth) > 100:
                                step = len(expert_wealth) // 100
                                expert_wealth = expert_wealth[::step]
                            trace_update['wealth_trace'] = {
                                'steps': list(range(len(expert_wealth))),
                                'expert_wealth': expert_wealth,
                                'num_experts': wealth_data.get('num_experts', 0),
                                'num_layers': wealth_data.get('num_layers', 0)
                            }
                        
                        # Get current steering trace (subsampled)
                        if homeostat:
                            homeo_stats = homeostat.get_alignment_stats()
                            strength_history = homeo_stats.get('strength_history', [])
                            alignment_history = homeo_stats.get('alignment_history', [])
                            
                            # Subsample to max 100 points
                            if len(strength_history) > 100:
                                step = len(strength_history) // 100
                                strength_history = strength_history[::step]
                                alignment_history = alignment_history[::step] if alignment_history else []
                            
                            if strength_history:
                                trace_update['steering_trace'] = {
                                    'strength_history': strength_history,
                                    'alignment_history': alignment_history,
                                    'target_alignment': STEERING_CONFIG.target_alignment,
                                }
                        
                        yield f"data: {json.dumps(trace_update)}\n\n"
                    except Exception as e:
                        logger.warning(f"Error sending trace update: {e}")
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