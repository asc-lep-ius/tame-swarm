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
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# MoB Configuration (Module 1: Agential Swarm)
MOB_CONFIG = MoBConfig(
    num_experts=8,          # 8 expert FFNs per layer
    top_k=2,                # 2 experts per token (sparse)
    hidden_dim=4096,        # Mistral hidden dimension
    intermediate_dim=14336, # Mistral FFN intermediate
    initial_wealth=100.0,
    wealth_decay=0.99,
    jitter_std=0.01
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
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    logger.info(f"[GESTATIONAL] Base model loaded on {model.device}")
    
    # Phase 2: Morphogenesis - Transform to Agential Swarm (MoB)
    logger.info("[MORPHOGENESIS] Applying Mixture of Bidders transformation...")
    
    # Apply MoB to middle layers (where most reasoning happens)
    num_layers = len(model.model.layers)
    layers_to_modify = list(range(8, num_layers - 4))  # Skip early/late layers
    
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