# TAME Cortex: Agential Swarm Node

A bio-inspired LLM inference server implementing the **TAME (Technological Approach to Mind Everywhere)** architecture based on Michael Levin's framework for biological cognition.

## Architecture Overview

This implementation transforms a standard monolithic LLM into an **Agential Swarm** with persistent goal-directed behavior:

### Module 1: Mixture of Bidders (MoB) - The "Body"

Replaces the standard Feed-Forward Networks with an **economic auction-based routing system**:

- **Expert Pool**: Each FFN is split into 8 specialized experts
- **VCG Auction**: Experts bid on tokens using confidence × wealth
- **Emergent Specialization**: Experts naturally specialize through economic competition
- **Sparse Computation**: Only top-k (default: 2) experts process each token

```
Token → [Expert Bids] → VCG Auction → Top-k Selection → Sparse FFN → Output
```

This mimics biological tissue where cells compete and cooperate, leading to emergent organ differentiation.

### Module 2: Cognitive Homeostasis - The "Mind"

Implements **Active Inference** approximation using **Activation Steering Vectors**:

- **Steering Vectors**: Directions in activation space encoding goals (truthful, reasoning, safe)
- **Adaptive Control**: P-controller adjusts steering strength based on drift
- **Goal Persistence**: Constant "homeostatic force" resists behavioral drift
- **Orthogonal Projection**: Prevents steering from damaging general capabilities

```
h' = h + α(t) × v_steer

where α(t) = k_p × (target_alignment - current_alignment)
```

This provides the "purpose" that drives the agent toward its goals.

## API Endpoints

### `GET /health`
System health check with architecture status.

### `POST /generate`
Agential text generation with:
- `prompt`: Input text
- `max_tokens`: Generation limit
- `temperature`: Sampling temperature
- `steering_strength`: Override adaptive steering (optional)
- `goal`: Behavioral goal (truthful/reasoning/safe)
- `return_stats`: Include MoB routing statistics

### `GET /swarm/status`
Expert swarm economy status:
- Wealth distribution across experts
- Usage statistics showing specialization patterns

### `GET /homeostasis/status`
Homeostatic alignment metrics:
- Current alignment to steering goal
- Adaptive strength values

### `POST /steering/update`
Runtime modification of steering goals.

## Running

### Docker (Recommended)

```bash
docker build -t tame-cortex .
docker run --gpus all -p 8000:8000 tame-cortex
```

### Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration

Key parameters in `main.py`:

```python
# Model selection
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# MoB Configuration
MOB_CONFIG = MoBConfig(
    num_experts=8,          # Expert count per layer
    top_k=2,                # Experts per token
    initial_wealth=100.0,   # Starting credits
    wealth_decay=0.99,      # Prevents runaway accumulation
)

# Steering Configuration  
STEERING_CONFIG = SteeringConfig(
    steering_layers=range(10, 22),  # Middle layers
    base_strength=0.3,              # Injection strength
    adaptive=True,                  # P-controller
    target_alignment=0.7,           # Homeostatic setpoint
)
```

## Project Status vs Architecture Document

| Module | Status | Notes |
|--------|--------|-------|
| **Module 1: MoB** | ✅ Implemented | VCG auction, wealth tracking, upcycling |
| **Module 2: Steering** | ✅ Implemented | Adaptive control, contrastive extraction |
| **Module 3: RMT Memory** | 🔲 Planned | Recurrent memory for bioelectric persistence |
| **Module 4: Physicome** | 🔲 Planned | Physics engine integration |

## References

- Levin, M. - TAME Framework for biological cognition
- Activation Steering / Activation Engineering research
- Mixture of Experts / Sparse Transformers literature
- Active Inference / Free Energy Principle
