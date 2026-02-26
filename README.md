<p align="center">
  <h1 align="center">TAME-Swarm</h1>
  <p align="center">
    <strong>Bio-Inspired Multi-Scale Competency Architecture for LLMs</strong>
  </p>
  <p align="center">
    Transforming monolithic language models into homeostatic agential swarms
    <br/>
    grounded in Michael Levin's <em>Technological Approach to Mind Everywhere</em> (TAME) framework
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> · <a href="#quickstart">Quickstart</a> · <a href="#training">Training</a> · <a href="#development">Development</a> · <a href="#api-reference">API</a>
  </p>
</p>

---

## Why This Exists

Contemporary LLMs are **monolithic next-token predictors** — sophisticated but fundamentally passive. They lack:

| Problem | Biological Analogue |
|---------|-------------------|
| No internal modularity — improving one skill degrades another | Organs specialise; the liver doesn't interfere with the brain |
| No persistent goals — behaviour drifts with context length | Homeostasis keeps body temperature at 37 °C regardless of weather |
| No grounding — "hallucinations" about the physical world are unchecked | Organisms are embodied; physics constrains morphogenesis |

TAME-Swarm addresses the first two by re-architecting an LLM's internals using two bio-inspired modules that run **today** on consumer GPUs.

---

## Architecture

<a name="architecture"></a>

```
                         ┌─────────────────────────────────────────────┐
  User Prompt ──────────►│           TAME-Swarm Agent                  │
                         │                                             │
                         │  ┌───────────────────────────────────────┐  │
                         │  │  Module 1 · Mixture of Bidders (MoB)  │  │
                         │  │                                       │  │
                         │  │  Token ─► Expert Bids ─► VCG Auction  │  │
                         │  │       ─► Top-k Routing ─► Sparse FFN  │  │
                         │  └───────────────────────────────────────┘  │
                         │         ▲                                   │
                         │         │ adaptive α(t)                     │
                         │  ┌──────┴────────────────────────────────┐  │
                         │  │  Module 2 · Cognitive Homeostasis     │  │
                         │  │                                       │  │
                         │  │  Steering Vector injection at each    │  │
                         │  │  layer with PID drift correction      │  │
                         │  └───────────────────────────────────────┘  │
                         └──────────────────────────────┬──────────────┘
                                                        │
                                                        ▼
                                                   Response
```

### Module 1 — Mixture of Bidders (MoB): *The Body*

Standard Mixture-of-Experts uses a learned router — a centralised command economy. MoB replaces it with a **VCG (Vickrey-Clarke-Groves) second-price auction**: each expert maintains a *wallet* of credits, bids `confidence × wealth` for every token, and only the top-k winners are activated.

**Why it matters:**

- **Truthful bidding** — the VCG mechanism mathematically incentivises experts to bid their true value.
- **Emergent specialisation** — experts that reduce loss earn more credits, reinforcing their niche (code, reasoning, grammar…).
- **No router collapse** — the decentralised market avoids the single-point-of-failure of a learned gating network.
- **Memory-efficient** — shared base weights + LoRA-rank adapters keep VRAM overhead to ~3 MB per expert per layer.

### Module 2 — Cognitive Homeostasis: *The Mind*

Activation **Steering Vectors** encode goals (truthfulness, safety, reasoning) as linear directions in the model's hidden space. A proportional controller injects these vectors at every layer, dynamically adjusting strength based on how far the model's activations have drifted from the target:

$$\alpha(t) = k_p \cdot (\text{target\_alignment} - \cos(h_t,\; v_{\text{steer}}))$$

- Zero context-window cost (no system-prompt tokens consumed)
- Resilient to jailbreaks — operates on the *latent space*, not text
- Orthogonal projection prevents capability damage to the base model

### Planned Modules

| Module | Purpose | Status |
|--------|---------|--------|
| **Recurrent Memory (RMT)** | Persistent "bioelectric" state across segments — infinite context | Planned |
| **Physicome (WorldCoder)** | Physics-engine grounding to eliminate physical hallucinations | Planned |

---

## Quickstart

<a name="quickstart"></a>

### Prerequisites

- **Python 3.10+**
- **CUDA 12.x** with a GPU that has ≥ 16 GB VRAM (RTX 4090, A100, etc.)
- **Docker** (optional, recommended for reproducibility)

### Option A — Docker (Recommended)

```bash
cd tame
docker build -t tame-swarm .
docker run --gpus all -p 8000:8000 tame-swarm
```

### Option B — Local

```bash
cd tame
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The first run downloads the base model (~5 GB for Gemma-2-2B). Subsequent runs use the local cache.

### Verify

```bash
# Health check
curl http://localhost:8000/health

# Generate with homeostatic steering
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement", "max_tokens": 200}'

# Inspect swarm economy
curl http://localhost:8000/swarm/status
```

---

## Training

<a name="training"></a>

Training develops expert specialisation within the MoB layers. Without it, all experts start with identical weights and a Gini coefficient of ≈ 0 (no differentiation).

### Supported Base Models

| Profile | Params | Train Speed | Quality | Access |
|---------|--------|-------------|---------|--------|
| `gemma-2-2b` | 2 B | ~3.5× faster | Medium | Open |
| `llama-3.2-3b` | 3 B | ~2.5× faster | Good | Requires Meta approval |
| `mistral-7b` | 7 B | 1× (baseline) | Best | Open |

Switch models by changing `ACTIVE_MODEL` in both [tame/main.py](tame/main.py) and [tame/train.py](tame/train.py).

### Quick Test (verify setup)

```powershell
# PowerShell
cd tame
.\train.ps1 test

# CMD
train.bat test

# Direct
python setup_tame.py --mode test --steps 100
```

### Full Training

```powershell
# 5 000 steps (~2-4 h on A100, ~6-8 h on RTX 4090)
.\train.ps1 train

# Custom step count
.\train.ps1 train 10000

# Memory-constrained (< 24 GB VRAM) — add LoRA
.\train.ps1 train -UseLora
```

### What Happens During Training

| Phase | Description |
|-------|-------------|
| **Wealth Updates** | Experts that reduce loss gain credits; poor performers lose them |
| **VCG Auction Routing** | Wealth differentials cause tokens to be routed to the most competent expert |
| **Confidence Calibration** | Each expert's confidence head learns to predict its actual contribution |
| **Checkpoint Persistence** | `mob_state.pt` saves the full economic state for later inference |

### Training Outputs

```
tame_checkpoints/
├── checkpoint-1000/
│   ├── model.safetensors     # Model weights
│   ├── mob_state.pt          # Expert wealth & auction state
│   └── training_state.pt     # Optimizer state (for resume)
└── checkpoint-5000/
    └── ...

tame_inference/               # Automatically exported for the API server
├── mob_state.pt
├── inference_config.json
└── loader_snippet.py
```

### VRAM Requirements

| Mode | VRAM | Notes |
|------|------|-------|
| Inference | ~8–16 GB | bfloat16, forward pass only |
| Training (full) | ~24–32 GB | Gradients + optimizer states |
| Training (LoRA) | ~16–20 GB | Only adapter gradients |

---

## Development

<a name="development"></a>

### Project Structure

```
tame-swarm/
├── README.md               ← You are here
└── tame/                   ← Core implementation
    ├── main.py             ← FastAPI inference server + API endpoints
    ├── mob.py              ← Mixture of Bidders: VCG auction, experts, wealth
    ├── steering.py         ← Cognitive Homeostasis: steering vectors, PID control
    ├── train.py            ← Training loop with MoB economic dynamics
    ├── setup_tame.py       ← End-to-end train → export workflow
    ├── chat_ui.py          ← Gradio chat interface with live wealth visualisation
    ├── requirements.txt    ← Python dependencies
    ├── Dockerfile          ← Production container (CUDA 12.6)
    ├── Dockerfile.chat     ← Lightweight chat UI container
    ├── dev.ps1 / dev.bat   ← One-click dev server (hot-reload)
    └── train.ps1 / train.bat ← One-click training scripts
```

### Dev Server (Hot Reload)

For the fastest iteration loop — file saves trigger automatic server restart:

```bash
cd tame
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```powershell
cd tame
.\dev.ps1
```

### Chat UI

A Gradio interface ships with live VCG auction visualisations — watch expert wealth diverge in real time:

```bash
# Start the API server first, then:
cd tame
python chat_ui.py
# Open http://localhost:7860
```

### Key Concepts for Contributors

| Concept | File | What to Know |
|---------|------|--------------|
| **VCG Auction** | `mob.py` | Second-price auction guarantees truthful bidding; `ConfidenceHead` predicts expert value |
| **Wealth Economy** | `mob.py` | `expert_wealth` buffers persist across batches; `wealth_decay` and `reward_scale` control dynamics |
| **Steering Vectors** | `steering.py` | Extracted via difference-in-means on contrastive prompt pairs; injected as residual-stream additions |
| **Adaptive Control** | `steering.py` | P-controller with `kp`, `target_alignment`, and `max_strength` — tunes itself at each token |
| **Model Profiles** | `main.py` | `MODEL_PROFILES` dict maps model names to dimensions; change `ACTIVE_MODEL` to switch |

### Configuration

All tuneable parameters are documented in-line. The most impactful knobs:

```python
# tame/main.py

MOB_CONFIG = MoBConfig(
    num_experts=4,           # 4–8 for meaningful auction dynamics
    top_k=2,                 # Experts activated per token
    initial_wealth=75.0,     # Starting credits
    wealth_decay=0.997,      # Decay rate per step
    reward_scale=2.0,        # How strongly loss reduction is rewarded
    adapter_rank=32,         # LoRA rank per expert (memory vs expressiveness)
)

STEERING_CONFIG = SteeringConfig(
    base_strength=0.3,       # Injection coefficient α
    adaptive=True,           # Enable PID drift correction
    target_alignment=0.7,    # Cosine-similarity setpoint
)
```

---

## API Reference

<a name="api-reference"></a>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check with architecture status |
| `/generate` | POST | Agential text generation (supports streaming) |
| `/swarm/status` | GET | Expert wealth distribution & specialisation metrics |
| `/homeostasis/status` | GET | Current steering alignment & adaptive strength |
| `/steering/update` | POST | Runtime modification of steering goals |

### Example: Generate with steering

```json
POST /generate
{
  "prompt": "Explain the second law of thermodynamics",
  "max_tokens": 300,
  "temperature": 0.7,
  "goal": "reasoning",
  "return_stats": true
}
```

The response includes MoB routing statistics showing which experts were activated and their wealth changes — useful for debugging specialisation.

---

## Theoretical Foundation

This project implements ideas from the following research areas:

- **TAME Framework** — Michael Levin's theory that intelligence is an emergent property of competent sub-agents cooperating under homeostatic pressure, not a monolithic central process.
- **Active Inference / Free Energy Principle** — Steering vectors approximate active inference by maintaining a "preferred state" in activation space; the adaptive controller minimises drift from this setpoint.
- **Activation Engineering** — Steering vectors discovered via contrastive activation analysis provide zero-cost behavioural control in latent space.
- **Sparse Mixture of Experts** — Token-level routing enables efficient scaling; TAME-Swarm extends this with decentralised economic allocation.

### From Biology to Code

| Biological Principle | TAME-Swarm Implementation | Status |
|---------------------|---------------------------|--------|
| Multicellular tissue with specialised organs | Expert pool with VCG auction routing | Implemented |
| Homeostatic setpoints (temperature, pH) | Steering vectors as target directions in activation space | Implemented |
| Gap junctions synchronising bioelectric state | Recurrent Memory Transformer (RMT) for persistent internal state | Planned |
| Morphogenetic adaptation under physical constraint | Physics-engine grounding (WorldCoder) | Planned |

---

## Roadmap

- [x] Mixture of Bidders (VCG auction routing)
- [x] Cognitive Homeostasis (adaptive steering vectors)
- [x] Training pipeline with wealth-based specialisation
- [x] Chat UI with live auction visualisation
- [x] Multi-model support (Gemma 2B, Llama 3B, Mistral 7B)
- [ ] Recurrent Memory Transformer (bioelectric persistence)
- [ ] Neuro-symbolic physics grounding (WorldCoder)
- [ ] Contrastive steering vector extraction tooling
- [ ] Benchmark suite (Machiavelli, Needle-in-Haystack, PhyQA)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built as a practical exploration of bio-inspired AI architectures.</sub>
</p>
