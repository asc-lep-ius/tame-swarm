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
    <a href="#architecture">Architecture</a> · <a href="#quickstart">Quickstart</a> · <a href="#training">Training</a> · <a href="#development">Development</a> · <a href="#api-reference">API</a> · <a href="#roadmap">Roadmap</a>
  </p>
</p>

---

## Why This Exists

> "There is no truly monadic, indivisible agent: all minds reside in physical systems made of components of various complexity. The Self is a dynamical construct—a multiscale holobiont where the activities of competent, lower-level agents give rise to something truly more than the sum of its parts."
>
> — Michael Levin, [*Technological Approach to Mind Everywhere (TAME)*](https://arxiv.org/abs/2201.10346) 

### From Monolith to Holobiont

In the TAME framework, intelligence isn't a "thing" you have; it's a collective competency across scales. Traditional LLMs are like a single, giant, frozen cell. TAME-Swarm unfetters this architecture by treating the model as a tissue of sub-agents:

**Mixture of Bidders (MoB)** represents the Evolutionary Economy. It recognizes that "competence without comprehension" is the engine of life. By forcing experts to compete and profit, we replicate the bio-economic pressure that drives cellular specialization. The core novelty is replacing the standard learned MoE router (a central planner) with a **VCG second-price auction** — a mechanism from economic theory that provably incentivises truthful bidding. Each expert accumulates wealth by performing well, creating emergent specialisation without any supervised routing signal.

**Cognitive Homeostasis** represents the Bioelectric Target Pattern. Just as an embryo "knows" to build a face even if the cells are scrambled, our steering vectors act as a "moral and logical pH balance," pulling the swarm back to its goal-state whenever the stochasticity of the auction drifts too far. The controller dynamically adjusts injection strength based on how far the model's latent representation has drifted from the target.

> In this architecture, "alignment" is a homeostatic state the system is physically incapable of leaving for long.

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
                         │  │  layer with P-controller correction   │  │
                         │  └───────────────────────────────────────┘  │
                         └──────────────────────────────┬──────────────┘
                                                        │
                                                        ▼
                                                   Response
```

> **Current state:** The two modules operate independently — MoB routes tokens, Steering corrects the output afterward. [Phase 1](#phase-1--steering-economy-coupling) will couple them so the goal state shapes routing directly.

### Module 1 — Mixture of Bidders (MoB): *The Body*

Standard Mixture-of-Experts uses a learned router — a centralised command economy. MoB replaces it with a **VCG (Vickrey-Clarke-Groves) second-price auction**: each expert maintains a *wallet* of credits, bids `confidence × wealth` for every token, and only the top-k winners are activated.

**Why it matters:**

- **Truthful bidding** — the VCG mechanism (from mechanism design theory) mathematically incentivises experts to bid their true value; no expert can gain by misreporting confidence.
- **Emergent specialisation** — experts that reduce loss earn more credits, reinforcing their niche. The Gini coefficient of the wealth distribution measures specialisation: higher Gini = more differentiation.
- **No router collapse** — the decentralised market avoids the single-point-of-failure of a learned gating network.
- **Memory-efficient** — shared base weights + LoRA-rank adapters keep VRAM overhead to ~3 MB per expert per layer at rank 32.

**Implementation details:**

- **Upcycling, not training from scratch.** MoB layers are initialised by copying the pretrained FFN weights to a shared base. Each expert starts as the identity transform (LoRA B-matrices zeroed) plus small Gaussian jitter to break symmetry. This preserves the original model's behaviour on day zero.
- **Layer selection matters.** Only middle layers (20–70% of model depth) are converted to MoB. Early layers handle tokenisation/syntax and late layers handle output formatting — modifying them degrades base performance.
- **Two forward-pass modes.** Training uses dense computation (all tokens through all experts, masked by routing) for gradient checkpointing compatibility. Inference uses sparse computation (only selected tokens through assigned experts) for speed.
- **Three wealth-update paths** exist today: loss-based feedback (training, primary), local output-quality proxy (inference), and participation-based (fallback). [Phase 2](#phase-2--economy-stabilisation) will unify these into a single parameterised mechanism.

### Module 2 — Cognitive Homeostasis: *The Mind*

Activation **Steering Vectors** encode goals (truthfulness, safety, reasoning) as linear directions in the model's hidden space. A proportional controller injects these vectors at every selected layer, dynamically adjusting strength based on how far the model's activations have drifted from the target:

```math
\alpha(t) = k_p \cdot (\text{target\_alignment} - \cos(h_t,\; v_{\text{steer}}))
```

- **Zero context-window cost** — no system-prompt tokens consumed; steering operates entirely in weight/activation space.
- **Latent-space operation** — acts on the residual stream, not on text tokens. This makes it harder (though not impossible) for prompt-based attacks to circumvent, since the correction bypasses the text channel entirely. Formal adversarial evaluation is planned but not yet complete.
- **Orthogonal projection** prevents capability damage by projecting steering vectors to be orthogonal to the model's general-capability subspace, avoiding the "lobotomy" problem where steering degrades base performance.

**Implementation details:**

- **Contrastive extraction.** Steering vectors are computed via the Difference-in-Means method: run positive and negative prompt sets through the model, capture activations at each target layer, and take $v_{\text{steer}} = \text{mean}(h^+) - \text{mean}(h^-)$. The resulting vector points in the direction of the desired behavioural trait.
- **Current limitation: thin contrastive data.** The default `STEERING_TEMPLATES` use only 4 contrastive pairs per goal. The activation engineering literature (Turner et al., 2023; Rimsky et al., 2024) recommends 50–200 diverse pairs for robust trait directions. With 4 pairs, the vector may capture prompt-surface features rather than the genuine latent trait. [Phase 1b](#phase-1--steering-economy-coupling) addresses this.
- **P-controller only (currently).** The controller is proportional-only — it adjusts strength based on instantaneous alignment error. Under stochastic sampling (temperature > 0), this produces oscillation around the target without convergence. [Phase 1c](#phase-1--steering-economy-coupling) upgrades to a full PID controller with anti-windup.
- **Runtime modifiable.** Steering goals can be changed at runtime via the `/steering/update` endpoint without restarting the server. New vectors are extracted on-the-fly.

### Planned Modules

| Module | Purpose | Phase | Status |
|--------|---------|-------|--------|
| **Steering–Economy Coupling** | Goal state shapes expert routing, not just post-hoc correction | [Phase 1](#phase-1--steering-economy-coupling) | Planned |
| **Economy Stabilisation** | Unified wealth dynamics with formal stability guarantees | [Phase 2](#phase-2--economy-stabilisation) | Planned |
| **Concept-Level Agency** | Chunk-level routing + per-expert memory within forward pass | [Phase 3](#phase-3--concept-level-agency) | Planned |
| **Multi-Scale Hierarchy** | Inter-layer wealth coupling + hierarchical VCG auction | [Phase 4](#phase-4--multi-scale-hierarchy) | Planned |
| **Recurrent Memory (RMT)** | Persistent "bioelectric" state across segments — infinite context | [Phase 5](#phase-5--persistent-memory-gap-junctions) | Planned |
| **Allostasis** | Meta-controller that adapts homeostatic setpoints under sustained pressure | [Phase 5](#phase-5--persistent-memory-gap-junctions) | Planned |

The system currently has two decoupled modules (MoB body, Steering mind) operating at token granularity within a single context window. The [full roadmap](#roadmap) lays out the path from this foundation to the complete TAME vision: coupled body–mind dynamics, stable economy, concept-level agency, multi-scale hierarchy, and persistent memory with allostatic stress response.

See the [Roadmap](#roadmap) for the dependency graph and implementation order.

---

## Quickstart

<a name="quickstart"></a>

### Prerequisites

- **Python 3.10+**
- **CUDA 12.x** with a GPU that has ≥ 16 GB VRAM (RTX 4090, A100, etc.)
- **Docker** (optional, recommended for reproducibility)

### Option A — Docker (Recommended)

Docker commands are the same on all platforms:

```bash
cd tame
docker compose -f docker-compose.dev.yml up --build
```

### Option B — Local

**Linux / macOS:**

```bash
cd tame
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**

```powershell
cd tame
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Windows (CMD):**

```cmd
cd tame
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The first run downloads the base model (~5 GB for Gemma-2-2B). Subsequent runs use the local cache.

### Verify

**Linux / macOS:**

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

**Windows (PowerShell):**

```powershell
# Health check
curl http://localhost:8000/health

# Generate with homeostatic steering
Invoke-RestMethod -Method Post -Uri http://localhost:8000/generate `
  -ContentType "application/json" `
  -Body '{"prompt": "Explain quantum entanglement", "max_tokens": 200}'

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

> **Known limitation:** `MODEL_PROFILES` is duplicated between `main.py` and `train.py` and must be kept in sync manually. If you train with one model profile and serve with another, `mob_state.pt` will fail to load due to dimension mismatches. [Phase 0a](#phase-0--foundation-engineering-hygiene) extracts these into a shared `config.py`.

### Quick Test (verify setup)

```bash
cd tame
python setup_tame.py --mode test --steps 100
```

Or via Docker:

```bash
cd tame
docker-compose -f docker-compose.train.yml run --rm train --mode test
```

### Full Training

```bash
cd tame

# 5 000 steps (~2-4 h on A100, ~6-8 h on RTX 4090)
python setup_tame.py --mode train --steps 5000

# Custom step count
python setup_tame.py --mode train --steps 10000

# Memory-constrained (< 24 GB VRAM) — add LoRA
python setup_tame.py --mode train --steps 5000 --use_lora

# Full pipeline: train + export in one step
python setup_tame.py --mode full --steps 5000
```

Or via Docker:

```bash
cd tame
docker-compose -f docker-compose.train.yml run --rm train --mode train --steps 5000

# With LoRA
docker-compose -f docker-compose.train.yml run --rm train --mode train --steps 5000 --use_lora
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
    ├── docker-compose.dev.yml   ← Dev server with hot-reload
    └── docker-compose.train.yml ← Containerised training
```

### Dev Server (Hot Reload)

For the fastest iteration loop — file saves trigger automatic server restart:

```bash
cd tame
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or via Docker (mounts local directory, auto-reloads on save):

```bash
cd tame
docker-compose -f docker-compose.dev.yml up --build
```

### Chat UI

A Gradio interface ships with live VCG auction visualisations — watch expert wealth diverge in real time.
Start the API server first (see above), then in a separate terminal:

**Any platform:**

```bash
cd tame
python chat_ui.py
# Open http://localhost:7860
```

### Key Concepts for Contributors

| Concept | File | What to Know |
|---------|------|--------------|
| **VCG Auction** | `mob.py` | Second-price auction from mechanism design theory; guarantees truthful bidding. `ConfidenceHead` predicts each expert's value. Winners pay the highest losing bid. |
| **Wealth Economy** | `mob.py` | `expert_wealth` buffers persist across batches. Three update paths exist (loss-based, quality-proxy, participation); `wealth_decay` and `reward_scale` control dynamics. The Gini coefficient is the primary health metric — too low (< 0.1) means experts aren't specialising, too high (> 0.6) means monopoly risk. |
| **Steering Vectors** | `steering.py` | Extracted via Difference-in-Means on contrastive prompt pairs; injected as residual-stream additions. Currently uses 4 contrastive pairs (thin). Orthogonal projection prevents capability damage. |
| **Adaptive Control** | `steering.py` | P-controller (not PID yet) with `kp`, `target_alignment`, and `max_strength`. Adjusts injection strength at each forward pass based on cosine alignment with the goal direction. |
| **Model Profiles** | `main.py`, `train.py` | `MODEL_PROFILES` dict maps model names to hidden dimensions and layer ranges. Duplicated across files (see Phase 0a). Change `ACTIVE_MODEL` to switch. |
| **Upcycling** | `mob.py` | `from_pretrained_ffn()` copies pretrained FFN weights to MoB shared base. Experts start as identity + jitter. No training-from-scratch required. |
| **Inference vs Training** | `mob.py` | Training uses dense forward pass (all tokens through all experts, masked) for gradient checkpointing. Inference uses sparse forward pass (only selected tokens) for speed. Wealth dynamics differ: faster decay and exploration bonus in inference mode. |

### Configuration

All tuneable parameters are documented in-line. The most impactful knobs:

```python
# tame/main.py

MOB_CONFIG = MoBConfig(
    num_experts=4,           # 4–8 for meaningful auction dynamics
    top_k=2,                 # Experts activated per token (2 is sweet spot)
    initial_wealth=75.0,     # Starting credits (lower = more room to grow)
    wealth_decay=0.997,      # Decay rate per step (0.997=aggressive, 0.999=slow)
    reward_scale=2.0,        # How strongly loss reduction is rewarded
    adapter_rank=32,         # LoRA rank per expert (32–64 sufficient; memory vs expressiveness)
    min_wealth=15.0,         # Floor prevents expert death
    max_wealth=750.0,        # Cap prevents monopoly
    jitter_std=0.08,         # Symmetry-breaking noise on initialisation
)

STEERING_CONFIG = SteeringConfig(
    base_strength=0.3,       # Injection coefficient α
    adaptive=True,           # Enable proportional drift correction
    target_alignment=0.7,    # Cosine-similarity setpoint
    kp=0.5,                  # Proportional gain (higher = more aggressive correction)
    max_strength=1.5,        # Safety cap on injection strength
    orthogonal_projection=True,  # Prevent capability damage from steering
)
```

### Tuning Guide & Diagnostics

The training loop logs comprehensive statistics every `log_frequency` steps. Here's how to interpret them:

| Metric | Healthy Range | What It Means |
|--------|---------------|---------------|
| **Loss** | Decreasing | Standard language modelling loss |
| **Perplexity** | Decreasing | Exponential of loss; lower = more confident predictions |
| **Calibration Loss** | 0.01–0.1 | Confidence head accuracy; should decrease as heads learn to predict their contribution |
| **Mean Wealth** | 50–500 | Average expert credits; should be stable, not pinned at floor or ceiling |
| **Wealth Std Dev** | > 10 | Divergence between experts; low std = no specialisation |
| **Gini Coefficient** | 0.10–0.50 | Wealth inequality. < 0.10 = experts converging (increase `reward_scale` or `jitter_std`). > 0.60 = monopoly risk (increase `min_wealth` or decrease `max_wealth`). |
| **Performance EMA** | Positive | Mean loss reduction vs baseline; negative = experts underperforming |

**Common failure modes:**

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Gini stays near 0 | Experts not differentiating | Increase `jitter_std` (0.08 → 0.15), increase `reward_scale`, or train longer |
| Gini > 0.6, one expert dominates | Wealth monopoly | Increase `min_wealth`, decrease `max_wealth`, or increase `wealth_decay` |
| Mean wealth pinned at ceiling | Rewards too generous | Decrease `reward_scale` or increase `wealth_decay` |
| Mean wealth pinned at floor | Decay too aggressive | Decrease `wealth_decay` (0.997 → 0.999) or increase `reward_scale` |
| NaN in loss or hidden states | Numerical instability | Check bfloat16 clamping; reduce `adapter_rank`; enable `orthogonal_projection` |
| Steering degrades output quality | Over-steering | Lower `base_strength` (0.3 → 0.15); enable `orthogonal_projection` |

---

## API Reference

<a name="api-reference"></a>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check with architecture status, GPU info, and module state |
| `/generate` | POST | Agential text generation with MoB routing and steering |
| `/generate/stream` | POST | Streaming generation via SSE with token-by-token output, periodic wealth traces, and steering traces |
| `/swarm/status` | GET | Expert wealth distribution & specialisation metrics (per-expert wealth, usage counts) |
| `/homeostasis/status` | GET | Current steering alignment, adaptive strength, and drift history |
| `/steering/update` | POST | Runtime modification of steering goals without server restart |
| `/traces/wealth` | GET | Aggregated VCG auction wealth traces for visualisation (Gini coefficient, per-expert history) |
| `/traces/steering` | GET | Homeostatic steering traces — alignment and strength (α_t) history |

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

### Streaming

The `/generate/stream` endpoint returns Server-Sent Events (SSE) with three event types:

| Event type | Frequency | Payload |
|------------|-----------|----------|
| `token` | Every token | `{"content": "..."}` |
| `progress` | Every 10 tokens | Token count, leading expert info |
| `trace_update` | Every 25 tokens | Intermediate wealth and steering traces for live chart updates |
| `complete` | Final | Full usage stats, homeostasis summary, wealth/steering traces |

---

## Theoretical Foundation

This project implements ideas from the following research areas:

- **TAME Framework** — Michael Levin's theory that intelligence is an emergent property of competent sub-agents cooperating under homeostatic pressure, not a monolithic central process. Cognition scales from cells to tissues to organisms through the same mechanisms; TAME-Swarm applies this to transformer layers. See [Levin 2022](https://arxiv.org/abs/2201.10346).
- **Mechanism Design Theory** — The VCG auction (Vickrey 1961, Clarke 1971, Groves 1973) is the only general mechanism that guarantees truthful bidding — no expert can benefit from misreporting its confidence. This is a provable property, not an empirical hope, making the routing economy formally incentive-compatible.
- **Activation Engineering** — Steering vectors discovered via contrastive activation analysis (Turner et al., 2023; Rimsky et al., 2024) provide zero-cost behavioural control in latent space. TAME-Swarm uses the Difference-in-Means extraction method and adds adaptive proportional control for dynamic strength.
- **Active Inference / Free Energy Principle** — The steering controller approximates active inference by maintaining a "preferred state" in activation space. The system minimises the distance between its current hidden state and the target direction, analogous to how biological systems minimise free energy relative to their homeostatic setpoint.
- **Sparse Mixture of Experts** — Token-level routing enables efficient scaling (Shazeer et al., 2017; Fedus et al., 2021). Standard MoE uses a learned gating network — a centralised router prone to collapse. TAME-Swarm replaces this with decentralised economic allocation where routing emerges from competitive bidding.

### From Biology to Code

| Biological Principle | TAME-Swarm Implementation | Status |
|---------------------|---------------------------|--------|
| Multicellular tissue with specialised organs | Expert pool with VCG auction routing | Implemented |
| Homeostatic setpoints (temperature, pH) | Steering vectors as target directions in activation space | Implemented |
| Morphogenetic field shaping cell behaviour | Steering signal coupled into expert confidence & routing | Phase 1 |
| Metabolic homeostasis (energy regulation) | Unified wealth economy with formal stability analysis | Phase 2 |
| Organ-level agency (not single-cell reflexes) | Chunk-level VCG routing with per-expert working memory | Phase 3 |
| Multi-scale nested agents (cells → tissues → organs) | Inter-layer wealth coupling + hierarchical auction | Phase 4 |
| Gap junctions synchronising bioelectric state | Recurrent Memory Transformer (RMT) for persistent internal state | Phase 5 |
| HPA axis / stress response (allostasis) | Meta-controller adapting steering setpoints under pressure | Phase 5 |

---

## Roadmap

<a name="roadmap"></a>

Improvements are ordered by **dependency** — each phase unlocks multiplicative returns for later phases.

```
Phase 0: Config + Split + Tests
    │
    ▼
Phase 1: Steering ↔ Economy Coupling → Better Contrastive Data → PID Controller
    │
    ▼
Phase 2: Stability Analysis → Unified Wealth Updater
    │
    ▼
Phase 3: Chunk-Level Routing → Expert Memory
    │
    ▼
Phase 4: Inter-Layer Coupling → Hierarchical Auction
    │
    ▼
Phase 5: RMT Gap Junctions → Allostasis
```

### Completed

- [x] Mixture of Bidders — VCG auction routing with LoRA-adapter experts
- [x] Cognitive Homeostasis — adaptive steering vectors with P-controller
- [x] Training pipeline — loss-based wealth updates, confidence calibration, checkpointing
- [x] Chat UI — Gradio interface with live VCG auction & steering visualisation
- [x] Multi-model support — Gemma 2B, Llama 3B, Mistral 7B

---

### Phase 0 — Foundation (Engineering Hygiene)

*"You can't study emergent dynamics in a system you can't reliably test."*

| Task | Description | Status |
|------|-------------|--------|
| **0a. Shared config module** | Extract `MODEL_PROFILES` and `ACTIVE_MODEL` into a single `config.py`, eliminating sync bugs between `main.py` and `train.py` | Not started |
| **0b. Split `mob.py`** | Decompose the 1,500-line module into `auction.py`, `experts.py`, `economy.py`, `serialization.py` | Not started |
| **0c. Test suite** | Numerical stability (NaN/Inf under bfloat16), checkpoint round-trips, wealth convergence smoke tests, steering extraction determinism | Not started |
| **0d. Request-scoped state** | Replace global `model`/`homeostat` with FastAPI dependency injection; eliminate race conditions under concurrent requests | Not started |

---

<a name="phase-1--steering-economy-coupling"></a>

### Phase 1 — Steering–Economy Coupling

*"The mind must influence the body, not just observe it."*

This is the **single highest-impact architectural change**. Currently MoB and Steering are parallel systems — MoB routes tokens, Steering corrects the output afterward. In TAME, the morphogenetic goal doesn't just *fix* deviations — it *shapes which cells activate in the first place*.

| Task | Description | Status |
|------|-------------|--------|
| **1a. Inject steering into confidence** | Modify `ConfidenceHead` so steering alignment modulates expert bids: $\text{bid}_i = c_i \times W_i \times (1 + \beta \cdot \cos(E_i(h),\, v_{\text{steer}}))$ — experts that move the representation *toward* the goal bid higher | Not started |
| **1b. Enrich contrastive data** | Expand `STEERING_TEMPLATES` from 4 to 50–200 diverse contrastive pairs per goal, producing genuine latent-trait directions instead of prompt-surface features | Not started |
| **1c. PID controller** | Upgrade P-only controller to full PID with anti-windup — integral term eliminates steady-state error, derivative term dampens oscillation under stochastic sampling | Not started |

**Why first:** Creates a feedback loop between goal and routing. Without it, improvements to steering and routing are additive. With it, they're multiplicative — better goals → better routing → better representations → easier steering.

---

<a name="phase-2--economy-stabilisation"></a>

### Phase 2 — Economy Stabilisation

*"An economy with hand-tuned magic numbers is a planned economy; planned economies collapse."*

| Task | Description | Status |
|------|-------------|--------|
| **2a. Formal stability analysis** | Fixed-point analysis on decay × reward equilibrium, eigenvalue analysis for oscillation conditions, empirical Gini-stability mapping | Not started |
| **2b. Unified wealth updater** | Merge the three wealth-update paths (`update_wealth_from_loss`, `_update_wealth_local_quality`, `_update_wealth_participation`) into a single `WealthUpdater` class with a pluggable reward signal | Not started |

**Why after Phase 1:** Steering–economy coupling changes the wealth dynamics. Stabilising before coupling would require re-doing the analysis.

---

<a name="phase-3--concept-level-agency"></a>

### Phase 3 — Concept-Level Agency

*"Cells don't decide one amino acid at a time."*

TAME posits agents operating at the *concept* level. Token-level routing limits experts to single-hidden-state decisions with no memory of what they bid on previously.

| Task | Description | Status |
|------|-------------|--------|
| **3a. Chunk-level routing** | Group tokens into 16–32 token spans (or attention-derived semantic chunks) and have experts bid on entire spans — enables specialisation on reasoning chains, code blocks, factual claims; reduces auction overhead by 16–32× | Not started |
| **3b. Expert memory (intra-forward)** | Lightweight per-expert recurrent state (EMA of past hidden states within current generation) — turns experts from reflexes into simple agents with short-term context | Not started |

**Why after Phase 2:** Chunk-level routing changes the reward signal granularity (one reward per chunk, not per token). Stable economy dynamics are needed before changing this shape.

---

<a name="phase-4--multi-scale-hierarchy"></a>

### Phase 4 — Multi-Scale Hierarchy

*"The whole point of TAME: from cells to tissues to organs."*

Currently the architecture is single-scale: individual experts competing flat within each layer. There's no mechanism for experts across layers to form coalitions or exhibit higher-order agency.

| Task | Description | Status |
|------|-------------|--------|
| **4a. Inter-layer wealth coupling** | "Tissue" abstraction: groups of 2–3 adjacent MoB layers share a pooled wealth component, enabling vertical specialisation (e.g., layers 8–10 form a "reasoning pathway" that co-evolves) | Not started |
| **4b. Hierarchical VCG auction** | Two-level auction: experts bid within their layer (inner), then layer-groups bid against each other for output influence (outer) — the computational analogue of Levin's nested agents | Not started |

**Why after Phase 3:** Multi-scale hierarchy only produces emergent structure when individual agents are meaningful. Phase 3 gives experts concept-level scope and memory; hierarchically organising token-level reflexes produces nothing.

---

<a name="phase-5--persistent-memory-gap-junctions"></a>

### Phase 5 — Persistent Memory (Gap Junctions)

*"Expanding the Cognitive Light Cone."*

While MoB provides the body and Steering provides the goal, the system currently lives only in the immediate present of its context window. In TAME, true scaling of cognition requires Gap Junctions: physical links that allow sub-agents to share their internal states, merging several small "selves" into one larger "Self."

| Task | Description | Status |
|------|-------------|--------|
| **5a. Recurrent Memory Transformer** | Memory tokens that persist across context window segments, acting as virtual gap junctions — the "bioelectric state" of the swarm survives beyond the context boundary, expanding the system's Cognitive Light Cone | Not started |
| **5b. Allostasis / stress response** | Meta-controller monitoring system-level statistics (mean alignment, Gini, loss trend) and adapting control setpoints — tightens steering under sustained adversarial pressure, relaxes when stable; the computational analogue of the HPA axis | Not started |

**Why last:** RMT and allostasis amplify whatever dynamics exist. If the economy is unstable (pre-Phase 2), persistent memory would propagate instability across segments. If steering is decoupled from routing (pre-Phase 1), persistent memory just remembers the wrong things.

---

### Future Directions

- [ ] Benchmark suite (Machiavelli alignment benchmark, Needle-in-Haystack for RMT)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built as a practical exploration of bio-inspired AI architectures.</sub>
</p>
