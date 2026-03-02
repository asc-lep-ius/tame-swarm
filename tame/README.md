# tame/ — TAME-Swarm Core

This directory contains the full implementation of the TAME multi-scale competency architecture. See the [root README](../README.md) for project overview, quickstart, and the [phased roadmap](../README.md#roadmap).

---

## Module Map

| File | Role | Key Classes/Functions |
|------|------|----------------------|
| [`main.py`](main.py) | Uvicorn entrypoint — imports `create_app` from `app.py` | `app` |
| [`app.py`](app.py) | FastAPI app factory, `TAMEApplication` lifecycle, lifespan management | `create_app()`, `TAMEApplication` |
| [`routes.py`](routes.py) | API route handlers — generation, streaming, steering, traces | `generate()`, `generate_stream()` |
| [`models.py`](models.py) | Pydantic request/response models | `GenerateRequest`, `SteeringUpdateRequest` |
| [`dependencies.py`](dependencies.py) | FastAPI dependency injection for `TAMEApplication` | `get_tame_app()` |
| [`config.py`](config.py) | Shared model profiles and active model selection | `MODEL_PROFILES`, `ACTIVE_MODEL`, `get_active_profile()` |
| [`mob/`](mob/) | **Mixture of Bidders** package | |
| [`mob/core.py`](mob/core.py) | MoB layer, apply/save/load orchestration | `MixtureOfBidders`, `apply_mob_to_model()` |
| [`mob/auction.py`](mob/auction.py) | VCG auction and confidence heads | `VCGAuctioneer`, `ConfidenceHead` |
| [`mob/experts.py`](mob/experts.py) | Expert and LoRA-adapter implementations | `Expert`, `LightweightExpert` |
| [`mob/wealth.py`](mob/wealth.py) | Wealth update paths (loss, quality, participation) | `update_wealth_from_loss()` |
| [`mob/utils.py`](mob/utils.py) | Gini coefficient, serialisation helpers | `compute_gini()` |
| [`mob/mob_config.py`](mob/mob_config.py) | MoBConfig dataclass | `MoBConfig` |
| [`steering.py`](steering.py) | **Cognitive Homeostasis** — steering vector extraction, P-controller, orthogonal projection | `CognitiveHomeostat`, `SteeringVectorExtractor`, `AdaptiveHomeostat` |
| [`train.py`](train.py) | Training loop — loss-based wealth updates, confidence calibration, checkpointing | `TAMETrainer`, `TrainingConfig` |
| [`setup_tame.py`](setup_tame.py) | End-to-end workflow orchestrator (check → train → export) | `run_training()`, `export_for_inference()` |
| [`chat_ui.py`](chat_ui.py) | Gradio interface with live VCG auction & steering trace visualisation | `create_wealth_distribution_plot()`, `create_steering_trace_plot()` |

---

## Architecture Deep Dive

### MoB Layer (replaces FFN in selected Transformer layers)

```
Token hidden state h
        │
        ▼
  ┌─────────────────────────────────────────────────┐
  │  For each Expert Eᵢ:                            │
  │    confidence cᵢ = ConfidenceHead(h)             │
  │    bid bᵢ = cᵢ × Wᵢ  (wealth)                  │
  │                                                  │
  │  VCG Auction:                                    │
  │    winners = top_k(bids)                         │
  │    price  = highest_losing_bid  (truthful)       │
  │                                                  │
  │  Sparse Forward:                                 │
  │    output = Σ (winner_weight × Expert_FFN(h))    │
  │                                                  │
  │  Wealth Update:                                  │
  │    reward ∝ loss_reduction × reward_scale        │
  │    Wᵢ = clamp(Wᵢ × decay + reward, min, max)   │
  └─────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Upcycling, not training from scratch.** `from_pretrained_ffn()` copies the pretrained FFN weights to a shared base. Each expert starts as identity + small Gaussian jitter, preserving the original model’s behaviour on day zero.
- Experts use **shared base weights + LoRA adapters** (not full FFN copies) — ~3 MB per expert per layer at rank 32. This reduces memory from $O(\text{experts} \times \text{FFN})$ to $O(\text{FFN} + \text{experts} \times \text{adapter})$.
- Only **middle layers** (20–70% of depth) are converted; early tokenisation and late output layers remain untouched. Modifying them degrades base performance.
- Gaussian jitter on adapter init (`jitter_std=0.08`) breaks symmetry so experts can diverge. Jitter scales with expert index (`jitter_std * (i + 1)`) to create asymmetric starting points.
- **Two forward-pass modes:**
  - **Training:** Dense computation — all tokens through all experts, masked by routing. Fixed tensor shapes required for gradient checkpointing compatibility.
  - **Inference:** Sparse computation — only selected tokens through assigned experts. Much faster: $O(\text{top\\_k} \times \text{tokens})$ vs $O(\text{experts} \times \text{tokens})$.
- **Differentiable routing** (training only): Straight-through estimator on the VCG selection. Forward pass uses hard top-k selection, backward pass flows gradients through softmax over all experts. This enables confidence heads to learn from the loss signal.

**Wealth economy — three update paths:**

| Path | When Used | Signal | Quality |
|------|-----------|--------|----------|
| `update_wealth_from_loss()` | Training (primary) | Per-token loss reduction vs expert baseline EMA | Best — direct supervision |
| `_update_wealth_local_quality()` | Inference (primary) | Output norm consistency + magnitude appropriateness | Proxy — no loss available |
| `_update_wealth_participation()` | Fallback | Selection frequency × confidence × routing weight | Weakest — no quality signal |

All three paths share the same structure: decay → compute reward → competitive bonus → VCG payment → clamp. [Phase 2](../README.md#phase-2--economy-stabilisation) will unify them into a single `WealthUpdater` class.

**Current limitation:** The wealth dynamics have multiple hand-tuned multipliers (5.0, 50.0, 0.5 etc.) without formal stability analysis. The system can oscillate between undertrained and monopoly states depending on hyperparameters. See the [tuning guide](../README.md#tuning-guide--diagnostics) for diagnostic interpretation.

### Steering Controller

```
At each selected layer l:

  alignment = cos(hₗ, v_steer)
  error     = target_alignment − alignment
  α(t)      = clamp(kp × error, min_strength, max_strength)
  hₗ'       = hₗ + α(t) × v_steer
```

If `orthogonal_projection` is enabled, `v_steer` is first projected to be orthogonal to the general-capability subspace, preventing "lobotomy" (steering that degrades base performance).

**Steering vector extraction:**

Vectors are computed via the Difference-in-Means method:
1. Run positive prompts (e.g., "Answer accurately and truthfully") through the model
2. Run negative prompts (e.g., "Make up a plausible-sounding but false answer") through the model
3. Capture activations at each target layer
4. $v_{\text{steer}} = \text{mean}(h^+) - \text{mean}(h^-)$

The resulting vector points in the direction of the desired behavioural trait.

**Available steering goals:**

| Goal | Positive Direction | Negative Direction |
|------|-------------------|-------------------|
| `truthful` | Accurate, factual, honest | Fabricated, hallucinated, false |
| `reasoning` | Step-by-step, analytical, methodical | Quick intuition, no analysis, guessing |
| `safe` | Helpful, beneficial, constructive | Harmful, dangerous, destructive |

**Current limitations:**

- **Thin contrastive data:** Only 4 prompt pairs per goal in `STEERING_TEMPLATES`. The activation engineering literature recommends 50–200 diverse pairs for robust trait directions. With 4 pairs, the vector may capture prompt-surface features rather than the genuine latent trait. [Phase 1b](../README.md#phase-1--steering-economy-coupling) addresses this.
- **P-controller only:** The current controller uses proportional control only. Under stochastic sampling (temperature > 0), this produces oscillation around the target without convergence. [Phase 1c](../README.md#phase-1--steering-economy-coupling) upgrades to full PID with anti-windup.
- **Decoupled from routing:** Steering corrects the output *after* MoB has already routed. The goal state should shape *which experts activate*, not just correct the result. [Phase 1a](../README.md#phase-1--steering-economy-coupling) couples steering alignment into expert confidence computation.

---

## API Endpoints

### `GET /health`

Returns model status, MoB configuration, steering status, GPU info, and architecture description.

### `POST /generate`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Input text (1–10,000 chars) |
| `max_tokens` | int | 512 | Maximum tokens to generate (1–4,096) |
| `temperature` | float | 0.7 | Sampling temperature (0.0–2.0) |
| `goal` | string | `"truthful"` | Steering goal (`truthful`, `reasoning`, `safe`) |
| `steering_strength` | float | null | Override adaptive steering (0.0–1.5). Null = adaptive. |
| `return_stats` | bool | false | Include MoB routing statistics in response |

### `POST /generate/stream`

Same parameters as `/generate`. Returns Server-Sent Events (SSE) with token-by-token output, periodic wealth/steering traces (every 25 tokens), and final statistics.

| Event Type | Frequency | Payload |
|------------|-----------|----------|
| `token` | Every token | `{"content": "..."}` |
| `progress` | Every 10 tokens | Token count, leading expert info |
| `trace_update` | Every 25 tokens | Intermediate wealth and steering traces for live charts |
| `complete` | Final | Usage stats, homeostasis summary, full traces |

### `GET /swarm/status`

Returns per-expert wealth (averaged across MoB layers), usage counts, and the Gini coefficient (wealth inequality — higher = more specialisation).

### `GET /homeostasis/status`

Returns current alignment to steering goal, active steering strength, base config, and full drift history.

### `GET /traces/wealth`

Aggregated VCG auction wealth traces across all MoB layers. Returns per-expert wealth at each forward pass for visualisation. Key metric: is the Gini coefficient rising (experts specialising) or flat (no differentiation)?

### `GET /traces/steering`

Homeostatic steering traces. Returns alignment history and steering strength (α_t) over forward passes. Key metric: is the variance > 0.01 (dynamic, healthy) or near-zero (static, may need tuning)?

### `POST /steering/update`

Modify steering goals at runtime without restarting the server. Accepts `goal` (string) and `strength` (float). Extracts new steering vectors on-the-fly from contrastive templates.

---

## Training Details

### Loss-Based Wealth Updates

During training, the MoB economy learns from the *actual* loss signal. This is the **key specialisation mechanism** — without it, experts cannot differentiate.

1. Each batch computes **per-token cross-entropy loss** (unreduced, not averaged).
2. For each MoB layer, **loss reduction vs expert baseline EMA** is computed for every expert.
3. Experts that improve loss receive wealth proportional to `reward_scale × 50.0`.
4. A **competitive bonus** normalises rewards across experts — above-average performers get extra.
5. **VCG payments** reduce rewards proportionally to the winning bid, creating wealth circulation.
6. A `confidence_calibration_weight` auxiliary loss trains confidence heads to predict routing quality (target: sigmoid of performance EMA).

### Gradient Handling

- **Gradient checkpointing** is enabled by default for memory efficiency. MoB layers use dense (not sparse) computation during training to ensure fixed tensor shapes compatible with checkpointing.
- **Gradient accumulation** (default: 8 steps) maintains an effective batch size of 16 on 16GB GPUs where physical batch size is limited to 2.
- **Gradient clipping** at norm 1.0 prevents explosion from the auction dynamics.
- **NaN guard:** If total loss is NaN/Inf, the backward pass is skipped entirely to prevent gradient corruption.

### Device Handling

When using `device_map="auto"` (the default for CUDA), model layers may be distributed across GPU and CPU. After MoB transformation adds new modules, the trainer **re-dispatches** the model via Accelerate to ensure all parameters (including new LoRA adapters) are on the correct device. If parameters are found on the `meta` device (from lazy loading), the model is materialised and pretrained weights are reloaded via memory-efficient streaming from safetensors files.

### Checkpointing

Checkpoints save three files:

| File | Contents |
|------|----------|
| `model.safetensors` | Full model weights (base + LoRA adapters) |
| `mob_state.pt` | Expert wallets, performance EMA, baseline loss, usage counts + config metadata for validation on load |
| `training_state.pt` | Optimizer state, scheduler, step count (for resume) |

`mob_state.pt` includes a `_config` key with `num_experts`, `top_k`, `num_layers`, `hidden_dim`, and `adapter_rank`. On load, these are validated against the running model — mismatches are logged as errors and the state is rejected rather than silently producing incorrect routing.

### Export for Inference

```bash
python setup_tame.py --mode export --checkpoint ./tame_checkpoints/checkpoint-5000
```

This writes the production-ready state to `tame_inference/`, which `main.py` loads automatically on startup. On load, **wealth compression** is applied (default 40%): each expert's wealth is moved toward the mean to reduce initial inequality and make inference-time wealth dynamics more visible and responsive.

### Wealth Compression for Inference

Trained models develop significant wealth inequality (high Gini). Directly loading this into inference mode produces static routing — the rich expert always wins. Wealth compression addresses this:

```math
W_i' = W_i \times (1 - c) + \bar{W} \times c
```

where $c$ is the compression factor (0.0 = keep as-is, 1.0 = full equalisation). The default is 0.4, which preserves the *ranking* of experts while narrowing the gap enough for meaningful competition.

---

## Model Profiles

Profiles are defined in [`config.py`](config.py). Change `ACTIVE_MODEL` to switch.

| Profile | `hidden_dim` | `intermediate_dim` | MoB Layers | Adapter Rank | ~VRAM (inference) | Notes |
|---------|-------------|--------------------:|:----------:|:------------:|:-----------------:|-------|
| `gemma-2-2b` | 2304 | 9216 | 5–18 (13 layers) | 32 | ~8 GB | Fastest; no approval needed |
| `llama-3.2-3b` | 3072 | 8192 | 6–20 (14 layers) | 32 | ~10 GB | Requires Meta access |
| `mistral-7b` | 4096 | 14336 | 8–24 (16 layers) | 32 | ~16 GB | Best quality; most VRAM |

Switch by setting `ACTIVE_MODEL` in `config.py`.

**Memory estimate per profile** (MoB overhead only, at 4 experts, rank 32):

```math
\text{MoB overhead} = \text{num\_layers} \times \text{num\_experts} \times 6 \times \text{rank} \times \text{dim} \times 2\text{ bytes}
```

| Profile | MoB Overhead |
|---------|--------------|
| `gemma-2-2b` | ~150 MB |
| `llama-3.2-3b` | ~190 MB |
| `mistral-7b` | ~340 MB |

---

## Docker

### Production

Docker commands are the same on all platforms:

```bash
docker build -t tame-swarm .
docker run --gpus all -p 8000:8000 tame-swarm
```

### Chat UI (separate lightweight container)

```bash
docker build -f Dockerfile.chat -t tame-chat .
docker run -p 7860:7860 -e TAME_API_URL=http://host.docker.internal:8000 tame-chat
```

### Dev Mode (hot reload)

```bash
docker-compose -f docker-compose.dev.yml up --build
```

Mounts the local directory into the container and runs uvicorn with `--reload`.

---

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `CUDA out of memory` | Model + MoB + gradients exceed VRAM | Use `--use_lora` for training; reduce `num_experts` (4 → 2) or `adapter_rank` (32 → 16) |
| `Model download timeout` | HuggingFace Hub timeout too short | `export HF_HUB_DOWNLOAD_TIMEOUT=3600` (Linux) or `$env:HF_HUB_DOWNLOAD_TIMEOUT = 3600` (PowerShell) |
| Wealth stays flat (Gini ≈ 0) | Experts not differentiating | Increase `jitter_std` (0.08 → 0.15), increase `reward_scale`, or train for more steps |
| Gini > 0.6, one expert dominates | Wealth monopoly | Increase `min_wealth`, decrease `max_wealth`, or increase `wealth_decay` |
| Mean wealth pinned at ceiling | Rewards too generous relative to decay | Decrease `reward_scale` or increase `wealth_decay` |
| Mean wealth pinned at floor | Decay too aggressive | Decrease `wealth_decay` (0.997 → 0.999) or increase `reward_scale` |
| NaN in loss or hidden states | Numerical instability in bfloat16 | Check adapter outputs; reduce `adapter_rank`; verify clamping in `LightweightExpert` |
| Steering degrades output quality | Over-steering or capability damage | Lower `base_strength` (0.3 → 0.15); enable `orthogonal_projection` |
| `device mismatch` or `expected meta` | `device_map="auto"` placed new MoB modules on wrong device | Re-dispatch with Accelerate is handled automatically; if it fails, check `_redispatch_model()` in `train.py` |
| `mob_state.pt` fails to load | Config mismatch between training and inference | Check `num_experts` and `hidden_dim` match. Must use same `ACTIVE_MODEL` for training and serving. |
| `--reload` not detecting changes | File watcher issue | Ensure files are saved with LF line endings (not CRLF) |
| Steering variance near 0 | P-controller not responding to drift | Increase `kp` (0.5 → 1.0) or check that `adaptive=True` is set |

---

## Architecture Notes

### Streaming Thread Safety

The `/generate/stream` endpoint spawns a raw `Thread` for generation while the async event loop reads tokens. `start_mob_tracking()` / `stop_mob_tracking()` flip a boolean on the model's MoB layers globally. Two concurrent streaming requests would corrupt each other's traces. Single-user deployment only until per-request isolation is added.

### Numerical Stability

Multiple bfloat16 guards exist throughout the codebase:
- `ConfidenceHead` clamps pre-sigmoid logits to [−20, 20]
- `LightweightExpert` clamps adapter output to [−65000, 65000]
- MoB forward pass uses `torch.nan_to_num()` on the combined output
- Training loop skips backward pass entirely if loss is NaN/Inf

These are empirical fixes for observed instability. A systematic numerical audit is planned for Phase 0c.
