# tame/ — TAME-Swarm Core

This directory contains the full implementation of the TAME multi-scale competency architecture. See the [root README](../README.md) for project overview and quickstart.

---

## Module Map

| File | Role | Lines |
|------|------|-------|
| [`main.py`](main.py) | FastAPI inference server — model loading, API endpoints, streaming generation | Entry point |
| [`mob.py`](mob.py) | **Mixture of Bidders** — VCG auction, expert wallets, confidence heads, wealth economy | Core module |
| [`steering.py`](steering.py) | **Cognitive Homeostasis** — steering vector extraction, adaptive P-controller, orthogonal projection | Core module |
| [`train.py`](train.py) | Training loop — loss-based wealth updates, confidence calibration, checkpointing | Training |
| [`setup_tame.py`](setup_tame.py) | End-to-end workflow orchestrator (check → train → export) | CLI tool |
| [`chat_ui.py`](chat_ui.py) | Gradio interface with live VCG auction wealth visualisation | UI |

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

- Experts use **shared base weights + LoRA adapters** (not full FFN copies) — ~3 MB per expert per layer at rank 32.
- Only **middle layers** (20–70 % of depth) are converted; early tokenisation and late output layers remain untouched.
- Gaussian jitter on adapter init (`jitter_std=0.08`) breaks symmetry so experts can diverge.

### Steering Controller

```
At each selected layer l:

  alignment = cos(hₗ, v_steer)
  error     = target_alignment − alignment
  α(t)      = clamp(kp × error, min_strength, max_strength)
  hₗ'       = hₗ + α(t) × v_steer
```

If `orthogonal_projection` is enabled, `v_steer` is first projected to be orthogonal to the general-capability subspace, preventing "lobotomy" (steering that degrades base performance).

---

## API Endpoints

### `GET /health`

Returns model status, MoB configuration, steering status, and device info.

### `POST /generate`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `max_tokens` | int | 200 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `goal` | string | `"truthful"` | Steering goal (`truthful`, `reasoning`, `safe`) |
| `steering_strength` | float | null | Override adaptive steering (0.0–1.5) |
| `return_stats` | bool | false | Include MoB routing statistics in response |

### `GET /swarm/status`

Returns per-expert wealth, usage counts, and the Gini coefficient (wealth inequality — higher = more specialisation).

### `GET /homeostasis/status`

Returns current alignment to steering goal, active steering strength, and drift history.

### `POST /steering/update`

Modify steering goals at runtime without restarting the server.

---

## Training Details

### Loss-Based Wealth Updates

During training, the MoB economy learns from the *actual* loss signal:

1. Each batch computes per-token loss.
2. For each MoB layer, the loss reduction attributable to each expert is estimated.
3. Experts that improve loss receive wealth proportional to `reward_scale`.
4. A `confidence_calibration_weight` auxiliary loss trains confidence heads to predict routing quality.

### Checkpointing

Checkpoints save three files:

| File | Contents |
|------|----------|
| `model.safetensors` | Full model weights (base + LoRA adapters) |
| `mob_state.pt` | Expert wallets, confidence head weights, wealth history |
| `training_state.pt` | Optimizer state, scheduler, step count (for resume) |

### Export for Inference

```bash
python setup_tame.py --mode export --checkpoint ./tame_checkpoints/checkpoint-5000
```

This writes the production-ready state to `tame_inference/`, which `main.py` loads automatically on startup.

---

## Model Profiles

Profiles are defined in both `main.py` and `train.py` — **keep them in sync**.

| Profile | `hidden_dim` | `intermediate_dim` | MoB Layers | Notes |
|---------|-------------|--------------------:|:----------:|-------|
| `gemma-2-2b` | 2304 | 9216 | 5–18 | Fastest; no approval needed |
| `llama-3.2-3b` | 3072 | 8192 | 6–20 | Requires Meta access |
| `mistral-7b` | 4096 | 14336 | 8–24 | Best quality; most VRAM |

Switch by setting `ACTIVE_MODEL` at the top of each file.

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

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Use `--use_lora` for training; reduce `num_experts` or `adapter_rank` |
| `Model download timeout` | Set the timeout environment variable: Linux/macOS: `export HF_HUB_DOWNLOAD_TIMEOUT=3600`; PowerShell: `$env:HF_HUB_DOWNLOAD_TIMEOUT = 3600` |
| Wealth stays flat (Gini ≈ 0) | Train for more steps; increase `jitter_std` or `reward_scale` |
| Steering degrades output quality | Lower `base_strength`; enable `orthogonal_projection` |
| `--reload` not detecting changes | Ensure files are saved with LF line endings (not CRLF) |
