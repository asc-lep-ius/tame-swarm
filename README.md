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

**Mixture of Bidders (MoB)** represents the Evolutionary Economy. It recognizes that "competence without comprehension" is the engine of life. By forcing experts to compete and profit, we replicate the bio-economic pressure that drives cellular specialization. The core novelty is replacing the standard learned MoE router (a central planner) with a **VCG (Vickrey-Clarke-Groves) auction** — a mechanism from economic theory under which no expert can improve its own payoff by misreporting its confidence for a token. Each expert accumulates wealth by performing well, creating emergent specialisation without any supervised routing signal.

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

Standard Mixture-of-Experts uses a learned router — one gating network trained by the global loss, which is a central planner by construction. MoB replaces it with a **VCG (Vickrey-Clarke-Groves) auction**: each expert maintains a *wallet* of credits, bids `confidence × wealth` for every token, and the top-k winners split the output evenly.

The substantive difference from a learned router is *where the training signal comes from*. No gradient from the language-modelling loss reaches a confidence head: winners share the output equally, so the routing decision is not differentiable with respect to any report. Each head is trained solely on the value that expert realised on the tokens it personally won. The experts' FFN adapters are still trained by the global loss — it is the *router* that is decentralised, not the whole layer.

**Why it matters:**

- **Truthful bidding** — for a single token, an expert cannot raise its own payoff by misreporting. The allocation is monotone in an expert's report, each winner is charged its critical value `b₍ₖ₊₁₎ / wᵢ`, and every winner receives the same `1/k` share regardless of what it reported. That is the standard strategyproofness argument for a single-parameter mechanism, and `tests/test_auction.py` checks it by exhaustive deviation rather than by assertion.
  The property also survives into the economy, which is the part that is easy to get wrong. Report, price, reward and charge are all denominated in loss reduction, and reward and charge share one coefficient, so wealth moves by `A·(value − price)` and its break-even sits **exactly** at the price — the same threshold the auction allocates on. An earlier revision scaled rewards ×100 against charges ×0.3; the mechanism was still strategyproof about a payoff nothing optimised, and an expert maximising *wealth* profited by overreporting. `test_wealth_threshold_coincides_with_the_auction_threshold` pins the crossing.
  **Scope:** this is a property of the per-token stage game. Wealth persists across tokens, so an expert's reports shape its future bids and prices; the repeated game is *not* covered, and no claim is made that a head's report is a *correct* estimate of its value — only that it has no incentive to distort whatever estimate it holds.
- **Emergent specialisation (not demonstrated)** — the intent is that experts reducing loss earn credits and reinforce a niche. No measurement supports this yet. In the only synthetic setting tried, wealth tracked the `ConfidenceHead` bias initialisation rather than expert competence: `r(wealth, expert index) ≈ −0.93` held across seeds regardless of which expert was actually the strongest, and the apparent competence correlation vanished once the competence vector was shuffled away from index order.
  **The Gini coefficient is not the measure of this and never was.** It measures dispersion of a wealth vector produced by an EMA with a tuned decay and a hard clamp to `[15, 750]`, so its value is largely a property of that update rule's fixed point, and a Gini of 0.12–0.35 is entirely consistent with every expert computing the same function. Its *direction* is wrong too: wealth multiplies the report inside the bid, so a rising Gini mechanically increases wealth's share of the routing decision and decreases the report's. Gini remains a reasonable economy-health diagnostic and is logged as `wealth/gini`; the specialisation measures are the `spec/` metrics from the held-out probe ([#12](#phase-05--mechanism-correction), shipped) — pairwise expert output divergence on identical hidden states, per-expert token-category routing profiles, and report decisiveness. A capability claim additionally needs the noise floor from [#13](#phase-05--mechanism-correction).
- **No router collapse (untested)** — the argument is that a market with per-expert wealth has no single gating network to collapse. It is an argument, not a result. The learned-router control that would evidence it now exists — `--router softmax`, the same confidence heads with the economy switched off ([#12](#phase-05--mechanism-correction)) — but no comparison has been run at a scale or seed count that would support a claim either way; the noise floor is [#13](#phase-05--mechanism-correction). Until then treat this as motivation. Note also that a wealth monopoly is a collapse mode of its own: `spec/routing_js_from_corpus` reads 0 both when routing ignores the token and when one expert wins everything, so it is read beside `expert_token_share`.
- **Memory-efficient** — shared base weights + LoRA-rank adapters keep VRAM overhead to ~3 MB per expert per layer at rank 32.

**Implementation details:**

- **Upcycling, not training from scratch.** MoB layers are initialised by copying the pretrained FFN weights to a shared base. Each expert starts as the identity transform (LoRA B-matrices zeroed) plus small Gaussian jitter to break symmetry. This preserves the original model's behaviour on day zero.
- **Layer selection matters.** Only middle layers (20–70% of model depth) are converted to MoB. Early layers handle tokenisation/syntax and late layers handle output formatting — modifying them degrades base performance.
- **Sparse computation.** Both training and inference use sparse gather/scatter — only selected tokens pass through their assigned experts. This is $O(\text{top\_k} \times \text{tokens})$ rather than $O(\text{experts} \times \text{tokens})$.
- **Per-expert value objective.** Each `ConfidenceHead` is regressed onto the loss reduction its expert delivered on the tokens it won, measured against the baseline that expert held when it bid. Because the mechanism is strategyproof, an expert's utility-maximising report *is* its value, so this objective and utility maximisation have the same optimum — the discrete utility has zero gradient almost everywhere, and the regression is its tractable form. **Two limits.** Value is only observed where an expert won, so the targets carry the selection bias of any bandit-feedback signal. And the target is clamped at zero, so what a head learns is the *positive part* of loss reduction, not its mean — a biased estimate that sits well above realised value in practice. The mechanism is strategyproof about the value an expert reports; it does not make the trained report an unbiased estimate of that value. See the abstention limitation below.
- **Payments are redistributed, not burned.** VCG prices have no recipient in a pool of experts, and once correctly scaled the outflow dwarfs the reward inflow — every expert converges on `min_wealth`. The Cavallo (2006) / Guo–Conitzer rule rebates each expert from the (k+1)-th highest bid *among the others*, a quantity it cannot influence, so the budget returns without moving any threshold. The divisor is the pool's largest wealth rather than the recipient's own: dividing by `wᵢ` is right for a price but pays the poorest expert the most as a rebate, and feasibility then holds only in bid units rather than in the credits the ledger uses. Green–Laffont says budget balance, strategyproofness and efficiency cannot all hold; this keeps the first two, and the residual is what is given up. That residual is not small in every regime: roughly 6% of the collection is burned on a flat wealth vector, ~32% across the configured band, and over 96% when one expert sits at `max_wealth` and the rest at the floor — so the rebate is weakest in exactly the monopoly regime the tuning guide warns about. A tighter report-independent divisor exists — the harmonic mean of the *k* richest wealths — and the choice is recorded on the [#15](#phase-05--mechanism-correction) row below.
- **Known limitation — abstaining pays.** Rebates go to every expert while charges fall only on winners, and top-*k* fills every slot with no reserve price, so an expert whose value is below the going price is forced to win at a loss. Measured over 300 steps: mean realised value of a win `+0.017` against a mean price of `+0.239`, so winning is a loss-making trade and `r(wealth, win share) = −0.28`. An earlier revision measured `−0.97`; most of that was a defect in the rebate divisor, not the economy. Two causes: the value target is clamped at zero, so a head predicts the **positive part** of loss reduction rather than its mean; and `expert_baseline_loss` is an EMA of the expert's *own* loss, which makes loss reduction a zero-mean fluctuation by construction, leaving little persistent value to price. Tracked as [#15](#phase-05--mechanism-correction); it is [Phase 2](#phase-2--economy-stabilisation) work, not a knob to turn before [#12](#phase-05--mechanism-correction) provides baselines.
- **Quasi-linear wealth.** Wealth moves by `reward − payment` with a *single* coefficient, derived from `reward_scale`, the path's reward multiplier and `top_k` rather than fitted. `payment_scale` survives only as a dimensionless deviation from that balanced point, defaulting to `1.0`. Quasi-linearity is a precondition of every VCG result, and one coefficient per side is what it means.
- **The report is a value estimate, not a probability.** `ConfidenceHead` emits `softplus(logits)` — a non-negative, unbounded estimate of the loss reduction the expert expects to deliver. A bid of ~0 is how an expert abstains. A sigmoid report would be capped at 1.0 while the reward it predicts is not, so "win when report > price" and "profit when value > price" could not coincide.
- **Three wealth-update paths** exist today: loss-based feedback (training, primary), local output-quality proxy (inference), and participation-based (fallback). [Phase 2](#phase-2--economy-stabilisation) will unify these into a single parameterised mechanism.
- **Gate-swap baseline.** `routing_share="proportional"` restores an own-bid-weighted gate as the comparison arm for [#12](#phase-05--mechanism-correction). It is *not* incentive compatible — a winner can enlarge its own share of the output while its price stays fixed — which is the single property the swap is meant to isolate.
- **The baseline gate reads relative wealth, never its scale.** A winner's share is `bid ** (1 / routing_temperature)` normalised over the winners, i.e. a softmax over `log(confidence) + log(wealth)`. The earlier gate took a softmax over the bids themselves, and softmax is not scale invariant: measured at default initialisation over `softplus` reports, its top-1 weight had median ≈0.99 at `initial_wealth` and 1.000 at `max_wealth`, and across the configured wealth band its effective expert count was **1.000** — `top_k=2` paying for two experts and using one. Because the absolute wealth scale drifts as the economy runs, that made gate sharpness a moving confound in every number read through it, the Phase 1 coupling ablation included. In the log domain a uniform rescaling of all wealth is a constant shift that softmax absorbs exactly, so only *relative* wealth reaches the gate. Verified invariant to under 2e-7 — float32 rounding on the log — across sixteen orders of magnitude of wealth scale, and stationary over a 5000-step run: top-1 median moves ≤0.0085 and effective expert count ≤0.0017 between step 500 and step 5000 — two-fifths and one-sixth of the drift the test admits — against ≤0.153 and ≤0.164 for the gate it replaced. `routing_temperature` defaults to `1.0` and is scale invariant at every setting — exactly in the algebra, and measured under 1e-6 in float32 down to `tau=0.1` — so sharpness is a choice rather than, as the raw bid scale was, a side effect.
- **What the gate actually did is logged, not inferred.** Every forward records the realised top-1 routing weight (mean, median, fraction above 0.99) and `exp(entropy(routing_weights))` — the number of experts the output was genuinely mixed from — and `get_mob_statistics` aggregates them beside the wealth figures. `top_k` is a configuration; the effective expert count is the outcome, and it is the statistic that would have surfaced the saturation above without anyone going looking for it.

### Module 2 — Cognitive Homeostasis: *The Mind*

Activation **Steering Vectors** encode goals (truthfulness, safety, reasoning) as linear directions in the model's hidden space. A proportional controller injects these vectors at every selected layer, dynamically adjusting strength based on how far the model's activations have drifted from the target:

```math
\alpha(t) = k_p \cdot (\text{target\_alignment} - \cos(h_t,\; v_{\text{steer}}))
```

- **Zero context-window cost** — no system-prompt tokens consumed; steering operates entirely in weight/activation space.
- **Latent-space operation** — acts on the residual stream, not on text tokens. This makes it harder (though not impossible) for prompt-based attacks to circumvent, since the correction bypasses the text channel entirely. Formal adversarial evaluation is planned but not yet complete.
- **Orthogonal projection** targets the "lobotomy" problem, where steering degrades base performance. A general corpus is run through the model, the leading principal components of the per-token activations at each steered layer are taken as the capability subspace, and the steering vector is projected orthogonal to it before injection. When a goal lies almost entirely inside that subspace, less than 5% of it survives projection and steering falls back to the unprojected vector with a logged warning rather than amplifying rounding noise to unit norm. **Validation status:** the mechanism is wired end-to-end and tested, but capability *preservation* has not been measured — that needs the held-out benchmark in [#12](#phase-05--mechanism-correction). Treat it as a targeted mitigation, not a demonstrated one.

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

- **Docker** with **NVIDIA Container Toolkit**
- **GPU** with ≥ 16 GB VRAM (RTX 4090, A100, etc.)

```bash
cd tame
docker compose -f docker-compose.dev.yml up --build
```

The first run downloads the base model (~5 GB for Gemma-2-2B). If you want to use Gemma-2-2B, you need a huggingface account and request access from google then add the "HF_TOKEN" to the .env file, which they grant instantly upon request AFAIK. Subsequent runs use the local cache.

### Verify

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement", "max_tokens": 200}'

curl http://localhost:8000/swarm/status
```

---

## Training

<a name="training"></a>

Training is what differentiates the experts within a MoB layer. Upcycling starts them identical by construction — copied FFN weights with LoRA-B zeroed — so at step zero pairwise expert output divergence is exactly zero and the wealth vector is flat.

### Supported Base Models

| Profile | Params | Train Speed | Quality | Access |
|---------|--------|-------------|---------|--------|
| `gemma-2-2b` | 2 B | ~3.5× faster | Medium | Open |
| `llama-3.2-3b` | 3 B | ~2.5× faster | Good | Requires Meta approval |
| `mistral-7b` | 7 B | 1× (baseline) | Best | Open |

Switch models by changing `ACTIVE_MODEL` in [tame/config.py](tame/config.py).

### Quick Test (verify setup)

```bash
cd tame
docker compose -f docker-compose.train.yml run --rm train --mode test
```

### Full Training

```bash
cd tame

# 5 000 steps (~2-4 h on A100, ~6-8 h on RTX 4090)
docker compose -f docker-compose.train.yml run --rm train --mode train --steps 5000

# Custom step count
docker compose -f docker-compose.train.yml run --rm train --mode train --steps 10000

# Memory-constrained (< 24 GB VRAM) — add LoRA
docker compose -f docker-compose.train.yml run --rm train --mode train --steps 5000 --use_lora

# Full pipeline: train + export in one step
docker compose -f docker-compose.train.yml run --rm train --mode full --steps 5000
```

### Held-out Evaluation and Baseline Arms

Every capability claim rests on the held-out split, never on the training batch.
`--eval_steps` runs the evaluation loop with the economy frozen — no wealth
updates, no usage counts, no coupling step — and logs `eval/loss`,
`eval/perplexity` and the `spec/` specialisation probe to `metrics.jsonl`,
namespaced so they can never be confused with the `train/` statistics of the
batch just fitted.

The split is built once and frozen. It prefers the dataset's own `validation`
split, which is disjoint from training at *article* level; for a dataset that
ships none, it falls back to a stride-97 holdout of the training stream that the
training loader then skips. The fallback is disjoint by index but may share
articles with training data, so it warns and records which path produced it.

Three arms, identical in everything but the gate:

```bash
# The auction (default)
python train.py --router mob --max_steps 1000 --seed 42

# The learned-gate control: the same confidence heads, softmaxed, economy off
python train.py --router softmax --max_steps 1000 --seed 42

# The capability-preservation floor: original FFN, no routing
python train.py --router dense --max_steps 1000 --seed 42
```

Or all three plus the parity check and a comparison table:

```bash
# Smoke run: a local tiny model and synthetic corpus, CPU, no network
uv run python scripts/compare_routers.py

# A real comparison
uv run python scripts/compare_routers.py \
    --model_id google/gemma-2-2b-it --dataset wikitext \
    --steps 1000 --device cuda --use_lora
```

Parity is asserted, not assumed: each arm records a fingerprint over seed, data
order (a hash of the tokens it will actually train on), requested layer range,
adapter rank, step budget and held-out split, and the comparison raises rather
than print a table whose arms differ in anything but the router. One seed is not
a result — the between-seed spread measured on report decisiveness was ~46 points
— so repeat over seeds, and read them against the noise floor from
[#13](#phase-05--mechanism-correction).

### Reproducibility

<a name="reproducibility"></a>

Every number above this line is one sample. [#13](#phase-05--mechanism-correction)
is the harness that turns "one sample" into "measured, with a spread":

**Determinism.** `--deterministic` (on by default) seeds `torch`, `numpy`,
`random` and CUDA from one field, sets `CUBLAS_WORKSPACE_CONFIG` before the
first CUDA context exists, and calls `torch.use_deterministic_algorithms(True,
warn_only=True)` — forcing a deterministic kernel wherever one exists and
logging the rest as a known variance source rather than accepting it silently.
Two runs of an identical config produce bitwise-identical `train/loss` traces;
this is asserted by `tests/test_determinism.py` (`gpu`-marked) in CI, which no
longer allows `test-gpu` to fail silently. `--shuffle_buffer_size` seeds a
bounded shuffle of the streaming dataset (0, the default, keeps the current
unshuffled — and therefore already order-deterministic — stream).

**Multi-seed harness.** `scripts/run_seeds.py` runs one configuration over
several seeds and reports mean ± std for every headline metric:

```bash
uv run python scripts/run_seeds.py \
    --model_id Qwen/Qwen3-1.7B --dataset wikitext \
    --router mob --steps 500 --device cuda --use_lora \
    --adapter_rank 32 --layers 6:22 --max_seq_length 512 --seeds 0,1,2
```

Each seed writes to its own `<output_dir>/seed<N>/`, and a `seed_summary.json`
at the workspace root carries every per-seed value plus the aggregate — the
input `scripts/compare_runs.py` reads to answer "is this effect real" in one
command:

```bash
uv run python scripts/compare_runs.py --group_a runs/mob --group_b runs/softmax
```

For each metric both groups measured on ≥2 seeds, it reports the delta, the
pooled replicate spread (the noise floor), and the delta in units of that
spread — a `delta/std` well under 1 is not distinguishable from re-running the
same configuration.

**Noise floor.** Measured by running one configuration three times at a fixed
step budget and reading the spread `run_seeds.py` reports. Current number —
`mob`, Qwen3-1.7B, 500 steps, LoRA rank 32, 16 converted layers, wikitext-2,
seeds 0/1/2, `n=3`:

| metric | mean | std | relative |
|---|---|---|---|
| `eval/loss` | 2.7315 | 0.0004 | 0.02% |
| `eval/perplexity` | 15.355 | 0.006 | 0.04% |
| `spec/expert_cosine_distance` | ≈0 | ≈0 | — (no specialisation at this budget; expected — see the Module 1 design-limitations note above) |
| `spec/routing_js_from_corpus` | 0.0183 | 0.0020 | 11% |
| `spec/report_decisiveness` | 0.4291 | 0.0495 | 12% (±4.9 points absolute) |

This is a real 3-seed measurement, not a placeholder — but 500 steps on an
ungated Qwen3-1.7B substitute, run to keep the harness itself honest, not the
number Phase 1 ablations should be read against. Held-out loss and perplexity
are already tight enough to detect small effects; `report_decisiveness`'s ±4.9
points is the bar an ablation on *that* metric has to clear before it's a
result rather than seed noise, and it should be re-measured at whatever step
budget and model the first real ablation actually uses — `scripts/run_seeds.py`
is the one-command way to do that. (The issue that opened #13 named "Gini and
mean alignment" as the metrics to publish here; both predate #12's mechanism
correction, which replaced Gini as a specialisation measure with the `spec/`
probe above and never introduced a "mean alignment" metric, so the table
reports the project's current headline metrics instead.)

**Disk budget.** `--checkpoint_min_free_gb` (default 50) refuses to write a
checkpoint — raising rather than filling the disk — when free space on the
filesystem backing `output_dir` drops below the threshold. Retention
(first/best/final/`checkpoint_keep_last`, above) prunes *after* a save
completes, so on its own it cannot stop a single checkpoint from being the
write that fills a shared runner's disk; this is the check in front of it. 50GB
is a placeholder default, not a measured Hephaestus budget — tune it with this
one flag once that number is known.

### What Happens During Training

| Phase | Description |
|-------|-------------|
| **Wealth Updates** | Experts that reduce loss gain credits; poor performers lose them |
| **VCG Auction Routing** | Wealth differentials shift which experts can afford to win a token; whether the winners are the *most competent* ones is the hypothesis under test, not an established result |
| **Confidence Calibration** | Each expert's head is regressed onto the loss reduction that expert realised on the tokens it won — the only training signal a head receives, and it reaches the head alone, not the backbone |
| **Checkpoint Persistence** | `mob_state.pt` saves the full economic state for later inference |

### Training Outputs

```
tame_checkpoints/
├── held_out_split.pt         # Frozen, fingerprinted evaluation set (built once)
├── metrics.jsonl             # train/ eval/ spec/ wealth/ routing/ metrics per step
├── mlruns/                   # MLflow local tracking store — params, metrics, artifacts
├── checkpoint-1000/
│   ├── model.safetensors     # Model weights
│   ├── mob_state.pt          # Expert wealth & auction state
│   └── training_state.pt     # Optimizer state, arm fingerprint, eval history
└── checkpoint-5000/
    └── ...

tame_inference/               # Automatically exported for the API server
├── mob_state.pt
├── inference_config.json
└── loader_snippet.py
```

Retention keeps the first, best (by held-out `eval/loss`) and final checkpoint
on disk permanently, plus `--checkpoint_keep_last` most recent transiently
(evicted as newer checkpoints arrive), and `--checkpoint_min_free_gb` refuses
to write a checkpoint at all below a configured free-space floor — see
[Reproducibility](#reproducibility) for why retention alone can't be that
guarantee. Only the permanent set — first/best/final
— is ever archived to `mlruns/`; the `--checkpoint_keep_last` window is a local
disk convenience and is never uploaded, so archiving never outgrows what
retention permanently keeps. `mlflow ui --backend-store-uri
file:./tame_checkpoints/mlruns` opens the run comparison view. Comparing
several runs (e.g. `scripts/compare_routers.py`'s three arms, or several seeds
of one config) needs them writing into one store: either set
`MLFLOW_TRACKING_URI` to a shared location before training, or point
`--output_dir` at a common parent so a per-run default lines up.

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
├── README.md                    ← You are here
├── docker-compose.test.yml      ← Containerised test runner
├── pyproject.toml
├── tests/                       ← Test suite (pytest)
│   ├── conftest.py
│   ├── test_auction.py
│   ├── test_config.py
│   ├── test_experts.py
│   ├── test_mixture.py
│   ├── test_mob_config.py
│   ├── test_steering.py
│   ├── test_wealth_updates.py
│   └── test_api.py
└── tame/                        ← Core implementation
    ├── main.py                  ← Uvicorn entrypoint (imports create_app)
    ├── app.py                   ← FastAPI app factory + TAMEApplication lifecycle
    ├── routes.py                ← API route handlers
    ├── models.py                ← Pydantic request/response models
    ├── dependencies.py          ← FastAPI dependency injection
    ├── config.py                ← Shared model profiles + active model selection
    ├── mob/                     ← Mixture of Bidders package
    │   ├── __init__.py
    │   ├── core.py              ← MixtureOfBidders layer, apply/save/load
    │   ├── auction.py           ← VCGAuctioneer
    │   ├── experts.py           ← Expert, LightweightExpert, ConfidenceHead
    │   ├── wealth.py            ← Wealth update paths (loss, quality, participation)
    │   ├── softmax_router.py    ← Learned-gate control arm (same heads, no economy)
    │   ├── utils.py             ← Gini coefficient, frozen_economy, serialisation helpers
    │   └── mob_config.py        ← MoBConfig dataclass
    ├── steering.py              ← Cognitive Homeostasis: steering vectors, P-controller
    ├── evaluation.py            ← Frozen held-out split + evaluation loop
    ├── specialisation.py        ← Functional specialisation metrics (not Gini)
    ├── parity.py                ← Arm fingerprints; refuses an unmatched comparison
    ├── metrics.py               ← JSONL metric sink, forwards to tracking.py
    ├── tracking.py              ← MLflow wrapper — the only file that imports mlflow
    ├── train.py                 ← Training loop with MoB economic dynamics
    ├── setup_tame.py            ← End-to-end train → export workflow
    ├── chat_ui.py               ← Gradio chat interface with live wealth visualisation
    ├── requirements.txt
    ├── Dockerfile               ← Production container (CUDA 12.6)
    ├── Dockerfile.chat          ← Lightweight chat UI container
    ├── docker-compose.dev.yml   ← Dev server with hot-reload
    └── docker-compose.train.yml ← Containerised training
```

### Dev Server (Hot Reload)

File saves trigger automatic server restart:

```bash
cd tame
docker compose -f docker-compose.dev.yml up --build
```

### Chat UI

A Gradio interface ships with live VCG auction visualisations — watch expert wealth diverge in real time.
Start the API server first, then in a separate terminal:

```bash
cd tame
docker build -f Dockerfile.chat -t tame-chat .
docker run -p 7860:7860 -e TAME_API_URL=http://host.docker.internal:8000 tame-chat
```

### Testing

Run the full test suite inside the same CUDA container used by the app — no local Python needed:

```bash
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

47 tests across 8 modules covering auction properties, wealth dynamics, steering, API endpoints, config, and experts.

### Key Concepts for Contributors

| Concept | File(s) | What to Know |
|---------|---------|--------------|
| **VCG Auction** | `mob/auction.py` | Externality-priced top-*k* auction. Each winner pays `b₍ₖ₊₁₎ / wᵢ` — the displaced welfare, divided by its own weight so the price is in the units it reports — and receives a `1/k` share that ignores its own bid. Strategyproof per token, and the wealth update shares the auction's break-even; see the scope note under Module 1. `ConfidenceHead` reports each expert's estimated loss reduction and is trained only on that expert's own outcomes. |
| **Wealth Economy** | `mob/wealth.py` | `expert_wealth` buffers persist across batches. Three update paths exist (loss-based, quality-proxy, participation); `wealth_decay` and `reward_scale` control dynamics. Gini is an economy-health diagnostic — too low (< 0.1) means near-flat wealth, too high (> 0.6) means monopoly risk — and is **not** a specialisation measure; see `specialisation.py`. |
| **Held-out evaluation** | `evaluation.py` | The frozen, fingerprinted validation split and the eval loop that reads it with the economy frozen (no wealth updates, no usage counts, no coupling step). Prefers the dataset's own `validation` split; falls back to a stride-97 holdout of the training stream, which the training loader then skips. |
| **Specialisation metrics** | `specialisation.py` | What experts *do*: pairwise expert output divergence on identical held-out hidden states, per-expert token-category routing profiles against the corpus marginal, and report decisiveness (how often the top-1 winner is the top-1 report). Probe ≥ 4096 *unpadded* tokens — padding is excluded from every statistic, because a pad carries an id, takes a category and is routed like any other position. |
| **Baseline arms** | `train.py`, `mob/softmax_router.py` | `--router {mob,softmax,dense}`. `softmax` is the same confidence heads with the economy switched off; `dense` is the unrouted FFN. `parity.py` refuses a comparison whose arms differ in anything but the gate. |
| **Steering Vectors** | `steering.py` | Extracted via Difference-in-Means on contrastive prompt pairs; injected as residual-stream additions. Currently uses 4 contrastive pairs (thin). Orthogonal projection removes the leading principal components of general-corpus activations before injection; capability preservation is not yet measured. |
| **Adaptive Control** | `steering.py` | P-controller (not PID yet) with `kp`, `target_alignment`, and `max_strength`. Adjusts injection strength at each forward pass based on cosine alignment with the goal direction. |
| **Model Profiles** | `config.py` | `MODEL_PROFILES` dict maps model names to hidden dimensions and layer ranges. Change `ACTIVE_MODEL` to switch. |
| **Upcycling** | `mob/experts.py` | `from_pretrained_ffn()` copies pretrained FFN weights to MoB shared base. Experts start as identity + jitter. No training-from-scratch required. |
| **Inference vs Training** | `mob/core.py` | Both use sparse forward pass (only selected tokens through assigned experts via gather/scatter). Training adds a straight-through estimator for differentiable routing. Wealth dynamics differ: faster decay and exploration bonus in inference mode. |

### Configuration

All tuneable parameters are documented in-line. The most impactful knobs:

```python
# tame/config.py / tame/app.py

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
    orthogonal_projection=True,   # Project the goal out of the capability subspace
    capability_subspace_rank=8,   # Principal components treated as capability
)
```

### Tuning Guide & Diagnostics

The training loop logs comprehensive statistics every `log_frequency` steps. Here's how to interpret them:

| Metric | Healthy Range | What It Means |
|--------|---------------|---------------|
| **Loss** | Decreasing | Standard language modelling loss |
| **Perplexity** | Decreasing | Exponential of loss; lower = more confident predictions |
| **Calibration Loss** | 0.01–0.1 | Per-expert value objective; should decrease as heads learn to report the value they realise. A flat line at a constant means no gradient is reaching the heads |
| **Mean Wealth** | 50–500 | Average expert credits; should be stable, not pinned at floor or ceiling |
| **Wealth Std Dev** | > 10 | Divergence between experts; low std = no specialisation |
| **Gini Coefficient** | 0.10–0.50 | Wealth inequality, an economy diagnostic. < 0.10 = near-flat wealth (increase `reward_scale` or `jitter_std`). > 0.60 = monopoly risk (increase `min_wealth` or decrease `max_wealth`). Not a specialisation measure — a high Gini means *less* of the routing is decided by what experts report |
| **Performance EMA** | Positive | Mean loss reduction vs baseline; negative = experts underperforming |
| **`eval/perplexity`** | Decreasing | Held-out perplexity on the frozen split. The only perplexity that supports a capability claim; `train/perplexity` is a statistic of the batch just fitted |
| **`spec/expert_cosine_distance`** | > 0, rising | Pairwise divergence of expert outputs on identical hidden states. Exactly 0 at upcycling; stays ~0 if experts compute the same function however unequal their wealth |
| **`spec/routing_js_from_corpus`** | > 0 | How far each expert's intake diverges from the corpus token-category marginal. 0 means routing is blind to the token — including when one expert wins everything |
| **`spec/report_decisiveness`** | Context | Fraction of tokens whose top-1 winner is the top-1 report. 1.0 for the `softmax` arm by construction; below 1 for `mob` is wealth overturning reports. Measured at 31–33% on the synthetic economy at a wealth spread of only 2.0× |

**Common failure modes:**

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Gini stays near 0 | Wealth is near-flat, so the auction allocates almost entirely on reports (not by itself a fault) | Increase `jitter_std` (0.08 → 0.15), increase `reward_scale`, or train longer |
| `spec/expert_cosine_distance` stays ~0 | Experts genuinely are computing the same function, whatever the wealth spread says | Increase `jitter_std` or `adapter_rank`, or train longer — this is the real "not specialising" symptom |
| `spec/report_decisiveness` far below 1 | Wealth is overturning the reports; the auction's incentive machinery governs less of the routing than it appears to | Narrow the wealth band ([#16](#phase-05--mechanism-correction)) |
| Gini > 0.6, one expert dominates | Wealth monopoly | Increase `min_wealth`, decrease `max_wealth`, or increase `wealth_decay` |
| Mean wealth pinned at ceiling | Rewards too generous | Decrease `reward_scale` or increase `wealth_decay` |
| Mean wealth pinned at floor | Decay too aggressive | Decrease `wealth_decay` (0.997 → 0.999) or increase `reward_scale` |
| NaN in loss or hidden states | Numerical instability | Check bfloat16 clamping; reduce `adapter_rank` |
| Steering degrades output quality | Over-steering, or the goal overlaps the capability subspace | Lower `base_strength` (0.3 → 0.15); raise `capability_subspace_rank` (8 → 16) |
| Log warns the projection left < 5% of the steering vector | The goal direction lies inside the capability subspace, so steering runs unprojected | Lower `capability_subspace_rank`, or widen the contrastive pairs so the goal is less collinear with general activity |

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
- **Mechanism Design Theory** — The VCG family (Vickrey 1961, Clarke 1971, Groves 1973) prices each winner at the externality it imposes. MoB runs the single-parameter case: a monotone top-*k* allocation with critical-value payments, which is strategyproof for a single token (Myerson 1981). Four things are needed to make that statement mean anything here, and all four are in the code — the payment is divided by the winner's own wealth, because the allocation maximises a *weighted* welfare; a winner's share of the output does not depend on its own report; reports, prices and rewards share one unit, loss reduction; and reward and charge share one coefficient, so `reward − payment` is a genuine quasi-linear utility rather than two differently-scaled terms. The claim is bounded: it is about the per-token stage game with the wealth vector held fixed, not about the repeated game the wealth dynamics create.
- **Activation Engineering** — Steering vectors discovered via contrastive activation analysis (Turner et al., 2023; Rimsky et al., 2024) provide zero-cost behavioural control in latent space. TAME-Swarm uses the Difference-in-Means extraction method and adds adaptive proportional control for dynamic strength.
- **Active Inference / Free Energy Principle** — The steering controller approximates active inference by maintaining a "preferred state" in activation space. The system minimises the distance between its current hidden state and the target direction, analogous to how biological systems minimise free energy relative to their homeostatic setpoint.
- **Sparse Mixture of Experts** — Token-level routing enables efficient scaling (Shazeer et al., 2017; Fedus et al., 2021). Standard MoE trains one gating network from the global loss. TAME-Swarm removes that gradient path entirely: no language-modelling gradient reaches a confidence head, and each head is trained only on the value its own expert realised. Whether this routes *better* than a learned gate is unmeasured — see [#12](#phase-05--mechanism-correction).

### From Biology to Code

| Biological Principle | TAME-Swarm Implementation | Status |
|---------------------|---------------------------|--------|
| Multicellular tissue with specialised organs | Expert pool with VCG auction routing | Implemented; routing quality unmeasured |
| Homeostatic setpoints (temperature, pH) | Steering vectors as target directions in activation space | Implemented; 4 contrastive pairs per goal (thin) |
| Morphogenetic field shaping cell behaviour | Steering signal coupled into expert confidence & routing | `coupling.py` exists but nothing attaches it at runtime — Phase 1 |
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
Phase 0: Config + Split + Tests                           ✔ DONE
    │
    ▼
Phase 0.5: Mechanism Correction + Baselines               ◐ IN PROGRESS
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
- [x] Phase 0 — shared `config.py`, `mob/` package split, `main.py` split (app/routes/models/dependencies), test suite, security hardening, code quality cleanup
- [x] Phase 0.5 (partial) — VCG payments corrected and made quasi-linear; the auction made strategyproof (weighted price divided by own weight, share independent of own bid); confidence heads given a per-expert value objective in place of an inert calibration loss; capability subspace estimated and wired into steering; README mechanism claims reconciled with the code

---

### Phase 0 — Foundation (Engineering Hygiene) ✔️

*"You can't study emergent dynamics in a system you can't reliably test."*

| Task | Description | Status |
|------|-------------|--------|
| **0a. Shared config module** | Extract `MODEL_PROFILES` and `ACTIVE_MODEL` into `config.py` | Done |
| **0b. Split `mob.py`** | Decompose into `mob/` package: `core.py`, `auction.py`, `experts.py`, `wealth.py`, `utils.py`, `mob_config.py` | Done |
| **0c. Test suite** | VCG auction properties, numerical stability, checkpoint round-trips, wealth convergence, steering, API endpoints (8 test files) | Done |
| **0d. Split `main.py`** | Extract into `app.py` (factory + lifecycle), `routes.py`, `models.py`, `dependencies.py` with DI via `TAMEApplication` | Done |
| **0e. Security hardening** | Stop leaking `str(e)` to clients, add input validation on `/steering/update`, make `trust_remote_code` configurable | Done |
| **0f. Code quality** | `print()` → structured logging, `deque` for steering history, named constants for magic numbers, modern typing | Done |

---

<a name="phase-05--mechanism-correction"></a>

### Phase 0.5 — Mechanism Correction & Baselines

*"Fix the arithmetic before anything reads it."*

An audit of the mechanism claims against the implementation. The auction the architecture is named for was priced wrong, the objective that was supposed to train the confidence heads carried no gradient, and the documented mitigation for steering's best-known failure mode was a parameter nothing ever set.

| Task | Description | Status |
|------|-------------|--------|
| **#9. VCG payments** | Exclusion set took `k-1`, so every payment was identically zero and the `clamp(min=0)` hid it. Fixed, and payments made quasi-linear transfers rather than a multiplicative haircut on rewards | Done |
| **#10. Mechanism claims** | Weighted price divided by the winner's own weight; routing share made independent of own bids; confidence heads given a per-expert value objective; capability subspace estimated and wired through to injection; README reconciled | Done |
| **#11. Routing temperature** | Gate moved to the log domain, so a uniform wealth rescaling leaves routing weights unchanged and `routing_temperature` becomes a deliberate sharpness dial. Realised top-1 weight and effective expert count now logged per step. Re-deriving the wealth bounds and decay is deliberately deferred to [#16](#phase-05--mechanism-correction) — wealth no longer sets gate sharpness, so what those constants shape is selection, price magnitude and rebate size, and judging them needs a held-out metric | Done |
| **#12. Held-out eval & baselines** | `eval_steps` was declared and never read; the reported perplexity was `exp(loss)` on the *training* batch of a stream with no split. Now: a frozen, fingerprinted held-out split (the dataset's own `validation` shard where one exists, else a stride-97 train holdout the loader skips), an evaluation loop with the economy frozen, `--router {mob,softmax,dense}` with parity asserted programmatically, and functional specialisation metrics replacing Gini — expert output divergence, routing profiles, report decisiveness. A test fails if any `TrainingConfig` field is never read. **Not** closed: the three-arm comparison is shipped as a harness with a CPU smoke run, and a real multi-seed run needs [#13](#phase-05--mechanism-correction)'s noise floor before its numbers mean anything | Done (harness); results pending #13 |
| **#13. Reproducibility** | Determinism (`torch`/`numpy`/`random`/CUDA seeded from one field, `CUBLAS_WORKSPACE_CONFIG`, `use_deterministic_algorithms(warn_only=True)`) verified bitwise-identical on GPU CI, which no longer allows `test-gpu` to fail silently; `scripts/run_seeds.py` (multi-seed mean±std) and `scripts/compare_runs.py` (delta vs. pooled noise floor); a checkpoint disk-budget floor that raises instead of filling the disk. Noise floor measured — see [Reproducibility](#reproducibility) — but at a reduced, ungated-model scale to fit this measurement session, not Phase 1's actual budget; disk-budget default is a placeholder pending a real Hephaestus number | Done (provisional-scale noise floor; disk budget placeholder) |
| **#14. Coupling activation** | Warmup default and non-vacuous tests for the `coupling.py` path that nothing currently attaches | Not started |
| **#16. Wealth bounds** | Deferred from #11: `[15, 750]` and `decay=0.997` were tuned against a gate that no longer exists and before #9 corrected the payments. The held-out metric it was blocked on now exists, and `spec/report_decisiveness` gives it a direct objective — the band is what sets how far wealth can overturn reports. Measured on the synthetic economy, the ceiling is inert (the run settles a factor of three *below* `initial_wealth`) while the floor holds up the mean — the opposite of a band chosen for the dynamics it now has. No longer blocked by #12 for the metric; still blocked by #15 for the transfer leak | Not started |
| **#15. Abstention pays** | Surfaced by #10: trained reports overestimate value because the target is clamped at zero, and loss reduction against an expert's own EMA baseline is zero-mean by construction, so winning is a loss-making trade. Also carries the **choice of rebate divisor** — `w_max` is safe but under-rebates by over 96% in a `max_wealth` monopoly; the harmonic mean of the *k* richest wealths is report-independent, feasible and tighter. Routed to [Phase 2](#phase-2--economy-stabilisation); #12's harness is in place, so the remaining block is #13 | Not started |

**Why before Phase 1:** every Phase 1 ablation is measured on top of the routing this phase prices and the baselines it establishes. Correcting the mechanism afterwards invalidates whatever was collected in between.

---

<a name="phase-1--steering-economy-coupling"></a>

### Phase 1 — Steering–Economy Coupling

*"The mind must influence the body, not just observe it."*

This is the **single highest-impact architectural change**. Currently MoB and Steering are parallel systems — MoB routes tokens, Steering corrects the output afterward. In TAME, the morphogenetic goal doesn't just *fix* deviations — it *shapes which cells activate in the first place*.

| Task | Description | Status |
|------|-------------|--------|
| **1a. Inject steering into confidence** | Modify `ConfidenceHead` so steering alignment modulates expert bids: $\text{bid}_i = c_i \times W_i \times (1 + \beta \cdot \cos(E_i(h),\, v_{\text{steer}}))$ — experts that move the representation *toward* the goal bid higher | Partial — `coupling.py` implements the perception-mode coupling and `MixtureOfBidders.attach_coupling` wires it, but nothing attaches it at runtime |
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

All Rights Reserved — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built as a practical exploration of bio-inspired AI architectures.</sub>
</p>
