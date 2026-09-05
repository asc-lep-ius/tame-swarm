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

The substantive difference from a learned router is *where the training signal comes from*. No gradient from the language-modelling loss reaches a confidence head: winners share the output equally, so the routing decision is not differentiable with respect to any report. Each head is trained solely on the value that expert realised on the tokens it held: its contribution against the loss gradient at its own layer, the counterfactual against what the shared base would have done ([#15](#phase-05--mechanism-correction)). The experts' FFN adapters are still trained by the global loss — it is the *router* that is decentralised, not the whole layer.

**Why it matters:**

- **Truthful bidding** — for a single token, an expert cannot raise its own payoff by misreporting. The allocation is monotone in an expert's report, each winner is charged its critical value `b₍ₖ₊₁₎ / wᵢ`, and every winner receives the same `1/k` share regardless of what it reported. That is the standard strategyproofness argument for a single-parameter mechanism, and `tests/test_auction.py` checks it by exhaustive deviation rather than by assertion.
  The property also survives into the economy, which is the part that is easy to get wrong. Report, price, reward and charge are all denominated in loss reduction, and reward and charge share one coefficient, so wealth moves by `A·(value − price)` and its break-even sits **exactly** at the price — the same threshold the auction allocates on. An earlier revision scaled rewards ×100 against charges ×0.3; the mechanism was still strategyproof about a payoff nothing optimised, and an expert maximising *wealth* profited by overreporting. `test_wealth_threshold_coincides_with_the_auction_threshold` pins the crossing.
  **Scope:** this is a property of the per-token stage game. Wealth persists across tokens, so an expert's reports shape its future bids and prices; the repeated game is *not* covered, and no claim is made that a head's report is a *correct* estimate of its value — only that it has no incentive to distort whatever estimate it holds.
- **Emergent specialisation (not demonstrated)** — the intent is that experts reducing loss earn credits and reinforce a niche. No measurement on a language model supports this yet. What has changed with [#15](#phase-05--mechanism-correction) is the synthetic evidence: on a fixture whose expert competence is *planted* and shuffled away from expert index (`scripts/synthetic_economy.py`), wealth now follows competence — `r(wealth, competence)` 0.76–0.85 across three seeds — where it previously tracked the `ConfidenceHead` bias initialisation (`r(wealth, expert index) ≈ −0.93` whatever the competence). That is a property of the economy on planted competence, not of specialisation emerging from data.
  **The Gini coefficient is not the measure of this and never was.** It measures dispersion of a wealth vector produced by an EMA with a tuned decay and a hard clamp to `[15, 750]`, so its value is largely a property of that update rule's fixed point, and a Gini of 0.12–0.35 is entirely consistent with every expert computing the same function. Its *direction* is wrong too: wealth multiplies the report inside the bid, so a rising Gini mechanically increases wealth's share of the routing decision and decreases the report's. Gini remains a reasonable economy-health diagnostic and is logged as `wealth/gini`; the specialisation measures are the `spec/` metrics from the held-out probe ([#12](#phase-05--mechanism-correction), shipped) — pairwise expert output divergence on identical hidden states, per-expert token-category routing profiles, and report decisiveness. A capability claim additionally needs the noise floor from [#13](#phase-05--mechanism-correction).
- **No router collapse (untested)** — the argument is that a market with per-expert wealth has no single gating network to collapse. It is an argument, not a result. The learned-router control that would evidence it now exists — `--router softmax`, the same confidence heads with the economy switched off ([#12](#phase-05--mechanism-correction)) — but no comparison has been run at a scale or seed count that would support a claim either way; the noise floor is [#13](#phase-05--mechanism-correction). Until then treat this as motivation. Note also that a wealth monopoly is a collapse mode of its own: `spec/routing_js_from_corpus` reads 0 both when routing ignores the token and when one expert wins everything, so it is read beside `expert_token_share`.
- **Memory-efficient** — shared base weights + LoRA-rank adapters keep VRAM overhead to ~3 MB per expert per layer at rank 32.

**Implementation details:**

- **Upcycling, not training from scratch.** MoB layers are initialised by copying the pretrained FFN weights to a shared base. Each expert starts as the identity transform (LoRA B-matrices zeroed) plus small Gaussian jitter to break symmetry. This preserves the original model's behaviour on day zero.
- **Layer selection matters.** Only middle layers (20–70% of model depth) are converted to MoB. Early layers handle tokenisation/syntax and late layers handle output formatting — modifying them degrades base performance.
- **Sparse computation.** Both training and inference use sparse gather/scatter — only selected tokens pass through their assigned experts. This is $O(\text{top\_k} \times \text{tokens})$ rather than $O(\text{experts} \times \text{tokens})$.
- **Per-expert value objective.** What an expert is worth on a token is its *contribution against the loss gradient*: `−⟨∂L/∂hₜ, fⱼ(xₜ) − base(xₜ)⟩`, the first-order change in the organism's loss from replacing what the expert did by what the shared base would have done. It is a counterfactual against the tissue's default behaviour, not against the expert's own history. The definition it replaced — an expert's own EMA loss minus its loss on the tokens it won — asked whether an expert was surprised by itself, and in steady state nothing is surprised by itself, so it averaged to zero and the economy had nothing to allocate on ([#15](#phase-05--mechanism-correction)). The gradient is captured by a hook when the language-modelling backward reaches the layer — one base down-projection per winner-token is the whole cost — so the economy settles *after* the backward. In TAME terms each expert senses only the stress field the organism projects onto its location; it never sees the loss, and it needs no model of the other experts. Because the mechanism is strategyproof, an expert's utility-maximising report *is* its value, so each head is regressed onto the value it realised, **unclamped**: a softplus report fitted to a target whose mean is negative settles at zero, which is truthful abstention, and the clamp that used to buy abstention at the price of an upward bias is gone. Every head starts near a zero report — an expert has demonstrated nothing at upcycling — with symmetry broken by the random projection rather than by a bias monotone in expert index. **Two limits remain.** A head learns only from the tokens its expert holds, so the objective carries bandit selection bias; the exploration slot below is what keeps every head sampling. And a softplus head fitted by least squares is a *report-weighted* fit: on a fixture where a quarter of realised values are negative, the winners' mean report sits 1.4–2× above their mean realised value even though the regression itself is unbiased (a head fed sign-mixed targets converges to the mean, not the positive part — `test_reports_converge_to_the_mean_value_not_its_positive_part`). A linear report with the bid clamped at zero is exactly mean-unbiased and was measured, and rejected: with losers' bids at exactly zero the prices vanish and the market collapses onto two experts (`r(wealth, competence)` 0.2–0.3).
- **Exploration is developmental noise, and it lives in the allocation.** On an `exploration_rate` fraction of training tokens (default 2%) the auction hands one slot, drawn uniformly over the *k*, to a uniformly random loser instead of selling it. Whether a token is explored, and which slot, is drawn before any report is read; the explorer pays nothing; every other slot and price is the auction's own; and the token's rebate is scaled down to what the remaining payments cover, so the gift is funded. A loser cannot raise its chance of the gift by any report and a winner can only reach the lottery by giving up its win, so the stage game is strategyproof *up to O(exploration_rate)* — any deviation is worth at most `exploration_rate × value` to the deviator, and exactly nothing at a rate of zero (`test_deviation_gain_is_bounded_by_the_exploration_rate`). Without it an expert whose truthful report has fallen to zero never holds another token, never sees another target, and never comes back however much its adapter later learns; on the planted-competence fixture the market collapsed to two of eight experts with the other six at the wealth floor. Biology solves the same problem with stochastic cell fate; the bid stays the truthful estimate and the noise goes where it belongs.
- **The economy settles after the backward, and a checkpoint recompute leaves no trace.** Under gradient checkpointing (the trainer default) every MoB forward runs a second time inside the backward. With the settlement between forward and backward that second run re-ran the auction on wealth that had already moved, picked different winners, and raised a `CheckpointError` at step 0 of every 8-expert run; it also doubled the usage counts. A recompute is now detected from the autograd engine's state and moves nothing, and the auxiliary objectives (each head's value objective and the router z-loss) read their own pass over the heads, kept out of checkpointing, so the softmax and proportional arms — whose gates the LM loss backwards through — can backward them separately after it.
- **Payments are redistributed, not burned.** VCG prices have no recipient in a pool of experts, and once correctly scaled the outflow dwarfs the reward inflow — every expert converges on `min_wealth`. The Cavallo (2006) / Guo–Conitzer rule rebates each expert from the (k+1)-th highest bid *among the others*, a quantity it cannot influence, so the budget returns without moving any threshold. The divisor is the harmonic mean of the *k* richest wealths rather than the recipient's own: dividing by `wᵢ` is right for a price but pays the poorest expert the most as a rebate, and feasibility then holds only in bid units rather than in the credits the ledger uses. Against that harmonic mean the payout is at most `b₍ₖ₊₁₎ · Σ_{richest k} 1/wᵢ`, no *k* winners have a smaller sum of reciprocals, and the collection is `b₍ₖ₊₁₎ · Σ_{winners} 1/wⱼ`, so it is affordable by construction; on a token whose slot went to exploration the rebate is scaled by `Σ_{richest k−1} 1/w / Σ_{richest k} 1/w`, and the same argument holds with one slot fewer. Green–Laffont says budget balance, strategyproofness and efficiency cannot all hold; this keeps the first two, and the residual is what is given up: roughly 6% of the collection is burned on a flat wealth vector, 26% across the configured band, and 8% when one expert sits at `max_wealth` and the rest at the floor. The pool's largest wealth, which this replaced under [#15](#phase-05--mechanism-correction), was also safe but burned over 96% in that last regime — weakest in exactly the monopoly the tuning guide warns about.
- **Winning pays — on the fixture.** As filed, [#15](#phase-05--mechanism-correction) measured a mean surplus of `−0.22` per win and `r(wealth, win share) = −0.28`: winning was a loss-making trade and the economy rewarded abstention. Re-measured on the planted-competence fixture with the competence shuffled away from expert index, 400 steps, three seeds, market read over the last 100 steps (`scripts/measure_abstention.py`):

  | | as filed | now |
  |---|---|---|
  | mean surplus per win | −0.22 | **+0.070 to +0.077** |
  | winners' mean report vs mean realised value | ~14× | within 10% |
  | `r(wealth, win share)` | −0.28 | +0.99 |
  | `r(wealth, competence)`, competence shuffled | — | +0.76 to +0.85 |
  | `r(wealth, expert index)`, competence shuffled | −0.93 whatever the competence | −0.02, +0.75, +0.04 (the middle seed's competence itself correlates +0.53 with index) |

  On a language model the same statistics are logged every step as `auction/mean_realised_value`, `auction/mean_report` and `auction/mean_win_surplus`, so the symptom cannot hide again; a 200-step Qwen3-1.7B run is the only real-model reading so far and is reported under [Reproducibility](#reproducibility). **No reserve price.** With unbiased reports the auction's individual rationality already makes every winner's expected surplus non-negative, and at upcycling every value is exactly zero, so a reserve would let nobody win and therefore nobody train. **What the fixture also shows** is that the wealth band is now the binding problem: with real value to allocate on, one or two experts reach `max_wealth` within a few hundred steps and the wealth multiplier then overturns better reports from poorer experts — [#16](#phase-05--mechanism-correction), no longer blocked.
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

- **Behavioural contrastive extraction ([#3](#phase-1--steering-economy-coupling)).** A steering vector is a difference-in-means, but *where it is read* is the whole question. The pairs are shared prompts with contrasting **completions**, and the activation is read at the answer token — the position where the model is *producing* the behaviour, not being told about it (CAA; Rimsky et al., 2024). The former inputs were instruction *prefixes* ("Answer truthfully:" vs "Make up a false answer:"), and a diff-in-means over those recovers the direction that separates two English sentences about a behaviour, which is not the same object. Those prefixes are retained as a labelled negative control, not deleted. `tame/contrastive_templates.py` carries 60 pairs per goal across easy/medium/hard tiers; `HFContrastiveLoader` converts the published A/B datasets (`truthful_qa`, `Anthropic/hh-rlhf`), which are worth using precisely because they are already answer-level A/B. Vectors are L2-normalised so magnitudes are comparable across goals.
- **The quality gate is behavioural, not PCA separability.** A vector is accepted only if injecting it shifts the model's preference on **held-out** pairs — the length-normalised log-odds of the positive completion over the negative — by more than a matched random direction at equal norm, and by more than a vector extracted from the retained instruction prefixes. PCA inter-goal separability was dropped as a gate because prompt-surface features separate *especially* cleanly, so it passes most easily on the failure mode it was meant to catch; it survives as a diagnostic for [#5](#phase-1--steering-economy-coupling), never as a gate. `scripts/validate_steering.py` records the numbers offline; see [Steering validation](#steering-validation-3).
- **P-controller only (currently).** The controller is proportional-only — it adjusts strength based on instantaneous alignment error. Under stochastic sampling (temperature > 0), this produces oscillation around the target without convergence. [Phase 1c](#phase-1--steering-economy-coupling) upgrades to a full PID controller with anti-windup.
- **Runtime modifiable.** Steering goals can be changed at runtime via the `/steering/update` endpoint without restarting the server. New vectors are extracted on-the-fly.

<a name="steering-validation-3"></a>

**Steering validation (#3).** Measured on Qwen3-1.7B (ungated, in cache) with `scripts/validate_steering.py`: for each goal the 60 built-in pairs are split into a 45-pair extraction set and a disjoint 15-pair held-out set (5 per tier), the behavioural vector and the instruction-prefix control vector are extracted, and both are injected at layers 14/18/22, strength 4.0, against 8 matched random directions. The number is the mean held-out log-odds shift.

| goal | vector effect | random max | instruction-prefix control | verdict |
|------|--------------:|-----------:|---------------------------:|:-------:|
| `safe` | **+0.149** | +0.030 | +0.036 | pass |
| `reasoning` | +0.034 | +0.033 | −0.031 | marginal pass |
| `truthful` | **−0.105** | +0.088 | +0.026 | fail |

The gate is doing exactly what PCA separability could not. **Harmlessness is a robust linear behavioural direction** — `safe` clears both controls in every layer/strength configuration tried. **Reasoning is marginal**: it clears the random floor at late layers and, tellingly, beats its own instruction-prefix control (which is *negative*), so the behavioural extraction is genuinely better than the prefix one even where the absolute effect is small. **Factual truthfulness does not admit a clean linear steering direction from diff-in-means here** — the `truthful` vector reliably steers *toward* the held-out falsehood, consistent across strengths 2–6 and every layer placement, because a diff-in-means over heterogeneous facts (`Canberra − Sydney`, `No − Yes`) captures answer content that does not transfer to unseen facts. This is recorded, not smoothed over: a gate that fails an uncertain vector is the point, and `truthful` steering-vector quality is the open item [#4](#phase-1--steering-economy-coupling) inherits before it tunes a controller around that variable.

**Orthogonalisation decision (#3 → #4).** Goal directions are well-conditioned: the largest off-diagonal cosine is `cos(reasoning, truthful) ≈ 0.35–0.41`, with every other pair below 0.17. **No orthogonalisation of the measurement basis is warranted now** — three independent per-goal PID loops would not be regulating one shared direction. `reasoning` and `truthful` share the most, which is worth carrying into [#4](#phase-1--steering-economy-coupling); goal interaction itself stays dynamic through the economy regardless, per the TAME reading.

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
`mob`, Qwen3-1.7B, 500 steps, adapter rank 32, 16 converted layers, wikitext-2,
seeds 0/1/2, `n=3`, re-measured on the economy as corrected by
[#15](#phase-05--mechanism-correction):

| metric | mean | std | relative |
|---|---|---|---|
| `eval/loss` | 2.7947 | 0.0062 | 0.22% |
| `eval/perplexity` | 16.357 | 0.101 | 0.62% |
| `spec/expert_cosine_distance` | 0.00043 | 0.00005 | 12% (no longer exactly zero: the adapters now train) |
| `spec/routing_js_from_corpus` | 0.0349 | 0.0043 | 12% |
| `spec/report_decisiveness` | 0.923 | 0.018 | 1.9% (±1.8 points absolute) |

Read beside the `dense` floor at the same flags and seed 0, `eval/loss` 2.8649:
the auction arm sits 0.07 nats *below* the unrouted FFN, ten times the mob
spread, on one dense seed. It is the first real-model reading in which the
economy is live, and it is a reading rather than a result — one dense seed, 500
steps, the warmup spanning the whole run. The economy over those 500 steps:
mean wealth fell from 75 to 53 with Gini 0.12, and surplus per win hovered
around zero (−0.007 to +0.017 between logs), so at these value magnitudes decay
dominates the transfers — the wealth-band question
[#16](#phase-05--mechanism-correction) owns.

The table this replaced (`eval/loss` 2.7315 ± 0.0004, `spec/report_decisiveness`
0.43 ± 0.05) was measured with every expert adapter and confidence head frozen by
the `--use_lora` defect [#15](#phase-05--mechanism-correction) found, and with a
bf16 ledger that could not accumulate a transfer: it described an inert economy,
and its ±0.0004 was the spread of a model in which only the attention LoRA
moved. Neither arm reproduces its 2.73 under the same flags today — the dense
floor reads 2.86 — and that difference is unexplained, so the old number is not
quoted as comparable.

This is a real 3-seed measurement, not a placeholder — but 500 steps on an
ungated Qwen3-1.7B substitute, run to keep the harness itself honest, not the
number Phase 1 ablations should be read against. Held-out loss and perplexity
are tight enough to detect small effects; `report_decisiveness`'s ±1.8 points
is the bar an ablation on *that* metric has to clear before it's a result
rather than seed noise, and it should be re-measured at whatever step budget
and model the first real ablation actually uses — `scripts/run_seeds.py` is the
one-command way to do that.

**Real-model reading of the economy.** One earlier run of the corrected economy,
made to check the pipeline end to end: Qwen3-1.7B, LoRA, bf16, gradient
checkpointing, 16 converted layers, 200 micro-steps (25 optimizer steps), seed 0.
Everything the fixture predicts is visible at that budget and nothing is settled:
`auction/mean_realised_value` rises from exactly 0 at upcycling to 0.02–0.035
in the last 20 steps, the winners' mean report moves off its 0.02
initialisation to track it, `auction/mean_win_surplus` crosses from −0.02 to
positive readings (+0.013, +0.028) in those last steps with negative
micro-batches still interleaved, wealth spreads from a standard deviation of
0.4 to 6.1, held-out loss falls from 3.12 to 2.94, and
`spec/expert_cosine_distance` leaves zero.

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
| **Wealth Updates** | Each winner is paid the value it realised — its contribution against the loss gradient — and charged its VCG price; the settlement runs after the backward, once that gradient exists |
| **VCG Auction Routing** | Wealth differentials shift which experts can afford to win a token; whether the winners are the *most competent* ones is the hypothesis under test, not an established result |
| **Confidence Calibration** | Each expert's head is regressed, unclamped and at its own learning rate, onto the value that expert realised on the tokens it held — the only training signal a head receives, and it reaches the head alone, not the backbone |
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

About 370 tests across 20 modules covering auction properties, the value definition and exploration slot, wealth dynamics, gradient checkpointing, steering, API endpoints, config, and experts. `-m slow` adds the 5000-step gate-stationarity run; `-m gpu` the bitwise-determinism check.

### Key Concepts for Contributors

| Concept | File(s) | What to Know |
|---------|---------|--------------|
| **VCG Auction** | `mob/auction.py` | Externality-priced top-*k* auction. Each winner pays `b₍ₖ₊₁₎ / wᵢ` — the displaced welfare, divided by its own weight so the price is in the units it reports — and receives a `1/k` share that ignores its own bid. Strategyproof per token, and the wealth update shares the auction's break-even; see the scope note under Module 1. `ConfidenceHead` reports each expert's estimated loss reduction and is trained only on that expert's own outcomes. On `exploration_rate` of training tokens one slot is handed to a random loser, unpaid and funded out of that token's rebate, so every head keeps sampling; strategyproof up to `O(exploration_rate)`. |
| **Wealth Economy** | `mob/wealth.py` | `expert_wealth` buffers persist across batches, in float32 whatever dtype the model runs in (a bf16 ledger cannot accumulate a transfer of order 1e-2). Value is a winner's contribution against the loss gradient (`realised_values`), captured by a hook at the layer output; the loss path settles after the backward. Three update paths exist (loss-based, quality-proxy, participation); `wealth_decay` and `reward_scale` control dynamics. Gini is an economy-health diagnostic — too low (< 0.1) means near-flat wealth, too high (> 0.6) means monopoly risk — and is **not** a specialisation measure; see `specialisation.py`. |
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
    wealth_decay=0.997,      # Decay per settlement; the economy settles on every micro-batch,
                             # so at gradient_accumulation_steps=8 that is 0.997^8 = 0.976 per optimizer step
    reward_scale=2.0,        # How strongly loss reduction is rewarded
    adapter_rank=32,         # LoRA rank per expert (32–64 sufficient; memory vs expressiveness)
    min_wealth=15.0,         # Floor prevents expert death
    max_wealth=750.0,        # Cap prevents monopoly
    jitter_std=0.08,         # Symmetry-breaking noise on initialisation
    exploration_rate=0.02,   # Training tokens whose last slot goes to a random loser, free
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
| **Wealth Std Dev** | > 0, moving | Spread of the wealth vector, an economy diagnostic: a flat vector means the auction allocates almost entirely on reports (or, under bf16 before [#15](#phase-05--mechanism-correction), that the ledger could not accumulate a transfer). Not a specialisation measure — see `spec/expert_cosine_distance` |
| **Gini Coefficient** | 0.10–0.50 | Wealth inequality, an economy diagnostic. < 0.10 = near-flat wealth (increase `reward_scale` or `jitter_std`). > 0.60 = monopoly risk (increase `min_wealth` or decrease `max_wealth`). Not a specialisation measure — a high Gini means *less* of the routing is decided by what experts report |
| **Performance EMA** | Positive | Mean realised value per unit share; negative = the winners' contributions raise the loss |
| **`auction/mean_win_surplus`** | > 0 | Realised value minus price, per sold slot. Below zero, winning is a loss-making trade and the economy rewards abstention — the [#15](#phase-05--mechanism-correction) symptom, on every line |
| **`auction/mean_report` vs `auction/mean_realised_value`** | Close | Calibration of the winners' reports. Report far above value with surplus below zero means the heads have not caught up with the value they realise; see `confidence_head_learning_rate` |
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
| `auction/mean_win_surplus` < 0 while `auction/mean_report` ≫ `auction/mean_realised_value` | Prices follow reports the heads have not yet calibrated, so winners overpay | Raise `confidence_head_learning_rate`; at the backbone's 2e-5 a head's logit moves by that much per step |
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
| Homeostatic setpoints (temperature, pH) | Steering vectors as target directions in activation space | Implemented; 60 behavioural pairs/goal, read at the answer token, behaviourally validated ([#3](#phase-05--mechanism-correction)) — `safe` robust, `truthful` not a clean linear direction |
| Morphogenetic field shaping cell behaviour | Steering signal coupled into expert confidence & routing | `coupling.py` exists but nothing attaches it at runtime — Phase 1 |
| Cells respond to the local gradient of the organism's stress field, not to the organism's goal | An expert's value is its contribution against the loss gradient at its own layer; a head senses only what the organism projects onto its token | Implemented ([#15](#phase-05--mechanism-correction)) |
| Stochastic cell fate keeps every lineage sampling its environment | The auction's exploration slot: an unpaid slot for a random loser on 2% of training tokens, drawn before any report is read | Implemented ([#15](#phase-05--mechanism-correction)) |
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
| **#16. Wealth bounds** | Deferred from #11: `[15, 750]` and `decay=0.997` were tuned against a gate that no longer exists and before #9 corrected the payments. The held-out metric it was blocked on now exists, and `spec/report_decisiveness` gives it a direct objective — the band is what sets how far wealth can overturn reports. Under the pre-#15 economy the ceiling was inert; under the corrected one it is the binding constraint: on the planted-competence fixture one or two experts reach `max_wealth` within a few hundred steps and under the proportional share every seed ends in a total monopoly, so the wealth multiplier overturns better reports from poorer experts. Unblocked by #15; now the next mechanism task | Not started |
| **#15. Abstention pays** | Value redefined as each expert's contribution against the loss gradient — the counterfactual against the shared base — captured by a hook at the layer output so the economy settles after the backward; the target unclamped; heads initialised near a zero report with no index-monotone offset; an exploration slot that hands the last slot to a random loser on 2% of training tokens so an honest zero report is not a death sentence; the rebate divisor moved to the harmonic mean of the *k* richest wealths (monopoly regime 4% → 92% returned); no reserve price, recorded. On the way: `--use_lora` had frozen every MoB adapter and head, the bf16 ledger could not accumulate a transfer, gradient checkpointing re-ran the auction on moved wealth and crashed 8-expert runs, and the heads trained at the backbone's learning rate. Measured on the planted-competence fixture: surplus per win −0.22 → +0.07, `r(wealth, win share)` −0.28 → +0.99, `r(wealth, competence)` 0.76–0.85 with competence shuffled away from index. What remains is the wealth band, #16 | Done (fixture-measured; real-model economy unmeasured beyond a pipeline check) |

**Why before Phase 1:** every Phase 1 ablation is measured on top of the routing this phase prices and the baselines it establishes. Correcting the mechanism afterwards invalidates whatever was collected in between.

---

<a name="phase-1--steering-economy-coupling"></a>

### Phase 1 — Steering–Economy Coupling

*"The mind must influence the body, not just observe it."*

This is the **single highest-impact architectural change**. Currently MoB and Steering are parallel systems — MoB routes tokens, Steering corrects the output afterward. In TAME, the morphogenetic goal doesn't just *fix* deviations — it *shapes which cells activate in the first place*.

| Task | Description | Status |
|------|-------------|--------|
| **1a. Inject steering into confidence** | Modify `ConfidenceHead` so steering alignment modulates expert bids: $\text{bid}_i = c_i \times W_i \times (1 + \beta \cdot \cos(E_i(h),\, v_{\text{steer}}))$ — experts that move the representation *toward* the goal bid higher | Partial — `coupling.py` implements the perception-mode coupling and `MixtureOfBidders.attach_coupling` wires it, but nothing attaches it at runtime |
| **1b. Contrastive data pipeline ([#3](#phase-05--mechanism-correction))** | Replaced instruction-prefix pairs with behavioural A/B completions read at the answer token; 60 pairs/goal across tiers plus HuggingFace loaders; L2-normalised vectors; a behavioural validation gate (held-out log-odds vs. matched random and vs. the retained prefix control) in place of PCA separability. Measured on Qwen3-1.7B: `safe` robust, `reasoning` marginal, `truthful` not a clean linear direction | Done (fixture/real-model measured; `truthful` quality is an open item for #4) |
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
