<!-- ──────────────────────────────────────────────────────────────────────────
     MEASUREMENT
     A question the code answers with a number: an ablation, a sweep, a
     checkpoint, a replication, a calibration.
     For a change to what the organism does → mechanism.
     For code that does not do what it says → defect.
     For docs, CI and tooling → task.
     Delete the sections that do not apply; a blank section is worse than none.
     These templates are revisited after the Phase 1.5 issues close: the first
     seven uses decide what stays.
     ────────────────────────────────────────────────────────────────────────── -->

## Question

<!-- One sentence a number can answer, and the value of that number that answers
     it "no". Written before anything runs.
     Example: "Does a goal injected at the certified strength during training move
     routing toward goal-aligned experts? A win-share contrast inside one pooled
     spread says no." -->

**Mode:** <!-- exploratory | confirmatory. Exploratory: a fixture sweep, a curve,
     a look — the plan may move as it runs and the result is a hypothesis.
     Confirmatory: the plan below is fixed before the run and the result is a
     verdict. A GPU ablation is confirmatory; the fixture pass before it is not. -->

## What is already known, and the premise that could be wrong

<!-- The measurements this rests on, with their README anchors, and the reading
     of them this issue takes for granted. Say which premise a surprise would
     overturn. #25 was filed on the premise that expert_cosine_distance = 0.00044
     meant an undifferentiated tissue; the metric was diluted by the shared base and
     the experts' contributions differed by 0.5. -->

## Substrate and configuration

| Field | Value |
|---|---|
| Substrate | <!-- real model (id, converted layers, rank, seq length) / wired tiny fixture / quality fixture / differentiated fixture (type_signal) --> |
| Arms | <!-- what differs between arms, and what is asserted at parity (fingerprint) --> |
| Variation per run | <!-- which sources vary between runs — init seed, data order, the kernels that are not deterministic — and how many runs. At the ablation configuration a seed pins the data and not the trajectory, so runs, not seeds, are the unit; ≥ 3 per arm for any quoted number (#13) --> |
| Budget | <!-- steps; and memory horizons at the decay, if the economy is read (#16) --> |
| Command | <!-- the exact `uv run python scripts/…` invocation, so a re-run is a copy --> |
| Code | <!-- commit SHA the numbers come from; the fingerprint does not carry it --> |
| Cost | <!-- GPU-hours or CPU-minutes, stated before running; the GPU is one box --> |

## Noise floor and resolution

<!-- What every delta is read against. At the ablation configuration the floor is
     run-to-run, not seed-to-seed: two runs at identical fingerprints differ as
     much as two seeds do (#25). Quote the pooled spread and the resolution the
     seed count buys (three a side resolves ~3 spreads); never a p-value alone. -->

## What would falsify it

<!-- One primary contrast, declared here, decides the question. Everything else is
     reported and decides nothing: with fifteen rows the largest |delta / spread|
     under the null is about two, so "the biggest movement" is not a result.
     A measurement that cannot come back "no" is telemetry. -->

- **Primary contrast (one):** <!-- the single number, and the value that reads "no" -->
- **Guardrails (must not move):** <!-- what the intervention may not cost; the number that shows it did -->
- **Secondary (reported, not decided on):** <!-- every other row in the table -->

## Deviations from the plan

<!-- Filled BEFORE results are read, whenever anything above changed after the
     run started: the metric, the budget, the arms, the code. #25 changed its gate
     metric after looking at the fixture; that was right, and it belongs here as a
     deviation, not rewritten into the plan. Empty means the plan held. -->

- 

## TAME reading

<!-- The number's meaning at the scale it is measured — cell, tissue, organism —
     and which competency claim it bears on: sensing, acting, a setpoint, recovery.
     A claim about the tissue must not be tested on a cell. Say what the result
     could NOT say about the scale above it. -->

## Traps that apply

<!-- Keep the lines that apply and delete the rest; each has cost a re-run before
     (README, "measurement traps"). Add any new one this measurement is exposed to. -->

- [ ] A fixed step budget is a different number of memory horizons at each decay
- [ ] The two wealth clamps need different tolerances
- [ ] A sweep that moves two things together is collinear
- [ ] The exploration gift (`exploration_rate / top_k / (n − top_k)` = 0.0017) is not a market share
- [ ] `expert_cosine_distance` is diluted by the shared base; read the contribution metric beside it
- [ ] A Pearson contrast is blind to a constant injection; read the win-share contrast beside it
- [ ] Generating a corpus or running a probe on an unfrozen economy moves the economy
- [ ] The tiny fixture's served strength saturates the gate (alignment 0.998)
- [ ] Concurrent GPU load; a run and the GPU test job on the same box
- [ ] A `--use_lora` checkpoint holds no MoB expert adapters or heads: nothing can be re-probed later

## What it records

- [ ] The README block (anchor: `#…`) **extended, not replaced** — the earlier number stays as the measurement it was
- [ ] The primary contrast with a bootstrap interval on the paired per-run deltas; every other row as mean ± pooled std; the run directory named
- [ ] The script and flags committed; no stored number quoted where a re-run was possible
- [ ] The verdict, and what it selects (#8's gate row, a constant, the next measurement)

## Dependencies

| Relationship | Issue |
|---|---|
| Follows | <!-- #N or n/a --> |
| Feeds | <!-- #N or n/a --> |
| Hands off to, if "no" | <!-- #N or n/a --> |

## Acceptance criteria

- [ ] 
- [ ] 

## Result and decision

<!-- Filled at close, in the issue, so the loop closes where the question was
     asked: the primary contrast's value and interval, "yes" or "no", what it
     selects, and the follow-up issues it opened. The README block carries the
     table; this carries the verdict. -->

## Touches

<!-- One line: the modules and scripts this lands in, e.g. `tame/train.py`, `scripts/run_seeds.py`, README `#…`. Labels carry the type. -->


---

/label ~measurement
