<!-- ──────────────────────────────────────────────────────────────────────────
     MEASUREMENT
     A question the code answers with a number: an ablation, a sweep, a
     checkpoint, a replication, a calibration.
     For a change to what the organism does → mechanism.
     For code that does not do what it says → defect.
     For docs, CI and tooling → task.
     ────────────────────────────────────────────────────────────────────────── -->

## Question

<!-- One sentence a number can answer, and the value of that number that answers
     it "no". Written before anything runs.
     Example: "Does a goal injected at the certified strength during training move
     routing toward goal-aligned experts? A win-share contrast inside one pooled
     spread says no." -->

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
| Seeds | <!-- ≥ 3 per arm for any quoted number (#13); state what that resolves --> |
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

<!-- The value that reads as "no", and what happens then: which constant, mechanism
     or paper it selects. A measurement that cannot come back "no" is telemetry. -->

## TAME reading

<!-- The number's meaning at the scale it is measured — cell, tissue, organism —
     and which competency claim it bears on: sensing, acting, a setpoint, recovery.
     A claim about the tissue must not be tested on a cell. Say what the result
     could NOT say about the scale above it. -->

## Traps that apply

<!-- Tick what applies; each has cost a re-run before (README, "measurement traps").
     Add any new one this measurement is exposed to. -->

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
- [ ] Every headline number as mean ± pooled std, the resolution stated, the run directory named
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

## Affected module(s)

- [ ] `tame/mob` (auction, wealth, core, routing trace)
- [ ] `tame/homeostat`, `homeostat_calibration` (the tissue, the loop)
- [ ] `tame/steering`, `steering_pipeline`, `contrastive_data` (directions, certification)
- [ ] `tame/coupling` (perception modulation)
- [ ] `tame/specialisation`, `evaluation` (held-out probe)
- [ ] `tame/train`, `parity`, `metrics` (training arms)
- [ ] `tame/app`, `routes`, `observability`, `outcome_probe` (served surface)
- [ ] `scripts/` (measurement harness)
- [ ] `tests/`
- [ ] README / docs
- [ ] CI

---

/label ~measurement
