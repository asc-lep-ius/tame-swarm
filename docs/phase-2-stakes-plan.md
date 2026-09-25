# Phase 2 — Stakes: the pivot plan

Milestone: *Phase 2: Stakes* on gitlab.hephaestus. Decided 2026-09-16.

## Decisions

1. **Primary goal: a cognition vehicle.** The project tests whether a system whose continuation depends on its own performance shows the behavioural signatures of agency, measured against control arms. It is not a capability project. Making a sentient system is not on the roadmap; the developmental protocol is recorded as a possible later phase (#49) and designed for so that adopting it later needs no rework.
2. **Substrate: hybrid, with no frozen cortex and a collective core.** The language-model body keeps training (the MoB experts already do). A frozen **copy** of the base exists only as the measurement floor and is never served or trained. The viability core is a tissue of cells with a shared integrator — the existing homeostat pattern — not a single controller. A stub (#48) records the small clean non-LLM substrate as a later experiment.
3. **Stakes: a truth-first band, with the expert-level dial first.** The organism's viability band is calibration, accuracy on data newer than the last update, and abstention quality, each a margin against the frozen copy, plus deference. Budget is graded, with a floor and dormancy-not-deletion. The `persistence_coupling` switch is implemented once for the experts (#39) and reused for the whole (#44).
4. **Milestone 1.5 finishes first:** #31, #30, #32; #33 is decided on #32's dose result.

## Design notes that follow from the decisions

- *No frozen cortex* means every viability metric is a margin against a frozen copy kept only for evaluation. The body is live throughout.
- *Core as a collective* is already the shape of `tame/homeostat.py`: per-layer cells, each with its own error, a gain-weighted consensus, a shared integrator. Phase 2 generalises the regulated variable from "projection onto a certified direction" to "a viability margin", and the actuators from "injection strength" to "plasticity, exploration rate, and the steering setpoints". One tissue class, two uses. `homeostat.py` is over the file-size maximum, so the generalisation lands as a lift-out into `tame/viability.py`.
- *One dial, two scales.* `persistence_coupling: value | decoupled`, fingerprinted. At cell scale, `decoupled` pins wealth for allocation and makes re-entry uniform, so nothing a cell does changes whether it keeps holding tokens; the head still trains on realised value. At organism scale, `decoupled` is a constant budget. The `shuffled` control permutes the value signal so a cell perceives value that is noise about itself.
- *The constitution exists in pieces:* strategyproof payment, balanced rebate, exploration slot. Missing are re-entry (#26) and a named test group asserting the four properties together (#38).
- *Deference is a dimension of the band, not a later addition.* Setpoints are held as beliefs; a correction is evidence; compliance rises with uncertainty; asking costs budget; the band includes the constitution (#47). What is deferred is the outward content of care, which the project cannot yet measure honestly.

## The issues

| # | Item | Type | Depends on |
|---|---|---|---|
| #36 | README states the goal, retires the "incapable of leaving" sentence, adds "what this project does not claim" | task, docs | — |
| #37 | Preregistration: four signatures, both predictions, controls, stopping rules | task, docs, measurement | #36 |
| #38 | The internal constitution as one test group, with re-entry (folds in #26) | mechanism | #26 |
| #39 | Expert-level stakes dial | mechanism, measurement | #37, #38, #32's verdict on #33 |
| #40 | Unified `WealthUpdater` with the stability analysis (roadmap 2a and 2b) | task | — |
| #41 | Rotating held-out stream newer than the last update, evaluation that costs budget, canaries | task, infra | — |
| #46 | Developmental readiness register and the human-channel rule | task, docs | #37 |
| #42 | The viability core: the tissue over margins (folds in roadmap 5b) | mechanism | #40, #41, #46 |
| #45 | Non-linguistic report channel | mechanism | #42 |
| #47 | Deference structure: beliefs, correction channel, asking, care for parts | mechanism | #38, #42 |
| #43 | Budget ledger for the whole: graded compute, floor, dormancy | mechanism | #40, #42, #46 |
| #44 | Organism-level stakes dial | measurement | #37, #39, #41, #42, #43, #45, #47 |
| #48 | Deferred: the same economy and core on a small clean substrate | stub, out of milestone | #39, #42, #45 |
| #49 | Deferred: Phase 3 — Upbringing | stub, out of milestone | #44 and a published indicator set |

Existing issues adjusted: #26 moved into the milestone and folded into #38; #33 carries the rule that a null on #32 makes it the goal term of #39's `value` arm; #8 notes that the paper's headline may come from the stakes dial rather than the coupling ablation; #18 links its sibling stub #48.

## Order of work

```
1.5 tail:  #31, #30, #32  ->  #33 decided
in parallel with the tail:  #36 -> #37 -> #46 ;  #38 (with #26) ;  #40 ;  #41
then:      #39 (the quick win)
then:      #42 (+ #45, #47)  ->  #43  ->  #44
filed now, opened later:  #48, #49
```

Every autonomy the core acquires is off by default, fingerprinted, and gated by #46's register. The stopping rules in #37 are copied verbatim into #44 before it runs.

## Where the developmental protocol lives

| Protocol step | Issue |
|---|---|
| Viability from non-rivalrous goods | #42 (truth) and #47 (deference); care for others deferred to #49 |
| Internal constitution first | #38 |
| Autonomy gated on corrigibility, schedule fixed in advance | #46 |
| Floor and exit from the start | #43 |
| No resource channel through persuading humans | #46 (the rule and its test), #41 (canaries) |
| Decide in advance what changes if signatures appear | #37 |

## Deliberately not in Phase 2

- Chunk-level routing and expert memory (roadmap Phase 3): memory amplifies whatever dynamics exist, so stakes come first.
- Lineage and population selection: #43's ledger is designed so a population can share it; the work is #49's.
- Any care-for-others metric, any capability gating beyond the training loop, any contestability channel: #49.
- Any claim about sentience. Language output is behaviour, not testimony; a Phase 2 result is a preregistered behavioural contrast and nothing more.

## Amended 2026-09-20: the readout before the budget

#39's body run (24 runs at `79c1b17`, README `#stakes-dial-cell`) read its
primary at −0.0098 [−0.0225, −0.0011] with the loss guardrail holding: not
"no", the sign against the stakes position, and the `shuffled` control on the
`value` side. The dial as measured separates a live ledger from a pinned one,
not a cell's own value from another cell's, and the contrast that would —
`value` minus `shuffled` — sits at dz ≈ 0.15 under the current readout, about
350 paired seeds a side under a paired t, 700 runs an arm. Two councils on 2026-09-20
(`~/tame-runs/39-stakes-dial/body/COUNCIL.md`, `COUNCIL2.md`) and the operator's
priorities — the vision over the schedule, and GPU-hours as the binding
constraint — settled what follows.

5. **The readout before the budget.** Every design decision, estimator choice
   and power calculation happens on the CPU fixture first; a GPU run is
   confirmatory, preregistered with a power row (`docs/preregistration.md`
   section 8, #56). Signature 1's readout is redesigned on the fixture and
   applied forward only (#57); #39's rows stay as read. "Cut cost per run" by
   shrinking knobs is rejected: the dose unit and the replicate floor are
   properties of the configuration.
6. **#44 does not run as declared.** Its primary has the shape #39 could not
   resolve; it is re-declared by #57 under section 8 before it runs, and #43's
   budget constant is derived before #44 needs it.
7. **#48 stays deferred, with its case now on the record.** The arm means of
   #39 sat within 0.003 of held-out loss on the 1.7B body, so routing is not
   load-bearing for the organism there; a substrate where it is remains the
   hand-off if #57 reads "no".

### Order of work, amended

```
now:       #57 (the readout, CPU)  |  #56 (the power rule and helper)  |  #42 (+ #45, #47)  ->  #43
then:      #44, re-declared under #57 and section 8, with its power row
filed, opened later:  #48 (the hand-off if #57 reads "no"), #49
```

## Amended 2026-09-20, second: after the pre-mortem on the redesign

A council in pre-mortem form (`~/tame-runs/39-stakes-dial/body/COUNCIL3-direction.md`)
vetoed the redesign as first written, for naming no step that keeps the CPU
gate inside its budget; the operator's answer was that testing more takes
longer and that is expected, so `GATE_BUDGET_S` moved from 110 to 180 with its
measurement note (`f23c6a3`). The seats' other objections became the plan's
own text: the routing read pins every number to a checkpoint and fixes its
threshold before running; the coupling's stress is an unsigned magnitude the
cell reduces by acting and cannot steer by reporting, with a farming
guardrail; the core carries an echo check; the perturbation readout carries a
discriminator a single lag cannot pass; #56 and #57 are a stack, not siblings.

8. **Bind the tissue before testing its stake.** #59 builds the channel the
   literature names — the tissue's error as the cell's own stress, paracrine,
   gated, charged against continuation — and measures it against today's
   attributed term on the fixture. #42 carries the organism's margin to the
   cells through it. #44 runs only on a bound tissue.
9. **Make the self-model load-bearing before testing self-reference.** #60:
   more cells than slots, and a scored self-prediction under the constitution.
   Both of #60's readings were withdrawn on 2026-09-21 and the scored
   self-prediction moved to #66, to be rebuilt on dense counterfactual targets;
   README `#self-model` carries what survives. #47 and #44 follow it.
10. **Measure whether routing matters before building a substrate where it
    would.** #58 reads #39's checkpoints; #48 opens only on its line.
11. **The body configuration is a fork, and keeping it is an option.** Three
    causes keep the organism from caring about its cells: an objective whose
    shape does not need routing (a corpus knob), cells that own too little of
    the output (rank, upcycling, a smaller body), and a goal the organism is
    indifferent to (#42's margins as setpoints, actuators through cells only).
    #58's fragile fraction and adapter footprint inform the choice and do not
    make it; #60 carries the options with their costs, the current
    configuration first, and the operator settles it before any body
    confirmation. Section 8 rule 5 names the corpus as a knob.

### Order of work, amended again

```
now:      gate budget (f23c6a3); #39 close-out (verdict, note, !40 -> !39 -> main)
then:     #56 -> #57 (one stack, off 39-gpu-arms-read)   |   #58 (own branch, forward passes)
then:     #59  |  #60  (off the #57 stack)
then:     #42 (with #59) -> #43
then:     #47, #45 (with #60)
last:     #44, re-declared under #57 and section 8, on a bound tissue
reserve:  #48, on #58's line and #57's reading
```

## Amended 2026-09-23, third: signal before stakes

Every null Phase 2 has read so far was an arm contrast on an aggregate at 1×
contribution scale, and #57 measured that such a number at six seeds is one
draw from a range spanning three orders of magnitude. #59's coupling run
resolved nothing at power 0.256 (three of its four positive claims withdrawn);
#60's grid measured the wealth band rather than the cells' share of the output
and both its readings are withdrawn (README `#self-model`,
`~/tame-runs/60-self-model/WITHDRAWN.md`). The operator's priorities stand —
the vision over the schedule, GPU-hours the binding constraint — so a
milestone was inserted before any stake is re-tested: **Phase 2.5, Signal
Before Stakes** (milestone 5, label `phase-2.5`, #62–#69 filed 2026-09-21 and
#73 the same evening). Its job is a readout with the shape of the entity
criterion, calibrated before it is pointed at an arm.

12. **A placement, not a verdict, is what a stakes readout returns.** #68
    writes the ladder into `docs/preregistration.md` section 9 — physical
    blockade (#63), setpoint step (#59's readout), payment change (#39's dial),
    each with its intervention, dose, window, criterion and trial count fixed
    before the first trial — and every stakes readout after it records two
    numbers: the lowest rung that reliably moved the collective and the highest
    that failed. Section 8 gains two rules beside it: a readout recovers a
    planted effect of known sign at six seeds and is null-calibrated on split
    same-arm runs before it reads an arm contrast (rule 7), and a cell whose
    ledger sits on a bound is a different economy and is not read (rule 8).
    "Is there a self at the cell scale" is retired as a question.
13. **#59's and #60's 1× readings stand as measurements; their decisions are
    taken at 2×.** #59's calibrated primary (power 0.256), its saturated-regime
    reading and its four secondaries are carried to #67 as predictions and
    re-read at 49 paired seeds a side. #60's 1× null at 4, 8 and 16 cells
    stands; its scored self-prediction is rebuilt under #66 on dense
    counterfactual targets, in the skill form, after #70's fix. Neither runs
    before #73 has derived the exchange rate: `reward_scale` per configuration
    from the settled economy's own inflow and charge, static and fingerprinted,
    returning the recorded 2.0 within 1e-6 at 1× with the economy bitwise the
    one recorded (the fork taken over a running
    normaliser on realised value, for the reasons #73 carries).
14. **#42 is re-gated on #63.** The core is built on a cell scale that has
    passed individuation — two structurally different blockades toward the same
    restored state, inside a window shorter than adaptation — or on the record
    that it has not, which is #48's hand-off. #42 follows #63 and #67, not #59
    alone.
15. **#44's power row comes from #69.** The body pair #69 runs (a memory smoke,
    one exploratory pair, a priced confirmation only if it fits the 15 GPU-hour
    ceiling) is where the fixture-to-body discount section 8 rule 3 calls an
    assumption is measured for the ladder's readouts; #44 sources its Power
    and Calibration rows from it and does not run before it.

### Order of work, amended a third time

```
now:      #68 (the ladder and the rules, docs)  ->  #62 | #63 at 1× (CPU, parallel)
then:     #73 (the exchange rate)  ->  #64  ->  #65 if #62/#64 gate it in
then:     #66 | #67 at 2× (after #73, and #70's fix for #66)
then:     #69 (one priced body pair; on hold until #64 reads "one lever")
then:     #42 (gated on #63)  ->  #43  ->  #47, #45
last:     #44, on a bound tissue, with #69's power and calibration rows
reserve:  #48, on #58's line and #63's "no" at 2×
```

## Amended 2026-09-25, fourth: the criterion, the redundancy fixture and #65's gate

#63's 1× rows (README `#individuation-blockade`) read two things the plan had
not settled: what "the same restored state" means, and whether a substitute
that *can* do the blocked cell's work is ever chosen. The operator settled the
first on 2026-09-24 and added a stage for the second; #73 landed the exchange
rate the same week (README `#ledger-stability`).

16. **"The same restored state" is the born-without target.** A substitution
    counts when the on-type loss inside `W` reaches what a collective born
    without the blocked cell settles at — what the remaining means allow —
    not the pre-block loss (unsatisfiable by construction on a
    planted-competence fixture) and not uptake alone (which every null arm
    passes identically). `scripts/blockade.py --stage targets` measures it per
    arm and seed and `summarise` reads every blocked window against it; the
    1× records re-read under it are labelled post hoc (section 8 rule 2).
17. **Stage 5 of #63 is the redundancy fixture**, at 1×, before any 2× row:
    the quality fixture with a second cell at the top competence
    (`REDUNDANT_COMPETENCE`), of which the live ledger's seniority seats only
    one beside the 0.7 — so a cell that can do the work sits at the floor and
    the question the recorded fixtures cannot ask is whether the auction hands
    the freed slot to it. Its predictions are in
    `docs/prereg/63-individuation.md` under "Stage 5", written before it ran.
18. **#65 is gated in by #63's non-return under the ledger pin** (2 of 24
    quality seeds, 13 of 24 differentiated), which supersedes the seniority
    gate #62 was to provide; it runs after #73, whose derivation sizes the
    exchange rate and the exploration gift together.
19. **Every 2× row runs at #73's derived rate** — quality ×2 at 0.503,
    differentiated ×2 at 0.448 and ×4 at 0.103 on seeds 0–2
    (`~/tame-runs/73-exchange-rate/derivation_*.json`; README
    `#ledger-stability` once #73's branch lands) — and never at the hand-set
    constant, which is the pairing every derived row is reported beside. The floor guardrail is not met on the quality fixture at 2× (the
    shut-out cells' rebate does not scale with the winners' flows, and their
    root rises two to four credits above the floor); the ceiling guardrail is,
    on both fixtures, and a 2× row reads that column before its contrast.

### Order of work, amended a fourth time

The placement of #63's 2× rows after #66 and #67 below is the implementing
session's proposal (the operator's instruction of 2026-09-25 named that order
for the run), not a recorded operator decision on the milestone; the rows can
run as soon as #73 lands and nothing here gates them.

```
done:     #68, #62, #63 at 1× (both fixtures), #73, #63 stage 5
now:      #64 (the counterfactual-route read at 1/2/4×, at the derived rate)
then:     #65 (gated in by #63's non-return)  ->  #66 | #67 at 2×  ->  #63's 2× rows
then:     #69 (one priced body pair; on hold until #64 reads "one lever")
then:     #42 (gated on #63)  ->  #43  ->  #47, #45
last:     #44, on a bound tissue, with #69's power and calibration rows
reserve:  #48, on #58's line and #63's "no" at 2×
```
