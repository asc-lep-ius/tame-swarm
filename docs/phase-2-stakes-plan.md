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
