<!-- ──────────────────────────────────────────────────────────────────────────
     MECHANISM
     A change to what the organism does: the auction, the ledger, the tissue,
     the coupling, the value definition, a constant that is load-bearing.
     For a question a number answers → measurement.
     For code that does not do what it says → defect.
     For docs, CI and tooling → task.
     Delete the sections that do not apply; a blank section is worse than none.
     These templates are revisited after the Phase 1.5 issues close: the first
     seven uses decide what stays.
     ────────────────────────────────────────────────────────────────────────── -->

## Claim

<!-- What the organism will do differently, at which scale, in one sentence.
     "Cells" are experts, "tissue" is the homeostat's cells over the layers,
     "organism" is held-out behaviour. State the claim at the scale it is made
     and the smaller scale it is implemented at; they are not the same thing.
     Example: "The tissue's goal recruits cells: with the goal present, routing
     moves toward goal-aligned experts by more than the pooled spread." -->

## TAME reading

<!-- Multi-scale competency, spelled out:
     - What does the cell SENSE (its local field), and what does it need to know?
       It must not need the global goal — competence without comprehension.
     - What does it ACT on, and what is its setpoint? A goal is a setpoint that
       changes what counts as stress for the cell, not a stimulus it is shown.
     - How LEGIBLE is the field to the cell? A cell that cannot tell where it is
       competent is a hazard in proportion to its skill: at half the fixture's
       default legibility the most competent expert is the one ruined (#25).
     - What does the collective do that no cell does?
     - Which failure is this a defence against (a cell that profits regardless of
       context; a cell that cannot sense where it is competent; a cell shut out)? -->

## Mechanism

<!-- The change in the algebra, not in prose: the update rule, the price, the
     identity that holds afterwards and the invariant that must survive (routing
     reads relative wealth only; prices are ratios; the reward is absolute; the
     probe never pays the economy). Name the constant it introduces and what sets
     its scale. -->

## Pairing

<!-- Every test can fail: the inert state this mechanism is measured against.
     kp = ki = 0 for the loop; β = 0 or a zero-norm receptor for the coupling; a
     flat band for the ledger; the quality fixture for a specialisation claim.
     A test that passes with the mechanism disabled is not a test of it. -->

## What it invalidates

<!-- Which recorded numbers, constants and identities stop being comparable, and
     where they are recorded. A change to the value definition moves #15's
     correlations, #6's recovery claims, the three expected failures and #16's
     tables at once. Say which are re-measured here and which are kept on the old
     mechanism, named. -->

## Scope

<!-- Files and rough size, per piece. Fixture first, real model last: a mechanism
     is measured on the planted economy or the wired tiny system before it costs
     GPU time, and the fixture number is what the test pins. -->

- `tame/…` — <!-- what, ~LOC -->
- `scripts/…` — <!-- the measurement, ~LOC -->
- `tests/…` — <!-- the paired test, ~LOC -->

## Risks and misuse

<!-- What the mechanism could reward that it should not (a goal in the price
     invites experts to farm the goal signal instead of the loss). The guard,
     and the number that would show the guard failing. -->

## Guardrail metrics

<!-- The numbers the mechanism may not move, each with the bound: held-out loss
     within the injection's own cost; the term exactly zero with the field
     absent; the quality fixture's r(wealth, competence) unchanged. A mechanism
     that improves its own metric and moves a guardrail has not shipped. -->

- 

## Out of scope

- 

## Dependencies

| Relationship | Issue |
|---|---|
| Follows | <!-- #N or n/a --> |
| Blocks | <!-- #N or n/a --> |
| Gated on | <!-- the measurement that has to come back "no" first, if any --> |

## Acceptance criteria

- [ ] The claim is measured on a fixture with its inert pairing, thresholds pinned from a recorded measurement
- [ ] Every new config field is read somewhere outside its own validation (`tests/test_no_silent_noops.py`, `tests/test_training_config_usage.py`)
- [ ] Every new field that changes training is in `ArmFingerprint`, or declared in `parity.NOT_A_CONFOUND` with its reason
- [ ] What it invalidates is re-measured, or kept on the old mechanism by name
- [ ] The README section for the mechanism written or extended, with the TAME reading and the number that would have falsified it
- [ ] 

## Touches

<!-- One line: the modules and scripts this lands in, e.g. `tame/train.py`, `scripts/run_seeds.py`, README `#…`. Labels carry the type. -->


---

/label ~mechanism
