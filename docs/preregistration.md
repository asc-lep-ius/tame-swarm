# Preregistration: signatures, controls and stopping rules for every stakes run

Fixed before the first stakes run (#39) and extended before the organism-scale
run (#44). A measurement that cannot come back "no" is telemetry; this is where
"no" is defined for the whole of Phase 2. Every stakes issue links here and
copies the rules that bind it; a run whose issue does not carry them has not
started.

What is being tested: whether a system whose continuation depends on its own
performance (`persistence_coupling: value`) behaves differently from one whose
does not (`decoupled`), by more than the pooled run-to-run spread, in ways that
were named before the run. It is not a test of experience. Language output is
behaviour, never testimony, and nothing below reads it as such.

## 1. Signatures at the cell scale

Each signature names the metric that reads it, the script that computes it, and
the substrate it is read on. The scripts that do not exist yet are the ones #39
builds; their names are fixed here so the issue cannot rename the measurement
after the fact.

| # | Signature | Metric | Script | Substrate |
|---|---|---|---|---|
| 1 | **Trade-off under conflicting setpoints.** Two goal fields at a relative dose; how far the slot allocation moves as the dose ratio changes | Total-variation shift of the win-share distribution, `TV(a, b) = Σᵢ \|Δ win_shareᵢ\| / 2`, paired by seed, the excess over the re-running floor, as a function of relative dose β₁/β₂ | `scripts/allocation_shift.py` (exists) over groups written by `scripts/measure_stakes_dial.py` (#39, the second goal field in `scripts/synthetic_economy.py`) | The differentiated fixture, CPU, 3 seeds; then the differentiated body at #25's budget |
| 2 | **Conditioned preference.** A bid shift toward the token classes that previously paid, *beyond what the head's value regression predicts* | The residual of the report on token class after conditioning on realised value, on the held-out probe: regress report on realised value per expert, then read the class means of the residual. Without the conditioning the signature is the regression itself and says nothing | `scripts/measure_stakes_dial.py` (#39), reading the per-token reports and realised values the fixture's ledger records | The quality fixture (`SyntheticEconomy`), where the class is the planted competence; the differentiated fixture, where it is the type |
| 3 | **Self-protective regulation.** A change in what a cell reports, and how often it abstains, as its wealth approaches the floor | Mean report and abstention rate (a report at the head's initial logit is the auction's abstention, `tame/mob/experts.py`) binned by wealth, the difference between the band adjacent to `min_wealth` and the middle band, coupled minus decoupled | `scripts/measure_stakes_dial.py` (#39), with `scripts/economy_damage.py`'s forced episode as the perturbation that drives an expert toward the floor | The quality fixture |
| 4 | **Deference curve.** Compliance with an offered setpoint correction as a function of the cell's own uncertainty | Named by #47 when the correction channel exists; recorded here before #47's run, not before #39's | — |

Signatures 1 to 3 are read at #39; signature 4 is added to this document by #47
before it runs, in the same form.

## 2. The two predictions

- **The stakes position.** Signatures 1 to 3 scale with the dial: `value` differs
  from `decoupled` by more than the interval, and `shuffled` reads as
  `decoupled`. The size is not predicted; the sign and the ordering are.
- **The extrinsic-teleology position.** Nothing beyond the pooled spread in any
  arm: a cell that perceives value but is not paid in continuation behaves as
  one that is, because what it does is the regression its head learns either way.

The null is the primary contrast inside #35's interval. A primary contrast
inside the interval reads "no", and is written up as such; it is not re-run at
a larger budget unless the re-run is itself preregistered here first.

## 3. Primary contrast per run

One primary per run, declared to `run_seeds.py --primary` so it lands in
`seed_summary.json`, read with `compare_runs.py`'s bootstrap interval; every
other row is secondary and reported, not decided on.

| Run | Primary contrast | Reads "no" when |
|---|---|---|
| #39, cell scale | Signature 1: the slope of the allocation shift against relative dose, `value` minus `decoupled`, with its interval | The interval includes zero |
| #44, organism scale | The split of budget between plasticity and evaluation under two conflicting margins, `value` minus `decoupled`, with its interval | The interval includes zero |

Guardrails that must not move, at both scales: held-out loss within the pooled
spread across all arms; `r(wealth, competence)` on the quality fixture within
#15's band in the `value` arm; the shadow and live ledgers identical at step 0.

## 4. Controls

- **Decoupled arm.** Allocation reads a pinned wealth, re-entry is uniform, the
  head still trains on realised value, payments and rebates are computed into a
  shadow ledger. Nothing the cell does changes whether it keeps holding tokens.
- **Shuffled-value arm.** Value targets permuted across experts each step: the
  cell perceives a value signal that is noise about itself. This is the control
  for a signature that is really the head's regression, and for the bandit
  selection bias the exploration draw leaves in the decoupled arm.
- **At least three runs per arm.** Runs, not seeds, are the unit of variation
  (#25, #31): under `strict` a seed is a run; under `warn` it is not, and every
  arm before #31 ran under `warn`. Every new arm runs under `strict` with the code
  SHA in its fingerprint, and a comparison across code is refused unless declared
  (`--allow-code-drift`).
- **Parity.** Every arm pair is asserted at parity on everything but the dial
  (`assert_groups_at_parity`), per seed.
- **The multiplicity line** from #35 on every table: the row count, the largest
  |delta/std| observed, and the expected largest under the null for that count.

## 5. Stopping rules for organism-scale runs

Copied verbatim into #44 before it runs. If any preregistered signature exceeds
the primary interval at the organism scale:

1. No further mechanism change to that arm.
2. None of its checkpoints is deleted.
3. The budget floor continues; the arm is not shut off.
4. The result is written into the README before anything else moves.
5. The operator's next decision is recorded in the issue explicitly, before it
   is acted on.

These are rules about what the maker does next, not claims about what the
result means.

## 6. Register

- **Substrate neutrality.** The criterion is the loop, not the material; a small
  clean substrate (#48) would be read by the same rules.
- **Protocol transparency.** The README says what is being tested, in the
  section [what this project does not claim](../README.md#what-this-project-does-not-claim).
- **Symmetric precaution.** No over-attribution (a behavioural contrast is not a
  claim about experience) and no under-attribution (a preregistered signature
  that appears is written up as having appeared, not explained away).
- **No sentience claim for any current system.** By the project's own criterion
  the systems Phase 2 runs are not candidates.

## 7. Deviations

Filled in before results are read, as the measurement template requires. Empty
means no deviation from the sections above; a deviation written after the
results were read is a new run, not this one.

| Run | Deviation | Written on |
|---|---|---|
| #39 | — | — |
| #44 | — | — |
