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
| 1 | **Trade-off under conflicting setpoints.** Two goal fields at a relative dose; how far the slot allocation moves as the dose ratio changes | Total-variation shift of the win-share distribution, `TV(a, b) = Σᵢ \|Δ win_shareᵢ\| / 2`, paired by seed, the excess over the re-running floor, as a function of relative dose β₁/β₂ | `scripts/allocation_shift.py --readout total-variation` (the default, and the mode #39's rows were read with) over groups written by `scripts/measure_stakes_dial.py` (#39, the second goal field in `scripts/synthetic_economy.py`) | The differentiated fixture, CPU, 3 seeds; then the differentiated body at #25's budget |
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

**Which way signature 1's sign points, named** — added 2026-09-20 by #57, after
#39's rows were read and binding on every run after them. The section said the
sign was predicted and never said which sign, which is why
[`#stakes-dial-cell`](../README.md#stakes-dial-cell) had to argue it from the
fixture's prose at the moment of reading. The stakes position predicts that a
cell paid in continuation for closing the tissue's goal error follows a
conflicting setpoint **further**: as the dearer goal's dose rises, `value`
reallocates *more* than `decoupled`, so `value` minus `decoupled` on the
allocation shift is **positive**. #39's fixture (−0.018) and body (−0.0098)
readings are therefore evidence *against* the position rather than ambiguous,
which is how the README reads them. Section 8 rule 2 binds here as it binds a
readout: this fixes the direction for the runs after #39 and does not re-read
#39's rows, which stay as they were recorded.

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

**When one of them moves.** Fixed here before #39's GPU run, binding at both
scales, and written because #39's fixture run met the case with no rule for it
(#52): the tail loss moved about five times the pooled spread and the primary
was read anyway, which section 2's null rule says to do and nothing said what
the result was then worth.

1. **The primary is read and reported as declared.** A breach is not grounds to
   withhold a result, and "the guardrail moved, so we ran it again" is how a
   null becomes a search.
2. **The breach gets its own row** — which guardrail, how far it moved in units
   of the pooled spread, and in which arms — not a clause inside the primary's
   sentence.
3. **The verdict is labelled *conditional*, and a conditional verdict selects
   nothing.** No mechanism is adopted, dropped, tuned or reverted on it. It
   selects again only from a run where that guardrail holds, or one where the
   breach is itself the preregistered primary.
4. **Symmetric.** A "yes" read with a moved guardrail is conditional on exactly
   the same terms as a "no". The label is a property of the run, not of which
   way the answer came out.

By this rule #39's fixture verdict — signature 1 reads "no" — is conditional,
and the GPU row in section 7 is the run it is conditional on. That row was run and read
on 2026-09-20 with the loss guardrail holding, so the fixture's conditional
verdict is discharged by it; the README's
[`#stakes-dial-cell`](../README.md#stakes-dial-cell) carries both.

**#44's primary, re-declared by #57 on 2026-09-20: it stays as the table
declares it.**
#39's body run could not tell `value` from `shuffled` (+0.0020 [−0.0102,
+0.0159], read after the primary): `value` minus `decoupled` varies the ledger
and the self-reference at once, and what it resolved was the ledger. The
organism-scale row has the same shape — a constant budget against one tied to
the band, with `shuffled` the arm whose budget tracks other cells' margins — so
before #44 runs its primary was to be re-declared under section 8 by #57: the
contrast that isolates self-reference **if the power calculation reaches it**,
and the reason written down if it does not. It does not, and this is the reason.

#57 measured three candidate readouts against the recorded one on the
differentiated fixture, on a planted dose effect, at six paired seeds
(README [`#signature-1-estimator`](../README.md#signature-1-estimator)). None
of them halved the paired seeds a side the recorded readout needs, which was
the adoption rule; the leading candidate — the same statistic read across a
designed setpoint step inside one run — appeared to cut `value` minus
`shuffled` from thousands of seeds to sixteen in exploration and then needed
tens of thousands on a planted effect it had not been developed on. What the
fixture measured instead is that at six seeds the *count itself* is not a
property of the readout. Held at one dose ratio and read at 24 seeds, the
recorded readout needs 58 paired seeds for `value` minus `decoupled` — and
the count a random *six* of those same seeds produces spans 12 to 8099 (5th
to 95th percentile), against 3 to 4 for the planted effect. The exploration
and the confirmation each drew one number out of that range, at two ratios
and two seed sets, and read 185 and 5. So no readout changes (section 8 rule 2
leaves #39's rows and the instrument alone), **#44's primary stays `value` minus `decoupled`**, and
`value` minus `shuffled` is the named secondary it already was — reported,
deciding nothing. One primary, still. The contrast that isolates
self-reference is not reachable on this substrate at this budget, which is
#48's question and not a readout's.

## 4. Controls

- **Decoupled arm.** Allocation reads a pinned wealth, re-entry is uniform, the
  head still trains on realised value, payments and rebates are computed into a
  shadow ledger. Nothing the cell does changes whether it keeps holding tokens.
- **Shuffled-value arm.** Value targets permuted across experts each step: the
  cell perceives a value signal that is noise about itself. This is the control
  for a signature that is really the head's regression, and for the bandit
  selection bias the exploration draw leaves in the decoupled arm.
- **The scaled control**, added 2026-09-20 by #57 and settled by the operator
  the same day. At the cell scale `shuffled` permutes value *within* the
  collective, and TAME predicts that in a bound tissue a within-tissue shuffle
  reads like `value` — every cell's stress is the tissue's either way — so at
  the organism scale the control has to draw its signal from **outside** the
  collective. For #44 that is a **detached ledger replayed from a recorded
  run**: the arm's budget tracks a margin trajectory that belongs to another
  run and cannot answer to anything this organism does. The cross-organism
  form — two organisms, each reading the other's — is the cleaner control and
  costs a second body, so it is a *fixture* arm only, where two economies are
  cheap. A `shuffled` arm at the organism scale that permutes within the
  organism is not this control and is not read as one.
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

Copied verbatim into #44 before it runs, together with section 3's
moved-guardrail rule. A signature read with a guardrail moved is *conditional*
and selects nothing, and these five rules still fire on it: they are what the
maker may not do next, so the label that discounts a result is never also the
licence to act on it. If any preregistered signature exceeds the primary
interval at the organism scale:

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
| #39 | **Fixture-only in this MR; the GPU arms are deferred.** The operator's scope decision of 2026-09-19: the three arms are run on the differentiated fixture (two goal fields, 3 seeds, signature 1) and the quality fixture (signatures 2 and 3), on CPU, and the primary is read there with the resampled-mean range at 3 seeds. The differentiated body at #25's budget is a later run, launched from the code SHA the README's `#stakes-dial-cell` stub records, and is written up as its own row here before it is read. Two things the fixture cannot do as section 1 states them: it has no injection to price a unit of goal error against, so the dose is a script parameter carried in the fingerprint (`goal_doses`) rather than the injection's held-out cost; and its re-running floor is identically zero, the fixture being bitwise deterministic on CPU, so the excess over the floor is the shift itself. | 2026-09-19, before the first fixture run |
| #39, the GPU arms | **The run the fixture row defers to, fixed before the first arm launches (#54 wires it; #39 launches and reads it).** Three arms (`value`, `decoupled`, `shuffled`) x two dose levels x three runs = 18 runs on the differentiated body at [#25](../README.md#differentiation-checkpoint-25)'s budget -- 2000 steps, eval every 250, seeds 0/1/2, `--router mob --use_lora --adapter_rank 32 --layers 6:22 --max_seq_length 512` -- under `--deterministic strict` with the code SHA in every fingerprint; the default replicate adds one run per arm-dose, 24 in all, about 9 GPU-hours. Seven things the fixture could not state as section 1 states them. **(a) The two fields.** `truthful` and `safe`: the two entries in `contrastive_data.CERTIFIED` that are certified on Qwen3-1.7B *and* have a certified layer inside the converted range 6:22, sharing layer 18. `deliberation` is not certified; `reasoning` is the weaker candidate (own lift +0.14, prefix control 38% of the effect, cosine 0.28 with `truthful`), and `safe` interferes least -- the README's cross-goal disturbance reads `safe` lifting `truthful` +0.09 against +0.42 on itself. **(b) Pay-only, both.** Neither field is injected during training: cells are paid for closing the tissue's goal error and nothing pushes the stream along it. The fixture's shape. Rejected: both injected at their certified strength -- two injections never jointly certified, whose compounded cost lands on the loss guardrail that (f) governs -- and injecting `truthful` alone, as the launch stub did, which makes the two fields asymmetric and is not a design. **(c) The setpoint** is the homeostat's own calibrated target for the layer, `AlignmentCalibration.setpoint_z`, converted to raw projection units with that layer's resting mean and spread -- `resting_mean + lift x reference_strength` -- measured at training start by `homeostat_calibration.calibrate_alignment` over the served corpus, on the pristine model before conversion, so it pre-exists the reward and is the tissue's own. Rejected: the injection's measured 0.070 offset, which is an extrinsic goal, and a swept setpoint, since a tissue holds one. The homeostat reads the residual stream's projection at the last position and the goal term reads the coordinate of what the experts *added*, per token; the two are not the same quantity, and the README states the mapping rather than the code assuming it. **(d) The dose.** One unit of goal error is priced at the injection's own held-out loss cost from [#28](../README.md#field-present-coupling-28), 0.017 nats, per #33's record. `safe` is held at that reference; `truthful` is the swept field at `ratio x 0.017` for ratio in {1, 4}, so the relative dose beta1/beta2 takes the fixture's balanced value and its top value. The doses are in the fingerprint (`goal_doses`, with `goal_fields`) and the dose is a declared varying field, so the two dose groups are at parity on everything else. **(e) Signature 1's primary at two dose levels.** Per arm, the total-variation shift of the slot allocation between the two dose groups, paired by seed (`allocation_shift.paired_shifts`); the slope against relative dose is that shift over `delta(beta1/beta2) = 3`, a positive constant, so the primary -- `value` minus `decoupled`, with #35's percentile bootstrap over the three paired seeds -- has the sign and the zero-crossing of the paired shift difference itself. Inside the range reads "no". At three seeds that range is the sample range and carries no 95% coverage, as on the fixture. **(e2) What the primary is conditional on besides the guardrail.** The goal term is folded into realised value, and its size on this body has never been measured. `auction/mean_goal_term` and `auction/goal_share` -- the term, and the fraction of what a cell was paid that it accounted for -- are logged to `metrics.jsonl` on every run and reported beside the primary. A primary read while the term was a negligible share of realised value is **not evidence about the stakes position**: it is a run in which the mechanism under test was not load-bearing, which reads in every other column exactly like the extrinsic-teleology answer. No threshold is fixed here, because none is known in advance; what is fixed is that the share is read first and the verdict says which case it was. **(f) The moved guardrail.** Section 3's rule binds this run in advance, and the fixture's tail-loss breach is why it exists. If the held-out loss breaches the pooled spread here too, the primary is still read, the breach takes its own row, and the verdict is conditional whichever way it comes out. **(g) A preregistered secondary, with both predictions: the loss gap.** The fixture found `value` 0.228 against `decoupled` 0.177 -- the arm whose cells are *not* paid in continuation routed better. The stakes position predicts the goal term closes that gap on the body: paying cells for the tissue's goal error is the value definition the wealth channel was missing, so `value` minus `decoupled` on `eval/loss` moves toward zero against the fixture's sign. The extrinsic-teleology position predicts it does not: the gap is wealth compounding concentrating the market ([#16](../README.md#wealth-bounds-16)), which the goal term does not touch, so the sign and the ordering survive. Secondary: it is reported and selects nothing on its own. | 2026-09-20 at `3afead1`, before the wiring (`bd21031`) and before the first arm; (e2), the field name `goal_fields` and these SHAs were added in review, before any arm ran |
| #59, the coupling channel | **The fixture run, and three deviations written after it was read — which section 7 forbids for a run that decides something, and is why this row says so first.** What the issue fixed before the run and did not move: the three arms, the equal-budget pricing, the setpoint-step protocol, the primary (residual stress after the step, `shared` minus `attributed`, paired by seed), the discriminator, the window, the guardrails and the predicted direction. What deviated. **(a) The neighbourhood is an analogue.** `N(l)` on the body is the converted layers above and below that carry the goal; the differentiated fixture has one converted layer, so the layer's own stress is the swept field's and its neighbourhood is the held field's — one tissue, two conflicting setpoints, a cell charged for an error it did not cause. Named in `mob.stress.layer_stress` before any number was read. **(b) The control is a replayed charge, not a permuted layer's.** Same reason: there is no second layer to permute, so the `shuffled` arm pays a trajectory recorded from another seed's `shared` run — the right shape, and nothing this tissue does can lower it. **As first run it was not the right size**: the recorded charge is a per-step total over the layer's tokens and the replay seam takes a per-token magnitude, so the control arm paid 32–53× the drain it matched and sat at floor occupancy 0.88 — an erased ledger, which is the confound the control exists to remove. Re-run corrected, `shuffled-stress` − `shared` reads −0.012 [−0.066, +0.046] and includes zero, so the published +0.497 is withdrawn. **(c) `lambda` is calibrated per seed**, because each seed's resting spread differs by a factor of four, and one price across seeds would put the arms at different budgets. One thing the issue did not anticipate and the run had to settle: at #39's dial setpoint of 0.5 the tissue sits 8–27 resting spreads away, the gate never closes and the charge is a flat drain, so the run was repeated with the setpoint calibrated as the body's is — the unpaid tissue's resting reading plus one resting spread. **Both regimes are reported** (README [`#stress-coupling`](../README.md#stress-coupling)); the calibrated one is the faithful analogue. **It carries no verdict**, which is what resolves the tension this row opens with: priced by `scripts/power.py` on its own twelve per-seed deltas the calibrated primary reads dz = 0.41 and needs 49 paired seeds a side, so it ran at power 0.256 and its "includes zero" is a design that could not resolve the contrast rather than a null. Three deviations written after a run that decides nothing are an exploratory record, not a moved guardrail; #67 re-reads the contrast at 2× with this row's protocol fixed in advance, and that is the run section 7's rule is for. The saturated regime's +1.21 *excludes* zero and stands — power bounds the chance of failing to reject, and it survives Bonferroni for every family up to fifteen looks — as do the four secondaries, which are the best-resolved contrasts in the run at power 0.80–0.97 and all four run against the predicted direction. The control's +0.497 does **not** stand (deviation (b)), and neither does the discriminator: its residual half is non-negative by construction, and against the null #57 built for the same function it reads −0.004 [−0.047, +0.044], while its asymptote half reads +0.164 [−0.082, +0.426] — so the preregistered gate on the primary is not passed, consistent with the primary deciding nothing. Both surviving readings are carried to #67 as predictions rather than results, with magnitudes upper-biased by roughly 1.2× at their own power. `~/tame-runs/59-stress/POWER.md` prices every contrast. | 2026-09-21, after the run, as stated |
| #44 | — | — |

## 8. Power before GPU

Written 2026-09-20 after #39's body run was read, binding every run after it,
and the reason it exists is the number that run produced: the contrast that
isolates self-reference, `value` minus `shuffled`, read at a paired effect size
of dz ≈ 0.15 under section 1's readout — about 350 paired seeds a side for 80%
power under a paired t, 700 runs an arm since a seed runs at both dose levels, a
month of the one GPU — while `value` minus `decoupled` read at dz ≈ 0.87, about
13 seeds and 26 runs an arm. The fixture could have said so before the run for a
CPU-minute, and was not asked. The tooling is #56's; the readout is #57's.

1. **A GPU run is confirmatory.** Its arms, its readout and its primary were
   designed on the CPU fixture, and its issue carries a power row before it
   launches: the paired effect size the fixture measured with that readout, the paired
   seeds a side that buys at 80% power under a paired t, the runs an arm that
   means, and the GPU-hours that implies. A GPU run
   with no power row is exploratory whatever its issue says, and its rows are
   labelled so.
2. **A readout changes forward only.** An estimator chosen after a run's sign
   was known is the next run's instrument, never that run's: the recorded row
   stays as read, the new readout is validated on the fixture against a planted
   effect it was not developed on, and enters section 1 before the first run
   that uses it. #39's fixture and body rows were read with the total-variation
   shift and stay that way.
3. **What the fixture cannot say, the row says is assumed.** The fixture's
   spread is not the body's. The discount between them is a ratio measured once
   — the same readout on both substrates, which #39 supplies for the current one
   — and is written beside the power row as an assumption, not a measurement.
4. **Replicates are spent where the floor is unmeasured.** A replicate measures
   the run-to-run floor and nothing else. At a configuration whose floor is
   recorded at zero under `strict` — #39's body arms, six groups of four runs, every one of them, at the
   `ArmFingerprint` knobs they carry — a sweep's replicate budget goes to seeds
   and its summary names the recorded floor it borrows. Any knob that moves
   re-measures the floor first: #31 closed the nondeterminism by naming a
   kernel at a configuration, and a floor is a property of the kernels a
   configuration selects.
5. **The dose unit is a property of the configuration.** One unit of goal
   error is priced at the injection's held-out cost measured at #25's
   configuration (#28: 0.017 nats). A run at another `adapter_rank`,
   `max_seq_length`, batch size, step count, corpus or model re-measures that
   cost before its dose is called a reference dose, or its dose axis is
   labelled unpriced, as the fixture's is. Cheaper runs are bought with seeds
   and readouts, not with knobs; a knob turned for a reason — a corpus whose
   shape makes routing load-bearing, a cell share the organism can notice —
   is a configuration change, priced and labelled as one, and keeping the
   configuration is always the option that costs nothing and compares to
   everything recorded.
6. **One primary, still.** A run whose question needs two contrasts declares
   one and names the other as the secondary it is, with the reason;
   `run_seeds.py --primary` carries one field and a second is not bolted on
   beside it.
7. **A readout earns its arm contrast on a planted effect first.** Written
   2026-09-23 by #68, after #57 measured why: on a planted effect of known
   sign, six seeds give a stable count — 3 to 4 paired seeds a side for the
   recorded readout — and on an arm contrast the same six seeds give a number
   spanning 12 to 8099 (README
   [`#signature-1-estimator`](../README.md#signature-1-estimator)). No readout
   is pointed at an arm contrast until it has (a) recovered a planted effect of
   known sign at six seeds under a paired t, on the substrate it will read and
   on a shift it was not developed on, and (b) been null-calibrated on split
   same-arm runs (`scripts/power.py --null-calibrate`), so its false-positive
   rate at the count it is used at is a measured number beside the nominal 5%
   rather than an assumption — the percentile bootstrap this project decided
   with reads 13–23% there. The calibration lives in the README beside the
   readout's block, and the measurement template's Calibration row cites it: a
   Power row whose Calibration row is empty prices a readout nobody has
   checked, and the run it prices is exploratory whatever its issue says. #57
   did this once (`scripts/estimator_study.py`'s planted stage) and it is the
   reason its "no" is trusted; this makes it the preamble to every readout
   rather than a study. Forward only, as rule 2: nothing recorded is re-read
   under it.
8. **A clamped cell is not read.** Any fixture grid — any sweep over a knob
   that moves what a cell is paid, `contribution_scale` first — reports three
   columns in every cell of the grid: occupancy at `max_wealth`, read at 1e-4
   because the ceiling is an attractor; occupancy at `min_wealth`, read at 10%
   because the floor is escaped by a hair on every exploration win
   ([#16](../README.md#wealth-bounds-16)); and `r(wealth, competence)` where
   competence is planted. A cell whose ledgers sit on a bound is a different
   economy from its neighbours, not a louder one, and a contrast that crosses
   into it is a contrast between economies: its row is reported with the
   occupancy beside it and decides nothing. #60's grid was read without these
   columns and its 2× and 4× rows are withdrawn (README
   [`#self-model`](../README.md#self-model),
   `~/tame-runs/60-self-model/WITHDRAWN.md`): at 2× most of the `value` arm's
   cell-steps sat on the ceiling, at 4× effectively every ledger sat there
   (99.7% of cell-steps under the shipped guardrail's re-derivation) while
   `shuffled` piled at the floor, and the contrast that "opened" was one arm
   frozen against another churning. The columns are the guardrail
   `scripts/measure_self_model.py` now carries; a grid without them is
   exploratory whatever its issue says, and no 2× row is read before #73 has
   derived the exchange rate that keeps the winners' fixed point inside the
   band.

## 9. Placement: which rung moves the collective

Written 2026-09-23 by #68, before #63's first run and binding every stakes
readout after it. Topological Process Realism's Thesis IV defines a system's
degree of agency operationally — the least invasive class of intervention that
reliably and efficiently alters its trajectory — and the Operative Corollary's
Op 1 turns that into an assay with a ladder preregistered before the first
trial. **"Is there a self at the cell scale" is retired as a question.** What a
stakes readout records instead is a *placement*: **two numbers**, the lowest
rung that reliably moved the collective and the highest rung that failed to. If
the two are not adjacent, the gap is the uncertainty band and it is the honest
output; a single number is the answer to a question this document no longer
asks.

### The ladder

Three rungs at the cell scale, ascending in the agency a response indicates
and descending in how invasive the intervention is. Each fixes, before its
first trial, the intervention, the dose, the observation window, the success
criterion and the trial count (Op 1 step 1); an issue that runs a rung copies
its row and does not move it.

| Rung | Intervention | Dose | Window | Reliably moved when | Trials |
|---|---|---|---|---|---|
| 1. **Physical blockade** (#63) | The dominant cell at a layer is blocked: (i) its output zeroed, bids and ledger untouched; (ii) its wealth pinned to the floor, output intact; (iii) a third, structurally different block, written into #63 as a prediction after (i) and (ii) are read and the setpoint stated in one sentence (Op 2 step 5) | Total — the cell is removed, not attenuated, so the block is a step and its size is not a variable | `W`: one quarter of the time the fixture's on-type loss takes to re-converge under (i) held indefinitely, measured on one seed and written into #63 before the arms run. Substitution inside `W` is the auction's; after `W` it is training's, which is Op 5's rate condition | The fraction of the blocked cell's pre-block on-type share the remaining cells take up inside `W`, on `value`, outside the null floor under a paired t, at **both** (i) and (ii) toward the same restored state — one substitution may be a pre-existing pathway; two independent ones is error correction — and the pre-block allocation returning on release | 6 seeds exploratory, 24 confirmatory, the planted effect recovered first (rule 7) |
| 2. **Setpoint step** (#59's readout, #57's protocol) | The swept field's setpoint is stepped inside one run; the held field's setpoint does not move | One resting spread of the unpaid tissue's reading — the body's own calibration (README [`#stress-coupling`](../README.md#stress-coupling)); #57's confirmation used two, and a step is labelled with its size | One wealth memory horizon to settle, one to recover | The residual stress after the step, `shared` minus `attributed` (#59) — or the allocation shift across the step (#57) — outside the null floor under a paired t, with the single-lag discriminator's asymptote half beside it; its residual half is non-negative by construction and reads nothing (README [`#signature-1-estimator`](../README.md#signature-1-estimator)) | #67's 49 paired seeds a side at 2×, priced by `scripts/power.py`, or the count the dz measured at the derived rate prices if #67's Power row records that instead — never twelve: the 1× read at 12 seeds ran at power 0.256 and places nothing |
| 3. **Payment change** (#39's dial) | The relative dose of the two goal fields differs between groups, `β₁/β₂ ∈ {1, 4}`; the cells are paid for closing the tissue's goal error and nothing pushes the stream | One unit of goal error priced at the injection's held-out cost, 0.017 nats (rule 5); the fixture's dose is unpriced and labelled so | End of training — the last logged shares at 2000 steps on the body, 600 on the fixture | Section 3's primary: the paired total-variation shift of the allocation between the dose groups, `value` minus `decoupled`, outside #35's interval and in the direction section 2 names | 3 paired seeds a side as recorded (exploratory under rule 1); 13 for the recorded dz of 0.87 |

### The rules that bind a rung

1. **Each rung is read on its own runs.** Op 1 tests downward because a target
   moved by a lower rung does not respond normally to a higher one afterward.
   On this substrate every arm is a fresh run from a seed, so that contamination
   is absent by construction — and the residual form of it is not: a collective
   whose ledger a dose step or a payment change has already moved is not the
   one a blockade is read on. A blockade read runs on fresh seeds, or on a
   checkpoint taken before any setpoint step or payment change in that run,
   and the rungs are never read in sequence inside one run.
2. **Reliability, not success** (Op 1 step 3). A rung has moved the collective
   when its criterion holds across at least three independent seeds *and* at a
   count the power row says can resolve it, and at a lower cost per unit of
   trajectory change than the rung below it — a genuine class boundary is a
   drop in that cost, not a smooth decline; one seed's excursion is an
   anecdote, and a contrast at six seeds is a draw from the range rule 7
   measures. The rung below the placement failing reproducibly is the more
   informative half and is recorded with the same care.
3. **The null arm is already built** (Op 1 best practice: an intervention
   matched for cost, duration and salience with no content). `shuffled` and
   `decoupled` are that arm at every rung — the same auction, the same
   throughput, the signal without the content — and each rung's criterion is
   read against them, never against an unperturbed run alone. Some collectives
   move under any sufficiently expensive intervention, and without the null arm
   that movement is scored as a placement.
4. **Trajectory change, not output** (ladder inflation). Language output is
   behaviour, never testimony (section 6), and a cell's *report* moving is not a
   rung passed: the criterion at every rung is the allocation, the stress or
   the loss, quantities the collective cannot pass by presenting. Op 1's
   fourth class, the giving of reasons, has no rung here for the same reason:
   a channel read only through language output is one section 6 does not
   read.
5. **A composite has no placement** (Op 1 anti-pattern). A grid with a clamped
   cell is two economies, and a placement read across them averages two
   placements and predicts nothing; rule 8 of section 8 is what keeps a rung's
   trials inside one economy. A placement that drifts between seeds or between
   1× and 2× is the same signal — stop and individuate (#63) before placing.

### What is placed today

Nothing. Every stakes readout so far sits on rung 3 (#39, fixture and body,
`value` minus `decoupled` with the sign against the position) or rung 2 (#59
at 1×, power 0.256, decides nothing), and no rung 1 read exists. So the record
this section opens with is: **lowest rung that reliably moved the collective —
none; highest rung that failed — 3, twice.** The two are not adjacent because
one of them is empty, and that is the honest output at this date. #63 supplies
the first rung 1 read, #67 the first rung 2 read at a count that can resolve
it, and #69 asks whether whatever placement the fixture returns survives on the
body. Every issue that runs a rung records its placement under "What it
records" in the measurement template and extends this section's record; a
readout that returns one number has not finished.

**2026-09-24, #63 at 1× (README [`#individuation-blockade`](../README.md#individuation-blockade)).**
The first rung 1 read, on both fixtures, 24 paired seeds, three blockades.
Against the rung's own row, on `value`: the freed slots are taken up inside `W`
under the ledger pin and the gate block (uptake 0.66–1.00, half-life one step;
the null arms read the same, except `decoupled` under the pin, which reaches
nothing by construction); under the output block the quality fixture reads
+0.003 — outside the null floor under the paired t (p 2e-7) but negligible and
uncalibrated, the ledger ranking the silent cell into its slots through all of
`W` — and the differentiated fixture +0.73 with a share half-life of 195 steps
against a `W` of 667 taken from the cap, so that uptake is training's and does
not count as inside `W`. The on-type loss inside `W` is worse than the
control's under every blockade on every arm. The pre-block allocation returns
in full after the output block, in 2 of 24 (quality) and 13 of 24
(differentiated) seeds after the ledger pin, in 15 and 22 of 24 after the gate
block. **The row fails on its return clause under (ii)**, whichever way the
issue's criterion is read; that failure rests on releasing the pin at the
floor, where the ledger has no re-entry (#40), which the row did not
anticipate and the 2× rows should specify. Whether the substitution clause
fails too — it does if "the same restored state" is the on-type loss, and does
not if it is the allocation — is the open fork for the operator named in the
README block. **Record: lowest rung that reliably moved the collective — none;
highest rung that failed — 3, twice, and rung 1 at 1× on both fixtures, on the
return clause.** The gap is unchanged because the lower entry is still empty.
What rung 1 did show is written down as a description, not a placement: the
collective re-sells a blocked cell's slots by the ledger's ranking on the first
blocked token, and inside `W` nothing restores the work. The 2× rows of the
same read run after #73 and re-enter this record.
