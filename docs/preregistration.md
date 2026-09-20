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
property of the readout: the recorded readout's own requirement for `value`
minus `decoupled` read 185 paired seeds on one six-seed sample and 5 on
another. So no readout changes (section 8 rule 2 leaves #39's rows and the
instrument alone), **#44's primary stays `value` minus `decoupled`**, and
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
