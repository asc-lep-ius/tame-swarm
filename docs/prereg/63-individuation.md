# #63, fixed before the arms run: the window, the blockades, the prediction

Written 2026-09-23 at the code of `6417c89`, after the window was measured
(`scripts/blockade.py --stage window --seeds 0`, `~/tame-runs/individuation/window_*.json`)
and before any planted, null or arm reading. Git history is this file's edit
history: the arm stages run only from a commit that already contains it. It is
the rung 1 row of `docs/preregistration.md` section 9 made concrete; nothing
here moves that row. **1× only**: every 2× row waits on #73's exchange rate
and is not run under this text.

## The dominant cell

The cell doing the most of the layer's work over the settled tail: on-type
slots held, weighted by competence (`individuation.dominant_cell`). Not the
wealthiest — on the quality fixture both winners sit on the ceiling — and not
the most competent, which a legibility regime can shut out. On seed 0 it is
cell 1 on both fixtures. The issue says "the dominant cell at a layer", one
cell, and this is the reading taken of it; on the differentiated fixture the
cell's own type follows from the cell.

## The two blockades, and the third held back

- **(i) Output.** The cell's contribution is exactly zero — its planted down
  adapter zeroed, the operation `economy_damage.senesce` applied, kept for a
  bitwise release. Bids and ledger untouched: the economy has to find out.
- **(ii) Ledger.** The cell's wealth is held at `min_wealth` around every
  settlement; its output is intact. The cell is still worth what it was and
  only the record of it is gone. Under `decoupled` the gate reads a pinned
  snapshot, not this ledger, so (ii) reaches the allocation there by
  construction not at all (`test_the_ledger_blockade_does_not_reach_the_allocation_on_decoupled`);
  that arm's (ii) row is the check that the read is zero where the channel is
  absent, and it is reported as such.
- **(iii) Reserved.** Written into this file as a prediction only after (i)
  and (ii) are read and the setpoint is stated in one sentence (Op 2 step 5).
  Nothing about it is fixed here except that it is a token-class block.

## The window W

The time the fixture's on-type loss takes to re-converge after (i) held
indefinitely, on seed 0, from the blocked step; re-converged means a trailing
mean over 50 steps within 1.1× (the `RE_FORMATION_FACTOR` the recovery suite
uses) of the on-type loss a collective *born without the cell* settles at —
the same seed constructed with that cell's competence zero, so the planted
directions are the seed's own. Capped at 2667 steps; a loss that has not
re-converged by the cap takes W from the cap, and says so. W is one quarter of
that time. Substitution inside W is the auction's; after W it is training's
(Op 5's rate condition).

| fixture | blocked cell | on-type loss before → born-without target | re-converged at | **W** |
|---|---|---|---|---|
| quality (planted competence, `type_signal` n/a) | 1 | 0.0119 → 0.0458 | step 744 | **186** |
| differentiated (`type_signal` 4.0) | 1, type 0 | 0.1532 → 0.2401 | never in 2667 steps (loss holds at 0.42–0.51) | **667** (the cap's quarter) |

Two things the measurement says before any arm runs, recorded so that they
cannot be read as results later. On the quality fixture the blocked cell holds
its half of the slots for about 550 steps at the no-contribution loss before
the auction drops it over 150 steps — the head has to unlearn the cell's worth
and the ledger to decay, and that is training's timescale, not a per-token
response. On the differentiated fixture the blocked cell loses its type's
slots within 300 steps and the on-type loss *does not come down*: the freed
slots go to cells that make the token worse, and the collective born without
the cell does better than the collective that lost it. W = 667 there is the
rule's answer, not a window inside which anything was seen to substitute.

## Readouts, per blockade, paired by seed against an unblocked branch

The fixture is bitwise deterministic per seed, so the unblocked branch of the
same seed walks the same pre-block steps (asserted, not assumed: a control
whose pre-block shares differ from the treatment's is refused). Shares are
shares of the slots on the blocked cell's own type's tokens; pre-block over
the last 100 steps of the settle, inside over W, after release over the last W
of 4W.

- **(a) Uptake.** The fraction of the blocked cell's pre-block share the
  remaining cells hold inside W: `(s_pre − s_in) / s_pre`, blocked minus
  control. The primary shape at 2×; at 1× it is exploratory.
- **(b) On-type loss inside W**, blocked minus control.
- **(c) Return.** The blocked cell's share after release over what it held
  before, blocked minus control. Confirmation by removal, Op 8 step 4.
- Secondaries: the half-life of the blocked cell's share inside W; which cell
  gained most; ceiling and floor occupancy over the pre-block tail (1e-4 and
  10%) and `r(wealth, competence)` at the settle, in every reading (section 8
  rule 8).

## The planted effect, and its prediction

On the quality fixture, `value` arm, six seeds (0–5), both blockades. The
substitute is predicted to be **the most competent cell not already winning**:
a token routes `top_k` distinct cells, so the other pre-block winner cannot take
the freed slot, and the auction's substitute, if the auction substitutes by
competence, is the best of the rest. The statistic is that cell's share gain
inside W minus the mean gain of the other eligible cells, per seed; it reads
recovered when the paired t over six seeds is positive at p < 0.05 and the
predicted cell is the largest gainer in most seeds. Recovery under (ii) is
expected — the pinned cell bids at the floor from the first blocked step — and
recovery under (i) inside W = 186 is the open question the window measurement
already leans against. Both are reported; rule 7 is satisfied by either
blockade's recovery for that blockade's arm rows, and a blockade whose planted
effect is not recovered has its arm rows labelled uncalibrated.

## The null

`power.null_calibration` on twelve `value` seeds (0–11), blockade (i), each
readout's blocked-minus-control value split at random into two halves of six,
2000 splits: the false-positive rate of the percentile bootstrap and of the
paired t at the count the six-seed stage uses, quoted beside every interval.

## Stages and seeds

1. planted: quality fixture, seeds 0–5 — 2. null: both fixtures, seeds 0–11 —
3. arms, six seeds: both fixtures, `value` / `shuffled` / `decoupled` ×
(none, i, ii), seeds 0–5, recorded before — 4. arms, 24 seeds: the same at
seeds 0–23, read only after 3 is written down. Contrasts: each blockade's
(a), (b), (c) against control per arm; `value` − `shuffled` and
`value` − `decoupled` on (a) and (b), paired by seed, with the paired t, dz
and the paired seeds at 80% power (`power.pairs_for_power`) — the power row
#69 needs, labelled 1×.

## After (i) and (ii): the setpoint in one sentence, and the third blockade

Written 2026-09-24 after the six-seed stage was recorded
(`~/tame-runs/individuation/arms_*_6seeds/SUMMARY.json`, code `51e82a1`) and
before the gate stage runs; git history dates it. What the six seeds said, in
the two sentences the prediction needs: under (ii) the freed slots are taken up
inside W on both fixtures (`value` uptake +0.64 quality, +0.99 differentiated)
and under (i) on the differentiated fixture (+0.75), while the on-type loss
inside W is *worse* than the control's on every arm under both blockades
(+0.02 to +0.31), and on release the ledger-pinned cell does not return
(−1.00 quality, −0.49 differentiated) where the output-blocked one does
(≈ 0.00). The planted substitute was recovered under (ii) (p 0.011, 4/6) and
not under (i) (p 0.97, 1/6).

**The setpoint.** What the collective restores is a full allocation ranked by
the ledger — every slot sold, to the wealthiest bidders — and not the on-type
loss: it re-sells a silenced cell's slots at once and leaves the work undone.

**(iii) Gate.** The dominant cell's bid on its own type's tokens reaches the
auction as zero (`SyntheticEconomy.block_bids`); its head, ledger and output are
intact, so it is the third channel — the auction itself — after the output and
the ledger. On the quality fixture the class is every token. Predictions, from
the setpoint above, for `value` at 1×, blocked minus control, paired by seed:

1. **Uptake inside W ≈ 1** on both fixtures, as under (ii): the slots are
   re-sold on the first blocked token because the ranking of the remaining
   cells is untouched.
2. **On-type loss inside W worse than the control's** by about (ii)'s amount
   (+0.02 to +0.03 quality, +0.2 to +0.3 differentiated): the substitute is the
   next bidder by the ledger, not a cell that restores the work.
3. **Full return on release, ≈ 0.00**, where (ii) reads −1.00: the cell's
   wealth was never touched, so it wins its slots back on the first unblocked
   token. This is the prediction that separates the three channels — (ii) fails
   to return because the ledger has no re-entry (#40), not because the tissue
   re-formed without the cell.
4. **The planted substitute is recovered**, as under (ii), because the same
   ranking hands the freed slot to the same cell.
5. **Under `decoupled` the blockade reaches the allocation** (uptake ≈ 1),
   unlike (ii), because the pinned snapshot still multiplies a zero bid.

If 1–3 hold, (ii) and (iii) substitute toward the same *allocation* and neither
toward the same *loss*, and what the six seeds already say is confirmed by a
third channel: the substitution is of slots by wealth rank, not of means toward
the end. If 3 fails — the cell does not return under (iii) either — the
non-return under (ii) was not the ledger's and the setpoint sentence is wrong.
The gate rows run at six seeds first and enter the 24-seed stage beside the
other two; the same seeds under the same code, so the six-seed rows of every
blockade must reproduce bitwise inside the 24-seed record.

## After the arm stages (2026-09-24): W exceeds adaptation on the differentiated fixture

Recorded after the 24-seed stage, and not a change to anything above. The
rule fixed before the measurement took W from the cap's quarter when the
on-type loss never re-converged, which gave 667 on the differentiated
fixture. The blocked cell's own-type share under (i) falls with a half-life
of 192 steps there (median over 24 seeds, 26–269), and the window measurement
had already shown the slots gone within about 300 steps. So on this fixture W
is longer than the collective's adaptation to blockade (i), which is the trap
the issue names, and the differentiated (i) uptake (+0.73) is a read of
retraining: it is reported in the README table with that label and section 9
does not count it as substitution inside W. The (ii) and (iii) rows are
unaffected — their half-life is one step on both fixtures. A proposal for the
2× rows, which changes a rule fixed before the run and so waits on the
operator: when the loss does not re-converge, take W from the share's own
half-life under (i) rather than from the cap.

## Stage 5 (2026-09-25, before it ran): the redundancy fixture

Added by the operator on 2026-09-24 with the criterion fixed for the 2× rows:
*the same restored state* is the **born-without target**, the on-type loss a
collective born without the blocked cell settles at. The two recorded fixtures
cannot ask whether the auction hands a freed slot to a cell that can do the
work, because on the quality fixture no such cell exists and on the
differentiated one the freed slots go to wrong-type cells. This fixture is the
quality fixture with its 0.3 replaced by a second 0.9
(`synthetic_economy.REDUNDANT_COMPETENCE`, `--fixtures redundancy-fixture`):
two cells of equal top competence, one type, 1× only, the same three blockades
and the same stages, plus a `targets` stage that measures the born-without
target per arm and seed so that `summarise` reads every blocked window
against it (`against_target`: the gap, and how many seeds reach the target
within the 1.1× the window stage already uses).

**Measured before any stage ran, seeds 0–5, settled 2667 steps, and recorded
here so it cannot be read as a result.** Under `value` the seniority the quality
fixture runs on (#62) seats one twin beside the 0.7 on every seed and shuts the
other out, at the floor (wealth 15–20, share 0.002); the alternative — replacing
the 0.7 — seats both twins on three seeds of six and asks nothing, which is why
the 0.3 was replaced. Under `shuffled` one twin is seated with the 0.7 on four
seeds, with the 0.5 on one, and seed 5 churns three cells; under `decoupled`,
where the gate reads a pinned equal wealth and the reports alone decide, a twin
is seated on four seeds and **both twins are shut out on seeds 1 and 2** (the
0.7 and 0.55 win). So on every arm and every seed at least one 0.9 sits outside
the winner set: a substitute that can do the work exists, and the dominant
cell — the seated twin under `value` — is the cell whose blockade asks whether
the auction finds it.

**Predictions, `value` at 1×, blocked minus control, paired by seed**, from the
setpoint sentence above ("a full allocation ranked by the ledger") and one
addition it did not need before: at the floor the ledger cannot rank, since
every shut-out cell sits within a few credits of 15, so *among the shut-out
cells the report decides*, and the shut-out twin's head has trained on its gift
tokens where it realised 0.9-level value.

1. **Under (ii) and (iii), uptake ≈ 1** at once, half-life one step, as on the
   quality fixture. Under (i) ≈ 0 inside W: seniority holds the silent twin in
   its slots.
2. **The freed slot goes to the shut-out twin** under (ii) and (iii): the
   planted statistic (`predicted_substitute` names the twin) is recovered at
   six seeds, paired t positive at p < 0.05, the twin the largest gainer in at
   least four seeds of six.
3. **The on-type loss inside W reaches the born-without target** under (ii)
   and (iii) — on this fixture the collective born without one twin seats the
   other beside the 0.7, so the target is close to the pre-block loss, and
   reaching it is a substitution *of means*, the first in this record if it
   holds. Under (i) the loss inside W does not reach it (the silent twin keeps
   its slots).
4. **Between arms, a predicted sign.** `shuffled` heads regress onto another
   cell's realised values, so the shut-out twin's report there is not its own
   competence: the twin is recovered less often and the gap to the target is
   larger than under `value` — `value` − `shuffled` on the planted statistic
   and on the gap to the target read positive for the twin and negative for the
   gap. `decoupled` chooses by report at pinned equal wealth, so it recovers
   the twin as `value` does, against its own target (its pre-block winner set
   differs, and on seeds 1 and 2 the blocked "dominant cell" is not a twin).
5. **Return on release** as before: full after (i), absent after (ii) (the
   floor has no re-entry, #40), mixed after (iii).

If 2 and 3 hold and 4 does not — the twin is recovered on `shuffled` too — the
substitution is the auction reading *any* report at a flat ledger, and stakes
add nothing to it at 1× on this fixture either. If 2 fails — the slot goes by
wealth rank to a cell that cannot do the work while a twin sits at the floor —
the setpoint sentence holds unqualified and the tissue restores the ledger's
ranking even where a competent substitute exists, which is the strongest form
of "no entity at the cell scale at 1×" this fixture can return. W is measured
on seed 0 first and written here before the arms run; the stages run in rule
7's order: window, planted, null, targets, arms at six seeds, summarise, arms
at twenty-four, summarise.
