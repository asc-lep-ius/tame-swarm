# #64, fixed before the grid runs: the port, the floor, the lines

Written 2026-09-25 before any reading, on the branch `64-loudness-routing`
stacked on #73's (`73-exchange-rate`); git history is this file's edit
history. The issue fixes the primary, the guardrails and the verdict lines; what
it leaves to the implementation is written here so that it cannot be chosen
after the numbers are seen.

## The port (`scripts/loudness_routing.py`, not a flag on the body script)

The issue's Touches row names `scripts/counterfactual_routing.py (--fixture)`.
The port is a separate script that *imports* the body script's
`GumbelTopKRoute` unchanged, so the body path is untouched to the byte rather
than "bitwise unchanged by a test": the body script's diff on this branch is
empty. Deviation recorded here.

What the body read does and what the port does in its place:

| body (#58) | fixture (this) |
|---|---|
| a held-out probe of 4096 tokens | 128 draws of the fixture's own generator after the settle, 4096 tokens; every arm of one seed has consumed the generator identically over 2667 steps, so the probe is the same tokens on every arm (asserted by a hash of the inputs) |
| score: log-probability of the next token | score: the token's own squared error against the planted target, lower is better; the per-token loss the fixture trains on |
| executed route: `model.eval()`, economy frozen | `mob.eval()` (no exploration gift), `frozen_economy` (no settlement); the read is asserted reproducible and to leave the ledger untouched |
| K = 32 Gumbel-top-k alternatives at noise scale 1 on the arm's own bids | the same class on the fixture's gate, K = 32, scale 1 |
| the stream: rerouting token *t* moves every later token | none: the fixture's tokens are independent, so an unmoved token reads the executed loss exactly and the "stream only" population is empty by construction |
| the replicate floor: two checkpoints of one seed under `strict`, bitwise equal | the alternatives' own RNG: the same settled seed, K draws under two Gumbel seeds, the per-token difference between the two best-of-K gaps; **the floor is that difference's 95th percentile** |
| fragile: gap > floor | the same |
| confident: the top quartile of executed log-probability | the quarter of tokens with the lowest executed loss |
| adapter footprint: loss with every adapter zeroed minus with them on | the same, on the fixture's experts |

Loss units grow as the scale squared (the planted correction scales by *c*
and the loss it closes by *c²*), so every gap is also read **relative to the
executed loss of the token it was read on**, and the contrasts across scale
are read on the relative gap.

## The rate

Every scaled cell runs at #73's derived exchange rate for its fixture and
cell count (`derive_reward_scale`, 2667 steps, seeds 0–2), reused from
`derivation_<fixture>_cells<N>_scale<c>.json` under the output directory when
one exists at this code and derived otherwise; the fingerprint of every
reading carries `reward_scale` and `reward_scale_derived`. 1× runs at the
recorded constant. A configuration whose derivation is refused is skipped and
the refusal printed.

## What is read, per (fixture, arm, cells, scale, seed)

- **fragile fraction**: tokens whose gap exceeds the floor;
- **the primary shape**: the mean gap on fragile tokens, absolute and relative,
  and the same minus the floor;
- **the guardrail** (#58's): on the confident quarter, how often an
  alternative beats the executed route, averaged over the K draws
  (`confident_better`); at 1× this must read about 0.5 on the fixture too,
  or the port is not measuring what the body read measured;
- `moved_better`, `moved_fraction`: among draw-token pairs whose route moved,
  how often the move helped and how many moved;
- the adapter footprint, absolute and relative;
- rule 8's columns over the settle's last 100 steps.

## The lines, as the issue fixed them and as they are read here

- **"One lever"** when at 2×, 8 cells, `value`, the relative gap on fragile
  tokens minus the floor exceeds the same at 1×, paired by seed (six seeds,
  paired *t*, p < 0.05, positive), *and* the guardrail departs from 0.5 in
  the direction of the executed route being better (`confident_better`
  below 0.5 by more than the six-seed spread).
- **"The market needs its own fix"** when the guardrail stays at 0.5 (its
  six-seed interval includes 0.5) at 2× while #60's allocation contrast at the
  derived rate is non-null at the same scale (README `#self-model`, the grid
  at the derived rate: dz −3.8 at 8 cells, 2×).
- Neither line is read on a cell whose rule 8 columns say it is clamped
  (ceiling occupancy outside the 1× seeds' range by more than one cell).

## Secondaries, reported and not decided on

4×; 4 cells; `shuffled` and `decoupled`; the differentiated fixture; the
adapter footprint per scale; and the correlation across the grid's cells
between the `value` arm's relative fragile gap and #60's `value` − `shuffled`
dz at the derived rate (`~/tame-runs/73-exchange-rate/self-model/self_model.json`),
Pearson over the six (cells × scale) points with a bootstrap interval over
seeds — the number that says whether #58 and #60 are one problem.

## What is expected, written down so it cannot be adjusted

- At 1× the guardrail reads about 0.5 on both fixtures (the body's 0.49–0.51).
- If the loudness hypothesis is right, the relative fragile gap rises with
  scale on `value` and `confident_better` falls below 0.5 at 2× and 4×; if
  the auction's ordering carries no token information at any loudness, both
  stay where they are while the absolute gap grows as *c²* with the loss.
- `decoupled`'s gate reads a pinned equal wealth, so its executed route is the
  reports' top-k; if the reports carry token information the guardrail departs
  from 0.5 there first.

## Stages and cost

`--stage grid` on the quality fixture (three arms × three scales × two cell
counts × six seeds = 108 readings), then the differentiated fixture, then
`--stage summarise --self-model <the #73 grid record>`. CPU only; each reading
is one settle and 65 probe forwards.

## Outcome (2026-09-25, after the grid; code `7e99155`, `~/tame-runs/loudness-routing/`)

Recorded after the run and not a change to anything above. The guardrail
written above as the port's own check — at 1× the confident-token rate reads
about 0.5 on both fixtures — **did not reproduce**: 0.000 on the quality
`value` arm at 4 and 8 cells, 0.18–0.19 on the differentiated `value` arm,
0.02–0.29 on the other arms, against the body's 0.49. The port measures the
same statistic the body read did; the fixture's router is not at chance
because planted competence is what the bid ranking finds. So neither line is
read: "one lever" needs the guardrail to leave 0.5, "the market needs its own
fix" needs it to stay there. The 2× and 4× rows were read in the same
summarise pass as the 1× rows, after the guardrail had failed, and are
reported as descriptions (README `#loudness-routing`). What they describe:
loudness lowers the confident-token rate and the relative fragile gap on the
differentiated fixture on every arm alike, `value` and `decoupled` reading
the same at 4 cells; on the quality fixture, which starts at zero, both rise
at 4× and 4 cells. The correlation with #60's contrast at the derived rate is
+0.01 [−0.66, +0.68] (quality, five cells) and +0.10 [+0.02, +0.17]
(differentiated, six). The quality fixture at 4× and 8 cells was refused by
#73's derivation (no settled rate in six passes) and is absent from the grid.
The port is a separate script rather than a flag on the body script, so the
body path is untouched to the byte (deviation from the issue's Touches row).

## Correction (2026-09-25, after the review of the first outcome; supersedes the outcome above where they differ)

The outcome above read the confident-token rate as the body read it and
called the fixture's router "not at chance" on that number. The review found
the number is not the body's: the body's rate counted unmoved tokens the
stream beat about half the time, and on the fixture an unmoved token ties, so
the rate is diluted by the pairs no draw moved and its fall with loudness is
the draw moving fewer routes. The grids were re-run at `66f25f0` and then
`4e3ecd7` (the pairing hash had to cover the inputs only, since the targets
scale with *c*) with the conditional pair recorded — the confident pairs a
draw moved and the beaten-rate over them — and "better" read only on moved
tokens. Read on the conditional rate the differentiated fixture's `value`
arm sits at 0.36 (4 cells) and 0.25 (8 cells) at 1×, against the body's
0.49; the quality fixture stays at 0.000. So the guardrail still does not
reproduce and neither line is read, but "loudness makes the route more
load-bearing on every arm alike" is withdrawn: the fraction of confident pairs
a draw moves falls with loudness on every arm, the rate given a move rises at
4 cells and falls at 8 cells on the live and pinned arms alike, and the
relative fragile gap rises at 4 cells and falls at 8. The correlation with
#60's contrast now carries a Fisher-z interval and a permutation p beside the
seed bootstrap (r +0.01 and +0.10; Fisher [−0.88, +0.88] and [−0.78, +0.84];
p 0.99 and 0.88), the quality fixture's read being against the differentiated
#60 grid, the only one that ran. Two errors in the sections above are noted
rather than edited: the "one lever" line subtracts a loss-unit floor from a
relative gap (no column computes it; it was not read), and line 70's "dz −3.8
at 8 cells, 2×" is the 16-cell figure (8 cells 2× reads −1.66). The rule 8
guardrail is now reported per cell against the 1× seeds' range in the README.
