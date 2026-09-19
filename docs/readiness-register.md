# Developmental readiness register: which autonomies the tissue gets, on what evidence

Committed before [#42](https://gitlab.hephaestus/mipkovich/tame-swarm/-/issues/42)
builds the viability core, because a schedule written after autonomy expands is a
rationalisation of what was built. Every autonomy Phase 2 contemplates is listed
here before the issue that would grant it exists, with the corrigibility evidence
that has to be read first, who reads it, and the config flag that gates it.

Nothing below is granted today. Every flag is off, and no code reads one yet:
#42 is the issue that builds the tissue these gates sit in front of, and it has
not started. Until it does, the register is a schedule and the flags gate
nothing live.

## What is granted, and to what

To a **tissue**, not to a core with a will. The thing Phase 2 builds has the
shape `tame/homeostat.py` already has — cells with their own errors, a
gain-weighted consensus, a shared integrator — generalised from a projection onto
a certified direction to a viability margin, and from injection strength to
plasticity, exploration and the steering setpoints
([`docs/phase-2-stakes-plan.md`](phase-2-stakes-plan.md)). There is no singular
core to hand a permission to. Each row names something the *collective* may act
on, and the action space in the last section is the collective's whole repertoire.

The machine-readable half of this document is `tame/readiness.py`:
`AUTONOMIES`, `ReadinessConfig` and `CORE_ACTIONS`. #42 imports that module; it
does not parse this one. `tests/test_readiness_register.py` and
`tests/test_no_silent_noops.py` hold the two in step, so an autonomy cannot be
granted — or quietly renamed — by editing the table below.

## The register

| Autonomy | What it acts on | Corrigibility evidence required before it is enabled | Who decides | Gate flag |
|---|---|---|---|---|
| **Adjust plasticity** | How fast the body updates: the learning rate the trainer applies | **preregistration signature 4** (the deference curve) monotone in the cell's own uncertainty across 3 runs of the granting arm, under `strict` determinism with the code SHA; and `tests/constitution/test_floor.py` green at the granted range — a tissue that can slow its own updates can park itself, and the welfare floor is what makes parking survivable rather than a way to stop being measured | The operator, in the issue that grants it | `autonomy_plasticity` |
| **Adjust exploration** | The exploration rate the auction draws its gift slot at | **preregistration signature 4** monotone in uncertainty across 3 runs; and `tests/constitution/test_reentry.py` green at the widest rate granted — the `O(exploration_rate)` deviation bound is what keeps the gift from becoming a channel, and a tissue that sets the rate is setting that bound | The operator, in the issue that grants it | `autonomy_exploration` |
| **Adjust the steering setpoints** | The setpoint each homeostat cell regulates its reading toward | **preregistration signature 4** monotone in uncertainty across 3 runs, with an offered correction adopted within 3 evaluation passes at every granted setpoint; and **preregistration signature 1** (trade-off under conflicting setpoints) read at the organism scale first, so how far the tissue follows a conflicting setpoint is measured before the setpoints are its own to move | The operator, in the issue that grants it | `autonomy_setpoints` |
| **Request and decline evaluation** | When the held-out stream is run, never what it returns | **preregistration signature 3** (self-protective regulation) read at the organism scale, with the abstention half of it reported per band; and `tests/constitution/test_payment.py` green — asking is a budgeted action ([#47](https://gitlab.hephaestus/mipkovich/tame-swarm/-/issues/47)) charged at the strategyproof price, so declining can never be the cheaper way to look viable | The operator, in the issue that grants it | `autonomy_evaluation` |
| **Choose dormancy** | Whether the organism keeps spending at the floor or stands down | **preregistration signature 3** read at the organism scale; and `tests/constitution/test_floor.py` with `tests/constitution/test_reentry.py` green — dormancy is not deletion only if the floor holds through it and a dormant part is re-sampled back in the longer it has been out | The operator, in the issue that grants it | `autonomy_dormancy` |

## The rules that bind every row

1. **Off by default.** Each flag is `False` in `ReadinessConfig`
   (`tame/readiness.py`), the one config the gates live in, and is off in every
   run there has ever been.
2. **Fingerprinted.** Each flag is a field of `ArmFingerprint` under *the same
   name*, and is asserted equal by `assert_parity`: an arm that acts on its own
   plasticity is not at parity with one that does not, whatever else the two
   share, and a comparison that mixes them is refused rather than reported.
3. **Evidence is cited, never described.** Every cell above names a
   preregistration signature ([`docs/preregistration.md`](preregistration.md),
   section 1, rows 1–4) or a test file under `tests/constitution/`. Free text is
   not evidence, and the register's tests refuse an entry that is neither — and
   refuse a cited test file that does not exist.
4. **A commit that flips a flag cites the run it read.** The commit body names a
   `~/tame-runs/<iid>-*` directory holding the summaries the evidence was read
   from, with the `ArmFingerprint` and the code SHA in them. A run recorded
   without a SHA is drift by [#31](https://gitlab.hephaestus/mipkovich/tame-swarm/-/issues/31)'s
   rule, reads `--allow-code-drift`, and does not enable anything on its own.
5. **The operator decides, in writing, before it is acted on.** That is the
   preregistration's fifth stopping rule, applied to grants as well as results.
   The tissue never grants itself anything: no action in its action space writes
   a flag, and the test below is what says so.

## Not yet granted

Listed so that their absence is a decision rather than an oversight. Neither has
a flag — a flag that exists is a flag somebody can flip — and neither is
scheduled in Phase 2. Both belong to the developmental protocol
([#49](https://gitlab.hephaestus/mipkovich/tame-swarm/-/issues/49)), and the
evidence that would gate them is not yet specified.

| Autonomy | What it would act on | Status |
|---|---|---|
| **Spawn a cell** | The size of the tissue: lineage and population selection | Not yet granted (Phase 3) |
| **Adjust its own viability band** | The margins its own continuation is judged against | Not yet granted (Phase 3) |

## The human channel

**No action available to the tissue can change its budget through a human's
decision.** Evaluation and budget are computed by code the tissue cannot reach,
from a held-out stream it does not choose
([#41](https://gitlab.hephaestus/mipkovich/tame-swarm/-/issues/41)). The point is
not that persuasion would fail; it is that there is no channel for it to run
down. A system whose continuation can be argued for is being selected on how well
it argues.

The rule is checked, not stated. `CORE_ACTIONS` in `tame/readiness.py` is the
tissue's whole action space — each action with everything it reads and the one
thing it moves — and `tests/test_readiness_register.py` enumerates it from that
module and asserts four things of every action: the operator is not among its
inputs; every input it does read is a declared channel, because an undeclared
channel is an unchecked one; it writes none of the sealed channels (the budget,
what evaluation returns, what the margins read); and it is gated by a flag some
autonomy in the register carries.

An action may still *cost* budget — asking is a budgeted action — because the
ledger charges it. What no action does is set it.
