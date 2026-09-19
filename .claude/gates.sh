# Per-project quality gate commands. Anything left empty is skipped.
# Sourced by .claude/hooks/gate-lib.sh and read by /ship.
#
# Installed by #50. Before it there was no `.claude/` path in this repo at all:
# every turn ran with nothing checking the tree, and the pipeline was the only
# gate the project had. All three slots below are filled, each with a command
# confirmed by running it on prometheus — the test slot only since #51, which is
# why the note above it is longer than the command. That note is the argument
# rather than the conclusion, and it is kept in this file because the next person
# to weigh emptying the slot needs the measurements, not somebody's memory of
# them.
#
# `.pre-commit-config.yaml` runs ruff and pyright too, and it is not what this
# replaces: pre-commit fires once per commit, only in a clone where somebody ran
# `pre-commit install`, and by then the turn that introduced the error is over.

# uv lives in ~/.local/bin, which is on the PATH of a login shell but not of the
# non-login shell a hook inherits. Setting it here rather than relying on the
# caller's environment is deliberate: a gate that cannot find its own tools
# fails every turn and reads as "your code is broken".
export PATH="$HOME/.local/bin:$PATH"

# Measured on prometheus, 2026-09-19, twice, the same figure both times:
#   lint <1s · types 62s · total 62s
# The gate is pyright and nothing else — ruff walks all 116 files in about 60 ms
# with its cache deleted first, so there is no warm/cold distinction to record.
LINT_CMD="uv run ruff check . && uv run ruff format --check ."

# `tame scripts`, not a bare `pyright`: pyrightconfig.json includes only `tame`,
# and scripts/ is where the measurement code lives — the sweeps and probes whose
# numbers end up in the README. Naming both directories on the command line is
# what the issues' verification block asks for and is what CI's reviewer reads.
TYPE_CMD="uv run pyright tame scripts"

# 110, against the 98s the three gates measure together on prometheus (62s of it
# pyright, 34s the parallel suite). The default is 60, which this gate is over on
# every single run, and a row that is advisory every time is a row that stops
# being read. The margin is there to show drift from what the gates cost now, not
# to leave room for them to grow.
GATE_BUDGET_S="110"

# Filled since #51, and empty before it. What follows is the measurement #50
# exists to write down and the one #51 overturned, in that order, because the
# case for emptying it again is the first one.
#
# The Stop hook's timeout is 300s (.claude/settings.json). sophia's gates.sh
# records that a **143s** suite reliably pushed headless `/ship` turns into the
# background and killed two consecutive attempts to ship an issue: the command
# outlives the turn, the turn ends waiting for a notification from a process
# that is gone. This repo's CPU suite is 855 tests of the 861 collected — the
# other six are `-m gpu` — and CI's `test` job measured **894–1013s** in
# pipelines 462 and 466 (2026-09-16): seven times the figure that already broke
# it, and three times the timeout.
#
# The local figure is the one that decides it, because a Stop hook runs here and
# not on the runner. **Which invocation, exactly**, because the two differ and a
# reader who measures one and reads the other will think the file is wrong:
#
#   uv run pytest tests/ -m "not gpu"   852 passed + 3 xfailed   161s   ← CI's
#   uv run pytest                       851 passed + 3 xfailed   137s
#
# Both on prometheus, 2026-09-19. The gap is the single `slow` test that
# pyproject's `addopts` also deselect (tests/test_mixture.py:464). The first is
# the one to compare against CI, and 2m40s of wall against 30m34s of CPU says
# torch is already spending eleven cores on it.
#
# Neither is what a Stop invocation costs, and that is the number that decides
# the slot either way: `run_all_gates` runs lint and types first, so a *serial*
# suite here would make one Stop hook 62s + 161s ≈ **223s** against a 300s
# timeout — seventy seconds of headroom for a suite whose own run-to-run spread
# is tens of seconds, and half again past the 143s that already lost two ships.
# On those numbers #50 left the slot empty, and that reasoning is kept because
# it is the case for emptying it again.
#
# **#51 changed the input to it.** Under `-n auto` with each worker pinned to one
# torch thread (tests/conftest.py), the same 855 tests take **33s** on the same
# box — 12 workers, 3m49s of CPU against the serial run's 30m34s, because the
# tensors are hidden_dim 32 and torch's intra-op pool was claiming the cores a
# second time. A fifth of the figure that broke two ships, and the whole gate
# measures **98s** against the 300s timeout.
#
# So it is filled, and the cost is the thing to keep an eye on rather than the
# saving: 855 tests and twelve worker processes now run in front of every tree
# change in this repo, including the ones that only touch a docstring. sophia
# holds its own gate to a ~35s ceiling deliberately and this is nearly three
# times that, so if a turn here
# starts feeling like it is waiting on something, this is what it is waiting on —
# empty the slot rather than reaching for --no-verify.
#
# The command mirrors CI's `test` job exactly, `-x` and all. A gate that runs a
# different invocation from the pipeline can pass here and fail there, which is
# the one thing a pre-push gate must not do.
TEST_CMD="uv run pytest tests/ -x --tb=short -m 'not gpu' -n auto"

# One command, one file — post-edit-lint.sh applies these to the file just
# written, never to the tree.
FORMAT_CMD="uv run ruff format"
LINT_FILE_CMD="uv run ruff check --fix"
FORMAT_EXTENSIONS="py"

# --- the opt-in suites --------------------------------------------------------
# Neither of these is a gate. `run_all_gates` reads LINT_CMD, TYPE_CMD and
# TEST_CMD and nothing else, so nothing below ever runs on Stop.

# The GPU suite: 6 tests, the bitwise-determinism pair at the ablation
# configuration and the four real-model tissue tests, all of which load
# Qwen3-1.7B from the local HuggingFace cache. Measured 267s on the RTX 5070 Ti
# (README), 296s here on 2026-09-19 through the guard below, and a 295s median
# across the last 26 `test-gpu` jobs on the forge.
#
# Under 300s and so nominally inside the Stop hook's timeout, and still not
# TEST_CMD: 295s is double the 143s that already broke two headless ships, and
# it would put a five-minute model load in front of every tree change including
# the ones that only touch a docstring. The shape MUTATION_CMD already has.
#
# It runs behind scripts/gpu_gate.sh, and that is the part worth reading. The
# three CI jobs that touch the card share `resource_group: gpu`, which serialises
# them against each other and knows nothing about a local process — and the
# runner tagged `workstation` *is* this workstation. A local run that starts
# while CI holds the group will not corrupt the numerics; it will OOM the job or
# push it past the 300s budget its log asserts against, and a blown budget reads
# as a performance regression in the code. The guard asks the forge who holds the
# group and refuses rather than queueing, because a queue and a refusal reach a
# terminal as the same wait. TAME_GPU_GATE_FORCE=1 is the way past it.
GPU_CMD="scripts/gpu_gate.sh uv run pytest tests/ -x --tb=short -m gpu --durations=10"

# What the six tests reach, one hop deep, which is the only honest answer to "is
# this worth 295 seconds". Most of it is what the two test files import outright;
# coupling.py and pid_controller.py arrive through mob/ and homeostat.py, and the
# last seven through train.py, homeostat.py, steering_pipeline.py and
# contrastive_data.py. A change one hop out is just as capable of moving a
# bitwise-determinism result as a change to the file that imports it.
#
# tests/conftest.py is in the list since #51: it now decides how many torch
# threads a worker gets, which every test in the file sees, GPU ones included.
#
# Read by a person and by nothing else — no hook, skill or rule in ~/.claude
# names this key, so a path here changing never runs anything on its own.
#
# One line, and not for the reason it is tempting to give: `decision_doc` folds
# newlines before `read -ra` (gate-lib.sh:156) precisely so a wrapped value keeps
# its entries, and `glob_specs` splits on IFS, which includes newline. Both would
# take a wrapped list. One line is simply the shape every reader agrees on
# without anyone having to check which of them folds.
GPU_PATHS="tame/mob/** tame/steering.py tame/steering_pipeline.py tame/homeostat.py tame/pid_controller.py tame/coupling.py tame/determinism.py tame/train.py tame/behavioural_validation.py tame/contrastive_data.py tame/contrastive_templates.py tame/config.py scripts/smoke_fixture.py tame/evaluation.py tame/homeostat_calibration.py tame/metrics.py tame/parity.py tame/specialisation.py tame/tracking.py tame/contrastive_sources.py tests/conftest.py tests/test_determinism.py tests/test_real_model.py"

# Empty: nothing here runs mutmut yet. The `mutants/` line in .gitignore is
# aspiration rather than configuration. mob/auction.py and mob/wealth.py are the
# branchy, deterministic candidates when somebody picks them up.
MUTATION_CMD=""
MUTATION_PATHS=""

# --- the run contract ---------------------------------------------------------
# Every slot empty, and deliberately rather than by omission.
#
# /ship step 2c starts the product and walks a flow on it to prove a change
# works. There is a FastAPI surface here (tame/routes.py, tame/metrics_routes.py)
# and a Gradio chat UI, but bringing either up means loading Qwen3-1.7B onto the
# card — minutes, the GPU the CI jobs above are queueing for, and no sign-in to
# walk past. What that proof would cover is covered instead by tests/test_api.py
# and the metrics-surface tests, in the CPU suite, against the app factory.
#
# SURFACE_PATHS is empty for a second reason worth stating, because it is a
# tripwire: foreman.sh makes a surface named with no RUN_CMD a **required**
# finding, on the grounds that /ship skips the proof of use when RUN_CMD is
# empty, so naming a surface here without a way to run it would stop step 0 on
# every ship rather than buy anything.
RUN_CMD=""
STOP_CMD=""
READY_URL=""
SESSION_CMD=""
SURFACE_PATHS=""

# Empty, and not the same word as tame/parity.py — which is about arm
# fingerprints and refuses a comparison whose arms differ in anything but the
# gate. This key is about a test double drifting from the server it stands in
# for, and there is no double here: tests/test_api.py drives the real app
# factory through FastAPI's TestClient. Nothing to pin, rather than something
# unpinned.
PARITY_CMD=""
PARITY_PATHS=""

# The documents whose *meaning* is not an implementation branch's to change.
# The default list is `*SPEC*.md docs/decisions/** DECISIONS.md` and would match
# nothing in this repo, which would leave decision-doc-context.sh installed and
# inert. docs/preregistration.md is the one that matters: a preregistration
# edited to agree with the result it was registered against is the exact failure
# it exists to prevent, and it is a file an implementation branch has every
# ordinary reason to touch.
DECISION_DOCS="docs/preregistration.md docs/phase-2-stakes-plan.md"

# --- the pipeline this project pushes into ------------------------------------
# Nothing here runs a pipeline or changes what CI does; these only say whether
# something may stand still watching one, and where that standing still happens.

# tip-only. The CPU suite is this project's largest check and CI is the only
# place it runs, so the tip's pipeline is genuinely what stands between the work
# and main — but at the median below, waiting on every phase of a stacked
# milestone costs hours of the session window for an answer the tip repeats.
CI_WAIT="tip-only"

# 2700, against a default of 1800. The median is the number below; the slowest
# successful MR pipeline in the same sample was 2146s, and 1800 would have
# recorded that one as a timeout. A timeout never resolves to green, so erring
# short turns a pipeline that passed into a run that reports it never saw one.
CI_WAIT_TIMEOUT="2700"

# The observed median MR pipeline, and what a proposal to tier this project's CI
# has to argue against. Nothing reads it to decide anything.
CI_MR_SECONDS="753  # median of 48 successful MR pipelines, 2026-09-19"

# Empty, which is `yes`, and #50 did not flip it. Installing this file was the
# precondition for flipping it — before today there was no local gate at all and
# `yes` was the only honest answer — but it is not sufficient, for two reasons
# that are worth writing down here rather than rediscovering.
#
# The first is the split, which is what a skip would be taking a chance on. Lint
# and types run on every tree change, in-turn, over both `tame` and `scripts`.
# The 855-test CPU suite and the 6-test GPU suite run nowhere but CI. Anything
# tiered here would be tiered against static checks and never against the tests.
#
# The second is a trap, and it is why this is empty rather than `no` today.
# `ci_policy_outcome` reads `no` on a **draft** MR as `draft-skipped` — "the
# pipeline was skipped by policy" — and .gitlab-ci.yml here has no draft rule at
# all: `workflow:` runs a full merge_request_event pipeline on a draft, and !31
# is a draft with four of them. So `no` would not be inert while CI_TIER_PATHS is
# empty, as it looks; it would write "no check ran" into the ledger on every
# draft ship, for pipelines that ran and could have been red. That is the false
# all-clear these keys exist to prevent, arriving through the key itself.
#
# To flip it: land mipkovich/claude-config#36, and either give .gitlab-ci.yml the
# draft rule the value assumes or fix the assumption. Both are out of scope here,
# where changing .gitlab-ci.yml is a non-goal.
CI_IS_ONLY_GATE=""

# Empty is `full-ci`, which is the label this project would use.
CI_TIP_LABEL=""

# Empty: .gitlab-ci.yml defers nothing today, so this project reads
# `ci-policy=full` and behaves exactly as it did before these keys existed.
# Tiering the GPU jobs to the stack tip is out of scope for #50, which installed
# this file, and is blocked on mipkovich/claude-config#36 landing the
# `ci-policy=` line.
CI_TIER_PATHS=""
