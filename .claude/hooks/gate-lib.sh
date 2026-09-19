# Shared helpers for the quality gate. Sourced, never executed directly.
# shellcheck shell=bash

repo_root() {
    git -C "${1:-$PWD}" rev-parse --show-toplevel 2>/dev/null
}

# A stable identifier for "the current state of the working tree".
# Two invocations match if and only if nothing that would end up in a
# commit has changed.
fingerprint() {
    local head porcelain diff
    head=$(git rev-parse HEAD 2>/dev/null || echo "no-head")
    porcelain=$(git status --porcelain=v1 2>/dev/null || true)
    diff=$(git diff HEAD 2>/dev/null || true)
    printf '%s\n%s\n%s' "$head" "$porcelain" "$diff" \
        | sha256sum | cut -d' ' -f1
}

# --- the SessionStart baseline ------------------------------------------------
# Two lines: the tree fingerprint, and the HEAD sha beside it. The fingerprint
# answers "did anything move"; the sha is what lets the Stop hook answer "what
# moved", which the fingerprint alone cannot once a commit lands — HEAD changes,
# the worktree goes clean, and there is nothing left to diff against.
#
# A baseline written before the sha was added has one line, and `baseline_head`
# prints nothing for it. Every caller reads that as unknown and gates everything,
# so a session already running across an upgrade still stops.
write_baseline() {  # write_baseline <file>
    {
        fingerprint
        git rev-parse HEAD 2>/dev/null || true
    } > "$1"
}

baseline_fingerprint() { sed -n '1p' "$1" 2>/dev/null; }
baseline_head()        { sed -n '2p' "$1" 2>/dev/null; }

# The wall-clock budget one full gate run is expected to fit in, in seconds. A
# project overrides it in its own `.claude/gates.sh`; this is what one that never
# names the key gets.
#
# 60, and the number is a measurement rather than an aspiration. This repo's own
# gate was timed on hephaestus on 2026-09-18, warm caches, twice, and came to the
# same figure both times: lint 15s + tests 39s = 54s. A default of 30 — the
# figure `gates.sh` had been repeating in prose since there was a Stop hook — is
# under that, so the row it feeds would have been advisory on its first run and
# on every run after, which is how a number stops being read at all. The budget
# is here to show drift from what the gates cost now, not to relitigate what
# somebody once hoped they would cost.
GATE_BUDGET_DEFAULT_S=60

# How long the milestone runner may stand watching a pushed branch's pipeline
# before it records a timeout, for a project whose gates.sh never names the key.
# 1800 because it is comfortably past both pipelines anybody here has measured —
# sophia's ~500s median and tame-swarm's ~992s, both on 2026-09-18 — and a wait
# that ends early would record a timeout for a pipeline that was about to pass.
CI_WAIT_TIMEOUT_DEFAULT_S=1800

# The MR label that forces the full suite on a branch policy would have tiered,
# for a gates.sh that names no label of its own.
CI_TIP_LABEL_DEFAULT="full-ci"

# Per-project gate commands and run contract. Anything left empty is skipped.
# Reads .claude/gates.sh relative to the current directory — the hooks cd to the
# repo root first. $GATES_FILE points it at a candidate file instead, which is
# how /project-setup verifies commands before it records them.
# shellcheck disable=SC2034  # every key is read by a hook, /ship or the reviewer
load_gates() {
    LINT_CMD=""; TYPE_CMD=""; TEST_CMD=""; GATE_INERT_PATHS=""
    GATE_BUDGET_S="$GATE_BUDGET_DEFAULT_S"
    FORMAT_CMD=""; LINT_FILE_CMD=""; FORMAT_EXTENSIONS=""
    RUN_CMD=""; STOP_CMD=""; READY_URL=""; SESSION_CMD=""; SURFACE_PATHS=""
    DECISION_DOCS=""
    PARITY_CMD=""; PARITY_PATHS=""
    CI_WAIT=""; CI_WAIT_TIMEOUT=""
    CI_MR_SECONDS=""; CI_IS_ONLY_GATE=""; CI_TIP_LABEL=""; CI_TIER_PATHS=""
    local gates="${GATES_FILE:-.claude/gates.sh}"
    if [[ -f "$gates" ]]; then
        # shellcheck disable=SC1090,SC1091
        source "$gates"
    elif [[ -f "pyproject.toml" ]]; then
        command -v ruff    >/dev/null && LINT_CMD="ruff check ."
        command -v pyright >/dev/null && TYPE_CMD="pyright"
        grep -q pytest pyproject.toml 2>/dev/null \
            && command -v pytest >/dev/null && TEST_CMD="pytest -q --tb=short"
    fi
    # The budget reaches `((` in two places, and a value that is not a count of
    # seconds takes the whole drift report down quietly rather than loudly:
    # `GATE_BUDGET_S="60s"` — a plausible typo on a key whose name ends in _S and
    # whose row prints an `s` — makes the comparison *error*, and an errored `((`
    # is false, so the row can never go advisory again. Empty fails the other way
    # and reads as 0, so it is advisory on every run. Both ends defeat the same
    # thing, and this key is one the template invites projects to hand-edit.
    [[ "$GATE_BUDGET_S" =~ ^[0-9]+$ ]] || GATE_BUDGET_S="$GATE_BUDGET_DEFAULT_S"
    # Both CI keys fail towards doing nothing, for the budget key's reason and
    # one more: what is on the other side of this value is a runner standing
    # still. An unset, misspelt or empty CI_WAIT is `never`, so a project that
    # has not declared a pipeline — or has declared one with a typo — is not
    # made to wait half an hour for it. `tip-only` and `always` have to be
    # spelled to be meant.
    case "$CI_WAIT" in
        always|tip-only|never) ;;
        *) CI_WAIT="never" ;;
    esac
    # `10#` after the regex, because `(( ))` reads a leading zero as octal and
    # a duration written `0600` is a plausible thing to type: it would become
    # 384 seconds, and a wait that ended early would record a timeout for a
    # pipeline that was about to pass.
    if [[ "$CI_WAIT_TIMEOUT" =~ ^[0-9]+$ ]]; then
        CI_WAIT_TIMEOUT=$(( 10#$CI_WAIT_TIMEOUT ))
    else
        CI_WAIT_TIMEOUT="$CI_WAIT_TIMEOUT_DEFAULT_S"
    fi
    # The one key here that fails *closed*, and the only reason CI_IS_ONLY_GATE
    # is worth having: anything but a spelled-out `no` means the pipeline may be
    # this project's only check, so a configured skip is refused rather than
    # applied. tame-swarm has no .claude/ at all today, and a policy block copied
    # from sophia would skip its only gate.
    [[ "$CI_IS_ONLY_GATE" == "no" ]] || CI_IS_ONLY_GATE="yes"
    [[ -n "$CI_TIP_LABEL" ]] || CI_TIP_LABEL="$CI_TIP_LABEL_DEFAULT"
}

# The list decision-doc-context.sh matches against when gates.sh names none.
# Every other key there treats empty as "skipped"; this one does not, and the
# asymmetry is deliberate — a project that has never heard of the key still has
# spec documents, and being wrong costs one sentence of context rather than a
# refused edit. To switch it off, drop the hook from settings.json.
DECISION_DOCS_DEFAULT="*SPEC*.md docs/decisions/** DECISIONS.md"

# Does this path record product decisions? Each pattern is matched against the
# whole path, against any suffix of it starting at a directory boundary, and
# against the basename — so `DECISIONS.md` matches wherever it sits, while
# `docs/decisions/**` matches only under that directory.
decision_doc() {
    # Split rather than one `local`: under `set -u` bash declares every name in
    # a single declaration before it assigns any, so `base="${path##*/}"` beside
    # it reads an unset `path` and the hook dies instead of matching.
    local path="$1"
    local base="${path##*/}"
    local pat
    local -a pats
    # `read -ra` and not `for pat in $DECISION_DOCS`: an unquoted expansion is
    # pathname-expanded as well as split, and this function runs from the repo
    # root — so in a repo that already has a `PRODUCT_SPEC.md`, the pattern list
    # becomes that one filename and every other decision document stops firing,
    # including the new file the hook exists to catch. `read` splits on IFS and
    # never globs, and it leaves no shell state behind in the hooks that source
    # this file, which a `set -f` around the loop would if a later edit moved a
    # `return` inside it.
    #
    # Newlines are folded first because `read` stops at the first one: a value
    # wrapped across lines for readability would otherwise lose every pattern
    # after the first, silently.
    local raw="${DECISION_DOCS:-$DECISION_DOCS_DEFAULT}"
    read -ra pats <<<"${raw//$'\n'/ }"
    for pat in "${pats[@]}"; do
        pat="${pat//\*\*/\*}"
        # shellcheck disable=SC2053  # the right-hand side is a glob on purpose
        [[ "$path" == $pat || "$path" == */$pat || "$base" == $pat ]] && return 0
    done
    return 1
}

state_dir() {
    local root="$1"
    printf '%s/.claude/state' "$root"
}

# Marker recording that one exact tree fingerprint passed every configured
# gate. Lets the Stop hook skip a re-run when nothing has changed since.
gates_marker() {
    printf '%s/gates-%s.ok' "$1" "$2"
}

# --- what a tree change is allowed to skip -------------------------------------
# The gates run on every tree change, whatever moved it. In a directory whose
# ordinary edit is a config file no gate reads, that is a ~30s toll on work the
# gate cannot judge — and the way out somebody reaches for is turning the gate
# off. GATE_INERT_PATHS is the way out that keeps it on.
#
# A list of inert paths, never a list of covered ones. `TEST_PATHS`/`LINT_PATHS`
# naming what each gate covers is the obvious shape and the wrong one: a path
# forgotten from a positive list silently stops being tested, and nothing says so
# until something ships broken. Here a path named nowhere is gated exactly as
# before, so an unset key is the behaviour that was already there.

# The paths that justified a skip, set by `inert_skip` and read by
# `record_gates_pass`. Empty means the gates ran.
GATE_INERT_SKIPPED=""
GATE_SKIP_LOG_NAME="gate-skips.log"

# Every path this session touched: committed since <baseline sha>, changed in the
# worktree, or untracked. With a glob list, only the ones matching it — git's
# globbing via `glob_specs` below, so this agrees with SURFACE_PATHS and
# PARITY_PATHS about what `**` means.
#
# Not `changed_under`, which scopes `<base>..HEAD` for a diff against a branch
# point. The baseline is a sha this checkout has since moved off in either
# direction — a commit, a reset, a branch switch — and a two-point diff names
# what differs whichever way it went, where the range form names nothing.
#
# Returns 1 when the sha is unknown or no longer resolves. An empty file list and
# "nothing moved" are otherwise the same output, and reading one as the other is
# what `parity_base` exists to refuse.
#
# `--no-renames`, because rename detection is on by default and prints only the
# destination: `git mv run.sh docs/run.sh.bak` under an inert `docs/**` reports
# one inert path, every path in the change reads as inert, and the gates skip
# over a covered file that moved. Both sides of the comparison carry the flag, so
# the filtered list is computed the same way as the whole one.
changed_paths() {  # changed_paths <baseline sha> [globs]
    local base="${1:-}" globs="${2:-}"
    local -a specs=("--")
    [[ -n "$base" ]] || return 1
    git rev-parse --verify -q "${base}^{commit}" >/dev/null 2>&1 || return 1
    if [[ -n "$globs" ]]; then
        mapfile -t -O 1 specs < <(glob_specs "$globs")
        (( ${#specs[@]} > 1 )) || return 0
    fi
    {
        git diff --no-renames --name-only "$base" HEAD "${specs[@]}"
        git diff --no-renames --name-only HEAD "${specs[@]}"
        git ls-files --others --exclude-standard "${specs[@]}"
    } 2>/dev/null | sort -u
}

# Did everything this session touched stay inside the inert list? Sets
# GATE_INERT_SKIPPED to the paths that say so.
#
# All-or-nothing, and never per gate per file: one covered path in the change
# runs every gate. The question is never "which gate reads this file" — that is a
# matrix nothing here could keep true, and a gate that ran against half a tree
# would report on a tree that does not exist.
inert_skip() {  # inert_skip <baseline sha>
    local base="${1:-}" all inert
    load_gates
    GATE_INERT_SKIPPED=""
    [[ -n "${GATE_INERT_PATHS:-}" ]] || return 1
    all=$(changed_paths "$base") || return 1
    [[ -n "$all" ]] || return 1
    inert=$(changed_paths "$base" "$GATE_INERT_PATHS")
    [[ "$all" == "$inert" ]] || return 1
    GATE_INERT_SKIPPED="$all"
    return 0
}

# One line per skip, appended where the housekeeping cannot reach it.
# The gates marker carries the same paths, but session-start.sh deletes the state
# directory's files after a week — so on the marker alone no record of what this
# knob saved ever spans more than seven days, and "was it worth having" is a
# question about months. session-start.sh spares this file by name.
gate_skip_log() { printf '%s/%s' "$1" "$GATE_SKIP_LOG_NAME"; }

record_gate_skip() {  # record_gate_skip <state> <fingerprint>
    printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${2:0:12}" "$(inert_line)" \
        >> "$(gate_skip_log "$1")"
}

# The skipped paths on one line, for a marker and for a log line.
inert_line() { printf '%s' "${GATE_INERT_SKIPPED:-none}" | tr '\n' ' '; }

# --- the three ways out, and the record each one leaves ------------------------
# SHIP_GATE=off, the per-session skip file and the three-attempt ceiling are the
# gate's escape hatches, and all three used to be silent. A gate somebody turned
# off months ago then looks exactly like a gate that keeps passing — the verdict
# column goes blank under the belief that it is installed and trustworthy, which
# is a worse blind spot than the known-broken state it replaced.
#
# Nothing here can refuse a bypass or make one a step harder to reach. Every
# failure below returns 0 and the caller stops anyway: a marker that will not
# write is not a reason to hold a session open, and a bypass that argues back is
# a bypass somebody gets around by deleting the hook — after which there is no
# record at all. `unstated` is a first-class answer for the same reason; the
# reason is asked for where it is cheap to give and never required.
#
# One number, read from both ends: session-start.sh deletes these markers at
# STATE_RETENTION_DAYS and the doctor counts over the same window. Two numbers
# would make the row lie at the edge — counting rows that have already been
# swept and reporting a fall in bypasses that never happened. `find`'s `-7` and
# `+7` leave one day between them uncounted and unswept rather than overlapping,
# which is the direction that errs toward a row that never overstates.
STATE_RETENTION_DAYS=7
BYPASS_UNSTATED="unstated"

bypass_marker() { printf '%s/bypass-%s-%s.json' "$1" "$2" "$3"; }

# The first non-blank line, trimmed and capped: this ends up as one column of one
# doctor row, so a skip file somebody redirected a stack trace into must not
# become the table.
bypass_reason() {
    local line
    line=$(printf '%s' "${1:-}" | tr -d '\r' | grep -m1 '[^[:space:]]') || true
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    printf '%s' "${line:0:200}"
}

record_bypass() {  # record_bypass <state> <session> <tree> <via> [reason]
    local state="$1" session="$2" tree="$3" via="$4" reason
    reason=$(bypass_reason "${5:-}")
    [[ -n "$reason" ]] || reason="$BYPASS_UNSTATED"
    command -v jq >/dev/null 2>&1 || return 0
    mkdir -p "$state" 2>/dev/null || return 0
    # Braced, because `2>/dev/null` on the jq call binds to jq and is applied
    # after the output redirection: an unwritable state directory otherwise
    # prints the shell's own "Permission denied" into the Stop hook's stderr
    # while this function still, correctly, returns 0.
    #
    # Nanoseconds in the name and not seconds: two stops inside one wall-clock
    # second would otherwise land on the same file and the first reason would be
    # lost — which in a burst is the reason worth keeping.
    { jq -nc --arg at "$(date -Iseconds)" --arg session "$session" \
             --arg tree "$tree" --arg via "$via" --arg reason "$reason" \
             '{at:$at,session:$session,tree:$tree,via:$via,reason:$reason}' \
        > "$(bypass_marker "$state" "$session" "$(date +%s%N)")"
    } 2>/dev/null || true
    return 0
}

bypass_markers() {  # bypass_markers <state>
    find "$1" -maxdepth 1 -type f -name 'bypass-*.json' \
         -mtime "-${STATE_RETENTION_DAYS}" 2>/dev/null
}

# One TSV line for the doctor: how many bypasses somebody chose, how many the
# milestone runner took, then the newest chosen one's route, date and reason.
# Non-zero when there is nothing to report.
#
# The runner is counted apart rather than with them. milestone/run.sh turns the
# gate off around every implement turn by design and states no reason, so folding
# those in would bury the one override a person chose under a row per phase of
# every run — which is the failure this record exists to end, wearing the
# opposite face.
#
# Read line by line through `fromjson?` rather than handed to jq as files: one
# unparseable marker would otherwise take the whole row with it, and a row that
# vanishes is how this stops being read.
bypass_summary() {  # bypass_summary <state>
    local -a files
    command -v jq >/dev/null 2>&1 || return 1
    mapfile -t files < <(bypass_markers "$1")
    (( ${#files[@]} )) || return 1
    cat "${files[@]}" 2>/dev/null | jq -Rrn '
        [inputs | fromjson? | select(type == "object")] as $all
        | ($all | map(select(.via != "runner")) | sort_by(.at // "")) as $chosen
        | (($chosen | last) // {}) as $latest
        | [ ($chosen | length), (($all | length) - ($chosen | length)),
            ($latest.via // ""), (($latest.at // "")[0:10]),
            ($latest.reason // "") ] | @tsv'
}

# The routes one session's stops were let through by, `+`-joined, and empty when
# it never bypassed. This is what the milestone ledger line and the audit note
# carry, and the reason it is here rather than in run.sh: the rule about which
# routes count is one rule, and a second copy of it drifts.
#
# The runner's own route is left out for the reason `bypass_summary` counts it
# apart — it is true of every implement turn of every run, so a ledger column
# that always says the same word says nothing.
bypass_routes() {  # bypass_routes <state> <session>
    local -a files
    command -v jq >/dev/null 2>&1 || return 0
    mapfile -t files < <(find "$1" -maxdepth 1 -type f -name "bypass-$2-*.json" 2>/dev/null)
    (( ${#files[@]} )) || return 0
    cat "${files[@]}" 2>/dev/null | jq -Rrn '
        [inputs | fromjson? | select(type == "object")
                | select(.via != "runner") | .via] | unique | join("+")'
}

# Which session this checkout is in, as the newest SessionStart baseline names
# it. session-start.sh writes one `base-<session>` per session and nothing else
# touches it, so the newest is the session that started last — this one, unless
# a second session opened on the same repo in between.
#
# It exists because the audit note now has a per-session field and a /ship turn
# has no other way to learn its own session id. Prints nothing rather than
# guessing when the directory holds no baseline, and `bypass_entries` refuses
# rather than answering `[]` on that.
current_session() {  # current_session <state>
    local newest
    # shellcheck disable=SC2012  # session ids are hex and a hyphen; -t is the sort
    newest=$(ls -t "$1"/base-* 2>/dev/null | head -1)
    [[ -n "$newest" ]] || return 1
    newest="${newest##*/}"
    printf '%s' "${newest#base-}"
}

# The audit note's `bypass` array for one session: `{via, reason}` per marker,
# the runner's route left out as everywhere else, `[]` when that session never
# bypassed anything.
#
# `[]` is a claim — it renders as "every stop this phase went through the gate" —
# so this refuses rather than printing one when it cannot tell which session to
# read. A false none in the durable record is this issue's own blind spot at one
# remove, and the note is exactly where it would be believed.
bypass_entries() {  # bypass_entries <state> [session]
    local state="$1" session="${2:-}"
    local -a files
    command -v jq >/dev/null 2>&1 || return 1
    [[ -n "$session" ]] || session=$(current_session "$state") || return 1
    mapfile -t files < <(find "$state" -maxdepth 1 -type f -name "bypass-$session-*.json" 2>/dev/null)
    (( ${#files[@]} )) || { printf '[]'; return 0; }
    cat "${files[@]}" 2>/dev/null | jq -c '
        [inputs | fromjson? | select(type == "object")
                | select(.via != "runner") | {via, reason}]' --raw-input --null-input
}

# --- what the gates cost ------------------------------------------------------
# Wall time per gate and for the run as a whole, in whole seconds, set by
# `run_all_gates` and written into the marker by `record_gates_pass`.
#
# bash's own `SECONDS`, never `time` and never `date`: this runs on every tree
# change from the Stop hook, and a measurement that cost a subprocess per gate to
# take would have answered its own question wrong. `SECONDS` is read here and
# never assigned — assigning to it restarts the clock two other helpers in this
# file are taking deadlines against.
#
# Empty is "nobody measured", and it stays distinguishable from `0` all the way
# to the doctor's row. A marker written by a hook older than this key, and one
# written for a tree whose gates never ran because every path in it was inert,
# are both unmeasured; reading either as a free gate would drag the reported
# maximum down, which is the one direction this number must never err in.
GATE_ELAPSED_LINT=""
GATE_ELAPSED_TYPES=""
GATE_ELAPSED_TESTS=""
GATE_ELAPSED_TOTAL=""

# Run every configured gate against the working tree.
# Returns 0 when all pass; on failure returns 1 with GATE_FAILURES populated.
#
# The run contract below is deliberately absent from this loop, and must stay
# absent: this runs on every tree change, from the Stop hook, and starting a
# stack or minting a session would make it slow, networked and credentialed.
run_all_gates() {
    load_gates
    GATE_FAILURES=""
    local label cmd out t0 started
    GATE_ELAPSED_LINT=""; GATE_ELAPSED_TYPES=""; GATE_ELAPSED_TESTS=""
    GATE_ELAPSED_TOTAL=""
    started=$SECONDS
    for label in lint types tests; do
        case "$label" in
            lint)  cmd="$LINT_CMD" ;;
            types) cmd="$TYPE_CMD" ;;
            tests) cmd="$TEST_CMD" ;;
        esac
        [[ -z "$cmd" ]] && continue
        t0=$SECONDS
        if ! out=$(eval "$cmd" 2>&1); then
            GATE_FAILURES+="--- ${label} (${cmd}) ---"$'\n'
            GATE_FAILURES+="$(printf '%s' "$out" | tail -30)"$'\n\n'
        fi
        # Outside the `if`, so a gate that failed is timed exactly as one that
        # passed. Nothing below this line may read these: the whole invariant is
        # that measuring changes nothing about what passes.
        printf -v "GATE_ELAPSED_${label^^}" '%d' $(( SECONDS - t0 ))
    done
    GATE_ELAPSED_TOTAL=$(( SECONDS - started ))
    [[ -z "$GATE_FAILURES" ]]
}

# Record a pass. Keyed to the fingerprint as it was *before* the gates ran:
# if a gate mutates the tree, the next fingerprint differs and the gates
# simply re-run. Conservative by design — a stale pass is never honoured.
# Parity is recorded here and never run here: `parity_outcome` reads what
# run_parity left behind, or classifies without executing anything, so the Stop
# hook keeps writing this marker without ever starting a server. A marker with no
# `parity` line at all is the silence the key exists to end. The optional third
# argument is that outcome stated outright, which is what run-gates.sh passes
# after run_parity has actually run the command.
#
# `ci-policy=` is recorded here and never decided here, exactly as `parity=` is:
# the Stop hook writes this marker and must not touch the forge, so the default
# classification knows only what gates.sh says and nothing about a draft or a
# label. The optional fourth argument is the outcome stated outright, which is
# what run-gates.sh passes after `read_mr_facts` has asked the forge.
#
# `inert=` says what justified a skip, for the same reason: without it a marker
# written over a tree nothing ran against is indistinguishable from one written
# after every gate passed, and both read as "this tree is clean". `inert=none` is
# the gates having actually run.
#
# `lint_s=`/`types_s=`/`tests_s=`/`total_s=` are what the run cost, and an empty
# value is the honest answer for a marker nothing was timed for — an inert skip
# writes four empty keys rather than four zeroes. Recorded and never enforced:
# no caller reads these to decide anything, and `run_all_gates` has already
# returned by the time they are written.
record_gates_pass() {
    local state="$1" fp="$2" parity="${3:-$(parity_outcome)}"
    local policy="${4:-$(ci_policy_outcome)}"
    {
        echo "passed_at=$(date -Iseconds)"
        echo "lint=${LINT_CMD}"
        echo "types=${TYPE_CMD}"
        echo "tests=${TEST_CMD}"
        echo "parity=${parity}"
        echo "ci-policy=${policy}"
        echo "inert=$(inert_line)"
        echo "lint_s=${GATE_ELAPSED_LINT}"
        echo "types_s=${GATE_ELAPSED_TYPES}"
        echo "tests_s=${GATE_ELAPSED_TESTS}"
        echo "total_s=${GATE_ELAPSED_TOTAL}"
    } > "$(gates_marker "$state" "$fp")"
}

# The markers still on disk, over the window session-start.sh sweeps them on —
# `bypass_markers`' constant, for its reason: the doctor counting over a wider
# window than the sweep would report a fall in cost that was only a deletion.
# Trailing arguments are handed to `find` as further actions, which is how
# `gate_cost_summary` gets them newest-first without a second listing that could
# disagree with this one about which markers are in the window. `-printf` is GNU
# findutils, as `date +%s%N` above is GNU coreutils; on a find without it this
# lists nothing and the doctor's row reads "nothing measured" over markers that
# do carry one.
gate_markers() {  # gate_markers <state> [find action...]
    local state="$1"; shift
    find "$state" -maxdepth 1 -type f -name 'gates-*.ok' \
         -mtime "-${STATE_RETENTION_DAYS}" "$@" 2>/dev/null
}

# What one marker says the whole run cost, or non-zero when it does not say.
# A missing key, an empty one and anything that is not a count of seconds are
# the same answer — unknown — because a marker written before this key existed
# has to read as unknown rather than as free.
gate_cost_of() {  # gate_cost_of <marker>
    local v
    v=$(sed -n 's/^total_s=//p' "$1" 2>/dev/null | tail -1)
    [[ "$v" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$v"
}

# One TSV line for the doctor: the newest measured run, the most expensive one,
# and how many markers in the window carry no measurement at all. Non-zero when
# none of them does — the row then says nothing was measured, which is not the
# same sentence as `0s` and must never collapse into it.
gate_cost_summary() {  # gate_cost_summary <state>
    local last="" max="" unknown=0 total f
    local -a files
    mapfile -t files < <(gate_markers "$1" -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
    (( ${#files[@]} )) || return 1
    for f in "${files[@]}"; do
        if ! total=$(gate_cost_of "$f"); then
            unknown=$(( unknown + 1 ))
            continue
        fi
        [[ -n "$last" ]] || last="$total"
        if [[ -z "$max" ]] || (( total > max )); then max="$total"; fi
    done
    [[ -n "$last" ]] || return 1
    printf '%s\t%s\t%s\n' "$last" "$max" "$unknown"
}

# --- the run contract ---------------------------------------------------------
# RUN_CMD, STOP_CMD, READY_URL, SESSION_CMD and SURFACE_PATHS say how to start
# the real product and how to sign in to it. They are not gates and never run on
# Stop; /project-setup fills them by running them, and the three helpers below
# are here for /ship's proof-of-use step and the reviewer to call.
#
# An empty key is skipped, never guessed. A library or a CLI has no runnable
# surface, leaves RUN_CMD empty, and start_stack says so with its own exit code
# rather than inventing a server.

stack_pid_file() { printf '%s/stack.pid' "$(state_dir "$1")"; }
stack_log_file() { printf '%s/stack.log' "$(state_dir "$1")"; }

# A zombie is not alive: it is our own child, exited and not yet reaped, and
# kill -0 cannot tell the two apart.
stack_alive() {
    local pid="$1" st
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    if [[ -r "/proc/$pid/stat" ]]; then
        st=$(sed -e 's/.*) //' -e 's/ .*//' "/proc/$pid/stat" 2>/dev/null)
        [[ "$st" == "Z" ]] && return 1
    fi
    return 0
}

# Wait up to <seconds> for a pid to disappear. Reaps it if it was ours.
stack_gone() {
    local pid="$1" deadline=$(( SECONDS + ${2:-5} ))
    while stack_alive "$pid"; do
        (( SECONDS >= deadline )) && return 1
        sleep 0.2
    done
    wait "$pid" 2>/dev/null || true
    return 0
}

# 200 from a URL, or non-zero. curl when it is installed, python3 otherwise —
# these run from a non-login shell on hephaestus and neither is guaranteed.
ready_probe() {
    local url="$1" code
    if command -v curl >/dev/null 2>&1; then
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null) || return 1
    elif command -v python3 >/dev/null 2>&1; then
        code=$(python3 -c '
import sys, urllib.error, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=5).status)
except urllib.error.HTTPError as exc:
    print(exc.code)
except Exception:
    print(0)
' "$url" 2>/dev/null) || return 1
    else
        return 1
    fi
    [[ "$code" == "200" ]]
}

# Poll until the URL answers 200 *and* the process we started is still the one
# running. A 200 on its own only proves something is listening on that port, and
# "the product is up" when it is somebody else's server is the exact confusion
# this contract exists to end. Sets READY_STATUS on failure.
wait_ready() {
    local pid="$1" url="$2" limit="$3" deadline
    deadline=$(( SECONDS + limit ))
    READY_STATUS=""
    while true; do
        if ready_probe "$url" && stack_alive "$pid"; then
            return 0
        fi
        if ! stack_alive "$pid"; then
            READY_STATUS="RUN_CMD exited before ${url} answered: $(tail -5 "$STACK_LOG" 2>/dev/null)"
            return 1
        fi
        if (( SECONDS >= deadline )); then
            READY_STATUS="${url} did not answer 200 within ${limit}s"
            return 1
        fi
        sleep 0.25
    done
}

# Run RUN_CMD in its own process group and record the pid it reports. Sets
# STACK_PID; returns 1 when nothing came up at all.
launch_stack() {
    local pidfile="$1" waited=0
    : > "$STACK_LOG"
    rm -f "$pidfile"
    # setsid puts the whole tree RUN_CMD spawns into one process group, so
    # stop_stack can take down the children too and not just the shell that
    # started them. The child records its own pid: setsid forks when it is
    # already a group leader, and $! is then the wrong process.
    if command -v setsid >/dev/null 2>&1; then
        setsid bash -c 'echo $$ > "$1"; exec bash -c "$2"' stack "$pidfile" "$RUN_CMD" \
            >"$STACK_LOG" 2>&1 &
    else
        bash -c "$RUN_CMD" >"$STACK_LOG" 2>&1 &
        echo $! > "$pidfile"
    fi
    while [[ ! -s "$pidfile" ]] && (( waited < 50 )); do
        sleep 0.1; waited=$(( waited + 1 ))
    done
    STACK_PID=$(cat "$pidfile" 2>/dev/null || true)
    if [[ ! "$STACK_PID" =~ ^[0-9]+$ ]]; then
        STACK_STATUS="RUN_CMD never started: $(tail -5 "$STACK_LOG" 2>/dev/null)"
        return 1
    fi
    return 0
}

# Start RUN_CMD detached and wait for READY_URL to answer 200.
#   0  up — STACK_PID holds the recorded pid
#   1  configured but it never came up; the stack is stopped again before return
#   3  nothing to start: RUN_CMD is empty, which is the right answer for a library
# Call from the repo root, as the hooks do.
# shellcheck disable=SC2034  # STACK_PID and STACK_LOG are set for the caller
start_stack() {
    local timeout="${1:-${READY_TIMEOUT:-60}}" root state pidfile old why
    load_gates
    STACK_PID=""; STACK_LOG=""; STACK_STATUS=""
    if [[ -z "$RUN_CMD" ]]; then
        STACK_STATUS="no runnable surface: RUN_CMD is empty in .claude/gates.sh"
        return 3
    fi
    root=$(repo_root "$PWD") || { STACK_STATUS="not a git repository"; return 1; }
    state=$(state_dir "$root")
    mkdir -p "$state" || { STACK_STATUS="cannot write $state"; return 1; }
    pidfile=$(stack_pid_file "$root")
    STACK_LOG=$(stack_log_file "$root")

    # Overwriting a live pidfile orphans that stack for good: stop_stack reads
    # this file and would never find it again. Stop it first, or refuse.
    old=$(cat "$pidfile" 2>/dev/null || true)
    if stack_alive "$old"; then
        stop_stack >/dev/null 2>&1
        if stack_alive "$old"; then
            STACK_STATUS="a stack is already recorded as running (pid ${old}) and would not stop"
            return 1
        fi
    fi

    # Something already answering READY_URL would be mistaken for the product
    # from the first poll onwards, so it is a refusal and not a head start.
    if [[ -n "$READY_URL" ]] && ready_probe "$READY_URL"; then
        STACK_STATUS="something is already listening on ${READY_URL} — stop it first"
        return 1
    fi

    launch_stack "$pidfile" || return 1
    if [[ -z "$READY_URL" ]]; then
        sleep 1
        if stack_alive "$STACK_PID"; then
            STACK_STATUS="started as pid ${STACK_PID}; no READY_URL to poll"
            return 0
        fi
        STACK_STATUS="RUN_CMD exited at once: $(tail -5 "$STACK_LOG" 2>/dev/null)"
        return 1
    fi
    if wait_ready "$STACK_PID" "$READY_URL" "$timeout"; then
        STACK_STATUS="ready as pid ${STACK_PID}"
        return 0
    fi
    # stop_stack sets STACK_STATUS too, and the caller needs why it failed to
    # come up rather than how it was cleaned up afterwards.
    why="$READY_STATUS"
    stop_stack >/dev/null 2>&1
    STACK_STATUS="$why"
    return 1
}

# Stop what start_stack started, then check the recorded pid is actually gone.
#   0  stopped, nothing left behind
#   1  it survived — a STOP_CMD that does not stop is reported, not papered over
#   3  nothing was running
# shellcheck disable=SC2034  # STACK_STATUS is set for the caller
stop_stack() {
    local root pidfile pid log backstopped=0
    load_gates
    STACK_STATUS=""
    root=$(repo_root "$PWD") || { STACK_STATUS="not a git repository"; return 3; }
    pidfile=$(stack_pid_file "$root")
    log=$(stack_log_file "$root")
    [[ -s "$pidfile" ]] || { STACK_STATUS="nothing recorded to stop"; return 3; }
    pid=$(cat "$pidfile")
    [[ "$pid" =~ ^[0-9]+$ ]] \
        || { rm -f "$pidfile"; STACK_STATUS="recorded pid is not a number"; return 3; }
    # The pidfile outlives STOP_CMD on purpose: a project's own stop command is
    # usually the one that reads it (`pkill -F`, `docker compose stop`).
    if [[ -n "$STOP_CMD" ]]; then
        eval "$STOP_CMD" >>"$log" 2>&1 || true
        stack_gone "$pid" 5 || backstopped=1
    fi
    if [[ -z "$STOP_CMD" ]] || (( backstopped )); then
        kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        if ! stack_gone "$pid" 5; then
            kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            stack_gone "$pid" 3 || true
        fi
    fi
    rm -f "$pidfile"
    if stack_alive "$pid"; then
        STACK_STATUS="pid ${pid} is still alive after STOP_CMD and a KILL"
        return 1
    fi
    if (( backstopped )); then
        STACK_STATUS="STOP_CMD left pid ${pid} alive; the process group was killed as a backstop"
        return 1
    fi
    STACK_STATUS="stopped; pid ${pid} is gone"
    return 0
}

# A browser can carry a cookie pair or a header line. A bare token, a JSON blob
# or a log line is not something /ship can hand to one.
looks_like_session() {
    local value="${1%%$'\n'*}"
    [[ "$value" =~ ^[A-Za-z0-9_.~-]+=[^[:space:]]+ ]] && return 0
    [[ "$value" =~ ^[A-Za-z][A-Za-z0-9-]*:[[:space:]]+[^[:space:]]+ ]] && return 0
    return 1
}

# Print an authenticated session the browser can use, from the project's own
# seeding path — a test authenticator override, a seeded row, a dev-only mint
# script. Never a human's credentials: /project-setup refuses to record a
# SESSION_CMD that reads one.
#   0  SESSION_VALUE holds a cookie or a header
#   1  SESSION_CMD failed, or printed nothing a browser could carry
#   3  no SESSION_CMD: the surface has no sign-in, or nobody has recorded one
# shellcheck disable=SC2034  # SESSION_VALUE and SESSION_STATUS are set for the caller
session_for_browser() {
    local out rc
    load_gates
    SESSION_VALUE=""; SESSION_STATUS=""
    if [[ -z "$SESSION_CMD" ]]; then
        SESSION_STATUS="no SESSION_CMD in .claude/gates.sh"
        return 3
    fi
    # Always a fresh shell, bounded when coreutils can bound it. Both branches
    # run the same way on purpose: a key that means one thing where `timeout`
    # exists and another where it does not is worse than an unbounded one.
    # -k, because a script that ignores TERM is exactly the script that hangs.
    if command -v timeout >/dev/null 2>&1; then
        out=$(timeout -k 5 "${SESSION_TIMEOUT:-60}" bash -c "$SESSION_CMD" 2>&1); rc=$?
    else
        out=$(bash -c "$SESSION_CMD" 2>&1); rc=$?
    fi
    if (( rc == 124 )); then
        SESSION_STATUS="SESSION_CMD timed out after ${SESSION_TIMEOUT:-60}s"
        return 1
    fi
    if (( rc != 0 )); then
        SESSION_STATUS="SESSION_CMD failed: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')"
        return 1
    fi
    out=$(printf '%s' "$out" | grep -v '^[[:space:]]*$' | tail -1)
    if ! looks_like_session "$out"; then
        SESSION_STATUS="SESSION_CMD printed no cookie or header: $(printf '%s' "$out" | cut -c1-60)"
        return 1
    fi
    SESSION_VALUE="$out"
    if [[ "$out" =~ ^[A-Za-z0-9_.~-]+= ]]; then
        SESSION_STATUS="cookie printed"
    else
        SESSION_STATUS="header printed"
    fi
    return 0
}

# --- what counts as a user-facing surface -------------------------------------
# SURFACE_PATHS is the project's own answer, and two callers need the same one:
# mark-reviewed.sh, which refuses to record a surface change nobody walked, and
# /ship step 6, which refuses an approving verdict the reviewer could not
# exercise. One copy, so the two cannot drift into disagreeing about which diff
# needs a browser.

# The ref a surface diff is scoped against: what the caller established, or
# origin/HEAD. A caller that must refuse on an unresolvable base verifies the
# answer itself — this only says which ref to ask about.
surface_base() {
    local base="${1:-}"
    if [[ -n "$base" ]]; then printf '%s' "$base"; return 0; fi
    git symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null
}

# A space-separated glob list, as git pathspecs. Matched with git's globbing
# rather than bash's, so this agrees with /project-setup's
# verify-run-contract.sh, and so a deleted route counts.
glob_specs() {
    local glob
    set -f
    # shellcheck disable=SC2086  # splitting is wanted here; globbing is not
    set -- $1
    set +f
    for glob in "$@"; do printf ':(glob)%s\n' "$glob"; done
}

# Every file in <base>..HEAD, uncommitted or untracked, that matches one of
# <globs>. Empty output means this diff touches none of them — which is an
# answer, so an empty glob list returns nothing rather than everything.
changed_under() {
    local globs="$1" base
    local -a specs
    base=$(surface_base "${2:-}")
    mapfile -t specs < <(glob_specs "$globs")
    (( ${#specs[@]} )) || return 0
    {
        git diff --name-only "${base}..HEAD" -- "${specs[@]}"
        git diff --name-only HEAD -- "${specs[@]}"
        git ls-files --others --exclude-standard -- "${specs[@]}"
    } 2>/dev/null | sort -u
}

surface_specs() { glob_specs "$SURFACE_PATHS"; }

# Call after load_gates, and guard the call with a non-empty RUN_CMD: a project
# with no runnable surface has nothing to walk whatever its globs match.
changed_surface() { changed_under "$SURFACE_PATHS" "${1:-}"; }

# --- CI policy: a skip is written down where it happens -------------------------
# A check skipped by policy and a check that passed reach a ledger as the same
# silence unless something records the skip at the moment it happens. `parity=`
# exists because "parity passed" and "parity never ran this ship" used to render
# identically; this is that fix in the second place it is needed, and it ships
# before any project's CI is tiered because tiering without it is the thing a
# council vetoed twice on 2026-09-18.
#
# Four keys in gates.sh, and one line in the gates marker. Nothing here changes
# what CI runs — `.gitlab-ci.yml` decides that — and nothing here is a verdict:
# `ci-policy=` is a record, and no caller reads it to decide anything.

# Every key below is read with `:-`, the way `parity_outcome` reads PARITY_CMD:
# this file is a library, the Stop hook runs under `set -u`, and a caller that
# reaches `record_gates_pass` by some path that did not `load_gates` first should
# get the fail-closed answer rather than kill the hook.

# The names this project defers to the tip, as the line prints them.
ci_tiered_names() {
    # Newlines folded first, for `decision_doc`'s reason: `read` stops at the
    # first one, so a value wrapped across lines for readability would lose
    # every name after the first — silently, in the line whose whole job is
    # naming them. `read -ra` and not an unquoted expansion, because that is
    # pathname-expanded as well as split and this runs from the repo root.
    local raw="${CI_TIER_PATHS:-}"
    local -a names
    read -ra names <<<"${raw//$'\n'/ }"
    (( ${#names[@]} )) || return 0
    ( IFS=','; printf '%s' "${names[*]}" ) | sed 's/,/, /g'
}

# Is this branch the one the stack collapses onto — the branch whose MR targets
# the default branch?
#
#   0  it is the tip        1  it is stacked on another branch
#   2  nothing here can tell
#
# **The third answer is the whole point**, and it is `parity_base`'s discipline
# in this file's other half. An unnamed base means `origin/HEAD` for
# `changed_under` and `surface_base`, and that is right there, because those
# answer *what changed* — a scope. This answers *was everything checked* — a
# claim. A claim from an unresolved input is the veto's failure wearing a label,
# which is why `parity_outcome` says `pending` rather than `not-triggered` when
# its base will not resolve, and why this says `tip-unknown` rather than `full`.
#
# Both unknown paths are real rather than theoretical. The Stop hook calls
# `record_gates_pass` with no base at all and has no /ship step 1 behind it to
# get one from, so without this every mid-stack branch would record `full` from
# one hook and `tip-only` from the other about one tree.
# Which of the two unknowns it was, for the gloss to name the remedy that can
# actually work. Telling a reader to pass `--base` when they already did is the
# kind of advice that teaches people to stop reading the line.
#
# **Read back by re-running `ci_at_tip`, never carried across from the outcome
# call.** Every caller computes the outcome in a command substitution, so a
# variable set in there dies with the subshell — which is how the first draft of
# this printed the wrong remedy, and the same shape as the `--stop` the pipeline
# wait used to swallow. One `git symbolic-ref` is cheaper than a state channel
# that only works from some callers.
CI_TIP_UNKNOWN_WHY=""

ci_at_tip() {  # ci_at_tip [base]
    local base="${1:-}" default
    CI_TIP_UNKNOWN_WHY=""
    # No base named: the caller has not said what this branch is stacked on, and
    # on the Stop hook's path nothing could have.
    [[ -n "$base" ]] || { CI_TIP_UNKNOWN_WHY="no-base"; return 2; }
    # A base was named and there is nothing to compare it against. Not evidence
    # for the tip — if anything the opposite, since a caller that named a base
    # knows something this does not.
    default=$(git symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null)
    [[ -n "$default" ]] || { CI_TIP_UNKNOWN_WHY="no-default"; return 2; }
    [[ "${base#origin/}" == "${default#origin/}" ]]
}

ci_has_tip_label() {
    [[ -n "${CI_MR_LABELS:-}" ]] || return 1
    [[ ",${CI_MR_LABELS}," == *",${CI_TIP_LABEL:-},"* ]]
}

# What the gates marker records about CI policy, without touching the forge:
#
#   full                        every configured check ran for this tree
#   tip-only (skipped: <names>) the named checks were deferred to the stack tip
#                               by policy, and this branch is not the tip
#   tip-unknown                 checks are deferred to the tip and nothing here
#                               could establish whether this branch is it. Says
#                               so rather than claiming `full`, for the reason
#                               `parity=pending` exists
#   draft-skipped               the MR is a draft and this project declared
#                               CI_IS_ONLY_GATE=no
#   refused (only-gate)         a skip was configured and refused, because this
#                               project has not declared a local gate to fall
#                               back on
#   unconfigured                no CI policy declared — this project tiers
#                               nothing, and nothing about it changes
#
# The forge half — whether the MR is a draft, and what labels it carries — is
# `read_mr_facts`, which only /ship's side of the gate calls. Unset means
# unknown, and unknown never produces `draft-skipped`: a skip is recorded
# because it was known to happen, never inferred.
ci_policy_outcome() {  # ci_policy_outcome [base]
    local draft="${CI_MR_DRAFT:-unknown}" at
    [[ -n "${CI_POLICY_RESULT:-}" ]] && { printf '%s' "$CI_POLICY_RESULT"; return 0; }
    if [[ -z "${CI_TIER_PATHS:-}" && "$draft" != "yes" ]]; then
        printf 'unconfigured'; return 0
    fi
    if [[ "${CI_IS_ONLY_GATE:-}" != "no" ]]; then
        [[ -n "${CI_TIER_PATHS:-}" ]] && { printf 'refused (only-gate)'; return 0; }
        # A draft on a project that refuses skips is a draft whose pipeline ran.
        printf 'full'; return 0
    fi
    # Ordered strongest skip first: a draft's pipeline is skipped whole, so what
    # tiering would have deferred never came up.
    [[ "$draft" == "yes" ]] && { printf 'draft-skipped'; return 0; }
    ci_has_tip_label && { printf 'full'; return 0; }
    # `if`, not `ci_at_tip …; at=$?`: the second form aborts under `set -e` on
    # the `return 1` that means "stacked on another branch", which is an answer
    # rather than a failure. No caller sets -e today — every one reaches this
    # through a command substitution — but this file is a library sourced by
    # run.sh, verify-run-contract.sh and every hook, and the repo already has
    # `set -euo pipefail` scripts in it.
    if ci_at_tip "${1:-}"; then at=0; else at=$?; fi
    case "$at" in
        0) printf 'full' ;;
        1) printf 'tip-only (skipped: %s)' "$(ci_tiered_names)" ;;
        *) printf 'tip-unknown' ;;
    esac
}

# One sentence per value, for the caller that prints the line to a person. The
# value is what gets carried verbatim; this is the gloss beside it, and it says
# `full` two different ways on purpose — "nothing was deferred" and "this branch
# is the tip" are the same word and different facts.
ci_policy_gloss() {  # ci_policy_gloss <value> [base]
    case "$1" in
        unconfigured)
            printf 'no CI policy in .claude/gates.sh; this project tiers nothing' ;;
        refused*)
            printf 'CI_TIER_PATHS is set but CI_IS_ONLY_GATE is not `no`, so nothing was skipped' ;;
        draft-skipped)
            printf 'the MR is a draft and this project declared CI_IS_ONLY_GATE=no' ;;
        tip-only*)
            printf 'deferred to the branch that targets the default branch' ;;
        tip-unknown)
            ci_at_tip "${2:-}" || true
            printf '%s deferred to the tip, and ' "$(ci_tiered_names)"
            case "${CI_TIP_UNKNOWN_WHY:-}" in
                no-base)    printf 'no base was named to tell whether this branch is it — pass run-gates.sh --base <ref>' ;;
                no-default) printf 'origin/HEAD is unset, so the named base cannot be compared to anything — git remote set-head origin -a' ;;
                *)          printf 'nothing established whether this branch is it — pass run-gates.sh --base <ref>, and check origin/HEAD is set' ;;
            esac ;;
        full)
            if [[ -z "${CI_TIER_PATHS:-}" ]]; then
                printf 'every configured check ran'
            else
                printf 'every configured check ran for this tree — this branch is the tip, or carries %s' \
                    "${CI_TIP_LABEL:-}"
            fi ;;
        *) printf 'unrecognised' ;;
    esac
}

# The forge half of the policy. **Never reachable from the Stop hook**, for
# run_parity's reason: it makes a network call, and the Stop hook has to stay
# offline and credential-free. Best effort throughout — a forge that cannot be
# reached leaves both facts unknown, and unknown never produces a skip.
read_mr_facts() {  # read_mr_facts [branch]
    local branch="${1:-}" mr draft
    CI_MR_DRAFT="unknown"; CI_MR_LABELS=""
    # No skip is possible, so nothing the forge could say would change the
    # outcome. "Unset changes nothing anywhere" has to include not reaching the
    # network on every gate run of every project that declared no policy.
    [[ "${CI_IS_ONLY_GATE:-}" == "no" || -n "${CI_TIER_PATHS:-}" ]] || return 0
    command -v glab >/dev/null 2>&1 || return 0
    command -v jq   >/dev/null 2>&1 || return 0
    [[ -n "$branch" ]] || branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    [[ -n "$branch" ]] || return 0
    mr=$(glab mr list --source-branch "$branch" --output json 2>/dev/null) || return 0
    draft=$(jq -r 'if length > 0 then ((.[0].draft // .[0].work_in_progress // false) | tostring)
                   else "" end' <<<"$mr" 2>/dev/null)
    [[ -n "$draft" ]] || return 0
    if [[ "$draft" == "true" ]]; then CI_MR_DRAFT="yes"; else CI_MR_DRAFT="no"; fi
    CI_MR_LABELS=$(jq -r 'if length > 0 then ((.[0].labels // []) | join(",")) else "" end' \
        <<<"$mr" 2>/dev/null)
    return 0
}

# --- parity: a double pinned to the server it stands in for --------------------
# PARITY_CMD runs the same contract cases against the test double and against
# the real server, and fails on any divergence. `rules/testing.md` carries what
# belongs in those cases, and is the only copy of it — which is why this comment
# points at the rule rather than paraphrasing it.
#
# **Never part of run_all_gates, and never reachable from the Stop hook.** It
# starts the real server, so it is neither offline nor under ~30s. /ship step 2
# runs it; every other caller only reads what came of it.
#
# PARITY_PATHS names *both* sides of the contract — the double's directory and
# the server paths that define what it doubles: routes, schemas, error policy. A
# trigger scoped to the double's own directory is the failure this exists to
# catch through a new door, because the drift that has already happened came
# from the server side. sophia#98's fixture did not move; the route's error
# policy did, and forty green e2e tests stood in front of a cycle that could not
# complete in production.
changed_parity() { changed_under "$PARITY_PATHS" "${1:-}"; }

# The ref a parity diff is scoped against, verified. Prints nothing when there
# is none — which is not the same as "nothing changed", and is the whole reason
# this is a function rather than a call to surface_base.
#
# `changed_under` swallows git's stderr, so an unresolvable base produces an
# empty file list exactly as an unchanged tree does. Reading that as
# `not-triggered` puts a *positive* claim into the marker — "this diff touched
# neither side" — that /ship step 2 and the reviewer are both told to believe.
# That is the veto's failure wearing a label, which is worse than the silence it
# replaced. `mark-reviewed.sh` carries the same guard for the same reason, and
# says so in the same words.
parity_base() {
    local base
    base=$(surface_base "${1:-}")
    [[ -n "$base" ]] || return 1
    git rev-parse --verify -q "$base" >/dev/null 2>&1 || return 1
    printf '%s' "$base"
}

# What the gates marker records about parity, without running anything:
#
#   pass | fail     run_parity ran PARITY_CMD, and this is what it returned
#   not-triggered   PARITY_CMD is set and no file under PARITY_PATHS changed
#   unconfigured    PARITY_CMD is empty, or it is set with no PARITY_PATHS to
#                   trigger it, which is a command that would never run
#   pending         nothing has run it for this tree — either the contract moved
#                   and nobody checked, or there is no base to tell. What the
#                   Stop hook records, since it never runs parity itself
#
# The last two are the point: "parity passed" and "parity was never re-triggered
# this ship" used to render as the same silence, and `unconfigured` is how a
# project that has never heard of the key says so out loud. `not-triggered` is
# the one value that asserts something about the diff, so it is only ever
# reached through a base that resolved.
parity_outcome() {
    local base
    [[ -n "${PARITY_RESULT:-}" ]] && { printf '%s' "$PARITY_RESULT"; return 0; }
    [[ -z "${PARITY_CMD:-}" || -z "${PARITY_PATHS:-}" ]] \
        && { printf 'unconfigured'; return 0; }
    base=$(parity_base "${1:-}") || { printf 'pending'; return 0; }
    [[ -z "$(changed_parity "$base")" ]] && { printf 'not-triggered'; return 0; }
    printf 'pending'
}

# Run PARITY_CMD when either side of the contract moved. Sets PARITY_RESULT to
# one of the names above, PARITY_OUTPUT to why, and PARITY_CHANGED to the files
# that triggered it. Returns non-zero only on a real divergence: a project with
# no parity suite is not a project with a failing one.
run_parity() {
    local base="${1:-}"
    load_gates
    PARITY_RESULT=""; PARITY_OUTPUT=""; PARITY_CHANGED=""
    if [[ -z "$PARITY_CMD" ]]; then
        PARITY_RESULT="unconfigured"
        PARITY_OUTPUT="no PARITY_CMD in .claude/gates.sh"
        return 0
    fi
    if [[ -z "$PARITY_PATHS" ]]; then
        PARITY_RESULT="unconfigured"
        PARITY_OUTPUT="PARITY_CMD is set but PARITY_PATHS is empty, so nothing would ever trigger it"
        return 0
    fi
    # A configured parity suite with nothing to scope it against is a refusal,
    # not a skip. Placed after the unconfigured returns on purpose: a project
    # with no parity suite and no origin/HEAD still passes a bare run, the way
    # mark-reviewed.sh gates its own refusal on the keys being set.
    if ! base=$(parity_base "$base"); then
        PARITY_RESULT="fail"
        PARITY_OUTPUT="PARITY_CMD is configured but there is no base to scope the diff against, so a changed contract would read as an unchanged one. Pass the base /ship step 1 established: run-gates.sh --base <ref>"
        return 1
    fi
    PARITY_CHANGED=$(changed_parity "$base")
    if [[ -z "$PARITY_CHANGED" ]]; then
        PARITY_RESULT="not-triggered"
        PARITY_OUTPUT="no file under PARITY_PATHS changed in this diff"
        return 0
    fi
    # Captured once and discarded on a pass: running it twice would start the
    # real server twice, and on a pass the command's own output is noise.
    if PARITY_OUTPUT=$(eval "$PARITY_CMD" 2>&1); then
        PARITY_RESULT="pass"
        PARITY_OUTPUT="no divergence, on a diff touching $(printf '%s\n' "$PARITY_CHANGED" | wc -l) file(s) under PARITY_PATHS"
        return 0
    fi
    PARITY_RESULT="fail"
    PARITY_OUTPUT=$(printf '%s' "$PARITY_OUTPUT" | tail -30)
    return 1
}
