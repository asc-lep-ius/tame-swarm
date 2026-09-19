#!/usr/bin/env bash
# Stop — the involuntary half of the gate.
set -uo pipefail

INPUT=$(cat) || INPUT='{}'
command -v jq >/dev/null 2>&1 || exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gate-lib.sh"

SESSION_ID=$(jq -r '.session_id // "no-session"' <<<"$INPUT")
CWD=$(jq -r '.cwd // ""' <<<"$INPUT")

ROOT=$(repo_root "${CWD:-$PWD}") || exit 0
cd "$ROOT" || exit 0

STATE=$(state_dir "$ROOT")
mkdir -p "$STATE"

# --- escape hatches -------------------------------------------------------
# Three ways out, every one of them exactly as easy to reach as it has always
# been, and none of them silent any more: each writes a bypass marker before it
# lets go. `record_bypass` cannot refuse and cannot fail loudly, so the order is
# "record, then leave" in all three cases — recording is never what decides
# whether a session may end.
#
# The fingerprint is taken up here because the marker names the tree, and because
# a bypass over a tree this session never touched bypassed nothing: the baseline
# check below would have let it stop anyway. Recording those would put one row
# per read-only session into a table whose whole job is making the rare real
# override visible.
BASE_FILE="${STATE}/base-${SESSION_ID}"
CUR_FP=$(fingerprint)
BASE_FP=$(baseline_fingerprint "$BASE_FILE")

gated_tree() { [[ -n "$BASE_FP" && "$BASE_FP" != "$CUR_FP" ]]; }

bypassed() {  # bypassed <via> [reason]
    gated_tree && record_bypass "$STATE" "$SESSION_ID" "$CUR_FP" "$1" "${2:-}"
    exit 0
}

# Never let the gate wedge a session. Three blocks per session is the ceiling.
ATTEMPTS_FILE="${STATE}/attempts-${SESSION_ID}"
ATTEMPTS=$(cat "$ATTEMPTS_FILE" 2>/dev/null || echo 0)
[[ "$ATTEMPTS" =~ ^[0-9]+$ ]] || ATTEMPTS=0
if (( ATTEMPTS >= 3 )); then
    rm -f "$ATTEMPTS_FILE"
    echo "gate: attempt limit reached, allowing stop" >&2
    # The bypass the model reaches by failing three times, and the one that was
    # most silent of all: there is nobody here to have stated a reason, so it is
    # recorded `unstated` like any other and never refused for it.
    bypassed attempt-limit
fi
if [[ "${SHIP_GATE:-on}" == "off" ]]; then
    # The milestone runner turns the gate off around every implement turn by
    # design and states no reason, so those are their own route — see
    # `bypass_summary` for why they are counted apart rather than not at all. A
    # reason set beside it is a person saying something the runner never says,
    # which makes it an override again.
    [[ -n "${MILESTONE_RUN:-}" && -z "${SHIP_GATE_REASON:-}" ]] && bypassed runner
    bypassed env "${SHIP_GATE_REASON:-}"
fi
SKIP_FILE="${STATE}/skip-${SESSION_ID}"
if [[ -f "$SKIP_FILE" ]]; then
    bypassed skip-file "$(cat "$SKIP_FILE" 2>/dev/null || true)"
fi

block() {
    echo $(( ATTEMPTS + 1 )) > "$ATTEMPTS_FILE"
    printf '%s\n' "$1" >&2
    exit 2
}

pass() {
    rm -f "$ATTEMPTS_FILE"
    exit 0
}

# --- did this session change anything? ------------------------------------
# The baseline and the fingerprint were read above, where the bypasses needed
# them; this is the same test `gated_tree` makes, spelled out for the path that
# goes on to run the gates.
[[ -f "$BASE_FILE" ]] || pass          # no baseline — don't gate
BASE_SHA=$(baseline_head "$BASE_FILE")
[[ "$BASE_FP" == "$CUR_FP" ]] && pass  # read-only session

# --- gate 1: build health -------------------------------------------------
# Skipped when this exact tree has already passed, so the gates cost one run
# per change rather than one run per turn. Skipped too when every path this
# session touched is declared inert in gates.sh — nothing a gate reads moved, so
# there is nothing for one to say. Unset GATE_INERT_PATHS means `inert_skip` is
# always false and this is the gate it always was.
if [[ ! -f "$(gates_marker "$STATE" "$CUR_FP")" ]]; then
    if inert_skip "$BASE_SHA"; then
        record_gate_skip "$STATE" "$CUR_FP"
        printf 'gate: no gate reads what changed (%s) — not running them\n' \
               "$(inert_line)" >&2
    elif ! run_all_gates; then
        block "Quality gate failed — fix before finishing:

${GATE_FAILURES}"
    fi
    record_gates_pass "$STATE" "$CUR_FP"
fi

# --- gate 2: review required ----------------------------------------------
# /ship is user-invoked only (disable-model-invocation), so the model has no
# legal way to clear this gate itself. Ask once per tree, then get out of the
# way — blocking again only burns turns on an instruction that cannot be
# obeyed. Deliberately does not touch ATTEMPTS: that budget is for gate 1,
# whose failures the model can actually fix.
if [[ ! -f "${STATE}/reviewed-${CUR_FP}.ok" ]]; then
    NUDGE_FILE="${STATE}/nudged-${SESSION_ID}-${CUR_FP}"
    if [[ -f "$NUDGE_FILE" ]]; then
        echo "gate: review pending, already asked for this tree — allowing stop" >&2
        pass
    fi
    : > "$NUDGE_FILE"
    printf '%s\n' "Code changed in this session but has not been reviewed.

You cannot clear this gate yourself — /ship is user-invoked only. Do not run the
reviewer, mark-reviewed.sh, or the skip file on the user's behalf, and do not
reproduce the /ship workflow by other means.

End the turn by telling the user the work is unreviewed and that they should run
/ship (or /ship --quick if a review already ran and the tree has moved since).
This gate will not block again for this tree. To bypass it deliberately, the
user can: echo 'why' > ${STATE}/skip-${SESSION_ID}

The bypass is allowed either way and recorded either way — the line written
there is the reason it is recorded with, and the doctor prints it back." >&2
    exit 2
fi

pass
