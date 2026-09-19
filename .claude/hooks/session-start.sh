#!/usr/bin/env bash
# SessionStart — record the pre-session baseline so the gate can tell a session
# that wrote code from one that only answered questions, and which paths it
# touched; and report whether the harness is actually installed here and this
# checkout is actually current.
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
write_baseline "${STATE}/base-${SESSION_ID}"

# Housekeeping: markers older than a week are dead weight. The gate skip log is
# the exception and is spared by name — it is the only record of what
# GATE_INERT_PATHS saved, and a question about months cannot be answered by a
# file that never spans a week.
#
# The retention is gate-lib's constant and not a literal here, because the
# doctor's bypass row counts over the same window. A row measured over more days
# than the sweep keeps would report a fall in bypasses that is only this line
# deleting the evidence.
find "$STATE" -type f ! -name "$GATE_SKIP_LOG_NAME" \
     -mtime "+${STATE_RETENTION_DAYS}" -delete 2>/dev/null || true

# --- the readiness report -----------------------------------------------------
# Reporting only. The doctor never edits the project and a finding is never a
# reason to block: this hook always exits 0, so the worst case is a table nobody
# acts on. The model gets it as additionalContext, before its first tool call;
# the user gets the same table in the transcript.
#
# Invoked through `bash` rather than executed, so that a project whose hooks lost
# their exec bit — one of the things the doctor is here to notice — still gets
# told so instead of silently reporting nothing.
FOREMAN="${SCRIPT_DIR}/foreman.sh"
if [[ -r "$FOREMAN" ]]; then
    REPORT=$(bash "$FOREMAN" 2>&1) || true
    if [[ -n "$REPORT" ]]; then
        printf '%s\n' "$REPORT" >&2
        jq -n --arg r "$REPORT" '{
          hookSpecificOutput: {
            hookEventName: "SessionStart",
            additionalContext: $r
          }
        }'
    fi
fi
exit 0
