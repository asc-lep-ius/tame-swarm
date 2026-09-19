#!/usr/bin/env bash
# Called by /ship as its LAST step, after review passed and after any final
# commits. The marker is keyed by tree fingerprint, not by session.
#
#   mark-reviewed.sh <verdict> [--base <ref>]
#
# --base is the base /ship step 1 established; it scopes the diff the proof
# rule below is read against. Without it the diff is scoped against origin/HEAD.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gate-lib.sh"

usage() { echo "usage: $0 <verdict> [--base <ref>]" >&2; exit 1; }

# The verdict is the only payload, and a milestone run reads its ledger verdict
# straight out of the marker. Anything outside the reviewer's four names means
# the caller retyped or guessed it, so refuse rather than record a fiction.
VERDICT="${1:-}"
(( $# )) && shift
BASE=""
while (( $# )); do
    case "$1" in
        --base) [[ $# -ge 2 ]] || usage; BASE="$2"; shift 2 ;;
        *)      usage ;;
    esac
done

case "$VERDICT" in
    APPROVED | APPROVED_WITH_NOTES | NEEDS_REVISION | FAILED) ;;
    *)
        echo "refusing to mark reviewed: verdict '${VERDICT}' is not one of" \
             "APPROVED, APPROVED_WITH_NOTES, NEEDS_REVISION, FAILED" >&2
        echo "pass the verdict jq parsed out of the reviewer's JSON block." >&2
        exit 1
        ;;
esac

ROOT=$(repo_root "$PWD") || { echo "not a git repo — nothing to mark"; exit 0; }
cd "$ROOT" || exit 1

STATE=$(state_dir "$ROOT")
mkdir -p "$STATE"
FP=$(fingerprint)

# --- the proof rule -----------------------------------------------------------
# sophia#98 was eleven commits, four review rounds, twenty-four findings fixed
# and a green e2e job against a fixture that accepted every attempt. The first
# human to open the page found every grade returning 412. /ship step 2c walks
# the issue's flow on the real stack and leaves the evidence under
# .claude/state/proof-<fingerprint>/; this refuses to record a review of a
# surface nobody watched working.
#
# What a hook can check is that the walk happened and is still about this
# surface. Whether it reached the issue's outcome is the reviewer's judgement —
# a screenshot of a page loading passes every check made here.
load_gates

# `surface_specs` and `changed_surface` live in gate-lib.sh, because /ship step 6
# asks the same question one step earlier — whether this diff needed the walk the
# reviewer says it could not take.

# The exact tree's proof, or an earlier one no surface change has overtaken:
# step 2c walks the flow before the review, so a fix that touches nothing
# user-facing must not send anyone back to the browser. A fix that does touch
# the surface must — that proof is about a page which no longer exists.
current_proof() {
    local candidate f d
    local -a present=() candidates=()
    if [[ -f "${STATE}/proof-${FP}/proof.md" ]]; then
        printf '%s' "${STATE}/proof-${FP}/proof.md"
        return 0
    fi
    # A route the diff *deleted* leaves no file whose mtime could be newer than a
    # proof — but removing it bumped its directory's mtime, which dates the
    # deletion exactly as a write dates an edit. Watch the directory instead.
    # Refusing outright instead would be worse than the hole it closes: a
    # deletion stays in base..HEAD for the rest of the ship, so every later
    # commit — step 5's fixes included — would demand a fresh walk of a tree
    # that was walked.
    for f in "$@"; do
        if [[ -f "$f" ]]; then present+=("$f"); continue; fi
        d=$(dirname "$f")
        while [[ "$d" != "." && "$d" != "/" && ! -d "$d" ]]; do d=$(dirname "$d"); done
        [[ -d "$d" ]] && present+=("$d")
    done
    # Newest first. The glob is in fingerprint order, which has nothing to do
    # with walk order, so the marker could otherwise cite an older unrelated
    # walk — and that pointer is the whole value of the note's `Proof:` line.
    mapfile -t candidates < <(ls -1t "${STATE}"/proof-*/proof.md 2>/dev/null)
    (( ${#candidates[@]} )) || return 1
    for candidate in "${candidates[@]}"; do
        [[ -f "$candidate" ]] || continue
        # -maxdepth 0, so a directory carried in `present` is judged on its own
        # mtime and not on everything beneath it. A no-op for the regular files.
        (( ${#present[@]} )) \
            && [[ -n "$(find "${present[@]}" -maxdepth 0 -newer "$candidate" -print -quit 2>/dev/null)" ]] \
            && continue
        printf '%s' "$candidate"
        return 0
    done
    return 1
}

PROOF=""
if [[ -n "$SURFACE_PATHS" && -n "$RUN_CMD" ]]; then
    # A base that does not resolve scopes the diff against nothing, and a
    # surface change would then look like no change at all — the one way this
    # rule could pass a tree it exists to refuse.
    BASE=$(surface_base "$BASE")
    if ! git rev-parse --verify -q "${BASE:-HEAD-is-not-a-base}" >/dev/null 2>&1; then
        echo "refusing to mark reviewed: SURFACE_PATHS is set and there is no base" \
             "to scope the diff against." >&2
        echo "pass the base /ship step 1 established:" \
             "mark-reviewed.sh ${VERDICT} --base <ref>" >&2
        exit 1
    fi
    mapfile -t SURFACE_CHANGED < <(changed_surface "$BASE")
    if (( ${#SURFACE_CHANGED[@]} )); then
        if ! PROOF=$(current_proof "${SURFACE_CHANGED[@]}"); then
            echo "refusing to mark reviewed: this diff moves a user-facing surface" \
                 "and nothing here proves it was walked." >&2
            printf '  %s\n' "${SURFACE_CHANGED[@]:0:5}" >&2
            (( ${#SURFACE_CHANGED[@]} > 5 )) \
                && echo "  … ${#SURFACE_CHANGED[@]} files in all" >&2
            echo "run /ship step 2c: start the stack the way RUN_CMD starts it, walk" >&2
            echo "scenario 1 of the issue, and leave the screenshots and proof.md in" >&2
            echo "  ${STATE#"$ROOT"/}/proof-${FP}/" >&2
            echo "A stack that cannot be started here is 'blocked', not a deferral." >&2
            exit 1
        fi
    fi
fi

{
    echo "reviewed_at=$(date -Iseconds)"
    echo "head=$(git rev-parse HEAD 2>/dev/null || echo none)"
    echo "verdict=${VERDICT}"
    echo "proof=${PROOF#"$ROOT"/}"
} > "${STATE}/reviewed-${FP}.ok"

echo "Review recorded for tree ${FP:0:12}. The Stop gate will pass."
