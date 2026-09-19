#!/usr/bin/env bash
# Run the project's quality gates and, on success, record the pass so the Stop
# hook does not run them again for the same tree.
#
# Called by /ship step 2. Safe to run by hand at any time.
#
#   --base <ref>   the ref the diff is scoped against, as /ship step 1
#                  established it. Two things read it now. Parity, to decide
#                  whether either side of the contract moved — without it the
#                  scope is origin/HEAD, which on a stacked branch is six other
#                  issues wide and triggers parity on all of them. And the CI
#                  policy, to tell whether this branch is the one the stack
#                  collapses onto; without it the line can only say
#                  `tip-unknown`, because "every configured check ran" is a
#                  claim and nothing here would have the evidence for it.
set -uo pipefail

BASE=""
while (( $# )); do
    case "$1" in
        --base)
            [[ $# -ge 2 ]] || { echo "usage: $0 [--base <ref>]" >&2; exit 2; }
            BASE="$2"; shift 2 ;;
        *) echo "usage: $0 [--base <ref>]" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gate-lib.sh"

ROOT=$(repo_root "$PWD") || { echo "not a git repo — no gates to run"; exit 0; }
cd "$ROOT" || exit 1

STATE=$(state_dir "$ROOT")
mkdir -p "$STATE"
FP=$(fingerprint)

if ! run_all_gates; then
    printf 'Quality gates FAILED:\n\n%s\n' "$GATE_FAILURES"
    exit 1
fi

# Parity runs here and nowhere else: it starts the real server, so it belongs to
# the deliberate half of the gate and must never reach the Stop hook. A failure
# is a failure — the marker is not written, exactly as for a failing gate, so the
# next turn re-checks rather than inheriting a pass.
if ! run_parity "$BASE"; then
    printf 'Contract parity FAILED (%s):\n\n%s\n' "$PARITY_CMD" "$PARITY_OUTPUT"
    # Only when something did: a refusal over an unresolvable base has no file
    # list, and an empty "Triggered by" reads as one that triggered on nothing.
    [[ -n "$PARITY_CHANGED" ]] && printf '\nTriggered by:\n%s\n' "$PARITY_CHANGED"
    exit 1
fi

# The forge half of the CI policy, and the only network call in this script that
# is not PARITY_CMD's own. Best effort: an unreachable forge leaves the MR facts
# unknown, and unknown never produces a recorded skip.
read_mr_facts
CI_POLICY=$(ci_policy_outcome "$BASE")

record_gates_pass "$STATE" "$FP" "$PARITY_RESULT" "$CI_POLICY"
echo "Gates passed for tree ${FP:0:12}:"
[[ -n "$LINT_CMD" ]] && echo "  lint  ${LINT_CMD}"
[[ -n "$TYPE_CMD" ]] && echo "  types ${TYPE_CMD}"
[[ -n "$TEST_CMD" ]] && echo "  tests ${TEST_CMD}"
[[ -z "$LINT_CMD$TYPE_CMD$TEST_CMD" ]] && echo "  (none configured in .claude/gates.sh)"
# Printed on every run, including the three outcomes where nothing ran. "Parity
# passed" and "parity was never re-triggered this ship" are different facts and
# stopped being the same silence here.
echo "  parity ${PARITY_RESULT} — ${PARITY_OUTPUT}"
# Printed on every run for parity's reason, and `unconfigured` is a value rather
# than a blank: a check skipped by policy and a check that passed are the same
# silence until something names the skip where it happened.
echo "  ci-policy ${CI_POLICY} — $(ci_policy_gloss "$CI_POLICY" "$BASE")"
# Printed, and that is the whole of it. A gate that failed for being slow is a
# gate somebody turns off on the first loaded machine, so the number says what
# the run cost and leaves the decision to whoever reads it.
if [[ -n "$GATE_ELAPSED_TOTAL" ]]; then
    printf '  cost   %ss' "$GATE_ELAPSED_TOTAL"
    (( GATE_ELAPSED_TOTAL > GATE_BUDGET_S )) \
        && printf ' — over the %ss budget in .claude/gates.sh' "$GATE_BUDGET_S"
    printf '\n'
fi
echo "The Stop hook will not re-run them until the tree changes."
exit 0
