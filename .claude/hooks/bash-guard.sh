#!/usr/bin/env bash
# PreToolUse(Bash) — advisory guard. Anything that must be enforced regardless
# of what the model decides belongs in permissions.deny, not here.
set -uo pipefail
INPUT=$(cat) || exit 0
command -v jq >/dev/null 2>&1 || exit 0

COMMAND=$(jq -r '.tool_input.command // ""' <<<"$INPUT")
[[ -z "$COMMAND" ]] && exit 0

PATTERNS=(
    # Merging is the first thing DECISIONS.md forbids an unattended run, and was
    # the only item on that list with no guard behind the prose.
    'glab[[:space:]]+mr[[:space:]]+merge'
    'gh[[:space:]]+pr[[:space:]]+merge'
    'rm[[:space:]]+-[a-zA-Z]*[rf][a-zA-Z]*[[:space:]]+(/|~|\$HOME)'
    'git[[:space:]]+push[[:space:]].*(--force([^-]|$)|-f([[:space:]]|$))'
    'git[[:space:]].*--no-verify'
    'git[[:space:]]+reset[[:space:]]+--hard'
    'DROP[[:space:]]+(TABLE|DATABASE)'
    '(^|[[:space:]])truncate([[:space:]]|$)'
    'chmod[[:space:]]+777'
    'docker[[:space:]]+system[[:space:]]+prune[[:space:]].*-a'
)

for p in "${PATTERNS[@]}"; do
    if [[ "$COMMAND" =~ $p ]]; then
        jq -n --arg r "Destructive command matched guard pattern: ${p}. Needs explicit approval." '{
          hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "escalate",
            permissionDecisionReason: $r
          }
        }'
        exit 0
    fi
done
exit 0
