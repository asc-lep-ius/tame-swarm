#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit) — say what a decision document is, before it
# is edited. Context, never a block: whether this particular edit is a product
# fork is a judgement, and a hook that refuses spec edits outright would stop
# the typo fixes too.
#
# sophia#98 rewrote STUDY_SURFACE_SPEC.md from "the expected answer … [is]
# visible" to "there is no expected answer to reveal" inside an implementation
# branch, and four review rounds read past it. A branch is the wrong place to
# change the product's mind; rules/product-forks.md is what to do instead.
set -uo pipefail
INPUT=$(cat) || exit 0
command -v jq >/dev/null 2>&1 || exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gate-lib.sh"

FILE=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT")
CWD=$(jq -r '.cwd // ""' <<<"$INPUT")
# No -f test: a brand new SPEC.md is exactly the edit worth saying this about,
# and at PreToolUse a Write's file does not exist yet.
[[ -z "$FILE" ]] && exit 0

ROOT=$(repo_root "${CWD:-$PWD}") || exit 0
cd "$ROOT" || exit 0
load_gates

decision_doc "$FILE" || exit 0

jq -n --arg f "${FILE#"$ROOT"/}" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    additionalContext: ("\($f) records product decisions. A change to what it "
      + "says the product does is a fork to ask about — AskUserQuestion with "
      + "two options, or /defer under product-semantics when nobody is there "
      + "to ask. See rules/product-forks.md. Wording, formatting and a "
      + "decision already made elsewhere are yours to edit.")
  }
}'
exit 0
