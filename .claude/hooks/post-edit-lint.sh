#!/usr/bin/env bash
# PostToolUse — format the edited file and surface any lint still on it.
#
# The commands come from .claude/gates.sh (FORMAT_CMD / LINT_FILE_CMD), so this
# hook carries no per-language knowledge. When neither is configured it falls
# back to ruff for Python files, resolving it through `uv run` since ruff is
# often not on PATH.
set -uo pipefail
INPUT=$(cat) || exit 0
command -v jq >/dev/null 2>&1 || exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gate-lib.sh"

FILE=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty' <<<"$INPUT")
CWD=$(jq -r '.cwd // ""' <<<"$INPUT")
[[ -z "$FILE" || ! -f "$FILE" ]] && exit 0

ROOT=$(repo_root "${CWD:-$PWD}") || exit 0
cd "$ROOT" || exit 0
load_gates

report() {
    jq -n --arg f "$1" --arg l "$(printf '%s' "$2" | head -40)" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("Formatted \($f). Remaining lint issues:\n" + $l)
      }
    }'
}

# --- configured path -------------------------------------------------------
if [[ -n "${FORMAT_CMD}${LINT_FILE_CMD}" ]]; then
    if [[ -n "$FORMAT_EXTENSIONS" ]]; then
        extension="${FILE##*.}"
        matched=""
        for candidate in $FORMAT_EXTENSIONS; do
            [[ "$extension" == "$candidate" ]] && matched=1 && break
        done
        [[ -z "$matched" ]] && exit 0
    fi

    [[ -n "$FORMAT_CMD" ]] && eval "$FORMAT_CMD \"\$FILE\"" >/dev/null 2>&1
    if [[ -n "$LINT_FILE_CMD" ]]; then
        LINT=$(eval "$LINT_FILE_CMD \"\$FILE\"" 2>&1) || true
        [[ -n "$LINT" ]] && report "$FILE" "$LINT"
    fi
    exit 0
fi

# --- fallback: Python via ruff ---------------------------------------------
[[ "$FILE" != *.py ]] && exit 0

if command -v ruff >/dev/null 2>&1; then
    RUFF=(ruff)
elif command -v uv >/dev/null 2>&1 && [[ -f pyproject.toml ]] && uv run --no-sync ruff --version >/dev/null 2>&1; then
    RUFF=(uv run --no-sync ruff)
else
    exit 0
fi

"${RUFF[@]}" format "$FILE" >/dev/null 2>&1
LINT=$("${RUFF[@]}" check "$FILE" 2>&1) || true

if [[ -n "$LINT" && "$LINT" != *"All checks passed"* ]]; then
    report "$FILE" "$LINT"
fi
exit 0
