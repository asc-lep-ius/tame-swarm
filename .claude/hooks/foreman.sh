#!/usr/bin/env bash
# project-foreman — is the harness actually installed here, and is this checkout
# actually current?
#
# Reports; never installs. `/project-setup` is the only thing that writes, and a
# finding here is a sentence telling you to run it, never a quiet fix.
#
#   foreman.sh [path]        report on a repo, defaulting to the current one
#     --measuring           promote the checkout and local-default findings to
#                           required: a coverage number read off a stale tree is
#                           a number about code that is not there
#     --base <ref>          measure the checkout against this ref rather than
#                           origin/HEAD, for a branch stacked on another branch.
#                           It holds the local-default row at advisory too,
#                           unless the ref it names is that stale one
#     --offline             skip the fetch and the forge query
#
#   .claude/doctor-accept   accepts one named row per line, `<key>  <reason>`,
#                           `#` comments allowed. A line holds only once it is
#                           committed, so an acceptance arrives in review like
#                           any other change. An accepted required finding still
#                           prints — as `accept`, with its reason — and stops
#                           counting toward the exit code, until the line is 30
#                           days old and stops accepting anything.
#
#   exit 0  no required finding — advisory ones may still be printed, and an
#           accepted one is printed above the count
#   exit 1  at least one required finding nobody accepted; /ship step 0 stops here
#   exit 2  not a git repository, or an option this script does not have
#
# Project-agnostic by construction: every check below is about the *contract* —
# which files exist, which keys are set, what git says — and none of them knows
# or cares what stack fills it. Nothing here writes a file, so a `noexec`
# $TMPDIR and a non-login shell are both uneventful.
set -uo pipefail

# --- what the contract is -----------------------------------------------------
REQUIRED_HOOKS=(bash-guard.sh decision-doc-context.sh foreman.sh gate-lib.sh
                mark-reviewed.sh post-edit-lint.sh run-gates.sh session-start.sh
                stop-gate.sh)
TEMPLATE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/templates/project-claude"
# #18 owns this ledger and its format. Until it lands the file is simply absent,
# which is why every read of it degrades to a row saying so rather than an error.
STATE_DIR=".claude/state"
DEFERRAL_LEDGER="$STATE_DIR/deferrals.jsonl"
# The third answer to a required finding, next to satisfying it and teaching this
# script about the repo. Absent is the ordinary case: a project with nothing to
# accept never grows one.
ACCEPT_FILE=".claude/doctor-accept"
ACCEPT_MAX_AGE_DAYS=30
# Every label this script can print, and so every key an acceptance may name.
# Compared literally, never as a glob or a pattern: a key that matched more than
# one row would accept the finding nobody had read yet.
# `deferrals`, `bypasses` and `gate cost` are deliberately not among them: all
# three only ever report ok or advisory, so an acceptance naming one could accept
# nothing and would read as stale forever.
ACCEPTABLE_KEYS=(".claude/" "hooks" "gates.sh" "gate commands" ".gitignore"
                 "origin/HEAD" "checkout" "local default" "forge" "run contract")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The two things that can take real time are the fetch and the forge query, so
# both are bounded and degrade to a finding rather than a hang. Everything else
# is a stat, a grep or a git ref read.
NET_TIMEOUT="${FOREMAN_NET_TIMEOUT:-5}"

MEASURING=0
OFFLINE=0
BASE_REF=""
TARGET=""

# Set by load_gates once gate-lib.sh is sourced; initialised here so that a
# project missing gate-lib still reports rather than dying on an unset variable.
LINT_CMD=""; TYPE_CMD=""; TEST_CMD=""; GATE_INERT_PATHS=""; GATE_BUDGET_S=""
RUN_CMD=""; READY_URL=""; SESSION_CMD=""; SURFACE_PATHS=""

BASE=""
FETCH_NOTE=""
REQUIRED_FINDINGS=0
ADVISORY_FINDINGS=0
ACCEPTED_FINDINGS=0

declare -A ACCEPT_REASON
declare -A ACCEPT_DATE
declare -A ACCEPT_EPOCH
declare -A ACCEPT_VOID   # why this line's date is unusable, when it is
declare -A ACCEPT_USED
ACCEPT_ORDER=()     # the keys in force, in file order, so the rows are stable
ACCEPT_NOTES=()     # what is wrong with the file itself, one note per line
ACCEPTED_ROWS=()    # what was accepted this run, for the block above the count
EMITTED_ROWS=()     # "<class><tab><label>" per row this run printed

usage() {
    sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (( $# )); do
    case "$1" in
        --measuring) MEASURING=1 ;;
        --offline)   OFFLINE=1 ;;
        --base)      shift; BASE_REF="${1:-}" ;;
        -h|--help)   usage; exit 0 ;;
        -*)          printf 'project-foreman: unknown option %s\n' "$1" >&2; exit 2 ;;
        *)           TARGET="$1" ;;
    esac
    shift
done

# --- the report ---------------------------------------------------------------
row()      { printf '  %-6s  %-14s %s\n' "$1" "$2" "$3"; EMITTED_ROWS+=("$1"$'\t'"$2"); }
ok()       { row "ok"   "$1" "$2"; }
advisory() { row "warn" "$1" "$2"; ADVISORY_FINDINGS=$(( ADVISORY_FINDINGS + 1 )); }

# The one place acceptance is applied, so that no check has to know the file
# exists. An acceptance changes this row's class and appends why; it never drops
# the row and never edits the sentence the check wrote, because a finding you
# cannot read is not one anybody decided to live with.
required() {  # required <label> <text>
    local label="$1" text="$2" reason date void age
    reason="${ACCEPT_REASON[$label]:-}"
    if [[ -n "$reason" ]]; then
        ACCEPT_USED[$label]=1
        date="${ACCEPT_DATE[$label]}"
        void="${ACCEPT_VOID[$label]}"
        age=$(( ( $(date +%s) - ${ACCEPT_EPOCH[$label]:-0} ) / 86400 ))
        if [[ -n "$void" ]]; then
            text="$text — accepted with $void, which does not hold: $reason"
        elif (( age > ACCEPT_MAX_AGE_DAYS )); then
            text="$text — accepted $date, $age days ago, and an acceptance expires at $ACCEPT_MAX_AGE_DAYS: $reason"
        else
            row "accept" "$label" "$text — accepted $date: $reason"
            ACCEPTED_FINDINGS=$(( ACCEPTED_FINDINGS + 1 ))
            ACCEPTED_ROWS+=("$label — $reason (accepted $date)")
            return
        fi
    fi
    row "FAIL" "$label" "$text"
    REQUIRED_FINDINGS=$(( REQUIRED_FINDINGS + 1 ))
}

contains() {  # contains <needle> <item>...
    local needle="$1" item
    shift
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

join_with() {  # join_with <separator> <item>...
    local sep="$1" out="" item
    shift
    for item in "$@"; do
        [[ -n "$out" ]] && out+="$sep"
        out+="$item"
    done
    printf '%s' "$out"
}

run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 2 "$NET_TIMEOUT" "$@"
    else
        "$@"
    fi
}

# The whole point of the hooks directory is that the project's own copy is the
# one that runs, so that is the one to read. The template's copy is the fallback
# for the case this script is invoked from the template against another repo.
source_gate_lib() {
    local candidate
    for candidate in ".claude/hooks/gate-lib.sh" "${SCRIPT_DIR}/gate-lib.sh"; do
        if [[ -r "$candidate" ]]; then
            # shellcheck source=/dev/null
            source "$candidate" && return 0
        fi
    done
    return 1
}

# --- the acceptance file ------------------------------------------------------
# `.claude/doctor-accept` names a row and says why this repo lives with it. One
# key per line and nothing else: no globs, no `all`, no patterns. The finding a
# blanket accept covers is the one nobody read.
trim() {  # trim <string>
    [[ "$1" =~ ^[[:space:]]*(.*[^[:space:]])?[[:space:]]*$ ]] && printf '%s' "${BASH_REMATCH[1]}"
    return 0
}

epoch_of() {  # epoch_of <YYYY-MM-DD>
    date -d "$1" +%s 2>/dev/null || date -j -f '%Y-%m-%d' "$1" +%s 2>/dev/null
}

# A line that does not date itself is dated by the file that carries it: the
# file is tracked, so git already knows when the line was written, and blame is
# what asks. An explicit `<key>  <date>  <reason>` overrides the *age* and
# nothing else, which is how a line keeps its own age across a reformat.
#
# What blame really answers is whether anybody reviewed this line, and that is
# the question with teeth: an acceptance nobody committed is an acceptance
# nobody reviewed, and the review is the whole price of putting this in the
# repo. So the all-zero sha is refused rather than read. Blame attributes a
# working-tree line to it with `author-time` set to *now*, which would date
# every uncommitted line today and hand the file's one cost back — append a
# line to a tracked file, never commit it, and step 0 goes green forever.
blame_date() {  # blame_date <line number>; empty unless a commit holds the line
    local porcelain stamp
    porcelain=$(git blame --line-porcelain -L "$1,$1" -- "$ACCEPT_FILE" 2>/dev/null) || return 0
    [[ "${porcelain%% *}" =~ ^0+$ ]] && return 0
    stamp=$(printf '%s\n' "$porcelain" | awk '/^author-time /{print $2; exit}')
    [[ -n "$stamp" ]] || return 0
    date -d "@$stamp" +%F 2>/dev/null || date -r "$stamp" +%F 2>/dev/null
}

load_acceptances() {
    local lineno=0 line key reason date blanket='[][*?]' dated bare
    local committed void epoch now
    dated='^([0-9]{4}-[0-9]{2}-[0-9]{2})[[:space:]]+(.*)$'
    bare='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
    now=$(date +%s)
    [[ -r "$ACCEPT_FILE" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        lineno=$(( lineno + 1 ))
        line=$(trim "$line")
        [[ -z "$line" || "$line" == "#"* ]] && continue
        key="${line%%  *}"
        reason=$(trim "${line#"$key"}")
        if [[ -z "$reason" ]]; then
            ACCEPT_NOTES+=("line $lineno accepts \`$key\` with no reason — two spaces, then why")
            continue
        fi
        if [[ "$key" =~ $blanket ]] || [[ "$key" == "all" || "$key" == "any" ]]; then
            ACCEPT_NOTES+=("line $lineno names \`$key\` — one row label per line, matched literally; there is no blanket accept")
            continue
        fi
        if ! contains "$key" "${ACCEPTABLE_KEYS[@]}"; then
            ACCEPT_NOTES+=("line $lineno names \`$key\`, which is not a row this foreman prints — it accepts nothing")
            continue
        fi
        if [[ -n "${ACCEPT_REASON[$key]:-}" ]]; then
            ACCEPT_NOTES+=("line $lineno accepts \`$key\` a second time — the first line is the one in force")
            continue
        fi
        if [[ "$reason" =~ $bare ]]; then
            ACCEPT_NOTES+=("line $lineno accepts \`$key\` with a date and no reason")
            continue
        fi
        date=""
        if [[ "$reason" =~ $dated ]]; then
            date="${BASH_REMATCH[1]}"
            reason=$(trim "${BASH_REMATCH[2]}")
        fi
        # Asked whatever the line said, because the explicit date answers how old
        # the acceptance is and blame answers whether anyone ever reviewed it.
        # Only the second one can be skipped by writing a date by hand.
        committed=$(blame_date "$lineno")
        void=""
        epoch=""
        if [[ -z "$committed" ]]; then
            void="no commit behind the line"
        else
            [[ -n "$date" ]] || date="$committed"
            epoch=$(epoch_of "$date")
            if [[ -z "$epoch" ]]; then
                void="a date this script cannot read ($date)"
            elif (( epoch > now )); then
                # Without this, 2999-01-01 is an acceptance that never ages, and
                # ageing is the only thing that ever takes one back.
                void="a date that has not happened yet ($date)"
            fi
        fi
        ACCEPT_REASON["$key"]="$reason"
        ACCEPT_DATE["$key"]="$date"
        ACCEPT_EPOCH["$key"]="$epoch"
        ACCEPT_VOID["$key"]="$void"
        ACCEPT_ORDER+=("$key")
    done < "$ACCEPT_FILE"
}

# --- 1. .claude/ is a real directory ------------------------------------------
check_claude_dir() {
    if [[ -L ".claude" ]]; then
        required ".claude/" "a symlink — Claude Code skips symlinked config directories, so none of this loads"
        return 1
    fi
    if [[ ! -d ".claude" ]]; then
        required ".claude/" "not installed — run /project-setup; this check never installs anything itself"
        return 1
    fi
    ok ".claude/" "a real directory"
    return 0
}

# --- 2. the hooks are present, executable, and stamped with the template's version
stamp_of() { [[ -r "$1" ]] && tr -d '[:space:]' < "$1"; }

# `test -x` asks access(2), which answers no for every file on a `noexec` mount —
# so under a noexec $TMPDIR it would report every hook broken and be wrong
# about every one of them. The mode bit is what a `cp -r` with no `chmod +x`
# actually loses, and it is what this asks for.
has_exec_bit() {
    local mode
    mode=$(stat -c '%a' "$1" 2>/dev/null) || mode=$(stat -f '%Lp' "$1" 2>/dev/null) || return 1
    (( (8#$mode & 0111) != 0 ))
}

check_hooks() {
    local findings=() hook missing=0 installed template note=""
    for hook in "${REQUIRED_HOOKS[@]}"; do
        if [[ ! -f ".claude/hooks/$hook" ]]; then
            missing=$(( missing + 1 ))
            findings+=("$hook missing")
        elif ! has_exec_bit ".claude/hooks/$hook"; then
            findings+=("$hook not executable")
        fi
    done
    # A project that never had the hooks installed is the common case, and
    # naming every one of them one at a time buries that under its own detail.
    if (( missing == ${#REQUIRED_HOOKS[@]} )); then
        required "hooks" "not one of the ${#REQUIRED_HOOKS[@]} hooks is installed — run /project-setup"
        return
    fi
    installed=$(stamp_of ".claude/VERSION")
    template=$(stamp_of "$TEMPLATE_DIR/VERSION")
    if [[ -z "$installed" ]]; then
        findings+=("no .claude/VERSION — these hooks predate the stamp")
    elif [[ -z "$template" ]]; then
        note=" (no template to compare against at $TEMPLATE_DIR)"
    elif [[ "$installed" != "$template" ]]; then
        findings+=("stamped $installed, the template is $template")
    fi
    if (( ${#findings[@]} )); then
        required "hooks" "$(join_with '; ' "${findings[@]}") — run /project-setup"
        return
    fi
    ok "hooks" "${#REQUIRED_HOOKS[@]} present and executable, stamped ${installed}${note}"
}

# --- 3. gates.sh exists and has at least one gate in it -----------------------
check_gates_file() {
    if [[ ! -f ".claude/gates.sh" ]]; then
        required "gates.sh" "missing — /ship then hand-rolls its own gates every time, which is how a project runs ungated for months"
        return 1
    fi
    if ! bash -n ".claude/gates.sh" 2>/dev/null; then
        required "gates.sh" "does not parse as shell — every hook that sources it fails silently"
        return 1
    fi
    if ! source_gate_lib; then
        required "gates.sh" "cannot read .claude/hooks/gate-lib.sh, so the keys cannot be loaded"
        return 1
    fi
    load_gates
    if [[ -z "${LINT_CMD}${TYPE_CMD}${TEST_CMD}" ]]; then
        required "gates.sh" "present, but every gate is empty — each turn passes by having nothing to run"
        return 1
    fi
    # The inert list prints here and nowhere else, so widening it cannot be
    # quiet: every session in this project sees what it has stopped gating,
    # before its first tool call. Absent from the row when the key is unset,
    # which is most projects and is the behaviour that was always there.
    local inert=""
    [[ -n "${GATE_INERT_PATHS:-}" ]] && inert="; inert: ${GATE_INERT_PATHS}"
    ok "gates.sh" "$(join_with ', ' ${LINT_CMD:+lint} ${TYPE_CMD:+types} ${TEST_CMD:+tests}) configured${inert}"
    return 0
}

# --- 4. each configured gate command's first word resolves --------------------
# `command -v` only, never the command itself: this runs at SessionStart, where
# running a gate would put a test suite in front of the first tool call.
#
# A runner's own name resolving proves nothing about what it was asked to run —
# `npm run nosuchscript` exits 1 exactly like a test suite with one failure — so
# the `<runner> run <script>` form is checked against package.json as well. The
# looser `pnpm <script>` form is not: `pnpm exec` and `pnpm test` are the
# runner's own verbs, and telling those from a script name needs a list of
# builtins per runner that would be wrong by the next release.
node_script() {  # node_script <runner> <second word> <third word>
    case "$1" in
        npm|pnpm|yarn|bun) [[ "${2:-}" == "run" ]] && printf '%s' "${3:-}" ;;
    esac
    return 0
}

resolve_gate() {  # resolve_gate <command>; prints why it does not resolve
    local cmd="$1" words=() script
    read -r -a words <<<"$cmd"
    if ! command -v "${words[0]}" >/dev/null 2>&1; then
        printf '%s is not on PATH' "${words[0]}"
        return 1
    fi
    script=$(node_script "${words[0]}" "${words[1]:-}" "${words[2]:-}")
    [[ -z "$script" ]] && return 0
    if [[ ! -f "package.json" ]]; then
        printf 'no package.json to define the %s script' "$script"
        return 1
    fi
    command -v jq >/dev/null 2>&1 || return 0
    if ! jq -e --arg s "$script" '.scripts | has($s)' package.json >/dev/null 2>&1; then
        printf 'package.json has no %s script' "$script"
        return 1
    fi
    return 0
}

check_gate_commands() {
    local findings=() label cmd why
    while IFS=$'\t' read -r label cmd; do
        [[ -z "$cmd" ]] && continue
        why=$(resolve_gate "$cmd") || findings+=("$label ($cmd): $why")
    done < <(printf 'lint\t%s\ntypes\t%s\ntests\t%s\n' "$LINT_CMD" "$TYPE_CMD" "$TEST_CMD")
    if (( ${#findings[@]} )); then
        required "gate commands" "$(join_with '; ' "${findings[@]}")"
        return
    fi
    ok "gate commands" "every configured gate's first word resolves"
}

# --- 5. the state directory and the local settings are excluded ---------------
# Asked of git rather than grepped out of .gitignore, because the question is
# "would this be committed" and an allowlist-style ignore file answers it with
# no line for either path.
#
# git cannot resolve a pathspec that sits beyond a symlink, and a symlinked
# `.claude` is one of the shapes this script exists to catch — so exit 128 falls
# back to the literal lines /project-setup writes rather than reading as "not
# ignored" and inventing a second finding out of the first one.
path_ignored() {
    local path="$1" rc
    git check-ignore -q "$path" 2>/dev/null
    rc=$?
    (( rc < 2 )) && return "$rc"
    grep -qxF -e "$path" -e "/$path" .gitignore 2>/dev/null
}

check_gitignore() {
    local findings=() path
    for path in ".claude/state/" ".claude/settings.local.json"; do
        path_ignored "$path" || findings+=("$path")
    done
    if (( ${#findings[@]} )); then
        required ".gitignore" "does not exclude $(join_with ' or ' "${findings[@]}") — per-session markers would be committed"
        return
    fi
    ok ".gitignore" "excludes the state directory and the local settings"
}

# --- 6. origin/HEAD is set ----------------------------------------------------
check_origin_head() {
    if ! git remote get-url origin >/dev/null 2>&1; then
        required "origin/HEAD" "there is no origin remote, so nothing says what this checkout is behind"
        return 1
    fi
    if ! BASE=$(git symbolic-ref --short -q refs/remotes/origin/HEAD); then
        required "origin/HEAD" "unset — run \`git remote set-head origin -a\`; a guessed main scopes a diff against nothing"
        return 1
    fi
    ok "origin/HEAD" "$BASE"
    return 0
}

# --- 7. the checkout is current -----------------------------------------------
fetch_origin() {
    FETCH_NOTE=""
    (( OFFLINE )) && return 0
    git remote get-url origin >/dev/null 2>&1 || return 0
    local out
    out=$(run_bounded git fetch origin --prune 2>&1) && return 0
    FETCH_NOTE="not fetched ($(printf '%s' "$out" | tail -1 | cut -c1-60))"
    return 1
}

# Advisory for an ordinary turn and required for one that measures anything.
# The split lives here rather than in a skill's prose so that every caller gets
# the same answer to "does this finding stop the run".
classify_checkout() {
    if (( MEASURING )); then required "$@"; else advisory "$@"; fi
}

check_checkout() {
    local base="${BASE_REF:-$BASE}" parts=() behind
    if [[ -n "$base" ]] && git rev-parse --verify -q "$base" >/dev/null; then
        behind=$(git rev-list --count "HEAD..$base" 2>/dev/null || printf '0')
        (( behind > 0 )) && parts+=("$behind behind $base")
    fi
    [[ "$(git status -sb 2>/dev/null | head -1)" == *"[gone]"* ]] && parts+=("upstream gone")
    [[ -n "$FETCH_NOTE" ]] && parts+=("$FETCH_NOTE")
    if (( ${#parts[@]} == 0 )); then
        ok "checkout" "fetched, and current with ${base:-origin}"
        return
    fi
    classify_checkout "checkout" "$(join_with ', ' "${parts[@]}")"
}

# --- 8. the local ref named after the default branch is not stale -------------
# A different question from check_checkout, which asks whether *this branch* is
# built on current code. This one asks whether a stale ref is lying around that
# a diff might get scoped against by mistake — and on 2026-09-17 there was. Step
# 0 passed clean on a branch of sophia while local `master` sat 86 commits
# behind `origin/master`: `git diff master..HEAD` reported 69 files and +9311
# lines where the right base reported 8 and +285, and the first number was about
# to be posted to the issue as that phase's record. Nothing downstream could
# have caught it — which is also why the row names both refs, since "86 behind"
# without saying behind what is the ambiguity that made the mistake possible.
#
# Required under --measuring for the checkout row's reason, with one exception:
# a caller that passed --base has named the ref it measures against, so the
# stale local one is not what its diff is scoped to. /ship step 0 always passes
# --measuring and a stacked run drifts behind local master by construction, so
# without it every phase after the first would halt on a ref that has nothing to
# do with its diff. Unless the ref it named *is* the stale one: `--base master`
# is that rationale inverted, and the shape #29 was written about, so it stays
# required rather than slipping through the door the exception opened.
classify_default_ref() {
    local branch="${BASE#origin/}" named="${BASE_REF#refs/heads/}"
    if (( MEASURING )) && [[ -z "$named" || "$named" == "$branch" ]]; then
        required "$@"
    else
        advisory "$@"
    fi
}

check_default_ref() {
    local branch behind head
    # origin/HEAD unset, or no origin: check 6 said so, and nothing names a ref.
    [[ -n "$BASE" ]] || return
    branch="${BASE#origin/}"
    if ! git rev-parse --verify -q "refs/heads/$branch" >/dev/null; then
        ok "local default" "no local $branch to go stale"
        return
    fi
    # Checked out, `$branch..HEAD` is empty and there is nothing to mis-scope,
    # so the checkout row owns HEAD — but only where it is measuring against the
    # same ref. Under --base it is not, and handing it a staleness it never
    # looked at would leave the condition reported by neither row.
    head=$(git symbolic-ref --short -q HEAD)
    if [[ "$head" == "$branch" && ( -z "$BASE_REF" || "$BASE_REF" == "$BASE" ) ]]; then
        ok "local default" "$branch is checked out — the checkout row above owns it"
        return
    fi
    behind=$(git rev-list --count "refs/heads/$branch..$BASE" 2>/dev/null || printf '0')
    if (( behind == 0 )); then
        ok "local default" "$branch is current with $BASE"
        return
    fi
    classify_default_ref "local default" \
        "$branch is $behind behind $BASE — scope a diff against $BASE, not $branch"
}

# --- 9. the forge resolves to the host CI runs on -----------------------------
url_host() {
    local url="$1"
    url="${url#*://}"   # https://host/path, ssh://git@host:22/path
    url="${url#*@}"     # git@host:path
    url="${url%%/*}"
    printf '%s' "${url%%:*}"
}

glab_host() {
    local json
    json=$(run_bounded glab repo view -F json 2>/dev/null) || return 1
    url_host "$(printf '%s' "$json" | jq -r '.web_url // .http_url_to_repo // empty' 2>/dev/null)"
}

check_forge() {
    local remote url host origin_host="" gitlab_remotes=0 resolved
    while read -r remote url _; do
        host=$(url_host "$url")
        [[ "$remote" == "origin" ]] && origin_host="$host"
        [[ "$host" == *gitlab* ]] && gitlab_remotes=$(( gitlab_remotes + 1 ))
    done < <(git remote -v 2>/dev/null | awk '$3 == "(fetch)"')
    if (( gitlab_remotes == 0 )); then
        ok "forge" "no GitLab remote"
        return
    fi
    # glab picks its host from the remotes and prefers one named literally
    # `gitlab` over `origin`. That is how commands meant for the self-hosted
    # instance silently reached gitlab.com, and it is invisible in every log.
    if git remote 2>/dev/null | grep -qx "gitlab"; then
        required "forge" "a remote named \`gitlab\` outranks origin for glab — rename it, CI runs on origin's host"
        return
    fi
    if (( OFFLINE )); then
        ok "forge" "origin is $origin_host (glab not consulted)"
        return
    fi
    if ! command -v glab >/dev/null 2>&1; then
        advisory "forge" "origin is $origin_host but glab is not installed"
        return
    fi
    resolved=$(glab_host)
    if [[ -z "$resolved" ]]; then
        advisory "forge" "glab could not resolve this repo — origin is $origin_host"
    elif [[ "$resolved" != "$origin_host" ]]; then
        required "forge" "glab targets $resolved but origin is $origin_host — CI runs on origin's host"
    else
        ok "forge" "$resolved, the host origin points at"
    fi
}

# --- 10. the run contract ------------------------------------------------------
# The foreman checks the contract against itself and never against the stack:
# what counts as a runnable surface is the project's business, and SURFACE_PATHS
# is where the project has already said so.
#
# A surface with no RUN_CMD is required rather than advisory, because /ship step
# 2c skips the proof of use when RUN_CMD is empty — so on a UI project that pair
# is the bypass, and step 0 stopping on it is what closes it.
check_run_contract() {
    local findings=()
    if [[ -n "$SURFACE_PATHS" && -z "$RUN_CMD" ]]; then
        required "run contract" \
            "SURFACE_PATHS names a surface but RUN_CMD is empty — /ship cannot prove it works"
        return
    fi
    [[ -n "$RUN_CMD" && -z "$READY_URL" ]] \
        && findings+=("RUN_CMD is set but no READY_URL proves it came up")
    if (( ${#findings[@]} )); then
        advisory "run contract" "$(join_with '; ' "${findings[@]}")"
        return
    fi
    if [[ -z "$RUN_CMD" ]]; then
        ok "run contract" "no runnable surface recorded"
        return
    fi
    ok "run contract" "RUN_CMD and READY_URL set${SESSION_CMD:+, sign-in recorded}"
}

# --- 11. open deferred findings -----------------------------------------------
check_deferrals() {
    local summary count severity
    if [[ ! -f "$DEFERRAL_LEDGER" ]] || ! command -v jq >/dev/null 2>&1; then
        ok "deferrals" "no ledger recorded"
        return
    fi
    summary=$(jq -Rrn --argjson rank '{"CRITICAL":4,"HIGH":3,"MEDIUM":2,"LOW":1}' '
        [inputs | fromjson? | select(type == "object") | select((.resolved // false) | not)] as $open
        | ($open | map(.severity // "LOW" | ascii_upcase) | max_by($rank[.] // 0) // "none") as $worst
        | "\($open | length) \($worst)"
    ' < "$DEFERRAL_LEDGER" 2>/dev/null) || summary=""
    if [[ -z "$summary" ]]; then
        advisory "deferrals" "$DEFERRAL_LEDGER does not parse as one JSON object per line"
        return
    fi
    read -r count severity <<<"$summary"
    if (( count == 0 )); then
        ok "deferrals" "none open"
        return
    fi
    advisory "deferrals" "$count open, highest $severity"
}

# --- 12. bypasses of the Stop gate --------------------------------------------
# Advisory, always, and never required. A bypass that failed /ship step 0 would
# be a bypass somebody routes around by deleting the hook, and a deleted hook
# records nothing at all — which is the blind spot this row exists to close, not
# widen. Reporting first; refusing to merge a phase that shipped under one is
# deliberately out of scope.
#
# gate-lib carries the window and the reading, because session-start.sh sweeps
# these markers on the same constant. Sourced here when nothing above has: the
# rows before this one only load it once `.claude/` and gates.sh are both intact,
# and a project missing either still bypasses the gate.
check_bypasses() {
    local summary count runner via date reason
    declare -F bypass_summary >/dev/null || source_gate_lib || return
    declare -F bypass_summary >/dev/null || return
    summary=$(bypass_summary "$STATE_DIR") || summary=""
    if [[ -z "$summary" ]]; then
        ok "bypasses" "none in the last $STATE_RETENTION_DAYS days"
        return
    fi
    IFS=$'\t' read -r count runner via date reason <<<"$summary"
    if (( count == 0 && runner == 0 )); then
        # Reachable only through markers that do not parse: `bypass_summary`
        # returns non-zero when there are none at all. A corrupt marker reading
        # as `none` is this row's own failure mode wearing a smaller face.
        advisory "bypasses" "$(bypass_markers "$STATE_DIR" | wc -l) marker(s) in $STATE_DIR that do not parse as one JSON object each"
        return
    fi
    if (( count == 0 )); then
        # Not silence: the runner turning the gate off around every implement
        # turn is the ordinary case here, and a bare "none" over a week of them
        # would read as a gate nothing ever got past.
        ok "bypasses" "none chosen in the last $STATE_RETENTION_DAYS days · $runner by the milestone runner"
        return
    fi
    advisory "bypasses" \
        "$count in the last $STATE_RETENTION_DAYS days · last: $via, \"$reason\" ($date)"
}

# --- 13. what the gates cost --------------------------------------------------
# Advisory at the very most. `gates.sh` has said "30s is the budget the Stop hook
# is built around, not a target to drift past" since there was a Stop hook, and
# until now nothing measured it — the one figure anybody had was a comment and a
# stopwatch reading somebody took by hand. A gate that *failed* for being slow
# would be turned off on the first loaded machine, so this row reports the drift
# and stops there.
#
# A marker carrying no `total_s` is unknown and is counted apart, never folded in
# as `0s`: markers written before the key existed are exactly that, and reading
# one as a free run would pull the maximum down — the direction that hides the
# drift the row exists to show.
check_gate_cost() {
    local summary last max unknown note
    declare -F gate_cost_summary >/dev/null || source_gate_lib || return
    declare -F gate_cost_summary >/dev/null || return
    summary=$(gate_cost_summary "$STATE_DIR") || summary=""
    if [[ -z "$summary" ]]; then
        ok "gate cost" "nothing measured in the last $STATE_RETENTION_DAYS days · budget ${GATE_BUDGET_S}s"
        return
    fi
    IFS=$'\t' read -r last max unknown <<<"$summary"
    note="last ${last}s · max ${max}s · budget ${GATE_BUDGET_S}s"
    (( unknown )) && note+=" · ${unknown} unmeasured"
    if (( max > GATE_BUDGET_S )); then
        advisory "gate cost" "$note"
        return
    fi
    ok "gate cost" "$note"
}

# --- 14. the acceptance file accounts for itself ------------------------------
# Runs last because it is the only check that needs to know what the others
# printed. A key that accepted nothing is a row of its own rather than a
# silence: an acceptance outliving the finding it was written for is the way
# this file rots, and it rots in the direction of accepting more than anyone
# meant to.
check_acceptances() {
    local i note key
    for (( i = 0; i < ${#ACCEPT_NOTES[@]}; i++ )); do
        note="${ACCEPT_NOTES[$i]}"
        advisory "doctor-accept" "$note"
    done
    for (( i = 0; i < ${#ACCEPT_ORDER[@]}; i++ )); do
        key="${ACCEPT_ORDER[$i]}"
        [[ -n "${ACCEPT_USED[$key]:-}" ]] && continue
        # `ok` and not merely "not required this run". Three rows would
        # otherwise be called stale while in force: one whose check never ran,
        # because a missing `.claude/` stops everything under it, and `checkout`
        # and `local default`, both advisory here and required under
        # --measuring. Telling the operator to delete the line at every
        # SessionStart would break the /ship step 0 it was written for.
        contains "ok"$'\t'"$key" ${EMITTED_ROWS[@]+"${EMITTED_ROWS[@]}"} || continue
        advisory "doctor-accept" "\`$key\` is accepted and is not a finding here — delete the line"
    done
}

# --- the run ------------------------------------------------------------------
summarise() {
    local i
    # Above the verdict, never folded into it. /ship step 0 reports this block,
    # so shipping under an acceptance is something the reviewer sees rather than
    # something they would have had to notice in one row of a ten-row table.
    for (( i = 0; i < ${#ACCEPTED_ROWS[@]}; i++ )); do
        (( i == 0 )) && printf '  accepted, and shipping under it is on this record:\n'
        printf '    accept  %s\n' "${ACCEPTED_ROWS[$i]}"
    done
    if (( REQUIRED_FINDINGS == 0 && ADVISORY_FINDINGS == 0 && ACCEPTED_FINDINGS == 0 )); then
        printf '  nothing to report.\n'
        return 0
    fi
    printf '  %d required, %d advisory, %d accepted. A required finding fails /ship step 0.\n' \
        "$REQUIRED_FINDINGS" "$ADVISORY_FINDINGS" "$ACCEPTED_FINDINGS"
    (( REQUIRED_FINDINGS == 0 ))
}

main() {
    local root gates_loaded=0
    root=$(git -C "${TARGET:-$PWD}" rev-parse --show-toplevel 2>/dev/null) || {
        printf 'project-foreman: %s is not a git repository\n' "${TARGET:-$PWD}" >&2
        return 2
    }
    cd "$root" || return 2
    load_acceptances
    fetch_origin || true

    printf 'project-foreman — %s\n' "$root"
    if check_claude_dir; then
        check_hooks
        if check_gates_file; then
            check_gate_commands
            gates_loaded=1
        fi
    fi
    check_gitignore
    check_origin_head || true
    check_checkout
    check_default_ref
    check_forge
    (( gates_loaded )) && check_run_contract
    check_deferrals
    check_bypasses
    # Only with the gates loaded: without them there is no declared budget, and a
    # cost reported against no budget is a number with nothing to mean.
    (( gates_loaded )) && check_gate_cost
    check_acceptances
    summarise
}

main
exit $?
