#!/usr/bin/env bash
# Run a GPU command only while CI is not holding the card.
#
# `resource_group: gpu` in .gitlab-ci.yml serialises test-gpu, train-smoke and
# perf-regression against each other. It knows nothing about a process started
# by hand, and the runner tagged `workstation` *is* this workstation — so a
# local GPU run that begins while CI holds the group shares one 16 GB card with
# a job that measured its budget alone. It will not corrupt the numerics; it
# will OOM the job, or push it past the 300 s its log asserts against, and a
# blown budget reads as a performance regression in the code. That is the most
# expensive false red this repo can produce, which is why this refuses rather
# than queues: queueing would make the local run look merely slow, and a wait
# and a refusal reach a terminal as the same silence.
#
#   scripts/gpu_gate.sh                 report only; exit 0 when the group is free
#   scripts/gpu_gate.sh <cmd> [args…]   report, then run <cmd> when it is free
#
#   TAME_GPU_GATE_FORCE=1   run anyway, for a box with no route to the forge. It
#                           says so on the way past. Never the default: "nobody
#                           could be asked" and "nobody is using it" arrive as
#                           the same silence, and only one of them is safe.
#   TAME_GPU_GATE_TIMEOUT   seconds allowed for the forge query (default 10)
#
# exit 0  the group is free — and, with a command, whatever that command exited
# exit 1  the group is held, or nothing could establish that it is not
#
# It asks once, before starting. A CI job that begins after that still collides,
# and nothing here holds the group against the runner either — the guard closes
# the common case, which is starting a five-minute suite on top of a job already
# running, and not the race. Naming what it does not cover is cheaper than
# discovering it from a job log.
set -uo pipefail

CI_FILE=".gitlab-ci.yml"
RESOURCE_GROUP="gpu"
BUSY_SCOPES="scope[]=running&scope[]=pending"
JOBS_PER_PAGE=100
QUERY_TIMEOUT="${TAME_GPU_GATE_TIMEOUT:-10}"

say() { printf 'gpu-gate: %s\n' "$*" >&2; }

# The job names that share the resource group, read out of the CI file rather
# than listed here: a fourth GPU job added without touching this script would
# otherwise be invisible to it. `extends:` is deliberately not resolved — a
# template carrying the key would be missed — so a parse that finds nothing is
# reported as a refusal by the caller, never as "the group is free".
gpu_jobs() {  # gpu_jobs <ci file>
    awk -v group="$RESOURCE_GROUP" '
        /^[^[:space:]#]/ {
            job = ""
            # The name is what precedes the key colon, and nothing after it.
            # Two shapes break the obvious readings. `test:3.12:` is a real job
            # name — sophia names jobs that way — so truncating at the *first*
            # colon records it as `test`; and `test-gpu: &gpu-job` carries a YAML
            # anchor, so stripping only a *trailing* colon keeps the anchor in
            # the name. Either way the name matches no running job and the card
            # reads as free. Matching the key itself and taking what is left of
            # its colon handles both, and a trailing comment falls outside it.
            if (match($0, /^[A-Za-z_][A-Za-z0-9_.:-]*:/)) {
                job = substr($0, 1, RLENGTH - 1)
            }
            next
        }
        # Quotes stripped before comparing: `resource_group: "gpu"` is valid YAML
        # and would otherwise drop that job out of the guarded list silently.
        job != "" && $1 == "resource_group:" {
            value = $2
            gsub(/^["'"'"']|["'"'"']$/, "", value)
            if (value == group) print job
        }
    ' "$1"
}

# Named against origin explicitly. glab picks its host from the remotes and
# prefers one called literally `gitlab` over `origin`, which is how a query
# meant for the self-hosted instance silently reaches gitlab.com — and a
# gitlab.com answer of "nothing is running" would be true and useless.
# `error` rather than `empty` on a body that is not the jobs array: jq exits 5,
# pipefail carries it out, and the caller refuses. `empty` would map an HTTP 200
# carrying `{"message":"401 Unauthorized"}` onto "nothing is running", which is
# the one way this script could say the card is free without having been told so.
busy_jobs() {  # busy_jobs <origin url>; prints "<name>\t<status>\t<pipeline>"
    timeout "$QUERY_TIMEOUT" glab api -R "$1" \
        "projects/:id/jobs?${BUSY_SCOPES}&per_page=${JOBS_PER_PAGE}" 2>/dev/null \
        | jq -r 'if type == "array" then .[] | "\(.name)\t\(.status)\t\(.pipeline.id)"
                 else error("unexpected response from the jobs API") end'
}

# Which of the guarded jobs are in that list, one per line. Reads busy_jobs'
# output on stdin so the forge is asked exactly once per run.
held_by_ci() {  # held_by_ci <guarded job>…  < busy_jobs output
    local name status pipeline job
    while IFS=$'\t' read -r name status pipeline; do
        [[ -z "$name" ]] && continue
        for job in "$@"; do
            [[ "$name" == "$job" ]] && printf '  %s (%s, pipeline %s)\n' "$name" "$status" "$pipeline"
        done
    done
}

# 0 when the card is demonstrably free, 1 otherwise — including every case where
# the question could not be put. Prints the reason either way.
group_is_free() {
    local origin_url guarded=() busy held
    [[ -f "$CI_FILE" ]] || { say "no $CI_FILE here — nothing says which jobs share the card"; return 1; }
    mapfile -t guarded < <(gpu_jobs "$CI_FILE")
    (( ${#guarded[@]} )) || {
        say "$CI_FILE names no job with resource_group: $RESOURCE_GROUP — that is a change to the CI file, not a free card"
        return 1
    }
    origin_url=$(git remote get-url origin 2>/dev/null)
    [[ -n "$origin_url" ]] || { say "no origin remote to ask what is running"; return 1; }
    if ! command -v glab >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
        say "glab and jq are what ask the forge, and one of them is missing"
        return 1
    fi
    # An empty array and a failed query both print nothing, so the exit status is
    # what tells them apart and it is kept rather than re-derived from the text.
    if ! busy=$(busy_jobs "$origin_url"); then
        say "no usable answer from $origin_url about what is running — timed out after ${QUERY_TIMEOUT}s, or the reply was not the jobs array (jq says which, above)"
        return 1
    fi
    held=$(held_by_ci "${guarded[@]}" <<<"$busy")
    if [[ -n "$held" ]]; then
        say "CI holds resource_group: $RESOURCE_GROUP — wait for it to finish:"
        printf '%s\n' "$held" >&2
        return 1
    fi
    say "resource_group: $RESOURCE_GROUP is free (${guarded[*]} idle)"
    return 0
}

root=$(git rev-parse --show-toplevel 2>/dev/null)
[[ -n "$root" ]] || { say "not a git repository, so nothing names the CI file"; exit 1; }
cd "$root" || exit 1

if ! group_is_free; then
    if [[ "${TAME_GPU_GATE_FORCE:-}" != "1" ]]; then
        say "not running — re-run with TAME_GPU_GATE_FORCE=1 if you mean to share the card"
        exit 1
    fi
    say "TAME_GPU_GATE_FORCE=1 — going ahead anyway"
fi

(( $# == 0 )) && exit 0
exec "$@"
