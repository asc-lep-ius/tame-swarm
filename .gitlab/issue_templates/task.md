<!-- ──────────────────────────────────────────────────────────────────────────
     TASK
     Docs, CI, tooling, refactors, dependency updates, the paper — anything
     that changes neither what the organism does nor what is known about it.
     For a question a number answers → measurement. For a change to what the
     organism does → mechanism. For code that does not do what it says → defect.
     Delete the sections that do not apply; a blank section is worse than none.
     These templates are revisited after the Phase 1.5 issues close: the first
     seven uses decide what stays.
     ────────────────────────────────────────────────────────────────────────── -->

## Summary

<!-- What needs doing and why, in one short paragraph. The goal, not the steps. -->

## Why now

<!-- What this unblocks or what it stops from rotting. Link the issue or the
     README section. "homeostat.py is 916 lines against an 800 maximum and the
     lift-out has been deferred three times" is a reason; "cleanup" is not. -->

## Definition of done

- [ ] `uv run ruff check . && uv run ruff format --check .` clean
- [ ] `uv run pyright tame scripts` clean
- [ ] `uv run pytest` green (CPU); `uv run pytest -m gpu` green if the substrate or the served surface is touched, inside the 160 s budget
- [ ] README updated where it names the thing changed
- [ ] No recorded number changed — or, if one had to, it is re-measured and the change says so

## Verification

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright tame scripts
uv run pytest
```

## Out of scope

<!-- Opportunistic cleanup is capped at five lines outside the change asked for;
     anything larger is its own task. -->

- 

## Dependencies

| Relationship | Issue |
|---|---|
| Blocked by | <!-- #N or n/a --> |
| Unblocks | <!-- #N or n/a --> |

## Touches

<!-- One line: the modules and scripts this lands in, e.g. `tame/train.py`, `scripts/run_seeds.py`, README `#…`. Labels carry the type. -->


---

/label ~task
