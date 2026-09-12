<!-- ──────────────────────────────────────────────────────────────────────────
     DEFECT
     Code that does not do what it says: a silent no-op, a hidden confound, a
     measurement artefact, a crash. The first three are the ones this project
     has been burned by; a crash is the easy case.
     For a question a number answers → measurement. For a change to what the
     organism does → mechanism. For docs, CI and tooling → task.
     Delete the sections that do not apply; a blank section is worse than none.
     These templates are revisited after the Phase 1.5 issues close: the first
     seven uses decide what stays.
     ────────────────────────────────────────────────────────────────────────── -->

## Summary

<!-- What the code SAYS it does, and what it DOES, in two sentences.
     Example: "`_save_checkpoint` says it saves the model and the wealth state; under
     `--use_lora` it saves the LoRA attention adapters and the ledgers, and the trained
     MoB expert adapters and confidence heads are not written anywhere." -->

## Class

- [ ] Silent no-op — a field, flag or hook that is set and never acts (`tests/test_no_silent_noops.py` is the pattern)
- [ ] Hidden confound — two arms that differ in something the fingerprint does not carry
- [ ] Measurement artefact — a number that describes the instrument rather than the organism
- [ ] Crash or wrong value

## Where it showed

<!-- The measurement that exposed it, with the number that looked wrong and the
     README anchor or run directory. A defect found by a number is worth more
     than one found by reading, because the number says how much it mattered. -->

## Reproduce

```bash
# the exact command or test; a checkpoint path or run directory if it needs one
```

## Blast radius

<!-- Which recorded claims, constants and runs it touches, and which stay valid.
     Be specific: "every checkpoint written by a LoRA run since #15" is a
     different sentence from "checkpoints". Say what does NOT need re-measuring. -->

## Fix shape, and the test that would have caught it

<!-- The repo rule: every fix adds the test that would have failed before it.
     For a silent no-op that means a test on the exact code path the defect hid in
     (the PEFT-wrapped model, not the bare one). For a confound, a fingerprint
     field. For an artefact, a control the number must be read beside. -->

- Fix: 
- Test: 

## Out of scope

- 

## Dependencies

| Relationship | Issue |
|---|---|
| Found by | <!-- #N --> |
| Blocks | <!-- #N or n/a --> |

## Acceptance criteria

- [ ] The test that would have caught it fails on the current code and passes after the fix
- [ ] The blast radius is recorded where the touched numbers live (README, expected-failure reasons, audit notes)
- [ ] 

## Touches

<!-- One line: the modules and scripts this lands in, e.g. `tame/train.py`, `scripts/run_seeds.py`, README `#…`. Labels carry the type. -->


---

/label ~defect
