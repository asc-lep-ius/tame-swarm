"""Which configuration a recorded floor belongs to, and what borrowing it refuses (#56).

Section 8 rule 4 lets a sweep spend its replicate budget on seeds where the floor
is already recorded. What makes that safe rather than convenient is the refusal:
a floor measured at other knobs is another configuration's, and the kernels it
was a property of are not the ones this sweep ran.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from noise_floor import (  # noqa: E402
    FLOOR_KNOBS,
    BorrowedFloorError,
    borrow_floor,
    differing_floor_knobs,
    unclassified_floor_fields,
)

from .arm_fingerprints import BASE  # noqa: E402

ZERO_FLOOR = {"eval/loss": 0.0, "routing/win_share_e0": 0.0}
SHA = "79c1b1755f3a1e0c9d2b4a6e8f0c1d3e5a7b9c11"
# A borrow is between two real sweeps, and a real sweep records its code
# identity (#31). ``BASE`` leaves it unset, which is the legacy case and a
# refusal in its own right, so the fixtures that are *about* something else say
# what code they ran under.
IDENTIFIED = replace(BASE, code_sha=SHA, code_dirty=False)


def write_summary(path: Path, fingerprint, floor=ZERO_FLOOR, **overrides) -> Path:
    """A ``seed_summary.json`` with one seed, as ``run_seeds.py`` writes it."""
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "arm": fingerprint.arm,
        "replicate_seed": fingerprint.seed,
        "replication_std": floor,
        "replication_error": None,
        "floor_recorded_at": None,
        "fingerprints": {str(fingerprint.seed): fingerprint.as_dict()},
    }
    summary.update(overrides)
    (path / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    return path


def test_every_fingerprint_field_is_either_a_floor_knob_or_declared_not_to_be():
    """A field added later must be classified, not merely absent from both lists.

    The same guard ``unchecked_config_fields`` puts on the parity check: a field
    nobody thought about is indistinguishable from one deliberately left out,
    and here that difference is whether a borrowed floor is this run's floor.
    """
    assert unclassified_floor_fields() == ()


def test_the_arms_of_one_sweep_share_a_floor_and_the_seed_does_not_break_it():
    lender = replace(BASE, seed=0, persistence_coupling="value", goal_doses=(0.017,))
    borrower = replace(BASE, seed=7, persistence_coupling="decoupled", goal_doses=(0.068,))

    assert differing_floor_knobs(lender.as_dict(), borrower.as_dict()) == ()


def test_a_knob_that_selects_other_kernels_is_not_a_floor_to_borrow():
    lender = replace(BASE, adapter_rank=32, max_seq_length=512)
    borrower = replace(BASE, adapter_rank=8, max_seq_length=128)

    assert differing_floor_knobs(lender.as_dict(), borrower.as_dict()) == (
        "adapter_rank",
        "max_seq_length",
    )


def test_a_sequence_knob_survives_the_json_round_trip(tmp_path):
    """``requested_layers`` returns from JSON as a list and must not read as drift."""
    recorded = json.loads(json.dumps(BASE.as_dict()))

    assert isinstance(recorded["requested_layers"], list)
    assert differing_floor_knobs(recorded, BASE.as_dict()) == ()


def test_borrowing_records_the_lender_the_floor_and_the_knobs_it_was_measured_at(tmp_path):
    lender = write_summary(tmp_path / "value", replace(IDENTIFIED, seed=0))
    borrower = replace(IDENTIFIED, seed=1, persistence_coupling="shuffled")

    borrowed = borrow_floor(lender, {"1": borrower.as_dict()})

    assert borrowed.replication_std == ZERO_FLOOR
    assert borrowed.is_zero
    assert borrowed.replicate_seed == 0
    assert set(borrowed.knobs) == set(FLOOR_KNOBS)
    assert borrowed.as_dict()["knobs"]["requested_layers"] == list(BASE.requested_layers)
    # Outside the knobs, and recorded anyway: a reader of a borrowed floor can
    # ask what code measured it without going back to the lender's directory.
    assert borrowed.as_dict()["code_sha"] == SHA
    assert borrowed.as_dict()["code_dirty"] is False


def test_a_floor_measured_at_another_configuration_is_refused_by_name(tmp_path):
    lender = write_summary(tmp_path / "rank32", replace(IDENTIFIED, adapter_rank=32))
    borrower = replace(IDENTIFIED, adapter_rank=8)

    with pytest.raises(BorrowedFloorError, match="adapter_rank 32 vs 8"):
        borrow_floor(lender, {"0": borrower.as_dict()})


def test_a_floor_measured_by_other_code_is_refused_at_identical_knobs(tmp_path):
    """#31's finding is that the floor is a property of the kernels the code selects.

    So the check that matters here cannot be a floor knob: ``code_sha`` is
    excluded from ``FLOOR_KNOBS`` precisely because a *missing* SHA has to count
    as drift and field equality would read two absences as agreement. Three
    shapes, each its own refusal, and each of them accepted silently before.
    """
    other_sha = replace(IDENTIFIED, code_sha="0" * 40)
    dirty = replace(IDENTIFIED, code_dirty=True)
    legacy = replace(IDENTIFIED, code_sha=None, code_dirty=None)

    lender = write_summary(tmp_path / "clean", IDENTIFIED)
    with pytest.raises(BorrowedFloorError, match="different code"):
        borrow_floor(lender, {"0": other_sha.as_dict()})
    with pytest.raises(BorrowedFloorError, match="dirty tree"):
        borrow_floor(lender, {"0": dirty.as_dict()})
    with pytest.raises(BorrowedFloorError, match="no code SHA recorded"):
        borrow_floor(lender, {"0": legacy.as_dict()})

    # And the same three from the other side: a lender with no SHA is every
    # summary written before #31, and it may not lend to a sweep that has one.
    before_31 = write_summary(tmp_path / "legacy", legacy)
    with pytest.raises(BorrowedFloorError, match="no code SHA recorded"):
        borrow_floor(before_31, {"0": IDENTIFIED.as_dict()})


def test_code_drift_is_borrowable_when_the_operator_names_the_decision(tmp_path):
    """``--allow-code-drift``, the escape ``compare_runs.py`` already carries."""
    lender = write_summary(tmp_path / "legacy", replace(IDENTIFIED, code_sha=None, code_dirty=None))

    borrowed = borrow_floor(lender, {"0": IDENTIFIED.as_dict()}, allow_code_drift=True)

    assert borrowed.is_zero
    # The drift is recorded rather than erased by allowing it: the borrowed
    # floor says the lender had no SHA, so the summary that quotes it does too.
    assert borrowed.as_dict()["code_sha"] is None


def test_a_summary_that_measured_no_floor_has_none_to_lend(tmp_path):
    lender = write_summary(
        tmp_path / "no-floor",
        BASE,
        floor=None,
        replication_error="the replicate of seed 0 failed: CUDA out of memory",
    )

    with pytest.raises(BorrowedFloorError, match="recorded none.*out of memory"):
        borrow_floor(lender, {"0": BASE.as_dict()})


def test_a_floor_is_not_relayed_twice(tmp_path):
    """A borrowed floor is not lendable: two hops and nobody measured the configuration."""
    lender = write_summary(
        tmp_path / "borrower",
        BASE,
        floor=None,
        floor_recorded_at={"path": "somewhere/else", "replication_std": ZERO_FLOOR},
    )

    with pytest.raises(BorrowedFloorError, match="borrowed its own floor"):
        borrow_floor(lender, {"0": BASE.as_dict()})


def test_a_summary_written_before_fingerprints_existed_cannot_lend(tmp_path):
    lender = write_summary(tmp_path / "ancient", BASE, fingerprints={})

    with pytest.raises(BorrowedFloorError, match="no arm fingerprints"):
        borrow_floor(lender, {"0": BASE.as_dict()})


def test_a_missing_sweep_directory_is_refused_before_the_runs(tmp_path):
    with pytest.raises(BorrowedFloorError, match="no seed_summary.json"):
        borrow_floor(tmp_path / "never-ran", {})
