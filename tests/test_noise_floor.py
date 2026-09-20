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
    lender = write_summary(tmp_path / "value", replace(BASE, seed=0))
    borrower = replace(BASE, seed=1, persistence_coupling="shuffled")

    borrowed = borrow_floor(lender, {"1": borrower.as_dict()})

    assert borrowed.replication_std == ZERO_FLOOR
    assert borrowed.is_zero
    assert borrowed.replicate_seed == 0
    assert set(borrowed.knobs) == set(FLOOR_KNOBS)
    assert borrowed.as_dict()["knobs"]["requested_layers"] == list(BASE.requested_layers)


def test_a_floor_measured_at_another_configuration_is_refused_by_name(tmp_path):
    lender = write_summary(tmp_path / "rank32", replace(BASE, adapter_rank=32))
    borrower = replace(BASE, adapter_rank=8)

    with pytest.raises(BorrowedFloorError, match="adapter_rank 32 vs 8"):
        borrow_floor(lender, {"0": borrower.as_dict()})


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
