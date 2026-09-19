"""#32's primary is one slope over three doses; pin its arithmetic on hand-built shifts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dose_slope import per_seed_slopes, slope  # noqa: E402


def test_the_slope_is_least_squares_and_exact_on_a_line():
    assert slope([(0.1, 0.11), (0.3, 0.13), (1.0, 0.20)]) == pytest.approx(0.1)
    assert slope([(0.1, 0.2), (0.3, 0.2), (1.0, 0.2)]) == pytest.approx(0.0, abs=1e-12)


def test_a_slope_needs_two_distinct_doses():
    with pytest.raises(ValueError, match="at least two"):
        slope([(0.1, 0.2)])
    with pytest.raises(ValueError, match="same"):
        slope([(0.1, 0.2), (0.1, 0.3)])


def test_per_seed_slopes_pair_the_doses_by_seed_and_drop_unshared_seeds():
    shifts = {
        0.1: {"0": 0.11, "1": 0.30, "2": 0.5},
        0.3: {"0": 0.13, "1": 0.30},
        1.0: {"0": 0.20, "1": 0.30},
    }

    slopes = per_seed_slopes(shifts)

    assert slopes == {"0": pytest.approx(0.1), "1": 0.0}


def test_no_shared_seed_is_refused():
    with pytest.raises(ValueError, match="share no seed"):
        per_seed_slopes({0.1: {"0": 0.1}, 0.3: {"1": 0.1}})


def test_the_cap_column_reads_the_last_record_and_refuses_a_missing_run(tmp_path):
    from dose_slope import cap_readings

    run = tmp_path / "runs" / "seed0"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        '{"coupling/delta_fraction_max": 0.05, "coupling/delta_fraction_mean": 0.01}\n'
        '{"coupling/delta_fraction_max": 0.1006, "coupling/delta_fraction_mean": 0.094}\n'
    )

    assert cap_readings(tmp_path, ["0"]) == {
        "0": {"coupling/delta_fraction_max": 0.1006, "coupling/delta_fraction_mean": 0.094}
    }
    with pytest.raises(FileNotFoundError, match="seed 1"):
        cap_readings(tmp_path, ["0", "1"])
