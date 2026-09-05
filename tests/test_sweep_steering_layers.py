"""The sweep's pure parts: layer specs and how candidate sets are built from a profile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sweep_steering_layers import candidate_sets, parse_layers  # noqa: E402


def test_parse_layers_accepts_ranges_lists_and_unions():
    assert parse_layers("4-7") == [4, 5, 6, 7]
    assert parse_layers("14,18,22") == [14, 18, 22]
    assert parse_layers("13+16-18") == [13, 16, 17, 18]


def test_candidate_sets_come_from_the_profile_and_explicit_specs():
    profile = {
        layer: dict(margin=margin, passed=margin > 0)
        for layer, margin in {10: -0.1, 11: 0.02, 12: 0.08, 13: 0.05, 14: -0.01, 15: 0.01}.items()
    }
    sets = candidate_sets(
        profile, ["certified", "top2", "window3", "passing", "11+13-14"], certified=(14, 18, 22)
    )
    assert sets["certified"] == [14, 18, 22]
    assert sets["top2"] == [12, 13]
    assert sets["window3"] == [11, 12, 13]
    assert sets["passing"] == [11, 12, 13, 15]
    assert sets["11+13-14"] == [11, 13, 14]
