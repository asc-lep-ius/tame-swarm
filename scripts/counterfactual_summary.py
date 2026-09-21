"""What a counterfactual read writes down: the separation block, and the summary (#58).

Lifted out of ``counterfactual_routing`` when that file crossed the 800-line
maximum in ``~/.claude/rules/coding-standards.md``. The split is along the seam
the file already had: everything here shapes what a read *records*, and nothing
here runs a forward pass.

The summary is deliberately ``run_seeds.py``'s per-seed shape, with each
checkpoint's own ``ArmFingerprint`` copied in, so ``compare_runs.py`` tables the
readings across arms unchanged -- and ``None`` survives that shape as a missing
column rather than as a number, which is what keeps a rate over an empty
population out of a table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# The checkpoint directories a group holds, newest step last.
CHECKPOINT_GLOB = "checkpoint-*"


def separation(read: dict[str, Any], suffix: str) -> dict[str, Any]:
    """The own-route-against-stream pair, its magnitudes, and the pairs behind each.

    ``own_route_effect`` is ``None`` unless both populations cleared
    ``MIN_POPULATION``: a difference of two rates, one of which stands on six
    pairs, is not a reading and must not reach a table as one.
    """
    moved, unmoved = read["moved_better"], read["unmoved_better"]
    swing, unmoved_swing = read["moved_swing"], read["unmoved_swing"]
    return {
        f"counterfactual/moved_better{suffix}": moved,
        f"counterfactual/unmoved_better{suffix}": unmoved,
        f"counterfactual/moved_swing{suffix}": swing,
        f"counterfactual/unmoved_swing{suffix}": unmoved_swing,
        f"counterfactual/own_route_effect{suffix}": (
            moved - unmoved if moved is not None and unmoved is not None else None
        ),
        f"counterfactual/own_route_swing{suffix}": (
            swing - unmoved_swing if swing is not None and unmoved_swing is not None else None
        ),
        f"counterfactual/moved_pairs{suffix}": read["moved_pairs"],
        f"counterfactual/unmoved_pairs{suffix}": read["unmoved_pairs"],
        f"counterfactual/moved_fraction{suffix}": read["moved_fraction"],
        f"counterfactual/confident_better{suffix}": read["confident_better"],
    }


def summarise(readings: dict[str, dict[str, Any]], group: Path | None) -> dict[str, Any]:
    """``run_seeds.py``'s summary shape, so ``compare_runs.py`` tables it unchanged."""
    per_seed = {
        seed: {
            key: value
            for key, value in reading.items()
            if isinstance(value, (int, float)) and "/" in key
        }
        for seed, reading in readings.items()
    }
    fingerprints = {
        seed: reading["arm_fingerprint"]
        for seed, reading in readings.items()
        if reading.get("arm_fingerprint")
    }
    return {
        "arm": next(
            (
                reading["arm_fingerprint"].get("persistence_coupling", "unknown")
                for reading in readings.values()
                if reading.get("arm_fingerprint")
            ),
            "unknown",
        ),
        "router": "mob",
        "group": str(group) if group else None,
        "seeds": sorted(per_seed, key=str),
        "primary": None,
        "per_seed": per_seed,
        "fingerprints": fingerprints,
        "stats": {
            metric: {
                "mean": sum(values) / len(values),
                "std": float(torch.tensor(values).std()) if len(values) > 1 else float("nan"),
                "n": len(values),
            }
            for metric, values in (
                (
                    metric,
                    [row[metric] for row in per_seed.values() if metric in row],
                )
                for metric in sorted({key for row in per_seed.values() for key in row})
            )
        },
        "replicate_seed": None,
        "replicate": None,
        "replication_std": None,
        "replication_error": (
            "exploratory read, not a run: the floor is the replicate pair's per-token spread "
            "(#58), recorded per reading rather than as a seed spread"
        ),
        "floor_recorded_at": None,
        "readings": readings,
    }


def checkpoints_of(group: Path, include_replicates: bool = False) -> dict[str, Path]:
    """Every seed's last checkpoint under a ``run_seeds.py`` group, keyed by seed.

    The replicate is left out by default: it is seed 0 again, it is what the
    floor pass reads, and a group summary that carried it would table one seed
    twice as though it were two.
    """
    found: dict[str, Path] = {}
    for run in sorted((group / "runs").glob("seed*")):
        if not include_replicates and run.name.endswith("-replicate"):
            continue
        checkpoints = sorted(
            run.glob(CHECKPOINT_GLOB), key=lambda path: int(path.name.split("-")[-1])
        )
        if checkpoints:
            found[run.name.replace("seed", "")] = checkpoints[-1]
    if not found:
        raise FileNotFoundError(f"no checkpoints under {group}/runs/seed*/")
    return found
