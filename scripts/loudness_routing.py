"""Loudness and the market: #58's counterfactual-route read on the fixture across scale (#64).

#58 read, on the body, that the executed route is statistically indistinguishable
from a Gumbel sample of the arm's own bids: on the quarter of tokens the body is
surest of, an alternative beats the executed route 0.4949 +/- 0.0438 of the time
(README ``#counterfactual-routing``). #60 hypothesised that at 3.7% of the residual
norm the per-token value differences between cells fall below the confidence
heads' regression noise, so bids carry no token-specific information -- and that
one knob, the cells' share of the output, would fix #58's chance reading and
#60's null together. This is that read, ported to the fixture where the knob
exists (``contribution_scale``) and run at #73's derived exchange rate, so a
louder fixture is still an economy rather than a clamp.

**The port.** The body read forwards a held-out probe once on the executed
route and ``K`` times on Gumbel-top-k alternatives drawn from the arm's own
bids, and scores each token by the log-probability of the token that followed.
The fixture has no next token: a token is scored by its own squared error
against the planted target, lower is better, and the tokens are independent --
there is no stream upstream of a token for a reroute to move, so the body's
"stream only" control is empty by construction and every difference between
two reads is the route's own. The alternatives are the body's
``GumbelTopKRoute`` on the fixture's gate, unchanged, with the ledger frozen
(``frozen_economy``) and the layer in eval mode so the executed route is the
gate's top-k with no exploration gift in it.

**The floor is the alternatives' own RNG.** The fixture is bitwise deterministic
per seed, so the executed read of a settled seed is reproducible to the bit and
the only noise in the read is which 32 alternatives were drawn. The floor is
read as the pair: the same settled seed, the same K, two Gumbel seeds, and the
per-token difference between the two best-of-K gaps; a token is *fragile* when
its gap exceeds the pair's 95th percentile. What is read is in the fixture's
loss units, which grow as the scale squared, so every gap is also reported
relative to the executed loss of the token it was read on.

    uv run python scripts/loudness_routing.py --stage grid --fixtures quality-fixture \\
        --out ~/tame-runs/loudness-routing/
    uv run python scripts/loudness_routing.py --stage summarise \\
        --self-model ~/tame-runs/73-exchange-rate/self-model/self_model.json \\
        --out ~/tame-runs/loudness-routing/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from counterfactual_routing import (  # noqa: E402
    DEFAULT_ALTERNATIVES,
    DEFAULT_NOISE_SCALE,
    GumbelTopKRoute,
)
from measure_ledger_stability import (  # noqa: E402
    DEFAULT_SEEDS as DERIVATION_SEEDS,
)
from measure_ledger_stability import (
    DIFFERENTIATED,
    FIXTURES,
    QUALITY,
    ClampedLedgerError,
    build_economy,
    derivation_path,
    derive_reward_scale,
)
from measure_stakes_dial import fixture_fingerprint  # noqa: E402
from power import pairs_for_power  # noqa: E402
from sweep_wealth_bounds import CEILING_TOLERANCE, FLOOR_TOLERANCE  # noqa: E402
from synthetic_economy import BASE_CONFIG, SyntheticEconomy, pearson  # noqa: E402

from individuation import paired_t  # noqa: E402
from mob import (  # noqa: E402
    PERSISTENCE_DECOUPLED,
    PERSISTENCE_SHUFFLED,
    PERSISTENCE_VALUE,
    LightweightExpert,
)
from mob.auction import VCGAuctioneer  # noqa: E402
from mob.utils import frozen_economy  # noqa: E402
from parity import code_identity  # noqa: E402

ARMS = (PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED, PERSISTENCE_DECOUPLED)
SCALES = (1.0, 2.0, 4.0)
CELL_COUNTS = (4, 8)
SEEDS = tuple(range(6))
# Eight memory horizons, the settle every fixture read in this project uses,
# and the tail rule 8's occupancy columns are read over.
SETTLE_STEPS = 2667
TAIL = 100
# 128 batches of 2 x 16 tokens: the body read's probe budget (4096 tokens).
PROBE_BATCHES = 128
# The quarter of tokens the executed route is surest of -- lowest loss -- on
# which #58's guardrail is read.
CONFIDENT_QUANTILE = 0.25
# The two Gumbel seeds of the floor pair, per fixture seed.
FLOOR_PAIR = (1, 2)
FLOOR_QUANTILE = 0.95
RECORDED_RATE = BASE_CONFIG.reward_scale


@dataclass(frozen=True)
class Job:
    fixture: str
    arm: str
    seed: int
    cells: int
    scale: float
    reward_scale: float
    reward_scale_derived: bool
    alternatives: int = DEFAULT_ALTERNATIVES
    noise_scale: float = DEFAULT_NOISE_SCALE
    probe_batches: int = PROBE_BATCHES


@dataclass(frozen=True)
class Read:
    """One forward over the probe: each token's loss and the route that produced it."""

    losses: torch.Tensor
    routes: torch.Tensor

    @property
    def tokens(self) -> int:
        return int(self.losses.numel())


def rate_for(
    fixture: str, cells: int, scale: float, out: Path, derivation_steps: int
) -> tuple[float, bool, dict[str, Any] | None]:
    """The exchange rate a configuration runs at: the recorded constant at 1x, derived elsewhere.

    A derivation already on disk under ``out`` at this code is reused rather
    than re-run, so every seed of a configuration runs at one rate -- the
    issue's "derive once per configuration" -- and the record says which.
    """
    if scale == 1.0:
        return RECORDED_RATE, False, None
    path = derivation_path(out, fixture, cells, scale)
    sha, _ = code_identity()
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("code_sha") == sha and not record.get("code_dirty"):
            return float(record["derived"]), True, record
    derivation = derive_reward_scale(
        fixture, scale, seeds=DERIVATION_SEEDS, steps=derivation_steps, cells=cells
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = derivation.as_dict()
    path.write_text(json.dumps(record, indent=2, default=str))
    return derivation.derived, True, record


def settle(job: Job, steps: int = SETTLE_STEPS) -> tuple[SyntheticEconomy, dict[str, float]]:
    """The economy at its settled state, and rule 8's columns over its last ``TAIL`` steps."""
    config = replace(BASE_CONFIG, persistence_coupling=job.arm, reward_scale=job.reward_scale)
    economy = build_economy(job.fixture, job.seed, config, job.scale, job.cells)
    ceiling = economy.config.max_wealth * (1 - CEILING_TOLERANCE)
    floor = economy.config.min_wealth * (1 + FLOOR_TOLERANCE)
    at_ceiling = at_floor = 0
    tail = min(TAIL, steps)
    for step in range(steps):
        economy.step()
        if step >= steps - tail:
            wealth = economy.mob.expert_wealth
            at_ceiling += int((wealth >= ceiling).sum())
            at_floor += int((wealth <= floor).sum())
    cell_steps = tail * economy.config.num_experts
    return economy, {
        "guardrail/ceiling_occupancy": at_ceiling / cell_steps,
        "guardrail/floor_occupancy": at_floor / cell_steps,
        "guardrail/cell_steps": float(cell_steps),
        "guardrail/r_wealth_competence": pearson(economy.mob.expert_wealth, economy.competence),
    }


def draw_probe(economy: SyntheticEconomy, batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The probe: ``batches`` draws from the fixture's own generator, after the settle.

    Every arm of one seed has consumed the generator identically over the
    settle, so the probe is the same tokens on every arm -- the pairing the
    contrasts below rest on, asserted by the caller through the batch hash.
    """
    return [economy._draw() for _ in range(batches)]


@contextmanager
def routed_by(economy: SyntheticEconomy, gate: torch.nn.Module) -> Iterator[None]:
    original = economy.mob.gate
    economy.mob.gate = gate
    try:
        yield
    finally:
        economy.mob.gate = original


@contextmanager
def adapters_zeroed(economy: SyntheticEconomy) -> Iterator[None]:
    """Every cell's contribution removed: the fixture's adapter footprint."""
    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for module in economy.mob.experts:
                expert = cast(LightweightExpert, module)
                for name in ("gate_adapter_B", "up_adapter_B", "down_adapter_B"):
                    weight = getattr(expert, name).weight
                    saved.append((weight, weight.detach().clone()))
                    weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for weight, value in saved:
                weight.copy_(value)


def read(economy: SyntheticEconomy, probe: list[tuple[torch.Tensor, torch.Tensor]]) -> Read:
    """Forward the probe with the ledger frozen and nothing paid, in eval mode.

    Eval mode is what takes the exploration gift out of the executed route: the
    gate explores only while training, and the read is of the auction's own
    top-k. The routes come back sorted per token so that two routes that seat
    the same cells compare equal whatever order the top-k listed them in.
    """
    losses: list[torch.Tensor] = []
    routes: list[torch.Tensor] = []
    was_training = economy.mob.training
    economy.mob.eval()
    try:
        with torch.no_grad(), frozen_economy(economy.mob):
            for x, target in probe:
                output = economy.mob(x)
                losses.append(((output - target) ** 2).sum(-1).flatten())
                stats = economy.mob.last_stats
                assert stats is not None
                selected = stats.selected_experts.reshape(-1, economy.config.top_k)
                routes.append(selected.sort(dim=-1).values)
    finally:
        economy.mob.train(was_training)
    return Read(torch.cat(losses), torch.cat(routes))


def alternatives_read(
    economy: SyntheticEconomy,
    probe: list[tuple[torch.Tensor, torch.Tensor]],
    executed: Read,
    alternatives: int,
    noise_scale: float,
    gumbel_seed: int,
) -> dict[str, Any]:
    """Executed against the best of ``alternatives`` equal-compute routes, per token.

    The body read's ``counterfactual_gaps`` with the stream taken out: a token
    whose route no draw moved reads exactly the executed loss, so the unmoved
    population is the identity and only the moved one is counted.
    """
    gate = economy.mob.gate
    if not isinstance(gate, VCGAuctioneer):
        raise TypeError("a counterfactual route is defined against the auction's own bids")
    generator = torch.Generator().manual_seed(gumbel_seed)
    best = executed.losses.clone()
    better = torch.zeros_like(executed.losses)
    moved_total = moved_better = moved_swing = 0.0
    confident = executed.losses <= float(executed.losses.quantile(CONFIDENT_QUANTILE))
    confident_better = 0.0
    # The body's rate counted every draw-token pair, and on the body an
    # unmoved token was still beaten about half the time by the stream
    # upstream of it; on the fixture an unmoved token ties exactly, so the
    # same rate is diluted by the pairs no draw moved. The conditional pair
    # -- moved and confident -- is the one that compares with the body.
    confident_moved = confident_moved_better = 0.0
    for _ in range(alternatives):
        wrapper = GumbelTopKRoute(gate, generator, noise_scale, subset=None)
        with routed_by(economy, wrapper):
            alternative = read(economy, probe)
        if alternative.tokens != executed.tokens:
            raise ValueError("an alternative route read a different number of tokens")
        moved = (alternative.routes != executed.routes).any(dim=-1)
        # A token no draw moved reads the executed loss up to the order the
        # same cells' outputs were summed in, so "better" is only ever read on
        # a moved token: a rounding flip on an unmoved one is not a route.
        improved = (alternative.losses < executed.losses) & moved
        best = torch.minimum(best, alternative.losses)
        better += improved.float()
        moved_total += float(moved.sum())
        moved_better += float((improved & moved).sum())
        moved_swing += float((alternative.losses - executed.losses).abs()[moved].sum())
        confident_better += float((improved & confident).sum())
        confident_moved += float((moved & confident).sum())
        confident_moved_better += float((improved & moved & confident).sum())
    gap = executed.losses - best
    return {
        "gap": gap,
        "alternatives_better": better / alternatives,
        "moved_fraction": moved_total / (alternatives * executed.tokens),
        "moved_better": moved_better / moved_total if moved_total else math.nan,
        "moved_swing": moved_swing / moved_total if moved_total else math.nan,
        "confident_better": confident_better / (alternatives * float(confident.sum())),
        "confident_tokens": int(confident.sum()),
        "confident_moved_fraction": confident_moved / (alternatives * float(confident.sum())),
        "confident_moved_better": (
            confident_moved_better / confident_moved if confident_moved else math.nan
        ),
    }


def floor_of(gap_a: torch.Tensor, gap_b: torch.Tensor) -> dict[str, float]:
    """The read's own noise: the same seed's best-of-K gap under two Gumbel seeds."""
    difference = (gap_a - gap_b).abs()
    return {
        "floor": float(difference.quantile(FLOOR_QUANTILE)),
        "floor_max": float(difference.max()),
        "floor_mean": float(difference.mean()),
    }


def probe_hash(probe: list[tuple[torch.Tensor, torch.Tensor]]) -> str:
    """A hash of the probe's inputs and targets, in order, to show two arms of one seed share it."""
    digest = hashlib.sha256()
    for x, target in probe:
        digest.update(x.contiguous().numpy().tobytes())
        digest.update(target.contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def read_one(job: Job, settle_steps: int = SETTLE_STEPS) -> dict[str, Any]:
    """One reading: settle, freeze, the executed route, K alternatives twice, the footprint."""
    economy, guardrail = settle(job, settle_steps)
    probe = draw_probe(economy, job.probe_batches)
    executed = read(economy, probe)
    again = read(economy, probe)
    if not torch.equal(again.losses, executed.losses):
        raise AssertionError("the executed read is not reproducible; the economy moved under it")
    reads = [
        alternatives_read(
            economy, probe, executed, job.alternatives, job.noise_scale, job.seed * 100 + offset
        )
        for offset in FLOOR_PAIR
    ]
    first, second = reads
    floor = floor_of(first["gap"], second["gap"])
    gap = first["gap"]
    fragile = gap > floor["floor"]
    relative = gap / executed.losses.clamp_min(torch.finfo(torch.float32).tiny)
    with adapters_zeroed(economy):
        without = read(economy, probe)
    code_sha, code_dirty = code_identity()
    reading: dict[str, Any] = {
        **asdict(job),
        "probe_tokens": executed.tokens,
        "probe_hash": probe_hash(probe),
        "probe_loss": float(executed.losses.mean()),
        "counterfactual/mean_gap": float(gap.mean()),
        "counterfactual/mean_relative_gap": float(relative.mean()),
        "counterfactual/gap_p50": float(gap.median()),
        "counterfactual/gap_p95": float(gap.quantile(0.95)),
        "counterfactual/alternatives_better": float(first["alternatives_better"].mean()),
        "counterfactual/moved_fraction": first["moved_fraction"],
        "counterfactual/moved_better": first["moved_better"],
        "counterfactual/moved_swing": first["moved_swing"],
        "counterfactual/confident_better": first["confident_better"],
        "counterfactual/confident_better_pair": second["confident_better"],
        "counterfactual/confident_tokens": first["confident_tokens"],
        "counterfactual/confident_moved_fraction": first["confident_moved_fraction"],
        "counterfactual/confident_moved_better": first["confident_moved_better"],
        "counterfactual/confident_moved_better_pair": second["confident_moved_better"],
        **floor,
        "counterfactual/fragile_fraction": float(fragile.float().mean()),
        "counterfactual/gap_on_fragile": float(gap[fragile].mean()) if bool(fragile.any()) else 0.0,
        "counterfactual/relative_gap_on_fragile": (
            float(relative[fragile].mean()) if bool(fragile.any()) else 0.0
        ),
        "counterfactual/gap_on_fragile_above_floor": (
            float(gap[fragile].mean()) - floor["floor"] if bool(fragile.any()) else 0.0
        ),
        "footprint/loss_without_adapters": float(without.losses.mean()),
        "footprint/loss_with_adapters": float(executed.losses.mean()),
        "footprint/adapter_footprint": float((without.losses - executed.losses).mean()),
        "footprint/relative_adapter_footprint": float(
            ((without.losses - executed.losses) / without.losses.clamp_min(1e-12)).mean()
        ),
        **guardrail,
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "fingerprint": asdict(
            fixture_fingerprint(
                job.fixture,
                job.seed,
                job.arm,
                (),
                settle_steps,
                cells=job.cells,
                contribution_scale=job.scale,
                reward_scale=job.reward_scale,
                reward_scale_derived=job.reward_scale_derived,
            )
        ),
    }
    return reading


def _single_thread() -> None:
    torch.set_num_threads(1)


def run_jobs(jobs: list[Job], workers: int) -> list[dict[str, Any]]:
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_single_thread, mp_context=spawn
    ) as pool:
        return list(pool.map(read_one, jobs))


def reading_path(out: Path, reading: dict[str, Any]) -> Path:
    return (
        out
        / reading["fixture"]
        / (
            f"{reading['arm']}_cells{reading['cells']}_scale{reading['scale']:g}"
            f"_seed{reading['seed']}.json"
        )
    )


def stage_grid(args: argparse.Namespace) -> None:
    for fixture in args.fixtures:
        jobs: list[Job] = []
        for cells in args.cells:
            for scale in args.scales:
                try:
                    rate, derived, _ = rate_for(
                        fixture, cells, scale, args.out, args.derivation_steps
                    )
                except ClampedLedgerError as refused:
                    print(
                        f"{fixture} cells {cells} scale {scale:g}: derivation refused -- {refused}"
                    )
                    continue
                print(
                    f"{fixture} cells {cells} scale {scale:g}: reward_scale {rate:.5f}"
                    f"{' (derived)' if derived else ' (recorded)'}"
                )
                jobs += [
                    Job(fixture, arm, seed, cells, scale, rate, derived, args.alternatives)
                    for arm in ARMS
                    for seed in args.seeds
                ]
        readings = run_jobs(jobs, args.workers)
        for reading in readings:
            path = reading_path(args.out, reading)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reading, indent=2, default=str))
        print(f"{fixture}: {len(readings)} readings written under {args.out / fixture}")


def load_readings(
    out: Path, fixture: str
) -> dict[tuple[str, int, float], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[str, int, float], dict[int, dict[str, Any]]] = {}
    for path in sorted((out / fixture).glob("*_cells*_scale*_seed*.json")):
        reading = json.loads(path.read_text())
        key = (reading["arm"], int(reading["cells"]), float(reading["scale"]))
        grouped.setdefault(key, {})[int(reading["seed"])] = reading
    return grouped


COLUMNS = (
    "counterfactual/fragile_fraction",
    "counterfactual/relative_gap_on_fragile",
    "counterfactual/gap_on_fragile_above_floor",
    "counterfactual/mean_relative_gap",
    "counterfactual/confident_better",
    "counterfactual/confident_moved_fraction",
    "counterfactual/confident_moved_better",
    "counterfactual/moved_better",
    "counterfactual/moved_fraction",
    "floor",
    "footprint/relative_adapter_footprint",
    "footprint/adapter_footprint",
    "guardrail/ceiling_occupancy",
    "guardrail/floor_occupancy",
    "guardrail/r_wealth_competence",
)
# The contrast the issue names: the relative gap on fragile tokens at a scale
# against the same at 1x, paired by seed, on every arm and cell count.
PRIMARY = "counterfactual/relative_gap_on_fragile"


def pooled(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seeds": sorted(rows)}
    for column in COLUMNS:
        values = [row[column] for row in rows.values() if not math.isnan(row[column])]
        summary[column] = {
            "mean": statistics.fmean(values) if values else math.nan,
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    summary["probe_hash"] = sorted({row["probe_hash"] for row in rows.values()})
    summary["reward_scale"] = rows[min(rows)]["reward_scale"]
    summary["reward_scale_derived"] = rows[min(rows)]["reward_scale_derived"]
    return summary


def scale_contrast(
    grouped: dict[tuple[str, int, float], dict[int, dict[str, Any]]],
    arm: str,
    cells: int,
    scale: float,
    column: str,
) -> dict[str, Any] | None:
    """``column`` at ``scale`` minus the same at 1x, paired by seed, on one arm and cell count."""
    scaled, base = grouped.get((arm, cells, scale)), grouped.get((arm, cells, 1.0))
    if not scaled or not base:
        return None
    seeds = sorted(set(scaled) & set(base))
    for seed in seeds:
        if scaled[seed]["probe_hash"] != base[seed]["probe_hash"]:
            raise AssertionError(f"seed {seed}: the two scales did not read the same probe tokens")
    deltas = [scaled[seed][column] - base[seed][column] for seed in seeds]
    if len(deltas) < 2:
        return None
    test = paired_t(deltas)
    return {
        **test.as_dict(),
        "pairs_at_80": pairs_for_power(test.dz) if test.sd > 0 else None,
        "per_seed": dict(zip(map(str, seeds), deltas, strict=True)),
    }


def self_model_contrasts(path: Path) -> dict[tuple[int, float], float]:
    """#60's `value` - `shuffled` dz per grid cell at the derived rate (README #self-model)."""
    record = json.loads(path.read_text())
    contrasts: dict[tuple[int, float], float] = {}
    for cell in record["lever_one"].values():
        rates = cell["rates"]
        row = rates.get("derived") or rates.get("hand-set")
        if row is not None:
            contrasts[(int(cell["cells"]), float(cell["scale"]))] = float(row["dz"])
    return contrasts


def swing_contrast_correlation(
    grouped: dict[tuple[str, int, float], dict[int, dict[str, Any]]],
    contrasts: dict[tuple[int, float], float],
    resamples: int = 2000,
) -> dict[str, Any]:
    """Across the grid's cells, r between the `value` arm's fragile swing and #60's contrast.

    Six points at most, so the interval is what carries the number: seeds are
    resampled with replacement inside every cell and r recomputed, and the 2.5
    and 97.5 percentiles are reported beside the point estimate.
    """
    cells = sorted(key for key in contrasts if (PERSISTENCE_VALUE, key[0], key[1]) in grouped)
    if len(cells) < 3:
        return {"r": math.nan, "cells": len(cells), "interval": [math.nan, math.nan]}
    rows = [grouped[(PERSISTENCE_VALUE, c, s)] for c, s in cells]
    ys = torch.tensor([contrasts[key] for key in cells], dtype=torch.float64)

    def r_of(picks: list[list[int]]) -> float:
        xs = torch.tensor(
            [
                statistics.fmean(row[seed][PRIMARY] for seed in pick)
                for row, pick in zip(rows, picks, strict=True)
            ],
            dtype=torch.float64,
        )
        return pearson(xs, ys)

    point = r_of([sorted(row) for row in rows])
    generator = torch.Generator().manual_seed(0)
    draws = []
    for _ in range(resamples):
        picks = []
        for row in rows:
            seeds = sorted(row)
            index = torch.randint(0, len(seeds), (len(seeds),), generator=generator)
            picks.append([seeds[int(i)] for i in index])
        draws.append(r_of(picks))
    sample = torch.tensor([d for d in draws if not math.isnan(d)])
    interval = (
        [float(sample.quantile(0.025)), float(sample.quantile(0.975))]
        if sample.numel()
        else [math.nan, math.nan]
    )
    # The seed bootstrap holds the cells fixed, so its interval is the seed
    # noise *given* these five or six points and says nothing about how far
    # a correlation over that few points is from zero. The Fisher z interval
    # and a permutation p over the cells carry that half.
    n = len(cells)
    fisher = [math.nan, math.nan]
    if n > 3 and not math.isnan(point):
        z = math.atanh(max(-0.999999, min(0.999999, point)))
        half = 1.96 / math.sqrt(n - 3)
        fisher = [math.tanh(z - half), math.tanh(z + half)]
    xs = torch.tensor(
        [statistics.fmean(row[seed][PRIMARY] for seed in row) for row in rows], dtype=torch.float64
    )
    permuted = 0
    for _ in range(resamples):
        order = torch.randperm(n, generator=generator)
        if abs(pearson(xs, ys[order])) >= abs(point):
            permuted += 1
    return {
        "r": point,
        "cells": len(cells),
        "interval": interval,
        "interval_is": "seed bootstrap, cells fixed",
        "fisher_interval": fisher,
        "permutation_p": permuted / resamples,
        "points": {
            f"cells{c}-scale{s:g}": {
                "swing": statistics.fmean(row[seed][PRIMARY] for seed in row),
                "contrast_dz": contrasts[(c, s)],
            }
            for (c, s), row in zip(cells, rows, strict=True)
        },
    }


def _columns(cell: dict[str, Any]) -> str:
    """One pooled grid cell under the headings ``stage_summarise`` prints."""
    mean = {column: cell[column]["mean"] for column in COLUMNS}
    return (
        f"{mean['counterfactual/fragile_fraction']:>9.3f}"
        f"{mean['counterfactual/relative_gap_on_fragile']:>9.4f}"
        f"{mean['counterfactual/gap_on_fragile_above_floor']:>9.4f}"
        f"{mean['counterfactual/confident_better']:>8.3f}"
        f"{mean['counterfactual/confident_moved_fraction']:>8.3f}"
        f"{mean['counterfactual/confident_moved_better']:>8.3f}"
        f"{mean['counterfactual/moved_better']:>8.3f}"
        f"{100 * mean['footprint/relative_adapter_footprint']:>7.1f}%"
        f"{mean['guardrail/ceiling_occupancy']:>7.3f}{mean['guardrail/floor_occupancy']:>7.3f}"
        f"{mean['guardrail/r_wealth_competence']:>+7.3f}"
    )


def stage_summarise(args: argparse.Namespace) -> None:
    code_sha, code_dirty = code_identity()
    for fixture in args.fixtures:
        grouped = load_readings(args.out, fixture)
        if not grouped:
            print(f"{fixture}: no readings under {args.out / fixture}")
            continue
        summary: dict[str, Any] = {
            "fixture": fixture,
            "code_sha": code_sha,
            "code_dirty": code_dirty,
            "cells": {},
            "scale_contrasts": {},
            "correlation_with_self_model": None,
        }
        print(f"\n== {fixture} ==")
        print(
            f"  {'arm':>10}{'cells':>6}{'scale':>6}{'rate':>8}{'fragile':>9}{'rel gap':>9}"
            f"{'>floor':>9}{'conf<':>8}{'c.moved':>8}{'c.mv<':>8}{'moved<':>8}{'foot%':>8}"
            f"{'ceil':>7}{'floor':>7}{'r':>7}"
        )
        for (arm, cells, scale), rows in sorted(grouped.items()):
            cell = pooled(rows)
            summary["cells"][f"{arm}/cells{cells}/scale{scale:g}"] = cell
            print(
                f"  {arm:>10}{cells:>6}{scale:>6.1f}{cell['reward_scale']:>8.4f}" + _columns(cell)
            )
        for arm in ARMS:
            for cells in CELL_COUNTS:
                for scale in SCALES[1:]:
                    for column in (
                        PRIMARY,
                        "counterfactual/confident_better",
                        "counterfactual/confident_moved_better",
                        "counterfactual/confident_moved_fraction",
                    ):
                        contrast = scale_contrast(grouped, arm, cells, scale, column)
                        if contrast is not None:
                            summary["scale_contrasts"][
                                f"{arm}/cells{cells}/scale{scale:g}-vs-1/{column}"
                            ] = contrast
                            what = column.split("/")[-1]
                            print(
                                f"  {arm}/cells{cells}: {scale:g}x - 1x on {what}: "
                                f"mean {contrast['mean']:+.4f} dz {contrast['dz']:+.2f} "
                                f"p {contrast['p']:.4f} pairs at 80% {contrast['pairs_at_80']}"
                            )
        if args.self_model is not None and args.self_model.exists():
            correlation = swing_contrast_correlation(grouped, self_model_contrasts(args.self_model))
            summary["correlation_with_self_model"] = correlation
            low, high = correlation["interval"]
            f_low, f_high = correlation["fisher_interval"]
            against = "" if fixture == DIFFERENTIATED else " -- against the differentiated #60 grid"
            print(
                f"  r(swing on fragile tokens, #60 contrast dz) over {correlation['cells']} grid "
                f"cells: {correlation['r']:+.3f}; seed bootstrap [{low:+.3f}, {high:+.3f}], "
                f"Fisher z [{f_low:+.3f}, {f_high:+.3f}], "
                f"permutation p {correlation['permutation_p']:.3f}{against}"
            )
        (args.out / fixture / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))


def parse_seeds(text: str) -> tuple[int, ...]:
    if "-" in text:
        low, high = text.split("-")
        return tuple(range(int(low), int(high) + 1))
    return tuple(int(part) for part in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=("grid", "summarise"), required=True)
    parser.add_argument("--fixtures", nargs="+", choices=FIXTURES, default=[QUALITY])
    parser.add_argument("--cells", nargs="+", type=int, default=list(CELL_COUNTS))
    parser.add_argument("--scales", nargs="+", type=float, default=list(SCALES))
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS, help="`0-5` or `0,1,2`")
    parser.add_argument("--alternatives", type=int, default=DEFAULT_ALTERNATIVES)
    parser.add_argument("--derivation-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--self-model",
        type=Path,
        default=None,
        help="#60's grid record at the derived rate, for the swing-to-contrast correlation",
    )
    parser.add_argument("--out", type=Path, default=Path.home() / "tame-runs" / "loudness-routing")
    return parser


def hide_the_card() -> None:
    """A CPU read on a box with a GPU never touches the card; set at run time, never at import."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> None:
    hide_the_card()
    args = build_parser().parse_args()
    {"grid": stage_grid, "summarise": stage_summarise}[args.stage](args)


if __name__ == "__main__":
    main()
