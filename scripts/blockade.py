"""Individuation by blockade on the planted fixtures (#63): means-substitution inside a window.

The entity criterion (Operative Corollary, Op 2) is not persistence but
*means-substitution under blockade*: block the means an entity uses toward its
end and it recruits new ones toward the same end, inside a window shorter than
adaptation; a part does not. ``scripts/economy_damage.py`` measured persistence
-- does the market re-form after damage -- and this measures the other thing:
when the dominant cell at the layer is blocked, do the remaining cells take up
its work *inside the window*, through the auction, toward the same restored
state under two structurally different blockades?

Three blockades, all on the fixture's own hooks: **output** (the cell's
contribution zeroed, bids and ledger untouched -- the economy has to find out),
**ledger** (its wealth pinned to the floor, output intact -- the cell is still
worth what it was, and only the record of it is gone) and **gate** (the cell's
bid on its own type's tokens reaches the auction as zero, head, ledger and
output intact -- the third, written into the prereg file as a prediction after
the first two were read, Op 2 step 5). The **window**
``W`` is a quarter of the time the on-type loss takes to re-converge under the
output blockade held indefinitely, measured first on one seed (``--stage
window``) and written into ``docs/prereg/63-individuation.md`` before any arm
runs. Every reading pairs against an unblocked branch of the same seed: the
fixture is bitwise deterministic, so the pre-block trajectory is identical and
the control is the same tokens with nothing blocked.

Section 8 rule 7 binds the order: the planted effect (``--stage planted``, the
quality fixture, where the substitute is predicted to be the next most competent
cell) and the null calibration (``--stage null``) come before any arm contrast
(``--stage arms``, then ``--stage summarise``). Rule 8 puts ceiling and floor
occupancy and ``r(wealth, competence)`` beside every reading.

**Stage 5, the redundancy fixture** (``--fixtures redundancy-fixture``): the
quality fixture with a second cell at the top competence, which the seniority
the live ledger runs on (#62) seats beside the 0.7 on no seed -- so a cell that
can do the blocked cell's work sits at the floor, and the question the recorded
fixtures cannot ask is whether the auction hands the freed slot to it. The
criterion the operator fixed for it is the *born-without target*: the on-type
loss a collective born without the blocked cell settles at, which ``--stage
targets`` measures per arm and seed and ``--stage summarise`` reads every
blocked window against.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from economy_damage import RE_FORMATION_FACTOR  # noqa: E402
from measure_stakes_dial import fixture_fingerprint  # noqa: E402
from power import null_calibration, pairs_for_power  # noqa: E402
from sweep_wealth_bounds import CEILING_TOLERANCE, FLOOR_TOLERANCE  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    REDUNDANT_COMPETENCE,
    DifferentiatedEconomy,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from individuation import (  # noqa: E402
    dominant_cell,
    gains,
    half_life,
    paired_t,
    planted_statistic,
    predicted_substitute,
    reconvergence_step,
    returned,
    type_shares,
    uptake,
    winners,
)
from mob import PERSISTENCE_DECOUPLED, PERSISTENCE_SHUFFLED, PERSISTENCE_VALUE  # noqa: E402
from parity import code_identity  # noqa: E402

QUALITY = "quality-fixture"
DIFFERENTIATED = "differentiated-fixture"
REDUNDANCY = "redundancy-fixture"
FIXTURES = (QUALITY, DIFFERENTIATED, REDUNDANCY)
# The two fixtures the 1x stages were recorded on, which is what a stage runs
# on when none is named; stage 5's fixture is named explicitly.
RECORDED_FIXTURES = (QUALITY, DIFFERENTIATED)
# Where a planted substitute is predicted: one type, so the best cell not
# already winning is defined; the differentiated fixture has two cells a type
# and nothing to predict.
PLANTED_FIXTURES = (QUALITY, REDUNDANCY)
ARMS = (PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED, PERSISTENCE_DECOUPLED)
NONE, OUTPUT, LEDGER, GATE = "none", "output", "ledger", "gate"
BLOCKADES = (NONE, OUTPUT, LEDGER, GATE)
READ_BLOCKADES = (OUTPUT, LEDGER, GATE)
# Eight wealth memory horizons at the decay, #16's convention for a settled
# ledger; the issue's budget.
SETTLE_STEPS = 2667
# The pre-block read: the settled tail every fixture reading in the project is
# taken over, and the window the occupancy guardrail counts cell-steps in.
TAIL = 100
# How long after release the return is watched: one re-convergence time, since
# W is a quarter of one. The return is read over the last W of it.
RELEASE_MULTIPLE = 4
# A blocked-minus-control return within this of zero is a cell that came back.
RETURN_TOLERANCE = 0.05
# The trailing mean the re-convergence is read on, and the cap past which the
# blocked loss is called not re-converging and W is taken from the cap.
TRAILING = 50
RECONVERGENCE_CAP = SETTLE_STEPS
EXPLORATORY_SEEDS = tuple(range(6))
CONFIRMATORY_SEEDS = tuple(range(24))
NULL_SEEDS = tuple(range(12))


def build(
    fixture: str, arm: str, seed: int, scale: float, without: int | None = None
) -> SyntheticEconomy:
    """The fixture at ``seed``; with ``without``, the same collective born without that cell.

    Born without means competence zero at construction: the plant's random
    draws do not depend on the competence values, so the planted directions
    are the seed's own and only the one cell's correction is gone -- the target
    ``economy_damage.floor_without`` reads re-formation against.
    """
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    planted = REDUNDANT_COMPETENCE if fixture == REDUNDANCY else DEFAULT_COMPETENCE
    competence = shuffled(planted, seed)
    if without is not None:
        competence = competence.clone()
        competence[without] = 0.0
    if fixture in PLANTED_FIXTURES:
        if scale != 1.0:
            raise ValueError("contribution_scale is the differentiated fixture's knob")
        return SyntheticEconomy(competence, seed=seed, config=config)
    if fixture == DIFFERENTIATED:
        return DifferentiatedEconomy(competence, seed=seed, config=config, contribution_scale=scale)
    raise ValueError(f"unknown fixture {fixture!r}")


def expert_types(economy: SyntheticEconomy) -> torch.Tensor:
    """Each cell's type; the quality fixture has one type and every token is of it."""
    if isinstance(economy, DifferentiatedEconomy):
        return economy.expert_types
    return torch.zeros(economy.config.num_experts, dtype=torch.long)


def token_types(economy: SyntheticEconomy, shape: torch.Size) -> torch.Tensor:
    if isinstance(economy, DifferentiatedEconomy):
        assert economy.last_types is not None
        return economy.last_types
    return torch.zeros(shape, dtype=torch.long)


@dataclass
class Window:
    """What one stretch of steps looked like, per type and per cell.

    ``wins_by_type[t, i]`` is the number of slots cell ``i`` held on tokens of
    type ``t``; ``loss_by_type`` the per-step mean loss on each type's tokens;
    ``blocked_share`` the per-step share of the watched cell on its type's
    tokens, kept only when a cell is being watched.
    """

    wins_by_type: torch.Tensor
    loss_by_type: list[list[float]]
    blocked_share: list[float]
    at_ceiling: int = 0
    at_floor: int = 0

    def on_type_loss(self, cell_type: int) -> float:
        return statistics.fmean(self.loss_by_type[cell_type])

    def shares(self, cell_type: int) -> torch.Tensor:
        return type_shares(self.wins_by_type[cell_type])

    def own_type_wins(self, types: torch.Tensor) -> torch.Tensor:
        """Slots each cell held on tokens of its own type."""
        return self.wins_by_type[types, torch.arange(types.numel())]


def observe(economy: SyntheticEconomy, steps: int, watch: tuple[int, int] | None = None) -> Window:
    """Run ``steps`` steps and count; consumes no randomness of its own."""
    num_types = getattr(economy, "num_types", 1)
    cells = economy.config.num_experts
    ceiling = economy.config.max_wealth * (1 - CEILING_TOLERANCE)
    floor = economy.config.min_wealth * (1 + FLOOR_TOLERANCE)
    window = Window(torch.zeros(num_types, cells), [[] for _ in range(num_types)], [])
    for _ in range(steps):
        record = economy.step()
        assert record.per_token_loss is not None
        selected = record.selected_experts
        types = token_types(economy, record.per_token_loss.shape)
        for cell_type in range(num_types):
            mask = types == cell_type
            if not mask.any():
                continue
            window.wins_by_type[cell_type] += torch.bincount(
                selected[mask].flatten(), minlength=cells
            ).float()
            window.loss_by_type[cell_type].append(float(record.per_token_loss[mask].mean()))
        if watch is not None:
            cell, cell_type = watch
            held = selected[types == cell_type]
            # A step with no token of the type is skipped rather than recorded
            # as NaN, which would poison every running mean read over it.
            if held.numel():
                window.blocked_share.append(float((held == cell).float().mean()))
        wealth = economy.mob.expert_wealth
        window.at_ceiling += int((wealth >= ceiling).sum())
        window.at_floor += int((wealth <= floor).sum())
    return window


def settle(fixture: str, arm: str, seed: int, scale: float) -> tuple[SyntheticEconomy, Window]:
    """An economy at its settled state, and its last ``TAIL`` steps."""
    economy = build(fixture, arm, seed, scale)
    observe(economy, SETTLE_STEPS - TAIL)
    return economy, observe(economy, TAIL)


def apply_blockade(economy: SyntheticEconomy, blockade: str, cell: int) -> None:
    if blockade == OUTPUT:
        economy.block_output(cell)
    elif blockade == LEDGER:
        economy.pin_wealth(cell)
    elif blockade == GATE:
        # The class is the cell's own type; the quality fixture has one type, so
        # there the class is every token.
        own_type = (
            int(expert_types(economy)[cell]) if isinstance(economy, DifferentiatedEconomy) else None
        )
        economy.block_bids(cell, own_type)
    elif blockade != NONE:
        raise ValueError(f"unknown blockade {blockade!r}")


def release_blockade(economy: SyntheticEconomy, blockade: str, cell: int) -> None:
    if blockade == OUTPUT:
        economy.release_output(cell)
    elif blockade == LEDGER:
        economy.release_wealth(cell)
    elif blockade == GATE:
        economy.release_bids(cell)


@dataclass(frozen=True)
class Job:
    fixture: str
    arm: str
    seed: int
    blockade: str
    window: int
    scale: float = 1.0


def read(job: Job) -> dict[str, Any]:
    """One reading: settle, block the dominant cell, watch W, release, watch the return."""
    economy, pre = settle(job.fixture, job.arm, job.seed, job.scale)
    types = expert_types(economy)
    competence = economy.competence
    blocked = dominant_cell(pre.own_type_wins(types), competence)
    cell_type = int(types[blocked])
    on_type = types == cell_type
    top_k = economy.config.top_k
    pre_shares = pre.shares(cell_type)
    wealth_at_settle = economy.mob.expert_wealth.detach().clone()

    apply_blockade(economy, job.blockade, blocked)
    inside = observe(economy, job.window, watch=(blocked, cell_type))
    release_blockade(economy, job.blockade, blocked)
    observe(economy, (RELEASE_MULTIPLE - 1) * job.window)
    post = observe(economy, job.window)

    inside_shares = inside.shares(cell_type)
    cell_gains = gains(pre_shares, inside_shares)
    predicted = predicted_substitute(competence, on_type, pre_shares, blocked, top_k)
    others = cell_gains.clone()
    others[blocked] = -torch.inf
    by_wealth = next_by_wealth(wealth_at_settle, pre_shares, blocked, top_k)
    code_sha, code_dirty = code_identity()
    reading: dict[str, Any] = {
        **asdict(job),
        "blocked_cell": blocked,
        "blocked_type": cell_type,
        "blocked_competence": float(competence[blocked]),
        "pre_shares": pre_shares.tolist(),
        "inside_shares": inside_shares.tolist(),
        "post_shares": post.shares(cell_type).tolist(),
        "uptake": uptake(pre_shares, inside_shares, blocked),
        "returned": returned(pre_shares, post.shares(cell_type), blocked),
        "pre_on_type_loss": pre.on_type_loss(cell_type),
        "inside_on_type_loss": inside.on_type_loss(cell_type),
        "post_on_type_loss": post.on_type_loss(cell_type),
        "largest_gainer": int(others.argmax()),
        "predicted_substitute": predicted,
        # Stage 5's other candidate: the wealthiest cell outside the pre-block
        # winner set, which is where the setpoint sentence says the slot goes.
        "next_by_wealth": by_wealth,
        "wealth_hit": int(others.argmax()) == by_wealth,
        "wealth_at_settle": wealth_at_settle.tolist(),
        "half_life": half_life(
            inside.blocked_share, float(pre_shares[blocked]), float(inside_shares[blocked])
        ),
        "guardrail/ceiling_occupancy": pre.at_ceiling / (TAIL * competence.numel()),
        "guardrail/floor_occupancy": pre.at_floor / (TAIL * competence.numel()),
        "guardrail/cell_steps": TAIL * competence.numel(),
        "guardrail/r_wealth_competence": pearson(wealth_at_settle, competence),
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "fingerprint": asdict(
            fixture_fingerprint(
                job.fixture, job.seed, job.arm, (), SETTLE_STEPS, contribution_scale=job.scale
            )
        ),
    }
    if predicted is not None:
        reading["planted_statistic"] = planted_statistic(
            cell_gains, on_type, pre_shares, blocked, predicted, top_k
        )
        reading["predicted_hit"] = int(others.argmax()) == predicted
    return reading


def next_by_wealth(wealth: torch.Tensor, pre_shares: torch.Tensor, blocked: int, top_k: int) -> int:
    """The wealthiest cell not already holding a slot, the blocked cell aside."""
    excluded = set(winners(pre_shares, top_k)) | {blocked}
    candidates = [index for index in range(wealth.numel()) if index not in excluded]
    return max(candidates, key=lambda index: float(wealth[index]))


def measure_window(fixture: str, seed: int, scale: float) -> dict[str, Any]:
    """T, the re-convergence time of the on-type loss under the output blockade, and W = T / 4.

    The target is what a collective born without the cell settles at
    (``economy_damage.floor_without``'s construction, the cell's competence
    zeroed), within ``RE_FORMATION_FACTOR``, on a trailing mean of ``TRAILING``
    steps. A loss that has not re-converged by the cap is reported as capped
    and W is taken from the cap: that is the honest window, not a failure.
    """
    economy, pre = settle(fixture, PERSISTENCE_VALUE, seed, scale)
    types = expert_types(economy)
    blocked = dominant_cell(pre.own_type_wins(types), economy.competence)
    cell_type = int(types[blocked])

    # The exploration draw reads the global stream, so the born-without run is
    # taken to its settled state *before* the blocked run continues: a second
    # economy constructed midway reseeds that stream, and the blocked run would
    # then draw differently from the arm readings that block the same cell.
    born_without = build(fixture, PERSISTENCE_VALUE, seed, scale, without=blocked)
    observe(born_without, SETTLE_STEPS - TAIL)
    target = observe(born_without, TAIL).on_type_loss(cell_type)

    economy, _ = settle(fixture, PERSISTENCE_VALUE, seed, scale)
    economy.block_output(blocked)
    blocked_run = observe(economy, RECONVERGENCE_CAP, watch=(blocked, cell_type))
    losses = blocked_run.loss_by_type[cell_type]
    step = reconvergence_step(losses, target, RE_FORMATION_FACTOR, TRAILING)
    reconvergence = step if step is not None else RECONVERGENCE_CAP
    return {
        "fixture": fixture,
        "seed": seed,
        "scale": scale,
        "blocked_cell": blocked,
        "blocked_type": cell_type,
        "pre_on_type_loss": pre.on_type_loss(cell_type),
        "born_without_on_type_loss": target,
        "reconvergence_factor": RE_FORMATION_FACTOR,
        "trailing": TRAILING,
        "reconvergence_step": step,
        "capped": step is None,
        "cap": RECONVERGENCE_CAP,
        "window": max(1, -(-reconvergence // 4)),
        "loss_every_50": [statistics.fmean(losses[i : i + 50]) for i in range(0, len(losses), 50)],
        "blocked_share_every_50": [
            statistics.fmean(blocked_run.blocked_share[i : i + 50])
            for i in range(0, len(losses), 50)
        ],
        **identity(),
    }


@dataclass(frozen=True)
class TargetJob:
    fixture: str
    arm: str
    seed: int
    scale: float = 1.0


def measure_target(job: TargetJob) -> dict[str, Any]:
    """The born-without target for one arm and seed: what the remaining means allow.

    The dominant cell is read off the arm's own settled state -- under a
    pinned ledger the winner set can differ from the live arm's -- and the
    collective born without it is the same seed with that cell's competence
    zero (``build``'s ``without``), settled the same way. Its on-type loss is
    the state the operator fixed as "the same restored state" (#63, 2026-09-24):
    substitution counts when the loss inside W reaches what the means allow.
    """
    economy, pre = settle(job.fixture, job.arm, job.seed, job.scale)
    types = expert_types(economy)
    blocked = dominant_cell(pre.own_type_wins(types), economy.competence)
    cell_type = int(types[blocked])
    born_without = build(job.fixture, job.arm, job.seed, job.scale, without=blocked)
    observe(born_without, SETTLE_STEPS - TAIL)
    tail = observe(born_without, TAIL)
    return {
        **asdict(job),
        "blocked_cell": blocked,
        "blocked_type": cell_type,
        "pre_on_type_loss": pre.on_type_loss(cell_type),
        "born_without_on_type_loss": tail.on_type_loss(cell_type),
        "born_without_shares": tail.shares(cell_type).tolist(),
        **identity(),
    }


def _single_thread() -> None:
    torch.set_num_threads(1)


def run_jobs(jobs: list[Job] | list[TargetJob], workers: int) -> list[dict[str, Any]]:
    # Spawned, never forked: a forked child of a CUDA-initialised parent dies
    # in Adam's stream check, and the parent hides the card (``hide_the_card``)
    # so a body run on the same box is not disturbed by a fixture that has no
    # use for it.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_single_thread, mp_context=spawn
    ) as pool:
        if jobs and isinstance(jobs[0], TargetJob):
            return list(pool.map(measure_target, jobs))
        return list(pool.map(read, jobs))


def group(readings: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    """Readings keyed by (fixture, arm, blockade), then by seed."""
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for reading in readings:
        key = (reading["fixture"], reading["arm"], reading["blockade"])
        grouped.setdefault(key, {})[str(reading["seed"])] = reading
    return grouped


def paired_against_control(
    blocked: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]], field: str
) -> dict[str, float]:
    """``field`` on the blocked branch minus the same field on the unblocked branch, per seed.

    The two branches share every pre-block step bitwise, which is asserted here
    rather than assumed: a control whose pre-block shares differ from the
    treatment's was not paired with it.
    """
    deltas: dict[str, float] = {}
    for seed, reading in blocked.items():
        twin = control[seed]
        if (
            reading["pre_shares"] != twin["pre_shares"]
            or reading["blocked_cell"] != twin["blocked_cell"]
        ):
            raise AssertionError(
                f"seed {seed}: the control's pre-block state is not the treatment's"
            )
        deltas[seed] = reading[field] - twin[field]
    return deltas


def identity() -> dict[str, Any]:
    """The code every record was read at; a dirty tree is recorded as one."""
    code_sha, code_dirty = code_identity()
    return {"code_sha": code_sha, "code_dirty": code_dirty}


def write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))


def stage_window(args: argparse.Namespace) -> None:
    for fixture in args.fixtures:
        record = measure_window(fixture, args.seeds[0], args.scale)
        write_json(args.out / f"window_{fixture}.json", record)
        print(
            f"{fixture}: blocked cell {record['blocked_cell']} (type {record['blocked_type']}); "
            f"on-type loss {record['pre_on_type_loss']:.4f} -> "
            f"target {record['born_without_on_type_loss']:.4f}; "
            f"re-converged at {record['reconvergence_step']} (capped: {record['capped']}); "
            f"W = {record['window']}"
        )


def load_window(out: Path, fixture: str) -> int:
    return int(json.loads((out / f"window_{fixture}.json").read_text())["window"])


def stage_planted(args: argparse.Namespace) -> None:
    """Rule 7's first half: the predicted substitute takes the freed share, six seeds, paired t.

    One record per planted fixture, ``planted_<fixture>.json``; the quality
    fixture's 1x record was written before the name carried the fixture and
    stays at ``planted.json`` in ``~/tame-runs/individuation/``.
    """
    for fixture in args.fixtures:
        if fixture not in PLANTED_FIXTURES:
            print(f"{fixture}: no planted substitute to predict; skipped")
            continue
        _planted(args, fixture)


def _planted(args: argparse.Namespace, fixture: str) -> None:
    window = load_window(args.out, fixture)
    jobs = [
        Job(fixture, PERSISTENCE_VALUE, seed, b, window) for seed in args.seeds for b in BLOCKADES
    ]
    readings = run_jobs(jobs, args.workers)
    grouped = group(readings)
    record: dict[str, Any] = {
        "fixture": fixture,
        "window": window,
        "seeds": list(args.seeds),
        "blockades": {},
        **identity(),
    }
    for blockade in READ_BLOCKADES:
        rows = grouped[(fixture, PERSISTENCE_VALUE, blockade)]
        statistic = {seed: row["planted_statistic"] for seed, row in rows.items()}
        test = paired_t(list(statistic.values()))
        hits = sum(int(row["predicted_hit"]) for row in rows.values())
        wealth_hits = sum(int(row["wealth_hit"]) for row in rows.values())
        record["blockades"][blockade] = {
            "planted_statistic": statistic,
            "paired_t": test.as_dict(),
            "predicted_hits": hits,
            "next_by_wealth_hits": wealth_hits,
            "uptake": paired_against_control(
                rows, grouped[(fixture, PERSISTENCE_VALUE, NONE)], "uptake"
            ),
            "recovered": test.p < 0.05 and test.mean > 0,
            "readings": rows,
        }
        print(
            f"planted, {fixture}, {blockade}: statistic mean {test.mean:+.4f} dz {test.dz:+.3f} "
            f"t {test.t:+.2f} "
            f"p {test.p:.4f}; predicted substitute the largest gainer in {hits}/{len(rows)} "
            f"seeds, the next cell by wealth in {wealth_hits}/{len(rows)}"
        )
    write_json(args.out / f"planted_{fixture}.json", record)


def stage_targets(args: argparse.Namespace) -> None:
    """The born-without target per arm and seed, read by ``summarise`` against every window."""
    for fixture in args.fixtures:
        jobs = [TargetJob(fixture, arm, seed, args.scale) for arm in ARMS for seed in args.seeds]
        readings = run_jobs(jobs, args.workers)
        record: dict[str, Any] = {
            "fixture": fixture,
            "seeds": list(args.seeds),
            "targets": {arm: {} for arm in ARMS},
            **identity(),
        }
        for reading in readings:
            record["targets"][reading["arm"]][str(reading["seed"])] = reading
        write_json(args.out / f"targets_{fixture}.json", record)
        for arm in ARMS:
            rows = record["targets"][arm]
            gap = [
                row["born_without_on_type_loss"] - row["pre_on_type_loss"] for row in rows.values()
            ]
            print(
                f"targets, {fixture}, {arm}: born-without minus pre-block on-type loss "
                f"{statistics.fmean(gap):+.4f} (min {min(gap):+.4f}, max {max(gap):+.4f}) "
                f"over {len(rows)} seeds"
            )


def load_targets(out: Path, fixture: str) -> dict[str, dict[str, dict[str, Any]]] | None:
    path = out / f"targets_{fixture}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["targets"]


def stage_null(args: argparse.Namespace) -> None:
    """Rule 7's second half: the false-positive rate of each readout on split same-arm runs."""
    for fixture in args.fixtures:
        window = load_window(args.out, fixture)
        jobs = [
            Job(fixture, PERSISTENCE_VALUE, s, b, window, args.scale)
            for s in args.seeds
            for b in (NONE, OUTPUT)
        ]
        grouped = group(run_jobs(jobs, args.workers))
        blocked, control = (
            grouped[(fixture, PERSISTENCE_VALUE, OUTPUT)],
            grouped[(fixture, PERSISTENCE_VALUE, NONE)],
        )
        record: dict[str, Any] = {
            "fixture": fixture,
            "window": window,
            "seeds": list(args.seeds),
            **identity(),
        }
        for field in ("uptake", "inside_on_type_loss", "returned"):
            values = paired_against_control(blocked, control, field)
            calibration = null_calibration(
                list(values.values()), len(values) // 2, args.splits, 0.05, 0, args.resamples
            )
            record[field] = {"per_seed": values, "calibration": calibration}
            print(
                f"null, {fixture}, {field}: {len(values)} readings split {len(values) // 2} vs "
                f"{len(values) // 2}; bootstrap calls "
                f"{calibration['bootstrap_false_positive_rate']:.3f}, "
                f"paired t {calibration['paired_t_false_positive_rate']:.3f}"
            )
        write_json(args.out / f"null_{fixture}.json", record)


def stage_arms(args: argparse.Namespace) -> None:
    for fixture in args.fixtures:
        window = load_window(args.out, fixture)
        jobs = [
            Job(fixture, arm, seed, blockade, window, args.scale)
            for arm in ARMS
            for seed in args.seeds
            for blockade in args.blockades
        ]
        readings = run_jobs(jobs, args.workers)
        stage = args.out / f"arms_{fixture}_{len(args.seeds)}seeds"
        for (_, arm, blockade), rows in group(readings).items():
            write_json(
                stage / f"{arm}_{blockade}.json",
                {"contrast": {seed: row["uptake"] for seed, row in rows.items()}, "readings": rows},
            )
        print(f"{fixture}: {len(readings)} readings written under {stage}")


def load_stage(stage: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    loaded: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    # ``<arm>_<blockade>.json`` only: the stage's own SUMMARY.json sits beside them.
    for path in sorted(stage.glob("*_*.json")):
        arm, blockade = path.stem.rsplit("_", 1)
        loaded[(arm, blockade)] = json.loads(path.read_text())["readings"]
    return loaded


def stage_fixture(stage: Path) -> str:
    """``arms_<fixture>_<n>seeds`` names the fixture its readings came from."""
    return stage.name[len("arms_") :].rsplit("_", 1)[0]


def against_target(
    rows: dict[str, dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The loss inside W against what the means allow, per seed: the operator's criterion.

    A window whose on-type loss is within ``RE_FORMATION_FACTOR`` of the
    born-without target has substituted to what the remaining means allow --
    the same factor the window stage reads re-convergence with. The target is
    the arm's own, since a pinned ledger can seat a different winner set, and a
    row whose blocked cell is not the target's is refused rather than compared.
    """
    gaps: dict[str, float] = {}
    reached = 0
    for seed, row in rows.items():
        target = targets[seed]
        if target["blocked_cell"] != row["blocked_cell"]:
            raise AssertionError(
                f"seed {seed}: the target was read without cell {target['blocked_cell']} and "
                f"the window blocked cell {row['blocked_cell']}"
            )
        gaps[seed] = row["inside_on_type_loss"] - target["born_without_on_type_loss"]
        reached += (
            row["inside_on_type_loss"] <= RE_FORMATION_FACTOR * target["born_without_on_type_loss"]
        )
    return {
        "gap_to_target": gaps,
        "mean_gap": statistics.fmean(gaps.values()),
        "reached_target": reached,
        "factor": RE_FORMATION_FACTOR,
    }


def summarise_stage(
    stage: Path, targets: dict[str, dict[str, dict[str, Any]]] | None = None
) -> dict[str, Any]:
    """Every contrast in one stage, against the control and between arms, with dz and its seeds."""
    loaded = load_stage(stage)
    summary: dict[str, Any] = {
        "stage": str(stage),
        "against_control": {},
        "between_arms": {},
        "targets_read": targets is not None,
        **identity(),
    }
    against: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for blockade in READ_BLOCKADES:
        if not all((arm, blockade) in loaded for arm in ARMS):
            continue
        for arm in ARMS:
            rows, control = loaded[(arm, blockade)], loaded[(arm, NONE)]
            for field in ("uptake", "inside_on_type_loss", "returned"):
                deltas = paired_against_control(rows, control, field)
                test = paired_t(list(deltas.values()))
                against.setdefault(blockade, {}).setdefault(field, {})[arm] = deltas
                row: dict[str, Any] = {
                    **test.as_dict(),
                    "pairs_at_80": pairs_for_power(test.dz) if test.sd > 0 else None,
                    "per_seed": deltas,
                    "guardrail": guardrail_columns(rows),
                }
                if field == "returned":
                    # How many seeds came back within RETURN_TOLERANCE of the
                    # control: the "returned in full" count the README quotes.
                    row["within_tolerance"] = sum(
                        abs(d) <= RETURN_TOLERANCE for d in deltas.values()
                    )
                if field == "inside_on_type_loss" and targets is not None:
                    row["against_target"] = against_target(rows, targets[arm])
                if field == "uptake":
                    # Who took the freed slot: the cell competence predicts, or
                    # the next cell by wealth. Counted over the seeds of one arm.
                    row["substitute"] = {
                        "predicted_hits": sum(
                            int(r.get("predicted_hit", False)) for r in rows.values()
                        ),
                        "next_by_wealth_hits": sum(int(r["wealth_hit"]) for r in rows.values()),
                    }
                summary["against_control"][f"{blockade}/{arm}/{field}"] = row
        for field in ("uptake", "inside_on_type_loss"):
            for other in (PERSISTENCE_SHUFFLED, PERSISTENCE_DECOUPLED):
                value, control_arm = (
                    against[blockade][field][PERSISTENCE_VALUE],
                    against[blockade][field][other],
                )
                deltas = {seed: value[seed] - control_arm[seed] for seed in value}
                test = paired_t(list(deltas.values()))
                summary["between_arms"][f"{blockade}/{field}/value-{other}"] = {
                    **test.as_dict(),
                    "pairs_at_80": pairs_for_power(test.dz) if test.sd > 0 else None,
                    "per_seed": deltas,
                }
    return summary


def guardrail_columns(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Rule 8's three columns, pooled over the seeds of one arm."""
    return {
        key: statistics.fmean(row[key] for row in rows.values())
        for key in (
            "guardrail/ceiling_occupancy",
            "guardrail/floor_occupancy",
            "guardrail/r_wealth_competence",
        )
    }


def stage_summarise(args: argparse.Namespace) -> None:
    for stage in sorted(path for path in args.out.glob("arms_*") if path.is_dir()):
        fixture = stage_fixture(stage)
        if fixture not in args.fixtures:
            continue
        summary = summarise_stage(stage, load_targets(args.out, fixture))
        write_json(stage / "SUMMARY.json", summary)
        print(f"\n== {stage.name} (targets {'read' if summary['targets_read'] else 'absent'}) ==")
        for name, row in summary["against_control"].items():
            target = row.get("against_target")
            reached = ""
            if target:
                reached = (
                    f"  to target {target['mean_gap']:+.4f}, "
                    f"reached {target['reached_target']}/{row['n']:.0f}"
                )
            print(
                f"  {name:<40} mean {row['mean']:+.4f}  dz {row['dz']:+.3f}  p {row['p']:.4f}"
                f"  ceiling {row['guardrail']['guardrail/ceiling_occupancy']:.3f}"
                f"  floor {row['guardrail']['guardrail/floor_occupancy']:.3f}"
                f"  r {row['guardrail']['guardrail/r_wealth_competence']:+.3f}{reached}"
            )
        for name, row in summary["between_arms"].items():
            print(
                f"  {name:<40} mean {row['mean']:+.4f}  dz {row['dz']:+.3f}  p {row['p']:.4f}"
                f"  pairs at 80% {row['pairs_at_80']}"
            )


def parse_seeds(text: str) -> tuple[int, ...]:
    if "-" in text:
        low, high = text.split("-")
        return tuple(range(int(low), int(high) + 1))
    return tuple(int(part) for part in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=("window", "planted", "null", "targets", "arms", "summarise"),
        required=True,
    )
    parser.add_argument("--fixtures", nargs="+", choices=FIXTURES, default=list(RECORDED_FIXTURES))
    parser.add_argument("--seeds", type=parse_seeds, default=None, help="`0-5` or `0,1,2`")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="contribution_scale; 1.0 is the recorded fixture"
    )
    parser.add_argument(
        "--blockades",
        nargs="+",
        choices=BLOCKADES,
        default=list(BLOCKADES),
        help="the arm stage's conditions; the control is always run",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", type=int, default=2000)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path.home() / "tame-runs" / "individuation")
    return parser


DEFAULT_SEEDS = {
    "window": (0,),
    "planted": EXPLORATORY_SEEDS,
    "null": NULL_SEEDS,
    "targets": CONFIRMATORY_SEEDS,
    "arms": EXPLORATORY_SEEDS,
    "summarise": (),
}
STAGES = {
    "window": stage_window,
    "planted": stage_planted,
    "null": stage_null,
    "targets": stage_targets,
    "arms": stage_arms,
    "summarise": stage_summarise,
}


def hide_the_card() -> None:
    """A CPU fixture on a box with a GPU never touches the card.

    Set here, when the script runs, and never at import: pytest imports every
    test module at collection, so an import-time ``CUDA_VISIBLE_DEVICES=""``
    hid the card from the GPU suite in the same process (pipelines 546 and
    549). CUDA initialises lazily, so setting it before the first CUDA call is
    enough, and the spawned workers inherit it. ``setdefault`` keeps an
    operator's explicit choice.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> None:
    hide_the_card()
    args = build_parser().parse_args()
    if args.seeds is None:
        args.seeds = DEFAULT_SEEDS[args.stage]
    if args.scale != 1.0:
        raise SystemExit(
            "every 2x row waits on #73's exchange rate; this script runs the 1x rows only"
        )
    if NONE not in args.blockades:
        args.blockades = [NONE, *args.blockades]
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
