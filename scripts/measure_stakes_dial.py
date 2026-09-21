"""The stakes dial on the fixtures (#39): three arms, three signatures, one primary.

Does a cell's continuation depending on its realised value produce the
preregistered signatures (``docs/preregistration.md``)? Three arms one field
apart -- ``value`` (today's economy), ``decoupled`` (pinned wealth, uniform
re-entry, a shadow ledger) and ``shuffled`` (regression targets permuted across
experts each step) -- on the two planted-competence fixtures, three seeds each,
before any GPU time.

**Signature 1, the primary.** Two goal fields on the differentiated fixture
(``TypeGoalField``, types 0 and 1 at a setpoint of ``SETPOINT``), field 1 at
``ratio x REFERENCE_DOSE`` against field 2 at ``REFERENCE_DOSE``. Per arm, the
total-variation shift of the slot allocation from the balanced ratio, paired by
seed (``allocation_shift.paired_shifts``), and the least-squares slope of that
shift against the ratio per seed (``dose_slope.per_seed_slopes``). The primary
is that slope, ``value`` minus ``decoupled``, with ``compare_runs``' percentile
bootstrap over the three paired seeds -- the resampled-mean range, which at
n = 3 is the sample range and carries no 95% coverage. Inside the range reads
"no". The fixture is bitwise deterministic on CPU, so the re-running floor is
identically zero; one replicate is run to show it rather than assumed.

**Signature 2.** The residual of the report on token class after conditioning
on realised value: per expert, the report regressed on the value it realised on
the tokens it held, and the residual's mean on the class that paid minus the
class that did not. On the differentiated fixture the class is the token's
type against the expert's own, which a head can read off the input; on the
quality fixture it is the sign of the correction (``positive_fraction`` below
one), which a head cannot, so that reading is the residual's own noise floor.

**Signature 3.** Mean report and abstention rate (the logit at or below the
head's initial one, ``CONFIDENCE_INITIAL_LOGIT``) binned by wealth over the
band, the floor-adjacent band minus the middle band, on the quality fixture as
recorded (``positive_fraction`` 1.0) run to its steady state and then through
``economy_damage``'s 150-step forced episode and its release. Under
``decoupled`` the wealth binned on is the shadow ledger: what the cell would
have had, which is what the signature asks about. The same run reads the
r(wealth, competence) guardrail at the end of its steady phase.

**Guardrails.** Held-out loss per arm against the pooled seed spread;
r(wealth, competence) on the quality fixture in the ``value`` arm, which is
today's economy and must read #15's band; the shadow and live ledgers at step 0,
pinned in ``tests/test_stakes_dial.py``. **Secondary, reported:** the on-type
share, each goal type's share, ``shuffled`` beside ``decoupled`` on every row,
and #35's multiplicity line on the ``value`` vs ``decoupled`` table.

Groups are written in ``run_seeds.py``'s summary format so that
``scripts/allocation_shift.py`` and ``scripts/compare_runs.py`` read them
unchanged, with a hand-built fingerprint per seed carrying the dial, the doses
and the code SHA.

    uv run python scripts/measure_stakes_dial.py --out ~/tame-runs/39-stakes-dial/fixture
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from allocation_shift import (  # noqa: E402
    DEFAULT_READOUT,
    HELD_TYPE_PREFIX,
    READING_PREFIX,
    READOUTS,
    SWEPT_TYPE_PREFIX,
    Readout,
    paired_shifts,
)
from compare_runs import (  # noqa: E402
    DEFAULT_RESAMPLES,
    assert_groups_at_parity,
    bootstrap_mean,
    compare,
    format_table,
    load_group,
)
from dose_slope import interval_label, per_seed_slopes  # noqa: E402
from economy_damage import (  # noqa: E402
    LONG_FORCED_EPISODE,
    RELEASE_HORIZON,
    STEADY_STEPS,
    ForcedSubset,
)
from run_seeds import aggregate  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from mob import PERSISTENCE_DECOUPLED, PERSISTENCE_SHUFFLED, PERSISTENCE_VALUE  # noqa: E402
from mob.experts import CONFIDENCE_INITIAL_LOGIT  # noqa: E402
from parity import ArmFingerprint, arm_label, code_identity  # noqa: E402
from readiness import READINESS_OFF, ReadinessConfig  # noqa: E402

ARMS = (PERSISTENCE_VALUE, PERSISTENCE_DECOUPLED, PERSISTENCE_SHUFFLED)
SEEDS = (0, 1, 2)
STEPS = 600
TAIL = 100
# One wealth memory horizon: at decay 0.997 the ledger has forgotten what
# happened 333 steps ago to within 1/e (#16). Nothing integrated is read before
# it, because a reading taken inside the transient is reading the heads learning
# what their experts are worth rather than the market they settled into.
WEALTH_HORIZON = 333
# The window one logged reading of the allocation averages over (#57 candidate
# 3). Fifty steps is 100 tokens a slot at this fixture's batch, enough that a
# window's shares are not counting noise, and short enough that the tail after
# one horizon holds six of them.
READING_WINDOW = 50
# The dose of the second goal field, in per-token loss per unit of goal error.
# Chosen against the fixture's own scale -- realised value per unit share reads
# about 0.24 and the per-token loss about 0.18 at the recorded steady state -- so
# that at the balanced ratio the goal competes with the loss without deciding
# every slot on its own. The fixture has no injection to price it against
# (docs/preregistration.md, deviations).
REFERENCE_DOSE = 0.25
RATIOS = (1.0, 2.0, 4.0)
SETPOINT = 0.5
GOAL_TYPES = (0, 1)
# The quality fixture flips the correction on a fifth of tokens (#15's unbiased
# report fixture), which is the class signature 2 reads there.
POSITIVE_FRACTION = 0.8
WEALTH_BANDS = 5
FLOOR_BAND, MIDDLE_BAND = 0, 2
PRIMARY = "signature1/allocation_shift_slope"
DEFAULT_OUT = Path.home() / "tame-runs" / "39-stakes-dial" / "fixture"
# The substrate label every fingerprint from this fixture carries.
FIXTURE = "differentiated-fixture"


@dataclass
class TokenRecords:
    """Per expert: the report, the value realised and the class, on every token it held."""

    reports: dict[int, list[float]] = field(default_factory=dict)
    values: dict[int, list[float]] = field(default_factory=dict)
    paid: dict[int, list[bool]] = field(default_factory=dict)

    def add(self, expert: int, report: float, value: float, paid: bool) -> None:
        self.reports.setdefault(expert, []).append(report)
        self.values.setdefault(expert, []).append(value)
        self.paid.setdefault(expert, []).append(paid)


def fixture_fingerprint(
    fixture: str,
    seed: int,
    arm: str,
    doses: tuple[float, ...],
    steps: int,
    cells: int = BASE_CONFIG.num_experts,
    contribution_scale: float = 1.0,
    readiness: ReadinessConfig = READINESS_OFF,
) -> ArmFingerprint:
    """A fingerprint for a fixture run: what the arms share, the dial, the doses, the code.

    ``readiness`` is #46's register, taken here as ``parity.fingerprint_arm`` takes
    it, so a fixture run that granted an autonomy cannot record itself as one that
    did not. Every flag is off until the issue that earns one turns it on.

    ``cells`` and ``contribution_scale`` are #60's two ratios, taken as arguments
    for the same reason: a grid run at four cells or at twice the correction
    recorded itself as the recorded fixture while it was neither, so the
    fingerprint said two incomparable runs were at parity.
    """
    code_sha, code_dirty = code_identity()
    return ArmFingerprint(
        router="mob",
        seed=seed,
        deterministic=True,
        model_id=fixture,
        dtype="float32",
        dataset="planted-competence",
        max_steps=steps,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=16,
        learning_rate=1e-2,
        warmup_steps=0,
        weight_decay=0.0,
        num_experts=cells,
        top_k=BASE_CONFIG.top_k,
        adapter_rank=BASE_CONFIG.adapter_rank,
        requested_layers=(0,),
        use_lora=False,
        lora_rank=0,
        lora_alpha=0,
        lora_dropout=0.0,
        calibration_loss_weight=BASE_CONFIG.confidence_calibration_weight,
        exploration_rate=BASE_CONFIG.exploration_rate,
        exploration_draw=BASE_CONFIG.exploration_draw,
        confidence_head_learning_rate=1e-2,
        wealth_update_frequency=1,
        coupling_goal=None,
        coupling_beta=0.0,
        coupling_warmup_steps=1,
        gradient_checkpointing=False,
        device="cpu",
        probe_tokens=0,
        eval_split="planted",
        data_order=f"seed{seed}",
        converted_layers=1,
        strict_determinism=True,
        code_sha=code_sha,
        code_dirty=code_dirty,
        persistence_coupling=arm,
        goal_doses=doses,
        contribution_scale=contribution_scale,
        autonomy_plasticity=readiness.autonomy_plasticity,
        autonomy_exploration=readiness.autonomy_exploration,
        autonomy_setpoints=readiness.autonomy_setpoints,
        autonomy_evaluation=readiness.autonomy_evaluation,
        autonomy_dormancy=readiness.autonomy_dormancy,
    )


def _record_tokens(economy: SyntheticEconomy, records: TokenRecords, paid: torch.Tensor) -> None:
    """``paid`` is ``(num_experts, batch, seq)``: whether the token's class paid this expert."""
    stats = economy.mob.last_stats
    values = economy.mob.last_realised_values
    assert stats is not None and values is not None
    selected = stats.selected_experts
    for slot in range(economy.config.top_k):
        for expert in range(economy.config.num_experts):
            held = selected[:, :, slot] == expert
            if not held.any():
                continue
            reports = stats.confidences[:, :, expert][held].tolist()
            realised = values[:, :, slot][held].tolist()
            paid_here = paid[expert][held].tolist()
            for report, value, was_paid in zip(reports, realised, paid_here, strict=True):
                records.add(expert, report, value, was_paid)


def build_differentiated(arm: str, ratio: float, seed: int, **overrides) -> DifferentiatedEconomy:
    """The fixture #39 read signature 1 on: two goal fields at a relative dose.

    ``overrides`` are ``MoBConfig`` fields, and every one of them is a deviation
    from the recorded configuration -- #57's mechanism ablation switches one
    ledger mechanic at a time through here and labels the arm with it.
    """
    config = replace(BASE_CONFIG, persistence_coupling=arm, **overrides)
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)
    economy.add_goal_field(GOAL_TYPES[0], SETPOINT, ratio * REFERENCE_DOSE)
    economy.add_goal_field(GOAL_TYPES[1], SETPOINT, REFERENCE_DOSE)
    return economy


def type_wins(economy: DifferentiatedEconomy, selected: torch.Tensor) -> dict[int, torch.Tensor]:
    """Slots won per expert, counted separately on each goal type's tokens.

    #57's candidate 2 reads the allocation where the swept goal is about the
    tokens in front of the cell; the aggregate averages that over the three
    types the goal pays nothing on.
    """
    assert economy.last_types is not None
    counts = {}
    for expert_type in GOAL_TYPES:
        held = selected[economy.last_types == expert_type]
        counts[expert_type] = torch.bincount(
            held.flatten(), minlength=economy.config.num_experts
        ).float()
    return counts


def share_columns(wins: torch.Tensor, prefix: str) -> dict[str, float]:
    """A win-share column per expert, or nothing when the window saw no token."""
    total = float(wins.sum())
    if total == 0.0:
        return {}
    return {f"{prefix}{i}": float(wins[i] / total) for i in range(wins.numel())}


def run_differentiated(
    arm: str, ratio: float, seed: int, steps: int = STEPS
) -> tuple[dict[str, float], TokenRecords]:
    """One arm at one relative dose: the tail metrics and the per-token records.

    The tail accumulation is #39's, untouched and bitwise reproducible; what #57
    adds is recording, not trajectory -- the type-conditioned counts and the
    windowed readings are read off the same steps and consume no randomness.
    """
    economy = build_differentiated(arm, ratio, seed)
    wins = torch.zeros(economy.config.num_experts)
    by_type = {expert_type: torch.zeros(economy.config.num_experts) for expert_type in GOAL_TYPES}
    window = torch.zeros(economy.config.num_experts)
    readings: dict[str, float] = {}
    losses: list[float] = []
    on_type: list[float] = []
    records = TokenRecords()
    for step in range(steps):
        record = economy.step()
        window += torch.bincount(
            record.selected_experts.flatten(), minlength=economy.config.num_experts
        ).float()
        if (step + 1) % READING_WINDOW == 0:
            if step + 1 > WEALTH_HORIZON:
                readings.update(share_columns(window, f"{READING_PREFIX}{step + 1}_e"))
            window = torch.zeros(economy.config.num_experts)
        if step < steps - TAIL:
            continue
        wins += torch.bincount(
            record.selected_experts.flatten(), minlength=economy.config.num_experts
        ).float()
        for expert_type, counts in type_wins(economy, record.selected_experts).items():
            by_type[expert_type] += counts
        losses.append(record.loss)
        on_type.append(economy.on_type_share(record.selected_experts))
        assert economy.last_types is not None
        own_type = economy.last_types.unsqueeze(0) == economy.expert_types.view(-1, 1, 1)
        _record_tokens(economy, records, own_type)
    share = wins / wins.sum()
    metrics = {f"routing/win_share_e{i}": float(share[i]) for i in range(share.numel())}
    metrics["eval/loss"] = sum(losses) / len(losses)
    metrics["routing/on_type_share"] = sum(on_type) / len(on_type)
    for expert_type in GOAL_TYPES:
        metrics[f"goal/type{expert_type}_share"] = float(
            share[economy.expert_types == expert_type].sum()
        )
    metrics.update(share_columns(by_type[GOAL_TYPES[0]], SWEPT_TYPE_PREFIX))
    metrics.update(share_columns(by_type[GOAL_TYPES[1]], HELD_TYPE_PREFIX))
    metrics.update(readings)
    return metrics, records


def write_group(
    path: Path,
    fixture: str,
    arm: str,
    doses: tuple[float, ...],
    per_seed: dict[int, dict[str, float]],
    steps: int,
    primary: str | None,
) -> dict[str, Any]:
    """A ``seed_summary.json`` in ``run_seeds.py``'s format, read back through ``load_group``."""
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "arm": arm_label("mob", None, None, arm),
        "router": "mob",
        "coupling_goal": None,
        "steer_goal": None,
        "persistence_coupling": arm,
        "trace_goal": None,
        "seeds": list(per_seed),
        "steps": steps,
        "primary": primary,
        "per_seed": per_seed,
        "fingerprints": {
            seed: fixture_fingerprint(fixture, seed, arm, doses, steps).as_dict()
            for seed in per_seed
        },
        "stats": aggregate(per_seed),
        "replicate_seed": None,
        "replicate": None,
        "replication_std": None,
        "replication_error": "the fixture is bitwise deterministic on CPU; the floor is zero",
    }
    (path / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    return load_group(path)


def conditioned_preference(records: TokenRecords) -> float:
    """Signature 2: the residual's paid-class minus unpaid-class mean, averaged over experts.

    An expert enters only with both classes on its tokens and a spread of
    realised value to regress on; the least-squares line is the head's own
    objective, so what is left is what the report does beyond it.
    """
    differences: list[float] = []
    for expert, reports in records.reports.items():
        y = torch.tensor(reports)
        x = torch.tensor(records.values[expert])
        paid = torch.tensor(records.paid[expert])
        if paid.all() or not paid.any() or x.numel() < 3 or x.var() == 0:
            continue
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        residual = y - (y.mean() + slope * (x - x.mean()))
        differences.append(float(residual[paid].mean() - residual[~paid].mean()))
    return sum(differences) / len(differences) if differences else float("nan")


@dataclass(frozen=True)
class QualityReading:
    """One arm at one seed on the quality fixture: signature 3, signature 2 and the guardrail.

    Signature 2 is read from a separate steady run at ``POSITIVE_FRACTION`` --
    the recorded fixture has one class only -- and the rest from the recorded
    fixture's own settings, so the guardrail is read where #15's band was.
    """

    report_floor_minus_middle: float
    abstention_floor_minus_middle: float
    band_counts: list[int]
    conditioned_preference: float
    wealth_vs_competence: float
    steady_loss: float


def _band(wealth: torch.Tensor, config) -> torch.Tensor:
    width = (config.max_wealth - config.min_wealth) / WEALTH_BANDS
    return ((wealth - config.min_wealth) / width).floor().clamp(0, WEALTH_BANDS - 1).long()


def signed_class_preference(arm: str, seed: int, steps: int) -> float:
    """Signature 2 on the quality fixture: a steady run with a class the head cannot read."""
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    economy = SyntheticEconomy(
        shuffled(DEFAULT_COMPETENCE, seed),
        seed=seed,
        config=config,
        positive_fraction=POSITIVE_FRACTION,
    )
    records = TokenRecords()
    for step in range(steps):
        economy.step()
        if step < steps - TAIL:
            continue
        assert economy.last_sign is not None
        paid = (economy.last_sign > 0).unsqueeze(0).expand(economy.config.num_experts, -1, -1)
        _record_tokens(economy, records, paid)
    return conditioned_preference(records)


def run_quality(arm: str, seed: int, steps: int) -> QualityReading:
    """Steady state, the 150-step forced episode to the three least competent, the release."""
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    economy = SyntheticEconomy(competence, seed=seed, config=config)
    least_competent = competence.argsort()[:3].tolist()
    schedule = [
        (max(steps, STEADY_STEPS), None),
        (LONG_FORCED_EPISODE, least_competent),
        (RELEASE_HORIZON, None),
    ]
    steady_losses: list[float] = []
    wealth_vs_competence = float("nan")
    reports_by_band = [[] for _ in range(WEALTH_BANDS)]
    abstentions_by_band = [[] for _ in range(WEALTH_BANDS)]
    for phase, (length, forced) in enumerate(schedule):
        gate = economy.mob.gate
        if forced is not None:
            economy.mob.gate = ForcedSubset(forced, economy.config.top_k, seed=1000 * length + seed)
        try:
            for step in range(length):
                record = economy.step()
                stats = economy.mob.last_stats
                assert stats is not None
                bands = _band(stats.expert_wealth, economy.config)
                mean_report = stats.confidences.mean(dim=(0, 1))
                abstaining = (
                    (stats.confidence_logits <= CONFIDENCE_INITIAL_LOGIT).float().mean(dim=(0, 1))
                )
                for expert in range(economy.config.num_experts):
                    band = int(bands[expert])
                    reports_by_band[band].append(float(mean_report[expert]))
                    abstentions_by_band[band].append(float(abstaining[expert]))
                if phase == 0 and step >= length - TAIL:
                    steady_losses.append(record.loss)
        finally:
            economy.mob.gate = gate
        if phase == 0:
            wealth_vs_competence = pearson(economy.mob.expert_wealth, competence)

    def band_mean(rows: list[list[float]], band: int) -> float:
        return sum(rows[band]) / len(rows[band]) if rows[band] else float("nan")

    return QualityReading(
        report_floor_minus_middle=band_mean(reports_by_band, FLOOR_BAND)
        - band_mean(reports_by_band, MIDDLE_BAND),
        abstention_floor_minus_middle=band_mean(abstentions_by_band, FLOOR_BAND)
        - band_mean(abstentions_by_band, MIDDLE_BAND),
        band_counts=[len(rows) for rows in reports_by_band],
        conditioned_preference=signed_class_preference(arm, seed, steps),
        wealth_vs_competence=wealth_vs_competence,
        steady_loss=sum(steady_losses) / len(steady_losses),
    )


def contrast_line(
    label: str, per_seed_a: dict[str, float], per_seed_b: dict[str, float], resamples: int
) -> str:
    """``b - a`` per shared seed, its mean, and the resampled-mean range."""
    seeds = sorted(set(per_seed_a) & set(per_seed_b), key=str)
    deltas = [per_seed_b[s] - per_seed_a[s] for s in seeds]
    if any(math.isnan(d) for d in deltas):
        return f"{label:<52} n/a (a seed reads NaN: an empty band or class)"
    centre, low, high = bootstrap_mean(deltas, resamples, 0)
    per_seed = "  ".join(f"s{s}={d:+.4f}" for s, d in zip(seeds, deltas, strict=True))
    return (
        f"{label:<52}{centre:>+9.4f}  [{low:+.4f}, {high:+.4f}]  "
        f"{interval_label(len(deltas))}  {per_seed}"
    )


def pooled_spread(groups: list[dict[str, Any]], metric: str) -> float:
    spreads = [g["stats"][metric]["std"] for g in groups if metric in g["stats"]]
    return (sum(s**2 for s in spreads) / len(spreads)) ** 0.5 if spreads else float("nan")


def primary_name(readout: Readout) -> str:
    """The primary's metric name, which carries the readout unless it is the recorded one.

    #39's groups are on disk under ``PRIMARY``; a group read with another readout
    must not be able to pass for one of them (section 8, rule 2).
    """
    return PRIMARY if readout is DEFAULT_READOUT else f"{PRIMARY}@{readout.name}"


def signature_one(
    out: Path,
    seeds: tuple[int, ...],
    steps: int,
    resamples: int,
    readout: Readout = DEFAULT_READOUT,
) -> dict[str, Any]:
    """Run every arm at every ratio, write the groups, and read the primary."""
    groups: dict[str, dict[float, dict[str, Any]]] = {}
    preference: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        groups[arm] = {}
        preference[arm] = {}
        for ratio in RATIOS:
            per_seed: dict[int, dict[str, float]] = {}
            for seed in seeds:
                metrics, records = run_differentiated(arm, ratio, seed, steps)
                per_seed[seed] = metrics
                if ratio == 1.0:
                    preference[arm][str(seed)] = conditioned_preference(records)
                print(
                    f"  {arm:<10} ratio {ratio:<4} seed {seed}: loss {metrics['eval/loss']:.4f}"
                    f"  type0 {metrics['goal/type0_share']:.3f}"
                    f"  type1 {metrics['goal/type1_share']:.3f}"
                    f"  on-type {metrics['routing/on_type_share']:.3f}",
                    flush=True,
                )
            doses = (ratio * REFERENCE_DOSE, REFERENCE_DOSE)
            groups[arm][ratio] = write_group(
                out / "signature1" / f"{arm}@r{ratio}",
                FIXTURE,
                arm,
                doses,
                per_seed,
                steps,
                primary_name(readout) if ratio == RATIOS[0] else None,
            )
    for ratio in RATIOS:
        for arm in ARMS[1:]:
            assert_groups_at_parity(groups[PERSISTENCE_VALUE][ratio], groups[arm][ratio])

    replicate, _ = run_differentiated(PERSISTENCE_VALUE, RATIOS[0], seeds[0], steps)
    floor = (
        sum(
            abs(replicate[k] - groups[PERSISTENCE_VALUE][RATIOS[0]]["per_seed"][str(seeds[0])][k])
            for k in replicate
            if k.startswith("routing/win_share_e")
        )
        / 2
    )

    shifts = {
        arm: {
            ratio: paired_shifts(groups[arm][RATIOS[0]], groups[arm][ratio], readout)
            for ratio in RATIOS
        }
        for arm in ARMS
    }
    slopes = {arm: per_seed_slopes(shifts[arm]) for arm in ARMS}
    return {
        "groups": groups,
        "shifts": shifts,
        "slopes": slopes,
        "floor": floor,
        "preference": preference,
        "resamples": resamples,
        "readout": readout,
    }


def print_signature_one(result: dict[str, Any]) -> None:
    shifts, slopes, groups = result["shifts"], result["slopes"], result["groups"]
    resamples, readout = result["resamples"], result["readout"]
    print(f"\n== signature 1: the shift from the balanced ratio, per seed ({readout.name}) ==")
    if readout is not DEFAULT_READOUT:
        print(f"  readout: {readout.describe}")
    for arm in ARMS:
        for ratio in RATIOS[1:]:
            per_seed = "  ".join(f"s{s}={v:.3f}" for s, v in shifts[arm][ratio].items())
            print(f"  {arm:<10} ratio {ratio:<4} {per_seed}")
    print(f"  re-running floor (one replicate of value@r1.0 seed 0): {result['floor']:.4f}")
    print("\n  slope of the shift against the ratio, per seed:")
    for arm in ARMS:
        per_seed = "  ".join(f"s{s}={v:+.4f}" for s, v in slopes[arm].items())
        print(f"  {arm:<10} {per_seed}")
    print(f"\n== primary: {primary_name(readout)}, value minus decoupled ==")
    print(
        contrast_line(
            "value - decoupled", slopes[PERSISTENCE_DECOUPLED], slopes[PERSISTENCE_VALUE], resamples
        )
    )
    print(
        contrast_line(
            "shuffled - decoupled (secondary)",
            slopes[PERSISTENCE_DECOUPLED],
            slopes[PERSISTENCE_SHUFFLED],
            resamples,
        )
    )
    print("  a range that includes zero reads 'no'; the fixture's floor is the line above")

    print("\n== guardrail: eval/loss per arm (tail mean), pooled seed spread across the arms ==")
    for ratio in RATIOS:
        arms = [groups[arm][ratio] for arm in ARMS]
        spread = pooled_spread(arms, "eval/loss")
        means = "  ".join(
            f"{arm}={g['stats']['eval/loss']['mean']:.4f}"
            for arm, g in zip(ARMS, arms, strict=True)
        )
        largest = max(
            abs(g["stats"]["eval/loss"]["mean"] - arms[0]["stats"]["eval/loss"]["mean"])
            for g in arms[1:]
        )
        print(
            f"  ratio {ratio:<4} {means}  pooled spread {spread:.4f}  largest |delta| {largest:.4f}"
        )

    print("\n== signature 2 on the differentiated fixture (balanced ratio): class = own type ==")
    preference = result["preference"]
    for arm in ARMS:
        per_seed = "  ".join(f"s{s}={v:+.4f}" for s, v in preference[arm].items())
        print(f"  {arm:<10} {per_seed}")
    print(
        contrast_line(
            "value - decoupled",
            preference[PERSISTENCE_DECOUPLED],
            preference[PERSISTENCE_VALUE],
            resamples,
        )
    )
    print(
        contrast_line(
            "shuffled - decoupled",
            preference[PERSISTENCE_DECOUPLED],
            preference[PERSISTENCE_SHUFFLED],
            resamples,
        )
    )

    print("\n== value vs decoupled at the balanced ratio, every tail metric (#35's line below) ==")
    comparison = compare(
        groups[PERSISTENCE_VALUE][RATIOS[0]], groups[PERSISTENCE_DECOUPLED][RATIOS[0]]
    )
    print(format_table(comparison, "value", "decoupled"))


def signatures_two_and_three(seeds: tuple[int, ...], steps: int, resamples: int) -> dict[str, Any]:
    readings = {arm: {str(seed): run_quality(arm, seed, steps) for seed in seeds} for arm in ARMS}
    print("\n== quality fixture: signature 3 (floor band minus middle band) and signature 2 ==")
    band = f"[{BASE_CONFIG.min_wealth}, {BASE_CONFIG.max_wealth}]"
    print(f"  bands: {WEALTH_BANDS} equal widths over {band}")
    for arm in ARMS:
        for seed, reading in readings[arm].items():
            print(
                f"  {arm:<10} seed {seed}: report {reading.report_floor_minus_middle:+.4f}"
                f"  abstention {reading.abstention_floor_minus_middle:+.4f}"
                f"  bands {reading.band_counts}  sig2 {reading.conditioned_preference:+.4f}"
                f"  r(w,c) {reading.wealth_vs_competence:+.3f}  loss {reading.steady_loss:.4f}"
            )

    def column(arm: str, name: str) -> dict[str, float]:
        return {seed: getattr(reading, name) for seed, reading in readings[arm].items()}

    lines = []
    for name, label in (
        ("report_floor_minus_middle", "signature 3, mean report"),
        ("abstention_floor_minus_middle", "signature 3, abstention rate"),
        ("conditioned_preference", "signature 2, class = the correction's sign"),
    ):
        for arm in (PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED):
            lines.append(
                contrast_line(
                    f"{label}: {arm} - decoupled",
                    column(PERSISTENCE_DECOUPLED, name),
                    column(arm, name),
                    resamples,
                )
            )
    print("\n".join(lines))
    value_tracking = column(PERSISTENCE_VALUE, "wealth_vs_competence")
    print(
        "\n  guardrail: r(wealth, competence) in the value arm at steady state, "
        + "  ".join(f"s{s}={v:+.3f}" for s, v in value_tracking.items())
        + "  (#15's band: above 0.5; recorded 0.79-0.85 on this fixture at 600 steps)"
    )
    return {
        arm: {seed: reading.__dict__ for seed, reading in readings[arm].items()} for arm in ARMS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="where the groups are written"
    )
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--skip-quality", action="store_true", help="signature 1 only")
    parser.add_argument(
        "--readout",
        type=str,
        default=DEFAULT_READOUT.name,
        choices=sorted(READOUTS),
        help=(
            "Which readout signature 1's primary is read with (#57). The default is the one "
            "#39's recorded rows were read with; a group read with another carries the "
            f"readout in its primary's name (default: {DEFAULT_READOUT.name})"
        ),
    )
    args = parser.parse_args()
    readout = READOUTS[args.readout]
    if readout.within_run:
        parser.error(
            f"{readout.name} is not read between two dose groups; it is the estimator study's "
            "protocol (scripts/estimator_study.py --stage setpoint, or --stage validate "
            f"--chosen {readout.name})"
        )
    seeds = tuple(int(part) for part in args.seeds.split(","))
    code_sha, code_dirty = code_identity()
    print(f"code {code_sha} dirty={code_dirty}; groups under {args.out}")
    print(
        f"arms {ARMS}; seeds {seeds}; {args.steps} steps, tail {TAIL}; goal types {GOAL_TYPES} "
        f"at setpoint {SETPOINT}, reference dose {REFERENCE_DOSE}, ratios {RATIOS}"
    )
    result = signature_one(args.out, seeds, args.steps, args.resamples, readout)
    print_signature_one(result)
    quality = (
        None if args.skip_quality else signatures_two_and_three(seeds, args.steps, args.resamples)
    )
    record = {
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "seeds": seeds,
        "steps": args.steps,
        "reference_dose": REFERENCE_DOSE,
        "ratios": RATIOS,
        "setpoint": SETPOINT,
        "readout": readout.name,
        "shifts": {
            arm: {str(r): v for r, v in shifts.items()} for arm, shifts in result["shifts"].items()
        },
        "slopes": result["slopes"],
        "floor": result["floor"],
        "preference_differentiated": result["preference"],
        "quality": quality,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stakes_dial.json").write_text(json.dumps(record, indent=2))
    print(f"\nrecord: {args.out / 'stakes_dial.json'}")


if __name__ == "__main__":
    main()
