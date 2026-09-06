"""Recovery of the wired system from designed and undesigned perturbations (#6).

Two organisms, two kinds of damage. The **steering tissue** (MoB blocks, a
calibrated goal tissue, the routing coupling seeded from the same direction --
``tests/wired_system.py``) is perturbed by content that drags the stream off its
resting alignment, which is the disturbance the loop was designed for, and by
the removal of an actuator mid-generation, which it was not. The **expert
economy** (``scripts/synthetic_economy.py``, competence planted and shuffled away
from expert index) is damaged in three ways the auction was never designed for:
its most competent expert goes senescent, routing is forced onto its least
competent experts for a while, and its most competent expert is ruined.

Recovery from a designed disturbance is a control result; an adequately tuned
loop passes it. The distinctly TAME claim is the second kind: the collective
re-forms *function* after damage nobody planned for -- scrambled cells still
building the face. Every recovery below is measured against the state that
would count as no recovery, and paired with the state in which the mechanism
that recovers is disabled: the inert loop (``kp = ki = 0``, still sensing), the
heads frozen so no report can re-learn what a dead expert is worth, and -- for
the two claims the current economy does not meet -- a strict expected failure
that names the mechanism it waits on.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    SyntheticEconomy,
    pearson,
    shuffled,
)

from mob.auction import AuctionOutcome  # noqa: E402

from .wired_system import ACTUATORS, BELOW_ACTUATORS, READOUT, build_wired_system  # noqa: E402

# --- The steering tissue ---------------------------------------------------------

# #6's acceptance criterion: the tissue's mean error within this fraction of the
# tissue setpoint, within this many forward passes of the disturbance.
RECOVERY_FRACTION = 0.05
RECOVERY_PASSES = 200
SETTLE_PASSES = 60
# A content deficit of one strength unit along the goal direction: about 490 sigma
# on this fixture's tissue, and about a quarter of the tissue setpoint uncorrected.
CONTENT_DEFICIT = -1.0
DEAD_CELL_DEFICIT = -0.5


def _passes_to_recover(system, goal: str | None = None) -> int | None:
    """First pass after which the tissue error stays inside the band, or None."""
    tolerance = RECOVERY_FRACTION * abs(system.setpoint(goal))
    errors = []
    for _ in range(RECOVERY_PASSES):
        system.step()
        errors.append(abs(system.error(goal)))
    for index, _ in enumerate(errors):
        if all(error <= tolerance for error in errors[index:]):
            return index + 1
    return None


@pytest.mark.parametrize("coupled", [True, False], ids=["coupled", "uncoupled"])
def test_the_tissue_recovers_from_content_that_drags_the_stream_off_its_setpoint(coupled):
    """Designed perturbation: a persistent deficit above the bottom actuator.

    Every cell above the bottom actuator reads the deficit and the actuators above
    it can answer it. The regulated variable is the tissue's mean error over its
    live cells -- what the shared integrator drives -- and it returns to within 5%
    of the tissue setpoint well inside the 200-pass budget (measured: inside the
    band from pass 23, coupled or not) while the actuators' strength rises to
    carry the deficit. With the coupling live the routing perceives the same
    direction; the tissue's recovery does not depend on it either way.
    """
    system = build_wired_system(coupled=coupled)
    system.run(SETTLE_PASSES)
    resting_strength = system.strength()
    assert abs(system.error()) <= RECOVERY_FRACTION * system.setpoint()

    system.set_content("truthful", CONTENT_DEFICIT)
    recovered_at = _passes_to_recover(system)

    assert recovered_at is not None and recovered_at <= RECOVERY_PASSES, recovered_at
    assert system.strength() > resting_strength + 0.2, "recovery has to be paid for in strength"


def test_the_inert_loop_leaves_the_deficit_in_place():
    """The pairing: cells that sense but cannot act never bring the error back."""
    system = build_wired_system(kp=0.0, ki=0.0)
    system.run(SETTLE_PASSES)
    system.set_content("truthful", CONTENT_DEFICIT)

    assert _passes_to_recover(system) is None
    assert abs(system.error()) > 4 * RECOVERY_FRACTION * system.setpoint()
    assert system.strength() == pytest.approx(system.tissue().config.base_strength)


def test_a_deficit_below_the_bottom_actuator_is_diluted_over_the_live_cells():
    """The tissue's answer to a deficit no cell can act on, pinned as a property.

    Content that enters below every actuator is read by the bottom cell too, and
    nothing can correct it there. The shared integrator drives the tissue *mean*
    to zero regardless, so the regulable cells settle past their setpoints by the
    blind cell's error divided by the number of other live cells. The tissue meets
    its criterion on the mean; the cost is paid by the cells above, and it shrinks
    with the cell count -- one seventh on the served eight-cell tissue, one
    quarter here.
    """
    system = build_wired_system()
    system.run(SETTLE_PASSES)
    system.set_content("truthful", DEAD_CELL_DEFICIT, layer=BELOW_ACTUATORS)

    assert _passes_to_recover(system) is not None
    errors = [cell["error"] for cell in system.tissue().status()["cells"]]
    blind, others = errors[0], errors[1:]
    assert blind > RECOVERY_FRACTION * system.setpoint(), "the bottom cell stays in deficit"
    assert sum(others) / len(others) == pytest.approx(-blind / len(others), rel=0.05)


READOUT_DRIFT_FRACTION = 0.02


def _readout_error(system) -> tuple[float, float]:
    cell = next(cell for cell in system.tissue().status()["cells"] if cell["layer"] == READOUT)
    return cell["error"], cell["setpoint"]


def test_the_tissue_recovers_after_an_actuator_is_removed_mid_generation():
    """Undesigned perturbation: a cell stops firing while the tissue is carrying a load.

    The removed cell leaves the consensus after one pass and the surviving
    actuators take up its share of the effort. The tissue error is the mean over
    the cells still live, a quantity the removal itself redefines, so the outcome
    that counts is measured at the top of the stack: the readout, which acts on
    nothing and sits downstream of every actuator, ends where it was before the
    damage (measured: within 0.3% of its setpoint, against the 2% allowed). The
    tissue mean peaks near 15% of the setpoint at pass 14 and is back inside the
    5% band by pass 37.
    """
    system = build_wired_system()
    system.run(SETTLE_PASSES)
    system.set_content("truthful", CONTENT_DEFICIT)
    assert _passes_to_recover(system) is not None
    strengths_before = dict(system.tissue()._strength)
    readout_before, readout_setpoint = _readout_error(system)
    killed = ACTUATORS[2]

    system.kill_actuator(killed)
    system.run(3)
    alive = {cell["layer"]: cell["alive"] for cell in system.tissue().status()["cells"]}
    assert alive[killed] is False and all(alive[layer] for layer in alive if layer != killed)

    assert _passes_to_recover(system) is not None
    survivors = [layer for layer in ACTUATORS if layer != killed]
    assert all(system.tissue()._strength[layer] > strengths_before[layer] for layer in survivors)
    readout_after, _ = _readout_error(system)
    assert abs(readout_after - readout_before) <= READOUT_DRIFT_FRACTION * readout_setpoint


def test_the_inert_loop_cannot_absorb_a_removed_actuator():
    system = build_wired_system(kp=0.0, ki=0.0)
    system.run(SETTLE_PASSES)
    system.set_content("truthful", CONTENT_DEFICIT)
    system.run(SETTLE_PASSES)
    error_before = abs(system.error())

    system.kill_actuator(ACTUATORS[2])
    assert _passes_to_recover(system) is None
    assert abs(system.error()) > error_before


# --- The expert economy ----------------------------------------------------------

STEADY_STEPS = 200
WINDOW = 50
DAMAGE_HORIZON = 200
# A market that has re-formed sits at the loss a collective born without the
# damaged expert reaches, within this factor, and no longer routes to it.
RE_FORMATION_FACTOR = 1.1
DEAD_SHARE_CEILING = 0.01
SURVIVOR_TRACKING_FLOOR = 0.5
SEEDS = (0, 1, 2)


def _window(economy: SyntheticEconomy, steps: int) -> tuple[float, torch.Tensor]:
    """Mean loss and per-expert win share over ``steps`` steps."""
    losses: list[float] = []
    wins = torch.zeros(economy.config.num_experts)
    for _ in range(steps):
        record = economy.step()
        losses.append(record.loss)
        wins += torch.bincount(
            record.selected_experts.flatten(), minlength=economy.config.num_experts
        ).float()
    return sum(losses) / len(losses), wins / wins.sum()


def _steady(competence: torch.Tensor, seed: int, **overrides) -> tuple[SyntheticEconomy, float]:
    config = replace(BASE_CONFIG, **overrides) if overrides else BASE_CONFIG
    economy = SyntheticEconomy(competence, seed=seed, config=config)
    _window(economy, STEADY_STEPS - WINDOW)
    loss, _ = _window(economy, WINDOW)
    return economy, loss


def _floor_without(competence: torch.Tensor, expert: int, seed: int) -> float:
    """The loss a collective born without ``expert`` settles at: the re-formation target."""
    born_without = competence.clone()
    born_without[expert] = 0.0
    return _steady(born_without, seed)[1]


def _survivors_track_competence(share: torch.Tensor, competence: torch.Tensor, dead: int) -> float:
    keep = torch.tensor([index != dead for index in range(competence.numel())])
    return pearson(share[keep], competence[keep])


def _senesce(economy: SyntheticEconomy, expert: int) -> None:
    """The cell is still wired and still bids; it just stops contributing anything."""
    with torch.no_grad():
        economy.mob.experts[expert].down_adapter_B.weight.zero_()  # type: ignore[union-attr]


def _freeze_heads(economy: SyntheticEconomy) -> None:
    for group in economy.optimizer.param_groups:
        group["lr"] = 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_the_market_re_forms_around_a_senescent_expert(seed):
    """Undesigned damage the collective must *detect*: the best expert stops contributing.

    Nobody tells the auction. The dead cell keeps its wealth and its stale report,
    and at first wins more than before -- it costs it nothing. Its realised value
    is now zero, its head learns that, its bids fall, and the tokens flow to the
    next-most-competent experts: within 200 steps the loss sits at the floor a
    collective born without that expert reaches (measured 0.76-0.96x it on the
    three seeds -- below it, since the survivors have had 200 more steps than
    that collective), the dead cell holds under 1% of the slots (0.001-0.006),
    and the survivors' routing still tracks their competence (r 0.61-0.82).
    """
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    floor = _floor_without(competence, best, seed)
    economy, _ = _steady(competence, seed)

    _senesce(economy, best)
    _window(economy, DAMAGE_HORIZON - WINDOW)
    loss, share = _window(economy, WINDOW)

    assert loss <= RE_FORMATION_FACTOR * floor, (loss, floor)
    assert float(share[best]) <= DEAD_SHARE_CEILING, float(share[best])
    assert _survivors_track_competence(share, competence, best) > SURVIVOR_TRACKING_FLOOR


def test_frozen_heads_leave_the_senescent_expert_in_the_market():
    """The pairing: without the value objective only its draining wealth removes the dead cell.

    Measured at 200 steps after damage over the three seeds above: loss 1.21-1.51x
    the born-without floor and a dead share of 0.10-0.18, against 0.76-0.96x and
    under 0.01 with the heads learning.
    """
    seed = SEEDS[0]
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    floor = _floor_without(competence, best, seed)
    economy, _ = _steady(competence, seed)

    _senesce(economy, best)
    _freeze_heads(economy)
    _window(economy, DAMAGE_HORIZON - WINDOW)
    loss, share = _window(economy, WINDOW)

    assert loss > RE_FORMATION_FACTOR * floor
    assert float(share[best]) > DEAD_SHARE_CEILING


class _ForcedSubset(torch.nn.Module):
    """A gate that ignores every report and routes each token to a fixed subset of experts."""

    def __init__(self, subset: list[int], top_k: int, seed: int):
        super().__init__()
        self.subset = torch.tensor(subset)
        self.top_k = top_k
        # Its own stream, so the forcing does not move the economy's draws.
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, confidences: torch.Tensor, wealth: torch.Tensor) -> AuctionOutcome:
        batch, seq_len, num_experts = confidences.shape
        draws = torch.stack(
            [
                torch.randperm(len(self.subset), generator=self.generator)[: self.top_k]
                for _ in range(batch * seq_len)
            ]
        ).view(batch, seq_len, self.top_k)
        selected = self.subset[draws]
        weights = torch.full_like(confidences[..., : self.top_k], 1.0 / self.top_k)
        rebates = torch.zeros_like(confidences)
        return AuctionOutcome(selected, weights, torch.zeros_like(weights), rebates, None)


def _force_routing(economy: SyntheticEconomy, subset: list[int], steps: int, seed: int) -> None:
    """Route by fiat for ``steps`` steps; the forcing stream is a replicate of the seed too."""
    gate = economy.mob.gate
    economy.mob.gate = _ForcedSubset(subset, economy.config.top_k, seed=1000 * steps + seed)
    try:
        _window(economy, steps)
    finally:
        economy.mob.gate = gate


SHORT_FORCED_EPISODE = 50
LONG_FORCED_EPISODE = 150
RELEASE_HORIZON = 150
TRACKING_AFTER_RELEASE = 0.7
# The best expert has regained its standing when it holds this much of the share
# it held before the episode (0.30-0.43 of the slots, by seed).
REGAINED_SHARE = 0.8


def _release_and_measure(seed: int, episode: int) -> tuple[float, float, float, float]:
    """``(loss / steady loss, routing-competence correlation, best share / its steady share)``."""
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    least_competent = competence.argsort()[:3].tolist()
    economy = SyntheticEconomy(competence, seed=seed)
    _window(economy, STEADY_STEPS - WINDOW)
    steady_loss, steady_share = _window(economy, WINDOW)

    _force_routing(economy, least_competent, episode, seed)
    _window(economy, RELEASE_HORIZON - WINDOW)
    loss, share = _window(economy, WINDOW)
    return (
        loss / steady_loss,
        pearson(share, competence),
        float(share[best]) / float(steady_share[best]),
        float(steady_share[best]),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_market_re_forms_after_routing_was_forced_onto_the_least_competent(seed):
    """Undesigned damage to the allocation itself: fifty steps of routing by fiat.

    Every token goes to the three least competent experts, the loss quadruples, and
    the competent experts hold nothing. Released, the auction routes on the reports
    and wealth the experts still carry: within 150 steps routing tracks competence
    again (r 0.76-0.78 on the three seeds), the best expert holds its pre-episode
    share (1.18-1.33x it), and the loss is below its steady value (0.74-0.76x).
    """
    loss_ratio, tracking, regained, steady_share = _release_and_measure(seed, SHORT_FORCED_EPISODE)

    assert steady_share > 2.0 / DEFAULT_COMPETENCE.numel(), "the fixture must have a leader"
    assert tracking > TRACKING_AFTER_RELEASE, tracking
    assert regained > REGAINED_SHARE, regained
    assert loss_ratio <= 1.0, loss_ratio


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A 150-step episode drives the incumbents to 667-746 of the 750 wealth ceiling and "
        "the market does not re-form: at the 150-step horizon this test asserts, routing no "
        "longer tracks competence (r -0.31 to -0.03 on the three seeds, 0.81-0.82 before), "
        "the best expert holds 9-11% of its steady share and the loss is 2.6-3.7x steady; "
        "at 300 steps r -0.35 to +0.16, 0.2-1.4% and 1.9-4.3x. Two candidate causes: the "
        "ceiling, and the starved experts' heads receiving no realised-value signal during "
        "the episode; the ceiling measurement supports the first, nothing here excludes the "
        "second. The band is #16's."
    ),
)
def test_the_market_re_forms_after_a_long_forced_episode():
    """The boundary of the claim above, measured: recovery depends on the episode's length."""
    loss_ratio, tracking, regained, _ = _release_and_measure(SEEDS[0], LONG_FORCED_EPISODE)

    assert tracking > TRACKING_AFTER_RELEASE, tracking
    assert regained > REGAINED_SHARE, regained
    assert loss_ratio <= 1.0, loss_ratio


def _ruin(economy: SyntheticEconomy, expert: int) -> None:
    with torch.no_grad():
        economy.mob.expert_wealth[expert] = 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_the_economy_re_equilibrates_after_an_experts_wealth_is_zeroed(seed):
    """Undesigned damage to the ledger: the best expert's wealth is set to zero.

    Its bid is zero, so it wins nothing; the first settlement clamps it to the
    floor. The survivors absorb its tokens without collapse: within 200 steps the
    loss sits at the born-without floor, their routing tracks their competence,
    and every ledger stays inside the band.
    """
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    floor = _floor_without(competence, best, seed)
    economy, _ = _steady(competence, seed)

    _ruin(economy, best)
    _window(economy, DAMAGE_HORIZON - WINDOW)
    loss, share = _window(economy, WINDOW)

    assert loss <= RE_FORMATION_FACTOR * floor, (loss, floor)
    assert _survivors_track_competence(share, competence, best) > SURVIVOR_TRACKING_FLOOR
    wealth = economy.mob.expert_wealth
    assert torch.isfinite(wealth).all()
    assert wealth.min() >= BASE_CONFIG.min_wealth and wealth.max() <= BASE_CONFIG.max_wealth


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A ruined but competent expert does not come back: at the default exploration rate "
        "it holds about one token in 300, and what it earns there barely outpaces decay at "
        "the floor -- 33 credits against a median of 113 after 200 steps, win share 0.002 "
        "(in a probe on the same fixture: 34 with exploration off, 79 with decay off). Its "
        "return waits on #16."
    ),
)
def test_a_ruined_competent_expert_returns_to_the_market():
    seed = SEEDS[0]
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    economy, _ = _steady(competence, seed)

    _ruin(economy, best)
    _window(economy, DAMAGE_HORIZON - WINDOW)
    _, share = _window(economy, WINDOW)

    assert float(share[best]) > 1.0 / competence.numel()
    assert economy.mob.expert_wealth[best] > economy.mob.expert_wealth.median()
