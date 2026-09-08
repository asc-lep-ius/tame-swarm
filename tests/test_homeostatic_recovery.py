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

from economy_damage import (  # noqa: E402
    DAMAGE_HORIZON,
    DEAD_SHARE_CEILING,
    LONG_FORCED_EPISODE,
    RE_FORMATION_FACTOR,
    REGAINED_SHARE,
    SEEDS,
    SHORT_FORCED_EPISODE,
    SURVIVOR_TRACKING_FLOOR,
    TRACKING_AFTER_RELEASE,
    WINDOW,
    floor_without,
    freeze_heads,
    release_and_measure,
    ruin,
    ruin_and_measure,
    senesce,
    steady,
    survivors_track_competence,
    window,
)
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    shuffled,
)

from homeostat_calibration import ConsensusWeighting  # noqa: E402

from .auction_mutations import flat_band  # noqa: E402
from .wired_system import (  # noqa: E402
    ACTUATORS,
    BELOW_ACTUATORS,
    READOUT,
    WiredSystem,
    build_wired_system,
)

# --- The steering tissue ---------------------------------------------------------

# #6's acceptance criterion: the tissue's consensus error within this fraction of
# the tissue setpoint, within this many forward passes of the disturbance.
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
    it can answer it. The regulated variable is the tissue's consensus error over
    its live cells, each weighted by its gain -- what the shared integrator drives
    -- and it returns to within 5% of the tissue setpoint well inside the 200-pass
    budget (measured: inside the band from pass 22, coupled or not) while the
    actuators' strength rises to carry the deficit. With the coupling live the
    routing perceives the same direction; the tissue's recovery does not depend on
    it either way.
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


def _consensus_weighting(system, weighting: ConsensusWeighting) -> None:
    """Re-weight the consensus: ``uniform`` is the rule before #21, ``calibrated`` before #22."""
    tissue = system.tissue()
    assert tissue.calibration is not None
    tissue.calibration = replace(tissue.calibration, weighting=weighting)
    # The gains derive from the calibration's gain, so the controller is rebuilt the
    # way set_gains rebuilds it; the plain mean's gains are the ones #4 recorded.
    tissue.controller.config = tissue._pid_config()


def _blind_and_others(system) -> tuple[dict, list[dict]]:
    cells = system.tissue().status()["cells"]
    return cells[0], cells[1:]


def test_content_below_the_bottom_actuator_leaves_the_regulable_cells_at_their_own_setpoints():
    """A deficit no cell can act on (#21): the blind cell reports it and nobody pays for it.

    Content that enters below every actuator is read by the bottom cell too, and
    nothing can correct it there. Its weight in the consensus is its gain, zero, so
    its error stands (+167 sigma, a quarter of the tissue setpoint) and is reported,
    while the shared integrator regulates the cells that can be moved: each of the
    four settles within 5% of its *own* setpoint (measured: +2.7%, 0.0%, -0.5%,
    -0.6%), not only the tissue mean. The consensus itself is barely disturbed by
    construction -- the blind cell no longer votes, so it is in band from the first
    pass -- and the per-cell assertion is the property. The test below is the
    pairing this replaced.
    """
    system = build_wired_system()
    system.run(SETTLE_PASSES)
    system.set_content("truthful", DEAD_CELL_DEFICIT, layer=BELOW_ACTUATORS)

    assert _passes_to_recover(system) is not None
    blind, others = _blind_and_others(system)
    assert blind["weight"] == 0.0
    assert blind["error"] > RECOVERY_FRACTION * system.setpoint(), (
        "the bottom cell stays in deficit"
    )
    assert all(abs(cell["error"]) <= RECOVERY_FRACTION * cell["setpoint"] for cell in others)


def test_the_uniform_consensus_dilutes_the_blind_deficit_over_the_live_cells():
    """The pairing: counted as one live cell of five, the blind error is zeroed by the other four.

    The shared integrator drives the plain mean to zero regardless, so the regulable
    cells settle past their setpoints by the blind cell's error divided by the
    number of other live cells -- one quarter here, one seventh on the served
    eight-cell tissue -- and the tissue meets its criterion on the mean while no
    cell is at its own setpoint (measured: -12.4%, -6.9%, -5.7%, -5.1%).
    """
    system = build_wired_system()
    _consensus_weighting(system, "uniform")
    system.run(SETTLE_PASSES)
    system.set_content("truthful", DEAD_CELL_DEFICIT, layer=BELOW_ACTUATORS)

    assert _passes_to_recover(system) is not None
    blind, others = _blind_and_others(system)
    errors = [cell["error"] for cell in others]
    assert sum(errors) / len(errors) == pytest.approx(-blind["error"] / len(others), rel=0.05)
    assert any(abs(cell["error"]) > RECOVERY_FRACTION * cell["setpoint"] for cell in others)


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
    damage (measured: within 1.4% of its setpoint, against the 2% allowed). The
    consensus error peaks near 17% of the setpoint at pass 14 and is back inside
    the 5% band by pass 38.
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


SURVIVORS_OF_THE_BOTTOM = (*ACTUATORS[2:], READOUT)


def _remove_bottom_actuator(
    weighting: ConsensusWeighting = "gain",
) -> tuple[WiredSystem, int | None, dict[int, float], dict[int, dict]]:
    """Settle under load, then remove the bottom actuator.

    The system, the pass at which its consensus was back in band (or None), the
    actuators' strengths before the damage, and the per-cell status after it.
    """
    system = build_wired_system()
    _consensus_weighting(system, weighting)
    system.run(SETTLE_PASSES)
    system.set_content("truthful", CONTENT_DEFICIT)
    assert _passes_to_recover(system) is not None
    strengths_before = dict(system.tissue()._strength)
    system.kill_actuator(ACTUATORS[0])
    recovered_at = _passes_to_recover(system)
    cells = {cell["layer"]: cell for cell in system.tissue().status()["cells"]}
    return system, recovered_at, strengths_before, cells


def _survivors_consensus(cells: dict[int, dict]) -> tuple[float, float]:
    """The cells above the new bottom cell: their gain-weighted mean error and squared-error sum."""
    survivors = [cells[layer] for layer in SURVIVORS_OF_THE_BOTTOM]
    weights = sum(cell["weight"] for cell in survivors)
    weighted = sum(cell["weight"] * cell["error"] for cell in survivors) / weights
    return weighted, sum(cell["error"] ** 2 for cell in survivors)


def test_removing_the_bottom_actuator_blinds_the_cell_above_it_and_the_survivors_hold_consensus():
    """Undesigned damage that moves the bottom of the tissue (#22).

    With actuator 1 gone nothing live injects below cell 2: it reads content alone,
    its error is its whole setpoint plus the deficit (+615 sigma, reported), and no
    action can change it. Its weight follows the live tissue below it, zero, so the
    shared integrator regulates the three cells it can still move: their
    gain-weighted consensus -- which is the tissue error once cell 2 weighs
    nothing, so the recovery helper asserts it -- is back inside the 5% band at
    pass 24 and ends at 0.01% of the setpoint, every surviving actuator raises its
    strength, and the tissue setpoint is theirs (688 sigma against the calibrated
    641, with ``error`` still ``setpoint - process_variable``). The survivors do
    not each return to their own setpoints (measured +19.8%, -2.7%, -9.6%): the
    removal changed every survivor's real gain, unevenly, so one common strength
    can no longer reach all three, and what remains is the least-squares residual
    of the damaged plant.
    The pairing below shows the part of that residual the rule removes.
    """
    system, recovered_at, strengths_before, cells = _remove_bottom_actuator()
    tissue = system.tissue()
    new_bottom = cells[ACTUATORS[1]]

    assert cells[ACTUATORS[0]]["alive"] is False
    assert new_bottom["alive"] and new_bottom["weight"] == 0.0
    assert new_bottom["error"] > new_bottom["setpoint"], "its whole setpoint plus the deficit"
    assert recovered_at is not None and recovered_at <= RECOVERY_PASSES, recovered_at
    assert all(tissue._strength[layer] > strengths_before[layer] for layer in ACTUATORS[1:])

    status = tissue.status()
    survivors = [cells[layer] for layer in SURVIVORS_OF_THE_BOTTOM]
    survivors_setpoint = sum(c["weight"] * c["setpoint"] for c in survivors) / sum(
        c["weight"] for c in survivors
    )
    assert status["setpoint"] == pytest.approx(survivors_setpoint)
    assert status["setpoint"] != pytest.approx(tissue.nominal_setpoint)
    assert status["setpoint"] - status["process_variable"] == pytest.approx(status["error"])


def test_the_calibrated_weighting_dilutes_the_new_bottom_cells_error_over_the_survivors():
    """The pairing: cell 2 keeps its calibrated weight and votes on an error nobody can fix.

    The shared integrator zeroes a consensus that includes it, so the survivors'
    own consensus is pushed past setpoint by 2's weighted error over their weights
    -- 19% of the tissue setpoint, the #21 dilution reappearing one cell up -- and
    the tissue reports itself in band (from pass 39, against 24 under the live
    rule) while the cells it can move are not. Their squared-error sum is 3.4x the
    live rule's: the fixed point is the least-squares common strength for a set
    that includes a cell with no gain at all.
    """
    _, live_recovered_at, _, live_cells = _remove_bottom_actuator("gain")
    system, recovered_at, _, cells = _remove_bottom_actuator("calibrated")
    new_bottom = cells[ACTUATORS[1]]
    survivor_weights = sum(cells[layer]["weight"] for layer in SURVIVORS_OF_THE_BOTTOM)

    assert new_bottom["weight"] > 0.0
    assert recovered_at is not None
    weighted, squared = _survivors_consensus(cells)
    dilution = -new_bottom["weight"] * new_bottom["error"] / survivor_weights
    assert weighted == pytest.approx(dilution, rel=0.02)
    assert abs(weighted) > RECOVERY_FRACTION * system.setpoint()
    _, live_squared = _survivors_consensus(live_cells)
    assert squared > 2 * live_squared, (squared, live_squared)
    assert live_recovered_at is not None and live_recovered_at < recovered_at


# --- The expert economy ----------------------------------------------------------
#
# The protocols themselves live in ``scripts/economy_damage.py`` so that #16's
# wealth-bound sweep runs the same damage this file asserts on. The two expected
# failures below are that sweep's acceptance criterion; a sweep with its own copy
# of the protocol could report a flip the suite does not see.


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
    floor = floor_without(competence, best, seed)
    economy, _ = steady(competence, seed)

    senesce(economy, best)
    window(economy, DAMAGE_HORIZON - WINDOW)
    loss, share = window(economy, WINDOW)

    assert loss <= RE_FORMATION_FACTOR * floor, (loss, floor)
    assert float(share[best]) <= DEAD_SHARE_CEILING, float(share[best])
    assert survivors_track_competence(share, competence, best) > SURVIVOR_TRACKING_FLOOR


def test_frozen_heads_leave_the_senescent_expert_in_the_market():
    """The pairing: without the value objective only its draining wealth removes the dead cell.

    Measured at 200 steps after damage over the three seeds above: loss 1.21-1.51x
    the born-without floor and a dead share of 0.10-0.18, against 0.76-0.96x and
    under 0.01 with the heads learning.
    """
    seed = SEEDS[0]
    competence = shuffled(DEFAULT_COMPETENCE, seed)
    best = int(competence.argmax())
    floor = floor_without(competence, best, seed)
    economy, _ = steady(competence, seed)

    senesce(economy, best)
    freeze_heads(economy)
    window(economy, DAMAGE_HORIZON - WINDOW)
    loss, share = window(economy, WINDOW)

    assert loss > RE_FORMATION_FACTOR * floor
    assert float(share[best]) > DEAD_SHARE_CEILING


@pytest.mark.parametrize("seed", SEEDS)
def test_the_market_re_forms_after_routing_was_forced_onto_the_least_competent(seed):
    """Undesigned damage to the allocation itself: fifty steps of routing by fiat.

    Every token goes to the three least competent experts, the loss quadruples, and
    the competent experts hold nothing. Released, the auction routes on the reports
    and wealth the experts still carry: within 150 steps routing tracks competence
    again (r 0.76-0.78 on the three seeds), the best expert holds its pre-episode
    share (1.18-1.33x it), and the loss is below its steady value (0.74-0.76x).
    """
    loss_ratio, tracking, regained, steady_share = release_and_measure(seed, SHORT_FORCED_EPISODE)

    assert steady_share > 2.0 / DEFAULT_COMPETENCE.numel(), "the fixture must have a leader"
    assert tracking > TRACKING_AFTER_RELEASE, tracking
    assert regained > REGAINED_SHARE, regained
    assert loss_ratio <= 1.0, loss_ratio


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A 150-step episode drives the incumbents to the wealth ceiling and the market does "
        "not re-form: routing no longer tracks competence (r -0.31 to -0.03 on the three "
        "seeds, 0.81-0.82 before), the best expert holds 9-11% of its steady share and the "
        "loss is 2.6-3.7x steady. #16 swept the band and settled which of the two candidate "
        "causes it is: NOT the band. With wealth pinned flat -- selection on reports alone, "
        "the auction still pricing -- the same episode still fails, and at this seed "
        "recovers 0.37 of the steady share against 0.09 and 1.66x the loss against 2.64x "
        "while the rank correlation does not improve at all (-0.20 against -0.03); the "
        "companion test below pins that. So the band carries part of the damage and no band "
        "in the swept space flips this test. What remains is the cause the old reason could "
        "not exclude: a head is trained only on the value its own expert realises, so 150 "
        "steps of holding no tokens leaves it with a stale report, and the uniform 2% "
        "exploration slot re-samples it no faster for having been starved longer. Waits on "
        "the count-based exploration in #26."
    ),
)
def test_the_market_re_forms_after_a_long_forced_episode():
    """The boundary of the claim above, measured: recovery depends on the episode's length."""
    loss_ratio, tracking, regained, _ = release_and_measure(SEEDS[0], LONG_FORCED_EPISODE)

    assert tracking > TRACKING_AFTER_RELEASE, tracking
    assert regained > REGAINED_SHARE, regained
    assert loss_ratio <= 1.0, loss_ratio


def test_the_long_episode_does_not_recover_with_the_ledger_pinned_flat():
    """#16's decomposition: how much of the damage above belongs to the wealth band.

    ``flat_band`` pins ``min == max == initial``, so a constant multiplier cannot
    reorder ``confidence x wealth`` and selection is the report ranking alone --
    while the auction still prices, unlike ``wealth_blind_forward``, which zeroes
    every payment and rebate too. That isolates the band rather than the whole
    mechanism, which is what makes this an attribution and not just a second
    failure.

    Two of the three statistics improve a lot and the third does not move the way
    a band explanation predicts: the best expert recovers 0.37 of its steady share
    against 0.09, the loss falls to 1.66x steady from 2.64x, and the rank
    correlation over eight experts -- the noisiest of the three -- goes the other
    way, -0.03 to -0.20. Recovery is markedly better with the ledger out of the
    way and still nowhere near any of the three thresholds, which is why the
    expected failure above no longer names the band as what it waits on.
    """
    banded_loss, _, banded_regained, _ = release_and_measure(SEEDS[0], LONG_FORCED_EPISODE)
    loss_ratio, tracking, regained, _ = release_and_measure(
        SEEDS[0], LONG_FORCED_EPISODE, flat_band(BASE_CONFIG)
    )

    assert loss_ratio < banded_loss, (loss_ratio, banded_loss)
    assert regained > banded_regained, (regained, banded_regained)

    assert loss_ratio > 1.0, loss_ratio
    assert tracking < TRACKING_AFTER_RELEASE, tracking
    assert regained < REGAINED_SHARE, regained


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
    floor = floor_without(competence, best, seed)
    economy, _ = steady(competence, seed)

    ruin(economy, best)
    window(economy, DAMAGE_HORIZON - WINDOW)
    loss, share = window(economy, WINDOW)

    assert loss <= RE_FORMATION_FACTOR * floor, (loss, floor)
    assert survivors_track_competence(share, competence, best) > SURVIVOR_TRACKING_FLOOR
    wealth = economy.mob.expert_wealth
    assert torch.isfinite(wealth).all()
    assert wealth.min() >= BASE_CONFIG.min_wealth and wealth.max() <= BASE_CONFIG.max_wealth


@pytest.fixture(scope="module")
def ruined():
    """One ruin protocol, read by the two claims below, which used to be one test."""
    return ruin_and_measure(SEEDS[0])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A ruined but competent expert does not come back: at the default exploration rate "
        "it holds about one token in 300, and what it earns there barely outpaces decay at "
        "the floor -- win share 0.002 after 200 steps (in a probe on the same fixture: 34 "
        "credits with exploration off, 79 with decay off). #16 measured the band's part in "
        "it: the share only clears chance with wealth pinned flat (0.215), and is already "
        "back to 0.063 at a band ratio of 4, so no band that leaves the ledger doing "
        "anything fixes this. Waits on the count-based exploration in #26."
    ),
)
def test_a_ruined_competent_expert_returns_to_the_market(ruined):
    """The claim this test is named for: it wins tokens again, at better than chance."""
    share, _, _ = ruined

    assert share > 1.0 / DEFAULT_COMPETENCE.numel()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The strictly stronger claim, split out of the test above by #16: not merely back in "
        "the market but back to its standing, out-earning four of the other seven within 200 "
        "steps. It reads 0.29 of the median wealth at the shipped band. Kept separate because "
        "it is the half a band can move -- it rises to 0.81 at decay 0.99 and reaches 1.00 "
        "with wealth flat, where it is unsatisfiable by construction rather than by the "
        "economy, every wealth being identical. Waits on #26 with the claim above."
    ),
)
def test_a_ruined_competent_expert_regains_its_standing(ruined):
    _, wealth, median = ruined

    assert wealth > median
