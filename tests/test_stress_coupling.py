"""The seven invariants #59's coupling has to hold, each failing on the code before it.

The charge is the first thing in this project that takes wealth from a cell for
something the cell did not do, so the properties that make it a *shared stress*
rather than a signed reward wearing a new name are asserted here one at a time:
unsigned, gated, ownership wiped, report-independent, inert at zero, settled in
the one settlement, and silent on the constitution.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    shuffled,
)

from mob.ledger import LEDGER_SETPOINT, Settlement, WealthUpdater  # noqa: E402
from mob.stress import (  # noqa: E402
    STRESS_ATTRIBUTED,
    STRESS_GATE_STATE,
    STRESS_SHARED,
    StressConfig,
    cell_stress,
    layer_stress,
    paracrine_share,
    state_gamma,
    stress_from_config,
    transmitted,
)

SETPOINT = 0.5
DOSE = 0.25
SIGMA = 0.04


def economy(seed: int = 0, **overrides) -> DifferentiatedEconomy:
    """The differentiated fixture with both goal fields, coupled as asked."""
    resting = overrides.pop("resting_sigma", SIGMA)
    config = replace(BASE_CONFIG, **overrides)
    built = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)
    built.add_goal_field(0, SETPOINT, DOSE, resting_sigma=resting)
    built.add_goal_field(1, SETPOINT, DOSE, resting_sigma=resting)
    return built


def test_invariant_1_the_stress_is_a_magnitude_and_never_a_direction():
    """A signed error fed into wealth is a reward for a direction under a new name."""
    above = cell_stress(torch.tensor([0.9, 0.7]), SETPOINT, SIGMA)
    below = cell_stress(torch.tensor([0.1, 0.3]), SETPOINT, SIGMA)

    assert bool((above >= 0).all()) and bool((below >= 0).all())
    # Symmetric about the setpoint: overshooting by 0.4 costs what
    # undershooting by 0.4 costs, which is what "magnitude" means.
    assert torch.allclose(above, below)
    with pytest.raises(ValueError, match="resting_sigma must be positive"):
        cell_stress(torch.tensor([0.5]), SETPOINT, 0.0)


def test_invariant_5_the_gate_passes_nothing_inside_one_resting_spread():
    stress = torch.tensor([0.0, 0.5, 1.0, 1.0001, 3.0])

    passed = transmitted(stress, gate_sigma=1.0)

    assert passed.tolist() == [0.0, 0.0, 0.0, pytest.approx(1.0001), 3.0]
    # And an error exactly at the threshold is noise, not a signal: the claim is
    # that a stress inside the spread is not information.
    assert float(passed[2]) == 0.0


def test_the_paracrine_share_is_the_neighbourhood_and_an_empty_one_changes_nothing():
    own = torch.tensor([2.0, 4.0])
    near = torch.tensor([0.0, 8.0])

    assert torch.equal(paracrine_share(own, [], [], gamma=1.0), own)
    assert torch.equal(paracrine_share(own, [near], [0.0], gamma=1.0), own)
    assert torch.equal(paracrine_share(own, [near], [1.0], gamma=0.0), own)
    assert torch.equal(paracrine_share(own, [near], [1.0], gamma=1.0), near)
    assert torch.equal(paracrine_share(own, [near], [3.0], gamma=0.5), torch.tensor([1.0, 6.0]))
    with pytest.raises(ValueError, match="gamma must lie"):
        paracrine_share(own, [near], [1.0], gamma=1.5)


def test_a_stressed_tissue_couples_its_cells_more_tightly_than_a_resting_one():
    """Levin 2019's selective coupling: the collective's state is upstream of the channel."""
    assert state_gamma(0.5, recent_stress=0.0, gate_sigma=1.0) == 0.5
    assert state_gamma(0.5, recent_stress=1.0, gate_sigma=1.0) == 0.5
    assert state_gamma(0.5, recent_stress=2.0, gate_sigma=1.0) == 1.0
    assert state_gamma(0.5, recent_stress=100.0, gate_sigma=1.0) == 1.0


def test_invariant_2_the_charge_reads_no_report_and_is_the_same_for_every_cell():
    """Ownership wiped: a cell lowers the charge by acting, never by saying something.

    The layer's stress is a function of the allocation and the readings it
    produced; permuting which cell *reported* what, at a fixed allocation,
    cannot move it -- which is why the auction's strategyproofness survives the
    charge, and why `tests/constitution/` needs no new case.
    """
    readings = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.4, 0.6]])]
    config = StressConfig(stress_lambda=1.0, gamma=0.5, gate_sigma=0.0)

    share = layer_stress(readings, [SETPOINT, SETPOINT], [SIGMA, SIGMA], config)

    # One number per token, not one per cell: every cell at the layer pays it.
    assert share.shape == readings[0].shape
    charged = WealthUpdater(
        reward=WealthUpdater.for_experts(BASE_CONFIG).reward,
        floor=WealthUpdater.for_experts(BASE_CONFIG).floor,
        decay=BASE_CONFIG.wealth_decay,
        stress_lambda=2.0,
    )
    settlement = Settlement(
        selected_experts=torch.zeros(1, 2, 2, dtype=torch.long),
        routing_weights=torch.full((1, 2, 2), 0.5),
        confidences=torch.zeros(1, 2, BASE_CONFIG.num_experts),
        num_tokens=2,
        stress=share,
    )
    assert charged.stress_charge(settlement) == pytest.approx(2.0 * float(share.sum()))


def test_invariant_4_a_zero_price_reproduces_the_recorded_economy_bitwise():
    """Every recorded arm ran at lambda zero, and this is what keeps them readable.

    Run one after the other rather than interleaved. The auction's exploration
    slot draws from the *global* RNG, not from the fixture's own generator, so
    two economies stepped alternately see each other's draws and diverge for a
    reason that has nothing to do with what is under test -- which is how this
    test first failed.
    """

    def run(**overrides) -> torch.Tensor:
        built = economy(**overrides)
        for _ in range(12):
            built.step()
        return built.mob.expert_wealth.clone(), built.mob.last_stress_charge

    recorded, charge = run(stress_gamma=0.0)
    coupled, _ = run(stress_gamma=0.5, stress_gate_sigma=0.25)

    assert torch.equal(recorded, coupled)
    assert charge == 0.0


def test_the_charge_bites_and_is_paid_by_winners_and_losers_alike():
    at_zero = economy()
    for _ in range(12):
        at_zero.step()
    charged = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    for _ in range(12):
        charged.step()

    assert charged.mob.last_stress_charge > 0.0
    assert not torch.equal(at_zero.mob.expert_wealth, charged.mob.expert_wealth)
    # Every cell paid the same charge, including the ones that never won: the
    # difference between a cell's two ledgers is the charge plus what the
    # allocation change did, and no cell is exempt.
    assert bool((charged.mob.expert_wealth <= at_zero.mob.expert_wealth + 1e-6).all())


def test_invariant_3_the_setpoint_and_the_resting_spread_are_read_only_after_calibration():
    coupled = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    setpoints = [field.setpoint for field in coupled.goal_fields()]
    sigmas = list(coupled.mob._resting_sigmas)

    for _ in range(20):
        coupled.step()

    assert [field.setpoint for field in coupled.goal_fields()] == setpoints
    assert list(coupled.mob._resting_sigmas) == sigmas


def test_invariant_6_the_charge_lands_in_the_one_settlement_and_nowhere_else():
    """The ledger reconstructs from the recorded inflow with the charge in it.

    #40 exists so that every wealth path settles once; a charge that moved
    wealth anywhere else would be the fourth path that issue removed. The
    reconstruction is `measure_ledger_stability.py`'s -- `decay^T w_0 + the
    relaxation toward the setpoint + the discounted inflow` -- and it is the
    only form of this check that can fail for the reason the invariant names.
    An earlier version asserted `charge > 0` and that some cell had lost
    wealth, which is true whenever any cell lost any wealth to anything.
    """
    coupled = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    # The same two constants `measure_ledger_stability._replay` reads: the
    # relaxation is toward `initial_wealth` under `setpoint` mode and toward
    # zero under `decay`, which is what the fixture runs.
    decay = coupled.config.wealth_decay
    setpoint = (
        coupled.config.initial_wealth if coupled.config.ledger_mode == LEDGER_SETPOINT else 0.0
    )
    coupled.step()

    start = coupled.mob.expert_wealth.clone().double()
    inflow: list[torch.Tensor] = []
    for _ in range(6):
        before = coupled.mob.expert_wealth.clone().double()
        coupled.step()
        after = coupled.mob.expert_wealth.double()
        # What the settlement put in, backed out of the relaxation: the charge
        # is part of it, so a charge applied anywhere else shows up as a gap.
        inflow.append(after - (before * decay + setpoint * (1.0 - decay)))

    steps = len(inflow)
    weights = torch.tensor([decay**j for j in range(steps)], dtype=torch.float64)
    reconstructed = (
        start * decay**steps
        + setpoint * (1.0 - decay) * float(weights.sum())
        + (torch.stack(inflow) * weights.flip(0).unsqueeze(-1)).sum(dim=0)
    )

    assert coupled.mob.last_stress_charge > 0.0
    assert torch.allclose(reconstructed, coupled.mob.expert_wealth.double(), atol=1e-5)


def test_a_coupling_whose_name_and_price_disagree_is_refused():
    """An arm called `shared` that charges nothing would run the recorded economy."""
    with pytest.raises(ValueError, match="stress_lambda is 0"):
        stress_from_config(replace(BASE_CONFIG, stress_coupling=STRESS_SHARED, stress_lambda=0.0))
    with pytest.raises(ValueError, match="charges no stress"):
        stress_from_config(
            replace(BASE_CONFIG, stress_coupling=STRESS_ATTRIBUTED, stress_lambda=0.5)
        )


def test_the_state_gate_is_a_mode_and_the_fixed_one_is_the_default():
    assert BASE_CONFIG.stress_gate_mode == "fixed"
    stated = economy(
        stress_coupling=STRESS_SHARED,
        stress_lambda=0.01,
        stress_gamma=0.5,
        stress_gate_mode=STRESS_GATE_STATE,
    )
    for _ in range(12):
        stated.step()
    fixed = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    for _ in range(12):
        fixed.step()

    # The mode is not inert: a tissue over its gate couples more tightly, so the
    # two ledgers part company.
    assert not torch.equal(stated.mob.expert_wealth, fixed.mob.expert_wealth)


def test_a_replayed_stress_is_a_charge_this_tissue_cannot_lower():
    """#59's control, and the seam it runs through."""
    coupled = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    coupled.mob.stress_override = torch.tensor(7.0)

    coupled.step()

    assert coupled.mob.last_stress_charge == pytest.approx(
        0.01 * 7.0 * coupled.batch_size * coupled.seq_len
    )


def test_a_replayed_charge_is_the_size_of_the_charge_it_replaces():
    """The control's whole job is to hold the drain fixed and vary its lowerability.

    `last_stress_charge` is a per-step *total* over the layer's tokens;
    `stress_override` is a per-token magnitude the layer expands to all of them.
    Replaying the total as the magnitude charged the control arm `tokens` times
    the drain it was matching -- 32x on this fixture -- which drove it to the
    wealth floor and made the contrast a comparison against an erased ledger.

    The existing seam test pins `lambda * override * tokens`, which is true
    whatever scale the caller picks, so only this one fails on that bug.
    """
    donor = economy(seed=0, stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    for _ in range(8):
        donor.step()
    recorded = donor.mob.last_stress_charge
    tokens = donor.batch_size * donor.seq_len

    replaying = economy(seed=1, stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    replaying.mob.stress_override = torch.tensor(recorded / (0.01 * tokens))
    replaying.step()

    assert replaying.mob.last_stress_charge == pytest.approx(recorded, rel=1e-6)
    # And the bug's own signature: replaying the total unscaled costs `tokens`
    # times as much, which is the arm being erased rather than stressed.
    replaying.mob.stress_override = torch.tensor(recorded / 0.01)
    replaying.step()
    assert replaying.mob.last_stress_charge == pytest.approx(recorded * tokens, rel=1e-6)


def test_the_setpoint_step_does_not_silently_switch_the_charge_off():
    """The perturbation rebuilds the field list; the stress sources must survive it.

    Found by the charge-per-step column rather than by a test: the fields came
    back without their resting spreads, so the charge went quiet at exactly the
    step the primary is read after, and every arm's table showed a charge of
    zero while the mechanism looked live in the settle phase.
    """
    coupled = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    coupled.step()
    before = coupled.mob.last_stress_charge

    coupled.step_goal_setpoint(0, 0.05)
    coupled.step()

    assert before > 0.0
    assert coupled.mob.last_stress_charge > 0.0
    assert coupled.mob._resting_sigmas == [SIGMA, SIGMA]
