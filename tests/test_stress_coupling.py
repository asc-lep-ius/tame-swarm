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

from measure_stress_coupling import Calibration, replay_magnitudes  # noqa: E402
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


def test_invariant_6_the_charge_lands_in_the_one_settlement_and_nowhere_else(monkeypatch):
    """The wealth update is exactly `settle`'s four terms, the charge among them.

    #40 exists so that every wealth path settles once; a charge that moved
    wealth anywhere else would be the fourth path that issue removed.

    **Spying on the terms, not backing them out of the wealth.** An earlier
    version of this test measured the inflow as
    `w_next - (w * decay + setpoint * (1 - decay))` and then reconstructed
    `w_next` from it, which telescopes: it held to 4e-16 on an arbitrary random
    path and held again with a deliberately wrong decay. A charge applied twice,
    applied per-winner, or applied outside the settlement would have been
    absorbed into the measured "inflow" and the assertion would still have
    passed. `measure_ledger_stability.py` avoids this by recording the inflow
    independently, and so does this: each term is captured as `settle` computes
    it, and the equation is checked against wealth the test never fed itself.
    """
    coupled = economy(stress_coupling=STRESS_SHARED, stress_lambda=0.01, stress_gamma=0.5)
    updater = coupled.mob.wealth_updater
    terms: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    real_settle = WealthUpdater.settle

    # Patched on the class: `WealthUpdater` is frozen, which is the property
    # that makes `settle` return the charge rather than store it.
    def spying(self, wealth: torch.Tensor, settlement, charge) -> float:
        before = wealth.clone()
        # The two signed pieces `settle` composes beside the relaxation: what
        # the reward paid, and what the auction charged. Recomputed from the
        # same settlement rather than differenced out of the ledger.
        paid = self.reward(wealth.clone(), settlement)
        priced = charge(
            settlement.payments,
            settlement.selected_experts,
            settlement.num_tokens,
            self.reward.multiplier,
            settlement.rebates,
            settlement.valid_mask,
        )
        stress = real_settle(self, wealth, settlement, charge)
        terms.append((before, paid - priced, stress))
        return stress

    monkeypatch.setattr(WealthUpdater, "settle", spying)
    for _ in range(6):
        coupled.step()

    assert terms, "the settlement never ran"
    assert any(stress > 0.0 for _, _, stress in terms)
    for index, (before, transfer, stress) in enumerate(terms):
        after = terms[index + 1][0] if index + 1 < len(terms) else coupled.mob.expert_wealth
        relaxed = before * updater.decay
        if updater.mode == LEDGER_SETPOINT:
            relaxed = relaxed + updater.rate * updater.setpoint
        # Every cell pays the same stress, winner or not: that is the invariant,
        # and it is why the charge is a scalar subtracted from the whole vector.
        assert torch.allclose(after, relaxed + transfer - stress, atol=1e-5), (
            f"step {index}: wealth moved by something other than "
            "relax + reward - price - stress, so the charge is not in the one settlement"
        )


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

    This pins the seam's arithmetic, which is true whatever scale the caller
    picks -- so it documents the mechanism and would *not* have caught the bug.
    `test_a_replayed_charge_round_trips_to_the_donors_own_total` is the one that
    would, because it reaches the caller's scaling.
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


def test_a_replayed_charge_round_trips_to_the_donors_own_total():
    """The scaling the control arm's whole validity rests on, at the caller's level.

    `read_control` divided a recorded per-step *total* by a price and handed it
    to a seam that reads a per-token *magnitude*, so the control arm paid
    `tokens` times the drain it was matching -- 32x here -- and sat at the
    wealth floor while `shared` did not. No test could have caught it: the
    arithmetic lived inside `read_control`, and nothing imports that module.

    Round-tripping is the whole check. Undo the donor's price and token count,
    then put them back, and the donor's own total must come out.
    """
    donor = Calibration(
        resting_sigma=0.04,
        resting_reading=0.18,
        setpoint=0.22,
        mean_stress=0.125,
        attributed_per_token=0.03,
        equal_budget_lambda=0.0008,
        tokens=32,
    )
    recorded = [0.1137, 0.2028, 0.0044]

    magnitudes = replay_magnitudes(recorded, donor)

    for charge, magnitude in zip(recorded, magnitudes, strict=True):
        assert magnitude * donor.equal_budget_lambda * donor.tokens == pytest.approx(charge)
    # The bug's own signature, so the guard is against a scale and not a typo:
    # dividing by the price alone leaves a magnitude `tokens` times too large.
    assert magnitudes[0] * donor.tokens == pytest.approx(recorded[0] / donor.equal_budget_lambda)
    # And a donor that charged nothing lends nothing rather than dividing by zero.
    assert replay_magnitudes(recorded, replace(donor, equal_budget_lambda=0.0)) == [0.0, 0.0, 0.0]


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
