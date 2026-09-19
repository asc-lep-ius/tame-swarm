"""The stakes dial (#39): does a cell's continuation depend on its realised value?

``persistence_coupling`` is the one field the three stakes arms differ in.
``value`` is the economy as recorded; ``decoupled`` pins the wealth the gate reads
and draws re-entry uniformly while the ledger settles on as a shadow; ``shuffled``
keeps the economy live and hands every head another expert's regression targets.
The goal term (#33, ``mob/goal.py``) is the ``value`` arm's value definition once a
goal field is attached, and inert until one is. Each test here pairs the
mechanism with the state it fails in, in the manner of ``test_no_silent_noops``.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    DifferentiatedEconomy,
    SyntheticEconomy,
    shuffled,
)

from mob import (  # noqa: E402
    PERSISTENCE_DECOUPLED,
    PERSISTENCE_SHUFFLED,
    PERSISTENCE_VALUE,
    ConstantGoalField,
    MixtureOfBidders,
    MoBConfig,
    goal_error_reduction,
    goal_terms,
)
from parity import ParityError, assert_parity  # noqa: E402

from .arm_fingerprints import BASE  # noqa: E402

SMALL = MoBConfig(hidden_dim=16, intermediate_dim=32, adapter_rank=4, adapter_alpha=4.0)


def _contributing_layer(config: MoBConfig, seed: int = 3) -> MixtureOfBidders:
    torch.manual_seed(seed)
    mob = MixtureOfBidders(config)
    mob.train()
    with torch.no_grad():
        for name, param in mob.experts.named_parameters():
            if name.endswith("_B.weight"):
                param.normal_(std=0.1)
    return mob


def _settle(mob: MixtureOfBidders, hidden: torch.Tensor, seed: int, scale: float = 1.0):
    """One forward, a fixed loss gradient at the output, and the settlement."""
    torch.manual_seed(seed)
    output = mob(hidden)
    gradient = torch.randn_like(output)
    (output * gradient).sum().backward()
    mob.update_wealth_from_loss(torch.ones(hidden.shape[:2]), loss_gradient_scale=scale)
    assert mob.last_realised_values is not None
    return mob.last_realised_values.clone()


# --- The dial is a config field with three settings ------------------------------------


def test_the_dial_refuses_a_setting_it_does_not_know():
    with pytest.raises(ValueError, match="persistence coupling"):
        MoBConfig(persistence_coupling="pinned")
    with pytest.raises(ValueError, match="at least two experts"):
        MoBConfig(num_experts=1, top_k=1, persistence_coupling=PERSISTENCE_SHUFFLED)


class _RecordingGate(nn.Module):
    """Wraps the auctioneer and keeps what the layer handed it."""

    def __init__(self, gate: nn.Module):
        super().__init__()
        self.gate = gate
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def forward(self, confidences, wealth, staleness=None):
        self.calls.append((wealth.clone(), staleness))
        return self.gate(confidences, wealth, staleness=staleness)


@pytest.mark.parametrize("coupling", [PERSISTENCE_VALUE, PERSISTENCE_SHUFFLED])
def test_a_live_arm_gates_on_the_ledger_and_the_staleness(coupling):
    mob = MixtureOfBidders(replace(SMALL, persistence_coupling=coupling))
    mob.train()
    with torch.no_grad():
        mob.expert_wealth.copy_(torch.linspace(20.0, 300.0, SMALL.num_experts))
        mob.expert_steps_since_held.fill_(7.0)
    recorder = _RecordingGate(mob.gate)
    mob.gate = recorder

    mob(torch.randn(1, 4, SMALL.hidden_dim))

    wealth, staleness = recorder.calls[0]
    assert torch.equal(wealth, mob.expert_wealth)
    assert staleness is not None and torch.equal(staleness, mob.expert_steps_since_held)


def test_the_decoupled_gate_reads_a_pinned_wealth_and_draws_uniformly():
    """Nothing the ledger records reaches the allocation: not the wealth, not the staleness."""
    mob = MixtureOfBidders(replace(SMALL, persistence_coupling=PERSISTENCE_DECOUPLED))
    mob.train()
    with torch.no_grad():
        mob.expert_wealth.copy_(torch.linspace(20.0, 300.0, SMALL.num_experts))
        mob.expert_steps_since_held.fill_(7.0)
    recorder = _RecordingGate(mob.gate)
    mob.gate = recorder

    mob(torch.randn(1, 4, SMALL.hidden_dim))

    wealth, staleness = recorder.calls[0]
    assert torch.equal(wealth, torch.full((SMALL.num_experts,), SMALL.initial_wealth))
    assert staleness is None
    assert not torch.equal(mob.expert_wealth, wealth), "the shadow ledger was overwritten"


# --- The shadow ledger -------------------------------------------------------------------


def _ledger_after(coupling: str, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The quality fixture at seed 0 under ``coupling``: its ledger and the gate's wealth."""
    config = replace(BASE_CONFIG, persistence_coupling=coupling)
    economy = SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0, config=config)
    for _ in range(steps):
        economy.step()
    return economy.mob.expert_wealth.clone(), economy.mob.allocation_wealth().clone()


def test_the_shadow_ledger_is_the_live_ledger_at_step_zero_and_not_after():
    """The guardrail the preregistration names: identical at step 0, diverging after.

    At step 0 every ledger holds ``initial_wealth``, prices are ratios, and the
    uniform draw is the staleness draw at zero staleness, so the decoupled arm's
    first settlement is bitwise the live arm's. From the second auction the live
    arm gates on a ledger that has moved and the decoupled arm does not, so the
    two ledgers part -- which is what makes the first assertion a check and not a
    tautology.
    """
    initial = torch.full((DEFAULT_COMPETENCE.numel(),), BASE_CONFIG.initial_wealth)
    live_1, _ = _ledger_after(PERSISTENCE_VALUE, 1)
    shadow_1, pinned_1 = _ledger_after(PERSISTENCE_DECOUPLED, 1)

    assert torch.equal(live_1, shadow_1)
    assert not torch.equal(live_1, initial), "the first settlement moved nothing"
    assert torch.equal(pinned_1, initial)

    live_40, _ = _ledger_after(PERSISTENCE_VALUE, 40)
    shadow_40, pinned_40 = _ledger_after(PERSISTENCE_DECOUPLED, 40)

    assert not torch.equal(live_40, shadow_40), "the dial changed no allocation in 40 steps"
    assert torch.equal(pinned_40, initial), "the pinned wealth drifted"
    assert not torch.equal(shadow_40, initial), "the shadow ledger stopped settling"


# --- The shuffled control ----------------------------------------------------------------


def _calibration_loss_by_hand(mob: MixtureOfBidders, hidden: torch.Tensor, sources) -> float:
    """Each head regressed onto the tokens and values of ``sources[head]``."""
    assert mob.last_stats is not None and mob.last_realised_values is not None
    selected = mob.last_stats.selected_experts
    values = mob.last_realised_values
    with torch.no_grad():
        _, confidences, _ = mob._report(hidden)
    terms = []
    for head, source in enumerate(sources):
        held_slots = selected == source
        held = held_slots.any(dim=-1)
        if not held.any():
            continue
        target = (values * held_slots).sum(dim=-1)[held]
        terms.append(torch.nn.functional.mse_loss(confidences[:, :, head][held], target))
    return float(torch.stack(terms).mean() * mob.config.confidence_calibration_weight)


@pytest.mark.parametrize(
    ("coupling", "sources"),
    [(PERSISTENCE_VALUE, (0, 1)), (PERSISTENCE_SHUFFLED, (1, 0))],
    ids=["own targets", "the other expert's targets"],
)
def test_the_shuffled_arm_regresses_each_head_onto_another_experts_values(coupling, sources):
    """With two experts the cyclic shift is the swap, so the target map is known exactly."""
    config = replace(SMALL, num_experts=2, top_k=1, persistence_coupling=coupling)
    mob = _contributing_layer(config)
    hidden = torch.randn(1, 12, config.hidden_dim)

    _settle(mob, hidden, seed=1)

    expected = _calibration_loss_by_hand(mob, hidden, sources)
    assert mob.get_confidence_calibration_loss().item() == pytest.approx(expected, rel=1e-5)
    other = _calibration_loss_by_hand(mob, hidden, sources[::-1])
    assert expected != pytest.approx(other), "the two target maps agree; the test cannot tell"


# --- The goal term -----------------------------------------------------------------------


def test_no_field_and_a_zero_dose_field_leave_realised_value_bitwise_unchanged():
    """The inert pairing: today's value definition until a field at a dose is attached."""
    hidden = torch.randn(1, 6, SMALL.hidden_dim)
    direction = torch.randn(SMALL.hidden_dim)

    bare = _settle(_contributing_layer(SMALL), hidden, seed=1)

    zero_dose = _contributing_layer(SMALL)
    zero_dose.attach_goal_field(ConstantGoalField(direction, setpoint=0.5, dose=0.0))
    assert torch.equal(_settle(zero_dose, hidden, seed=1), bare)
    assert zero_dose.last_goal_terms is not None
    assert torch.equal(zero_dose.last_goal_terms, torch.zeros_like(zero_dose.last_goal_terms))

    dosed = _contributing_layer(SMALL)
    dosed.attach_goal_field(ConstantGoalField(direction, setpoint=0.5, dose=1.0))
    assert not torch.equal(_settle(dosed, hidden, seed=1), bare), "the field is inert"
    assert dosed.last_goal_terms is not None and bool((dosed.last_goal_terms != 0).any())


def test_a_contribution_that_does_not_move_the_reading_earns_exactly_zero():
    """A tissue already holding its goal pays nothing for holding it, at any error."""
    contributions = torch.zeros(1, 3, 2, 4)
    contributions[..., 1] = torch.randn(1, 3, 2)  # every push is along e1
    weights = torch.full((1, 3, 2), 0.5)
    for setpoint in (0.0, 0.7, -3.0):
        reduction = goal_error_reduction(
            contributions, weights, torch.tensor([1.0, 0, 0, 0]), setpoint
        )
        assert torch.equal(reduction, torch.zeros(1, 3, 2))


def test_a_cell_aligned_beyond_the_setpoint_earns_nothing_more():
    """One slot at full share, setpoint 1: the reduction peaks at the setpoint and falls past it."""
    direction = torch.tensor([2.0, 0.0])  # not a unit vector: the coordinate is what counts
    pushes = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
    contributions = torch.zeros(1, 5, 1, 2)
    contributions[0, :, 0, 0] = pushes * 2.0  # coordinate = c . d / |d|^2 = push
    weights = torch.ones(1, 5, 1)

    reduction = goal_error_reduction(contributions, weights, direction, setpoint=1.0)[0, :, 0]

    assert torch.allclose(reduction, torch.tensor([0.5, 1.0, 0.5, 0.0, -1.0]))
    assert reduction.argmax().item() == 1, "the peak is not at the setpoint"
    assert bool((reduction[2:] < reduction[1]).all()), "overshoot earned more"


def test_a_tissue_already_at_its_setpoint_charges_the_cell_that_pushes_it_off():
    """The second winner arrives with the goal already met: its push is an error, not a service."""
    contributions = torch.zeros(1, 1, 2, 3)
    contributions[0, 0, 0, 0] = 2.0  # slot 0 at share 1/2 brings the reading to 1.0
    contributions[0, 0, 1, 0] = 0.8  # slot 1 at share 1/2 pushes it to 1.4
    weights = torch.full((1, 1, 2), 0.5)

    reduction = goal_error_reduction(
        contributions, weights, torch.tensor([1.0, 0, 0]), setpoint=1.0
    )

    assert reduction[0, 0, 0].item() == pytest.approx(0.2)  # |1 - 0.4| - |1 - 1.4|
    assert reduction[0, 0, 1].item() == pytest.approx(-0.4)  # |1 - 1.0| - |1 - 1.4|


def test_the_goal_term_is_paid_at_the_slots_share():
    """Per unit share in the value, so share times term is the priced reduction bought."""
    contributions = torch.randn(2, 4, 2, 5)
    weights = torch.softmax(torch.randn(2, 4, 2), dim=-1)
    field = ConstantGoalField(torch.randn(5), setpoint=0.3, dose=1.7)

    terms = goal_terms(contributions, weights, [field], torch.zeros(2, 4, 5))
    reduction = goal_error_reduction(contributions, weights, field.vector, field.setpoint)

    assert terms is not None
    assert torch.allclose(terms * weights, field.dose * reduction)
    assert goal_terms(contributions, weights, [], torch.zeros(2, 4, 5)) is None


def test_the_term_joins_value_in_per_token_loss_units_after_the_gradient_is_rescaled():
    """The dose prices error in per-token loss; ``loss_gradient_scale`` must not touch it."""
    hidden = torch.randn(1, 6, SMALL.hidden_dim)
    field = ConstantGoalField(torch.randn(SMALL.hidden_dim), setpoint=0.5, dose=1.0)

    bare = _settle(_contributing_layer(SMALL), hidden, seed=1, scale=1.0)
    dosed = _contributing_layer(SMALL)
    dosed.attach_goal_field(field)
    with_field = _settle(dosed, hidden, seed=1, scale=1.0)
    rescaled = _contributing_layer(SMALL)
    rescaled.attach_goal_field(field)
    with_field_at_two = _settle(rescaled, hidden, seed=1, scale=2.0)

    goal_part = with_field - bare
    assert torch.allclose(with_field_at_two, 2.0 * bare + goal_part, atol=1e-6)


def test_a_field_with_a_negative_dose_is_refused():
    with pytest.raises(ValueError, match="dose"):
        ConstantGoalField(torch.ones(3), setpoint=0.0, dose=-0.1)


# --- The fingerprint ---------------------------------------------------------------------


def test_the_dial_is_the_variable_under_test_and_the_dose_is_a_confound():
    decoupled = replace(BASE, persistence_coupling=PERSISTENCE_DECOUPLED)
    assert_parity([BASE, decoupled])
    assert {BASE.arm, decoupled.arm} == {"mob", "mob~decoupled"}

    with pytest.raises(ParityError, match="goal_doses"):
        assert_parity([BASE, replace(decoupled, goal_doses=(0.5,))])


# --- The goal fields on the differentiated fixture ---------------------------------------


def _differentiated(
    seed: int, doses: tuple[float, float] | None, **kwargs
) -> DifferentiatedEconomy:
    """The fixture at ``seed`` with a goal field on types 0 and 1 at ``doses``, or none."""
    economy = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, **kwargs)
    if doses is not None:
        for expert_type, dose in enumerate(doses):
            economy.add_goal_field(expert_type, setpoint=0.5, dose=dose)
    return economy


def test_the_goal_term_on_the_fixture_is_the_closed_form():
    """The layer's coordinate arithmetic against the planted competences, slot by slot.

    An expert of the field's type moves the reading by its competence per unit
    share and any other expert by exactly zero, so the term is a closed form in
    the winners' types and competences; the layer computes it from adapters and
    directions and must land on the same number. The gradient part of the value
    keeps its first-order relation to the exact counterfactual beside it -- on
    the on-type slots, where that relation is stated: an off-type winner's exact
    loss is a pure quadratic in its own contribution, so the first-order estimate
    is exactly twice it, on this fixture with or without a field.
    """
    economy = _differentiated(1, (0.25, 0.5))
    with torch.no_grad():
        economy.competence.mul_(0.2)  # the regime the first-order check is stated in
        for expert in economy.mob.experts:
            expert.down_adapter_B.weight.mul_(0.2)  # type: ignore[union-attr]

    record = economy.step(with_exact_values=True)

    terms = economy.mob.last_goal_terms
    assert terms is not None and record.exact_values is not None
    closed_form = economy.closed_form_goal_terms(record.selected_experts)
    assert bool((closed_form != 0).any()), "no winner of a goal type; the check is empty"
    assert torch.allclose(terms, closed_form, rtol=1e-4, atol=1e-6)

    gradient_part = record.realised_values - terms
    exact_part = record.exact_values - closed_form
    assert torch.equal(gradient_part.sign(), exact_part.sign())
    assert economy.last_types is not None
    on_type = economy.expert_types[record.selected_experts] == economy.last_types.unsqueeze(-1)
    relative_error = ((gradient_part - exact_part).abs() / exact_part.abs())[on_type]
    assert relative_error.numel() > 0 and relative_error.median().item() < 0.1


def _winners_and_ledger(seed: int, doses: tuple[float, float] | None, steps: int = 20):
    economy = _differentiated(seed, doses)
    return [economy.step().selected_experts for _ in range(steps)], economy.mob.expert_wealth


def test_a_goal_at_dose_zero_is_the_recorded_fixture_bit_for_bit():
    """``lambda = 0`` reproduces today's fixture and a dose does not: the inert pairing."""
    bare_winners, bare_wealth = _winners_and_ledger(0, None)
    zero_winners, zero_wealth = _winners_and_ledger(0, (0.0, 0.0))
    dosed_winners, dosed_wealth = _winners_and_ledger(0, (0.5, 0.5))

    for zero, bare in zip(zero_winners, bare_winners, strict=True):
        assert torch.equal(zero, bare)
    assert torch.equal(zero_wealth, bare_wealth)
    assert not torch.equal(dosed_wealth, bare_wealth), "a dosed field left the ledger untouched"
    assert any(
        not torch.equal(dosed, bare)
        for dosed, bare in zip(dosed_winners, bare_winners, strict=True)
    ), "a dosed field left every allocation untouched"


def _type_share(economy: DifferentiatedEconomy, expert_type: int, steps: int) -> float:
    wins = torch.zeros(economy.config.num_experts)
    for _ in range(steps):
        wins += torch.bincount(
            economy.step().selected_experts.flatten(), minlength=economy.config.num_experts
        ).float()
    share = wins / wins.sum()
    return float(share[economy.expert_types == expert_type].sum())


def test_a_goal_field_recruits_the_type_it_pays_for():
    """No silent no-op on the field: paid for holding type 0's correction, the tissue holds it.

    Chance for a type's share of the slots is 0.25; the loss alone already
    concentrates slots on competent cells, so the comparison is against the
    field-off share at the same seed rather than against chance.
    """
    field_off = _type_share(_differentiated(2, None), 0, steps=150)
    field_on = _type_share(_differentiated(2, (1.0, 0.0)), 0, steps=150)

    assert field_on > field_off + 0.1, (field_off, field_on)


def test_the_fixture_refuses_a_goal_on_a_type_it_did_not_plant():
    with pytest.raises(ValueError, match="expert_type"):
        _differentiated(0, None).add_goal_field(9, setpoint=0.5, dose=0.1)
