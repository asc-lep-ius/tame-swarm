from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mob import MixtureOfBidders, MoBConfig
from mob.utils import get_mob_statistics, get_total_router_z_loss
from train import TAMETrainer, TrainingConfig

STABILITY_CONFIG = MoBConfig(
    num_experts=2,
    top_k=1,
    hidden_dim=32,
    intermediate_dim=64,
    adapter_rank=4,
    adapter_alpha=4.0,
    use_shared_base=True,
    use_vcg_payments=True,
    use_differentiable_routing=True,
    use_loss_feedback=True,
    use_local_quality=True,
)


def _build_training_mob():
    mob = MixtureOfBidders(STABILITY_CONFIG)
    mob.train()
    return mob


def test_wealth_stays_bounded_after_many_updates():
    mob = _build_training_mob()
    x = torch.randn(1, 8, 32)

    for _ in range(1000):
        mob(x)
        per_token_loss = torch.randn(1, 8).abs()
        mob.update_wealth_from_loss(per_token_loss)

    assert (mob.expert_wealth >= STABILITY_CONFIG.min_wealth).all()
    assert (mob.expert_wealth <= STABILITY_CONFIG.max_wealth).all()


@pytest.mark.parametrize("loss_value", [0.0])
def test_wealth_no_nan_on_zero_loss(loss_value):
    mob = _build_training_mob()
    x = torch.randn(1, 8, 32)

    mob(x)
    per_token_loss = torch.full((1, 8), loss_value)
    mob.update_wealth_from_loss(per_token_loss)

    assert not torch.isnan(mob.expert_wealth).any()
    assert not torch.isinf(mob.expert_wealth).any()


@pytest.mark.parametrize("loss_value", [1e6, 1e8])
def test_wealth_no_nan_on_large_loss(loss_value):
    mob = _build_training_mob()
    x = torch.randn(1, 8, 32)

    mob(x)
    per_token_loss = torch.full((1, 8), loss_value)
    mob.update_wealth_from_loss(per_token_loss)

    assert not torch.isnan(mob.expert_wealth).any()
    assert not torch.isinf(mob.expert_wealth).any()


def test_wealth_decay_applied():
    inference_config = MoBConfig(
        num_experts=2,
        top_k=1,
        hidden_dim=32,
        intermediate_dim=64,
        adapter_rank=4,
        adapter_alpha=4.0,
        use_shared_base=True,
        use_vcg_payments=True,
        use_differentiable_routing=False,
        use_loss_feedback=False,
        use_local_quality=True,
    )
    mob = MixtureOfBidders(inference_config)
    mob.eval()
    initial_wealth = mob.expert_wealth.clone()
    x = torch.randn(1, 8, 32)

    mob(x)

    assert not torch.equal(mob.expert_wealth, initial_wealth), (
        "Wealth should change after forward pass with decay and quality updates active"
    )


def test_gini_between_zero_and_one(mob_layer, random_hidden_states):
    class FakeModel(torch.nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    fake_model = FakeModel(mob_layer)

    stats = get_mob_statistics(fake_model)
    if stats:
        gini = stats["wealth_gini"].item()
        assert 0.0 <= gini <= 1.0, f"Gini coefficient should be in [0, 1], got {gini}"


def test_usage_count_increments(training_mob_layer, random_hidden_states):
    initial_usage = training_mob_layer.expert_usage_count.clone()

    training_mob_layer(random_hidden_states)

    assert (training_mob_layer.expert_usage_count > initial_usage).any(), (
        "Expert usage count should increase after forward pass"
    )


def test_performance_ema_updates():
    mob = _build_training_mob()
    x = torch.randn(1, 8, 32)

    initial_ema = mob.expert_performance_ema.clone()

    mob(x)
    per_token_loss = torch.randn(1, 8).abs() + 0.5
    mob.update_wealth_from_loss(per_token_loss)

    changed = (mob.expert_performance_ema != initial_ema).any()
    assert changed, "Performance EMA should change after update_wealth_from_loss"


def test_calibration_loss_finite():
    mob = _build_training_mob()
    x = torch.randn(1, 8, 32)

    mob(x)
    per_token_loss = torch.randn(1, 8).abs()
    mob.update_wealth_from_loss(per_token_loss)

    cal_loss = mob.get_confidence_calibration_loss()
    assert torch.isfinite(cal_loss).all(), f"Calibration loss should be finite, got {cal_loss}"


def test_total_router_z_loss_aggregates_live_layer_losses() -> None:
    mob_a = _build_training_mob()
    mob_b = _build_training_mob()
    model = torch.nn.Sequential(mob_a, mob_b)
    hidden_states = torch.randn(2, 4, STABILITY_CONFIG.hidden_dim, requires_grad=True)

    model(hidden_states)
    router_z_loss = get_total_router_z_loss(model)

    assert torch.isfinite(router_z_loss)
    assert router_z_loss.item() >= 0.0
    assert router_z_loss.requires_grad

    router_z_loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in mob_a.confidence_heads.parameters()
    )


class TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 8
        vocab_size = 11
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.mob = MixtureOfBidders(
            MoBConfig(
                num_experts=2,
                top_k=1,
                hidden_dim=hidden_dim,
                intermediate_dim=16,
                adapter_rank=2,
                adapter_alpha=2.0,
                use_shared_base=True,
                use_vcg_payments=True,
                use_differentiable_routing=True,
                use_loss_feedback=True,
                use_local_quality=True,
                confidence_z_loss_weight=0.1,
            )
        )
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)
        self.forward_coupling_step: int | None = None

        self.mob.attach_coupling(
            torch.ones(hidden_dim),
            config=None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, labels, use_cache
        self.forward_coupling_step = int(self.mob.coupling._coupling_step.item())
        hidden_states = self.embedding(input_ids)
        hidden_states = self.mob(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def test_train_step_reports_router_z_loss_and_sets_coupling_step() -> None:
    trainer = TAMETrainer(
        TrainingConfig(
            device="cpu",
            dtype="float32",
            gradient_accumulation_steps=2,
            wealth_update_frequency=1,
        )
    )
    model = TinyCausalModel()
    trainer.model = model
    trainer.global_step = 7

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }

    metrics = trainer.train_step(batch)

    assert model.forward_coupling_step == trainer.global_step
    assert torch.isfinite(torch.tensor(metrics["router_z_loss"]))
    assert metrics["router_z_loss"] >= 0.0
    assert metrics["total_loss"] == pytest.approx(
        metrics["loss"] + metrics["calibration_loss"] + metrics["router_z_loss"],
        rel=1e-6,
    )


QUASI_LINEAR_CONFIG = MoBConfig(
    num_experts=4,
    top_k=2,
    hidden_dim=32,
    intermediate_dim=64,
    adapter_rank=4,
    adapter_alpha=4.0,
    use_shared_base=True,
    use_vcg_payments=True,
    use_differentiable_routing=True,
    use_loss_feedback=True,
    use_local_quality=True,
)


def _uniform_selection(batch: int, seq: int) -> torch.Tensor:
    """Expert 0 takes every slot-0 win, expert 1 every slot-1 win."""
    selected = torch.zeros(batch, seq, 2, dtype=torch.long)
    selected[:, :, 1] = 1
    return selected


def _split_selection() -> torch.Tensor:
    """Four tokens: experts 0 and 2 split slot 0; expert 1 holds slot 1 throughout.

    Every winner here has a token share of 0.5 or 1.0 rather than a uniform 1.0,
    so a charge that ignores the share weighting is distinguishable.
    """
    selected = torch.zeros(1, 4, 2, dtype=torch.long)
    selected[0, :, 0] = torch.tensor([0, 0, 2, 2])
    selected[0, :, 1] = 1
    return selected


def _cross_slot_selection() -> torch.Tensor:
    """Expert 0 wins slot 0 on the first two tokens and slot 1 on the last two.

    Experts really do win different slots on different tokens, so the helper must
    accumulate across slots rather than overwrite.
    """
    selected = torch.zeros(1, 4, 2, dtype=torch.long)
    selected[0, :, 0] = torch.tensor([0, 0, 3, 3])
    selected[0, :, 1] = torch.tensor([1, 1, 0, 0])
    return selected


def _slot_priced_payments() -> torch.Tensor:
    """Distinct price per slot, so a charge cannot pick the wrong slot unnoticed."""
    payments = torch.empty(1, 4, 2)
    payments[..., 0] = 2.0
    payments[..., 1] = 6.0
    return payments


def _charges_for(mob: MixtureOfBidders, payment_value: float) -> torch.Tensor:
    payments = torch.full((1, 4, 2), payment_value)
    return mob._vcg_charges(payments, _uniform_selection(1, 4), num_tokens=4)


def test_vcg_charge_matches_hand_computed_transfer():
    """charge = mean payment x token share x payment_scale, in reward units."""
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)

    charges = _charges_for(mob, payment_value=2.0)

    # Experts 0 and 1 win every token, so token share is 1.0 for each.
    expected = 2.0 * 1.0 * QUASI_LINEAR_CONFIG.payment_scale
    assert charges[0].item() == pytest.approx(expected, abs=1e-6)
    assert charges[1].item() == pytest.approx(expected, abs=1e-6)
    assert charges[2].item() == 0.0
    assert charges[3].item() == 0.0


def test_vcg_charge_independent_of_expert_wealth():
    """Quasi-linearity marker: the transfer is in reward units, not a wealth fraction.

    The multiplicative form this replaced divided the payment by expert wealth, so
    a rich expert was charged less for the same win. A transfer cannot do that.
    """
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)

    mob.expert_wealth.fill_(20.0)
    poor = _charges_for(mob, payment_value=2.0).clone()

    mob.expert_wealth.fill_(500.0)
    rich = _charges_for(mob, payment_value=2.0)

    assert torch.allclose(poor, rich)


def test_vcg_charge_scales_linearly_with_payments():
    """Additive transfer: doubling the price doubles the charge."""
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)

    single = _charges_for(mob, payment_value=1.0).clone()
    double = _charges_for(mob, payment_value=2.0)

    assert torch.allclose(double, 2.0 * single)


def test_vcg_charge_is_zero_when_payments_disabled():
    config = replace(QUASI_LINEAR_CONFIG, use_vcg_payments=False)
    mob = MixtureOfBidders(config)

    assert (_charges_for(mob, payment_value=2.0) == 0).all()


def test_payments_reduce_wealth_by_exactly_the_charge():
    """End-to-end quasi-linearity: enabling payments subtracts the charge, nothing else.

    Under the multiplicative haircut the gap between the two runs would scale with
    each expert's reward; under a transfer it is the charge on its own.
    """
    hidden_states = torch.randn(1, 8, 32)
    per_token_loss = torch.randn(1, 8).abs()

    def run(use_payments: bool) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(11)
        config = replace(QUASI_LINEAR_CONFIG, use_vcg_payments=use_payments)
        mob = MixtureOfBidders(config)
        mob.train()
        mob(hidden_states)
        charges = mob._vcg_charges(
            mob._cached_payments,
            mob._cached_selected_experts,
            num_tokens=hidden_states.size(0) * hidden_states.size(1),
        ).clone()
        mob.update_wealth_from_loss(per_token_loss)
        return mob.expert_wealth.clone(), charges

    # This computes its expectation by calling _vcg_charges, so it pins the wiring
    # -- that the charge is subtracted additively rather than multiplied into the
    # reward -- and not the formula inside. The unit tests above cover the formula.
    wealth_paying, charges = run(use_payments=True)
    wealth_free, _ = run(use_payments=False)

    assert (charges > 0).any(), "fixture charges nothing; the comparison proves nothing"
    assert torch.allclose(wealth_paying, wealth_free - charges, atol=1e-5)


def test_vcg_charge_weights_by_token_share():
    """An expert winning half the tokens pays half as much for the same price."""
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)
    scale = QUASI_LINEAR_CONFIG.payment_scale

    charges = mob._vcg_charges(_slot_priced_payments(), _split_selection(), num_tokens=4)

    # Experts 0 and 2 split slot 0 (price 2.0, share 0.5 each); expert 1 holds
    # slot 1 outright (price 6.0, share 1.0).
    assert charges[0].item() == pytest.approx(2.0 * 0.5 * scale, abs=1e-6)
    assert charges[2].item() == pytest.approx(2.0 * 0.5 * scale, abs=1e-6)
    assert charges[1].item() == pytest.approx(6.0 * 1.0 * scale, abs=1e-6)
    assert charges[3].item() == 0.0


def test_vcg_charge_accumulates_across_slots():
    """One expert winning in two different slots pays for both."""
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)
    scale = QUASI_LINEAR_CONFIG.payment_scale

    charges = mob._vcg_charges(_slot_priced_payments(), _cross_slot_selection(), num_tokens=4)

    # Expert 0: slot 0 on two tokens at 2.0, slot 1 on two tokens at 6.0.
    expected = (2.0 * 0.5 + 6.0 * 0.5) * scale
    assert charges[0].item() == pytest.approx(expected, abs=1e-6)


def _spy_on_charges(mob: MixtureOfBidders) -> list[SimpleNamespace]:
    """Record every _vcg_charges call so a dropped call site cannot pass silently."""
    calls = []
    original = mob._vcg_charges

    def spy(payments, selected_experts, num_tokens):
        charge = original(payments, selected_experts, num_tokens)
        calls.append(SimpleNamespace(num_tokens=num_tokens, charge=charge.clone()))
        return charge

    mob._vcg_charges = spy
    return calls


@pytest.mark.parametrize(
    "path,overrides",
    [
        ("local_quality", {"use_loss_feedback": False, "use_local_quality": True}),
        ("participation", {"use_loss_feedback": False, "use_local_quality": False}),
    ],
)
def test_non_loss_paths_apply_the_transfer(path, overrides):
    """The local-quality and participation paths must charge too, with right num_tokens.

    Both run inside forward() rather than update_wealth_from_loss, so neither is
    reached by the loss-path tests above.
    """
    torch.manual_seed(23)
    hidden_states = torch.randn(2, 8, 32)

    def run(use_payments: bool):
        torch.manual_seed(17)
        config = replace(QUASI_LINEAR_CONFIG, use_vcg_payments=use_payments, **overrides)
        mob = MixtureOfBidders(config)
        mob.train()
        calls = _spy_on_charges(mob)
        mob(hidden_states)
        return mob.expert_wealth.clone(), calls

    wealth_paying, calls = run(use_payments=True)
    wealth_free, _ = run(use_payments=False)

    assert len(calls) == 1, f"{path} path did not apply a VCG charge"
    assert calls[0].num_tokens == 2 * 8, "charge normalised by the wrong token count"
    # A magnitude floor, not just a sign: float noise would satisfy `> 0`.
    assert calls[0].charge.max() > 1e-3, "fixture charges nothing; the comparison is vacuous"
    assert torch.allclose(wealth_paying, wealth_free - calls[0].charge, atol=1e-5)
