from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mob import MixtureOfBidders, MoBConfig
from mob.utils import get_mob_statistics, get_total_router_z_loss
from mob.wealth import (
    LOCAL_REWARD_MULTIPLIER,
    LOSS_REWARD_MULTIPLIER,
    PARTICIPATION_REWARD_MULTIPLIER,
)
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


def _expected_coefficient(config: MoBConfig, reward_multiplier: float) -> float:
    """The transfer scale restated from the quasi-linearity requirement.

    A reward is ``value x (1/top_k) x reward_scale x reward_multiplier``. For
    ``reward - charge`` to be one utility, the charge must carry that same scale;
    ``payment_scale`` is only a dimensionless deviation from it. Written out here so
    the tests constrain the definition rather than echo the implementation.
    """
    return config.payment_scale * config.reward_scale * reward_multiplier / config.top_k


def _charges_for(
    mob: MixtureOfBidders,
    payment_value: float,
    reward_multiplier: float = LOSS_REWARD_MULTIPLIER,
) -> torch.Tensor:
    payments = torch.full((1, 4, 2), payment_value)
    return mob._vcg_charges(
        payments, _uniform_selection(1, 4), num_tokens=4, reward_multiplier=reward_multiplier
    )


def test_vcg_charge_matches_hand_computed_transfer():
    """charge = mean payment x token share x the reward's own scale."""
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)

    charges = _charges_for(mob, payment_value=2.0)

    # Experts 0 and 1 win every token, so token share is 1.0 for each.
    expected = 2.0 * 1.0 * _expected_coefficient(QUASI_LINEAR_CONFIG, LOSS_REWARD_MULTIPLIER)
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


def test_payments_reduce_wealth_by_exactly_the_charge(monkeypatch):
    """End-to-end quasi-linearity: enabling payments subtracts the charge, nothing else.

    Under the multiplicative haircut the gap between the two runs would scale with
    each expert's reward; under a transfer it is the charge on its own.

    The competitive bonus is switched off because it is now computed on the net,
    i.e. downstream of the charge, so enabling payments legitimately moves it too.
    That coupling is the subject of test_wealth_is_flat_at_the_price; isolating the
    transfer is the subject here.
    """
    monkeypatch.setattr("mob.wealth.COMPETITIVE_BONUS_FACTOR", 0.0)
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
            reward_multiplier=LOSS_REWARD_MULTIPLIER,
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
    scale = _expected_coefficient(QUASI_LINEAR_CONFIG, LOSS_REWARD_MULTIPLIER)

    charges = mob._vcg_charges(
        _slot_priced_payments(),
        _split_selection(),
        num_tokens=4,
        reward_multiplier=LOSS_REWARD_MULTIPLIER,
    )

    # Experts 0 and 2 split slot 0 (price 2.0, share 0.5 each); expert 1 holds
    # slot 1 outright (price 6.0, share 1.0).
    assert charges[0].item() == pytest.approx(2.0 * 0.5 * scale, abs=1e-6)
    assert charges[2].item() == pytest.approx(2.0 * 0.5 * scale, abs=1e-6)
    assert charges[1].item() == pytest.approx(6.0 * 1.0 * scale, abs=1e-6)
    assert charges[3].item() == 0.0


def test_vcg_charge_accumulates_across_slots():
    """One expert winning in two different slots pays for both."""
    mob = MixtureOfBidders(QUASI_LINEAR_CONFIG)
    scale = _expected_coefficient(QUASI_LINEAR_CONFIG, LOSS_REWARD_MULTIPLIER)

    charges = mob._vcg_charges(
        _slot_priced_payments(),
        _cross_slot_selection(),
        num_tokens=4,
        reward_multiplier=LOSS_REWARD_MULTIPLIER,
    )

    # Expert 0: slot 0 on two tokens at 2.0, slot 1 on two tokens at 6.0.
    expected = (2.0 * 0.5 + 6.0 * 0.5) * scale
    assert charges[0].item() == pytest.approx(expected, abs=1e-6)


def _spy_on_charges(mob: MixtureOfBidders) -> list[SimpleNamespace]:
    """Record every _vcg_charges call so a dropped call site cannot pass silently."""
    calls = []
    original = mob._vcg_charges

    def spy(payments, selected_experts, num_tokens, reward_multiplier):
        charge = original(payments, selected_experts, num_tokens, reward_multiplier)
        calls.append(
            SimpleNamespace(
                num_tokens=num_tokens,
                reward_multiplier=reward_multiplier,
                charge=charge.clone(),
            )
        )
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
def test_non_loss_paths_apply_the_transfer(path, overrides, monkeypatch):
    """The local-quality and participation paths must charge too, with right num_tokens.

    Both run inside forward() rather than update_wealth_from_loss, so neither is
    reached by the loss-path tests above. The competitive bonus is disabled for the
    same reason as in test_payments_reduce_wealth_by_exactly_the_charge.
    """
    monkeypatch.setattr("mob.wealth.COMPETITIVE_BONUS_FACTOR", 0.0)
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
    expected_multiplier = (
        LOCAL_REWARD_MULTIPLIER if path == "local_quality" else PARTICIPATION_REWARD_MULTIPLIER
    )
    assert calls[0].reward_multiplier == expected_multiplier, (
        f"{path} path priced at another path's reward scale"
    )
    # A magnitude floor, not just a sign: float noise would satisfy `> 0`.
    assert calls[0].charge.max() > 1e-3, "fixture charges nothing; the comparison is vacuous"
    assert torch.allclose(wealth_paying, wealth_free - calls[0].charge, atol=1e-5)


def _confidence_head_grads(mob: MixtureOfBidders) -> list[float]:
    return [
        0.0 if head.proj.weight.grad is None else head.proj.weight.grad.abs().sum().item()
        for head in mob.confidence_heads
    ]


def test_value_objective_carries_gradient():
    """The objective has to reach an optimiser, not just be added to a scalar.

    It previously did not: the loss was built from the detached confidence cache
    inside ``torch.no_grad()``, so it summed into the training loss as a constant
    and trained nothing. A finiteness check passed the whole time.
    """
    mob = _build_training_mob()
    mob(torch.randn(1, 8, 32))
    mob.update_wealth_from_loss(torch.randn(1, 8).abs())

    objective = mob.get_confidence_calibration_loss()

    assert objective.requires_grad, "value objective is detached and trains nothing"
    assert objective.grad_fn is not None

    objective.backward()
    assert any(grad > 0.0 for grad in _confidence_head_grads(mob))


def test_value_objective_is_the_only_gradient_reaching_confidence_heads():
    """No central planner: the language-modelling loss must not route.

    Backpropagating the layer output alone has to leave every confidence head
    untouched. If a gradient appears here, the uniform share has been reverted and
    the heads are once again slices of the global objective rather than agents.
    """
    mob = _build_training_mob()

    output = mob(torch.randn(1, 8, 32))
    output.sum().backward()

    assert all(grad == 0.0 for grad in _confidence_head_grads(mob))
    assert any(
        expert.gate_adapter_B.weight.grad.abs().sum().item() > 0.0 for expert in mob.experts
    ), "the language-modelling loss must still train the expert adapters"


def test_value_objective_is_local_to_each_expert():
    """Expert i's realised value may only move expert i's report.

    An expert at zero wealth bids zero and cannot win a slot, so it realises no
    value and must receive no gradient. If it does, the objective is reading
    outcomes that belong to other experts.
    """
    config = replace(STABILITY_CONFIG, num_experts=4, top_k=1)
    mob = MixtureOfBidders(config)
    mob.train()
    mob.expert_wealth[3] = 0.0

    mob(torch.randn(2, 8, 32))
    winners = set(mob.last_stats.selected_experts.flatten().tolist())
    assert 3 not in winners, "a bankrupt expert cannot win a slot"

    mob.update_wealth_from_loss(torch.randn(2, 8).abs())
    mob.get_confidence_calibration_loss().backward()

    for expert_idx, grad in enumerate(_confidence_head_grads(mob)):
        if expert_idx in winners:
            assert grad > 0.0, f"winner {expert_idx} got no value signal"
        else:
            assert grad == 0.0, f"expert {expert_idx} was trained on a token it never won"


def test_value_objective_excludes_masked_tokens():
    """Padding is dropped, not scored as a token of zero loss.

    The reward path multiplies masked positions by zero, which as a *target* reads
    as the largest loss reduction an expert can achieve -- so padding would teach
    every winner to report maximum confidence.

    Asserted positively against the definition on one module, and paired with the
    mask-ignoring value it must not equal. Comparing two separately built modules
    proves nothing: their inputs and buffers differ, so the numbers differ whether
    or not masking is implemented.
    """
    mob = _build_training_mob()
    mob(torch.randn(1, 8, 32))

    confidences = mob.last_stats.confidences
    selected = mob.last_stats.selected_experts
    baselines = mob.expert_baseline_loss.clone()

    per_token_loss = torch.randn(1, 8).abs() + 0.5
    token_mask = torch.ones(1, 8)
    token_mask[0, 4:] = 0.0
    mob.update_wealth_from_loss(per_token_loss, token_mask)

    def expected(valid: torch.Tensor | None) -> float:
        terms = []
        for expert_idx in range(mob.config.num_experts):
            won = (selected == expert_idx).any(dim=-1)
            if valid is not None:
                won = won & valid
            if not won.any():
                continue
            target = (baselines[expert_idx] - per_token_loss[won]).clamp_min(0.0)
            terms.append(torch.nn.functional.mse_loss(confidences[:, :, expert_idx][won], target))
        return (torch.stack(terms).mean() * mob.config.confidence_calibration_weight).item()

    masked = expected(token_mask > 0)
    unmasked = expected(None)

    assert masked != pytest.approx(unmasked, abs=1e-6), (
        "fixture must distinguish the two; the mask has to change the objective"
    )
    assert mob.get_confidence_calibration_loss().item() == pytest.approx(masked, abs=1e-6)


def test_stale_value_objective_is_not_backwarded_twice():
    """A forward without a wealth update must not reuse the previous step's graph.

    `train.py` calls update_all_mob_from_loss only every
    `wealth_update_frequency` steps but adds get_total_calibration_loss into the
    training loss on every step. The old calibration loss was a detached constant,
    so a stale one was harmless; the value objective holds a live graph, and
    backwarding last step's graph raises "backward through the graph a second time".
    """
    mob = _build_training_mob()

    for step in range(4):
        output = mob(torch.randn(1, 8, 32))
        if step % 2 == 0:
            mob.update_wealth_from_loss(torch.randn(1, 8).abs())

        # Mirror train.py: the objective is summed into a loss that always carries
        # a graph, so a stale cached objective is backwarded rather than skipped.
        total = output.sum() + mob.get_confidence_calibration_loss()
        total.backward()

    assert mob.get_confidence_calibration_loss().item() == 0.0


def test_value_objective_is_zero_when_no_update_ran():
    mob = _build_training_mob()
    mob(torch.randn(1, 8, 32))
    mob.update_wealth_from_loss(torch.randn(1, 8).abs())
    assert mob.get_confidence_calibration_loss().item() > 0.0

    mob(torch.randn(1, 8, 32))
    assert mob.get_confidence_calibration_loss().item() == 0.0


def test_value_objective_releases_the_graph_after_use():
    """The live confidence cache pins a graph; it must not survive the update."""
    mob = _build_training_mob()
    mob(torch.randn(1, 8, 32))
    assert mob._live_confidences is not None

    mob.update_wealth_from_loss(torch.randn(1, 8).abs())
    assert mob._live_confidences is None


def test_value_objective_releases_the_graph_on_every_exit_path():
    """Not just the happy path -- a pinned graph outlives the step that made it."""
    mob = _build_training_mob()

    mob(torch.randn(1, 8, 32))
    mob._loss_feedback_pending = False
    mob.update_wealth_from_loss(torch.randn(1, 8).abs())
    assert mob._live_confidences is None, "leaked on the not-pending return"

    mob(torch.randn(1, 8, 32))
    mob.update_wealth_from_loss(torch.randn(1, 32).abs())
    assert mob._live_confidences is None, "leaked on the seq-len mismatch return"

    mob(torch.randn(1, 8, 32))
    assert mob._live_confidences is not None
    mob(torch.randn(1, 8, 32))
    assert mob.get_confidence_calibration_loss().item() == 0.0


def test_value_objective_matches_its_definition():
    """Rebuild the objective from the mechanism statement rather than the code.

    Each winner's target is ``clamp_min(baseline - loss, 0)`` on the tokens it
    won, where ``baseline`` is the value it held when it bid -- not the one the
    reward loop leaves behind, which has already absorbed the very batch the expert
    is being judged on.
    """
    mob = _build_training_mob()
    hidden = torch.randn(1, 8, 32)
    mob(hidden)

    confidences = mob.last_stats.confidences
    selected = mob.last_stats.selected_experts
    baselines = mob.expert_baseline_loss.clone()

    per_token_loss = torch.full((1, 8), 4.0)
    mob.update_wealth_from_loss(per_token_loss)

    assert not torch.allclose(baselines, mob.expert_baseline_loss), (
        "fixture must actually move the baseline, or the snapshot is untested"
    )

    terms = []
    for expert_idx in range(mob.config.num_experts):
        won = (selected == expert_idx).any(dim=-1)
        if not won.any():
            continue
        target = torch.sigmoid(5.0 * (baselines[expert_idx] - per_token_loss[won]))
        terms.append(torch.nn.functional.mse_loss(confidences[:, :, expert_idx][won], target))

    expected = torch.stack(terms).mean() * mob.config.confidence_calibration_weight
    assert mob.get_confidence_calibration_loss().item() == pytest.approx(expected.item(), abs=1e-6)


def test_value_objective_does_not_train_the_backbone():
    """Heads observe the representation; they must not reshape it.

    Every head reads the same hidden states, so an undetached routing path turns
    each expert's private objective into a shared auxiliary loss on everything
    below the layer — weighted 0.15 and summed over every MoB layer. That is the
    central planner the auction exists to replace, arriving through the back door.
    """
    trunk = torch.nn.Linear(32, 32)
    mob = _build_training_mob()

    mob(trunk(torch.randn(1, 8, 32)))
    mob.update_wealth_from_loss(torch.randn(1, 8).abs())
    mob.get_confidence_calibration_loss().backward()

    assert trunk.weight.grad is None or trunk.weight.grad.abs().sum().item() == 0.0
    assert any(grad > 0.0 for grad in _confidence_head_grads(mob)), (
        "detaching must not also starve the heads"
    )


def test_language_model_loss_still_trains_the_backbone():
    """The detach is scoped to the routing path, not to the layer output."""
    trunk = torch.nn.Linear(32, 32)
    mob = _build_training_mob()

    mob(trunk(torch.randn(1, 8, 32))).sum().backward()

    assert trunk.weight.grad is not None
    assert trunk.weight.grad.abs().sum().item() > 0.0


def test_detached_routing_path_still_trains_the_coupling():
    """Detaching the input, not the coupling output, keeps issue #2's module live."""
    from coupling import SteeringCouplingConfig

    mob = _build_training_mob()
    mob.attach_coupling(torch.randn(32), SteeringCouplingConfig(hidden_dim=32))
    torch.nn.init.normal_(mob.coupling.projection.weight, std=0.05)
    mob.set_coupling_step(10)

    mob(torch.randn(1, 8, 32))
    mob.update_wealth_from_loss(torch.randn(1, 8).abs())
    mob.get_confidence_calibration_loss().backward()

    grad = mob.coupling.projection.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0.0


def _net_wealth_change(value: float, price: float) -> float:
    """Wealth moved by one expert that won every token, at a known value and price.

    Decay is switched off so the result is the transfer alone, and the baseline is
    set explicitly so the realised loss reduction is exactly ``value``.
    """
    config = replace(QUASI_LINEAR_CONFIG, wealth_decay=1.0, num_experts=4, top_k=2)
    mob = MixtureOfBidders(config)
    mob.train()
    mob(torch.randn(1, 4, 32))

    mob.expert_baseline_loss.fill_(1.0 + value)
    mob._cached_selected_experts = _uniform_selection(1, 4)
    mob._cached_routing_weights = torch.full((1, 4, 2), 0.5)
    mob._cached_payments = torch.full((1, 4, 2), price)

    before = mob.expert_wealth[0].item()
    mob.update_wealth_from_loss(torch.full((1, 4), 1.0))
    return mob.expert_wealth[0].item() - before


def test_wealth_threshold_coincides_with_the_auction_threshold():
    """The economy pays exactly when the auction would let a truthful expert win.

    This is what quasi-linearity buys and what a mismatched pair of scales destroys.
    The auction lets an expert win when ``report > price``; with a truthful report
    that is ``value > price``. Wealth must move the same way across that same
    boundary, or an expert maximising wealth wants a different allocation than the
    one the mechanism prices -- and overreporting pays.

    Before reward and charge shared a coefficient the reward side outweighed the
    charge side ~167x, so winning was profitable far below the price and the
    strategyproofness result said nothing about the realised economy.
    """
    assert _net_wealth_change(value=2.0, price=1.0) > 0.0, (
        "a truthful winner above its price must gain"
    )
    assert _net_wealth_change(value=0.5, price=2.0) < 0.0, (
        "winning below the price must cost, or overreporting to win is free money"
    )


def test_wealth_is_flat_at_the_price():
    """The boundary itself: value == price nets nothing but the competitive bonus.

    Pinning the crossing point is what stops the test above from passing on any
    monotone-but-misscaled pair of coefficients.
    """
    below = _net_wealth_change(value=0.9, price=1.0)
    at = _net_wealth_change(value=1.0, price=1.0)
    above = _net_wealth_change(value=1.1, price=1.0)

    assert below < at < above
    assert at == pytest.approx(0.0, abs=1e-4), (
        "break-even must sit exactly at the price; a competitive bonus paid on "
        "gross wins rather than surplus moves it below"
    )


def test_each_wealth_path_is_charged_at_its_own_reward_scale():
    """One shared constant would only be quasi-linear for the path it came from.

    The three wealth paths use different reward multipliers, so the transfer has to
    follow the path rather than a single config value.
    """
    config = QUASI_LINEAR_CONFIG
    loss_coefficient = _expected_coefficient(config, LOSS_REWARD_MULTIPLIER)
    local_coefficient = _expected_coefficient(config, LOCAL_REWARD_MULTIPLIER)
    participation_coefficient = _expected_coefficient(config, PARTICIPATION_REWARD_MULTIPLIER)

    assert len({loss_coefficient, local_coefficient, participation_coefficient}) == 3, (
        "fixture cannot tell the paths apart"
    )

    mob = MixtureOfBidders(config)
    for multiplier, coefficient in (
        (LOSS_REWARD_MULTIPLIER, loss_coefficient),
        (LOCAL_REWARD_MULTIPLIER, local_coefficient),
        (PARTICIPATION_REWARD_MULTIPLIER, participation_coefficient),
    ):
        charges = _charges_for(mob, payment_value=2.0, reward_multiplier=multiplier)
        assert charges[0].item() == pytest.approx(2.0 * coefficient, abs=1e-6)
