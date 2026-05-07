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
