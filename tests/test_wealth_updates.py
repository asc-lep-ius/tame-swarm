import torch
import pytest

from mob import MoBConfig, MixtureOfBidders
from mob.utils import get_mob_statistics


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
