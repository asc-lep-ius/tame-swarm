import logging
from operator import contains

import pytest
import torch
import torch.nn as nn

from mob import (
    MixtureOfBidders,
    MoBConfig,
    MoBStats,
    apply_mob_to_model,
    load_mob_state,
    save_mob_state,
)


def test_forward_output_shape(mob_layer, random_hidden_states):
    out = mob_layer(random_hidden_states)
    assert out.shape == random_hidden_states.shape


def test_forward_no_nan_inf(mob_layer, random_hidden_states):
    out = mob_layer(random_hidden_states)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_updates_last_stats(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    stats = mob_layer.last_stats
    assert isinstance(stats, MoBStats)
    assert stats.confidence_logits.shape == (1, 8, mob_layer.config.num_experts)
    assert stats.confidences.shape == (1, 8, mob_layer.config.num_experts)
    assert stats.selected_experts.shape == (1, 8, mob_layer.config.top_k)
    assert stats.routing_weights.shape == (1, 8, mob_layer.config.top_k)
    assert stats.expert_wealth.shape == (mob_layer.config.num_experts,)
    assert stats.expert_usage.shape == (mob_layer.config.num_experts,)
    assert stats.expert_performance.shape == (mob_layer.config.num_experts,)
    assert stats.coupling_metrics is None


def test_forward_records_detached_finite_router_z_loss(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    stats = mob_layer.last_stats
    assert isinstance(stats, MoBStats)
    assert torch.isfinite(stats.router_z_loss)
    assert stats.router_z_loss >= 0.0
    assert not stats.router_z_loss.requires_grad
    assert not stats.confidence_logits.requires_grad
    assert not stats.confidences.requires_grad


def test_router_z_loss_matches_logsumexp_formula_with_default_weight(tiny_config):
    mob = MixtureOfBidders(tiny_config)
    logits = torch.tensor(
        [
            [[0.25, -1.5], [2.0, 0.5]],
            [[-0.75, 1.25], [0.0, -0.25]],
        ],
        dtype=torch.float16,
    )

    assert tiny_config.confidence_z_loss_weight == pytest.approx(0.0001)
    expected = (
        torch.logsumexp(logits.float(), dim=-1).square().mean()
        * tiny_config.confidence_z_loss_weight
    )

    assert torch.allclose(mob._compute_router_z_loss(logits), expected)


def test_mob_stats_does_not_support_mapping_style_access(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    stats = mob_layer.last_stats
    assert isinstance(stats, MoBStats)
    assert stats.expert_wealth.shape == (mob_layer.config.num_experts,)

    with pytest.raises(TypeError):
        contains(stats, "expert_wealth")
    with pytest.raises(TypeError):
        stats["expert_wealth"]
    with pytest.raises(AttributeError):
        stats.keys()


def test_save_load_roundtrip(tmp_path, tiny_config):
    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    mob.train()
    x = torch.randn(1, 8, 32)
    mob(x)
    per_token_loss = torch.randn(1, 8).abs()
    mob.update_wealth_from_loss(per_token_loss)

    original_wealth = mob.expert_wealth.clone()
    original_ema = mob.expert_performance_ema.clone()
    original_baseline = mob.expert_baseline_loss.clone()

    model = FakeModel(mob)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(model, save_path)

    mob2 = MixtureOfBidders(tiny_config)
    model2 = FakeModel(mob2)
    load_mob_state(model2, save_path)

    assert torch.allclose(mob2.expert_wealth, original_wealth, atol=1e-5)
    assert torch.allclose(mob2.expert_performance_ema, original_ema, atol=1e-5)
    assert torch.allclose(mob2.expert_baseline_loss, original_baseline, atol=1e-5)


def test_load_state_strict_mismatch(tmp_path, tiny_config):
    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    model = FakeModel(mob)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(model, save_path)

    different_config = MoBConfig(
        num_experts=4,
        top_k=1,
        hidden_dim=32,
        intermediate_dim=64,
        adapter_rank=4,
        adapter_alpha=4.0,
        use_shared_base=True,
    )
    mob2 = MixtureOfBidders(different_config)
    model2 = FakeModel(mob2)

    with pytest.raises(ValueError, match="Expert count mismatch"):
        load_mob_state(model2, save_path, strict=True)


def test_tracking_records_history(mob_layer, random_hidden_states):
    mob_layer.start_tracking()
    mob_layer(random_hidden_states)
    mob_layer(random_hidden_states)

    history = mob_layer.get_wealth_history()
    assert len(history) >= 2
    mob_layer.stop_tracking()


def test_training_and_eval_produce_same_output(tiny_config):
    mob = MixtureOfBidders(tiny_config)
    hidden = torch.randn(1, 4, tiny_config.hidden_dim)

    mob.train()
    train_out = mob(hidden, update_wealth=False)

    mob.eval()
    eval_out = mob(hidden, update_wealth=False)

    assert torch.allclose(train_out, eval_out, atol=1e-5)


def test_sparse_forward_skips_unselected_experts(tiny_config):
    mob = MixtureOfBidders(tiny_config)
    mob.eval()
    mob.expert_usage_count.zero_()
    hidden = torch.randn(1, 1, tiny_config.hidden_dim)
    mob(hidden, update_wealth=True)
    nonzero_experts = (mob.expert_usage_count > 0).sum().item()
    assert nonzero_experts == tiny_config.top_k


def test_apply_mob_replaces_mlp(tiny_config):
    class FakeFFN(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class FakeTransformerLayer(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim):
            super().__init__()
            self.mlp = FakeFFN(hidden_dim, intermediate_dim)

    class FakeTransformer(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim, num_layers):
            super().__init__()
            self.layers = nn.ModuleList(
                [FakeTransformerLayer(hidden_dim, intermediate_dim) for _ in range(num_layers)]
            )

    class FakeModel(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim, num_layers):
            super().__init__()
            self.model = FakeTransformer(hidden_dim, intermediate_dim, num_layers)

    hd, inter = tiny_config.hidden_dim, tiny_config.intermediate_dim
    fake_model = FakeModel(hd, inter, num_layers=8)

    target_layer = 5
    apply_mob_to_model(fake_model, tiny_config, layers_to_modify=[target_layer])

    replaced = fake_model.model.layers[target_layer].mlp
    assert isinstance(replaced, MixtureOfBidders)

    untouched = fake_model.model.layers[0].mlp
    assert not isinstance(untouched, MixtureOfBidders)


def test_restored_wealth_is_clamped_to_the_configured_bounds(tmp_path, tiny_config, caplog):
    """A checkpoint is the fourth writer to expert_wealth, and the only external one.

    The three update paths clamp; `load_mob_state` did not. The auction divides each
    winner's price by its own wealth, so a restored zero or negative entry reaches
    that division rather than the boundary validation in `MoBConfig` — and a
    negative one small enough to sit inside the payment-negativity tolerance would
    reach that division instead of an invariant; `auction.py` records the window and
    the price it produces.
    """

    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(FakeModel(mob), save_path)

    # A checkpoint that a truncated write, a hand edit, or an older config could
    # plausibly produce: one bankrupt expert, one above the ceiling.
    state = torch.load(save_path, weights_only=True)
    state["layer_0"]["wealth"] = [-1e-4, tiny_config.max_wealth * 10]
    torch.save(state, save_path)

    restored = MixtureOfBidders(tiny_config)
    with caplog.at_level(logging.WARNING, logger="mob.utils"):
        load_mob_state(FakeModel(restored), save_path)

    assert (restored.expert_wealth >= tiny_config.min_wealth).all()
    assert (restored.expert_wealth <= tiny_config.max_wealth).all()
    assert any("repaired" in message for message in caplog.messages), (
        "every other rejection in load_mob_state logs; this one must too"
    )


def test_restored_wealth_repairs_non_finite_entries(tmp_path, tiny_config, caplog):
    """`clamp_` passes NaN through, so a diverged run's checkpoint would survive it.

    That matters most where the guard is thinnest: under `-O` the auction's
    finiteness assert is compiled out, so a NaN wealth would reach the bid silently.
    A non-finite entry is reset to `initial_wealth` rather than to a bound, because
    its true value is unknown rather than extreme.
    """

    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(FakeModel(mob), save_path)

    state = torch.load(save_path, weights_only=True)
    state["layer_0"]["wealth"] = [float("nan"), float("inf")]
    torch.save(state, save_path)

    restored = MixtureOfBidders(tiny_config)
    with caplog.at_level(logging.WARNING, logger="mob.utils"):
        load_mob_state(FakeModel(restored), save_path)

    assert torch.isfinite(restored.expert_wealth).all()
    assert restored.expert_wealth[0].item() == pytest.approx(tiny_config.initial_wealth)
    assert restored.expert_wealth[1].item() == pytest.approx(tiny_config.max_wealth)
    assert any("repaired" in message for message in caplog.messages), (
        "a silent repair is the one thing this function does not do elsewhere"
    )
