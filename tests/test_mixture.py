import torch
import torch.nn as nn

import pytest

from mob import MoBConfig, MixtureOfBidders, apply_mob_to_model, save_mob_state, load_mob_state


def test_forward_output_shape(mob_layer, random_hidden_states):
    out = mob_layer(random_hidden_states)
    assert out.shape == random_hidden_states.shape


def test_forward_no_nan_inf(mob_layer, random_hidden_states):
    out = mob_layer(random_hidden_states)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_updates_last_stats(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    expected_keys = {
        "confidences",
        "selected_experts",
        "routing_weights",
        "expert_wealth",
        "expert_usage",
        "expert_performance",
    }
    assert expected_keys.issubset(mob_layer.last_stats.keys())


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
            self.layers = nn.ModuleList([
                FakeTransformerLayer(hidden_dim, intermediate_dim)
                for _ in range(num_layers)
            ])

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
