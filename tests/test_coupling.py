import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mob import (
    CouplingMetrics,
    MixtureOfBidders,
    MoBConfig,
    SteeringCoupling,
    SteeringCouplingConfig,
)


def test_public_tame_mob_imports_work_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tame.mob import MixtureOfBidders, SteeringCouplingConfig",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _clone_mob_with_same_weights(source: MixtureOfBidders) -> MixtureOfBidders:
    clone = MixtureOfBidders(source.config)
    clone.load_state_dict(source.state_dict())
    clone.eval()
    return clone


def test_step_zero_coupling_matches_uncoupled_baseline(tiny_config):
    torch.manual_seed(17)
    baseline = MixtureOfBidders(tiny_config)
    baseline.eval()
    coupled = _clone_mob_with_same_weights(baseline)
    hidden_states = torch.randn(1, 4, tiny_config.hidden_dim)

    coupled.attach_coupling(
        torch.randn(tiny_config.hidden_dim),
        SteeringCouplingConfig(
            hidden_dim=tiny_config.hidden_dim,
            coupling_beta=0.5,
            warmup_steps=4,
            max_coupling_fraction=0.1,
        ),
    )
    coupled.set_coupling_step(0)

    baseline_output = baseline(hidden_states, update_wealth=False)
    coupled_output = coupled(hidden_states, update_wealth=False)

    assert torch.allclose(coupled_output, baseline_output, atol=1e-6)
    assert coupled.last_stats is not None
    assert coupled.last_stats.coupling_metrics is not None
    assert float(coupled.last_stats.coupling_metrics.beta_effective.item()) == pytest.approx(0.0)


def test_attach_coupling_registers_trainable_parameters_and_moves_with_module(tiny_config):
    mob = MixtureOfBidders(tiny_config)

    coupling = mob.attach_coupling(
        torch.randn(tiny_config.hidden_dim),
        SteeringCouplingConfig(hidden_dim=tiny_config.hidden_dim),
    )

    coupling_parameters = {
        name: parameter
        for name, parameter in mob.named_parameters()
        if name.startswith("coupling.")
    }
    assert coupling_parameters
    assert all(parameter.requires_grad for parameter in coupling_parameters.values())
    assert "coupling" in dict(mob.named_modules())

    mob.to(dtype=torch.float64)

    assert next(coupling.parameters()).dtype == torch.float64
    assert coupling.steering_direction.dtype == torch.float64


def test_detach_coupling_removes_submodule_and_clears_stale_state(tiny_config):
    torch.manual_seed(23)
    baseline = MixtureOfBidders(tiny_config)
    baseline.eval()
    coupled = _clone_mob_with_same_weights(baseline)
    hidden_states = torch.randn(1, 4, tiny_config.hidden_dim)

    coupled.attach_coupling(torch.randn(tiny_config.hidden_dim))
    coupled.set_coupling_step(1)
    coupled(hidden_states, update_wealth=False)
    assert coupled.last_stats is not None
    assert coupled.last_stats.coupling_metrics is not None

    coupled.detach_coupling()

    assert "coupling" not in coupled._modules
    assert "coupling" not in dict(coupled.named_modules())
    assert not hasattr(coupled, "coupling")
    assert coupled.last_stats is None

    baseline_output = baseline(hidden_states, update_wealth=False)
    detached_output = coupled(hidden_states, update_wealth=False)

    assert torch.allclose(detached_output, baseline_output, atol=1e-6)
    assert coupled.last_stats is not None
    assert coupled.last_stats.coupling_metrics is None


def test_coupling_warmup_is_explicit_and_deterministic(tiny_config):
    coupling = SteeringCoupling(
        SteeringCouplingConfig(
            hidden_dim=tiny_config.hidden_dim,
            coupling_beta=0.4,
            warmup_steps=4,
        ),
        torch.randn(tiny_config.hidden_dim),
    )
    hidden_states = torch.randn(1, 2, tiny_config.hidden_dim)

    coupling.set_coupling_step(0)
    coupling(hidden_states)
    assert coupling.last_metrics is not None
    assert float(coupling.last_metrics.beta_effective.item()) == pytest.approx(0.0)

    coupling.set_coupling_step(2)
    coupling(hidden_states)
    assert coupling.last_metrics is not None
    assert float(coupling.last_metrics.beta_effective.item()) == pytest.approx(0.2)

    coupling.set_coupling_step(4)
    coupling(hidden_states)
    assert coupling.last_metrics is not None
    assert float(coupling.last_metrics.beta_effective.item()) == pytest.approx(0.4)

    coupling(hidden_states)
    assert coupling.last_metrics is not None
    assert float(coupling.last_metrics.beta_effective.item()) == pytest.approx(0.4)


def test_coupling_delta_norm_is_capped_per_token(tiny_config):
    max_fraction = 0.05
    coupling = SteeringCoupling(
        SteeringCouplingConfig(
            hidden_dim=tiny_config.hidden_dim,
            coupling_beta=1.0,
            warmup_steps=1,
            max_coupling_fraction=max_fraction,
        ),
        torch.ones(tiny_config.hidden_dim),
    )
    with torch.no_grad():
        coupling.projection.weight.copy_(torch.eye(tiny_config.hidden_dim) * 20.0)

    hidden_states = torch.randn(2, 3, tiny_config.hidden_dim)
    hidden_states[0, 0].zero_()
    coupling.set_coupling_step(1)

    coupled_states = coupling(hidden_states)
    delta = coupled_states - hidden_states
    hidden_norm = hidden_states.norm(dim=-1)
    delta_fraction = torch.where(
        hidden_norm > 0,
        delta.norm(dim=-1) / hidden_norm.clamp_min(1e-8),
        torch.zeros_like(hidden_norm),
    )

    assert torch.all(delta_fraction <= max_fraction + 1e-6)
    assert coupling.last_metrics is not None
    assert float(coupling.last_metrics.delta_norm_fraction_max.item()) <= max_fraction + 1e-6


def test_mob_stats_include_coupling_metrics_only_when_coupling_is_attached(
    mob_layer, random_hidden_states
):
    mob_layer(random_hidden_states, update_wealth=False)
    assert mob_layer.last_stats is not None
    assert mob_layer.last_stats.coupling_metrics is None

    mob_layer.attach_coupling(torch.randn(mob_layer.config.hidden_dim))
    mob_layer.set_coupling_step(1)
    mob_layer(random_hidden_states, update_wealth=False)

    assert mob_layer.last_stats is not None
    assert isinstance(mob_layer.last_stats.coupling_metrics, CouplingMetrics)


def test_active_coupling_preserves_vcg_auction_inputs() -> None:
    config = MoBConfig(
        num_experts=3,
        top_k=2,
        hidden_dim=4,
        intermediate_dim=8,
        adapter_rank=2,
        adapter_alpha=2.0,
        use_shared_base=True,
        use_vcg_payments=True,
        use_differentiable_routing=False,
        use_loss_feedback=True,
        use_local_quality=False,
        exploration_rate=0.0,  # the assertion below is on the auction's own allocation
    )
    baseline = MixtureOfBidders(config)
    coupled = MixtureOfBidders(config)
    coupled.load_state_dict(baseline.state_dict())

    with torch.no_grad():
        baseline.expert_wealth.copy_(torch.tensor([1.0, 1.3, 0.7]))
        coupled.expert_wealth.copy_(baseline.expert_wealth)
        for coefficient, confidence_head in zip(
            [1.2, -0.8, 0.4], baseline.confidence_heads, strict=True
        ):
            confidence_head.proj.weight.zero_()
            confidence_head.proj.weight[0, 0] = coefficient
            confidence_head.proj.bias.zero_()
        coupled.load_state_dict(baseline.state_dict())

    coupled.attach_coupling(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        SteeringCouplingConfig(
            hidden_dim=config.hidden_dim,
            coupling_beta=1.0,
            warmup_steps=1,
            max_coupling_fraction=2.0,
        ),
    )
    with torch.no_grad():
        coupled.coupling.projection.weight.zero_()
        coupled.coupling.projection.weight[0, 0] = 1.0
    coupled.set_coupling_step(1)

    hidden_states = torch.tensor([[[0.5, 0.1, 0.0, 0.0], [-0.25, 0.0, 0.2, 0.0]]])
    baseline.train()
    coupled.train()

    baseline(hidden_states, update_wealth=False)
    coupled(hidden_states, update_wealth=False)

    assert baseline.last_stats is not None
    assert coupled.last_stats is not None
    assert not torch.allclose(coupled.last_stats.confidences, baseline.last_stats.confidences)

    bids = coupled.last_stats.confidences * coupled.expert_wealth.view(1, 1, -1)
    expected_selected = torch.topk(bids, config.top_k, dim=-1).indices
    expected_payments = coupled.gate(
        coupled.last_stats.confidences,
        coupled.expert_wealth,
    ).payments

    assert torch.equal(coupled.last_stats.selected_experts, expected_selected)
    assert coupled._cached_payments is not None
    assert torch.allclose(coupled._cached_payments, expected_payments, atol=1e-6)
