"""The coupling's mechanism, asserted where it is non-trivial.

#2's acceptance criteria could not tell a working coupling from an inert one: with
the receptor zero-initialised, "detach restores the uncoupled forward" is true by
construction, and a warmup of one step made ``beta_effective`` a constant. Every
measurement below is paired with the state in which it must fail -- ``inert()``
forces ``beta_effective`` to zero, and each test asserts its own statistic
collapses under it -- so a test here can fail, and is therefore a test.
"""

import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from coupling import DEFAULT_WARMUP_STEPS, MIN_WARMUP_STEPS_FOR_RAMP
from mob import (
    CouplingMetrics,
    MixtureOfBidders,
    MoBConfig,
    SteeringCoupling,
    SteeringCouplingConfig,
)

HIDDEN = 32


@contextmanager
def inert() -> Iterator[None]:
    """Force ``beta_effective`` to zero: the hook attached and doing nothing."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            SteeringCoupling,
            "_effective_beta",
            lambda self, hidden: torch.zeros((), device=hidden.device, dtype=hidden.dtype),
        )
        yield


def _unit(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(dim, generator=generator)
    return vector / vector.norm()


def _coupling(
    direction: torch.Tensor, detector: torch.Tensor | None = None, **overrides
) -> SteeringCoupling:
    settings = dict(
        hidden_dim=HIDDEN, coupling_beta=1.0, warmup_steps=10, max_coupling_fraction=0.5
    )
    settings.update(overrides)
    coupling = SteeringCoupling(SteeringCouplingConfig(**settings), direction)
    if detector is not None:
        with torch.no_grad():
            coupling.detector.copy_(detector)
    coupling.set_coupling_step(settings["warmup_steps"])
    coupling.eval()
    return coupling


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


# --- Warmup ----------------------------------------------------------------


def test_default_warmup_is_the_specified_hundred_steps() -> None:
    assert DEFAULT_WARMUP_STEPS == 100
    assert SteeringCouplingConfig(hidden_dim=4).warmup_steps == 100


@pytest.mark.parametrize("warmup_steps", [1, MIN_WARMUP_STEPS_FOR_RAMP - 1])
def test_config_warns_when_warmup_is_effectively_bypassed(warmup_steps, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="coupling"):
        SteeringCouplingConfig(hidden_dim=4, warmup_steps=warmup_steps)

    assert any("step change" in record.message for record in caplog.records)


@pytest.mark.parametrize("warmup_steps", [MIN_WARMUP_STEPS_FOR_RAMP, DEFAULT_WARMUP_STEPS])
def test_config_is_quiet_when_warmup_is_a_ramp(warmup_steps, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="coupling"):
        SteeringCouplingConfig(hidden_dim=4, warmup_steps=warmup_steps)

    assert not caplog.records


def _beta_at(coupling: SteeringCoupling, step: int, hidden: torch.Tensor) -> float:
    coupling.set_coupling_step(step)
    coupling(hidden)
    assert coupling.last_metrics is not None
    return float(coupling.last_metrics.beta_effective.item())


def test_warmup_ramps_strictly_through_intermediate_steps() -> None:
    """Zero at step 0, strictly increasing to ``coupling_beta`` at ``warmup_steps``, flat after."""
    warmup, beta = 8, 0.4
    coupling = _coupling(_unit(0), warmup_steps=warmup, coupling_beta=beta)
    hidden = torch.randn(1, 2, HIDDEN)

    betas = [_beta_at(coupling, step, hidden) for step in range(warmup + 4)]

    assert betas[0] == 0.0
    for step in range(1, warmup + 1):
        assert betas[step] > betas[step - 1]
        assert betas[step] == pytest.approx(beta * step / warmup)
    assert betas[warmup:] == pytest.approx([beta] * 4)

    with inert():
        assert all(_beta_at(coupling, step, hidden) == 0.0 for step in range(warmup + 4))


# --- Parity, where it is not true by construction --------------------------


def test_step_zero_coupling_matches_uncoupled_baseline(tiny_config):
    torch.manual_seed(17)
    baseline = MixtureOfBidders(tiny_config)
    baseline.eval()
    coupled = MixtureOfBidders(tiny_config)
    coupled.load_state_dict(baseline.state_dict())
    coupled.eval()
    hidden_states = torch.randn(1, 4, tiny_config.hidden_dim)

    coupled.attach_coupling(
        torch.randn(tiny_config.hidden_dim),
        SteeringCouplingConfig(
            hidden_dim=tiny_config.hidden_dim,
            coupling_beta=0.5,
            warmup_steps=40,
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


def test_detach_restores_the_uncoupled_forward_after_the_receptor_has_trained():
    """#2's parity criterion, asserted after the receptor has moved off zero.

    The receptor is trained through the real path -- the heads' value objective
    after a settled step -- so the state detached from is one training produced,
    not one a test wrote. Before detaching, the coupled forward must *differ*
    from the uncoupled one, and only because the hook is live.
    """
    from tests.test_wealth_updates import _build_training_mob, _settle

    torch.manual_seed(5)
    mob = _build_training_mob()
    mob.attach_coupling(
        _unit(3),
        SteeringCouplingConfig(
            hidden_dim=HIDDEN, coupling_beta=1.0, warmup_steps=10, max_coupling_fraction=0.5
        ),
    )
    mob.set_coupling_step(10)
    optimizer = torch.optim.SGD(mob.coupling.parameters(), lr=0.5)
    for _ in range(3):
        _settle(mob, torch.randn(1, 8, HIDDEN))
        mob.get_confidence_calibration_loss().backward()
        optimizer.step()
        optimizer.zero_grad()
    assert float(mob.coupling.detector.detach().norm()) > 0.0, "the value objective must move it"

    baseline = MixtureOfBidders(mob.config)
    baseline.load_state_dict(
        {key: value for key, value in mob.state_dict().items() if not key.startswith("coupling.")}
    )
    baseline.eval()
    mob.eval()
    hidden = torch.randn(1, 8, HIDDEN)
    baseline_output = baseline(hidden, update_wealth=False)
    assert baseline.last_stats is not None
    baseline_reports = baseline.last_stats.confidences

    # The auction splits the output evenly among winners, so the reports are where
    # a live coupling shows first: they must differ, and only because it is live.
    mob(hidden, update_wealth=False)
    assert mob.last_stats is not None
    assert not torch.allclose(mob.last_stats.confidences, baseline_reports)
    with inert():
        mob(hidden, update_wealth=False)
        assert mob.last_stats is not None
        assert torch.allclose(mob.last_stats.confidences, baseline_reports)

    mob.detach_coupling()

    assert torch.equal(mob(hidden, update_wealth=False), baseline_output)
    assert mob.last_stats is not None
    assert torch.equal(mob.last_stats.confidences, baseline_reports)


# --- The cap, on a path where it binds ------------------------------------


def test_norm_cap_binds_where_the_raw_delta_exceeds_it():
    """Construct a delta larger than the cap and check the cap is what comes out."""
    cap = 0.05
    direction = torch.eye(HIDDEN)[0]
    gain = 20.0
    hidden = torch.randn(2, 4, HIDDEN)
    hidden[..., 0] += 3.0  # a strong goal component: raw fraction = gain * |h0| / |h|
    hidden[0, 0].zero_()
    hidden_norm = hidden.norm(dim=-1)
    nonzero = hidden_norm > 0

    raw_fraction = gain * hidden[..., 0].abs() / hidden_norm.clamp_min(1e-8)
    uncapped = _coupling(direction, gain * direction, max_coupling_fraction=1e6)
    uncapped_fraction = (uncapped(hidden) - hidden).norm(dim=-1) / hidden_norm.clamp_min(1e-8)
    assert torch.allclose(uncapped_fraction[nonzero], raw_fraction[nonzero], atol=1e-5)
    assert torch.all(raw_fraction[nonzero] > cap), "the fixture must exceed the cap everywhere"

    capped = _coupling(direction, gain * direction, max_coupling_fraction=cap)
    delta = capped(hidden) - hidden
    fraction = delta.norm(dim=-1) / hidden_norm.clamp_min(1e-8)

    assert torch.allclose(fraction[nonzero], torch.full_like(fraction[nonzero], cap), atol=1e-5)
    assert torch.all(delta[~nonzero] == 0.0)
    cosine = torch.nn.functional.cosine_similarity(
        delta[nonzero], (uncapped(hidden) - hidden)[nonzero]
    )
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5), (
        "the cap rescales, never turns"
    )
    assert capped.last_metrics is not None
    assert float(capped.last_metrics.delta_norm_fraction_max.item()) == pytest.approx(cap, abs=1e-5)

    with inert():
        assert torch.all(capped(hidden) == hidden)


# --- The step guard ----------------------------------------------------------


def test_training_forward_requires_the_step_to_have_been_set():
    """A coupling whose counter never advances is a no-op that looks attached."""
    coupling = SteeringCoupling(SteeringCouplingConfig(hidden_dim=HIDDEN), _unit(1))
    hidden = torch.randn(1, 2, HIDDEN)

    coupling.train()
    with pytest.raises(RuntimeError, match="set_coupling_step"):
        coupling(hidden)

    coupling.eval()
    coupling(hidden)  # inference serves whatever step the checkpoint carries

    coupling.train()
    coupling.set_coupling_step(0)
    coupling(hidden)


def test_a_training_mob_refuses_to_forward_a_coupling_nobody_stepped(training_mob_layer):
    training_mob_layer.attach_coupling(_unit(2))

    with pytest.raises(RuntimeError, match="set_coupling_step"):
        training_mob_layer(torch.randn(1, 4, HIDDEN))

    training_mob_layer.set_coupling_step(0)
    training_mob_layer(torch.randn(1, 4, HIDDEN))


# --- Influence direction, against a matched random control -----------------

# The stated minimum effect: the goal-aligned experts' mean routing share must
# rise by at least this much under the goal direction ...
MIN_ALIGNED_SHIFT = 0.05
# ... and by at least this multiple of the largest shift any of the matched
# random directions produces.
RANDOM_CONTROL_MARGIN = 3.0
RANDOM_CONTROLS = 8


def _contested_mob(direction: torch.Tensor, aligned: torch.Tensor) -> MixtureOfBidders:
    """Four experts whose heads lean weakly with or against the goal direction.

    Weakly, so routing is contested: heads that already read the goal direction
    strongly win every token and leave the coupling nothing to shift.
    """
    config = MoBConfig(
        num_experts=4,
        top_k=2,
        hidden_dim=HIDDEN,
        intermediate_dim=64,
        adapter_rank=4,
        adapter_alpha=4.0,
        use_shared_base=True,
        use_vcg_payments=True,
        exploration_rate=0.0,
    )
    mob = MixtureOfBidders(config)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for expert, head in enumerate(mob.confidence_heads):
            lean = 0.5 if aligned[expert] else -0.5
            head.proj.weight.copy_(
                (0.2 * torch.randn(HIDDEN, generator=generator) + lean * direction).unsqueeze(0)
            )
            head.proj.bias.zero_()
    mob.eval()
    return mob


def _aligned_share(mob: MixtureOfBidders, hidden: torch.Tensor, aligned: torch.Tensor) -> float:
    """Mean over tokens of the routing weight the goal-aligned experts receive."""
    mob(hidden, update_wealth=False)
    stats = mob.last_stats
    assert stats is not None
    share = torch.zeros(*stats.selected_experts.shape[:-1], mob.config.num_experts)
    share.scatter_(-1, stats.selected_experts, stats.routing_weights.float())
    return float(share[..., aligned].sum(dim=-1).mean().item())


def _aligned_shift(
    mob: MixtureOfBidders, hidden: torch.Tensor, aligned: torch.Tensor, direction: torch.Tensor
) -> float:
    """Gain in aligned share under a coupling seeded with ``direction``, receptor ``direction``.

    The receptor is the one a coupling seeded with this direction learns first --
    perceive the stream's own component along it -- so the random controls are
    matched in construction, not only in norm.
    """
    baseline = _aligned_share(mob, hidden, aligned)
    mob.attach_coupling(
        direction,
        SteeringCouplingConfig(
            hidden_dim=HIDDEN, coupling_beta=1.0, warmup_steps=10, max_coupling_fraction=0.5
        ),
    )
    with torch.no_grad():
        mob.coupling.detector.copy_(direction)
    mob.set_coupling_step(10)
    shift = _aligned_share(mob, hidden, aligned) - baseline
    mob.detach_coupling()
    return shift


def test_coupling_shifts_routing_toward_goal_aligned_experts_more_than_random_directions():
    """Routing moves toward the experts that read the goal direction, by a stated margin.

    On a stream carrying a positive goal component -- the steered regime the
    residual-stream injection produces -- the perceived state's goal component is
    amplified, so heads leaning with the goal report more and win more. Random
    directions of equal norm, coupled the same way, do not know which experts
    lean with the goal, so their shift is noise around zero.
    """
    direction = _unit(7)
    aligned = torch.tensor([True, True, False, False])
    mob = _contested_mob(direction, aligned)
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(8, 32, HIDDEN, generator=generator) + 1.0 * direction

    goal_shift = _aligned_shift(mob, hidden, aligned, direction)
    random_shifts = [
        _aligned_shift(mob, hidden, aligned, _unit(100 + k)) for k in range(RANDOM_CONTROLS)
    ]

    assert goal_shift >= MIN_ALIGNED_SHIFT, (goal_shift, random_shifts)
    assert goal_shift >= RANDOM_CONTROL_MARGIN * max(random_shifts), (goal_shift, random_shifts)

    with inert():
        assert _aligned_shift(mob, hidden, aligned, direction) == 0.0


# --- Wiring ----------------------------------------------------------------


def test_attach_coupling_registers_the_receptor_and_moves_with_module(tiny_config):
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
    assert set(coupling_parameters) == {"coupling.detector"}
    assert coupling_parameters["coupling.detector"].shape == (tiny_config.hidden_dim,)
    assert all(parameter.requires_grad for parameter in coupling_parameters.values())
    assert "coupling" in dict(mob.named_modules())
    assert [p for p in mob.routing_parameters() if p is coupling.detector]

    mob.to(dtype=torch.float64)

    assert coupling.detector.dtype == torch.float64
    assert coupling.steering_direction.dtype == torch.float64


def test_detach_coupling_removes_submodule_and_clears_stale_state(tiny_config):
    torch.manual_seed(23)
    baseline = MixtureOfBidders(tiny_config)
    baseline.eval()
    coupled = MixtureOfBidders(tiny_config)
    coupled.load_state_dict(baseline.state_dict())
    coupled.eval()
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
    assert coupled.routing_parameters() == list(coupled.confidence_heads.parameters())

    baseline_output = baseline(hidden_states, update_wealth=False)
    detached_output = coupled(hidden_states, update_wealth=False)

    assert torch.allclose(detached_output, baseline_output, atol=1e-6)
    assert coupled.last_stats is not None
    assert coupled.last_stats.coupling_metrics is None


def test_metrics_report_the_receptor_norm():
    coupling = _coupling(_unit(4), 3.0 * _unit(5))

    coupling(torch.randn(1, 2, HIDDEN))

    assert coupling.last_metrics is not None
    assert float(coupling.last_metrics.detector_norm.item()) == pytest.approx(3.0)


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
            warmup_steps=10,
            max_coupling_fraction=2.0,
        ),
    )
    with torch.no_grad():
        coupled.coupling.detector.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    coupled.set_coupling_step(10)

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
