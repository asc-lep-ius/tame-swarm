"""Mechanisms that are supposed to be active are: the generalisation of #9's lesson (#6).

Every no-op in this codebase so far was silent. #9's payments were identically
zero under a clamp; #14's coupling shipped inert behind a one-step warmup; #12's
``eval_steps`` was declared and never read; #10's ``capability_subspace`` was a
constructor argument nothing assigned. Each test here fails if the mechanism it
names is inert, and each is paired with the inert state it fails in.

The training-loop half of the list -- ``set_coupling_step`` is actually called by
``train_step`` -- lives with the trainer's own tests
(``tests/test_wealth_updates.py::test_train_step_reports_router_z_loss_and_sets_coupling_step``,
mutation-verified in #14) and the ``TrainingConfig`` field scan in
``tests/test_training_config_usage.py``; neither is repeated here.
"""

from dataclasses import fields

import pytest
import torch

from homeostat import CognitiveHomeostat
from mob import MixtureOfBidders, MoBConfig, SteeringCouplingConfig
from mob.auction import ROUTING_SHARE_PROPORTIONAL, VCGAuctioneer
from pid_controller import PIDConfig
from steering import SteeringConfig, SteeringVector

from .auction_mutations import pre_nine_payments
from .config_reads import read_names
from .conftest import TINY_HIDDEN_DIM, build_tiny_causal_lm

# --- Every config field is read by something -----------------------------------------

CONFIGS = (MoBConfig, SteeringConfig, SteeringCouplingConfig, PIDConfig)


@pytest.mark.parametrize(
    ("config_class", "field"),
    [(config_class, spec.name) for config_class in CONFIGS for spec in fields(config_class)],
    ids=lambda value: value.__name__ if isinstance(value, type) else value,
)
def test_every_config_field_is_read_somewhere(config_class, field):
    """A field read only by its own dataclass is a field that steers nothing.

    The scanner's own failability is pinned in ``test_training_config_usage``; it
    is the same scanner.
    """
    assert field in read_names(), (
        f"{config_class.__name__}.{field} is declared and never read outside its own "
        "validation. A field that looks like it steers the mechanism and does not is "
        "the defect this test exists to prevent -- wire it up or delete it."
    )


# --- The auction charges a price ---------------------------------------------------------

# The default economy at a hidden size a test can afford: eight experts, top-2,
# the served wealth band, the exploration slot and the value objective all on.
REALISTIC = MoBConfig(hidden_dim=64, intermediate_dim=128, adapter_rank=8, adapter_alpha=8.0)


def _payments_at(config: MoBConfig) -> torch.Tensor:
    torch.manual_seed(0)
    mob = MixtureOfBidders(config)
    mob.eval()
    mob(torch.randn(2, 16, config.hidden_dim), update_wealth=False)
    assert mob.last_stats is not None
    outcome = mob.gate(mob.last_stats.confidences, mob.expert_wealth)  # type: ignore[operator]
    assert outcome.payments is not None
    assert mob.last_stats.mean_payment is not None
    return outcome.payments


MEANINGFUL_PAYMENT = 1e-4


def _assert_strictly_positive(payments: torch.Tensor) -> None:
    assert bool((payments > 0).all()), payments.min()
    assert float(payments.mean()) > MEANINGFUL_PAYMENT, payments.mean()


def test_vcg_payments_are_strictly_positive_at_the_default_economy():
    """Not merely non-negative: with eight bidders for two slots every winner displaces someone."""
    _assert_strictly_positive(_payments_at(REALISTIC))


def test_the_payment_check_fails_on_the_pre_nine_auction(monkeypatch):
    """Zero up to the rounding of two equal welfare sums, which the clamp turned positive."""
    monkeypatch.setattr(VCGAuctioneer, "_compute_vcg_payments", pre_nine_payments)

    with pytest.raises(AssertionError):
        _assert_strictly_positive(_payments_at(REALISTIC))


# --- Routing mixes more than one expert ----------------------------------------------------

EFFECTIVE_EXPERTS_FLOOR = 1.5


def _effective_experts(config: MoBConfig) -> float:
    torch.manual_seed(0)
    mob = MixtureOfBidders(config)
    mob.eval()
    mob(torch.randn(2, 16, config.hidden_dim), update_wealth=False)
    assert mob.last_stats is not None
    return float(mob.last_stats.routing.effective_experts)


def test_routing_mixes_more_than_one_and_a_half_experts_at_the_default_configuration():
    """#11: ``top_k=2`` paying for two experts and using one is the collapse this metric shows."""
    assert _effective_experts(REALISTIC) > EFFECTIVE_EXPERTS_FLOOR


def test_the_effective_count_falls_below_the_floor_on_a_sharpened_gate():
    """The pairing: the own-bid-weighted share at a low temperature approaches argmax."""
    collapsed = MoBConfig(
        hidden_dim=64,
        intermediate_dim=128,
        adapter_rank=8,
        adapter_alpha=8.0,
        routing_share=ROUTING_SHARE_PROPORTIONAL,
        routing_temperature=0.01,
    )

    assert _effective_experts(collapsed) < EFFECTIVE_EXPERTS_FLOOR


# --- The capability subspace reaches the injected direction ----------------------------------

LAYERS = (1, 2)


def _attached_homeostat(orthogonal_projection: bool) -> tuple[CognitiveHomeostat, torch.Tensor]:
    torch.manual_seed(0)
    model = build_tiny_causal_lm()
    raw = torch.randn(TINY_HIDDEN_DIM)
    config = SteeringConfig(
        steering_layers=list(LAYERS), adaptive=False, orthogonal_projection=orthogonal_projection
    )
    homeostat = CognitiveHomeostat(config)
    homeostat.add_steering_vectors(
        {layer: SteeringVector("truthful", raw.clone(), layer) for layer in LAYERS}
    )
    # A basis the raw direction overlaps by about 0.8 in cosine, so the projection
    # removes a measurable part of it and leaves well above the 5% fallback floor.
    leaning = raw / raw.norm() + 0.7 * torch.randn(TINY_HIDDEN_DIM)
    basis = torch.linalg.qr(torch.stack([leaning, torch.randn(TINY_HIDDEN_DIM)], dim=1))[0].T
    homeostat.set_capability_subspaces({layer: basis for layer in LAYERS})
    homeostat.attach_to_model(model)
    return homeostat, raw / raw.norm()


def test_the_capability_subspace_is_wired_through_to_what_the_hooks_inject():
    """#10's Claim 3, end to end: the hook injects the projected direction, not the raw vector."""
    homeostat, raw = _attached_homeostat(orthogonal_projection=True)

    for layer in LAYERS:
        injected = homeostat.hooks[layer]._direction(torch.device("cpu"), torch.float32)
        subspace = homeostat.capability_subspaces[layer]
        assert torch.allclose(subspace @ injected, torch.zeros(subspace.shape[0]), atol=1e-5)
        assert not torch.allclose(injected, raw, atol=1e-3)
        assert 0.0 < homeostat.get_capability_retention()[layer] < 1.0
    homeostat.detach_from_model()


def test_the_projection_switch_is_live():
    """The pairing: with the projection off the hook injects the raw direction, all of it."""
    homeostat, raw = _attached_homeostat(orthogonal_projection=False)

    for layer in LAYERS:
        injected = homeostat.hooks[layer]._direction(torch.device("cpu"), torch.float32)
        assert torch.allclose(injected, raw, atol=1e-6)
        assert homeostat.get_capability_retention()[layer] == 1.0
    homeostat.detach_from_model()
