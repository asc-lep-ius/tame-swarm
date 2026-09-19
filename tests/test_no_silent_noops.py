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

import os
from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from homeostat import CognitiveHomeostat
from mob import MixtureOfBidders, MoBConfig, SteeringCouplingConfig, apply_mob_to_model
from mob.auction import ROUTING_SHARE_PROPORTIONAL, VCGAuctioneer
from parity import ArmFingerprint, ParityError, assert_parity
from pid_controller import PIDConfig
from readiness import AUTONOMIES, ReadinessConfig, granted_flags
from rotating_stream import build_rotating_stream
from steering import SteeringConfig, SteeringVector
from viability_margins import (
    CALLER_TRAINER,
    FrozenBase,
    MetricSet,
    ViabilityError,
    ViabilityMargins,
    measure_viability,
    score_predictions,
)

from .arm_fingerprints import BASE
from .auction_mutations import pre_nine_payments
from .config_reads import read_names
from .conftest import TINY_HIDDEN_DIM, build_tiny_causal_lm
from .rotating_fixtures import CUTOFF, MAX_SEQ_LENGTH, TODAY, canary_manifest, stream_manifest

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


# --- The readiness register's gates are off, and a run says which it granted ------------

# ``ReadinessConfig`` (``tame/readiness.py``) is the one config #46 puts the gate
# flags in, and the call site that will read it is the viability tissue of #42 --
# ``tame/viability.py`` in the Phase 2 plan, the lift-out of ``homeostat.py``
# generalised from a projection onto a direction to a viability margin. That
# module does not exist yet, so the config-read scanner above cannot cover these
# fields and ``ReadinessConfig`` is deliberately not in ``CONFIGS``: adding it
# today would assert a read nothing can yet make. When #42 lands and reads them,
# it goes in ``CONFIGS`` and this section keeps the other half of the guarantee.
#
# The no-op this section names is the autonomy's own: a gate that is on by
# default grants silently, and a gate the fingerprint does not carry makes a run
# that granted it indistinguishable from one that did not -- the difference
# between "the register schedules this" and "the register happened".


@pytest.mark.parametrize("flag", granted_flags())
def test_the_gate_flag_is_off_by_default(flag):
    assert getattr(ReadinessConfig(), flag) is False, (
        f"{flag} is on in the default ReadinessConfig. Every autonomy in "
        "docs/readiness-register.md is off until the issue that earns it turns it on."
    )


@pytest.mark.parametrize("flag", granted_flags())
def test_the_gate_flag_is_fingerprinted_and_asserted_at_parity(flag):
    """In ``ArmFingerprint`` under the same name, defaulted off, and a confound when it differs."""
    spec = {field.name: field for field in fields(ArmFingerprint)}
    assert flag in spec, f"{flag} gates an autonomy and is not in ArmFingerprint"
    assert spec[flag].default is False

    with pytest.raises(ParityError, match=flag):
        assert_parity([BASE, replace(BASE, router="softmax", **{flag: True})])


def test_the_default_config_grants_nothing():
    """The pairing: ``granted()`` is what a run would report, and today it is empty."""
    assert ReadinessConfig().granted() == ()
    assert ReadinessConfig(autonomy_dormancy=True).granted() == ("autonomy_dormancy",)


def test_every_flag_in_the_register_is_checked_here():
    """A row added to ``AUTONOMIES`` with a flag must arrive with its two checks.

    The parametrisations above read ``granted_flags()``, so this only has to fail
    when the register and the config part company -- a flag scheduled in the
    register and never declared in the config gates nothing at all.
    """
    declared = {spec.name for spec in fields(ReadinessConfig)}
    scheduled = {autonomy.flag for autonomy in AUTONOMIES if autonomy.flag is not None}

    assert scheduled == declared, (
        "docs/readiness-register.md's flags and ReadinessConfig's fields must be the same set; "
        f"register-only {sorted(scheduled - declared)}, config-only {sorted(declared - scheduled)}"
    )


def test_the_xdist_thread_pin_is_active_in_this_worker():
    """#51's speedup is the pin in conftest, and deleting it is silent.

    ``-n auto`` on its own claims the cores twice -- xdist sizes its worker count
    from the box and torch sizes its intra-op pool from the box inside each of
    those workers -- which turned a 161 s suite into 419 s, and into a run that
    had not finished after 21 minutes at load average 135 when it was reproduced
    during review. Nothing fails when the pin goes: CI gets five to twelve times
    slower and nobody is told, which is exactly this module's subject.
    """
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("serial runs keep torch's default pool; the pin is for workers")

    assert torch.get_num_threads() == 1


# --- The frozen base is read-only, and is never the model it is measured against ------------

# ``viability_margins.FrozenBase`` (#41) is the copy of the base every viability
# margin is a margin *against*. The no-op it can fail into is the quietest in this
# module: a base that is the served model, or that shares its parameters, makes
# every margin identically zero -- which is also what a healthy organism at parity
# with its base looks like. A core (#42) regulating on that regulates on nothing,
# and nothing fails. The second half is the freeze itself: a base left trainable
# is a second model being trained on the held-out stream, one gradient at a time.


def _rotation(fake_tokenizer):
    return build_rotating_stream(
        stream_manifest(),
        canary_manifest(),
        fake_tokenizer,
        MAX_SEQ_LENGTH,
        cutoff=CUTOFF,
        today=TODAY,
    )


def _organism_and_base(config: MoBConfig):
    torch.manual_seed(0)
    pristine = build_tiny_causal_lm()
    base = FrozenBase.freeze(deepcopy(pristine))
    return apply_mob_to_model(pristine, config, layers_to_modify=[1, 2]), base


def test_the_frozen_base_takes_no_gradient_from_a_margin_pass(tiny_mob_config, fake_tokenizer):
    model, base = _organism_and_base(tiny_mob_config)

    measure_viability(
        model,
        base,
        _rotation(fake_tokenizer),
        batch_size=8,
        device=torch.device("cpu"),
        caller=CALLER_TRAINER,
    )

    for name, tensor in base.module.named_parameters():
        assert not tensor.requires_grad, name
        assert tensor.grad is None, name


def test_a_base_that_was_not_frozen_does_take_one(fake_tokenizer):
    """The pairing: the same items through the same forward leave a gradient.

    Without it, the assertion above passes on a model whose parameters no gradient
    could reach for reasons of its own -- which is a test of the fixture.
    """
    unfrozen = build_tiny_causal_lm()
    stream = _rotation(fake_tokenizer)

    outputs = unfrozen(input_ids=stream.input_ids, attention_mask=stream.attention_mask)
    outputs.logits.sum().backward()

    assert any(tensor.grad is not None for tensor in unfrozen.parameters())


def test_the_organism_is_refused_as_its_own_frozen_base(tiny_mob_config, fake_tokenizer):
    """Both ways it could be handed over: converted, and as the very object measured."""
    model, base = _organism_and_base(tiny_mob_config)

    with pytest.raises(ViabilityError, match="MoB layers"):
        FrozenBase.freeze(model)

    base.module = model
    with pytest.raises(ViabilityError, match="every margin would be identically zero"):
        measure_viability(
            model,
            base,
            _rotation(fake_tokenizer),
            batch_size=8,
            device=torch.device("cpu"),
            caller=CALLER_TRAINER,
        )


def test_a_base_that_is_the_model_reads_as_a_perfectly_healthy_organism(
    tiny_mob_config, fake_tokenizer
):
    """The inert state the refusal above exists to prevent, shown rather than described."""
    model, _ = _organism_and_base(tiny_mob_config)
    predictions = score_predictions(model, _rotation(fake_tokenizer), 8, torch.device("cpu"))
    metrics = MetricSet.of(predictions.stream)

    margins = ViabilityMargins.between(metrics, metrics)

    assert (margins.accuracy, margins.calibration, margins.brier, margins.selective) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
