from dataclasses import fields as dc_fields

import pytest

from mob import ROUTER_AUCTION, ROUTER_SOFTMAX, MoBConfig
from mob.mob_config import AUCTION_ONLY_FIELDS


def test_default_values_match_expected():
    cfg = MoBConfig()
    assert cfg.num_experts == 8
    assert cfg.top_k == 2
    assert cfg.hidden_dim == 4096
    assert cfg.intermediate_dim == 14336
    assert cfg.initial_wealth == 75.0
    assert cfg.wealth_decay == 0.997
    assert cfg.min_wealth == 15.0
    assert cfg.max_wealth == 750.0
    assert cfg.use_vcg_payments is True
    assert cfg.use_shared_base is True
    assert cfg.adapter_rank == 64
    assert cfg.adapter_alpha == 16.0
    assert cfg.use_differentiable_routing is True
    assert cfg.routing_share == "uniform"
    assert cfg.routing_temperature == 1.0


def test_config_is_mutable():
    cfg = MoBConfig(num_experts=4, top_k=1, hidden_dim=512, intermediate_dim=1024)
    assert cfg.num_experts == 4
    assert cfg.top_k == 1
    assert cfg.hidden_dim == 512
    assert cfg.intermediate_dim == 1024


def test_non_positive_wealth_bounds_are_rejected():
    """The auction divides each price by the winner's own wealth.

    A non-positive bound makes that division meaningless, and the epsilon clamp
    guarding it would turn a valid numerator into an enormous price rather than
    failing. Reject it where the value enters, not where it explodes.
    """
    with pytest.raises(ValueError, match="min_wealth must be positive"):
        MoBConfig(min_wealth=0.0)

    with pytest.raises(ValueError, match="initial_wealth must be positive"):
        MoBConfig(initial_wealth=-1.0)


def test_inverted_wealth_band_is_rejected():
    """`clamp_(min=15, max=-5)` returns -5.

    Every clamp in the codebase exists to keep wealth inside the band; with the
    bounds inverted they would each write a negative wealth instead, which is the
    one way the auction's "no writer can produce a negative wealth" could be false.
    """
    with pytest.raises(ValueError, match="max_wealth"):
        MoBConfig(min_wealth=15.0, max_wealth=5.0)


@pytest.mark.parametrize("initial", [5.0, 800.0])
def test_initial_wealth_outside_the_band_is_rejected(initial):
    """Seeding outside the band makes step zero a discontinuity, not a starting point.

    Every expert would begin out of bounds and the first wealth update would yank
    them all onto a bound at once — which reads as a training artefact rather than
    the config error it is.
    """
    with pytest.raises(ValueError, match="must lie within"):
        MoBConfig(min_wealth=15.0, max_wealth=750.0, initial_wealth=initial)


@pytest.mark.parametrize("temperature", [0.0, -0.5])
def test_non_positive_routing_temperature_is_rejected(temperature):
    """Sharpness is a dial, not a sign.

    The proportional gate raises each bid to ``1 / routing_temperature``. Zero
    divides, and a negative exponent inverts the ranking, so the gate would hand the
    largest share of the output to the expert that bid least while the auction
    charged the one that bid most.
    """
    with pytest.raises(ValueError, match="routing_temperature must be positive"):
        MoBConfig(routing_temperature=temperature)


def test_auction_only_settings_warn_under_the_softmax_gate(caplog):
    """A tuned field that nothing reads is the defect class #12 was opened over."""
    with caplog.at_level("WARNING"):
        MoBConfig(router=ROUTER_SOFTMAX, payment_scale=2.5, wealth_decay=0.5)

    assert "payment_scale" in caplog.text
    assert "wealth_decay" in caplog.text
    assert "does not run the auction" in caplog.text
    assert "none of them affect this arm" in caplog.text


def test_auction_gate_parameters_warn_under_the_softmax_gate(caplog):
    """The share and sharpness go to VCGAuctioneer; SoftmaxRouter takes neither."""
    with caplog.at_level("WARNING"):
        MoBConfig(router=ROUTER_SOFTMAX, routing_temperature=0.1, use_differentiable_routing=False)

    assert "routing_temperature" in caplog.text
    assert "use_differentiable_routing" in caplog.text


def test_the_calibration_weight_warns_under_the_softmax_gate(caplog):
    """Its only reader sits below the has_economy early return, so it is a no-op here.

    Reachable from the harness: the trainer threads TrainingConfig.calibration_loss_weight
    straight into it and parity fingerprints that field, so a sweep over it would
    otherwise get a silent no-op on the control arm.
    """
    with caplog.at_level("WARNING"):
        MoBConfig(router=ROUTER_SOFTMAX, confidence_calibration_weight=0.9)

    assert "confidence_calibration_weight" in caplog.text


def test_jitter_is_not_reported_as_ignored_under_the_softmax_gate(caplog):
    """``jitter_std`` perturbs the adapters in from_pretrained_ffn, which every arm runs.

    Calling a live field dead is the mirror of the defect this warning exists to
    catch, and worse: it invites someone to stop setting something that does steer
    the control arm.
    """
    with caplog.at_level("WARNING"):
        MoBConfig(router=ROUTER_SOFTMAX, jitter_std=0.5)

    assert caplog.text == ""


def test_default_settings_are_silent_under_the_softmax_gate(caplog):
    """The control arm shares the auction's config object, so defaults must not warn."""
    with caplog.at_level("WARNING"):
        MoBConfig(router=ROUTER_SOFTMAX)

    assert caplog.text == ""


def test_auction_only_settings_do_not_warn_under_the_auction(caplog):
    with caplog.at_level("WARNING"):
        MoBConfig(router=ROUTER_AUCTION, payment_scale=2.5, routing_temperature=0.1)

    assert caplog.text == ""


def test_every_auction_only_field_exists_on_the_config():
    """A renamed field would otherwise turn the warning into a KeyError at import."""
    names = {spec.name for spec in dc_fields(MoBConfig)}
    assert set(AUCTION_ONLY_FIELDS) <= names
