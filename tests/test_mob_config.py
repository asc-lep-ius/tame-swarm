import pytest

from mob import MoBConfig


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
