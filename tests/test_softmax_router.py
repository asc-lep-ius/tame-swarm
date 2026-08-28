"""The #12 control arm: same experts, same heads, no economy."""

import pytest
import torch

from mob import (
    ROUTER_AUCTION,
    ROUTER_SOFTMAX,
    MixtureOfBidders,
    MoBConfig,
    SoftmaxRouter,
    update_all_mob_from_loss,
)


def _config(**overrides):
    base = {
        "num_experts": 4,
        "top_k": 2,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "adapter_rank": 4,
        "adapter_alpha": 4.0,
    }
    base.update(overrides)
    return MoBConfig(**base)


def test_rejects_unknown_router():
    with pytest.raises(ValueError, match="Unsupported router"):
        _config(router="vickrey")


def test_has_economy_tracks_the_router():
    assert _config(router=ROUTER_AUCTION).has_economy
    assert not _config(router=ROUTER_SOFTMAX).has_economy


def test_routing_weights_are_normalised_over_winners():
    router = SoftmaxRouter(num_experts=4, top_k=2)
    logits = torch.randn(2, 8, 4)

    outcome = router(logits)

    assert outcome.routing_weights.shape == (2, 8, 2)
    assert torch.allclose(outcome.routing_weights.sum(dim=-1), torch.ones(2, 8), atol=1e-6)
    assert outcome.payments is None
    assert outcome.rebates is None


def test_winners_are_the_highest_logits():
    """The control has to be an ordinary argmax gate, or it is not a control."""
    router = SoftmaxRouter(num_experts=5, top_k=2)
    logits = torch.tensor([[[0.1, 5.0, -2.0, 3.0, 0.0]]])

    outcome = router(logits)

    assert outcome.selected_experts[0, 0, 0].item() == 1
    assert outcome.selected_experts[0, 0, 1].item() == 3


def test_share_depends_only_on_the_winners():
    """The losers' mass cancels in the renormalisation, exactly.

    Softmax-then-renormalise and softmax-over-the-top-k are the same function, so a
    winner's share is fixed by the gaps between the winners' logits alone. Pinned
    because it is easy to assume otherwise and then read a share as evidence of how
    firmly the gate rejected the experts it dropped.
    """
    router = SoftmaxRouter(num_experts=3, top_k=2)

    close_loser = router(torch.tensor([[[2.0, 1.0, 0.9]]])).routing_weights
    distant_loser = router(torch.tensor([[[2.0, 1.0, -8.0]]])).routing_weights
    top_k_only = torch.softmax(torch.tensor([[[2.0, 1.0]]]), dim=-1)

    assert torch.allclose(close_loser, distant_loser, atol=1e-6)
    assert torch.allclose(close_loser, top_k_only, atol=1e-6)


def test_softmax_arm_never_moves_wealth():
    layer = MixtureOfBidders(_config(router=ROUTER_SOFTMAX))
    layer.train()
    before = layer.expert_wealth.clone()

    layer(torch.randn(2, 8, 32))
    layer.update_wealth_from_loss(torch.rand(2, 8))

    assert torch.equal(layer.expert_wealth, before)
    assert layer._cached_payments is None
    assert not layer._loss_feedback_pending


def test_softmax_arm_adds_no_value_objective():
    """The economy's objective is what is being removed, so it must be absent.

    The router z-loss is deliberately still live: it regularises the same logits in
    both arms, so it is not part of the difference under test.
    """
    layer = MixtureOfBidders(_config(router=ROUTER_SOFTMAX))
    layer.train()

    layer(torch.randn(2, 8, 32))
    layer.update_wealth_from_loss(torch.rand(2, 8))

    assert layer.get_confidence_calibration_loss().item() == 0.0
    assert layer.get_router_z_loss().item() > 0.0


def test_language_model_loss_reaches_the_heads_in_the_control_arm():
    """A learned router is learned from the LM loss; the auction's heads are not.

    Under the uniform share the auction returns a constant share, so the only
    gradient into a confidence head is its own value objective. The control has no
    value objective, so if the LM loss did not reach the heads its gate would never
    learn anything and the baseline would be a straw man.
    """
    layer = MixtureOfBidders(_config(router=ROUTER_SOFTMAX))
    layer.train()

    output = layer(torch.randn(2, 8, 32))
    output.sum().backward()

    head_gradients = [
        head.proj.weight.grad
        for head in layer.confidence_heads
        if head.proj.weight.grad is not None
    ]
    assert head_gradients
    assert any(gradient.abs().sum() > 0 for gradient in head_gradients)


def test_auction_arm_keeps_its_heads_out_of_the_language_model_loss():
    """The other half of the asymmetry above, asserted rather than assumed."""
    layer = MixtureOfBidders(_config(router=ROUTER_AUCTION))
    layer.train()

    output = layer(torch.randn(2, 8, 32))
    output.sum().backward()

    for head in layer.confidence_heads:
        assert head.proj.weight.grad is None or head.proj.weight.grad.abs().sum() == 0


def test_both_arms_build_the_same_experts():
    """Parity where it matters most: the arms must differ only in the gate.

    Seeded before each construction and compared weight by weight, not by shape:
    matching shapes would still hold if one arm initialised its experts differently,
    and "the same experts" is the claim the control arm rests on.
    """
    torch.manual_seed(7)
    auction = MixtureOfBidders(_config(router=ROUTER_AUCTION))
    torch.manual_seed(7)
    control = MixtureOfBidders(_config(router=ROUTER_SOFTMAX))

    assert type(auction.experts[0]) is type(control.experts[0])
    assert len(auction.experts) == len(control.experts)
    assert len(auction.confidence_heads) == len(control.confidence_heads)

    for expected, actual in zip(
        auction.experts.parameters(), control.experts.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    for expected, actual in zip(
        auction.confidence_heads.parameters(), control.confidence_heads.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_model_level_update_is_a_no_op_for_the_control_arm(tiny_causal_lm, tiny_mob_config):
    """The trainer calls the same update on every arm; it must be inert here."""
    from dataclasses import replace

    from mob import apply_mob_to_model, get_mob_layers

    model = apply_mob_to_model(
        tiny_causal_lm,
        replace(tiny_mob_config, router=ROUTER_SOFTMAX),
        layers_to_modify=[1, 2],
    )
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))
    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)

    before = [mob.expert_wealth.clone() for mob in get_mob_layers(model)]
    update_all_mob_from_loss(model, torch.rand(2, 7), torch.ones(2, 7))

    for mob, wealth in zip(get_mob_layers(model), before, strict=True):
        assert torch.equal(mob.expert_wealth, wealth)
