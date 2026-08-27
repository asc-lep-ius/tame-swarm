import pytest
import torch

from mob import VCGAuctioneer

PAYMENT_TOLERANCE = 1e-5


def _make_auction(num_experts=4, top_k=2, differentiable=True):
    auctioneer = VCGAuctioneer(num_experts, top_k, differentiable)
    return auctioneer


def test_vcg_top_k_selection():
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    wealth = torch.tensor([1.0, 1.0, 1.0, 1.0])

    selected, _, _ = auctioneer(confidences, wealth)
    selected_set = set(selected[0, 0].tolist())
    assert 1 in selected_set
    assert 2 in selected_set


def test_vcg_routing_weights_sum_to_one():
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.randn(2, 8, 4).abs()
    wealth = torch.ones(4)

    _, routing_weights, _ = auctioneer(confidences, wealth)
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_vcg_output_shapes():
    batch, seq, num_experts, top_k = 2, 8, 4, 2
    auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
    auctioneer.eval()

    confidences = torch.randn(batch, seq, num_experts).abs()
    wealth = torch.ones(num_experts)

    selected, routing_weights, payments = auctioneer(confidences, wealth)
    assert selected.shape == (batch, seq, top_k)
    assert routing_weights.shape == (batch, seq, top_k)
    assert payments.shape == (batch, seq, top_k)


def test_vcg_higher_bid_wins():
    auctioneer = _make_auction(num_experts=3, top_k=1)
    auctioneer.eval()

    confidences = torch.tensor([[[0.1, 0.2, 0.8]]])
    wealth = torch.tensor([1.0, 1.0, 1.0])

    selected, _, _ = auctioneer(confidences, wealth)
    assert selected[0, 0, 0].item() == 2

    wealth_adjusted = torch.tensor([10.0, 1.0, 1.0])
    selected_adj, _, _ = auctioneer(confidences, wealth_adjusted)
    assert selected_adj[0, 0, 0].item() == 0


def test_vcg_differentiable_mode():
    auctioneer = _make_auction(num_experts=4, top_k=2, differentiable=True)
    auctioneer.train()

    confidences = torch.randn(1, 4, 4).abs().requires_grad_(True)
    wealth = torch.ones(4)

    _, routing_weights, _ = auctioneer(confidences, wealth)
    objective_weights = torch.arange(
        1,
        routing_weights.numel() + 1,
        device=routing_weights.device,
        dtype=routing_weights.dtype,
    ).reshape_as(routing_weights)
    loss = (routing_weights * objective_weights).sum()
    loss.backward()
    assert confidences.grad is not None
    assert (confidences.grad.abs() > 0).any()


def _bids(confidences: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
    """The auction's bid rule, restated here so tests never import it."""
    return confidences * wealth.unsqueeze(0).unsqueeze(0)


def _kth_highest(bid_row: torch.Tensor, index: int) -> float:
    ordered, _ = torch.sort(bid_row, descending=True)
    return ordered[index].item()


def test_vcg_payment_equals_replacement_bid_hand_computed():
    """Every winner pays b_(k+1) -- the bid of the expert its win displaced.

    Hand-computed: unit wealth makes bids equal confidences, so the sorted bids
    are 0.9, 0.5, 0.3, 0.1. The top two win and the third bid, 0.3, is the price.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    wealth = torch.ones(4)

    selected, _, payments = auctioneer(confidences, wealth)

    assert set(selected[0, 0].tolist()) == {1, 2}
    for j in range(2):
        assert payments[0, 0, j].item() == pytest.approx(0.3, abs=PAYMENT_TOLERANCE)


def test_vcg_payment_equals_replacement_bid_nonuniform_wealth():
    """Wealth enters only through the bid, so the price is still b_(k+1).

    Hand-computed: confidences 0.5/0.4/0.3/0.2 against wealth 1/2/3/0.5 give bids
    0.5, 0.8, 0.9, 0.1. Experts 2 and 1 win; expert 0's bid of 0.5 is the price.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.tensor([[[0.5, 0.4, 0.3, 0.2]]])
    wealth = torch.tensor([1.0, 2.0, 3.0, 0.5])

    selected, _, payments = auctioneer(confidences, wealth)

    assert set(selected[0, 0].tolist()) == {1, 2}
    for j in range(2):
        assert payments[0, 0, j].item() == pytest.approx(0.5, abs=PAYMENT_TOLERANCE)


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_vcg_payment_equals_replacement_bid_across_k(top_k):
    """The theorem, checked per token against an independently sorted bid vector."""
    num_experts = 5
    auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
    auctioneer.eval()

    torch.manual_seed(top_k)
    confidences = torch.rand(2, 6, num_experts)
    wealth = torch.rand(num_experts) * 10.0

    _, _, payments = auctioneer(confidences, wealth)
    bids = _bids(confidences, wealth)

    for b in range(bids.size(0)):
        for t in range(bids.size(1)):
            expected = _kth_highest(bids[b, t], top_k)
            for j in range(top_k):
                assert payments[b, t, j].item() == pytest.approx(expected, abs=PAYMENT_TOLERANCE)


def test_vcg_payment_invariant_to_winners_own_bid():
    """Incentive core of VCG: a winner cannot move its own price by bidding higher."""
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    wealth = torch.ones(4)
    confidences = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    selected, _, payments = auctioneer(confidences, wealth)

    # Raise the top winner's confidence well clear of the field; the winner set
    # is unchanged, so its payment must not move.
    perturbed = confidences.clone()
    perturbed[0, 0, 1] = 0.99
    selected_perturbed, _, payments_perturbed = auctioneer(perturbed, wealth)

    assert selected_perturbed[0, 0].tolist() == selected[0, 0].tolist()
    assert payments_perturbed[0, 0, 0].item() == pytest.approx(
        payments[0, 0, 0].item(), abs=PAYMENT_TOLERANCE
    )


def test_vcg_individual_rationality():
    """A truthful winner never pays more than its own bid, so surplus stays >= 0."""
    auctioneer = _make_auction(num_experts=6, top_k=2)
    auctioneer.eval()

    torch.manual_seed(7)
    confidences = torch.rand(2, 8, 6)
    wealth = torch.rand(6) * 10.0

    selected, _, payments = auctioneer(confidences, wealth)
    bids = _bids(confidences, wealth)
    winner_bids = torch.gather(bids, -1, selected)

    surplus = winner_bids - payments
    assert (surplus >= -PAYMENT_TOLERANCE).all()


def test_vcg_payments_non_negative_and_non_vacuous():
    """No positive transfers -- and at least one price is strictly positive.

    The strict-positivity half is what makes this assertion mean anything: the
    all-zero payments this replaces satisfied ``>= 0`` perfectly well.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    torch.manual_seed(3)
    confidences = torch.rand(2, 8, 4)
    wealth = torch.rand(4) * 10.0

    _, _, payments = auctioneer(confidences, wealth)

    assert (payments >= 0).all()
    assert (payments > PAYMENT_TOLERANCE).any()


def test_vcg_zero_payment_regression():
    """Guards the k-1 exclusion-set bug that made every payment identically zero.

    Excluding a winner frees all k slots, so the counterfactual allocation is the
    top k of the remaining bids. Taking k-1 instead makes the exclusion set exactly
    the other winners, and the welfare difference collapses to zero for every
    winner in every configuration.
    """
    auctioneer = _make_auction(num_experts=8, top_k=2)
    auctioneer.eval()

    # Bids are well separated, so the price cannot be zero for float reasons.
    confidences = torch.tensor([[[0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]]])
    wealth = torch.ones(8)

    _, _, payments = auctioneer(confidences, wealth)

    assert (payments > 0.1).all(), "payments collapsed to zero: k-1 exclusion set is back"
    for j in range(2):
        assert payments[0, 0, j].item() == pytest.approx(0.6, abs=PAYMENT_TOLERANCE)


def test_vcg_top_k_equals_num_experts_zero_payments():
    """No externality exists when every expert wins, so every price is zero."""
    auctioneer = _make_auction(num_experts=3, top_k=3)
    auctioneer.eval()

    confidences = torch.tensor([[[0.5, 0.3, 0.8]]])
    wealth = torch.ones(3)

    _, _, payments = auctioneer(confidences, wealth)
    assert (payments == 0).all()


@pytest.mark.skipif(not __debug__, reason="bare asserts are compiled out under -O")
def test_vcg_negative_price_raises_instead_of_being_clamped():
    """A broken invariant must fail loudly rather than be silently repaired.

    Negative wealth violates the mechanism's precondition and drives b_(k+1)
    negative. The old ``clamp(min=0)`` rewrote that to a plausible 0.0 -- the same
    guard that hid the zero-payment defect. It must raise now.
    """
    auctioneer = _make_auction(num_experts=3, top_k=1)
    auctioneer.eval()

    confidences = torch.tensor([[[1.0, 2.0, 3.0]]])
    wealth = torch.tensor([1.0, -1.0, -1.0])  # bids [1, -2, -3]; b_(k+1) = -2

    with pytest.raises(AssertionError, match="VCG payment is negative"):
        auctioneer(confidences, wealth)


def test_vcg_payment_survives_bfloat16_router_specialisation():
    """bfloat16 is the default training dtype and expert_wealth is cast to it.

    The price is a difference of two welfare sums far larger than the price itself.
    Accumulated in bf16 that cancellation quantises a real price of ~0.08 to exactly
    0.0 -- silently reinstating the zero payments this module was fixed to stop --
    and drives the non-negativity assert on a specialised router.
    """
    auctioneer = _make_auction(num_experts=8, top_k=2)
    auctioneer.eval()

    # Specialised router: two clear winners, six experts pushed near zero.
    logits = torch.full((1, 4, 8), -8.0)
    logits[..., 3] = 4.0
    logits[..., 6] = 3.5
    confidences = torch.sigmoid(logits)
    wealth = torch.linspace(75.0, 300.0, 8)

    _, _, payments_fp32 = auctioneer(confidences, wealth)
    _, _, payments_bf16 = auctioneer(confidences.bfloat16(), wealth.bfloat16())

    assert (payments_fp32 > 0).all(), "fixture prices nothing; the comparison is vacuous"
    assert payments_bf16.dtype == torch.bfloat16, "payments must return in the input dtype"
    assert (payments_bf16 > 0).all(), "bf16 price collapsed to zero"
    assert torch.allclose(payments_bf16.float(), payments_fp32, rtol=0.05)


@pytest.mark.skipif(not __debug__, reason="bare asserts are compiled out under -O")
def test_vcg_nan_bids_report_as_non_finite_not_as_negative():
    """A NaN bid must name its own fault rather than masquerade as a negative price."""
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.tensor([[[0.1, 0.9, float("nan"), 0.3]]])
    wealth = torch.ones(4)

    with pytest.raises(AssertionError, match="not finite"):
        auctioneer(confidences, wealth)


def test_vcg_payment_preserves_float64_precision():
    """A float64 caller must not be silently downcast by the float32 accumulation.

    The fp32 guard against bfloat16 cancellation has to lift precision, never cap
    it. A hard ``.float()`` holds the error here around 1e-06; genuine float64
    accumulation reaches ~2e-15, and the tolerance sits between the two regimes
    with roughly 375x headroom on the passing side.
    """
    auctioneer = _make_auction(num_experts=6, top_k=2)
    auctioneer.eval()

    torch.manual_seed(19)
    confidences = torch.rand(1, 4, 6, dtype=torch.float64)
    wealth = torch.rand(6, dtype=torch.float64) * 10.0

    _, _, payments = auctioneer(confidences, wealth)
    bids = _bids(confidences, wealth)

    # Guards the cast-back, not the accumulation dtype: `.to(out_dtype)` restores
    # float64 even when the interior arithmetic is fp32, so the tolerance below is
    # what actually pins the precision.
    assert payments.dtype == torch.float64, "float64 input was downcast"
    for t in range(bids.size(1)):
        expected = _kth_highest(bids[0, t], 2)
        for j in range(2):
            assert abs(payments[0, t, j].item() - expected) < 1e-12
