import pytest
import torch

from mob import VCGAuctioneer

PAYMENT_TOLERANCE = 1e-5


def _make_auction(num_experts=4, top_k=2, differentiable=True, routing_share="uniform"):
    auctioneer = VCGAuctioneer(num_experts, top_k, differentiable, routing_share=routing_share)
    return auctioneer


def test_vcg_top_k_selection():
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    wealth = torch.tensor([1.0, 1.0, 1.0, 1.0])

    selected, _, _, _ = auctioneer(confidences, wealth)
    selected_set = set(selected[0, 0].tolist())
    assert 1 in selected_set
    assert 2 in selected_set


def test_vcg_routing_weights_sum_to_one():
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.randn(2, 8, 4).abs()
    wealth = torch.ones(4)

    _, routing_weights, _, _ = auctioneer(confidences, wealth)
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_vcg_output_shapes():
    batch, seq, num_experts, top_k = 2, 8, 4, 2
    auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
    auctioneer.eval()

    confidences = torch.randn(batch, seq, num_experts).abs()
    wealth = torch.ones(num_experts)

    selected, routing_weights, payments, _ = auctioneer(confidences, wealth)
    assert selected.shape == (batch, seq, top_k)
    assert routing_weights.shape == (batch, seq, top_k)
    assert payments.shape == (batch, seq, top_k)


def test_vcg_higher_bid_wins():
    auctioneer = _make_auction(num_experts=3, top_k=1)
    auctioneer.eval()

    confidences = torch.tensor([[[0.1, 0.2, 0.8]]])
    wealth = torch.tensor([1.0, 1.0, 1.0])

    selected, _, _, _ = auctioneer(confidences, wealth)
    assert selected[0, 0, 0].item() == 2

    wealth_adjusted = torch.tensor([10.0, 1.0, 1.0])
    selected_adj, _, _, _ = auctioneer(confidences, wealth_adjusted)
    assert selected_adj[0, 0, 0].item() == 0


def test_vcg_differentiable_mode():
    """The straight-through gate survives, but only as the softmax baseline.

    Under the uniform share this gradient path is deliberately absent; see
    ``test_uniform_share_carries_no_gradient_into_confidences``.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2, differentiable=True, routing_share="softmax")
    auctioneer.train()

    confidences = torch.randn(1, 4, 4).abs().requires_grad_(True)
    wealth = torch.ones(4)

    _, routing_weights, _, _ = auctioneer(confidences, wealth)
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

    selected, _, payments, _ = auctioneer(confidences, wealth)

    assert set(selected[0, 0].tolist()) == {1, 2}
    for j in range(2):
        assert payments[0, 0, j].item() == pytest.approx(0.3, abs=PAYMENT_TOLERANCE)


def test_vcg_payment_equals_replacement_bid_nonuniform_wealth():
    """Wealth weights the welfare, so the price is b_(k+1) over the winner's weight.

    Hand-computed: confidences 0.5/0.4/0.3/0.2 against wealth 1/2/3/0.5 give bids
    0.5, 0.8, 0.9, 0.1. Experts 2 and 1 win and expert 0's bid of 0.5 is the
    displaced welfare -- but that is denominated in bid units. Dividing by each
    winner's own weight restates it in the units that winner reports: expert 2 pays
    0.5/3 and expert 1 pays 0.5/2. Charging both the raw 0.5 is the weighted-VCG
    error that makes truthful reporting stop being optimal.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.tensor([[[0.5, 0.4, 0.3, 0.2]]])
    wealth = torch.tensor([1.0, 2.0, 3.0, 0.5])

    selected, _, payments, _ = auctioneer(confidences, wealth)

    assert selected[0, 0].tolist() == [2, 1]
    assert payments[0, 0, 0].item() == pytest.approx(0.5 / 3.0, abs=PAYMENT_TOLERANCE)
    assert payments[0, 0, 1].item() == pytest.approx(0.5 / 2.0, abs=PAYMENT_TOLERANCE)


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_vcg_payment_equals_replacement_bid_across_k(top_k):
    """The theorem, checked per token against an independently sorted bid vector."""
    num_experts = 5
    auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
    auctioneer.eval()

    torch.manual_seed(top_k)
    confidences = torch.rand(2, 6, num_experts)
    wealth = torch.rand(num_experts) * 10.0

    selected, _, payments, _ = auctioneer(confidences, wealth)
    bids = _bids(confidences, wealth)

    for b in range(bids.size(0)):
        for t in range(bids.size(1)):
            displaced = _kth_highest(bids[b, t], top_k)
            for j in range(top_k):
                expected = displaced / wealth[selected[b, t, j]].item()
                assert payments[b, t, j].item() == pytest.approx(expected, abs=PAYMENT_TOLERANCE)


def test_vcg_payment_invariant_to_winners_own_bid():
    """Incentive core of VCG: a winner cannot move its own price by bidding higher."""
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    wealth = torch.ones(4)
    confidences = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    selected, _, payments, _ = auctioneer(confidences, wealth)

    # Raise the top winner's confidence well clear of the field; the winner set
    # is unchanged, so its payment must not move.
    perturbed = confidences.clone()
    perturbed[0, 0, 1] = 0.99
    selected_perturbed, _, payments_perturbed, _ = auctioneer(perturbed, wealth)

    assert selected_perturbed[0, 0].tolist() == selected[0, 0].tolist()
    assert payments_perturbed[0, 0, 0].item() == pytest.approx(
        payments[0, 0, 0].item(), abs=PAYMENT_TOLERANCE
    )


def test_vcg_individual_rationality():
    """A truthful winner never pays more than the value it reported.

    Price and value must be compared in the same currency. The payment is already
    divided by the winner's weight, so the comparison is against the reported
    confidence -- not against the wealth-scaled bid.
    """
    auctioneer = _make_auction(num_experts=6, top_k=2)
    auctioneer.eval()

    torch.manual_seed(7)
    confidences = torch.rand(2, 8, 6)
    wealth = torch.rand(6) * 10.0

    selected, _, payments, _ = auctioneer(confidences, wealth)
    winner_values = torch.gather(confidences, -1, selected)

    surplus = winner_values - payments
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

    _, _, payments, _ = auctioneer(confidences, wealth)

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

    _, _, payments, _ = auctioneer(confidences, wealth)

    assert (payments > 0.1).all(), "payments collapsed to zero: k-1 exclusion set is back"
    for j in range(2):
        assert payments[0, 0, j].item() == pytest.approx(0.6, abs=PAYMENT_TOLERANCE)


def test_vcg_top_k_equals_num_experts_zero_payments():
    """No externality exists when every expert wins, so every price is zero."""
    auctioneer = _make_auction(num_experts=3, top_k=3)
    auctioneer.eval()

    confidences = torch.tensor([[[0.5, 0.3, 0.8]]])
    wealth = torch.ones(3)

    _, _, payments, _ = auctioneer(confidences, wealth)
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

    _, _, payments_fp32, _ = auctioneer(confidences, wealth)
    _, _, payments_bf16, _ = auctioneer(confidences.bfloat16(), wealth.bfloat16())

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

    selected, _, payments, _ = auctioneer(confidences, wealth)
    bids = _bids(confidences, wealth)

    # Guards the cast-back, not the accumulation dtype: `.to(out_dtype)` restores
    # float64 even when the interior arithmetic is fp32, so the tolerance below is
    # what actually pins the precision.
    assert payments.dtype == torch.float64, "float64 input was downcast"
    for t in range(bids.size(1)):
        displaced = _kth_highest(bids[0, t], 2)
        for j in range(2):
            expected = displaced / wealth[selected[0, t, j]].item()
            assert abs(payments[0, t, j].item() - expected) < 1e-12


def test_routing_share_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported routing share"):
        VCGAuctioneer(4, 2, routing_share="proportional")


def test_uniform_share_is_flat_and_ignores_own_bid():
    """The share a winner receives must not read that winner's own report.

    VCG prices the externality of *winning*, not the size of the slice won. A share
    that rises with your own bid while your price does not is influence bought for
    free, so the uniform split is a precondition of the incentive claim, not a
    simplification of it.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    wealth = torch.ones(4)
    confidences = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    selected, weights, _, _ = auctioneer(confidences, wealth)

    assert torch.allclose(weights, torch.full_like(weights, 0.5))

    perturbed = confidences.clone()
    perturbed[0, 0, 1] = 0.99
    selected_perturbed, weights_perturbed, _, _ = auctioneer(perturbed, wealth)

    assert selected_perturbed[0, 0].tolist() == selected[0, 0].tolist()
    assert torch.equal(weights_perturbed, weights)


def test_uniform_share_carries_no_gradient_into_confidences():
    """No language-modelling gradient may reach a confidence head through routing.

    This is what separates the mechanism from a learned gating network: with a
    constant share, the only thing that trains a head is its own value objective.
    A gradient here would mean the global loss is still doing the routing.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2, differentiable=True)
    auctioneer.train()

    confidences = torch.rand(1, 4, 4, requires_grad=True)
    _, routing_weights, _, _ = auctioneer(confidences, torch.ones(4))

    assert not routing_weights.requires_grad


def test_softmax_share_lets_a_winner_buy_influence_for_free():
    """Negative control: the gate-swap baseline is not incentive compatible.

    Same winners, same price, strictly larger share. This is the defect the uniform
    share exists to remove, kept as a live test so the baseline is documented by
    behaviour rather than by assertion.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2, routing_share="softmax")
    auctioneer.eval()

    wealth = torch.ones(4)
    truthful = torch.tensor([[[0.1, 0.9, 0.5, 0.3]]])
    overreported = truthful.clone()
    overreported[0, 0, 1] = 0.99

    selected, weights, payments, _ = auctioneer(truthful, wealth)
    selected_over, weights_over, payments_over, _ = auctioneer(overreported, wealth)

    assert selected_over[0, 0].tolist() == selected[0, 0].tolist()
    assert payments_over[0, 0, 0].item() == pytest.approx(
        payments[0, 0, 0].item(), abs=PAYMENT_TOLERANCE
    )
    assert weights_over[0, 0, 0].item() > weights[0, 0, 0].item() + PAYMENT_TOLERANCE


def test_payment_is_the_winners_critical_value():
    """A winner wins exactly when its report exceeds its own price.

    Monotone allocation plus critical-value payment is the Myerson characterisation
    of a strategyproof single-parameter mechanism; this checks the threshold the
    weighted division is supposed to produce actually sits where the theorem says.
    """
    auctioneer = _make_auction(num_experts=5, top_k=2)
    auctioneer.eval()

    torch.manual_seed(23)
    wealth = torch.rand(5) * 8.0 + 1.0
    field = torch.rand(5)

    def outcome(report: float) -> tuple[bool, float]:
        confidences = field.clone()
        confidences[0] = report
        selected, _, payments, _ = auctioneer(confidences.view(1, 1, 5), wealth)
        slots = (selected[0, 0] == 0).nonzero()
        if slots.numel() == 0:
            return False, 0.0
        return True, payments[0, 0, slots[0, 0]].item()

    won, price = outcome(0.999)
    assert won, "fixture never wins; the threshold below is untested"

    assert outcome(price + 1e-3)[0], "report above the price must win"
    assert not outcome(price - 1e-3)[0], "report below the price must lose"


# Wealth and rival reports chosen so expert 0's threshold lands mid-sweep. Rival
# bids are 2*0.8=1.6, 3*0.5=1.5, 1*0.9=0.9 and 5*0.28=1.4; with two slots, expert 0
# enters the allocation once 4*c_0 clears the second-highest rival bid of 1.5, so
# its critical value is 0.375 and the sweep below straddles it in both directions.
_UTILITY_WEALTH = torch.tensor([4.0, 2.0, 3.0, 1.0, 5.0])
_UTILITY_FIELD = torch.tensor([0.0, 0.8, 0.5, 0.9, 0.28])
_UTILITY_CRITICAL_VALUE = 0.375


def _expert_zero_utility(auctioneer, report: float, true_value: float) -> float:
    """Quasi-linear payoff for expert 0: what it banks, less what it is charged.

    Influence is the winner's share renormalised by an equal split, so it is
    identically 1.0 under the uniform rule and the expression reduces to the
    textbook ``v * 1[win] - p``. It is written this way so the same utility is
    defined for the softmax baseline, where a winner's slice does move with its own
    report and the deviation test below must be able to see that.
    """
    confidences = _UTILITY_FIELD.clone()
    confidences[0] = report
    selected, weights, payments, _ = auctioneer(confidences.view(1, 1, -1), _UTILITY_WEALTH)

    slots = (selected[0, 0] == 0).nonzero()
    if slots.numel() == 0:
        return 0.0

    slot = slots[0, 0]
    influence = weights[0, 0, slot].item() * auctioneer.top_k
    return true_value * influence - payments[0, 0, slot].item()


def test_truthful_reporting_maximises_expert_utility():
    """The incentive statement itself, checked by exhaustive deviation.

    Sweeping expert 0's *report* across the whole range while its true value is held
    fixed, no misreport ever beats reporting truthfully. This is the property the
    README is allowed to claim, and it needs both halves of the mechanism: an
    undivided weighted price or an own-bid-dependent share each hand some deviation
    a strictly better payoff.
    """
    auctioneer = _make_auction(num_experts=5, top_k=2)
    auctioneer.eval()

    winning_outcomes = set()
    for true_value in torch.linspace(0.05, 0.95, 10).tolist():
        truthful = _expert_zero_utility(auctioneer, true_value, true_value)
        winning_outcomes.add(true_value > _UTILITY_CRITICAL_VALUE)

        assert truthful >= -PAYMENT_TOLERANCE, "truthful reporting must never lose money"

        for report in torch.linspace(0.0, 1.0, 41).tolist():
            deviation = _expert_zero_utility(auctioneer, report, true_value)
            assert deviation <= truthful + PAYMENT_TOLERANCE, (
                f"misreporting {report:.3f} beat truthful {true_value:.3f}: "
                f"{deviation:.6f} > {truthful:.6f}"
            )

    assert winning_outcomes == {True, False}, "sweep must straddle the critical value"


def test_softmax_baseline_rewards_overreporting():
    """Negative control for the deviation sweep above.

    The same utility, the same fixture, the same truthful report -- but with the
    own-bid-weighted gate restored there is a strictly profitable lie. Asserting the
    baseline *fails* the property is what stops the test above from passing for
    reasons unrelated to the mechanism.
    """
    auctioneer = _make_auction(num_experts=5, top_k=2, routing_share="softmax")
    auctioneer.eval()

    true_value = 0.5
    truthful = _expert_zero_utility(auctioneer, true_value, true_value)
    best_lie = max(
        _expert_zero_utility(auctioneer, report, true_value)
        for report in torch.linspace(0.0, 1.0, 41).tolist()
    )

    assert best_lie > truthful + PAYMENT_TOLERANCE


def test_negative_wealth_trips_the_negativity_assert():
    """Why there is no wealth assert beside the epsilon clamp.

    A negative-wealth expert *can* win: with mixed signs a negative bid still places
    in the top k when the alternatives are worse. A negative-wealth winner sits in
    the top k, so b_(k+1) is at most its own negative bid, and the payment-negativity
    assert fires on the numerator before the division that the epsilon clamp guards
    -- unless that bid is inside the assert's tolerance of zero, which needs a wealth
    around -1e-7. MoBConfig rejects non-positive wealth bounds and every update path
    clamps to min_wealth, so that residual case cannot arise in the pipeline. A
    second assert here would be unreachable code documenting a protection that never
    runs.
    """
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()
    confidences = torch.tensor([[[0.9, 0.7, 0.5, 0.3]]])

    # Mixed signs: bids are [-0.45, +0.70, -1.00, -0.90], so expert 0 wins a slot
    # on negative wealth even though a positive bid exists.
    with pytest.raises(AssertionError, match="negative"):
        auctioneer(confidences, torch.tensor([-0.5, 1.0, -2.0, -3.0]))

    with pytest.raises(AssertionError, match="negative"):
        auctioneer(confidences, torch.tensor([-1.0, -2.0, -3.0, -4.0]))


def test_zero_wealth_is_accepted_and_prices_at_zero():
    """The case the epsilon clamp exists for: a bankrupt expert bids, and pays, zero."""
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    _, _, payments, _ = auctioneer(torch.zeros(1, 1, 4), torch.zeros(4))

    assert torch.isfinite(payments).all()
    assert (payments == 0).all()


def test_rebate_is_independent_of_the_recipients_own_bid():
    """The property the whole redistribution rests on.

    Cavallo pays expert i out of a quantity computed from everyone *but* i, so no
    report an expert can make moves the money it gets back. A rebate that did depend
    on it — an even split of the collected pot, say — would shift that expert's
    threshold away from its price, which is the Green–Laffont trade this rule exists
    to avoid.
    """
    auctioneer = _make_auction(num_experts=6, top_k=2)
    auctioneer.eval()

    torch.manual_seed(31)
    wealth = torch.rand(6) * 8.0 + 1.0
    confidences = torch.rand(1, 1, 6)

    baseline = auctioneer(confidences, wealth).rebates[0, 0, 0].item()

    for own_bid in torch.linspace(0.0, 1.0, 21).tolist():
        perturbed = confidences.clone()
        perturbed[0, 0, 0] = own_bid
        rebate = auctioneer(perturbed, wealth).rebates[0, 0, 0].item()
        assert rebate == pytest.approx(baseline, abs=PAYMENT_TOLERANCE), (
            f"reporting {own_bid:.2f} moved expert 0's own rebate"
        )


def test_rebate_never_exceeds_what_the_auction_collected():
    """Budget feasibility, in the currency the wealth ledger actually uses.

    Both sides are the per-expert quantities the wealth update consumes: payments
    already divided by each winner's own wealth, rebates divided by the pool's
    largest. Checking this in bid units instead — multiplying wealth back in — tests
    an inequality that holds even when the ledger's does not, which is exactly how a
    rebate that over-paid by 7.4x passed a feasibility test.
    """
    torch.manual_seed(37)
    for num_experts, top_k in ((6, 2), (8, 3), (5, 1), (4, 2)):
        auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
        auctioneer.eval()

        # Spanning the configured min_wealth..max_wealth band, not a narrow
        # random range: a per-recipient divisor only over-rebates once the
        # spread is wide, so a tight fixture cannot see it.
        wealth = torch.linspace(15.0, 750.0, num_experts)
        confidences = torch.rand(2, 6, num_experts)
        outcome = auctioneer(confidences, wealth)

        collected = outcome.payments.sum(dim=-1)
        returned = outcome.rebates.sum(dim=-1)

        assert (returned <= collected + PAYMENT_TOLERANCE).all(), (
            f"n={num_experts} k={top_k}: rebate exceeds revenue in credits"
        )
        assert (returned > 0).any(), "fixture returns nothing; feasibility is vacuous"

        # The classical Cavallo bound, which does not depend on the divisor: every
        # reference is at most b_(k+1), so sum_i (k/n) * ref_i <= k * b_(k+1). This
        # is tight (about 2% slack) on every fixture, whereas the credit assertion
        # above leaves 20-33% on the wide-spread ones — so a modest reference
        # inflation shows up here first.
        payout_in_bid_units = (outcome.rebates * wealth.max()).sum(dim=-1)
        displaced = torch.sort(_bids(confidences, wealth), dim=-1, descending=True)[0][..., top_k]
        # Relative, not absolute: these are bid-unit quantities of order 100, where
        # a 1e-5 absolute tolerance is really no tolerance at all.
        bound = top_k * displaced
        assert (payout_in_bid_units <= bound * (1 + PAYMENT_TOLERANCE)).all(), (
            f"n={num_experts} k={top_k}: exclusion rule returned too large a reference"
        )


def test_every_expert_is_rebated_not_only_winners():
    """Losers must be paid too — that is what keeps the rebate report-independent."""
    auctioneer = _make_auction(num_experts=5, top_k=2)
    auctioneer.eval()

    outcome = auctioneer(torch.rand(1, 3, 5), torch.rand(5) * 5.0 + 1.0)

    assert outcome.rebates.shape == (1, 3, 5)
    assert (outcome.rebates > 0).all()


def test_no_rebate_when_there_is_no_displaced_bid():
    """With fewer than k+2 experts there is no (k+2)-th bid to pay out of.

    The same boundary that makes payments zero when everyone wins.
    """
    for num_experts, top_k in ((3, 3), (3, 2)):
        auctioneer = _make_auction(num_experts=num_experts, top_k=top_k)
        auctioneer.eval()

        outcome = auctioneer(torch.rand(1, 2, num_experts), torch.ones(num_experts))
        assert (outcome.rebates == 0).all(), f"n={num_experts} k={top_k} rebated from nothing"


def test_rebate_leaves_the_deviation_sweep_intact():
    """The incentive result must survive the redistribution, not merely coexist with it."""
    auctioneer = _make_auction(num_experts=5, top_k=2)
    auctioneer.eval()

    for true_value in torch.linspace(0.05, 0.95, 10).tolist():
        truthful = _expert_zero_utility(auctioneer, true_value, true_value)
        for report in torch.linspace(0.0, 1.0, 41).tolist():
            assert _expert_zero_utility(auctioneer, report, true_value) <= (
                truthful + PAYMENT_TOLERANCE
            )


@pytest.mark.parametrize(
    ("regime", "wealth", "expected_return"),
    [
        ("flat", torch.full((8,), 100.0), 0.94),
        ("configured band", torch.linspace(15.0, 750.0, 8), 0.68),
        ("max_wealth monopoly", torch.tensor([750.0] + [15.0] * 7), 0.04),
    ],
)
def test_returned_fraction_matches_the_documented_regimes(regime, wealth, expected_return):
    """Pin the numbers both READMEs quote, and pin them from below.

    Feasibility only bounds the rebate from above, so a divisor returning 1% of
    Cavallo passes every other test here while making the documented 94/68/4%
    figures wrong. This is the test that fails when the divisor changes — which it
    is expected to, under #15 — and it should fail, because the prose changes with
    it.
    """
    torch.manual_seed(11)
    auctioneer = _make_auction(num_experts=8, top_k=2)
    auctioneer.eval()

    outcome = auctioneer(torch.rand(64, 16, 8), wealth)
    returned = outcome.rebates.sum(dim=-1).sum() / outcome.payments.sum(dim=-1).sum()

    assert returned.item() == pytest.approx(expected_return, abs=0.02), (
        f"{regime}: returned fraction moved; the README figures need updating too"
    )


def test_rebate_is_bounded_from_below_by_the_cavallo_reference():
    """A uniformly down-scaled reference is feasible, wrong, and otherwise invisible.

    Every other assertion here bounds the rebate from above. Without this one, a
    rule handing back a hundredth of what Cavallo specifies passes the whole file.
    """
    auctioneer = _make_auction(num_experts=6, top_k=2)
    auctioneer.eval()

    wealth = torch.linspace(15.0, 750.0, 6)
    confidences = torch.rand(4, 8, 6)
    outcome = auctioneer(confidences, wealth)

    bids = _bids(confidences, wealth)
    ranked = torch.sort(bids, dim=-1, descending=True)[0]
    # Every reference is at least b_(k+2), so the payout is at least (k/n)*n*b_(k+2).
    floor = (2 / 6) * 6 * ranked[..., 3] / wealth.max()

    assert (outcome.rebates.sum(dim=-1) >= floor - PAYMENT_TOLERANCE).all()
    assert (floor > 0).any(), "fixture has no lower bound to check"
