"""The VCG properties, stated from the mechanism definition and explored with hypothesis (#6).

#9's payment defect -- every price identically zero -- reached ``main`` because
the tests were written from the implementation: ``remaining_bids[:1]`` mirrored
the off-by-one and asserted ``0 == 0`` under a name that promised the opposite.
Nothing below reads ``auction.py`` for its expected value. Each property is the
textbook statement for a top-*k* unit-slot auction over weighted bids
``b_i = c_i w_i``, with the reference quantity recomputed here in float64 from
the definition (the (k+1)-th highest bid, the welfare-maximising subset by
enumeration), and hypothesis draws the markets.

Every property is paired with the mutation that breaks it, so each test is known
to be able to fail. #6 asked for every property to fail against the pre-#9 code,
and three of them cannot: individual rationality, own-bid independence and
allocation optimality are all *satisfied* by zero payments. Those three are
paired instead with the defect they would catch -- the undivided weighted price
(#10), a first-price rule, and a wealth-blind allocation.

Truthfulness is asserted here because #10 settled the utility it ranges over:
the quasi-linear payoff ``v x 1[win] - p`` that #15's value definition
denominates (see ``test_auction.test_truthful_reporting_maximises_expert_utility``
for the exhaustive sweep on a fixed fixture).

The properties are checked on reports a live coupling produced and on the same
layer's reports with the coupling detached: the coupling changes what the heads
report, never what the auction does with a report.
"""

import math
from dataclasses import dataclass
from itertools import combinations

import pytest
import torch
import torch.nn.functional as F
from hypothesis import find, given
from hypothesis import strategies as st

from mob import MixtureOfBidders, MoBConfig, SteeringCouplingConfig
from mob.auction import ROUTING_SHARE_PROPORTIONAL, AuctionOutcome, VCGAuctioneer

from .auction_mutations import (
    first_price_payments,
    pre_nine_payments,
    undivided_payments,
    wealth_blind_forward,
)

WEALTH_BAND = (MoBConfig().min_wealth, MoBConfig().max_wealth)
# Confidence logits at initialisation sit near -4 (a report of ~0.02) and a
# trained report is a loss-reduction estimate of order one; the range brackets
# both without spanning more orders of magnitude than float32 accumulation of
# the welfare sums can resolve against the smallest bid.
LOGIT_RANGE = (-4.5, 4.0)
# Float error headroom on a difference of welfare sums accumulated in float32,
# relative to the largest bid and the number of terms summed. Small enough that a
# zero payment where the (k+1)-th bid is the strategy's floor is always caught.
FLOAT32_EPSILON = torch.finfo(torch.float32).eps
SUM_ROUNDING_FACTOR = 8.0
HIDDEN = 16


@dataclass(frozen=True)
class Market:
    """One drawn auction: ``confidences`` is ``(1, tokens, experts)``, wealth per expert."""

    top_k: int
    wealth: torch.Tensor
    confidences: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.wealth.numel())

    @property
    def bids(self) -> torch.Tensor:
        return (self.confidences.double() * self.wealth.double()).squeeze(0)


@st.composite
def markets(draw, tokens: tuple[int, int] = (1, 3)) -> Market:
    num_experts = draw(st.integers(3, 8))
    top_k = draw(st.integers(1, num_experts - 1))
    num_tokens = draw(st.integers(*tokens))
    log_wealth = draw(
        st.lists(
            st.floats(math.log(WEALTH_BAND[0]), math.log(WEALTH_BAND[1])),
            min_size=num_experts,
            max_size=num_experts,
        )
    )
    logits = draw(
        st.lists(
            st.floats(*LOGIT_RANGE),
            min_size=num_experts * num_tokens,
            max_size=num_experts * num_tokens,
        )
    )
    wealth = torch.tensor(log_wealth, dtype=torch.float32).exp()
    confidences = F.softplus(torch.tensor(logits, dtype=torch.float32)).view(
        1, num_tokens, num_experts
    )
    return Market(top_k, wealth, confidences)


def _auction(market: Market, **overrides) -> VCGAuctioneer:
    settings = dict(differentiable=False, exploration_rate=0.0)
    settings.update(overrides)
    auction = VCGAuctioneer(market.num_experts, market.top_k, **settings)
    auction.eval()
    return auction


def _tolerance(market: Market) -> float:
    """Absolute headroom on a price numerator: float32 rounding of the welfare sums."""
    max_bid = float(market.bids.abs().max())
    return SUM_ROUNDING_FACTOR * FLOAT32_EPSILON * market.top_k * max(max_bid, 1.0)


def _kth_plus_one_bid(market: Market, token: int) -> float:
    ranked = market.bids[token].sort(descending=True).values
    return float(ranked[market.top_k])


def _optimal_welfare(market: Market, token: int) -> float:
    bids = market.bids[token]
    return max(
        float(bids[list(subset)].sum())
        for subset in combinations(range(market.num_experts), market.top_k)
    )


# --- The properties, each as a checker the tests and the mutation pairings share ---


def check_externality_pricing(market: Market, outcome: AuctionOutcome) -> None:
    """Winner *j* pays the (k+1)-th highest bid, in its own units: ``p_j = b_(k+1) / w_j``."""
    assert outcome.payments is not None
    tolerance = _tolerance(market)
    for token in range(market.confidences.shape[1]):
        displaced = _kth_plus_one_bid(market, token)
        for slot in range(market.top_k):
            winner = int(outcome.selected_experts[0, token, slot])
            numerator = float(outcome.payments[0, token, slot]) * float(market.wealth[winner])
            assert abs(numerator - displaced) <= tolerance, (token, slot, numerator, displaced)


def check_no_positive_transfers(market: Market, outcome: AuctionOutcome) -> None:
    """Every payment is non-negative -- and, with ``n > k`` and every bid positive, strictly so."""
    assert outcome.payments is not None
    assert float(outcome.payments.min()) > 0.0, outcome.payments


def check_individual_rationality(market: Market, outcome: AuctionOutcome) -> None:
    """Under truthful reporting the report is the value, and no winner pays more than it."""
    assert outcome.payments is not None
    for token in range(market.confidences.shape[1]):
        for slot in range(market.top_k):
            winner = int(outcome.selected_experts[0, token, slot])
            value = float(market.confidences[0, token, winner])
            price = float(outcome.payments[0, token, slot])
            headroom = _tolerance(market) / float(market.wealth[winner])
            assert value - price >= -headroom, (token, slot, value, price)


def check_allocation_optimality(market: Market, outcome: AuctionOutcome) -> None:
    """The winners maximise summed bids over every size-*k* subset."""
    for token in range(market.confidences.shape[1]):
        winners = outcome.selected_experts[0, token].tolist()
        assert len(set(winners)) == market.top_k
        welfare = float(market.bids[token][winners].sum())
        assert welfare >= _optimal_welfare(market, token) - _tolerance(market), (token, winners)


def check_own_bid_independence(market: Market, auction: VCGAuctioneer) -> None:
    """Raising a winner's report, while it keeps winning, leaves its own price unchanged."""
    outcome = auction(market.confidences, market.wealth)
    assert outcome.payments is not None
    tolerance = _tolerance(market)
    for token in range(market.confidences.shape[1]):
        for slot in range(market.top_k):
            winner = int(outcome.selected_experts[0, token, slot])
            for factor in (1.5, 4.0):
                raised = market.confidences.clone()
                raised[0, token, winner] *= factor
                perturbed = auction(raised, market.wealth)
                assert perturbed.payments is not None
                slots = (perturbed.selected_experts[0, token] == winner).nonzero()
                assert slots.numel() == 1, "a raised report must keep the slot"
                before = float(outcome.payments[0, token, slot]) * float(market.wealth[winner])
                after = float(perturbed.payments[0, token, slots[0, 0]]) * float(
                    market.wealth[winner]
                )
                assert abs(after - before) <= tolerance, (token, winner, factor, before, after)


def check_all(market: Market, auction: VCGAuctioneer) -> None:
    outcome = auction(market.confidences, market.wealth)
    check_externality_pricing(market, outcome)
    check_no_positive_transfers(market, outcome)
    check_individual_rationality(market, outcome)
    check_allocation_optimality(market, outcome)
    check_own_bid_independence(market, auction)


# --- The properties over drawn markets -----------------------------------------


@given(markets())
def test_every_winner_pays_the_displaced_bid_in_its_own_units(market):
    check_externality_pricing(market, _auction(market)(market.confidences, market.wealth))


@given(markets())
def test_payments_are_strictly_positive_whenever_a_bid_is_displaced(market):
    check_no_positive_transfers(market, _auction(market)(market.confidences, market.wealth))


@given(markets())
def test_truthful_winners_never_pay_more_than_their_value(market):
    check_individual_rationality(market, _auction(market)(market.confidences, market.wealth))


@given(markets())
def test_the_winners_are_the_welfare_maximising_subset(market):
    check_allocation_optimality(market, _auction(market)(market.confidences, market.wealth))


@given(markets())
def test_a_winners_price_does_not_move_with_its_own_report(market):
    check_own_bid_independence(market, _auction(market))


# --- Truthfulness, now that the utility exists -----------------------------------


def _expert_zero_utility(auction: VCGAuctioneer, market: Market, report: float, value: float):
    """Quasi-linear payoff for expert 0 on a one-token market: ``v x influence - p``.

    ``influence`` is the winner's share renormalised by an equal split -- exactly
    one under the uniform share, so the payoff is the textbook ``v 1[win] - p``;
    under the proportional baseline it moves with the report, which is what the
    negative control below needs to see.
    """
    confidences = market.confidences.clone()
    confidences[0, 0, 0] = report
    outcome = auction(confidences, market.wealth)
    assert outcome.payments is not None
    slots = (outcome.selected_experts[0, 0] == 0).nonzero()
    if slots.numel() == 0:
        return 0.0
    slot = slots[0, 0]
    influence = float(outcome.routing_weights[0, 0, slot]) * auction.top_k
    return value * influence - float(outcome.payments[0, 0, slot])


REPORT_GRID = torch.linspace(0.0, 5.0, 26).tolist()


def _best_deviation_gain(auction: VCGAuctioneer, market: Market, value: float) -> float:
    truthful = _expert_zero_utility(auction, market, value, value)
    return max(_expert_zero_utility(auction, market, report, value) for report in REPORT_GRID) - (
        truthful
    )


@given(markets(tokens=(1, 1)), st.floats(0.05, 4.0))
def test_no_misreport_beats_a_truthful_report(market, value):
    auction = _auction(market)
    headroom = _tolerance(market) / float(market.wealth[0])
    assert _expert_zero_utility(auction, market, value, value) >= -headroom
    assert _best_deviation_gain(auction, market, value) <= headroom


def test_the_proportional_baseline_admits_a_profitable_misreport():
    """The negative control: the property above is false for the own-bid-weighted gate.

    An existence claim, so it is a search rather than a universal check: hypothesis
    finds a drawn market on which some report on the grid strictly beats truth.
    """

    def has_profitable_lie(market: Market) -> bool:
        auction = _auction(market, routing_share=ROUTING_SHARE_PROPORTIONAL)
        return _best_deviation_gain(auction, market, 0.5) > 1e-3

    find(markets(tokens=(1, 1)), has_profitable_lie)


# --- With the coupling attached, and with it detached ----------------------------


def _coupled_layer(market: Market, seed: int, live: bool) -> tuple[MixtureOfBidders, torch.Tensor]:
    """A MoB layer whose heads report through a live coupling, or through none."""
    torch.manual_seed(seed)
    config = MoBConfig(
        num_experts=market.num_experts,
        top_k=market.top_k,
        hidden_dim=HIDDEN,
        intermediate_dim=2 * HIDDEN,
        adapter_rank=2,
        adapter_alpha=2.0,
        exploration_rate=0.0,
    )
    mob = MixtureOfBidders(config)
    mob.eval()
    with torch.no_grad():
        mob.expert_wealth.copy_(market.wealth)
    if live:
        generator = torch.Generator().manual_seed(seed + 1)
        direction = torch.randn(HIDDEN, generator=generator)
        receptor = torch.randn(HIDDEN, generator=generator)
        mob.attach_coupling(
            direction,
            SteeringCouplingConfig(
                hidden_dim=HIDDEN, coupling_beta=1.0, warmup_steps=10, max_coupling_fraction=0.5
            ),
        )
        with torch.no_grad():
            mob.coupling.detector.copy_(receptor / receptor.norm())
        mob.set_coupling_step(10)
    generator = torch.Generator().manual_seed(seed + 2)
    hidden = torch.randn(1, market.confidences.shape[1], HIDDEN, generator=generator)
    return mob, hidden


def _reports_through(market: Market, seed: int, live: bool) -> tuple[Market, VCGAuctioneer]:
    mob, hidden = _coupled_layer(market, seed, live)
    mob(hidden, update_wealth=False)
    assert mob.last_stats is not None
    reported = Market(market.top_k, market.wealth, mob.last_stats.confidences.clone())
    return reported, mob.gate  # type: ignore[return-value]


@given(markets(), st.integers(0, 2**16))
def test_the_properties_hold_on_reports_a_live_coupling_produced(market, seed):
    coupled, gate = _reports_through(market, seed, live=True)
    detached, _ = _reports_through(market, seed, live=False)
    assert not torch.equal(coupled.confidences, detached.confidences), "the coupling must be live"
    assert coupled.confidences.min() > 0.0

    check_all(coupled, gate)


@given(markets(), st.integers(0, 2**16))
def test_the_properties_hold_with_the_coupling_detached(market, seed):
    detached, gate = _reports_through(market, seed, live=False)
    assert not hasattr(gate, "coupling")

    check_all(detached, gate)


# --- Each property paired with the defect it catches ---------------------------------

EXAMPLE = Market(
    top_k=2,
    wealth=torch.tensor([40.0, 15.0, 300.0, 75.0, 120.0]),
    confidences=torch.tensor([[[0.9, 1.2, 0.05, 0.4, 0.3], [0.2, 0.7, 0.03, 0.9, 0.5]]]),
)


def test_externality_pricing_and_positivity_fail_on_the_pre_nine_exclusion_set(monkeypatch):
    monkeypatch.setattr(VCGAuctioneer, "_compute_vcg_payments", pre_nine_payments)
    outcome = _auction(EXAMPLE)(EXAMPLE.confidences, EXAMPLE.wealth)

    with pytest.raises(AssertionError):
        check_externality_pricing(EXAMPLE, outcome)
    with pytest.raises(AssertionError):
        check_no_positive_transfers(EXAMPLE, outcome)


def test_individual_rationality_fails_on_the_undivided_weighted_price(monkeypatch):
    original = VCGAuctioneer._compute_vcg_payments
    monkeypatch.setattr(VCGAuctioneer, "_compute_vcg_payments", undivided_payments(original))

    outcome = _auction(EXAMPLE)(EXAMPLE.confidences, EXAMPLE.wealth)
    with pytest.raises(AssertionError):
        check_individual_rationality(EXAMPLE, outcome)


def test_own_bid_independence_fails_under_a_first_price_rule(monkeypatch):
    monkeypatch.setattr(VCGAuctioneer, "_compute_vcg_payments", first_price_payments)

    with pytest.raises(AssertionError):
        check_own_bid_independence(EXAMPLE, _auction(EXAMPLE))


def test_allocation_optimality_fails_on_a_wealth_blind_allocation(monkeypatch):
    monkeypatch.setattr(VCGAuctioneer, "forward", wealth_blind_forward)

    with pytest.raises(AssertionError):
        check_allocation_optimality(EXAMPLE, _auction(EXAMPLE)(EXAMPLE.confidences, EXAMPLE.wealth))


def test_truthfulness_fails_under_the_proportional_share():
    auction = _auction(EXAMPLE, routing_share=ROUTING_SHARE_PROPORTIONAL)
    single = Market(EXAMPLE.top_k, EXAMPLE.wealth, EXAMPLE.confidences[:, :1])

    assert _best_deviation_gain(auction, single, 0.5) > 1e-3


def test_the_example_market_exercises_every_checker():
    """The pairings above are only evidence if the reference auction passes on the fixture."""
    check_all(EXAMPLE, _auction(EXAMPLE))
