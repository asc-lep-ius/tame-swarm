"""Re-entry (#38's fourth property, folding in #26): an expert that has been shut
out is re-sampled faster the longer it has been shut out, over N steps every
expert holds at least one token, and the uniform draw -- the inert state --
re-samples a starved expert no faster than a merely unlucky one.

The draw reads a ledger of settled steps since each expert last held a token.
No report in the token's own auction writes it; a report can only lose now to
be staler later, and the O(exploration_rate) deviation bound covers that: one
explored token carries one gift, so no expert's chance of it on any token
exceeds the rate whatever the weights."""

import pytest
import torch

from mob import MixtureOfBidders, MoBConfig, VCGAuctioneer

NUM_EXPERTS = 8
TOP_K = 2
TOKENS = 40000
STARVED = NUM_EXPERTS - 1


def _gift_share(draw: str, staleness: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Each expert's share of the explored slots, with expert 7 always a loser."""
    torch.manual_seed(seed)
    auctioneer = VCGAuctioneer(NUM_EXPERTS, TOP_K, exploration_rate=0.5, exploration_draw=draw)
    auctioneer.train()
    confidences = torch.rand(1, TOKENS, NUM_EXPERTS) * 0.5 + 0.5
    confidences[..., STARVED] = 0.0
    outcome = auctioneer(confidences, torch.ones(NUM_EXPERTS), staleness=staleness)
    assert outcome.explored is not None
    explorer = outcome.selected_experts[outcome.explored]
    return torch.bincount(explorer, minlength=NUM_EXPERTS).float() / explorer.numel()


def _starved_for(steps: int) -> torch.Tensor:
    staleness = torch.zeros(NUM_EXPERTS)
    staleness[STARVED] = float(steps)
    return staleness


def test_a_starved_expert_is_re_sampled_faster_the_longer_it_has_been_starved():
    shares = [
        _gift_share("staleness", _starved_for(steps))[STARVED].item() for steps in (0, 1, 10, 150)
    ]

    assert shares == sorted(shares), shares
    # Weight 1 + s against five other losers at weight 1: 1/6, 2/7, 11/16, 151/156.
    assert shares[0] == pytest.approx(1 / 6, abs=0.02)
    assert shares[1] == pytest.approx(2 / 7, abs=0.02)
    assert shares[3] > 0.9, shares


def test_the_uniform_draw_re_samples_a_starved_expert_no_faster():
    """The inert state the property fails in: every arm before #38."""
    unlucky = _gift_share("uniform", _starved_for(1))[STARVED].item()
    starved = _gift_share("uniform", _starved_for(150))[STARVED].item()

    assert unlucky == pytest.approx(1 / 6, abs=0.02)
    assert starved == pytest.approx(1 / 6, abs=0.02)


def test_the_staleness_draw_is_the_uniform_draw_when_nothing_is_stale():
    staleness = torch.zeros(NUM_EXPERTS)
    weighted = _gift_share("staleness", staleness)
    uniform = _gift_share("uniform", staleness)

    assert torch.allclose(weighted, uniform, atol=0.02)
    assert weighted[STARVED] == pytest.approx(1 / 6, abs=0.02)


def test_a_draw_outside_the_two_is_refused():
    with pytest.raises(ValueError, match="exploration draw"):
        VCGAuctioneer(NUM_EXPERTS, TOP_K, exploration_draw="ucb")
    with pytest.raises(ValueError, match="exploration draw"):
        MoBConfig(exploration_draw="ucb")


def _layer(exploration_rate: float, exploration_draw: str = "staleness") -> MixtureOfBidders:
    torch.manual_seed(0)
    config = MoBConfig(
        num_experts=4,
        top_k=2,
        hidden_dim=16,
        intermediate_dim=32,
        adapter_rank=4,
        adapter_alpha=4.0,
        exploration_rate=exploration_rate,
        exploration_draw=exploration_draw,
    )
    mob = MixtureOfBidders(config)
    mob.train()
    # Expert 3 reports far below the others, so the auction never picks it and
    # only the gift can.
    with torch.no_grad():
        mob.confidence_heads[3].proj.bias.fill_(-20.0)
    return mob


def test_the_ledger_ages_the_shut_out_and_resets_the_holders():
    mob = _layer(exploration_rate=0.0)
    for _ in range(3):
        mob(torch.randn(2, 8, 16))

    assert mob.expert_steps_since_held.tolist() == [0.0, 0.0, 0.0, 3.0]


def test_over_n_steps_every_expert_holds_a_token():
    """Under the staleness draw the shut-out expert's weight grows every step it
    is passed over, so it holds a token within a handful of steps even at the
    shipped rate on a small batch."""
    mob = _layer(exploration_rate=0.02)
    steps_until_held = None
    for step in range(200):
        mob(torch.randn(2, 8, 16))
        if mob.expert_steps_since_held[3].item() == 0.0:
            steps_until_held = step
            break

    assert steps_until_held is not None and steps_until_held < 20, steps_until_held
    assert mob.expert_usage_count[3].item() >= 1.0


def test_the_ledger_is_at_rest_before_the_first_step_and_frozen_does_not_move_it():
    mob = _layer(exploration_rate=0.02)
    assert mob.expert_steps_since_held.tolist() == [0.0] * 4

    mob(torch.randn(2, 8, 16), update_wealth=False)

    assert mob.expert_steps_since_held.tolist() == [0.0] * 4


def _expected_utility(
    auctioneer: VCGAuctioneer,
    report: float,
    true_value: float,
    draws: int,
    staleness: torch.Tensor,
    gift_per_slot: float = 0.0,
) -> float:
    """Expert 0's expected quasi-linear payoff over the exploration draws,
    ``test_exploration.py``'s helper with the staleness ledger passed in and,
    for #65, a credit per explored slot the expert drew."""
    confidences = torch.tensor([report, 0.30, 0.40, 0.15, 0.20]).view(1, 1, -1).expand(1, draws, -1)
    wealth = torch.tensor([1.0, 1.25, 0.9, 1.4, 1.5])
    outcome = auctioneer(confidences.contiguous(), wealth, staleness=staleness)

    holds = outcome.selected_experts == 0
    influence = (outcome.routing_weights * auctioneer.top_k * holds).sum(dim=-1)
    paid = (outcome.payments * holds).sum(dim=-1)
    assert outcome.explored is not None
    gifted = gift_per_slot * (outcome.explored & holds).sum(dim=-1).float()
    return (true_value * influence - paid + gifted).mean().item()


def test_the_deviation_bound_holds_for_a_starved_deviator():
    """The re-derived O(rate) bound at the point the weighting is strongest: the
    deviator is the stalest loser by far, so losing on purpose buys it nearly
    the whole lottery -- and the lottery is still worth at most rate x value."""
    rate = 0.1
    auctioneer = VCGAuctioneer(5, 2, exploration_rate=rate)
    auctioneer.train()
    staleness = torch.tensor([200.0, 0.0, 0.0, 0.0, 0.0])
    draws = 20000

    for true_value in (0.30, 0.365, 0.38, 0.45, 0.60):
        torch.manual_seed(5)
        truthful = _expected_utility(auctioneer, true_value, true_value, draws, staleness)
        for report in torch.linspace(0.0, 1.0, 21).tolist():
            torch.manual_seed(5)
            deviation = _expected_utility(auctioneer, report, true_value, draws, staleness)
            assert deviation <= truthful + rate * true_value + 1e-3, (
                f"value {true_value:.2f}: reporting {report:.2f} gained "
                f"{deviation - truthful:.4f} against a bound of {rate * true_value:.4f}"
            )
            if report > true_value:
                assert deviation <= truthful + 1e-3


def test_a_gift_per_explored_slot_raises_the_bound_by_the_whole_lottery():
    """#65's candidate 1 costs exactly what the staleness draw hands the stalest loser.

    A credit per explored slot is a credit for *losing*, and the starved
    deviator wins nearly every explored slot, so the O(rate) bound grows by
    ``rate x gift`` -- the whole lottery -- and not by the ``1 / (n - k)``
    share a merely unlucky loser would average. The negative control is the
    share bound: it is broken at the same gift, which is the difference
    between a mean and a bound.
    """
    rate = 0.1
    gift = 0.5
    auctioneer = VCGAuctioneer(5, 2, exploration_rate=rate)
    auctioneer.train()
    staleness = torch.tensor([200.0, 0.0, 0.0, 0.0, 0.0])
    draws = 20000
    share = rate / (5 - 2)
    share_bound_broken = False

    for true_value in (0.30, 0.365, 0.38, 0.45, 0.60):
        torch.manual_seed(5)
        truthful = _expected_utility(auctioneer, true_value, true_value, draws, staleness, gift)
        for report in torch.linspace(0.0, 1.0, 21).tolist():
            torch.manual_seed(5)
            deviation = _expected_utility(auctioneer, report, true_value, draws, staleness, gift)
            assert deviation <= truthful + rate * (true_value + gift) + 1e-3, (
                f"value {true_value:.2f}: reporting {report:.2f} gained "
                f"{deviation - truthful:.4f} against rate x (value + gift) "
                f"{rate * (true_value + gift):.4f}"
            )
            if deviation > truthful + rate * true_value + share * gift + 1e-3:
                share_bound_broken = True

    assert share_bound_broken, "the exploration-share bound held; the lottery is not the bound"
