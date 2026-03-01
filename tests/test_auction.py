import torch

from mob import VCGAuctioneer


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


def test_vcg_payments_non_negative():
    auctioneer = _make_auction(num_experts=4, top_k=2)
    auctioneer.eval()

    confidences = torch.randn(2, 8, 4).abs()
    wealth = torch.ones(4)

    _, _, payments = auctioneer(confidences, wealth)
    assert (payments >= 0).all()


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
    loss = routing_weights.sum()
    loss.backward()
    assert confidences.grad is not None
    assert (confidences.grad.abs() > 0).any()
