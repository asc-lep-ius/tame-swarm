import pytest
import torch
import torch.nn as nn

from mob import ConfidenceHead, Expert, LightweightExpert


def test_confidence_head_output_shape():
    head = ConfidenceHead(hidden_dim=32)
    x = torch.randn(2, 4, 32)
    out = head(x)
    assert out.shape == (2, 4, 1)


def test_confidence_head_reports_a_non_negative_unbounded_value():
    """The report is a loss-reduction estimate, not a probability.

    Non-negativity is what the auction needs -- a bid of zero is how an expert
    abstains. An upper bound of 1.0 is what it must NOT have: the reward the report
    predicts is unbounded, and a report capped below the value it reports cannot
    equal that value, which is what truthful reporting means.
    """
    head = ConfidenceHead(hidden_dim=32)
    out = head(torch.randn(2, 4, 32))
    assert (out >= 0.0).all()

    with torch.no_grad():
        head.proj.weight.fill_(1.0)
        head.proj.bias.fill_(5.0)
    assert (head(torch.ones(1, 1, 32)) > 1.0).all(), "report is capped in probability space"


def test_confidence_head_forward_logits_clamps_projection():
    head = ConfidenceHead(hidden_dim=2)
    with torch.no_grad():
        head.proj.weight.fill_(100.0)
        head.proj.bias.zero_()

    high_logits = head.forward_logits(torch.ones(1, 3, 2))
    low_logits = head.forward_logits(-torch.ones(1, 3, 2))

    assert high_logits.shape == (1, 3, 1)
    assert torch.equal(high_logits, torch.full_like(high_logits, 20.0))
    assert torch.equal(low_logits, torch.full_like(low_logits, -20.0))


def test_confidence_head_forward_is_softplus_of_logits():
    head = ConfidenceHead(hidden_dim=32)
    x = torch.randn(2, 4, 32)

    assert torch.allclose(head(x), torch.nn.functional.softplus(head.forward_logits(x)))


def test_confidence_head_report_is_bounded_by_the_logit_clamp():
    """Softplus is unbounded, but the logit clamp keeps bids finite.

    That is what stops one expert's runaway value estimate from taking every token
    without needing a separate calibration constant to cap the report.
    """
    head = ConfidenceHead(hidden_dim=2)
    with torch.no_grad():
        head.proj.weight.fill_(1e6)
        head.proj.bias.zero_()

    assert head(torch.ones(1, 1, 2)).item() == pytest.approx(20.0, abs=1e-4)


def test_expert_output_shape():
    expert = Expert(hidden_dim=32, intermediate_dim=64)
    x = torch.randn(2, 4, 32)
    out = expert(x)
    assert out.shape == (2, 4, 32)


def test_lightweight_expert_output_shape():
    base_gate = nn.Linear(32, 64, bias=False)
    base_up = nn.Linear(32, 64, bias=False)
    base_down = nn.Linear(64, 32, bias=False)

    lw_expert = LightweightExpert(
        hidden_dim=32,
        intermediate_dim=64,
        rank=4,
        alpha=4.0,
    )

    x = torch.randn(2, 4, 32)
    out = lw_expert(x, base_gate, base_up, base_down)
    assert out.shape == (2, 4, 32)


def test_lightweight_expert_zero_init():
    base_gate = nn.Linear(32, 64, bias=False)
    base_up = nn.Linear(32, 64, bias=False)
    base_down = nn.Linear(64, 32, bias=False)

    lw_expert = LightweightExpert(
        hidden_dim=32,
        intermediate_dim=64,
        rank=4,
        alpha=4.0,
    )

    x = torch.randn(1, 4, 32)

    base_only = base_down(torch.nn.functional.silu(base_gate(x)) * base_up(x))
    adapted = lw_expert(x, base_gate, base_up, base_down)

    delta = (adapted - base_only).abs().max().item()
    assert delta < 1e-4, f"Fresh LightweightExpert should produce near-zero delta, got {delta}"
