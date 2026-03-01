import torch
import torch.nn as nn

from mob import ConfidenceHead, Expert, LightweightExpert


def test_confidence_head_output_shape():
    head = ConfidenceHead(hidden_dim=32)
    x = torch.randn(2, 4, 32)
    out = head(x)
    assert out.shape == (2, 4, 1)


def test_confidence_head_output_range():
    head = ConfidenceHead(hidden_dim=32)
    x = torch.randn(2, 4, 32)
    out = head(x)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


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
