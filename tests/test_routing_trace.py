"""The bounded per-token routing record (#5).

Each property is paired with the state that disables its mechanism, as #6 requires:
the correlation is measured against a trace with no direction installed and against
one whose alignment is uncorrelated with the wins; the training exclusion against
the same forward in eval mode; the ring's bound against a window large enough to
hold everything.
"""

import pytest
import torch

from mob import DEFAULT_TRACE_TOKENS, MixtureOfBidders, MoBConfig, RoutingTrace
from mob.auction import ROUTING_SATURATION_THRESHOLD
from mob.routing_trace import MIN_TOKENS_FOR_CORRELATION

HIDDEN = 32
NUM_EXPERTS = 4
TOP_K = 2


def _config(**overrides) -> MoBConfig:
    return MoBConfig(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_dim=HIDDEN,
        intermediate_dim=64,
        adapter_rank=4,
        adapter_alpha=4.0,
        **overrides,
    )


def _layer() -> MixtureOfBidders:
    torch.manual_seed(0)
    layer = MixtureOfBidders(_config())
    layer.eval()
    return layer


def test_an_eval_forward_records_every_token_and_a_training_forward_records_none():
    """The trace is serve-time telemetry: training's tokens are a regime nobody asked about."""
    layer = _layer()
    trace = layer.enable_routing_trace(maxlen=512)
    hidden = torch.randn(2, 20, HIDDEN)

    layer(hidden)
    assert trace.tokens == 40

    layer.train()
    layer(hidden)
    assert trace.tokens == 40, "a training forward must leave the window untouched"

    layer.eval()
    layer(hidden)
    assert trace.tokens == 80


def test_disable_stops_recording_and_clear_empties_the_window():
    layer = _layer()
    trace = layer.enable_routing_trace(maxlen=512)
    layer(torch.randn(1, 8, HIDDEN))
    assert trace.tokens == 8

    trace.clear()
    assert trace.tokens == 0 and trace.summary().tokens == 0

    layer.disable_routing_trace()
    layer(torch.randn(1, 8, HIDDEN))
    assert layer.routing_trace is None and trace.tokens == 0


def test_the_window_is_bounded_and_keeps_the_most_recent_tokens():
    """A ring, not a list: an unbounded trace on a long-running server is a leak."""
    trace = RoutingTrace(NUM_EXPERTS, TOP_K, maxlen=16)
    weights = torch.full((1, 40, TOP_K), 0.5)
    experts = torch.zeros(1, 40, TOP_K, dtype=torch.long)
    experts[..., 1] = 1
    trace.record(weights, experts, torch.randn(1, 40, HIDDEN))
    assert trace.tokens == 16

    # One forward wider than the whole window leaves only its own tail.
    trace.clear()
    trace.record(weights, experts, torch.randn(1, 40, HIDDEN))
    assert trace.tokens == 16
    assert trace.summary().win_share == [1.0, 1.0, 0.0, 0.0]


def test_the_summary_reduces_the_window_to_the_gate_statistics():
    """``win_share`` sums to ``top_k``: every token buys ``top_k`` slots, not one."""
    trace = RoutingTrace(NUM_EXPERTS, TOP_K, maxlen=256)
    weights = torch.tensor([[[0.995, 0.005], [0.6, 0.4]]])
    experts = torch.tensor([[[0, 1], [2, 3]]])
    trace.record(weights, experts, torch.randn(1, 2, HIDDEN))

    summary = trace.summary()
    assert summary.tokens == 2
    assert summary.top1_mean == pytest.approx(0.7975)
    assert summary.top1_saturated_fraction == 0.5, ROUTING_SATURATION_THRESHOLD
    assert sum(summary.win_share) == TOP_K
    # exp(entropy) of a near-degenerate split is ~1, of a 0.6/0.4 split ~1.96.
    assert 1.4 < summary.effective_experts < 1.6
    assert summary.goal_alignment_mean is None and summary.goal_correlation is None


def test_the_correlation_is_none_without_a_direction_and_a_number_with_one():
    """Without an installed direction there is no independent variable to correlate against."""
    tokens = 4 * MIN_TOKENS_FOR_CORRELATION
    generator = torch.Generator().manual_seed(0)
    direction = torch.zeros(HIDDEN)
    direction[0] = 1.0

    hidden = torch.randn(1, tokens, HIDDEN, generator=generator)
    # Expert 0 wins exactly on the tokens that point along the direction.
    aligned = hidden[0, :, 0] > 0
    experts = torch.where(
        aligned.view(1, -1, 1), torch.tensor([[[0, 1]]]), torch.tensor([[[2, 3]]])
    ).expand(1, tokens, TOP_K)
    weights = torch.full((1, tokens, TOP_K), 0.5)

    blind = RoutingTrace(NUM_EXPERTS, TOP_K, maxlen=tokens)
    blind.record(weights, experts, hidden)
    assert blind.summary().goal_correlation is None

    seeing = RoutingTrace(NUM_EXPERTS, TOP_K, maxlen=tokens)
    seeing.set_direction(direction)
    seeing.record(weights, experts, hidden)
    correlation = seeing.summary().goal_correlation
    assert correlation is not None
    assert correlation[0] is not None and correlation[0] > 0.5
    assert correlation[2] is not None and correlation[2] < -0.5


def test_an_expert_that_wins_everything_correlates_with_nothing():
    """A constant column has no correlation; 0.0 would read as "the goal does not move it"."""
    tokens = 2 * MIN_TOKENS_FOR_CORRELATION
    trace = RoutingTrace(NUM_EXPERTS, TOP_K, maxlen=tokens)
    trace.set_direction(torch.eye(HIDDEN)[0])
    experts = torch.zeros(1, tokens, TOP_K, dtype=torch.long)
    experts[..., 1] = 1
    trace.record(
        torch.full((1, tokens, TOP_K), 0.5),
        experts,
        torch.randn(1, tokens, HIDDEN, generator=torch.Generator().manual_seed(1)),
    )

    correlation = trace.summary().goal_correlation
    assert correlation is not None
    assert correlation[0] is None and correlation[1] is None
    assert correlation[2] is None and correlation[3] is None


def test_a_window_too_short_for_a_correlation_reports_none_rather_than_noise():
    trace = RoutingTrace(NUM_EXPERTS, TOP_K, maxlen=256)
    trace.set_direction(torch.eye(HIDDEN)[0])
    short = MIN_TOKENS_FOR_CORRELATION - 1
    experts = torch.zeros(1, short, TOP_K, dtype=torch.long)
    experts[..., 1] = 1
    trace.record(torch.full((1, short, TOP_K), 0.5), experts, torch.randn(1, short, HIDDEN))

    summary = trace.summary()
    assert summary.tokens == short
    assert summary.goal_correlation is None
    assert summary.goal_alignment_mean is not None, "the alignment itself is still measured"


def test_enabling_the_trace_twice_replaces_the_window():
    layer = _layer()
    first = layer.enable_routing_trace(maxlen=DEFAULT_TRACE_TOKENS)
    layer(torch.randn(1, 4, HIDDEN))
    assert first.tokens == 4

    second = layer.enable_routing_trace(maxlen=8)
    assert second is not first and second.maxlen == 8 and second.tokens == 0
