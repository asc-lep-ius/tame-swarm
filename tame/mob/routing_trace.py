"""What the gate did over the last N served tokens, and what the goal did to it (#5).

At serve time a decode forward carries one token, so ``MoBStats.routing`` is a
statistic of that token and a correlation over it is undefined. This keeps the
last ``maxlen`` tokens per layer instead: each token's routing weights, the
experts that won its slots, and how far its own hidden state pointed along the
goal direction. Those are the columns ``steering_routing_correlation`` needs --
the live form of what ``tests/test_coupling.py`` measures offline, and the number
that says whether the goal shapes *which cells activate* rather than only what
they output.

**The record path is the hot path**, and it is written to be as close to nothing
as a record can be. Each forward copies three tensors into preallocated device
buffers through plain contiguous slices -- no host synchronisation, no advanced
indexing, no statistics. Every reduction (top-1 share, effective experts, win
share, the correlation) is deferred to :meth:`RoutingTrace.summary`, which a
metrics route calls once over at most ``maxlen`` rows. This matters more than it
looks: on a 1.7B model a decode step is a few milliseconds spread over sixteen
converted layers, so the cost of telemetry here is counted in kernel launches per
layer per token, not in arithmetic. Computing the statistics eagerly measured
2.8% of the forward pass and 7.7% of throughput; deferring them brings it to 0.9%
and 1.9%, which is what makes the trace cheap enough to leave on
(``scripts/benchmark_coupling.py``).

The trace is serve-time telemetry: :class:`~mob.core.MixtureOfBidders` records
only outside training mode, so a training step's ``(batch, seq)`` tokens never
enter it and the training hot path is untouched whether or not a trace is
installed.
"""

from dataclasses import dataclass

import torch

from .auction import ENTROPY_PROBABILITY_FLOOR, ROUTING_SATURATION_THRESHOLD

# Enough tokens for a correlation with a usable standard error (1/sqrt(n) is 2%
# here) at a few tens of kilobytes per layer, and small enough that the window is
# recent: a served answer of a few hundred tokens is the unit anyone reads this
# about.
DEFAULT_TRACE_TOKENS = 2048
# Below this many tokens a per-expert correlation is noise; the summary reports
# None rather than a number nobody should act on.
MIN_TOKENS_FOR_CORRELATION = 64


@dataclass(frozen=True)
class RoutingTraceSummary:
    """The window reduced to host floats: one device read, at read time.

    ``win_share`` is each expert's share of the tokens in the window, so it sums
    to ``top_k`` rather than to one. ``goal_correlation`` is, per expert, the
    Pearson correlation over the window between a token's alignment with the goal
    direction and whether that expert won a slot on it -- ``None`` when no
    direction is installed, when the window is too short, or when nothing varies.
    """

    tokens: int
    top1_mean: float
    top1_median: float
    top1_saturated_fraction: float
    effective_experts: float
    win_share: list[float]
    goal_alignment_mean: float | None
    goal_correlation: list[float | None] | None


class RoutingTrace:
    """A bounded per-token record of one MoB layer's routing, kept on the device."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        maxlen: int = DEFAULT_TRACE_TOKENS,
        device: torch.device | None = None,
    ):
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.maxlen = maxlen
        self._weights = torch.zeros(maxlen, top_k, device=device)
        self._experts = torch.zeros(maxlen, top_k, dtype=torch.long, device=device)
        self._alignment = torch.zeros(maxlen, device=device)
        self._direction: torch.Tensor | None = None
        self._written = 0
        self._next = 0

    @property
    def tokens(self) -> int:
        """How many tokens the window holds."""
        return min(self._written, self.maxlen)

    def set_direction(self, direction: torch.Tensor | None) -> None:
        """Install the goal direction this layer's tokens are measured against.

        The stream the confidence heads read is already carrying the tissue's
        injection at a steered block, so this is the alignment the goal produced,
        whether or not a :class:`~coupling.SteeringCoupling` mediates it -- with no
        coupling attached the correlation it feeds is the additive baseline the
        coupled arm has to beat, which is the comparison #5 exists to make live.
        Kept unit-norm and on the buffers' device so the record path is one matrix
        product, not a normalisation and a transfer per token.
        """
        if direction is None:
            self._direction = None
            self.clear()
            return
        flat = direction.detach().reshape(-1).float()
        norm = flat.norm()
        if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
            raise ValueError("goal direction has zero or non-finite norm")
        self._direction = (flat / norm).to(self._alignment.device)
        # Rows recorded before a direction was installed hold no alignment; keeping
        # them would fold a zero into the mean and the correlation as though it had
        # been measured against this direction.
        self.clear()

    def clear(self) -> None:
        self._written = 0
        self._next = 0

    @torch.no_grad()
    def record(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Append one forward's tokens. Three slice copies and, at most, one projection.

        ``routing_weights`` and ``selected_experts`` are ``(batch, seq, top_k)``;
        ``hidden_states`` is the ``(batch, seq, hidden)`` stream the routing path
        read, before any coupling shifted it, so the alignment is the token's own.
        Nothing is reduced and nothing leaves the device.
        """
        weights = routing_weights.detach().reshape(-1, self.top_k)
        experts = selected_experts.detach().reshape(-1, self.top_k)
        count = weights.shape[0]
        if count == 0:
            return

        # A single forward wider than the window can only leave its own tail.
        if count > self.maxlen:
            weights, experts = weights[-self.maxlen :], experts[-self.maxlen :]
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])[-self.maxlen :]
            count = self.maxlen

        alignment = self._alignment_of(hidden_states)
        # Contiguous slices, in at most two pieces: a decode step is always one
        # token and takes the first branch, so the common case is three copies
        # with no index tensor to build.
        start = self._next
        end = start + count
        if end <= self.maxlen:
            self._write(slice(start, end), weights, experts, alignment)
        else:
            # Every column is split at the same token, or the alignment stops
            # describing the routing it sits beside.
            split = self.maxlen - start
            head, tail = (
                (None, None) if alignment is None else (alignment[:split], alignment[split:])
            )
            self._write(slice(start, self.maxlen), weights[:split], experts[:split], head)
            self._write(slice(0, end - self.maxlen), weights[split:], experts[split:], tail)
        self._next = end % self.maxlen
        self._written += count

    def _write(
        self,
        window: slice,
        weights: torch.Tensor,
        experts: torch.Tensor,
        alignment: torch.Tensor | None,
    ) -> None:
        """Copy one contiguous run. Every argument already holds exactly ``window``'s rows."""
        self._weights[window] = weights
        self._experts[window] = experts
        if alignment is not None:
            self._alignment[window] = alignment

    def _alignment_of(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """``cos(h, d)`` per token, or ``None`` when no direction is installed.

        ``None`` rather than a NaN fill: with no direction the column is never
        read, and writing one would put a kernel on the hot path for nothing.
        """
        if self._direction is None:
            return None
        stream = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        return (stream @ self._direction) / stream.norm(dim=-1).clamp_min(1e-12)

    def summary(self) -> RoutingTraceSummary:
        """Reduce the window to host floats. One device read; safe to call per request."""
        tokens = self.tokens
        if tokens == 0:
            return RoutingTraceSummary(0, 0.0, 0.0, 0.0, 0.0, [0.0] * self.num_experts, None, None)

        window = slice(0, tokens)
        weights = self._weights[window].float().cpu()
        experts = self._experts[window].cpu()
        alignment = self._alignment[window].cpu()

        top1 = weights.amax(dim=-1)
        safe = weights.clamp_min(ENTROPY_PROBABILITY_FLOOR)
        effective = (-(safe * safe.log()).sum(dim=-1)).exp()
        wins = torch.zeros(tokens, self.num_experts)
        wins.scatter_(1, experts, 1.0)

        return RoutingTraceSummary(
            tokens=tokens,
            top1_mean=float(top1.mean()),
            top1_median=float(top1.median()),
            top1_saturated_fraction=float((top1 > ROUTING_SATURATION_THRESHOLD).float().mean()),
            effective_experts=float(effective.mean()),
            win_share=wins.mean(dim=0).tolist(),
            goal_alignment_mean=(float(alignment.mean()) if self._direction is not None else None),
            goal_correlation=self._correlation(alignment, wins, tokens),
        )

    def _correlation(
        self, alignment: torch.Tensor, wins: torch.Tensor, tokens: int
    ) -> list[float | None] | None:
        """Pearson correlation of each expert's win indicator with the token's alignment.

        ``None`` rather than zero when it cannot be estimated: no direction, too
        short a window, or an expert that won every token or none of them -- a
        constant column has no correlation, and reporting 0.0 there would read as
        "the goal does not move this expert" when the truth is "nothing varied".
        """
        if self._direction is None or tokens < MIN_TOKENS_FOR_CORRELATION:
            return None
        centred_alignment = alignment - alignment.mean()
        alignment_norm = centred_alignment.norm()
        if not bool(torch.isfinite(alignment_norm)) or float(alignment_norm) == 0.0:
            return None
        centred_wins = wins - wins.mean(dim=0, keepdim=True)
        win_norms = centred_wins.norm(dim=0)
        correlation = torch.where(
            win_norms > 0,
            (centred_alignment @ centred_wins) / (alignment_norm * win_norms.clamp_min(1e-12)),
            torch.full_like(win_norms, float("nan")),
        )
        return [None if value != value else float(value) for value in correlation.tolist()]
