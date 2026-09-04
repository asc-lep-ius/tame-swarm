import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIDENCE_LOGIT_MIN = -20.0
CONFIDENCE_LOGIT_MAX = 20.0

# Every head starts reporting a value of about 0.02: an expert has demonstrated
# nothing at upcycling, and its bid says so. The offset this replaced seeded each
# bias monotone in expert index, which made report and participation monotone in
# index at step 0 and left r(wealth, index) ~ -0.9 across every seed -- the #15
# initialisation artefact -- and it started every report near 0.7 while every
# true value was zero, so each early winner paid for value that did not exist and
# the experts that lost got rich on the rebates. Measured on the planted-competence
# fixture (scripts/synthetic_economy.py, 600 steps, 3 seeds): from here,
# r(wealth, win share) ~ 1.0, r(wealth, competence) ~ 0.8, report equals realised
# value to three decimals and every win carries a surplus; from the old offset,
# r(wealth, competence) ranged -0.78 to +0.68 with the index artefact intact.
# Symmetry is broken by the random projection, which is token-dependent, rather
# than by a bias every token shares.
CONFIDENCE_INITIAL_LOGIT = -4.0


class ConfidenceHead(nn.Module):
    """Each expert's report of the value it expects to deliver on a token.

    The report is a *value estimate in loss-reduction units*, not a probability.
    That is what lets one currency run through the whole mechanism: the report is
    the bid, the price is the report's critical value, and the wealth update moves
    by ``reward - charge`` with a single coefficient. A sigmoid report would be
    bounded in (0, 1) while the reward it is supposed to predict is not, and the
    two thresholds -- "win when report > price" and "profit when value > price" --
    would not coincide.

    ``softplus`` is what makes truthful reporting expressible at the bottom of the
    range: an expert whose expected loss reduction is zero or negative would rather
    not win, and reports ~0, which is the auction's way of abstaining. The logit
    clamp bounds the report to roughly [0, 20] loss-reduction units, far above any
    realistic value, so bids stay finite without a separate calibration constant.

    Every head starts there, abstaining. Value appears as the adapters learn, and
    a head's report rises from zero only as the value it realises does -- see
    ``CONFIDENCE_INITIAL_LOGIT``.
    """

    def __init__(self, hidden_dim: int, expert_id: int = 0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=True)
        self.expert_id = expert_id

        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.constant_(self.proj.bias, CONFIDENCE_INITIAL_LOGIT)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Clamped confidence logits of shape (batch, seq_len, 1)
        """
        logits = self.proj(x)
        return torch.clamp(logits, min=CONFIDENCE_LOGIT_MIN, max=CONFIDENCE_LOGIT_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Non-negative value reports of shape (batch, seq_len, 1)
        """
        return F.softplus(self.forward_logits(x))


class Expert(nn.Module):
    """Individual expert FFN module using SwiGLU activation."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU FFN forward pass."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LightweightExpert(nn.Module):
    """Memory-efficient expert using LoRA-style adapters on a shared base."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        rank: int = 64,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.gate_adapter_A = nn.Linear(hidden_dim, rank, bias=False)
        self.gate_adapter_B = nn.Linear(rank, intermediate_dim, bias=False)

        self.up_adapter_A = nn.Linear(hidden_dim, rank, bias=False)
        self.up_adapter_B = nn.Linear(rank, intermediate_dim, bias=False)

        self.down_adapter_A = nn.Linear(intermediate_dim, rank, bias=False)
        self.down_adapter_B = nn.Linear(rank, hidden_dim, bias=False)

        for name, param in self.named_parameters():
            if "_A" in name:
                nn.init.kaiming_uniform_(param, a=5**0.5)
            else:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        base_gate: nn.Linear,
        base_up: nn.Linear,
        base_down: nn.Linear,
    ) -> torch.Tensor:
        """
        Forward pass using shared base + expert-specific adapters.

        Args:
            x: Input tensor
            base_gate, base_up, base_down: Shared base FFN projections
        """
        gate_out = base_gate(x)
        up_out = base_up(x)
        hidden = self._adapted_hidden(x, gate_out, up_out)
        return self._down(hidden, base_down)

    def forward_with_reference(
        self,
        x: torch.Tensor,
        base_gate: nn.Linear,
        base_up: nn.Linear,
        base_down: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The expert's output beside what the shared base alone would have produced.

        The difference between the two is the expert's *contribution*: everything
        its adapters add to the tissue's default behaviour on this token, and the
        thing the economy prices. Both outputs come from one evaluation of the base
        gate and up projections, so the reference costs a single extra down
        projection rather than a second FFN pass. It carries no graph -- it exists
        to be subtracted from a detached output, never to be trained through.
        """
        gate_out = base_gate(x)
        up_out = base_up(x)
        hidden = self._adapted_hidden(x, gate_out, up_out)
        with torch.no_grad():
            reference = self._clamp_half_precision(base_down(F.silu(gate_out) * up_out))
        return self._down(hidden, base_down), reference

    def _adapted_hidden(
        self, x: torch.Tensor, gate_out: torch.Tensor, up_out: torch.Tensor
    ) -> torch.Tensor:
        gate_out = gate_out + self.gate_adapter_B(self.gate_adapter_A(x)) * self.scaling
        up_out = up_out + self.up_adapter_B(self.up_adapter_A(x)) * self.scaling
        return F.silu(gate_out) * up_out

    def _down(self, hidden: torch.Tensor, base_down: nn.Linear) -> torch.Tensor:
        output = base_down(hidden) + self.down_adapter_B(self.down_adapter_A(hidden)) * self.scaling
        return self._clamp_half_precision(output)

    @staticmethod
    def _clamp_half_precision(output: torch.Tensor) -> torch.Tensor:
        if output.dtype == torch.bfloat16 or output.dtype == torch.float16:
            output = torch.clamp(output, min=-65000.0, max=65000.0)
        return output
