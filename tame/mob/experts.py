import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """Lightweight linear layer for each expert to predict its confidence in handling a given token."""

    def __init__(self, hidden_dim: int, expert_id: int = 0, num_experts: int = 8):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=True)
        self.expert_id = expert_id

        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        bias_offset = (expert_id - num_experts / 2) * 0.1
        nn.init.constant_(self.proj.bias, bias_offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Confidence scores of shape (batch, seq_len, 1)
        """
        logits = self.proj(x)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        return torch.sigmoid(logits)


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

        gate_out = gate_out + self.gate_adapter_B(self.gate_adapter_A(x)) * self.scaling
        up_out = up_out + self.up_adapter_B(self.up_adapter_A(x)) * self.scaling

        hidden = F.silu(gate_out) * up_out

        output = base_down(hidden) + self.down_adapter_B(self.down_adapter_A(hidden)) * self.scaling

        if output.dtype == torch.bfloat16 or output.dtype == torch.float16:
            output = torch.clamp(output, min=-65000.0, max=65000.0)

        return output
