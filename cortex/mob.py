"""
Mixture of Bidders (MoB) - Agential Swarm Architecture

This module implements the economic auction-based routing mechanism
described in the TAME architecture. Instead of a learned router (standard MoE),
we use a VCG (Vickrey-Clarke-Groves) auction where experts bid for tokens
based on their confidence and accumulated wealth.

Key concepts:
- Each expert has a "wallet" of credits
- Experts bid based on confidence × wealth
- Winners are selected via VCG second-price auction
- Wealth updates based on performance (loss reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MoBConfig:
    """Configuration for Mixture of Bidders module."""
    num_experts: int = 8  # Number of expert FFNs
    top_k: int = 2  # How many experts process each token
    hidden_dim: int = 4096  # Model hidden dimension
    intermediate_dim: int = 14336  # FFN intermediate dimension (Mistral default)
    initial_wealth: float = 100.0  # Starting credits for each expert
    wealth_decay: float = 0.99  # Wealth decay per step (prevents runaway accumulation)
    min_wealth: float = 1.0  # Minimum wealth (prevents bankruptcy death spiral)
    jitter_std: float = 0.01  # Gaussian noise for symmetry breaking


class ConfidenceHead(nn.Module):
    """
    Lightweight linear layer for each expert to predict its confidence
    in handling a given token. This is the expert's "valuation" in the auction.
    
    Maps: hidden_dim -> 1 (scalar confidence score)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=True)
        # Initialize to produce moderate confidence initially
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Confidence scores of shape (batch, seq_len, 1)
        """
        return torch.sigmoid(self.proj(x))


class Expert(nn.Module):
    """
    Individual expert FFN module. Mirrors the structure of the original
    model's FFN but with potential for specialized behavior.
    
    For Mistral/Llama: SwiGLU activation
    FFN(x) = (gate(x) * up(x)) @ down
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU FFN forward pass."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VCGAuctioneer(nn.Module):
    """
    Implements the VCG (Vickrey-Clarke-Groves) auction mechanism.
    
    VCG is a second-price auction that incentivizes truthful bidding:
    - Each expert bids: bid = confidence × wealth
    - Top-k experts win
    - Winners pay the k+1th highest bid (not their own bid)
    
    This creates emergent specialization as experts accumulate wealth
    by successfully processing tokens they're confident about.
    """
    
    def __init__(self, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(
        self, 
        confidences: torch.Tensor,  # (batch, seq, num_experts)
        wealth: torch.Tensor  # (num_experts,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the VCG auction.
        
        Args:
            confidences: Expert confidence scores (batch, seq_len, num_experts)
            wealth: Current wealth of each expert (num_experts,)
            
        Returns:
            selected_experts: Indices of winning experts (batch, seq_len, top_k)
            routing_weights: Normalized weights for selected experts (batch, seq_len, top_k)
            payments: What each winning expert pays (batch, seq_len, top_k)
        """
        batch_size, seq_len, _ = confidences.shape
        
        # Compute bids: confidence × wealth
        # Wealth is broadcasted across batch and sequence
        bids = confidences * wealth.unsqueeze(0).unsqueeze(0)  # (batch, seq, num_experts)
        
        # Select top-k experts for each token
        top_bids, selected_experts = torch.topk(bids, self.top_k, dim=-1)
        
        # VCG payment: pay the (k+1)th highest bid
        # For simplicity, we use the k+1 th bid if available, else 0
        if self.top_k < self.num_experts:
            # Get all bids, find the k+1th
            all_sorted, _ = torch.sort(bids, dim=-1, descending=True)
            payments = all_sorted[:, :, self.top_k:self.top_k+1].expand(-1, -1, self.top_k)
        else:
            payments = torch.zeros_like(top_bids)
            
        # Normalize routing weights (softmax over selected experts' bids)
        routing_weights = F.softmax(top_bids, dim=-1)
        
        return selected_experts, routing_weights, payments


class MixtureOfBidders(nn.Module):
    """
    The complete Mixture of Bidders layer that replaces a standard FFN.
    
    This implements the "Agential Swarm" concept from TAME:
    - Multiple expert FFNs act as semi-autonomous agents
    - They bid on tokens via economic auction (VCG)
    - Wealth accumulates based on performance
    - Emergent specialization occurs naturally
    
    This transforms static matrix multiplication into a dynamic
    competitive economy within the neural network.
    """
    
    def __init__(self, config: MoBConfig):
        super().__init__()
        self.config = config
        
        # Create the expert pool
        self.experts = nn.ModuleList([
            Expert(config.hidden_dim, config.intermediate_dim)
            for _ in range(config.num_experts)
        ])
        
        # Confidence head for each expert
        self.confidence_heads = nn.ModuleList([
            ConfidenceHead(config.hidden_dim)
            for _ in range(config.num_experts)
        ])
        
        # The auctioneer
        self.auctioneer = VCGAuctioneer(config.num_experts, config.top_k)
        
        # Expert wealth (persistent state across forward passes)
        # This is the "economy" that drives specialization
        self.register_buffer(
            'expert_wealth',
            torch.full((config.num_experts,), config.initial_wealth)
        )
        
        # Track expert usage for monitoring
        self.register_buffer(
            'expert_usage_count',
            torch.zeros(config.num_experts)
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        update_wealth: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the MoB layer.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            update_wealth: Whether to update expert wealth (False during eval)
            
        Returns:
            output: Processed tensor (batch, seq_len, hidden_dim)
            stats: Dictionary of routing statistics for monitoring
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 1. Each expert evaluates its confidence for each token
        confidences = torch.stack([
            head(hidden_states).squeeze(-1)
            for head in self.confidence_heads
        ], dim=-1)  # (batch, seq_len, num_experts)
        
        # 2. Run the auction
        selected_experts, routing_weights, payments = self.auctioneer(
            confidences, self.expert_wealth
        )
        
        # 3. Process tokens through selected experts
        # This is the sparse computation - only top_k experts run per token
        output = torch.zeros_like(hidden_states)
        
        for k in range(self.config.top_k):
            expert_indices = selected_experts[:, :, k]  # (batch, seq_len)
            weights = routing_weights[:, :, k:k+1]  # (batch, seq_len, 1)
            
            # Process through each expert (vectorized per expert)
            for expert_idx in range(self.config.num_experts):
                # Create mask for tokens assigned to this expert
                mask = (expert_indices == expert_idx)
                if not mask.any():
                    continue
                    
                # Get tokens for this expert
                expert_input = hidden_states[mask]  # (num_tokens, hidden_dim)
                expert_output = self.experts[expert_idx](expert_input)
                
                # Weight by routing weight and add to output
                weight_vals = weights.squeeze(-1)[mask]  # (num_tokens,)
                output[mask] += expert_output * weight_vals.unsqueeze(-1)
                
                # Update usage count
                if update_wealth:
                    self.expert_usage_count[expert_idx] += mask.sum().float()
        
        # 4. Update wealth based on performance (simplified reward)
        # In full implementation, this would use actual loss reduction
        if update_wealth and self.training:
            self._update_wealth(selected_experts, routing_weights, confidences)
        
        # Collect statistics
        stats = {
            'confidences': confidences.detach(),
            'selected_experts': selected_experts.detach(),
            'routing_weights': routing_weights.detach(),
            'expert_wealth': self.expert_wealth.clone(),
            'expert_usage': self.expert_usage_count.clone(),
        }
        
        return output, stats
    
    def _update_wealth(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        confidences: torch.Tensor
    ):
        """
        Update expert wealth based on their participation.
        
        In the full TAME architecture, this would incorporate:
        - Actual loss reduction from each expert
        - Long-term performance tracking
        - Penalties for overconfidence
        
        For the prototype, we use a simplified reward:
        - Experts gain wealth proportional to their confidence when selected
        - All experts decay slightly to prevent runaway accumulation
        """
        with torch.no_grad():
            # Decay all wealth
            self.expert_wealth *= self.config.wealth_decay
            
            # Reward selected experts based on their confidence
            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
                    if mask.any():
                        # Reward proportional to confidence when selected
                        reward = confidences[:, :, expert_idx][mask].mean()
                        self.expert_wealth[expert_idx] += reward * 0.1
            
            # Clamp to minimum wealth
            self.expert_wealth.clamp_(min=self.config.min_wealth)
    
    @classmethod
    def from_pretrained_ffn(
        cls,
        ffn_module: nn.Module,
        config: MoBConfig,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> 'MixtureOfBidders':
        """
        Initialize MoB by "upcycling" from a pretrained FFN.
        
        This is the key efficiency gain: we don't train from scratch.
        We duplicate the existing FFN weights across experts and add
        small random noise to break symmetry.
        
        Args:
            ffn_module: The original FFN module (e.g., from Mistral)
            config: MoB configuration
            device: Target device (auto-detected from ffn_module if None)
            dtype: Target dtype (auto-detected from ffn_module if None)
            
        Returns:
            Initialized MoB module with weights copied from FFN
        """
        # Detect device and dtype from the original FFN
        if device is None or dtype is None:
            if hasattr(ffn_module, 'gate_proj'):
                ref_param = ffn_module.gate_proj.weight
            elif hasattr(ffn_module, 'up_proj'):
                ref_param = ffn_module.up_proj.weight
            else:
                ref_param = next(ffn_module.parameters())
            
            if device is None:
                device = ref_param.device
            if dtype is None:
                dtype = ref_param.dtype
        
        logger.info(f"Creating MoB on device={device}, dtype={dtype}")
        
        # Create MoB and move to correct device/dtype
        mob = cls(config)
        mob = mob.to(device=device, dtype=dtype)
        
        # Copy weights from original FFN to each expert
        with torch.no_grad():
            for expert in mob.experts:
                # Copy gate, up, down projections
                if hasattr(ffn_module, 'gate_proj'):
                    expert.gate_proj.weight.copy_(ffn_module.gate_proj.weight)
                    expert.up_proj.weight.copy_(ffn_module.up_proj.weight)
                    expert.down_proj.weight.copy_(ffn_module.down_proj.weight)
                    
                # Add small noise for symmetry breaking
                expert.gate_proj.weight.add_(
                    torch.randn_like(expert.gate_proj.weight) * config.jitter_std
                )
                expert.up_proj.weight.add_(
                    torch.randn_like(expert.up_proj.weight) * config.jitter_std
                )
                expert.down_proj.weight.add_(
                    torch.randn_like(expert.down_proj.weight) * config.jitter_std
                )
        
        logger.info(
            f"Upcycled FFN to MoB with {config.num_experts} experts, "
            f"top-k={config.top_k}"
        )
        return mob


class TAMETransformerLayer(nn.Module):
    """
    A complete Transformer layer with MoB replacing the standard FFN.
    
    This wraps the attention mechanism and adds the MoB layer,
    creating a single "organ" in the agential swarm.
    """
    
    def __init__(
        self,
        attention_module: nn.Module,
        mob_config: MoBConfig,
        layer_norm_eps: float = 1e-5,
        hidden_dim: int = 4096
    ):
        super().__init__()
        self.attention = attention_module
        self.mob = MixtureOfBidders(mob_config)
        
        # Layer norms (standard Transformer architecture)
        self.input_layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the TAME layer.
        
        Returns both the hidden states and MoB routing statistics.
        """
        # Pre-norm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )[0]
        hidden_states = residual + hidden_states
        
        # Pre-norm + MoB (replaces standard FFN)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mob_stats = self.mob(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, mob_stats


def apply_mob_to_model(
    model: nn.Module,
    mob_config: MoBConfig,
    layers_to_modify: Optional[List[int]] = None
) -> nn.Module:
    """
    Apply MoB transformation to an existing pretrained model.
    
    This is the main entry point for "upcycling" a dense model
    into a TAME-compliant agential swarm architecture.
    
    Args:
        model: Pretrained HuggingFace model (Mistral, Llama, etc.)
        mob_config: Configuration for MoB modules
        layers_to_modify: Which layers to convert (None = all)
        
    Returns:
        Modified model with MoB layers
    """
    # Find the transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Cannot find transformer layers in model")
    
    num_layers = len(layers)
    if layers_to_modify is None:
        # Modify middle layers (where most reasoning happens)
        # Skip first and last few layers
        layers_to_modify = list(range(4, num_layers - 4))
    
    for layer_idx in layers_to_modify:
        layer = layers[layer_idx]
        
        # Find the FFN module (different names in different architectures)
        if hasattr(layer, 'mlp'):
            ffn = layer.mlp
        elif hasattr(layer, 'feed_forward'):
            ffn = layer.feed_forward
        else:
            logger.warning(f"Layer {layer_idx}: Cannot find FFN module, skipping")
            continue
            
        # Create MoB from the FFN
        mob = MixtureOfBidders.from_pretrained_ffn(ffn, mob_config)
        
        # Replace the FFN with MoB
        if hasattr(layer, 'mlp'):
            layer.mlp = mob
        else:
            layer.feed_forward = mob
            
        logger.info(f"Layer {layer_idx}: Replaced FFN with MoB")
    
    return model
