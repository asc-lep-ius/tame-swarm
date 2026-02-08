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
from typing import Optional, Tuple, List, Dict
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
    wealth_decay: float = 0.999  # Wealth decay per step (reduced from 0.99 - was too aggressive)
    min_wealth: float = 10.0  # Minimum wealth (raised to prevent death spiral)
    max_wealth: float = 500.0  # Maximum wealth (prevents runaway monopoly)
    jitter_std: float = 0.05  # Gaussian noise for symmetry breaking (increased for differentiation)
    reward_scale: float = 1.0  # Base reward multiplier
    use_vcg_payments: bool = True  # Whether winners pay (wealth transfer mechanism)
    # Memory-efficient options
    use_shared_base: bool = True  # Use shared base + adapters instead of full copies
    adapter_rank: int = 64  # Rank of LoRA-style adapters when use_shared_base=True
    adapter_alpha: float = 16.0  # Scaling factor for adapters
    # Specialization mechanisms
    use_loss_feedback: bool = True  # Enable loss-based wealth updates (requires training loop integration)
    use_local_quality: bool = True  # Use local quality signals when loss not available
    use_differentiable_routing: bool = True  # Straight-through estimator for confidence head gradients
    confidence_calibration_weight: float = 0.1  # Auxiliary loss weight for confidence calibration
    loss_ema_decay: float = 0.95  # EMA decay for baseline loss tracking


class ConfidenceHead(nn.Module):
    """
    Lightweight linear layer for each expert to predict its confidence
    in handling a given token. This is the expert's "valuation" in the auction.
    
    Maps: hidden_dim -> 1 (scalar confidence score)
    """
    
    def __init__(self, hidden_dim: int, expert_id: int = 0, num_experts: int = 8):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1, bias=True)
        self.expert_id = expert_id
        
        # Initialize with variation between experts to break symmetry
        # Different experts start with different "preferences"
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        # Bias varies per expert to create initial confidence differences
        # This is crucial for breaking the uniform equilibrium
        bias_offset = (expert_id - num_experts / 2) * 0.1
        nn.init.constant_(self.proj.bias, bias_offset)
        
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


class LightweightExpert(nn.Module):
    """
    Memory-efficient expert using LoRA-style adapters on a shared base.
    
    Instead of duplicating the full FFN weights for each expert, we:
    1. Keep one shared base FFN (not owned by this module)
    2. Add small low-rank adapter matrices per expert
    
    This reduces memory from O(num_experts * FFN_size) to 
    O(FFN_size + num_experts * adapter_size), where adapter_size << FFN_size.
    
    The expert's output becomes: base_output + adapter_output
    Where adapter uses low-rank decomposition: x @ A @ B
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        intermediate_dim: int, 
        rank: int = 64,
        alpha: float = 16.0
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # Low-rank adapters for each projection
        # gate adapter: hidden -> rank -> intermediate
        self.gate_adapter_A = nn.Linear(hidden_dim, rank, bias=False)
        self.gate_adapter_B = nn.Linear(rank, intermediate_dim, bias=False)
        
        # up adapter: hidden -> rank -> intermediate
        self.up_adapter_A = nn.Linear(hidden_dim, rank, bias=False)
        self.up_adapter_B = nn.Linear(rank, intermediate_dim, bias=False)
        
        # down adapter: intermediate -> rank -> hidden
        self.down_adapter_A = nn.Linear(intermediate_dim, rank, bias=False)
        self.down_adapter_B = nn.Linear(rank, hidden_dim, bias=False)
        
        # Initialize adapters (LoRA-style: A random, B zeros)
        for name, param in self.named_parameters():
            if '_A' in name:
                nn.init.kaiming_uniform_(param, a=5**0.5)
            else:  # '_B' layers
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        base_gate: nn.Linear,
        base_up: nn.Linear,
        base_down: nn.Linear
    ) -> torch.Tensor:
        """
        Forward pass using shared base + expert-specific adapters.
        
        Args:
            x: Input tensor
            base_gate, base_up, base_down: Shared base FFN projections
        """
        # Base computation
        gate_out = base_gate(x)
        up_out = base_up(x)
        
        # Add adapter contributions (scaled)
        gate_out = gate_out + self.gate_adapter_B(self.gate_adapter_A(x)) * self.scaling
        up_out = up_out + self.up_adapter_B(self.up_adapter_A(x)) * self.scaling
        
        # SwiGLU activation
        hidden = F.silu(gate_out) * up_out
        
        # Down projection with adapter
        output = base_down(hidden) + self.down_adapter_B(self.down_adapter_A(hidden)) * self.scaling
        
        return output


class VCGAuctioneer(nn.Module):
    """
    Implements the VCG (Vickrey-Clarke-Groves) auction mechanism.
    
    VCG is a second-price auction that incentivizes truthful bidding:
    - Each expert bids: bid = confidence × wealth
    - Top-k experts win
    - Winners pay the k+1th highest bid (not their own bid)
    
    This creates emergent specialization as experts accumulate wealth
    by successfully processing tokens they're confident about.
    
    Supports differentiable routing via straight-through estimator for
    gradient flow to confidence heads.
    """
    
    def __init__(self, num_experts: int, top_k: int = 2, differentiable: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.differentiable = differentiable
        
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
        
        # Differentiable routing via straight-through estimator
        # Forward: hard selection (top-k), Backward: soft gradients through all experts
        if self.differentiable and self.training:
            # Soft attention over ALL experts (for gradient flow)
            soft_weights = F.softmax(bids, dim=-1)  # (batch, seq, num_experts)
            
            # Hard selection mask
            hard_mask = torch.zeros_like(bids)
            hard_mask.scatter_(-1, selected_experts, 1.0)
            
            # Straight-through: use hard mask in forward, soft gradients in backward
            # The gradient flows through soft_weights but the value is hard_mask
            differentiable_mask = hard_mask + (soft_weights - soft_weights.detach())
            
            # Gather routing weights for selected experts with gradient flow
            routing_weights_full = differentiable_mask * F.softmax(bids, dim=-1)
            routing_weights = torch.gather(routing_weights_full, -1, selected_experts)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            # Standard routing (no gradient through selection)
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
    
    Supports two modes:
    1. Full experts: Each expert has complete FFN weights (high memory)
    2. Shared base + adapters: One shared FFN + lightweight adapters per expert (low memory)
    """
    
    def __init__(self, config: MoBConfig):
        super().__init__()
        self.config = config
        self.use_shared_base = config.use_shared_base
        
        if self.use_shared_base:
            # Memory-efficient mode: shared base + lightweight adapters
            # Base FFN (shared across all experts)
            self.base_gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.base_up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.base_down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
            
            # Lightweight adapters per expert
            self.experts = nn.ModuleList([
                LightweightExpert(
                    config.hidden_dim, 
                    config.intermediate_dim,
                    rank=config.adapter_rank,
                    alpha=config.adapter_alpha
                )
                for _ in range(config.num_experts)
            ])
        else:
            # Full expert mode (original, high memory)
            self.experts = nn.ModuleList([
                Expert(config.hidden_dim, config.intermediate_dim)
                for _ in range(config.num_experts)
            ])
        
        # Confidence head for each expert (with unique initialization for symmetry breaking)
        self.confidence_heads = nn.ModuleList([
            ConfidenceHead(config.hidden_dim, expert_id=i, num_experts=config.num_experts)
            for i in range(config.num_experts)
        ])
        
        # The auctioneer (with differentiable routing for confidence head training)
        self.auctioneer = VCGAuctioneer(
            config.num_experts, 
            config.top_k,
            differentiable=config.use_differentiable_routing
        )
        
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
        
        # EMA baseline loss for computing loss reduction (per expert)
        self.register_buffer(
            'expert_baseline_loss',
            torch.ones(config.num_experts)  # Start at 1.0 as neutral baseline
        )
        
        # Track expert performance (EMA of loss reduction)
        self.register_buffer(
            'expert_performance_ema',
            torch.zeros(config.num_experts)
        )
        
        # Last forward pass statistics (for monitoring)
        self.last_stats = {}
        
        # Wealth history tracking for VCG auction analysis
        # Stores wealth snapshots at each forward pass for visualization
        self.wealth_history: List[List[float]] = []
        self._track_wealth: bool = False  # Enable via start_tracking()
        
        # Caching for loss feedback mechanism
        # These are non-persistent and reset each forward pass
        self._cached_selected_experts: Optional[torch.Tensor] = None
        self._cached_routing_weights: Optional[torch.Tensor] = None
        self._cached_confidences: Optional[torch.Tensor] = None
        self._cached_payments: Optional[torch.Tensor] = None
        self._cached_expert_token_masks: Optional[List[torch.Tensor]] = None
        self._loss_feedback_pending: bool = False  # Flag to check if loss feedback was provided
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        update_wealth: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the MoB layer.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            update_wealth: Whether to update expert wealth (False during eval)
            
        Returns:
            output: Processed tensor (batch, seq_len, hidden_dim)
            
        Note:
            Routing statistics are stored in self.last_stats for monitoring.
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
        # Two modes:
        # - TRAINING: Dense computation (all tokens through all experts, then mask)
        #   Required for gradient checkpointing compatibility
        # - INFERENCE: Sparse computation (only selected tokens through expert)
        #   Much faster as we skip non-selected expert computations
        
        output = torch.zeros_like(hidden_states)
        
        if self.training:
            # ========== TRAINING MODE: Dense computation ==========
            # Process all tokens through each expert, mask by routing
            # This ensures fixed tensor shapes for gradient checkpointing
            flat_hidden = hidden_states.view(-1, hidden_dim)  # (batch * seq, hidden)
            flat_output = output.view(-1, hidden_dim)
            
            for k in range(self.config.top_k):
                expert_indices = selected_experts[:, :, k]  # (batch, seq_len)
                weights = routing_weights[:, :, k]  # (batch, seq_len)
                flat_expert_indices = expert_indices.view(-1)  # (batch * seq,)
                flat_weights = weights.view(-1)  # (batch * seq,)
                
                for expert_idx in range(self.config.num_experts):
                    expert_mask = (flat_expert_indices == expert_idx).float()
                    
                    if expert_mask.sum() == 0:
                        # Maintain computation graph for unused experts
                        if self.use_shared_base:
                            dummy = self.experts[expert_idx](
                                flat_hidden[:1], self.base_gate_proj,
                                self.base_up_proj, self.base_down_proj
                            )
                        else:
                            dummy = self.experts[expert_idx](flat_hidden[:1])
                        flat_output = flat_output + 0.0 * dummy[:1].sum() * expert_mask[:1].unsqueeze(-1)
                        continue
                    
                    # Dense: process ALL tokens
                    if self.use_shared_base:
                        all_expert_output = self.experts[expert_idx](
                            flat_hidden, self.base_gate_proj,
                            self.base_up_proj, self.base_down_proj
                        )
                    else:
                        all_expert_output = self.experts[expert_idx](flat_hidden)
                    
                    combined_weight = expert_mask * flat_weights
                    weighted_output = all_expert_output * combined_weight.unsqueeze(-1)
                    flat_output = flat_output + weighted_output
                    
                    if update_wealth:
                        self.expert_usage_count[expert_idx] += expert_mask.sum()
            
            output = flat_output.view(batch_size, seq_len, hidden_dim)
        
        else:
            # ========== INFERENCE MODE: Sparse computation ==========
            # Only process selected tokens through their assigned experts
            # Much faster: O(top_k * tokens) vs O(num_experts * tokens)
            for k in range(self.config.top_k):
                expert_indices = selected_experts[:, :, k]  # (batch, seq_len)
                weights = routing_weights[:, :, k:k+1]  # (batch, seq_len, 1)
                
                for expert_idx in range(self.config.num_experts):
                    mask = (expert_indices == expert_idx)
                    if not mask.any():
                        continue
                    
                    # Sparse: only selected tokens
                    expert_input = hidden_states[mask]  # (num_tokens, hidden_dim)
                    
                    if self.use_shared_base:
                        expert_output = self.experts[expert_idx](
                            expert_input, self.base_gate_proj,
                            self.base_up_proj, self.base_down_proj
                        )
                    else:
                        expert_output = self.experts[expert_idx](expert_input)
                    
                    # Weight and scatter back
                    weight_vals = weights.squeeze(-1)[mask]
                    weighted_expert_output = expert_output * weight_vals.unsqueeze(-1)
                    
                    # Use index_add_ for safe in-place update
                    mask_indices = mask.nonzero(as_tuple=False)
                    flat_indices = mask_indices[:, 0] * seq_len + mask_indices[:, 1]
                    output_flat = output.view(-1, hidden_dim)
                    output_flat.index_add_(0, flat_indices, weighted_expert_output)
                    
                    if update_wealth:
                        self.expert_usage_count[expert_idx] += mask.sum().float()
        
        # 4. Cache routing information for loss feedback
        # This enables update_wealth_from_loss() to be called after loss computation
        if self.training and self.config.use_loss_feedback:
            self._cached_selected_experts = selected_experts.detach()
            self._cached_routing_weights = routing_weights.detach()
            self._cached_confidences = confidences.detach()
            self._cached_payments = payments.detach() if payments is not None else None
            self._loss_feedback_pending = True
        
        # 5. Apply local quality signal if loss feedback not available
        # This provides a fallback specialization signal
        if update_wealth and self.training:
            if self.config.use_local_quality:
                # Compute local quality from output stability
                self._update_wealth_local_quality(
                    selected_experts, routing_weights, confidences, payments, output
                )
            else:
                # Fallback to participation-only updates (less effective)
                self._update_wealth_participation(
                    selected_experts, routing_weights, confidences, payments
                )
        
        # Store statistics for monitoring (accessible via self.last_stats)
        self.last_stats = {
            'confidences': confidences.detach(),
            'selected_experts': selected_experts.detach(),
            'routing_weights': routing_weights.detach(),
            'expert_wealth': self.expert_wealth.clone(),
            'expert_usage': self.expert_usage_count.clone(),
            'expert_performance': self.expert_performance_ema.clone(),
        }
        
        # Record wealth snapshot for history tracking
        if self._track_wealth:
            self.wealth_history.append(self.expert_wealth.cpu().tolist())
        
        return output
    
    def get_confidence_calibration_loss(self) -> torch.Tensor:
        """
        Compute auxiliary loss for confidence head calibration.
        
        This loss encourages confidence heads to be calibrated:
        - High confidence should correlate with low loss (after loss feedback)
        - Overconfident experts get penalized
        
        Should be added to the main loss with weight config.confidence_calibration_weight.
        
        Returns:
            Scalar calibration loss tensor (0 if no loss feedback available)
        """
        if not self._loss_feedback_pending or self._cached_confidences is None:
            return torch.tensor(0.0, device=self.expert_wealth.device)
        
        # Use performance EMA as target for confidence calibration
        # Experts should be confident when they typically perform well
        target_confidence = torch.sigmoid(self.expert_performance_ema * 5.0)  # Scale to [0, 1]
        
        # MSE between predicted confidence and performance-based target
        mean_confidences = self._cached_confidences.mean(dim=(0, 1))  # (num_experts,)
        calibration_loss = F.mse_loss(mean_confidences, target_confidence.detach())
        
        return calibration_loss * self.config.confidence_calibration_weight
    
    def update_wealth_from_loss(
        self, 
        per_token_loss: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None
    ):
        """
        Update expert wealth based on actual loss reduction.
        
        THIS IS THE KEY SPECIALIZATION MECHANISM.
        
        Called by the training loop after computing the loss:
        ```
        output = model(input_ids)
        loss = criterion(output, labels)  # per-token loss
        mob_layer.update_wealth_from_loss(loss)
        ```
        
        Args:
            per_token_loss: Loss per token, shape (batch, seq_len) or (batch * seq_len,)
                           Note: For causal LM, this is typically (batch, input_seq_len - 1)
                           due to next-token prediction shift.
            token_mask: Optional mask for valid tokens (1 = valid, 0 = padding)
        """
        if not self._loss_feedback_pending or self._cached_selected_experts is None:
            logger.warning("update_wealth_from_loss called without pending forward pass")
            return
        
        with torch.no_grad():
            selected_experts = self._cached_selected_experts
            routing_weights = self._cached_routing_weights
            confidences = self._cached_confidences
            payments = self._cached_payments
            
            batch_size, cached_seq_len, _ = confidences.shape
            
            # Reshape loss if needed
            if per_token_loss.dim() == 1:
                # Try to infer batch size from cached shape
                loss_seq_len = per_token_loss.numel() // batch_size
                per_token_loss = per_token_loss.view(batch_size, loss_seq_len)
            
            loss_seq_len = per_token_loss.size(1)
            
            # Handle sequence length mismatch (common in causal LM due to shift)
            # Loss is computed on tokens [1:N], routing was done on [0:N]
            # We align by taking the first loss_seq_len positions from routing
            if loss_seq_len != cached_seq_len:
                if loss_seq_len < cached_seq_len:
                    # Slice routing tensors to match loss length
                    # For causal LM: loss[i] corresponds to predicting token[i+1] from token[i]
                    # So loss[i] should use routing decisions from position [i]
                    selected_experts = selected_experts[:, :loss_seq_len, :]
                    routing_weights = routing_weights[:, :loss_seq_len, :]
                    confidences = confidences[:, :loss_seq_len, :]
                    if payments is not None:
                        payments = payments[:, :loss_seq_len, :]
                else:
                    logger.warning(f"Loss seq_len ({loss_seq_len}) > cached seq_len ({cached_seq_len}), skipping wealth update")
                    self._loss_feedback_pending = False
                    return
            
            seq_len = loss_seq_len  # Use the aligned sequence length
            
            # Apply token mask if provided
            if token_mask is not None:
                if token_mask.dim() == 1:
                    token_mask = token_mask.view(batch_size, -1)
                # Align token_mask to loss sequence length
                if token_mask.size(1) > seq_len:
                    token_mask = token_mask[:, :seq_len]
                elif token_mask.size(1) < seq_len:
                    # Pad with zeros if mask is shorter (shouldn't happen)
                    pad_size = seq_len - token_mask.size(1)
                    token_mask = F.pad(token_mask, (0, pad_size), value=0)
                per_token_loss = per_token_loss * token_mask
            
            # 1. Mild decay to all experts
            self.expert_wealth *= self.config.wealth_decay
            
            # 2. Compute loss reduction per expert
            expert_rewards = torch.zeros_like(self.expert_wealth)
            expert_token_counts = torch.zeros_like(self.expert_wealth)
            
            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
                    if not mask.any():
                        continue
                    
                    # Get losses for tokens this expert processed
                    expert_losses = per_token_loss[mask]
                    mean_loss = expert_losses.mean()
                    token_count = mask.sum().float()
                    expert_token_counts[expert_idx] += token_count
                    
                    # Compare to baseline (EMA of past losses for this expert)
                    baseline = self.expert_baseline_loss[expert_idx]
                    
                    # Loss reduction: positive = expert did better than baseline
                    loss_reduction = baseline - mean_loss
                    
                    # Reward proportional to loss reduction and selection weight
                    mean_weight = routing_weights[:, :, k][mask].mean()
                    reward = loss_reduction * mean_weight * token_count / (batch_size * seq_len)
                    
                    # Scale reward
                    expert_rewards[expert_idx] += reward * self.config.reward_scale * 50.0
                    
                    # Update baseline EMA
                    self.expert_baseline_loss[expert_idx] = (
                        self.config.loss_ema_decay * baseline + 
                        (1 - self.config.loss_ema_decay) * mean_loss
                    )
                    
                    # Update performance EMA (for confidence calibration)
                    self.expert_performance_ema[expert_idx] = (
                        self.config.loss_ema_decay * self.expert_performance_ema[expert_idx] +
                        (1 - self.config.loss_ema_decay) * loss_reduction
                    )
            
            # 3. Competitive bonus for top performers
            if expert_rewards.abs().max() > 1e-6:
                normalized_rewards = (expert_rewards - expert_rewards.mean()) / (expert_rewards.std() + 1e-6)
                competitive_bonus = F.relu(normalized_rewards) * expert_rewards.abs().mean() * 0.5
                expert_rewards += competitive_bonus
            
            # 4. Apply VCG payments (wealth transfer)
            if self.config.use_vcg_payments and payments is not None:
                for k in range(self.config.top_k):
                    for expert_idx in range(self.config.num_experts):
                        mask = (selected_experts[:, :, k] == expert_idx)
                        if mask.any():
                            mean_payment = payments[:, :, k][mask].mean()
                            # Payment proportional to bid, reduces reward
                            payment_fraction = mean_payment / (self.expert_wealth[expert_idx] + 1e-6)
                            expert_rewards[expert_idx] *= (1.0 - payment_fraction.clamp(0, 0.3))
            
            # 5. Apply rewards and clamp
            self.expert_wealth += expert_rewards
            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)
            
            # Clear cache
            self._loss_feedback_pending = False
    
    def _update_wealth_local_quality(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        confidences: torch.Tensor,
        payments: Optional[torch.Tensor],
        output: torch.Tensor
    ):
        """
        Update wealth using local quality signals (when external loss unavailable).
        
        Uses output stability as a proxy for quality:
        - Experts that produce consistent output magnitudes are rewarded
        - Experts with high variance in output norms are penalized
        
        This is less effective than loss feedback but works standalone.
        """
        with torch.no_grad():
            batch_size, seq_len, hidden_dim = output.shape
            num_tokens = batch_size * seq_len
            
            # 1. Mild decay
            self.expert_wealth *= self.config.wealth_decay
            
            # 2. Compute output quality per expert
            expert_rewards = torch.zeros_like(self.expert_wealth)
            output_norms = output.norm(dim=-1)  # (batch, seq_len)
            global_mean_norm = output_norms.mean()
            
            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
                    if not mask.any():
                        continue
                    
                    # Get output norms for this expert's tokens
                    expert_output_norms = output_norms[mask]
                    
                    # Quality signal 1: consistency (low variance is good)
                    norm_std = expert_output_norms.std()
                    consistency_reward = 1.0 / (1.0 + norm_std)
                    
                    # Quality signal 2: appropriate magnitude (close to global mean)
                    norm_mean = expert_output_norms.mean()
                    magnitude_diff = (norm_mean - global_mean_norm).abs()
                    magnitude_reward = 1.0 / (1.0 + magnitude_diff)
                    
                    # Combined quality signal
                    quality = (consistency_reward + magnitude_reward) / 2.0
                    
                    # Weight by confidence and selection fraction
                    mean_confidence = confidences[:, :, expert_idx][mask].mean()
                    mean_weight = routing_weights[:, :, k][mask].mean()
                    selection_fraction = mask.sum().float() / num_tokens
                    
                    # Reward = quality * confidence * weight * fraction
                    reward = quality * mean_confidence * mean_weight * selection_fraction
                    expert_rewards[expert_idx] += reward * self.config.reward_scale * 5.0
            
            # 3. Competitive bonus
            mean_reward = expert_rewards.mean()
            if mean_reward > 0:
                competitive_bonus = (expert_rewards - mean_reward) * 0.5
                expert_rewards += competitive_bonus.clamp(min=0)
            
            # 4. VCG payments
            if self.config.use_vcg_payments and payments is not None:
                for k in range(self.config.top_k):
                    for expert_idx in range(self.config.num_experts):
                        mask = (selected_experts[:, :, k] == expert_idx)
                        if mask.any():
                            mean_payment = payments[:, :, k][mask].mean()
                            payment_cost = mean_payment * 0.1 / (self.expert_wealth[expert_idx] + 1e-6)
                            expert_rewards[expert_idx] *= (1.0 - payment_cost.clamp(0, 0.5))
            
            # 5. Apply rewards
            self.expert_wealth += expert_rewards
            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)
    
    def _update_wealth_participation(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        confidences: torch.Tensor,
        payments: Optional[torch.Tensor] = None
    ):
        """
        Update expert wealth based on their participation and performance.
        
        Economics redesigned for proper differentiation:
        1. Decay is mild (0.999) - experts don't bleed out
        2. Rewards are scaled to match or exceed decay for active experts
        3. VCG payments create wealth transfer (not just accumulation)
        4. Competitive rewards: better confidence relative to peers = more reward
        
        The goal: Active, confident experts should grow wealth while
        inactive or low-confidence experts slowly decay, creating specialization.
        """
        with torch.no_grad():
            batch_size, seq_len, _ = confidences.shape
            num_tokens = batch_size * seq_len
            
            # 1. Mild decay to all experts (prevents permanent monopolies)
            wealth_before_decay = self.expert_wealth.clone()
            self.expert_wealth *= self.config.wealth_decay
            
            # 2. Calculate participation-based rewards
            # Key insight: rewards must be competitive, not absolute
            # An expert that gets selected more AND has higher confidence earns more
            
            expert_rewards = torch.zeros_like(self.expert_wealth)
            expert_selections = torch.zeros_like(self.expert_wealth)
            
            for k in range(self.config.top_k):
                for expert_idx in range(self.config.num_experts):
                    mask = (selected_experts[:, :, k] == expert_idx)
                    if mask.any():
                        # Count selections
                        selection_count = mask.sum().float()
                        expert_selections[expert_idx] += selection_count
                        
                        # Reward = selection_fraction * mean_confidence * routing_weight
                        # This creates differentiation: confident experts on important tokens win more
                        selection_fraction = selection_count / num_tokens
                        mean_confidence = confidences[:, :, expert_idx][mask].mean()
                        mean_weight = routing_weights[:, :, k][mask].mean()
                        
                        # Scale reward to match potential decay
                        # At equilibrium: reward ≈ decay for moderately active experts
                        base_reward = selection_fraction * mean_confidence * mean_weight
                        expert_rewards[expert_idx] += base_reward * self.config.reward_scale * 10.0
            
            # 3. Competitive bonus: experts above average confidence get extra
            mean_reward = expert_rewards.mean()
            if mean_reward > 0:
                competitive_bonus = (expert_rewards - mean_reward) * 0.5
                expert_rewards += competitive_bonus.clamp(min=0)  # Only reward over-performers
            
            # 4. Apply VCG payments if enabled (wealth transfer mechanism)
            if self.config.use_vcg_payments and payments is not None:
                # Winners "pay" by losing some wealth proportional to their payment
                # This prevents unchecked accumulation and creates circulation
                for k in range(self.config.top_k):
                    for expert_idx in range(self.config.num_experts):
                        mask = (selected_experts[:, :, k] == expert_idx)
                        if mask.any():
                            mean_payment = payments[:, :, k][mask].mean()
                            # Payment is a percentage of potential reward (creates cost for winning)
                            payment_cost = mean_payment * 0.1 / (self.expert_wealth[expert_idx] + 1e-6)
                            expert_rewards[expert_idx] *= (1.0 - payment_cost.clamp(0, 0.5))
            
            # 5. Apply rewards
            self.expert_wealth += expert_rewards
            
            # 6. Clamp to valid range
            self.expert_wealth.clamp_(min=self.config.min_wealth, max=self.config.max_wealth)
    
    def start_tracking(self):
        """Enable wealth history tracking for VCG auction analysis."""
        self._track_wealth = True
        self.wealth_history = []
        
    def stop_tracking(self):
        """Disable wealth history tracking."""
        self._track_wealth = False
        
    def get_wealth_history(self) -> List[List[float]]:
        """
        Get the wealth history.
        
        Returns:
            List of wealth snapshots, each containing wealth values per expert.
            Shape: [num_forward_passes, num_experts]
        """
        return self.wealth_history.copy()
    
    def reset_tracking(self):
        """Reset the wealth history without disabling tracking."""
        self.wealth_history = []
    
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
        
        For shared base mode (memory-efficient):
        - Copy the FFN weights to a single shared base
        - Adapters start at zero (no initial change to behavior)
        - Jitter applied to adapters only
        
        For full expert mode:
        - Duplicate the existing FFN weights across experts
        - Add small random noise to break symmetry
        
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
        
        mode_str = "shared-base" if config.use_shared_base else "full-expert"
        logger.info(f"Creating MoB ({mode_str}) on device={device}, dtype={dtype}")
        
        # Create MoB on CPU first for memory efficiency, then move to device
        mob = cls(config)
        
        # Copy weights from original FFN
        with torch.no_grad():
            if config.use_shared_base:
                # Shared base mode: copy weights to shared base only
                if hasattr(ffn_module, 'gate_proj'):
                    mob.base_gate_proj.weight.copy_(ffn_module.gate_proj.weight.cpu())
                    mob.base_up_proj.weight.copy_(ffn_module.up_proj.weight.cpu())
                    mob.base_down_proj.weight.copy_(ffn_module.down_proj.weight.cpu())
                
                # Adapters are already initialized (A: kaiming, B: zeros)
                # Add small jitter to A matrices to break symmetry between experts
                for i, expert in enumerate(mob.experts):
                    expert.gate_adapter_A.weight.add_(
                        torch.randn_like(expert.gate_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
                    expert.up_adapter_A.weight.add_(
                        torch.randn_like(expert.up_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
                    expert.down_adapter_A.weight.add_(
                        torch.randn_like(expert.down_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
            else:
                # Full expert mode: copy to each expert with jitter
                for expert in mob.experts:
                    if hasattr(ffn_module, 'gate_proj'):
                        expert.gate_proj.weight.copy_(ffn_module.gate_proj.weight.cpu())
                        expert.up_proj.weight.copy_(ffn_module.up_proj.weight.cpu())
                        expert.down_proj.weight.copy_(ffn_module.down_proj.weight.cpu())
                        
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
        
        # Move to target device/dtype
        mob = mob.to(device=device, dtype=dtype)
        
        logger.info(
            f"Upcycled FFN to MoB with {config.num_experts} experts, "
            f"top-k={config.top_k}, mode={mode_str}"
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
        hidden_states = self.mob(hidden_states)
        hidden_states = residual + hidden_states
        
        # Get routing statistics from MoB
        mob_stats = self.mob.last_stats
        
        return hidden_states, mob_stats
    
    def get_calibration_loss(self) -> torch.Tensor:
        """Get confidence calibration loss from MoB layer."""
        return self.mob.get_confidence_calibration_loss()
    
    def update_from_loss(self, per_token_loss: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        """Pass loss feedback to MoB layer for wealth updates."""
        self.mob.update_wealth_from_loss(per_token_loss, token_mask)


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


def get_mob_layers(model: nn.Module) -> List[MixtureOfBidders]:
    """
    Find all MoB layers in a model.
    
    Args:
        model: Model that may contain MoB layers
        
    Returns:
        List of MixtureOfBidders modules
    """
    mob_layers = []
    for module in model.modules():
        if isinstance(module, MixtureOfBidders):
            mob_layers.append(module)
    return mob_layers


def update_all_mob_from_loss(
    model: nn.Module,
    per_token_loss: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None
):
    """
    Update all MoB layers in a model with loss feedback.
    
    Call this after computing the loss in your training loop:
    
    ```python
    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits
    
    # Compute per-token loss (don't reduce yet)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    per_token_loss = per_token_loss.view(batch_size, seq_len)
    
    # Update MoB wealth from loss feedback
    update_all_mob_from_loss(model, per_token_loss, attention_mask)
    
    # Now reduce for backprop
    loss = per_token_loss.mean()
    loss.backward()
    ```
    
    Args:
        model: Model containing MoB layers
        per_token_loss: Loss per token, shape (batch, seq_len)
        token_mask: Optional mask for valid tokens
    """
    for mob in get_mob_layers(model):
        mob.update_wealth_from_loss(per_token_loss, token_mask)


def get_total_calibration_loss(model: nn.Module) -> torch.Tensor:
    """
    Sum calibration losses from all MoB layers.
    
    Add this to your main loss for confidence head training:
    
    ```python
    main_loss = criterion(outputs, labels)
    calibration_loss = get_total_calibration_loss(model)
    total_loss = main_loss + calibration_loss
    total_loss.backward()
    ```
    
    Args:
        model: Model containing MoB layers
        
    Returns:
        Sum of calibration losses from all MoB layers
    """
    total_loss = torch.tensor(0.0)
    for mob in get_mob_layers(model):
        cal_loss = mob.get_confidence_calibration_loss()
        if cal_loss.device != total_loss.device:
            total_loss = total_loss.to(cal_loss.device)
        total_loss = total_loss + cal_loss
    return total_loss


def get_mob_statistics(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Aggregate statistics from all MoB layers for monitoring.
    
    Args:
        model: Model containing MoB layers
        
    Returns:
        Dictionary with aggregated statistics:
        - 'mean_wealth': Mean wealth across all experts and layers
        - 'wealth_gini': Gini coefficient of wealth distribution
        - 'mean_performance': Mean performance EMA
        - 'layer_wealth': List of per-layer wealth tensors
    """
    mob_layers = get_mob_layers(model)
    if not mob_layers:
        return {}
    
    all_wealth = torch.stack([mob.expert_wealth for mob in mob_layers])
    all_performance = torch.stack([mob.expert_performance_ema for mob in mob_layers])
    
    # Compute Gini coefficient
    flat_wealth = all_wealth.flatten()
    sorted_wealth = torch.sort(flat_wealth)[0]
    n = len(sorted_wealth)
    cumsum = torch.cumsum(sorted_wealth, dim=0)
    gini = (2 * torch.sum((torch.arange(1, n+1, device=flat_wealth.device) * sorted_wealth))) / (n * torch.sum(sorted_wealth)) - (n + 1) / n
    
    return {
        'mean_wealth': all_wealth.mean(),
        'wealth_std': all_wealth.std(),
        'wealth_gini': gini.abs(),
        'mean_performance': all_performance.mean(),
        'layer_wealth': [mob.expert_wealth.clone() for mob in mob_layers],
        'layer_performance': [mob.expert_performance_ema.clone() for mob in mob_layers],
    }


def load_mob_state(
    model: nn.Module, 
    state_path: str,
    strict: bool = False
) -> int:
    """
    Load trained MoB state (wealth, performance EMA) into a model.
    
    Use this to restore expert specialization from a trained checkpoint
    when starting the inference server.
    
    Args:
        model: Model containing MoB layers
        state_path: Path to mob_state.pt file
        strict: If True, raise error on config mismatch
        
    Returns:
        Number of layers successfully loaded
        
    Raises:
        ValueError: If strict=True and config mismatch detected
        
    Example:
        # In main.py after apply_mob_to_model():
        from mob import load_mob_state
        loaded = load_mob_state(model, "./tame_inference/mob_state.pt")
        print(f"Loaded MoB state for {loaded} layers")
    """
    import logging
    logger = logging.getLogger(__name__)
    
    mob_state = torch.load(state_path, map_location="cpu")
    mob_layers = get_mob_layers(model)
    
    if not mob_layers:
        logger.warning("No MoB layers found in model")
        return 0
    
    # Validate config if present
    saved_config = mob_state.get("_config", {})
    if saved_config:
        # Check num_experts
        saved_experts = saved_config.get("num_experts")
        if saved_experts and mob_layers:
            model_experts = mob_layers[0].config.num_experts
            if saved_experts != model_experts:
                msg = (f"Expert count mismatch: trained with {saved_experts} experts, "
                       f"but model has {model_experts} experts")
                if strict:
                    raise ValueError(msg)
                logger.error(f"CONFIG MISMATCH: {msg}")
                logger.error("Wealth state will NOT be loaded. "
                            "Experts will start with default wealth.")
                return 0
        
        # Check number of layers
        saved_layers = saved_config.get("num_layers")
        if saved_layers and saved_layers != len(mob_layers):
            msg = (f"Layer count mismatch: trained {saved_layers} MoB layers, "
                   f"but model has {len(mob_layers)} MoB layers")
            if strict:
                raise ValueError(msg)
            logger.warning(f"CONFIG MISMATCH: {msg}")
            logger.warning("Will load state for available layers only.")
        
        # Log training config for reference
        logger.info(f"Loading state from: {state_path}")
        logger.info(f"  Trained with: {saved_experts} experts, {saved_layers} layers")
        if "top_k" in saved_config:
            logger.info(f"  top_k: {saved_config['top_k']}")
    else:
        logger.warning("No config metadata in mob_state.pt - cannot validate compatibility")
    
    loaded = 0
    skipped = 0
    for idx, mob in enumerate(mob_layers):
        key = f"layer_{idx}"
        if key not in mob_state:
            if strict:
                raise ValueError(f"Missing state for {key}")
            skipped += 1
            continue
        
        state = mob_state[key]
        device = mob.expert_wealth.device
        
        # Load wealth
        if "wealth" in state:
            wealth = torch.tensor(state["wealth"], device=device, dtype=mob.expert_wealth.dtype)
            if wealth.shape == mob.expert_wealth.shape:
                mob.expert_wealth.copy_(wealth)
            else:
                logger.warning(f"{key}: wealth shape mismatch "
                              f"(saved: {wealth.shape}, model: {mob.expert_wealth.shape}), skipping")
                skipped += 1
                continue
        
        # Load performance EMA
        if "performance_ema" in state:
            perf = torch.tensor(state["performance_ema"], device=device, dtype=mob.expert_performance_ema.dtype)
            if perf.shape == mob.expert_performance_ema.shape:
                mob.expert_performance_ema.copy_(perf)
        
        # Load baseline loss
        if "baseline_loss" in state:
            baseline = torch.tensor(state["baseline_loss"], device=device, dtype=mob.expert_baseline_loss.dtype)
            if baseline.shape == mob.expert_baseline_loss.shape:
                mob.expert_baseline_loss.copy_(baseline)
        
        # Load usage count
        if "usage_count" in state:
            usage = torch.tensor(state["usage_count"], device=device, dtype=mob.expert_usage_count.dtype)
            if usage.shape == mob.expert_usage_count.shape:
                mob.expert_usage_count.copy_(usage)
        
        loaded += 1
    
    if loaded > 0:
        logger.info(f"✓ Loaded MoB state for {loaded}/{len(mob_layers)} layers from {state_path}")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} layers due to missing/mismatched state")
    
    return loaded


def save_mob_state(model: nn.Module, save_path: str) -> bool:
    """
    Save MoB state (wealth, performance EMA) to file.
    
    Includes configuration metadata to validate compatibility on load.
    
    Args:
        model: Model containing MoB layers
        save_path: Path to save mob_state.pt
        
    Returns:
        True if successful
    """
    mob_layers = get_mob_layers(model)
    if not mob_layers:
        return False
    
    # Save config metadata for validation on load
    first_mob = mob_layers[0]
    mob_state = {
        "_config": {
            "num_experts": first_mob.config.num_experts,
            "top_k": first_mob.config.top_k,
            "num_layers": len(mob_layers),
            "hidden_dim": first_mob.config.hidden_dim,
            "adapter_rank": first_mob.config.adapter_rank if first_mob.config.use_shared_base else None,
        }
    }
    
    for idx, mob in enumerate(mob_layers):
        mob_state[f"layer_{idx}"] = {
            "wealth": mob.expert_wealth.cpu().tolist(),
            "performance_ema": mob.expert_performance_ema.cpu().tolist(),
            "baseline_loss": mob.expert_baseline_loss.cpu().tolist(),
            "usage_count": mob.expert_usage_count.cpu().tolist(),
        }
    
    torch.save(mob_state, save_path)
    return True
