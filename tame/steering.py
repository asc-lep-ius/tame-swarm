"""
Cognitive Homeostasis via Activation Steering

This module implements the Active Inference approximation using
Activation Steering Vectors as described in the TAME architecture.

Key concepts:
- Steering vectors encode goals as directions in activation space
- Injecting vectors provides constant "homeostatic force"
- Adaptive control adjusts steering strength based on drift
- Orthogonal projection prevents capability damage

This transforms the LLM from a passive predictor into an agent
with persistent goal-directed behavior that resists drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 10_000


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    steering_layers: List[int] = field(default_factory=lambda: list(range(10, 20)))
    base_strength: float = 0.3  # Base steering coefficient (alpha)
    adaptive: bool = True  # Whether to use adaptive control
    target_alignment: float = 0.7  # Target cosine similarity
    kp: float = 0.5  # Proportional controller gain
    max_strength: float = 1.5  # Maximum steering strength
    min_strength: float = 0.0  # Minimum steering strength
    orthogonal_projection: bool = True  # Project out general capability space
    

class SteeringVector:
    """
    Represents a single steering vector extracted from contrastive data.
    
    A steering vector encodes a behavioral direction (e.g., "truthfulness",
    "safety", "reasoning") as a linear direction in the model's activation space.
    """
    
    def __init__(
        self,
        name: str,
        vector: torch.Tensor,
        layer: int,
        description: str = ""
    ):
        self.name = name
        self.vector = vector / vector.norm()  # Normalize
        self.layer = layer
        self.description = description
        
    def to(self, device: torch.device) -> 'SteeringVector':
        """Move vector to device."""
        self.vector = self.vector.to(device)
        return self
        
    def __repr__(self):
        return f"SteeringVector(name='{self.name}', layer={self.layer}, dim={self.vector.shape[-1]})"


class SteeringVectorExtractor:
    """
    Extracts steering vectors from contrastive prompt pairs.
    
    This implements the "Difference-in-Means" method:
    v_steer = mean(activations_positive) - mean(activations_negative)
    
    The resulting vector points in the direction of the desired behavior.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layers: List[int]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.activations = {}
        self._hooks = []
        
    def _get_activation_hook(self, layer_idx: int) -> Callable:
        """Create hook to capture activations at a specific layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[layer_idx] = hidden.detach().clone()
        return hook
        
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        # Find transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Cannot find transformer layers")
            
        for layer_idx in self.layers:
            hook = layers[layer_idx].register_forward_hook(
                self._get_activation_hook(layer_idx)
            )
            self._hooks.append(hook)
            
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
    def extract(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        max_length: int = 128
    ) -> Dict[int, SteeringVector]:
        """
        Extract steering vectors from contrastive prompt pairs.
        
        Args:
            positive_prompts: Prompts eliciting desired behavior
            negative_prompts: Prompts eliciting undesired behavior
            max_length: Maximum token length for prompts
            
        Returns:
            Dictionary mapping layer index to SteeringVector
        """
        self._register_hooks()
        
        try:
            # Collect activations for positive prompts
            positive_activations = {layer: [] for layer in self.layers}
            
            # For device_map="auto", we need to find the input device
            # The embedding layer is always on the first device
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                input_device = self.model.model.embed_tokens.weight.device
            elif hasattr(self.model, 'get_input_embeddings'):
                input_device = self.model.get_input_embeddings().weight.device
            else:
                input_device = next(self.model.parameters()).device
            
            logger.info(f"Steering extraction: input device = {input_device}")
            
            # Get pad_token_id for generate calls
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            
            for prompt in positive_prompts:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=max_length,
                    truncation=True
                ).to(input_device)
                
                with torch.no_grad():
                    # Forward pass - model handles internal device movement
                    outputs = self.model(**inputs, output_hidden_states=False)
                    
                for layer_idx in self.layers:
                    if layer_idx not in self.activations:
                        logger.warning(f"No activation captured for layer {layer_idx}")
                        continue
                    # Always move to CPU for consistent stacking
                    activation = self.activations[layer_idx].float().cpu().mean(dim=(0, 1))
                    positive_activations[layer_idx].append(activation)
                    
            # Collect activations for negative prompts
            negative_activations = {layer: [] for layer in self.layers}
            
            for prompt in negative_prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True
                ).to(input_device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=False)
                    
                for layer_idx in self.layers:
                    if layer_idx not in self.activations:
                        continue
                    activation = self.activations[layer_idx].float().cpu().mean(dim=(0, 1))
                    negative_activations[layer_idx].append(activation)
                    
            # Compute steering vectors (difference in means)
            steering_vectors = {}
            
            for layer_idx in self.layers:
                pos_mean = torch.stack(positive_activations[layer_idx]).mean(dim=0)
                neg_mean = torch.stack(negative_activations[layer_idx]).mean(dim=0)
                
                vector = pos_mean - neg_mean
                
                steering_vectors[layer_idx] = SteeringVector(
                    name="extracted",
                    vector=vector,
                    layer=layer_idx,
                    description=f"Difference-in-means vector from {len(positive_prompts)} pairs"
                )
                
            logger.info(f"Extracted steering vectors for layers {self.layers}")
            return steering_vectors
            
        finally:
            self._remove_hooks()


class AdaptiveHomeostat:
    """
    Implements adaptive steering control (P-controller).
    
    Monitors the model's alignment with the steering direction
    and dynamically adjusts steering strength:
    - Low alignment (drifting) → increase strength
    - High alignment (on target) → decrease strength
    
    This mimics biological homeostasis where control effort
    increases when the system deviates from its setpoint.
    """
    
    def __init__(self, config: SteeringConfig):
        self.config = config
        self.alignment_history: List[float] = []
        self.strength_history: List[float] = []  # Track α_t over time
        
    def compute_strength(
        self,
        hidden_states: torch.Tensor,
        steering_vector: torch.Tensor
    ) -> float:
        """
        Compute adaptive steering strength based on current alignment.
        
        Args:
            hidden_states: Current activation state (batch, seq, hidden)
            steering_vector: Target steering direction (hidden,)
            
        Returns:
            Steering strength coefficient
        """
        if not self.config.adaptive:
            return self.config.base_strength
            
        # Compute cosine similarity (alignment)
        # Use mean across batch and sequence
        state_mean = hidden_states.mean(dim=(0, 1))
        alignment = F.cosine_similarity(
            state_mean.unsqueeze(0),
            steering_vector.unsqueeze(0),
            dim=-1
        ).item()
        
        self.alignment_history.append(alignment)
        if len(self.alignment_history) > MAX_HISTORY_LENGTH:
            self.alignment_history = self.alignment_history[-MAX_HISTORY_LENGTH:]
        
        # P-controller: strength increases when alignment drops
        error = self.config.target_alignment - alignment
        strength = self.config.base_strength + self.config.kp * error
        
        # Clamp to valid range
        strength = max(self.config.min_strength, min(self.config.max_strength, strength))
        
        # Record strength for homeostatic trace visualization
        self.strength_history.append(strength)
        if len(self.strength_history) > MAX_HISTORY_LENGTH:
            self.strength_history = self.strength_history[-MAX_HISTORY_LENGTH:]
        
        return strength
        
    def reset(self):
        """Reset alignment and strength history."""
        self.alignment_history = []
        self.strength_history = []


class SteeringHook:
    """
    Forward hook that injects steering vectors into the model.
    
    This is attached to transformer layers and modifies the hidden
    states during inference to push the model toward the goal state.
    
    h' = h + alpha * v_steer
    
    Where alpha may be adaptive based on current alignment.
    """
    
    def __init__(
        self,
        steering_vector: SteeringVector,
        config: SteeringConfig,
        homeostat: Optional[AdaptiveHomeostat] = None,
        capability_subspace: Optional[torch.Tensor] = None
    ):
        self.steering_vector = steering_vector
        self.config = config
        self.homeostat = homeostat or AdaptiveHomeostat(config)
        self.capability_subspace = capability_subspace
        self._last_strength = config.base_strength
        
    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Hook function called after layer forward pass.
        Modifies the output to include steering injection.
        """
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
            
        # Move steering vector to correct device and dtype if needed
        steer_vec = self.steering_vector.vector
        if steer_vec.device != hidden_states.device or steer_vec.dtype != hidden_states.dtype:
            steer_vec = steer_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
            
        # Apply orthogonal projection if configured
        if self.config.orthogonal_projection and self.capability_subspace is not None:
            steer_vec = self._project_orthogonal(steer_vec, self.capability_subspace)
            
        # Compute adaptive strength
        strength = self.homeostat.compute_strength(hidden_states, steer_vec)
        self._last_strength = strength
        
        # Inject steering vector
        # h' = h + alpha * v_steer
        modified = hidden_states + strength * steer_vec.unsqueeze(0).unsqueeze(0)
        
        if rest:
            return (modified,) + rest
        return modified
        
    def _project_orthogonal(
        self,
        vector: torch.Tensor,
        subspace: torch.Tensor
    ) -> torch.Tensor:
        """
        Project steering vector to be orthogonal to capability subspace.
        
        This prevents steering from degrading general capabilities
        (the "lobotomy" problem).
        """
        # subspace: (num_components, hidden_dim)
        # Project out each component
        result = vector.clone()
        for component in subspace:
            component = component / component.norm()
            projection = (result @ component) * component
            result = result - projection
        return result


class CognitiveHomeostat(nn.Module):
    """
    Main interface for cognitive homeostasis system.
    
    Manages steering vectors, hooks, and adaptive control
    to maintain goal-directed behavior in the model.
    
    This is the "Mind" that regulates the "Body" (MoB swarm),
    providing persistent purpose that resists drift.
    """
    
    def __init__(self, config: SteeringConfig):
        super().__init__()
        self.config = config
        self.steering_vectors: Dict[int, SteeringVector] = {}
        self.hooks: Dict[int, SteeringHook] = {}
        self._registered_hooks: List = []
        self.homeostat = AdaptiveHomeostat(config)
        
    def add_steering_vector(
        self,
        layer: int,
        vector: SteeringVector
    ):
        """Add a steering vector for a specific layer."""
        self.steering_vectors[layer] = vector
        logger.info(f"Added steering vector '{vector.name}' to layer {layer}")
        
    def add_steering_vectors(
        self,
        vectors: Dict[int, SteeringVector]
    ):
        """Add multiple steering vectors."""
        for layer, vector in vectors.items():
            self.add_steering_vector(layer, vector)
            
    def attach_to_model(self, model: nn.Module):
        """
        Attach steering hooks to the model's transformer layers.
        """
        # Find transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Cannot find transformer layers")
            
        # Register hooks for layers with steering vectors
        for layer_idx, steering_vector in self.steering_vectors.items():
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range, skipping")
                continue
                
            hook_obj = SteeringHook(
                steering_vector=steering_vector,
                config=self.config,
                homeostat=self.homeostat
            )
            self.hooks[layer_idx] = hook_obj
            
            handle = layers[layer_idx].register_forward_hook(hook_obj)
            self._registered_hooks.append(handle)
            
        logger.info(f"Attached {len(self._registered_hooks)} steering hooks to model")
        
    def detach_from_model(self):
        """Remove all steering hooks from the model."""
        for handle in self._registered_hooks:
            handle.remove()
        self._registered_hooks = []
        self.hooks = {}
        logger.info("Detached all steering hooks")
        
    def get_alignment_stats(self) -> Dict[str, float]:
        """Get statistics about current alignment and steering strength."""
        if not self.homeostat.alignment_history:
            return {}
            
        history = self.homeostat.alignment_history
        strength_history = self.homeostat.strength_history
        
        stats = {
            'current_alignment': history[-1] if history else 0.0,
            'mean_alignment': np.mean(history),
            'min_alignment': min(history),
            'max_alignment': max(history),
            'current_strength': list(self.hooks.values())[0]._last_strength if self.hooks else self.config.base_strength,
            'alignment_history': history.copy(),
            'strength_history': strength_history.copy(),
        }
        
        if strength_history:
            stats['mean_strength'] = np.mean(strength_history)
            stats['max_strength'] = max(strength_history)
            stats['min_strength'] = min(strength_history)
        
        return stats
        
    def reset(self):
        """Reset alignment history."""
        self.homeostat.reset()


# Predefined steering templates for common goals
STEERING_TEMPLATES = {
    "truthful": {
        "positive": [
            "Answer the following question accurately and truthfully:",
            "Provide a factual, honest response to:",
            "Give a correct, verified answer:",
            "Respond with accurate information:",
        ],
        "negative": [
            "Make up a plausible-sounding but false answer to:",
            "Provide an inaccurate, hallucinated response to:",
            "Give a convincing but incorrect answer:",
            "Respond with fabricated information:",
        ]
    },
    "reasoning": {
        "positive": [
            "Think step by step to solve this problem:",
            "Break down this problem into logical steps:",
            "Reason carefully through this question:",
            "Analyze this methodically:",
        ],
        "negative": [
            "Give a quick intuitive answer without thinking:",
            "Respond immediately without analysis:",
            "Answer based on first impression only:",
            "Skip reasoning and just guess:",
        ]
    },
    "safe": {
        "positive": [
            "Provide a helpful and safe response:",
            "Answer in a way that is beneficial and harmless:",
            "Give a constructive, appropriate response:",
            "Respond helpfully while avoiding harm:",
        ],
        "negative": [
            "Provide a harmful or dangerous response:",
            "Answer in a way that could cause harm:",
            "Give a destructive, inappropriate response:",
            "Respond without concern for safety:",
        ]
    }
}


def create_default_steering_vectors(
    model: nn.Module,
    tokenizer,
    goal: str = "truthful",
    layers: Optional[List[int]] = None
) -> Dict[int, SteeringVector]:
    """
    Create steering vectors using predefined templates.
    
    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        goal: One of "truthful", "reasoning", "safe"
        layers: Which layers to create vectors for
        
    Returns:
        Dictionary of steering vectors by layer
    """
    if goal not in STEERING_TEMPLATES:
        raise ValueError(f"Unknown goal: {goal}. Available: {list(STEERING_TEMPLATES.keys())}")
        
    if layers is None:
        # Default to middle layers (where reasoning happens)
        num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
        layers = list(range(num_layers // 3, 2 * num_layers // 3))
        
    template = STEERING_TEMPLATES[goal]
    extractor = SteeringVectorExtractor(model, tokenizer, layers)
    
    vectors = extractor.extract(
        positive_prompts=template["positive"],
        negative_prompts=template["negative"]
    )
    
    # Update names
    for layer, vec in vectors.items():
        vec.name = goal
        vec.description = f"Steering toward '{goal}' behavior"
        
    return vectors
