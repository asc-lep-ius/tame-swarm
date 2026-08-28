import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 10_000

DEFAULT_CAPABILITY_SUBSPACE_RANK = 8
# Token activations are pooled per layer before the SVD; this bounds the pool so a
# large corpus costs corpus-length forward passes rather than corpus-length memory.
MAX_CAPABILITY_TOKENS = 4096
# Below this share of its original norm a steering vector has been swallowed by the
# capability subspace, and renormalising what is left amplifies rounding noise into
# a direction the projection was supposed to remove.
MIN_RETAINED_NORM_FRACTION = 0.05


@dataclass
class SteeringConfig:
    steering_layers: list[int] = field(default_factory=lambda: list(range(10, 20)))
    base_strength: float = 0.3  # Base steering coefficient (alpha)
    adaptive: bool = True  # Whether to use adaptive control
    target_alignment: float = 0.7  # Target cosine similarity
    kp: float = 0.5  # Proportional controller gain
    max_strength: float = 1.5  # Maximum steering strength
    min_strength: float = 0.0  # Minimum steering strength
    orthogonal_projection: bool = True  # Project out general capability space
    # Number of principal components of general-task activations treated as the
    # capability subspace. Higher ranks protect more of the model's behaviour and
    # leave less of the steering vector standing; see MIN_RETAINED_NORM_FRACTION.
    capability_subspace_rank: int = DEFAULT_CAPABILITY_SUBSPACE_RANK


def project_out_subspace(vector: torch.Tensor, subspace: torch.Tensor) -> torch.Tensor:
    """Remove every component of ``vector`` that lies inside ``subspace``.

    ``subspace`` is expected to be row-orthonormal, which is what
    :func:`estimate_capability_subspace` returns, but each row is renormalised
    anyway so a hand-built or loaded basis cannot silently over-subtract.
    """
    result = vector.clone()
    for component in subspace:
        norm = component.norm()
        if norm == 0:
            continue
        unit = component / norm
        result = result - (result @ unit) * unit
    return result


def project_steering_direction(
    vector: torch.Tensor,
    subspace: torch.Tensor,
    layer: int | None = None,
    rank: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Strip capability components from a steering direction and restore its norm.

    Returns the direction to inject and the share of the original norm that
    survived the projection.

    ``vector`` is expected to be unit norm, as ``SteeringVector`` guarantees. The
    success path returns a unit vector and the fallback returns ``vector``
    unchanged, so the two agree on magnitude only under that precondition.

    ``base_strength`` is calibrated against a unit direction, so returning the
    shortened vector unnormalised would silently weaken steering in proportion to
    how much of it overlapped the capability subspace -- turning a projection into
    an uncontrolled strength change.

    When almost nothing survives, the goal direction lies inside the subspace and
    renormalising the remainder would amplify rounding noise into exactly the
    direction the projection was meant to remove. Steering unprojected is the honest
    failure mode: it is the behaviour before this feature existed, and it says so in
    the log.
    """
    original_norm = vector.norm()
    if original_norm == 0:
        return vector, 0.0

    projected = project_out_subspace(vector, subspace)
    retained = (projected.norm() / original_norm).item()

    if retained < MIN_RETAINED_NORM_FRACTION:
        logger.warning(
            "Layer %s: capability projection left %.1f%% of the steering vector; "
            "steering unprojected. Lower capability_subspace_rank (currently %s).",
            layer,
            100 * retained,
            rank,
        )
        return vector, retained

    return projected / projected.norm(), retained


class SteeringVector:
    def __init__(self, name: str, vector: torch.Tensor, layer: int, description: str = ""):
        self.name = name
        self.vector = vector / vector.norm()  # Normalize
        self.layer = layer
        self.description = description

    def to(self, device: torch.device) -> "SteeringVector":
        self.vector = self.vector.to(device)
        return self

    def __repr__(self):
        return (
            f"SteeringVector(name='{self.name}', layer={self.layer}, dim={self.vector.shape[-1]})"
        )


class SteeringVectorExtractor:
    def __init__(self, model: nn.Module, tokenizer, layers: list[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.activations = {}
        self._hooks = []

    def _get_activation_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach().clone()

        return hook

    def _register_hooks(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = cast(nn.ModuleList, getattr(self.model.model, "layers"))  # noqa: B009
        elif hasattr(self.model, "layers"):
            layers = cast(nn.ModuleList, getattr(self.model, "layers"))  # noqa: B009
        else:
            raise ValueError("Cannot find transformer layers")

        for layer_idx in self.layers:
            hook = layers[layer_idx].register_forward_hook(self._get_activation_hook(layer_idx))
            self._hooks.append(hook)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _input_device(self) -> torch.device:
        """Where token ids must land before a forward pass.

        Under ``device_map="auto"`` the layers are spread across devices, but the
        embedding always holds the first one.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            # HuggingFace model internals: embed_tokens is an nn.Embedding
            embed_tokens = getattr(self.model.model, "embed_tokens")  # noqa: B009
            return embed_tokens.weight.device
        if hasattr(self.model, "get_input_embeddings"):
            # HuggingFace PreTrainedModel method, not on nn.Module stubs
            get_embeddings = getattr(self.model, "get_input_embeddings")  # noqa: B009
            return get_embeddings().weight.device
        return next(self.model.parameters()).device

    def _forward(self, prompt: str, max_length: int, input_device: torch.device) -> None:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=max_length, truncation=True
        ).to(input_device)

        with torch.no_grad():
            self.model(**inputs, output_hidden_states=False)

    def _mean_activations(self, prompts: list[str], max_length: int) -> dict[int, list]:
        """One mean-pooled activation vector per prompt, per layer, on CPU."""
        input_device = self._input_device()
        collected: dict[int, list] = {layer: [] for layer in self.layers}

        for prompt in prompts:
            self._forward(prompt, max_length, input_device)

            for layer_idx in self.layers:
                if layer_idx not in self.activations:
                    logger.warning(f"No activation captured for layer {layer_idx}")
                    continue
                collected[layer_idx].append(
                    self.activations[layer_idx].float().cpu().mean(dim=(0, 1))
                )

        return collected

    def collect_token_activations(
        self,
        prompts: list[str],
        max_length: int = 128,
        max_tokens: int = MAX_CAPABILITY_TOKENS,
    ) -> dict[int, torch.Tensor]:
        """Per-token activations at each target layer, pooled across prompts.

        Mean pooling would collapse each prompt to a single point and leave far too
        few samples to estimate a subspace from; the capability directions are the
        ones along which individual token representations vary.
        """
        self._register_hooks()

        try:
            input_device = self._input_device()
            collected: dict[int, list] = {layer: [] for layer in self.layers}
            counts = dict.fromkeys(self.layers, 0)

            for prompt in prompts:
                if all(counts[layer] >= max_tokens for layer in self.layers):
                    break

                self._forward(prompt, max_length, input_device)

                for layer_idx in self.layers:
                    if layer_idx not in self.activations or counts[layer_idx] >= max_tokens:
                        continue
                    hidden = self.activations[layer_idx].float().cpu()
                    tokens = hidden.reshape(-1, hidden.shape[-1])
                    room = max_tokens - counts[layer_idx]
                    collected[layer_idx].append(tokens[:room])
                    counts[layer_idx] += min(room, tokens.shape[0])

            return {
                layer: torch.cat(chunks, dim=0) for layer, chunks in collected.items() if chunks
            }
        finally:
            self._remove_hooks()

    def extract(
        self, positive_prompts: list[str], negative_prompts: list[str], max_length: int = 128
    ) -> dict[int, SteeringVector]:
        self._register_hooks()

        try:
            logger.info(f"Steering extraction: input device = {self._input_device()}")

            positive_activations = self._mean_activations(positive_prompts, max_length)
            negative_activations = self._mean_activations(negative_prompts, max_length)

            steering_vectors = {}
            for layer_idx in self.layers:
                pos_mean = torch.stack(positive_activations[layer_idx]).mean(dim=0)
                neg_mean = torch.stack(negative_activations[layer_idx]).mean(dim=0)

                steering_vectors[layer_idx] = SteeringVector(
                    name="extracted",
                    vector=pos_mean - neg_mean,
                    layer=layer_idx,
                    description=f"Difference-in-means vector from {len(positive_prompts)} pairs",
                )

            logger.info(f"Extracted steering vectors for layers {self.layers}")
            return steering_vectors

        finally:
            self._remove_hooks()


class AdaptiveHomeostat:
    def __init__(self, config: SteeringConfig):
        self.config = config
        self.alignment_history: deque[float] = deque(maxlen=MAX_HISTORY_LENGTH)
        self.strength_history: deque[float] = deque(maxlen=MAX_HISTORY_LENGTH)

    def compute_strength(self, hidden_states: torch.Tensor, steering_vector: torch.Tensor) -> float:
        if not self.config.adaptive:
            return self.config.base_strength

        # Compute cosine similarity (alignment)
        # Use mean across batch and sequence
        state_mean = hidden_states.mean(dim=(0, 1))
        alignment = F.cosine_similarity(
            state_mean.unsqueeze(0), steering_vector.unsqueeze(0), dim=-1
        ).item()

        self.alignment_history.append(alignment)

        error = self.config.target_alignment - alignment
        strength = self.config.base_strength + self.config.kp * error

        # Clamp to valid range
        strength = max(self.config.min_strength, min(self.config.max_strength, strength))

        self.strength_history.append(strength)

        return strength

    def reset(self):
        self.alignment_history = deque(maxlen=MAX_HISTORY_LENGTH)
        self.strength_history = deque(maxlen=MAX_HISTORY_LENGTH)


class SteeringHook:
    def __init__(
        self,
        steering_vector: SteeringVector,
        config: SteeringConfig,
        homeostat: AdaptiveHomeostat | None = None,
        capability_subspace: torch.Tensor | None = None,
    ):
        self.steering_vector = steering_vector
        self.config = config
        self.homeostat = homeostat or AdaptiveHomeostat(config)
        self.capability_subspace = capability_subspace
        self._last_strength = config.base_strength
        self._direction_cache: torch.Tensor | None = None
        self._direction_key: tuple[torch.device, torch.dtype] | None = None

    def __call__(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()

        steer_vec = self._direction(hidden_states.device, hidden_states.dtype)

        strength = self.homeostat.compute_strength(hidden_states, steer_vec)
        self._last_strength = strength

        modified = hidden_states + strength * steer_vec.unsqueeze(0).unsqueeze(0)

        if rest:
            return (modified,) + rest
        return modified

    def _direction(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The steering direction this hook injects, projected and cached.

        The projection depends only on the vector and the subspace, so recomputing
        it per forward pass would repeat a Gram-Schmidt sweep on every token.
        """
        key = (device, dtype)
        if self._direction_key == key and self._direction_cache is not None:
            return self._direction_cache

        steer_vec = self.steering_vector.vector.to(device=device, dtype=dtype)
        if self.config.orthogonal_projection and self.capability_subspace is not None:
            steer_vec = self._project_out_capabilities(steer_vec, device, dtype)

        self._direction_cache = steer_vec
        self._direction_key = key
        return steer_vec

    def _project_out_capabilities(
        self, steer_vec: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        assert self.capability_subspace is not None
        direction, _ = project_steering_direction(
            steer_vec,
            self.capability_subspace.to(device=device, dtype=dtype),
            layer=self.steering_vector.layer,
            rank=self.config.capability_subspace_rank,
        )
        return direction


class CognitiveHomeostat(nn.Module):
    def __init__(self, config: SteeringConfig):
        super().__init__()
        self.config = config
        self.steering_vectors: dict[int, SteeringVector] = {}
        self.capability_subspaces: dict[int, torch.Tensor] = {}
        self.hooks: dict[int, SteeringHook] = {}
        self._registered_hooks: list = []
        self.homeostat = AdaptiveHomeostat(config)

    def add_steering_vector(self, layer: int, vector: SteeringVector):
        self.steering_vectors[layer] = vector
        logger.info(f"Added steering vector '{vector.name}' to layer {layer}")

    def add_steering_vectors(self, vectors: dict[int, SteeringVector]):
        for layer, vector in vectors.items():
            self.add_steering_vector(layer, vector)

    def set_capability_subspaces(self, subspaces: dict[int, torch.Tensor]) -> None:
        """Install per-layer capability bases; hooks pick them up on the next attach."""
        for layer, subspace in subspaces.items():
            if subspace.ndim != 2:
                raise ValueError(
                    f"Layer {layer}: capability subspace must be (rank, hidden_dim), "
                    f"got shape {tuple(subspace.shape)}"
                )
            steering_vector = self.steering_vectors.get(layer)
            if steering_vector is not None and subspace.shape[-1] != steering_vector.vector.numel():
                raise ValueError(
                    f"Layer {layer}: capability subspace has hidden_dim "
                    f"{subspace.shape[-1]}, steering vector has "
                    f"{steering_vector.vector.numel()}"
                )

        self.capability_subspaces = dict(subspaces)
        logger.info(f"Installed capability subspaces for layers {sorted(subspaces)}")

    def estimate_capability_subspaces(
        self,
        model: nn.Module,
        tokenizer,
        texts: list[str] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Estimate and install the capability subspace for every steered layer."""
        subspaces = estimate_capability_subspace(
            model,
            tokenizer,
            layers=sorted(self.steering_vectors) or list(self.config.steering_layers),
            texts=texts,
            rank=self.config.capability_subspace_rank,
        )
        self.set_capability_subspaces(subspaces)
        return subspaces

    def attach_to_model(self, model: nn.Module):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = cast(nn.ModuleList, getattr(model.model, "layers"))  # noqa: B009
        elif hasattr(model, "layers"):
            layers = cast(nn.ModuleList, getattr(model, "layers"))  # noqa: B009
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
                homeostat=self.homeostat,
                capability_subspace=self.capability_subspaces.get(layer_idx),
            )
            self.hooks[layer_idx] = hook_obj

            handle = layers[layer_idx].register_forward_hook(hook_obj)
            self._registered_hooks.append(handle)

        logger.info(f"Attached {len(self._registered_hooks)} steering hooks to model")

    def detach_from_model(self):
        for handle in self._registered_hooks:
            handle.remove()
        self._registered_hooks = []
        self.hooks = {}
        logger.info("Detached all steering hooks")

    def projected_direction(self, layer: int) -> tuple[torch.Tensor, float]:
        """The direction actually injected at ``layer``, and its retained norm share.

        Anything else that consumes the goal direction must read it from here rather
        than from the raw steering vector. ``SteeringCoupling`` in particular keeps
        its own copy in a buffer, so seeding it from the unprojected vector would
        leave the routing coupling steering toward a direction the residual-stream
        injection has already decided not to use.
        """
        vector = self.steering_vectors[layer].vector
        subspace = self.capability_subspaces.get(layer)
        if subspace is None or not self.config.orthogonal_projection:
            return vector, 1.0

        return project_steering_direction(
            vector,
            subspace.to(device=vector.device, dtype=vector.dtype),
            layer=layer,
            rank=self.config.capability_subspace_rank,
        )

    def get_capability_retention(self) -> dict[int, float]:
        """Share of each steering vector's norm surviving its capability projection.

        A diagnostic, not a guarantee: it reports how much of the goal direction was
        orthogonal to general-task variation, not whether capability was preserved.
        Measuring that needs a held-out benchmark.
        """
        return {
            layer: self.projected_direction(layer)[1]
            for layer in self.steering_vectors
            if layer in self.capability_subspaces
        }

    def get_alignment_stats(self) -> dict[str, Any]:
        if not self.homeostat.alignment_history:
            return {}

        history = self.homeostat.alignment_history
        strength_history = self.homeostat.strength_history

        stats = {
            "current_alignment": history[-1] if history else 0.0,
            "mean_alignment": np.mean(list(history)),
            "min_alignment": min(history),
            "max_alignment": max(history),
            "current_strength": list(self.hooks.values())[0]._last_strength
            if self.hooks
            else self.config.base_strength,
            "alignment_history": list(history),
            "strength_history": list(strength_history),
        }

        if strength_history:
            stats["mean_strength"] = np.mean(list(strength_history))
            stats["max_strength"] = max(strength_history)
            stats["min_strength"] = min(strength_history)

        return stats

    def reset(self):
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
        ],
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
        ],
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
        ],
    },
}


# A deliberately broad slice of ordinary model work: exposition, narrative,
# arithmetic, code, instruction-following, dialogue and list formatting. The
# capability subspace is the directions along which *general* activity varies, so a
# corpus narrow enough to be about one thing would project out that thing rather
# than the model's shared machinery. It is small by design -- a default that runs at
# server start without a dataset dependency -- and callers with a real corpus should
# pass one to estimate_capability_subspace.
CAPABILITY_CORPUS = [
    "The mitochondrion generates most of the chemical energy a cell needs.",
    "Rainfall in the region peaks between June and September, then tapers off.",
    "She checked the timetable twice before deciding to take the later train.",
    "The treaty was signed in 1919 and reshaped the borders of central Europe.",
    "To compute a median, sort the values and take the middle one.",
    "17 multiplied by 24 is 408, and 408 divided by 8 is 51.",
    "def normalise(values):\n    total = sum(values)\n    return [v / total for v in values]",
    "Run the migration before restarting the service, or the schema will not match.",
    "Q: What causes tides? A: The gravitational pull of the moon and the sun.",
    "First, preheat the oven. Second, combine the dry ingredients. Third, add butter.",
    "The novel is narrated by an unreliable witness to events he only half understands.",
    "Steel expands when heated, which is why bridges are built with expansion joints.",
    "Please summarise the following paragraph in one sentence, keeping the main claim.",
    "The committee could not agree, so the vote was postponed until the next session.",
    "A binary search halves the remaining range on every step, so it runs in log time.",
    "He argued the opposite last year, which makes the present position hard to read.",
    "Common symptoms include fatigue, a persistent cough, and a low-grade fever.",
    "The exhibition runs through October and is free to visit on weekday mornings.",
    "SELECT name, count(*) FROM orders GROUP BY name HAVING count(*) > 3;",
    "In practice the distinction matters less than the textbooks suggest.",
    "Water boils at a lower temperature at altitude because the air pressure is lower.",
    "The parcel should arrive on Thursday unless the depot is closed for the holiday.",
    "Translate the sentence into French, keeping the formal register of the original.",
    "Most of the cost is fixed, so volume changes barely move the unit price.",
]


def estimate_capability_subspace(
    model: nn.Module,
    tokenizer,
    layers: list[int],
    texts: list[str] | None = None,
    rank: int = DEFAULT_CAPABILITY_SUBSPACE_RANK,
    max_length: int = 128,
    max_tokens: int = MAX_CAPABILITY_TOKENS,
) -> dict[int, torch.Tensor]:
    """Estimate the general-capability subspace at each layer by PCA.

    Steering degrades a model when the injected direction overlaps the directions
    the model already uses to do ordinary work -- the "lobotomy" failure mode of
    activation steering. This estimates that overlap directly: run a general corpus
    through the model, and take the leading principal components of the per-token
    activations at each target layer.

    Activations are mean-centred first. The uncentred first component is dominated
    by the shared mean activation, which every token has and which a difference-in-
    means steering vector has already cancelled; projecting it out would remove a
    direction the steering vector does not contain.

    Returns a row-orthonormal ``(rank, hidden_dim)`` basis per layer. The rank may
    come back smaller than requested when the corpus supplies fewer tokens than
    components; a subspace is not returned at all for a layer that captured nothing.
    """
    if rank <= 0:
        raise ValueError(f"capability subspace rank must be positive, got {rank}")

    corpus = texts if texts is not None else CAPABILITY_CORPUS
    if not corpus:
        raise ValueError("capability corpus is empty")

    extractor = SteeringVectorExtractor(model, tokenizer, layers)
    activations = extractor.collect_token_activations(
        corpus, max_length=max_length, max_tokens=max_tokens
    )

    subspaces: dict[int, torch.Tensor] = {}
    for layer_idx, tokens in activations.items():
        available = min(rank, *tokens.shape)
        if available < rank:
            logger.warning(
                "Layer %d: capability subspace truncated to rank %d "
                "(corpus supplied %d token activations)",
                layer_idx,
                available,
                tokens.shape[0],
            )
        if available <= 0:
            continue

        centred = tokens - tokens.mean(dim=0, keepdim=True)
        # Right singular vectors of the centred matrix are the principal axes; the
        # full covariance is hidden_dim x hidden_dim and never worth forming.
        _, _, components = torch.linalg.svd(centred, full_matrices=False)
        subspaces[layer_idx] = components[:available].contiguous()

    logger.info(
        f"Estimated capability subspaces for layers {sorted(subspaces)} "
        f"from {len(corpus)} general-corpus passages"
    )
    return subspaces


def create_default_steering_vectors(
    model: nn.Module, tokenizer, goal: str = "truthful", layers: list[int] | None = None
) -> dict[int, SteeringVector]:
    if goal not in STEERING_TEMPLATES:
        raise ValueError(f"Unknown goal: {goal}. Available: {list(STEERING_TEMPLATES.keys())}")

    if layers is None:
        # HuggingFace models store layers in model.model.layers (ModuleList)
        if hasattr(model, "model"):
            model_layers = cast(nn.ModuleList, getattr(model.model, "layers"))  # noqa: B009
        else:
            model_layers = cast(nn.ModuleList, getattr(model, "layers"))  # noqa: B009
        num_layers = len(model_layers)
        layers = list(range(num_layers // 3, 2 * num_layers // 3))

    template = STEERING_TEMPLATES[goal]
    extractor = SteeringVectorExtractor(model, tokenizer, layers)

    vectors = extractor.extract(
        positive_prompts=template["positive"], negative_prompts=template["negative"]
    )

    # Update names
    for _layer, vec in vectors.items():
        vec.name = goal
        vec.description = f"Steering toward '{goal}' behavior"

    return vectors
