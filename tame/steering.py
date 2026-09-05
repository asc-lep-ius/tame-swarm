import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from contrastive_data import ContrastivePair

logger = logging.getLogger(__name__)

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
    """Where to inject, how hard, and how the loop that sets the strength is tuned.

    The loop's gains, setpoint and units come from a calibration measured against
    the loaded model (``homeostat.calibrate_alignment``); the fields here either
    pin a value or leave it ``None`` to be derived. ``target_alignment`` is the
    legacy cosine setpoint and is only read while no calibration exists.
    """

    steering_layers: list[int] = field(default_factory=lambda: list(range(10, 20)))
    base_strength: float = 0.3  # Reference injection strength; the loop's bias term
    adaptive: bool = True  # Whether the loop sets the strength, or base_strength is constant
    target_alignment: float = 0.7  # Legacy cosine setpoint, used only when uncalibrated
    kp: float | None = None  # Proportional gain; None = derived (LEGACY_KP when uncalibrated)
    ki: float | None = None  # Integral gain; None = derived from the calibration
    kd: float = 0.0  # Derivative gain; ships disabled (noise-sensitive on a per-token reading)
    derivative_filter_alpha: float = 0.1  # EMA weight on the derivative term
    # EMA weight on the sensor reading; its time constant (1 - alpha) / alpha tokens is
    # the plant's only dynamics, and what the gain derivation is computed against.
    measurement_filter_alpha: float = 0.1
    # Tokens over which the loop corrects a deviation; None = the filter time constant.
    closed_loop_tau: float | None = None
    # The sensor-only cell; None = the layer above the top actuator when a vector
    # exists there, else the top actuator is the top cell (every cell reads its own
    # layer before its own injection).
    readout_layer: int | None = None
    max_strength: float = 1.5  # Maximum steering strength
    min_strength: float = 0.0  # Minimum steering strength
    orthogonal_projection: bool = True  # Project out general capability space
    # Number of principal components of general-task activations treated as the
    # capability subspace. Higher ranks protect more of the model's behaviour and
    # leave less of the steering vector standing; see MIN_RETAINED_NORM_FRACTION.
    capability_subspace_rank: int = DEFAULT_CAPABILITY_SUBSPACE_RANK

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "kd"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        for name in ("derivative_filter_alpha", "measurement_filter_alpha"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if self.closed_loop_tau is not None and self.closed_loop_tau <= 0:
            raise ValueError(f"closed_loop_tau must be positive, got {self.closed_loop_tau}")
        if self.min_strength > self.max_strength:
            raise ValueError("min_strength must not exceed max_strength")


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
        norm = vector.norm()
        if norm <= 0 or not torch.isfinite(norm):
            # A degenerate diff-in-means (e.g. identical positive/negative
            # completions) would divide to NaN and steer with full confidence in a
            # meaningless direction. Keep the zero vector: a zero direction is an
            # honest no-op that the hook and the coupling both treat as inert.
            logger.warning("Steering vector '%s' has zero/non-finite norm; kept inert", name)
            self.vector = torch.zeros_like(vector)
        else:
            self.vector = vector / norm
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

    def extract_from_pairs(
        self, pairs: "Sequence[ContrastivePair]", max_length: int = 128
    ) -> dict[int, SteeringVector]:
        """Difference-in-means read at each pair's completion position.

        This is the behavioural extraction #3 exists to provide, and it differs
        from :meth:`extract` in *where it reads*. ``extract`` mean-pools every
        position of an instruction prefix, recovering the direction that separates
        two sentences about a behaviour. This reads the single token at which the
        model is *producing* the behaviour -- the answer token of a shared prompt's
        A/B completions -- so the difference is a behaviour direction (CAA; Rimsky
        et al., 2024).

        For each pair the positive and negative completions are read at their own
        recorded positions, and the per-layer means are differenced across pairs.
        A pair whose completion is entirely truncated away is skipped with a
        warning rather than read at a prompt token.
        """
        self._register_hooks()
        try:
            input_device = self._input_device()
            positives: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
            negatives: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
            used = 0

            for pair in pairs:
                pos = self._read_completion_activations(
                    pair.prompt,
                    pair.positive_completion,
                    pair.read_position,
                    max_length,
                    input_device,
                )
                neg = self._read_completion_activations(
                    pair.prompt,
                    pair.negative_completion,
                    pair.read_position,
                    max_length,
                    input_device,
                )
                if pos is None or neg is None:
                    continue
                for layer_idx in self.layers:
                    positives[layer_idx].append(pos[layer_idx])
                    negatives[layer_idx].append(neg[layer_idx])
                used += 1

            if used == 0:
                raise ValueError("no usable pairs: every completion was empty or truncated away")

            steering_vectors = {}
            for layer_idx in self.layers:
                pos_mean = torch.stack(positives[layer_idx]).mean(dim=0)
                neg_mean = torch.stack(negatives[layer_idx]).mean(dim=0)
                steering_vectors[layer_idx] = SteeringVector(
                    name="extracted",
                    vector=pos_mean - neg_mean,
                    layer=layer_idx,
                    description=f"Behavioural diff-in-means from {used} completion pairs",
                )

            total = len(pairs) if hasattr(pairs, "__len__") else used
            logger.info(
                "Behavioural extraction: %d/%d pairs used, layers %s", used, total, self.layers
            )
            return steering_vectors
        finally:
            self._remove_hooks()

    def _read_completion_activations(
        self,
        prompt: str,
        completion: str,
        read_position: int,
        max_length: int,
        input_device: torch.device,
    ) -> dict[int, torch.Tensor] | None:
        """Per-layer activation at the completion's read position, or None if unusable.

        The read position indexes the *completion*, so its absolute position is
        resolved against the token boundary tokenisation actually produced for
        ``prompt`` -- never a character offset, which would land mid-token on a
        subword vocabulary.
        """
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", max_length=max_length, truncation=True
        )["input_ids"]
        inputs = self.tokenizer(
            prompt + completion, return_tensors="pt", max_length=max_length, truncation=True
        ).to(input_device)

        seq_len = int(inputs["input_ids"].shape[1])
        completion_start = min(int(prompt_ids.shape[1]), seq_len)
        if completion_start >= seq_len:
            logger.warning("Completion truncated away for prompt %r; skipping pair", prompt[:40])
            return None

        if read_position >= 0:
            position = completion_start + read_position
        else:
            position = seq_len + read_position
        position = max(completion_start, min(position, seq_len - 1))

        with torch.no_grad():
            self.model(**inputs, output_hidden_states=False)

        result: dict[int, torch.Tensor] = {}
        for layer_idx in self.layers:
            if layer_idx not in self.activations:
                logger.warning(f"No activation captured for layer {layer_idx}")
                return None
            hidden = self.activations[layer_idx].float().cpu()
            result[layer_idx] = hidden[0, position]
        return result


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
