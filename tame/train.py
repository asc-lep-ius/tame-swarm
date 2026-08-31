"""
TAME Training Loop

This script implements the training loop for the Multi-Scale Competency Architecture
with proper integration of the Mixture of Bidders (MoB) economic dynamics.

Key features:
1. Loss-based wealth updates for expert specialization
2. Confidence head calibration via auxiliary loss
3. Wealth history tracking and Gini monitoring
4. Support for LoRA fine-tuning (memory efficient)
5. Gradient accumulation for larger effective batch sizes

Usage:
    python train.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --dataset wikitext
    python train.py --model_id meta-llama/Llama-2-7b-hf --dataset c4 --use_lora
"""

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

try:
    import datasets
    from datasets import load_dataset

    # Option D: Disable dataset caching to reduce RAM usage
    datasets.disable_caching()
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("'datasets' library not installed. Install with: pip install datasets")

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger.warning(
        "'peft' library not installed. LoRA support disabled. Install with: pip install peft"
    )

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        return iterable


try:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    logger.warning("'accelerate' library not installed. Model re-dispatch disabled.")

from config import get_active_profile
from evaluation import (
    DEFAULT_HELD_OUT_SEQUENCES,
    SOURCE_TRAIN_HOLDOUT,
    HeldOutSplit,
    build_held_out_split,
    evaluate,
    is_held_out_position,
)
from metrics import MetricSink
from mob import (
    ROUTER_AUCTION,
    ROUTER_SOFTMAX,
    ROUTING_SATURATION_THRESHOLD,
    MoBConfig,
    apply_mob_to_model,
    get_mob_layers,
    get_mob_statistics,
    get_total_calibration_loss,
    get_total_router_z_loss,
    save_mob_state,
    update_all_mob_from_loss,
)
from parity import ArmFingerprint, data_order_fingerprint, fingerprint_arm
from specialisation import SpecialisationReport, probe_specialisation

_profile = get_active_profile()

# The three arms of #12. ``mob`` and ``softmax`` are gates over the same upcycled
# experts and differ only in how reports become an allocation; ``dense`` is the
# absence of routing -- the original FFN, untouched -- and is the
# capability-preservation floor rather than a third gate.
ARM_MOB = "mob"
ARM_SOFTMAX = "softmax"
ARM_DENSE = "dense"
ARMS = (ARM_MOB, ARM_SOFTMAX, ARM_DENSE)

# Which MoB gate each routed arm configures. ``dense`` is absent because it builds
# no MoB layer at all.
ARM_ROUTERS = {ARM_MOB: ROUTER_AUCTION, ARM_SOFTMAX: ROUTER_SOFTMAX}

HELD_OUT_SPLIT_FILENAME = "held_out_split.pt"
METRICS_FILENAME = "metrics.jsonl"


# Below this, top_k slots are buying less than one extra expert's worth of mixing.
# Only meaningful for top_k > 1: at top_k == 1 the effective count is 1.0 and the
# saturated fraction 1.0 by construction, and neither is a fault.
MIN_HEALTHY_EFFECTIVE_EXPERTS = 1.5


def _scalar(value: object) -> float:
    """Sync one aggregate statistic to the host, or report it as missing.

    ``get_mob_statistics`` omits the gate diagnostics until every MoB layer has
    forwarded at least once, and a partial average would be worse than an absent
    number. NaN formats and stores as one; a zero would read as a collapsed gate.
    """
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float("nan")


@dataclass
class TrainingConfig:
    """
    Configuration for TAME training.

    Defaults are auto-configured from the active model profile in config.py.

    Memory Profile (defaults optimized for 16GB VRAM - RTX 5070 Ti, RTX 4080, etc.):
    - batch_size=2, seq_len=512, adapter_rank=32 → ~12GB peak usage
    - Increase batch_size/seq_len if you have 24GB+ (A10G, RTX 4090, etc.)
    """

    # Model (auto-configured from ACTIVE_MODEL)
    model_id: str = _profile["model_id"]
    output_dir: str = "./tame_checkpoints"

    # MoB settings (auto-configured from ACTIVE_MODEL)
    num_experts: int = 4
    top_k: int = 2
    mob_layers_start: int = _profile["mob_layers_start"]
    mob_layers_end: int = _profile["mob_layers_end"]
    adapter_rank: int = 32  # Reduced from 64 for 16GB GPUs (still effective)

    # Training hyperparameters (optimized for 16GB VRAM)
    batch_size: int = 2  # Reduced from 4 for memory efficiency
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size of 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    max_seq_length: int = 512  # Reduced from 1024 for 16GB GPUs

    # MoB-specific training
    calibration_loss_weight: float = (
        0.15  # Weight for confidence calibration loss (increased for stronger training)
    )
    wealth_update_frequency: int = 1  # How often to update wealth (every N steps)
    log_frequency: int = 10  # How often to log comprehensive training statistics (every N steps)

    # LoRA (optional)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"  # bfloat16, float16, or float32
    # Gradient checkpointing saves memory but requires deterministic forward pass
    # MoB layer now uses dense computation for checkpointing compatibility
    gradient_checkpointing: bool = True

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500

    # Experimental arm (#12). The gate is the only thing allowed to vary between
    # arms; parity.py refuses a comparison in which anything else did.
    router: str = ARM_MOB

    # Held-out evaluation. ``held_out_sequences`` fixes the split size so it is a
    # property of the experiment rather than of whichever run built the cache
    # first; ``probe_tokens`` is the specialisation probe, floored at 4096 because
    # below roughly a thousand tokens report decisiveness carries several points of
    # noise -- the same order as the effect being resolved between arms.
    held_out_sequences: int = DEFAULT_HELD_OUT_SEQUENCES
    probe_tokens: int = 4096

    # Misc
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject a config that would fail deep inside a run rather than at the boundary.

        argparse ``choices`` covers the CLI, but a config assembled in code -- which is
        how the comparison harness builds its arms -- reaches ``ARM_ROUTERS[router]``
        as a bare ``KeyError`` and ``step % eval_steps`` as a ``ZeroDivisionError``
        several minutes in, with the model loaded and the split already built.
        """
        if self.router not in ARMS:
            raise ValueError(f"Unsupported router '{self.router}'. Supported: {sorted(ARMS)}")

        positive_fields = (
            "max_steps",
            "gradient_accumulation_steps",
            "eval_steps",
            "save_steps",
            "log_frequency",
            "probe_tokens",
            "held_out_sequences",
            "wealth_update_frequency",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")


class TAMETrainer:
    """
    Trainer for TAME architecture with MoB wealth dynamics.

    This trainer implements the key training loop that enables expert specialization:
    1. Forward pass through model (MoB layers route tokens to experts)
    2. Compute per-token loss (for wealth update signal)
    3. Update expert wealth based on loss reduction (specialization pressure)
    4. Add calibration loss (confidence head training)
    5. Backward pass and optimizer step
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set dtype
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(config.dtype, torch.bfloat16)

        # Will be initialized later
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        self._last_avg_metrics = {
            "loss": 0.0,
            "calibration_loss": 0.0,
            "router_z_loss": 0.0,
            "perplexity": 0.0,
        }

        # Wealth history for analysis
        self.wealth_history: list[dict[str, Any]] = []

        # Held-out evaluation state, populated by setup()
        self.held_out_split: HeldOutSplit | None = None
        self.fingerprint: ArmFingerprint | None = None
        self.eval_history: list[dict[str, Any]] = []
        self.metrics = MetricSink(
            Path(config.output_dir) / METRICS_FILENAME,
            run_tags={"router": config.router, "seed": config.seed},
        )

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def setup(self):
        """Initialize model, tokenizer, optimizer, and data."""
        logger.info(f"Loading model: {self.config.model_id}")
        logger.info(f"Arm: --router {self.config.router}")

        # Created before anything writes into it: the held-out split is frozen here
        # so that later runs and sibling arms read the same file rather than
        # rebuilding one each and hoping they agree.
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            use_fast=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )

        # Apply gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply MoB transformation. The dense arm skips it entirely -- it is the
        # unrouted floor, so "convert nothing" is the arm rather than a degenerate
        # configuration of one.
        if self.config.router != ARM_DENSE:
            self._apply_mob()
        else:
            logger.info("Dense arm: leaving the original FFN in place, no MoB layers")

        # Apply LoRA if requested (before re-dispatch so all new modules are included)
        if self.config.use_lora:
            self._apply_lora()

        # Re-dispatch model after MoB + LoRA transformations to ensure
        # consistent device placement. Fixes meta device gradient error
        # when using device_map="auto". Must happen AFTER all model
        # modifications (MoB and LoRA) are complete.
        if self.device.type == "cuda" and HAS_ACCELERATE:
            self._redispatch_model()
        elif self.device.type != "cuda":
            # Move to device if not using CUDA
            self.model = self.model.to(self.device)  # pyright: ignore[reportArgumentType] # .to() overload expects Device, not torch.device

        # Setup optimizer
        self._setup_optimizer()

        # Held-out split before the training data: when the dataset ships no
        # validation split, the training stream has to skip exactly the rows the
        # evaluation set took, so the split has to exist before the loader is built.
        self._setup_held_out_split()

        # Setup data
        self._setup_data()

        # Setup scheduler
        self._setup_scheduler()

        # Parity fingerprint last: it hashes the data this arm will actually train
        # on, which needs the loader that was just built.
        self._record_fingerprint()

        logger.info(
            f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        logger.info(
            "Trainable parameters: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        # Log MoB statistics
        mob_layers = get_mob_layers(self.model)
        logger.info(f"Applied MoB to {len(mob_layers)} layers")

    def _apply_mob(self):
        """Apply Mixture of Bidders transformation to model."""
        assert self.model is not None
        logger.info("Applying MoB transformation...")

        # Determine hidden dimensions from model config
        model_config = self.model.config
        hidden_dim = getattr(model_config, "hidden_size", 4096)
        intermediate_dim = getattr(model_config, "intermediate_size", 14336)

        mob_config = MoBConfig(
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            use_shared_base=True,
            adapter_rank=self.config.adapter_rank,
            use_loss_feedback=True,
            use_local_quality=True,
            use_differentiable_routing=True,
            confidence_calibration_weight=self.config.calibration_loss_weight,
            router=ARM_ROUTERS[self.config.router],
        )

        # Determine which layers to modify
        layers_to_modify = list(range(self.config.mob_layers_start, self.config.mob_layers_end))

        self.model = apply_mob_to_model(self.model, mob_config, layers_to_modify=layers_to_modify)

    def _redispatch_model(self):
        """
        Re-dispatch model after MoB/LoRA transformations using Accelerate.

        This ensures all newly created modules (MoB, LoRA adapters) are properly placed
        on devices after modifying the model architecture. Without this, modules created
        during transformations may remain on 'meta' device causing gradient errors like:
        "RuntimeError: expected device meta but got cuda:0"
        """
        assert self.model is not None
        logger.info("Re-dispatching model after transformations...")

        # First, check for any parameters still on 'meta' device
        # This can happen with device_map="auto" lazy loading
        meta_params = []
        for name, param in self.model.named_parameters():
            if param.device.type == "meta":
                meta_params.append(name)

        if meta_params:
            logger.info(f"Found {len(meta_params)} parameters on meta device, materializing...")
            # Meta tensors require special handling - can't use .to() directly
            # Use to_empty() to allocate memory, then initialize weights
            self.model = self.model.to_empty(device=self.device)

            # Re-initialize any parameters that were on meta device
            # For most cases these are MoB adapter weights which should start near-zero anyway
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.isnan().any() or param.isinf().any() or (param == 0).all():
                        # Parameter needs initialization
                        if "weight" in name:
                            if param.dim() >= 2:
                                # Use kaiming for weight matrices
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            else:
                                # Small init for 1D weights
                                nn.init.uniform_(param, -0.01, 0.01)
                        elif "bias" in name:
                            nn.init.zeros_(param)
                        else:
                            # Default small random init
                            nn.init.uniform_(param, -0.01, 0.01)

            logger.info("Model materialized and re-initialized on device")

            # Now reload pretrained weights for the base model components
            # This preserves the original model weights while keeping new MoB/LoRA init
            self._reload_pretrained_weights()
            return

        try:
            # For PEFT models, we need to work with the underlying model
            model_to_dispatch = self.model
            is_peft = hasattr(self.model, "base_model")

            if is_peft:
                logger.info("Detected PEFT model, working with base model for dispatch")

            # Get balanced memory allocation
            max_memory = get_balanced_memory(
                model_to_dispatch,
                max_memory=None,  # Use all available memory
                no_split_module_classes=[
                    "MixtureOfBidders",
                    "LoraLayer",
                ],  # Don't split these modules
            )

            device_map = infer_auto_device_map(
                model_to_dispatch,
                max_memory=max_memory,
                no_split_module_classes=["MixtureOfBidders", "LoraLayer"],
            )

            # Log device distribution
            device_counts = {}
            for _module_name, device in device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            logger.info(f"Device map: {device_counts}")

            # Dispatch the model with the new device map
            self.model = dispatch_model(model_to_dispatch, device_map=device_map)
            logger.info("Model re-dispatched successfully")

        except Exception as e:
            logger.warning(
                f"Re-dispatch failed ({type(e).__name__}: {e}), falling back to simple device move"
            )
            # Fallback: check if we have meta tensors before calling .to()
            has_meta = any(p.device.type == "meta" for p in self.model.parameters())
            if has_meta:
                self.model = self.model.to_empty(device=self.device)
                self._reload_pretrained_weights()
            else:
                self.model = self.model.to(self.device)  # pyright: ignore[reportArgumentType] # .to() overload expects Device, not torch.device

    def _reload_pretrained_weights(self):
        """
        Reload pretrained weights after materializing from meta device.

        Uses memory-efficient streaming from safetensors files (<500MB RAM)
        instead of loading the full model (~14GB RAM).
        """
        assert self.model is not None
        logger.info("Reloading pretrained weights (streaming from safetensors)...")

        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            from safetensors import safe_open

            # Get list of safetensor files in the model repo
            repo_files = list_repo_files(self.config.model_id)
            safetensor_files = [f for f in repo_files if f.endswith(".safetensors")]

            if not safetensor_files:
                logger.warning("No safetensors files found, falling back to bin files")
                self._reload_pretrained_weights_legacy()
                return

            # Build mapping of current model keys (handling PEFT prefix)
            current_state_dict = self.model.state_dict()

            # PEFT wraps keys with "base_model.model." prefix
            # Build reverse mapping: original_key -> peft_key
            key_mapping = {}
            for peft_key in current_state_dict:
                # Strip PEFT prefixes to get original key
                original_key = peft_key
                for prefix in ["base_model.model.", "base_model."]:
                    if original_key.startswith(prefix):
                        original_key = original_key[len(prefix) :]
                        break
                key_mapping[original_key] = peft_key

            copied = 0
            skipped = 0

            # Stream each safetensor file and copy matching weights
            for sf_file in safetensor_files:
                try:
                    # Download file (uses cache if already downloaded)
                    local_path = hf_hub_download(
                        repo_id=self.config.model_id,
                        filename=sf_file,
                    )

                    # Open safetensors file for memory-mapped reading
                    with safe_open(local_path, framework="pt", device="cpu") as f:
                        for tensor_name in f.keys():  # noqa: SIM118
                            # Find matching key in current model
                            peft_key = key_mapping.get(tensor_name)

                            if peft_key and peft_key in current_state_dict:
                                src_tensor = f.get_tensor(tensor_name)
                                dst_tensor = current_state_dict[peft_key]

                                if src_tensor.shape == dst_tensor.shape:
                                    # Copy directly to device
                                    with torch.no_grad():
                                        dst_tensor.copy_(src_tensor.to(self.device))
                                    copied += 1
                                else:
                                    skipped += 1
                            else:
                                skipped += 1

                except Exception as e:
                    logger.warning(f"Error loading {sf_file}: {e}")
                    continue

            logger.info(
                f"Reloaded {copied} pretrained weight tensors (skipped {skipped} non-matching)"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError as e:
            logger.warning(f"Missing dependency for streaming: {e}")
            logger.warning("Install with: pip install safetensors huggingface_hub")
            self._reload_pretrained_weights_legacy()
        except Exception as e:
            logger.warning(f"Streaming reload failed: {e}")
            self._reload_pretrained_weights_legacy()

    def _reload_pretrained_weights_legacy(self):
        """
        Legacy fallback: reload weights by loading full model.
        Warning: Uses ~14GB RAM for 7B models.
        """
        assert self.model is not None
        logger.warning("Using legacy weight reload (high RAM usage)")

        try:
            from transformers import AutoModelForCausalLM

            # Load a fresh copy of weights (on CPU)
            fresh_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=self.dtype,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # At least try to reduce peak
            )
            fresh_state_dict = fresh_model.state_dict()
            current_state_dict = self.model.state_dict()

            # Build key mapping for PEFT
            key_mapping = {}
            for peft_key in current_state_dict:
                original_key = peft_key
                for prefix in ["base_model.model.", "base_model."]:
                    if original_key.startswith(prefix):
                        original_key = original_key[len(prefix) :]
                        break
                key_mapping[original_key] = peft_key

            copied = 0
            for original_key, param in fresh_state_dict.items():
                peft_key = key_mapping.get(original_key)
                if (
                    peft_key
                    and peft_key in current_state_dict
                    and current_state_dict[peft_key].shape == param.shape
                ):
                    with torch.no_grad():
                        current_state_dict[peft_key].copy_(param.to(self.device))
                    copied += 1

            logger.info(f"Reloaded {copied} pretrained weight tensors")

            del fresh_model
            del fresh_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Could not reload pretrained weights: {e}")
            logger.warning("Model will use randomly initialized weights for some components")

    def _apply_lora(self):
        """Apply LoRA adapters for memory-efficient training."""
        if not HAS_PEFT:
            logger.warning("PEFT not installed, skipping LoRA")
            return

        logger.info("Applying LoRA adapters...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )

        assert self.model is not None
        self.model = get_peft_model(cast(PreTrainedModel, self.model), lora_config)
        self.model.print_trainable_parameters()

    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay."""
        assert self.model is not None
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        num_training_steps = self.config.max_steps
        num_warmup_steps = self.config.warmup_steps

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _setup_held_out_split(self):
        """Build or restore the frozen held-out split for this experiment.

        Restored from ``output_dir`` when it is already there, which is what makes
        the number comparable across runs rather than merely across arms: a split
        rebuilt from a dataset that has since been revised is a different
        measurement wearing the same name.
        """
        assert self.tokenizer is not None
        if not HAS_DATASETS:
            raise ImportError("Install datasets: pip install datasets")

        cache_path = Path(self.config.output_dir) / HELD_OUT_SPLIT_FILENAME
        if cache_path.exists():
            self.held_out_split = HeldOutSplit.load(cache_path)
            self._assert_cache_matches_config(self.held_out_split, cache_path)
            logger.info(
                f"Held-out split restored from {cache_path} "
                f"(fingerprint {self.held_out_split.fingerprint}, "
                f"{self.held_out_split.leakage_risk})"
            )
            return

        self.held_out_split = build_held_out_split(
            dataset_name=self.config.dataset_name,
            dataset_config=self._dataset_config(),
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            load_dataset=load_dataset,
            num_sequences=self.config.held_out_sequences,
        )
        self.held_out_split.save(cache_path)

    def _assert_cache_matches_config(self, split: HeldOutSplit, cache_path: Path) -> None:
        """Refuse a cached split built for a different corpus, length or size.

        ``HeldOutSplit.load`` proves the file is internally consistent; it cannot
        know what this run asked for. Parity will not catch the difference either --
        every arm reads the same stale cache, so the arms agree and the comparison
        is a comparison on the wrong data.
        """
        expected_dataset = self._dataset_config()
        expected_dataset = (
            f"{self.config.dataset_name}/{expected_dataset}"
            if expected_dataset
            else self.config.dataset_name
        )
        cached_length = int(split.input_ids.shape[1])
        mismatches = []
        if split.dataset != expected_dataset:
            mismatches.append(
                f"dataset: cached {split.dataset!r} vs configured {expected_dataset!r}"
            )
        if cached_length != self.config.max_seq_length:
            mismatches.append(
                f"max_seq_length: cached {cached_length} vs configured {self.config.max_seq_length}"
            )
        # Asymmetric on purpose: a cache larger than the config was built for a
        # different experiment, while a smaller one is also what a source that ran
        # out of usable rows legitimately produces -- collect_documents already warns
        # about that, and refusing it would make a short corpus unusable.
        if split.num_sequences > self.config.held_out_sequences:
            mismatches.append(
                f"held_out_sequences: cached {split.num_sequences} vs configured "
                f"{self.config.held_out_sequences}"
            )
        elif split.num_sequences < self.config.held_out_sequences:
            logger.warning(
                f"Held-out split cached at {cache_path} has {split.num_sequences} sequences "
                f"against the configured {self.config.held_out_sequences}; the source may have "
                "run short, but delete the cache to be sure it is not stale"
            )
        if mismatches:
            raise ValueError(
                f"The held-out split cached at {cache_path} was not built for this run "
                "(" + "; ".join(mismatches) + "). Delete it and let the run rebuild it."
            )

    def _dataset_config(self) -> str | None:
        """The dataset config actually in force, by one rule rather than two.

        ``dataset_config`` names a wikitext variant and means nothing for any other
        dataset. The split builder and the parity fingerprint both need this, and
        computing it twice is how a fingerprint ends up recording a configuration
        the run did not use -- which is worse than recording nothing, because it
        looks like evidence.
        """
        return self.config.dataset_config if self.config.dataset_name == "wikitext" else None

    def _record_fingerprint(self):
        """Hash what this arm will train on, so a comparison can refuse a confound."""
        assert self.train_dataloader is not None
        assert self.held_out_split is not None
        assert self.model is not None

        # A fresh iterator over a streaming dataset restarts it, so this reads the
        # same rows the training iterator is about to read. Every arm pays the same
        # cost in the same order, which is what keeps the check from perturbing the
        # thing it is checking.
        order = data_order_fingerprint(iter(self.train_dataloader))

        self.fingerprint = fingerprint_arm(
            self.config,
            dataset_config=self._dataset_config(),
            eval_split_fingerprint=self.held_out_split.fingerprint,
            data_order=order,
            converted_layers=len(get_mob_layers(self.model)),
        )
        logger.info(f"Arm fingerprint: {self.fingerprint.as_dict()}")

    def evaluate_held_out(self, step: int) -> dict[str, float]:
        """Held-out loss, perplexity and the functional specialisation probe.

        The economy is frozen for the duration (``mob.frozen_economy``): no wealth
        moves, no usage count advances, no coupling step is set, and the training
        statistics the next log line reads are restored on the way out. That is the
        difference between an evaluation and an unlogged training step.
        """
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.held_out_split is not None

        result = evaluate(self.model, self.held_out_split, self.config.batch_size, self.device)
        measurements = dict(result.as_metrics())

        report: SpecialisationReport | None = probe_specialisation(
            self.model,
            self.held_out_split,
            self.tokenizer,
            self.device,
            batch_size=self.config.batch_size,
            probe_tokens=self.config.probe_tokens,
        )
        if report is not None:
            measurements.update(report.as_metrics())

        self.metrics.log(step, measurements)
        self.eval_history.append({"step": step, **measurements})

        logger.info(
            f"  eval @ {step}: loss {result.loss:.4f} | ppl {result.perplexity:.2f} "
            f"| {result.num_tokens} tokens | split {result.fingerprint}"
        )
        if report is not None:
            logger.info(
                f"  spec @ {step}: expert cos-dist {report.divergence.mean_cosine_distance:.4f}"
                f" | routing JS vs corpus {report.profile.mean_js_from_corpus:.4f}"
                f" | report-decisive {report.report_decisiveness:.1%}"
                f" over {report.probe_tokens} tokens"
            )
        return measurements

    def _setup_data(self):
        """Setup training data loader."""
        assert self.tokenizer is not None
        if not HAS_DATASETS:
            logger.error("datasets library required for training")
            raise ImportError("Install datasets: pip install datasets")

        logger.info(f"Loading dataset: {self.config.dataset_name}")

        # Load dataset
        # Option A: Use streaming=True to reduce RAM usage (~100MB vs ~2GB)
        if self.config.dataset_name == "wikitext":
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split="train",
                streaming=True,  # Stream chunks instead of loading full dataset
            )
            text_column = "text"
        elif self.config.dataset_name == "c4":
            dataset = load_dataset("c4", "en", split="train", streaming=True)
            text_column = "text"
        else:
            dataset = load_dataset(
                self.config.dataset_name,
                split="train",
                streaming=True,
            )
            text_column = "text"

        # The fallback holdout carved the evaluation set out of *this* stream, so
        # the same rows have to leave it here. Filtering on the raw row index is
        # what makes the two sides provably the same set rather than two filters
        # that happen to agree -- see evaluation.is_held_out_position.
        if self.held_out_split is not None and self.held_out_split.source == SOURCE_TRAIN_HOLDOUT:
            dataset = dataset.filter(
                lambda _example, index: not is_held_out_position(index),
                with_indices=True,
            )
            logger.info("Training stream filtered: held-out row positions removed")

        # Tokenize — capture tokenizer locally for pyright closure narrowing
        tokenizer = self.tokenizer

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

        # Process dataset (streaming datasets don't have column_names attribute)
        if hasattr(dataset, "column_names") and dataset.column_names is not None:
            # Non-streaming dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
        else:
            # Streaming dataset - columns are automatically handled
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=[text_column],
            )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )

        # Create dataloader
        self.train_dataloader = DataLoader(
            tokenized_dataset,  # pyright: ignore[reportArgumentType] # DataLoader stubs don't accept IterableDataset
            batch_size=self.config.batch_size,
            shuffle=hasattr(tokenized_dataset, "__len__"),
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single training step with MoB wealth updates.

        This is the core of the TAME training loop:
        1. Forward pass (experts route and process tokens)
        2. Compute per-token loss (provides specialization signal)
        3. Update expert wealth based on loss (key for differentiation!)
        4. Add calibration loss (trains confidence heads)
        5. Backward pass
        """
        assert self.model is not None
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        batch_size, seq_len = input_ids.shape

        for mob in get_mob_layers(self.model):
            mob.set_coupling_step(self.global_step)

        # Forward pass
        outputs = self.model(  # pyright: ignore[reportCallIssue] # model forward call signature varies by runtime model type
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We'll compute loss manually for per-token access
            use_cache=False,
        )

        logits = outputs.logits

        # Compute per-token loss (unreduced for wealth updates)
        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        # Flatten for cross entropy
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        # Reshape back to (batch, seq_len-1)
        per_token_loss = per_token_loss.view(batch_size, seq_len - 1)

        # =========================================================
        # KEY: Update MoB wealth based on loss (SPECIALIZATION!)
        # =========================================================
        # This is what makes experts actually specialize!
        # Experts that reduce loss get rewarded, others decay
        if self.global_step % self.config.wealth_update_frequency == 0:
            update_all_mob_from_loss(
                self.model,
                per_token_loss.detach(),  # Detach to prevent double gradients
                shift_mask,
            )

        # Compute mean loss for backprop
        valid_mask = (shift_labels != -100) & (shift_mask == 1)
        main_loss = (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # =========================================================
        # Add calibration loss for confidence head training
        # =========================================================
        # This teaches confidence heads to predict when they'll do well
        calibration_loss = get_total_calibration_loss(self.model)
        router_z_loss = get_total_router_z_loss(self.model)

        # Total loss
        total_loss = main_loss + calibration_loss + router_z_loss

        # NaN guard: skip backprop if loss is NaN to prevent gradient corruption
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            cal_val = calibration_loss.item() if isinstance(calibration_loss, torch.Tensor) else 0
            router_val = router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else 0
            logger.warning(
                f"Step {self.global_step}: NaN/Inf loss detected "
                f"(main={main_loss.item()}, cal={cal_val}, router_z={router_val}), "
                "skipping backward"
            )
            return {
                "loss": float("nan"),
                "calibration_loss": 0.0,
                "router_z_loss": 0.0,
                "total_loss": float("nan"),
                "perplexity": float("nan"),
            }

        # Scale for gradient accumulation
        scaled_loss = total_loss / self.config.gradient_accumulation_steps

        # Backward pass
        scaled_loss.backward()

        return {
            "loss": main_loss.item(),
            "calibration_loss": calibration_loss.item()
            if isinstance(calibration_loss, torch.Tensor)
            else 0.0,
            "router_z_loss": router_z_loss.item()
            if isinstance(router_z_loss, torch.Tensor)
            else 0.0,
            "total_loss": total_loss.item(),
            "perplexity": math.exp(min(main_loss.item(), 20)),  # Cap to prevent overflow
        }

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info("=" * 118)
        logger.info(
            f"{'Step':>6} | {'Prog':>5} | {'Loss':>7} | {'PPL':>10}"
            f" | {'Cal':>6} | {'Z':>6} | {'Mean Wealth':>12} | {'Std Dev':>8}"
            f" | {'Gini':>6} | {'Perf EMA':>9} | {'Top1':>6} | {'EffExp':>6}"
        )
        logger.info("-" * 118)

        assert self.model is not None
        assert self.train_dataloader is not None
        assert self.optimizer is not None
        assert self.scheduler is not None

        # Start wealth tracking for analysis
        for mob in get_mob_layers(self.model):
            mob.start_tracking()

        # Training loop
        self.model.train()
        accumulated_metrics = {
            "loss": 0.0,
            "calibration_loss": 0.0,
            "router_z_loss": 0.0,
            "perplexity": 0.0,
        }

        data_iter = iter(self.train_dataloader)

        progress_bar = (
            tqdm(range(self.config.max_steps), desc="Training")
            if HAS_TQDM
            else range(self.config.max_steps)
        )

        # The sink is closed however the loop leaves: an interrupted or failed
        # run still owns a metrics file, and a half-written one is easier to read
        # than one whose handle was never released.
        try:
            for step in progress_bar:
                self.global_step = step

                # Get batch (handle iterator exhaustion)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                # Training step
                metrics = self.train_step(batch)

                # Accumulate metrics
                for key in accumulated_metrics:
                    if key in metrics:
                        accumulated_metrics[key] += metrics[key]

                # Gradient accumulation step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Compute averaged metrics for this accumulation window (stored for next log)
                    self._last_avg_metrics = {
                        k: v / self.config.gradient_accumulation_steps
                        for k, v in accumulated_metrics.items()
                    }

                    # Reset accumulated metrics
                    accumulated_metrics = {k: 0.0 for k in accumulated_metrics}

                # Log comprehensive training statistics
                if step > 0 and step % self.config.log_frequency == 0:
                    self._log_training_step(step)

                # Held-out evaluation. This is what ``eval_steps`` has always claimed
                # to do and never did; every number the project reports as evidence of
                # capability comes from here rather than from the training batch above.
                if step > 0 and step % self.config.eval_steps == 0:
                    self.evaluate_held_out(step)

                # Save checkpoint
                if step > 0 and step % self.config.save_steps == 0:
                    self._save_checkpoint(step)

            # Final save and log
            self._log_training_step(self.config.max_steps)
            self.evaluate_held_out(self.config.max_steps)
            self._save_checkpoint(self.config.max_steps, final=True)
        finally:
            self.metrics.close()

        logger.info("=" * 118)
        logger.info("Training complete!")

    def _log_training_step(self, step: int):
        """
        Log comprehensive training statistics for fine-tuning analysis.

        Outputs: Step, Progress%, Loss, Perplexity, Calibration Loss, Router Z,
                 Mean Wealth, Std Dev, Gini Coefficient, Performance EMA,
                 mean top-1 routing weight, effective expert count.
        """
        assert self.model is not None
        # Get MoB wealth statistics
        stats = get_mob_statistics(self.model)

        # Calculate progress percentage
        progress = (step / self.config.max_steps) * 100

        # Use averaged metrics from gradient accumulation window
        metrics = self._last_avg_metrics
        loss = metrics.get("loss", float("nan"))
        ppl = metrics.get("perplexity", float("nan"))
        cal = metrics.get("calibration_loss", 0.0)
        router_z = metrics.get("router_z_loss", 0.0)

        # Recorded before the MoB branch: the dense arm has no wealth statistics and
        # would otherwise contribute no training curve at all, which is exactly the
        # arm a capability-preservation claim is read against.
        measurements = {
            "train/loss": loss,
            "train/perplexity": ppl,
            "train/calibration_loss": cal,
            "train/router_z_loss": router_z,
        }

        if stats:
            _mw = stats["mean_wealth"]
            assert isinstance(_mw, torch.Tensor)
            mean_wealth = float(_mw.item())

            _sw = stats["wealth_std"]
            assert isinstance(_sw, torch.Tensor)
            std_wealth = float(_sw.item())

            _gi = stats["wealth_gini"]
            assert isinstance(_gi, torch.Tensor)
            gini = float(_gi.item())

            _pe = stats["mean_performance"]
            assert isinstance(_pe, torch.Tensor)
            perf_ema = float(_pe.item())

            # What the gate actually did, as opposed to what top_k configures. These
            # are the statistics that would have surfaced a gate saturating on the
            # absolute wealth scale, so they belong on the line that is read every
            # run rather than in a field someone has to know to go and fetch.
            top1 = _scalar(stats.get("routing_top1_mean"))
            effective_experts = _scalar(stats.get("routing_effective_experts"))
            saturated = _scalar(stats.get("routing_top1_saturated_fraction"))

            # Store for analysis
            self.wealth_history.append(
                {
                    "step": step,
                    "progress": progress,
                    "loss": loss,
                    "perplexity": ppl,
                    "calibration_loss": cal,
                    "router_z_loss": router_z,
                    "mean_wealth": mean_wealth,
                    "wealth_std": std_wealth,
                    "wealth_gini": gini,
                    "mean_performance": perf_ema,
                    "routing_top1_mean": top1,
                    "routing_top1_saturated_fraction": saturated,
                    "routing_effective_experts": effective_experts,
                }
            )

            measurements.update(
                {
                    "wealth/mean": mean_wealth,
                    "wealth/std": std_wealth,
                    "wealth/gini": gini,
                    "wealth/mean_performance": perf_ema,
                    "routing/top1_mean": top1,
                    "routing/saturated_fraction": saturated,
                    "routing/effective_experts": effective_experts,
                }
            )

            # Format performance EMA with sign
            perf_sign = "+" if perf_ema >= 0 else ""

            # Log comprehensive line
            logger.info(
                f"{step:>6} | {progress:>4.0f}% | {loss:>7.4f}"
                f" | {ppl:>10.2f} | {cal:>6.4f} | {router_z:>6.4f}"
                f" | {mean_wealth:>12.2f} | {std_wealth:>8.2f}"
                f" | {gini:>6.4f} | {perf_sign}{perf_ema:>8.4f}"
                f" | {top1:>6.4f} | {effective_experts:>6.3f}"
            )

            # The sparse-computation claim is O(top_k x tokens), and it is paid
            # whether or not the second winner contributes anything. This says
            # whether it bought anything -- a question that only exists once there
            # is a second winner to buy, hence the top_k guard.
            if self.config.top_k > 1 and effective_experts < MIN_HEALTHY_EFFECTIVE_EXPERTS:
                logger.warning(
                    f"  ⚠ Effective expert count {effective_experts:.3f} with "
                    f"top_k={self.config.top_k} - {saturated:.0%} of tokens route "
                    f"above {ROUTING_SATURATION_THRESHOLD} to one expert. The gate "
                    "is paying for experts it is not mixing."
                )

            # Economy diagnostics, and only that. Gini measures dispersion of a
            # wealth vector produced by an EMA with a tuned decay and a hard clamp;
            # it says nothing about whether two experts compute the same function,
            # and its direction is the wrong way round for a specialisation reading
            # -- higher Gini means *more* of the routing decided by the wealth
            # scalar and less by what an expert reports about the token. The
            # specialisation numbers are the spec/ metrics from the held-out probe.
            if self.config.router == ARM_MOB:
                if gini < 0.10:
                    logger.warning(
                        f"  ⚠ Low Gini ({gini:.4f}) - wealth is near-flat, so the "
                        "auction is allocating almost entirely on reports. Not a "
                        "specialisation measure. Consider: ↑reward_scale, ↓wealth_decay"
                    )
                elif gini > 0.60:
                    logger.warning(
                        f"  ⚠ High Gini ({gini:.4f}) - wealth monopoly risk. "
                        "Consider: ↑min_wealth, ↓max_wealth"
                    )

                if mean_wealth > 0.9 * 750:
                    logger.warning(
                        f"  ⚠ Wealth near ceiling ({mean_wealth:.0f}/750) - consider ↑max_wealth"
                    )

                if perf_ema < -0.3:
                    logger.warning(
                        f"  ⚠ Negative performance EMA ({perf_ema:.4f}) "
                        "- experts underperforming vs baseline"
                    )
        else:
            # No MoB stats available
            logger.info(
                f"{step:>6} | {progress:>4.0f}% | {loss:>7.4f} | {ppl:>10.2f} | {cal:>6.4f} | "
                f"{'N/A':>12} | {'N/A':>8} | {'N/A':>6} | {'N/A':>9}"
            )

        self.metrics.log(step, measurements)

    def _save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint and wealth state."""
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.optimizer is not None
        assert self.scheduler is not None

        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        checkpoint_str = str(checkpoint_dir)
        if self.config.use_lora and HAS_PEFT:
            self.model.save_pretrained(checkpoint_str)  # pyright: ignore[reportCallIssue] # PeftModel.save_pretrained stubs incomplete
        else:
            self.model.save_pretrained(checkpoint_str)  # pyright: ignore[reportCallIssue] # PreTrainedModel.save_pretrained stubs incomplete

        self.tokenizer.save_pretrained(checkpoint_str)

        save_mob_state(cast(nn.Module, self.model), str(checkpoint_dir / "mob_state.pt"))

        # Save wealth history
        if self.wealth_history:
            torch.save(self.wealth_history, checkpoint_dir / "wealth_history.pt")

        # Save training state. The fingerprint and the eval history travel with the
        # checkpoint because a comparison assembled later needs to prove parity from
        # the artefacts, not from whatever the shell history says was run.
        training_state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "arm_fingerprint": self.fingerprint.as_dict() if self.fingerprint else None,
            "eval_history": self.eval_history,
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

        logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train TAME architecture")

    # Model arguments (defaults from ACTIVE_MODEL profile)
    parser.add_argument(
        "--model_id", type=str, default=_profile["model_id"], help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tame_checkpoints",
        help="Output directory for checkpoints",
    )

    # MoB arguments (defaults from ACTIVE_MODEL profile)
    parser.add_argument(
        "--num_experts", type=int, default=4, help="Number of experts per MoB layer"
    )
    parser.add_argument("--top_k", type=int, default=2, help="Top-k experts to route to")
    parser.add_argument(
        "--mob_layers_start",
        type=int,
        default=_profile["mob_layers_start"],
        help="First layer to apply MoB",
    )
    parser.add_argument(
        "--mob_layers_end",
        type=int,
        default=_profile["mob_layers_end"],
        help="Last layer to apply MoB (exclusive)",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # LoRA
    parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA for memory-efficient training"
    )
    parser.add_argument("--lora_rank", type=int, default=16)

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name (wikitext, c4, or HuggingFace dataset)",
    )

    # Experimental arm (#12)
    parser.add_argument(
        "--router",
        type=str,
        default=ARM_MOB,
        choices=list(ARMS),
        help=(
            "Routing arm: 'mob' is the auction, 'softmax' is the learned-gate control "
            "over the same experts, 'dense' is the original FFN with no routing. "
            "Everything else is held identical and parity is asserted."
        ),
    )

    # Held-out evaluation (#12)
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Run the held-out evaluation every N steps"
    )
    parser.add_argument(
        "--held_out_sequences",
        type=int,
        default=DEFAULT_HELD_OUT_SEQUENCES,
        help="Sequences in the frozen held-out split",
    )
    parser.add_argument(
        "--probe_tokens",
        type=int,
        default=4096,
        help=(
            "Tokens for the functional specialisation probe. Below ~1000 a single "
            "arm carries several points of noise on report decisiveness"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed; must match across arms")

    # Hardware
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        num_experts=args.num_experts,
        top_k=args.top_k,
        mob_layers_start=args.mob_layers_start,
        mob_layers_end=args.mob_layers_end,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        dataset_name=args.dataset,
        dtype=args.dtype,
        router=args.router,
        eval_steps=args.eval_steps,
        held_out_sequences=args.held_out_sequences,
        probe_tokens=args.probe_tokens,
        seed=args.seed,
    )

    # Create trainer and run
    trainer = TAMETrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
