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
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)

try:
    from datasets import load_dataset
    import datasets
    # Option D: Disable dataset caching to reduce RAM usage
    datasets.disable_caching()
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: 'peft' library not installed. LoRA support disabled. Install with: pip install peft")

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
    print("Warning: 'accelerate' library not installed. Model re-dispatch disabled.")

from mob import (
    MoBConfig, 
    MixtureOfBidders, 
    apply_mob_to_model,
    get_mob_layers,
    update_all_mob_from_loss,
    get_total_calibration_loss,
    get_mob_statistics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL PROFILES - Change ACTIVE_MODEL to switch between models
# =============================================================================
# Keep synchronized with main.py!

MODEL_PROFILES = {
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b-it",
        "hidden_dim": 2304,
        "intermediate_dim": 9216,
        "num_layers": 26,
        "mob_layers_start": 5,
        "mob_layers_end": 18,
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_dim": 3072,
        "intermediate_dim": 8192,
        "num_layers": 28,
        "mob_layers_start": 6,
        "mob_layers_end": 20,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_layers": 32,
        "mob_layers_start": 8,
        "mob_layers_end": 24,
    },
}

# =============================================================================
# >>> CHANGE THIS TO SWITCH MODELS <<<
# =============================================================================
ACTIVE_MODEL = "gemma-2-2b"  # Options: "gemma-2-2b", "llama-3.2-3b", "mistral-7b"
# =============================================================================

_profile = MODEL_PROFILES[ACTIVE_MODEL]


@dataclass
class TrainingConfig:
    """
    Configuration for TAME training.
    
    IMPORTANT: Keep ACTIVE_MODEL synchronized with main.py!
    Defaults are auto-configured from the active model profile.
    """
    # Model (auto-configured from ACTIVE_MODEL)
    model_id: str = _profile["model_id"]
    output_dir: str = "./tame_checkpoints"
    
    # MoB settings (auto-configured from ACTIVE_MODEL)
    num_experts: int = 4
    top_k: int = 2
    mob_layers_start: int = _profile["mob_layers_start"]
    mob_layers_end: int = _profile["mob_layers_end"]
    adapter_rank: int = 64
    
    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    max_seq_length: int = 1024
    
    # MoB-specific training
    calibration_loss_weight: float = 0.1  # Weight for confidence calibration loss
    wealth_update_frequency: int = 1  # How often to update wealth (every N steps)
    log_wealth_frequency: int = 100  # How often to log wealth statistics
    
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
    
    # Misc
    seed: int = 42


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
        self.best_loss = float('inf')
        
        # Wealth history for analysis
        self.wealth_history: List[Dict[str, Any]] = []
        
        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    def setup(self):
        """Initialize model, tokenizer, optimizer, and data."""
        logger.info(f"Loading model: {self.config.model_id}")
        
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
        
        # Apply MoB transformation
        self._apply_mob()
        
        # Apply LoRA if requested (before re-dispatch so all new modules are included)
        if self.config.use_lora:
            self._apply_lora()
        
        # Option 4: Re-dispatch model after MoB + LoRA transformations to ensure consistent device placement
        # This fixes the meta device gradient error when using device_map="auto"
        # Must happen AFTER all model modifications (MoB and LoRA) are complete
        if self.device.type == "cuda" and HAS_ACCELERATE:
            self._redispatch_model()
        elif self.device.type != "cuda":
            # Move to device if not using CUDA
            self.model = self.model.to(self.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup data
        self._setup_data()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Log MoB statistics
        mob_layers = get_mob_layers(self.model)
        logger.info(f"Applied MoB to {len(mob_layers)} layers")
    
    def _apply_mob(self):
        """Apply Mixture of Bidders transformation to model."""
        logger.info("Applying MoB transformation...")
        
        # Determine hidden dimensions from model config
        model_config = self.model.config
        hidden_dim = getattr(model_config, 'hidden_size', 4096)
        intermediate_dim = getattr(model_config, 'intermediate_size', 14336)
        
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
        )
        
        # Determine which layers to modify
        layers_to_modify = list(range(
            self.config.mob_layers_start, 
            self.config.mob_layers_end
        ))
        
        self.model = apply_mob_to_model(
            self.model,
            mob_config,
            layers_to_modify=layers_to_modify
        )
    
    def _redispatch_model(self):
        """
        Re-dispatch model after MoB/LoRA transformations using Accelerate.
        
        This ensures all newly created modules (MoB, LoRA adapters) are properly placed 
        on devices after modifying the model architecture. Without this, modules created 
        during transformations may remain on 'meta' device causing gradient errors like:
        "RuntimeError: expected device meta but got cuda:0"
        """
        logger.info("Re-dispatching model after transformations...")
        
        # First, check for any parameters still on 'meta' device
        # This can happen with device_map="auto" lazy loading
        meta_params = []
        for name, param in self.model.named_parameters():
            if param.device.type == 'meta':
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
                        if 'weight' in name:
                            if param.dim() >= 2:
                                # Use kaiming for weight matrices
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            else:
                                # Small init for 1D weights
                                nn.init.uniform_(param, -0.01, 0.01)
                        elif 'bias' in name:
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
            is_peft = hasattr(self.model, 'base_model')
            
            if is_peft:
                logger.info("Detected PEFT model, working with base model for dispatch")
            
            # Get balanced memory allocation
            max_memory = get_balanced_memory(
                model_to_dispatch,
                max_memory=None,  # Use all available memory
                no_split_module_classes=["MixtureOfBidders", "LoraLayer"],  # Don't split these modules
            )
            
            device_map = infer_auto_device_map(
                model_to_dispatch,
                max_memory=max_memory,
                no_split_module_classes=["MixtureOfBidders", "LoraLayer"],
            )
            
            # Log device distribution
            device_counts = {}
            for module_name, device in device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            logger.info(f"Device map: {device_counts}")
            
            # Dispatch the model with the new device map
            self.model = dispatch_model(model_to_dispatch, device_map=device_map)
            logger.info("Model re-dispatched successfully")
            
        except Exception as e:
            logger.warning(f"Re-dispatch failed ({type(e).__name__}: {e}), falling back to simple device move")
            # Fallback: check if we have meta tensors before calling .to()
            has_meta = any(p.device.type == 'meta' for p in self.model.parameters())
            if has_meta:
                self.model = self.model.to_empty(device=self.device)
                self._reload_pretrained_weights()
            else:
                self.model = self.model.to(self.device)
    
    def _reload_pretrained_weights(self):
        """
        Reload pretrained weights after materializing from meta device.
        
        Uses memory-efficient streaming from safetensors files (<500MB RAM)
        instead of loading the full model (~14GB RAM).
        """
        logger.info("Reloading pretrained weights (streaming from safetensors)...")
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            from safetensors import safe_open
            
            # Get list of safetensor files in the model repo
            repo_files = list_repo_files(self.config.model_id)
            safetensor_files = [f for f in repo_files if f.endswith('.safetensors')]
            
            if not safetensor_files:
                logger.warning("No safetensors files found, falling back to bin files")
                self._reload_pretrained_weights_legacy()
                return
            
            # Build mapping of current model keys (handling PEFT prefix)
            current_state_dict = self.model.state_dict()
            
            # PEFT wraps keys with "base_model.model." prefix
            # Build reverse mapping: original_key -> peft_key
            key_mapping = {}
            for peft_key in current_state_dict.keys():
                # Strip PEFT prefixes to get original key
                original_key = peft_key
                for prefix in ["base_model.model.", "base_model."]:
                    if original_key.startswith(prefix):
                        original_key = original_key[len(prefix):]
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
                        for tensor_name in f.keys():
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
            
            logger.info(f"Reloaded {copied} pretrained weight tensors (skipped {skipped} non-matching)")
            
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
            for peft_key in current_state_dict.keys():
                original_key = peft_key
                for prefix in ["base_model.model.", "base_model."]:
                    if original_key.startswith(prefix):
                        original_key = original_key[len(prefix):]
                        break
                key_mapping[original_key] = peft_key
            
            copied = 0
            for original_key, param in fresh_state_dict.items():
                peft_key = key_mapping.get(original_key)
                if peft_key and peft_key in current_state_dict:
                    if current_state_dict[peft_key].shape == param.shape:
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
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'LayerNorm' in name or 'layernorm' in name:
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
    
    def _setup_data(self):
        """Setup training data loader."""
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
            dataset = load_dataset(
                "c4", 
                "en", 
                split="train", 
                streaming=True
            )
            text_column = "text"
        else:
            dataset = load_dataset(
                self.config.dataset_name, 
                split="train",
                streaming=True,
            )
            text_column = "text"
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
        
        # Process dataset (streaming datasets don't have column_names attribute)
        if hasattr(dataset, 'column_names') and dataset.column_names is not None:
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
            tokenized_dataset,
            batch_size=self.config.batch_size,
            shuffle=True if hasattr(tokenized_dataset, '__len__') else False,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with MoB wealth updates.
        
        This is the core of the TAME training loop:
        1. Forward pass (experts route and process tokens)
        2. Compute per-token loss (provides specialization signal)
        3. Update expert wealth based on loss (key for differentiation!)
        4. Add calibration loss (trains confidence heads)
        5. Backward pass
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        batch_size, seq_len = input_ids.shape
        
        # Forward pass
        outputs = self.model(
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
            reduction='none',
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
                shift_mask
            )
        
        # Compute mean loss for backprop
        valid_mask = (shift_labels != -100) & (shift_mask == 1)
        main_loss = (per_token_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        # =========================================================
        # Add calibration loss for confidence head training
        # =========================================================
        # This teaches confidence heads to predict when they'll do well
        calibration_loss = get_total_calibration_loss(self.model)
        
        # Total loss
        total_loss = main_loss + calibration_loss
        
        # NaN guard: skip backprop if loss is NaN to prevent gradient corruption
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"Step {self.global_step}: NaN/Inf loss detected (main={main_loss.item()}, cal={calibration_loss.item() if isinstance(calibration_loss, torch.Tensor) else 0}), skipping backward")
            return {
                "loss": float('nan'),
                "calibration_loss": 0.0,
                "total_loss": float('nan'),
                "perplexity": float('nan'),
            }
        
        # Scale for gradient accumulation
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        return {
            "loss": main_loss.item(),
            "calibration_loss": calibration_loss.item() if isinstance(calibration_loss, torch.Tensor) else 0.0,
            "total_loss": total_loss.item(),
            "perplexity": math.exp(min(main_loss.item(), 20)),  # Cap to prevent overflow
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Start wealth tracking for analysis
        for mob in get_mob_layers(self.model):
            mob.start_tracking()
        
        # Training loop
        self.model.train()
        accumulated_loss = 0.0
        accumulated_metrics = {"loss": 0.0, "calibration_loss": 0.0, "perplexity": 0.0}
        
        data_iter = iter(self.train_dataloader)
        
        progress_bar = tqdm(range(self.config.max_steps), desc="Training") if HAS_TQDM else range(self.config.max_steps)
        
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
                
                # Log metrics
                avg_metrics = {k: v / self.config.gradient_accumulation_steps 
                              for k, v in accumulated_metrics.items()}
                
                if HAS_TQDM:
                    progress_bar.set_postfix({
                        "loss": f"{avg_metrics['loss']:.4f}",
                        "ppl": f"{avg_metrics['perplexity']:.2f}",
                        "cal": f"{avg_metrics['calibration_loss']:.4f}",
                    })
                
                # Reset accumulated metrics
                accumulated_metrics = {k: 0.0 for k in accumulated_metrics}
            
            # Log wealth statistics
            if step > 0 and step % self.config.log_wealth_frequency == 0:
                self._log_wealth_statistics()
            
            # Save checkpoint
            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)
        
        # Final save
        self._save_checkpoint(self.config.max_steps, final=True)
        
        logger.info("Training complete!")
    
    def _log_wealth_statistics(self):
        """Log MoB wealth distribution statistics."""
        stats = get_mob_statistics(self.model)
        
        if not stats:
            return
        
        # Store for analysis
        self.wealth_history.append({
            "step": self.global_step,
            **{k: v.item() if isinstance(v, torch.Tensor) else v 
               for k, v in stats.items() if k not in ['layer_wealth', 'layer_performance']}
        })
        
        # Log
        logger.info(
            f"Step {self.global_step} | Wealth: mean={stats['mean_wealth']:.2f}, "
            f"std={stats['wealth_std']:.2f}, gini={stats['wealth_gini']:.4f} | "
            f"Performance EMA: {stats['mean_performance']:.4f}"
        )
        
        # Check for healthy specialization
        gini = stats['wealth_gini'].item() if isinstance(stats['wealth_gini'], torch.Tensor) else stats['wealth_gini']
        if gini < 0.05:
            logger.warning("Low Gini coefficient - experts may not be specializing. Check loss feedback.")
        elif gini > 0.7:
            logger.warning("Very high Gini - potential wealth monopoly. Consider increasing min_wealth.")
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint and wealth state."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.config.use_lora and HAS_PEFT:
            self.model.save_pretrained(checkpoint_dir)
        else:
            self.model.save_pretrained(checkpoint_dir)
        
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save MoB state (wealth, performance EMA)
        mob_state = {}
        for idx, mob in enumerate(get_mob_layers(self.model)):
            mob_state[f"layer_{idx}"] = {
                "wealth": mob.expert_wealth.cpu().tolist(),
                "performance_ema": mob.expert_performance_ema.cpu().tolist(),
                "baseline_loss": mob.expert_baseline_loss.cpu().tolist(),
                "usage_count": mob.expert_usage_count.cpu().tolist(),
            }
        
        torch.save(mob_state, checkpoint_dir / "mob_state.pt")
        
        # Save wealth history
        if self.wealth_history:
            torch.save(self.wealth_history, checkpoint_dir / "wealth_history.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train TAME architecture")
    
    # Model arguments (defaults from ACTIVE_MODEL profile)
    parser.add_argument("--model_id", type=str, default=_profile["model_id"],
                       help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="./tame_checkpoints",
                       help="Output directory for checkpoints")
    
    # MoB arguments (defaults from ACTIVE_MODEL profile)
    parser.add_argument("--num_experts", type=int, default=4,
                       help="Number of experts per MoB layer")
    parser.add_argument("--top_k", type=int, default=2,
                       help="Top-k experts to route to")
    parser.add_argument("--mob_layers_start", type=int, default=_profile["mob_layers_start"],
                       help="First layer to apply MoB")
    parser.add_argument("--mob_layers_end", type=int, default=_profile["mob_layers_end"],
                       help="Last layer to apply MoB (exclusive)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for memory-efficient training")
    parser.add_argument("--lora_rank", type=int, default=16)
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name (wikitext, c4, or HuggingFace dataset)")
    
    # Hardware
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["bfloat16", "float16", "float32"])
    
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
    )
    
    # Create trainer and run
    trainer = TAMETrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
