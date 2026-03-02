#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    missing = []
    
    try:
        import torch
        logger.info("PyTorch %s", torch.__version__)
        if torch.cuda.is_available():
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
            logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        else:
            logger.info("CUDA not available, will use CPU (slow)")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        logger.info("Transformers %s", transformers.__version__)
    except ImportError:
        missing.append("transformers")
    
    try:
        import datasets
        logger.info("Datasets %s", datasets.__version__)
    except ImportError:
        missing.append("datasets")
    
    try:
        import peft
        logger.info("PEFT %s", peft.__version__)
    except ImportError:
        logger.warning("PEFT not installed (LoRA disabled)")
    
    try:
        from mob import MoBConfig, MixtureOfBidders
        logger.info("MoB module loaded")
    except ImportError as e:
        logger.error("MoB module import failed: %s", e)
        missing.append("mob (local)")
    
    try:
        from config import ACTIVE_MODEL, MODEL_PROFILES
        profile = MODEL_PROFILES[ACTIVE_MODEL]
        logger.info("Active model: %s (%s)", ACTIVE_MODEL, profile['model_id'])
    except ImportError as e:
        logger.warning("Could not load model profile: %s", e)
    
    if missing:
        logger.error("Missing dependencies: %s", ', '.join(missing))
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    return True


def run_training(args):
    from train import TrainingConfig, TAMETrainer
    
    logger.info("TAME TRAINING")
    
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
        max_steps=args.steps,
        warmup_steps=min(args.steps // 10, 500),
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        dataset_name=args.dataset,
        dtype=args.dtype,
        save_steps=max(args.steps // 5, 100),
        log_frequency=max(args.steps // 50, 10),
    )
    
    logger.info(
        "Configuration: model=%s dataset=%s steps=%d batch=%dx%d "
        "mob=%d experts top-%d layers %d-%d lora=%s output=%s",
        config.model_id, config.dataset_name, config.max_steps,
        config.batch_size, config.gradient_accumulation_steps,
        config.num_experts, config.top_k,
        config.mob_layers_start, config.mob_layers_end,
        "enabled" if config.use_lora else "disabled",
        config.output_dir,
    )
    
    trainer = TAMETrainer(config)
    
    logger.info("Initializing model...")
    trainer.setup()
    
    logger.info("Starting training...")
    trainer.train()
    
    # Find latest checkpoint
    checkpoints = sorted(Path(config.output_dir).glob("checkpoint-*"))
    if checkpoints:
        latest = checkpoints[-1]
        logger.info("Training complete. Latest checkpoint: %s", latest)
        return str(latest)
    
    return None


def export_for_inference(checkpoint_path: str, export_dir: str = "./tame_inference"):
    import torch
    
    checkpoint_path = Path(checkpoint_path)
    export_dir = Path(export_dir)
    
    logger.info("EXPORTING FOR INFERENCE")
    
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return False
    
    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Copying model from %s...", checkpoint_path)
    model_files = [
        "config.json",
        "generation_config.json", 
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    
    copied = 0
    for fname in model_files:
        src = checkpoint_path / fname
        if src.exists():
            shutil.copy2(src, export_dir / fname)
            copied += 1
    
    logger.info("Copied %d model files", copied)
    
    # Copy MoB state
    mob_state_path = checkpoint_path / "mob_state.pt"
    if mob_state_path.exists():
        shutil.copy2(mob_state_path, export_dir / "mob_state.pt")
        logger.info("Copied mob_state.pt")
        
        mob_state = torch.load(mob_state_path, map_location="cpu", weights_only=True)
        logger.info("MoB Wealth Summary:")
        for layer_key, state in mob_state.items():
            wealth = state.get("wealth", [])
            if wealth:
                logger.info(
                    "%s: min=%.1f, max=%.1f, spread=%.1f",
                    layer_key, min(wealth), max(wealth), max(wealth) - min(wealth),
                )
    else:
        logger.warning("No mob_state.pt found (wealth will start fresh)")
    
    # Copy training state for reference
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        shutil.copy2(training_state_path, export_dir / "training_state.pt")
        logger.info("Copied training_state.pt")
    
    # Create inference config
    inference_config = {
        "source_checkpoint": str(checkpoint_path),
        "export_time": datetime.now().isoformat(),
        "usage": "Load with: model = load_tame_model('./tame_inference')",
    }
    
    with open(export_dir / "inference_config.json", "w") as f:
        json.dump(inference_config, f, indent=2)
    
    logger.info("Export complete: %s", export_dir)
    logger.info("To use in main.py, update MODEL_PATH to: %s", export_dir.absolute())
    
    # Generate loader code snippet
    loader_code = f'''
# Add this to main.py to load trained MoB state:

def load_mob_state(model, mob_state_path="{export_dir}/mob_state.pt"):
    """Load trained MoB wealth state into model."""
    import torch
    from mob import get_mob_layers
    
    mob_state = torch.load(mob_state_path, map_location="cpu", weights_only=True)
    mob_layers = get_mob_layers(model)
    
    for idx, mob in enumerate(mob_layers):
        key = f"layer_{{idx}}"
        if key in mob_state:
            state = mob_state[key]
            mob.expert_wealth.copy_(torch.tensor(state["wealth"]))
            mob.expert_performance_ema.copy_(torch.tensor(state["performance_ema"]))
            mob.expert_baseline_loss.copy_(torch.tensor(state["baseline_loss"]))
    
    print(f"Loaded MoB state for {{len(mob_layers)}} layers")
'''
    
    with open(export_dir / "loader_snippet.py", "w") as f:
        f.write(loader_code)
    
    logger.info("Loader code saved to: %s/loader_snippet.py", export_dir)
    
    return True


def run_test(args):
    logger.info("QUICK TEST (100 steps)")
    
    args.steps = 100
    args.batch_size = 2
    args.gradient_accumulation_steps = 1
    args.output_dir = "./tame_test_run"
    
    checkpoint = run_training(args)
    
    if checkpoint:
        logger.info("Test completed successfully")
        logger.info("Checkpoint: %s", checkpoint)
        
        if args.cleanup:
            shutil.rmtree(args.output_dir, ignore_errors=True)
            logger.info("Cleaned up test files")
    else:
        logger.error("Test failed")
        return False
    
    return True


def main():
    # Import model profiles from train.py to use consistent defaults
    from config import ACTIVE_MODEL, MODEL_PROFILES, get_active_profile
    _profile = get_active_profile()
    
    parser = argparse.ArgumentParser(
        description="TAME Setup - Train and deploy MoB model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Active Model: {ACTIVE_MODEL} ({_profile['model_id']})
(Change by editing ACTIVE_MODEL in train.py)

Examples:
  # Quick test (verify setup)
  python setup_tame.py --mode test
  
  # Full training (5000 steps, ~2-4 hours on A100)
  python setup_tame.py --mode train --steps 5000
  
  # Train with LoRA (less memory)
  python setup_tame.py --mode train --steps 5000 --use_lora
  
  # Export for inference
  python setup_tame.py --mode export --checkpoint ./tame_checkpoints/checkpoint-5000
  
  # Full pipeline
  python setup_tame.py --mode full --steps 5000
        """
    )
    
    # Mode
    parser.add_argument("--mode", type=str, default="test",
                       choices=["test", "train", "export", "full", "check"],
                       help="Operation mode")
    
    # Model - default from ACTIVE_MODEL profile
    parser.add_argument("--model_id", type=str, 
                       default=_profile["model_id"],
                       help=f"HuggingFace model ID (default from ACTIVE_MODEL: {ACTIVE_MODEL})")
    
    # Training (defaults optimized for 16GB VRAM - RTX 5070 Ti, RTX 4080, etc.)
    parser.add_argument("--steps", type=int, default=5000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size per step (default: 2 for 16GB GPU)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation (default: 8, effective batch=16)")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Max sequence length (default: 512 for 16GB GPU)")
    
    # MoB - defaults from ACTIVE_MODEL profile
    parser.add_argument("--num_experts", type=int, default=4,
                       help="Number of experts (must match main.py)")
    parser.add_argument("--top_k", type=int, default=2,
                       help="Experts per token (must match main.py)")
    parser.add_argument("--mob_layers_start", type=int, default=_profile["mob_layers_start"],
                       help="First MoB layer (auto-configured from ACTIVE_MODEL)")
    parser.add_argument("--mob_layers_end", type=int, default=_profile["mob_layers_end"],
                       help="Last MoB layer exclusive (auto-configured from ACTIVE_MODEL)")
    
    # Options
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for memory-efficient training")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="./tame_checkpoints")
    
    # Export
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path for export mode")
    parser.add_argument("--export_dir", type=str, default="./tame_inference")
    
    # Misc
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up test files after test mode")
    
    args = parser.parse_args()
    
    logger.info("TAME SETUP - Multi-Scale Competency Architecture")
    
    if not check_dependencies():
        if args.mode != "check":
            sys.exit(1)
        return
    
    if args.mode == "check":
        logger.info("All dependencies OK")
        return
    
    # Execute mode
    if args.mode == "test":
        success = run_test(args)
        sys.exit(0 if success else 1)
    
    elif args.mode == "train":
        checkpoint = run_training(args)
        if not checkpoint:
            sys.exit(1)
        logger.info("Training complete")
        logger.info("Next step: python setup_tame.py --mode export --checkpoint %s", checkpoint)
    
    elif args.mode == "export":
        if not args.checkpoint:
            # Find latest checkpoint
            checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
            if checkpoints:
                args.checkpoint = str(checkpoints[-1])
                logger.info("Using latest checkpoint: %s", args.checkpoint)
            else:
                logger.error("No checkpoint specified and none found")
                logger.error("Run training first or specify --checkpoint")
                sys.exit(1)
        
        success = export_for_inference(args.checkpoint, args.export_dir)
        sys.exit(0 if success else 1)
    
    elif args.mode == "full":
        # Train + Export
        checkpoint = run_training(args)
        if not checkpoint:
            sys.exit(1)
        
        success = export_for_inference(checkpoint, args.export_dir)
        
        if success:
            logger.info("SETUP COMPLETE")
            logger.info("Trained model ready at: %s", args.export_dir)
            logger.info("To start inference server:")
            logger.info("1. Update MODEL_PATH in main.py to: %s", Path(args.export_dir).absolute())
            logger.info("2. Run: python main.py")
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
