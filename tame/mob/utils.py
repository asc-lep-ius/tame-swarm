import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import logging

from .core import MixtureOfBidders

logger = logging.getLogger(__name__)


def get_mob_layers(model: nn.Module) -> List[MixtureOfBidders]:
    """
    Find all MoB layers in a model.

    Args:
        model: Model that may contain MoB layers

    Returns:
        List of MixtureOfBidders modules
    """
    return [
        module
        for module in model.modules()
        if isinstance(module, MixtureOfBidders)
    ]


def update_all_mob_from_loss(
    model: nn.Module,
    per_token_loss: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
):
    """
    Update all MoB layers in a model with loss feedback.

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
        Dictionary with aggregated statistics.
    """
    mob_layers = get_mob_layers(model)
    if not mob_layers:
        return {}

    all_wealth = torch.stack([mob.expert_wealth for mob in mob_layers])
    all_performance = torch.stack([mob.expert_performance_ema for mob in mob_layers])

    flat_wealth = all_wealth.flatten()
    sorted_wealth = torch.sort(flat_wealth)[0]
    n = len(sorted_wealth)
    gini = (
        (2 * torch.sum((torch.arange(1, n + 1, device=flat_wealth.device) * sorted_wealth)))
        / (n * torch.sum(sorted_wealth))
        - (n + 1) / n
    )

    return {
        "mean_wealth": all_wealth.mean(),
        "wealth_std": all_wealth.std(),
        "wealth_gini": gini.abs(),
        "mean_performance": all_performance.mean(),
        "layer_wealth": [mob.expert_wealth.clone() for mob in mob_layers],
        "layer_performance": [mob.expert_performance_ema.clone() for mob in mob_layers],
    }


def load_mob_state(
    model: nn.Module,
    state_path: str,
    strict: bool = False,
    compress_wealth: float = 0.0,
) -> int:
    """
    Load trained MoB state (wealth, performance EMA) into a model.

    Args:
        model: Model containing MoB layers
        state_path: Path to mob_state.pt file
        strict: If True, raise error on config mismatch
        compress_wealth: Compression factor for inference mode (0.0-1.0).

    Returns:
        Number of layers successfully loaded

    Raises:
        ValueError: If strict=True and config mismatch detected
    """
    mob_state = torch.load(state_path, map_location="cpu", weights_only=False)
    mob_layers = get_mob_layers(model)

    if not mob_layers:
        logger.warning("No MoB layers found in model")
        return 0

    saved_config = mob_state.get("_config", {})
    if saved_config:
        saved_experts = saved_config.get("num_experts")
        if saved_experts and mob_layers:
            model_experts = mob_layers[0].config.num_experts
            if saved_experts != model_experts:
                msg = (
                    f"Expert count mismatch: trained with {saved_experts} experts, "
                    f"but model has {model_experts} experts"
                )
                if strict:
                    raise ValueError(msg)
                logger.error(f"CONFIG MISMATCH: {msg}")
                logger.error(
                    "Wealth state will NOT be loaded. Experts will start with default wealth."
                )
                return 0

        saved_layers = saved_config.get("num_layers")
        if saved_layers and saved_layers != len(mob_layers):
            msg = (
                f"Layer count mismatch: trained {saved_layers} MoB layers, "
                f"but model has {len(mob_layers)} MoB layers"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(f"CONFIG MISMATCH: {msg}")
            logger.warning("Will load state for available layers only.")

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

        if "wealth" in state:
            wealth = torch.tensor(state["wealth"], device=device, dtype=mob.expert_wealth.dtype)
            if wealth.shape == mob.expert_wealth.shape:
                mob.expert_wealth.copy_(wealth)
            else:
                logger.warning(
                    f"{key}: wealth shape mismatch "
                    f"(saved: {wealth.shape}, model: {mob.expert_wealth.shape}), skipping"
                )
                skipped += 1
                continue

        if "performance_ema" in state:
            perf = torch.tensor(
                state["performance_ema"], device=device, dtype=mob.expert_performance_ema.dtype
            )
            if perf.shape == mob.expert_performance_ema.shape:
                mob.expert_performance_ema.copy_(perf)

        if "baseline_loss" in state:
            baseline = torch.tensor(
                state["baseline_loss"], device=device, dtype=mob.expert_baseline_loss.dtype
            )
            if baseline.shape == mob.expert_baseline_loss.shape:
                mob.expert_baseline_loss.copy_(baseline)

        if "usage_count" in state:
            usage = torch.tensor(
                state["usage_count"], device=device, dtype=mob.expert_usage_count.dtype
            )
            if usage.shape == mob.expert_usage_count.shape:
                mob.expert_usage_count.copy_(usage)

        loaded += 1

    if loaded > 0:
        logger.info(f"Loaded MoB state for {loaded}/{len(mob_layers)} layers from {state_path}")

        if compress_wealth > 0:
            compress_wealth = min(1.0, max(0.0, compress_wealth))
            logger.info(f"[INFERENCE] Applying wealth compression factor: {compress_wealth:.2f}")

            for mob in mob_layers:
                mean_wealth = mob.expert_wealth.mean()
                mob.expert_wealth.copy_(
                    mob.expert_wealth * (1.0 - compress_wealth) + mean_wealth * compress_wealth
                )

            if mob_layers:
                sample_wealth = mob_layers[0].expert_wealth
                logger.info(
                    f"[INFERENCE] Post-compression wealth (layer 0): "
                    f"min={sample_wealth.min():.1f}, max={sample_wealth.max():.1f}, "
                    f"mean={sample_wealth.mean():.1f}"
                )

            for mob in mob_layers:
                mob.expert_usage_count.zero_()
            logger.info("[INFERENCE] Reset usage counts for fair exploration bonus")

    if skipped > 0:
        logger.warning(f"Skipped {skipped} layers due to missing/mismatched state")

    return loaded


def save_mob_state(model: nn.Module, save_path: str) -> bool:
    """
    Save MoB state (wealth, performance EMA) to file.

    Args:
        model: Model containing MoB layers
        save_path: Path to save mob_state.pt

    Returns:
        True if successful
    """
    mob_layers = get_mob_layers(model)
    if not mob_layers:
        return False

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
