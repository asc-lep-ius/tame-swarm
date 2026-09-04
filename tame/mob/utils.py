import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

from .core import MixtureOfBidders

logger = logging.getLogger(__name__)


def get_mob_layers(model: nn.Module) -> list[MixtureOfBidders]:
    """
    Find all MoB layers in a model.

    Args:
        model: Model that may contain MoB layers

    Returns:
        List of MixtureOfBidders modules
    """
    return [module for module in model.modules() if isinstance(module, MixtureOfBidders)]


@contextmanager
def frozen_economy(model: nn.Module) -> Iterator[None]:
    """Read the model without paying it, then leave the economy exactly as found.

    A held-out evaluation must not be a training step in disguise. ``model.eval()``
    alone is not enough: the inference forward still increments
    ``expert_usage_count`` (which the inference exploration bonus reads), still
    appends to the wealth history when tracking is on, and still overwrites
    ``last_stats`` -- so the training statistics logged after an eval would describe
    the eval batches instead. #12's acceptance criterion is "no wealth updates and
    no steering adaptation", and this is what makes that true rather than assumed.

    Steering needs nothing suppressed, only nothing driven: ``SteeringCoupling``
    keeps no state its forward pass mutates, and its one mutable buffer
    ``_coupling_step`` moves only when ``set_coupling_step`` is called. The
    contract is therefore that the caller does not advance it inside this block --
    the eval loop does not -- and the metrics it caches are restored below.

    Restoring rather than clearing matters: an eval at step N runs between the
    training step that produced ``last_stats`` and the log line that reads them.
    """
    mob_layers = get_mob_layers(model)
    saved = [
        (mob, mob._economy_frozen, mob.last_stats, mob._last_coupling_metrics) for mob in mob_layers
    ]
    for mob in mob_layers:
        mob._economy_frozen = True
    try:
        yield
    finally:
        for mob, was_frozen, stats, coupling_metrics in saved:
            mob._economy_frozen = was_frozen
            mob.last_stats = stats
            mob._last_coupling_metrics = coupling_metrics


def update_all_mob_from_loss(
    model: nn.Module,
    per_token_loss: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    loss_gradient_scale: float = 1.0,
):
    """Settle every MoB layer's economy for the step. Call *after* the loss backward.

    Args:
        model: Model containing MoB layers
        per_token_loss: Loss per token, shape (batch, seq_len)
        token_mask: Optional mask for valid tokens
        loss_gradient_scale: What the backwarded loss's gradient must be multiplied
            by to be the gradient of the summed per-token loss -- ``N x
            accumulation steps`` for a mean over ``N`` valid tokens. See
            ``WealthUpdateMixin.update_wealth_from_loss``.
    """
    for mob in get_mob_layers(model):
        mob.update_wealth_from_loss(per_token_loss, token_mask, loss_gradient_scale)


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


def get_total_router_z_loss(model: nn.Module) -> torch.Tensor:
    """
    Sum live router z-losses from all MoB layers.

    Args:
        model: Model containing MoB layers

    Returns:
        Sum of current non-detached router z-losses from all MoB layers
    """
    total_loss: torch.Tensor | None = None
    for mob in get_mob_layers(model):
        router_z_loss = mob.get_router_z_loss()
        if total_loss is None:
            total_loss = router_z_loss
            continue
        if router_z_loss.device != total_loss.device:
            total_loss = total_loss.to(router_z_loss.device)
        total_loss = total_loss + router_z_loss

    if total_loss is None:
        return torch.tensor(0.0)
    return total_loss


def get_mob_statistics(model: nn.Module) -> dict[str, torch.Tensor | list[torch.Tensor]]:
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
    gini = (2 * torch.sum(torch.arange(1, n + 1, device=flat_wealth.device) * sorted_wealth)) / (
        n * torch.sum(sorted_wealth)
    ) - (n + 1) / n

    statistics: dict[str, torch.Tensor | list[torch.Tensor]] = {
        "mean_wealth": all_wealth.mean(),
        "wealth_std": all_wealth.std(),
        "wealth_gini": gini.abs(),
        "mean_performance": all_performance.mean(),
        "layer_wealth": [mob.expert_wealth.clone() for mob in mob_layers],
        "layer_performance": [mob.expert_performance_ema.clone() for mob in mob_layers],
    }

    # Absent until every layer has forwarded at least once; a partial average over
    # whichever layers happen to have run would be worse than no number at all.
    routing = [mob.last_stats.routing for mob in mob_layers if mob.last_stats is not None]
    if len(routing) == len(mob_layers):
        # Every layer sees the same tokens, so a mean over layers of a per-layer mean
        # or fraction is the pooled quantity. A median is not composable that way, so
        # it is reported per layer rather than averaged into something that looks
        # like a median of the whole model and is not one.
        statistics["routing_top1_mean"] = torch.stack([r.top1_mean for r in routing]).mean()
        statistics["routing_top1_saturated_fraction"] = torch.stack(
            [r.top1_saturated_fraction for r in routing]
        ).mean()
        statistics["routing_effective_experts"] = torch.stack(
            [r.effective_experts for r in routing]
        ).mean()
        statistics["layer_routing_top1_median"] = [r.top1_median for r in routing]
        statistics["layer_routing_effective_experts"] = [r.effective_experts for r in routing]

    # Absent under a gate with no economy (softmax, dense) -- there is no payment
    # to report -- and, same rule as routing above, absent until every layer has
    # forwarded: a partial average would be worse than no number at all. Present
    # and legitimately able to read 0.0 under the auction gate once complete,
    # which is the case #9 existed to make visible: a broken VCG computation
    # reads as a flat zero line instead of an absent metric.
    payments = [
        mob.last_stats.mean_payment
        for mob in mob_layers
        if mob.last_stats is not None and mob.last_stats.mean_payment is not None
    ]
    if len(payments) == len(mob_layers):
        statistics["mean_payment"] = torch.stack(payments).mean()

    # What the last settlement actually paid for. Absent until every layer has
    # settled once, and absent under a gate with no economy. ``mean_win_surplus``
    # is the #15 symptom made visible on every step: below zero, winning is a
    # loss-making trade and the economy is rewarding abstention.
    summaries = [mob.last_value_summary for mob in mob_layers if mob.last_value_summary is not None]
    if len(summaries) == len(mob_layers):
        statistics["mean_realised_value"] = torch.stack(
            [s.mean_realised_value for s in summaries]
        ).mean()
        statistics["mean_report"] = torch.stack([s.mean_report for s in summaries]).mean()
        statistics["mean_win_surplus"] = torch.stack([s.mean_surplus for s in summaries]).mean()

    return statistics


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
    mob_state = torch.load(state_path, map_location="cpu", weights_only=True)
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
                # The three update paths clamp; a restore is the fourth writer and
                # the only one reading from outside the process. The auction divides
                # each price by the winner's own wealth, so a checkpoint carrying a
                # zero or negative entry -- truncated, hand-edited, or written by an
                # older config with different bounds -- would reach that division
                # rather than the boundary validation in MoBConfig.
                # NaN is the reason this is not a bare clamp: clamp_ passes it
                # through, and a diverged run's checkpoint is at least as likely as a
                # hand-edited one. Under -O the finiteness assert in the auction is
                # compiled out, so a NaN wealth would otherwise reach the bid
                # silently. Non-finite entries reset to initial_wealth rather than to
                # a bound, because their true value is unknown rather than extreme.
                mob.expert_wealth.nan_to_num_(
                    nan=mob.config.initial_wealth,
                    posinf=mob.config.max_wealth,
                    neginf=mob.config.min_wealth,
                )
                mob.expert_wealth.clamp_(min=mob.config.min_wealth, max=mob.config.max_wealth)
                repaired = int((mob.expert_wealth != wealth).sum().item())
                non_finite = int((~torch.isfinite(wealth)).sum().item())
                if repaired or non_finite:
                    # The count matters: a whole ledger reset to flat is a different
                    # event from one entry nudged onto a bound, and without it both
                    # produce the same line in a training log.
                    logger.warning(
                        f"{key}: {repaired} of {wealth.numel()} restored wealth values "
                        f"were not usable as saved ({non_finite} non-finite) and have "
                        f"been repaired into [{mob.config.min_wealth}, "
                        f"{mob.config.max_wealth}]; a checkpoint from a different "
                        f"wealth band has its spread flattened onto the current bounds, "
                        f"and a non-finite entry is reset to initial_wealth"
                    )
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
            "adapter_rank": first_mob.config.adapter_rank
            if first_mob.config.use_shared_base
            else None,
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
