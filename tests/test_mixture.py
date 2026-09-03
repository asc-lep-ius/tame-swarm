import logging
from operator import contains

import pytest
import torch
import torch.nn as nn

from mob import (
    MixtureOfBidders,
    MoBConfig,
    MoBStats,
    apply_mob_to_model,
    get_mob_statistics,
    load_mob_state,
    save_mob_state,
)


class _LayerListModel(nn.Module):
    """A model that is nothing but a list of MoB layers -- get_mob_statistics
    needs a module to walk, not a real forward pass through anything else."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


def test_forward_output_shape(mob_layer, random_hidden_states):
    out = mob_layer(random_hidden_states)
    assert out.shape == random_hidden_states.shape


def test_forward_no_nan_inf(mob_layer, random_hidden_states):
    out = mob_layer(random_hidden_states)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_updates_last_stats(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    stats = mob_layer.last_stats
    assert isinstance(stats, MoBStats)
    assert stats.confidence_logits.shape == (1, 8, mob_layer.config.num_experts)
    assert stats.confidences.shape == (1, 8, mob_layer.config.num_experts)
    assert stats.selected_experts.shape == (1, 8, mob_layer.config.top_k)
    assert stats.routing_weights.shape == (1, 8, mob_layer.config.top_k)
    assert stats.expert_wealth.shape == (mob_layer.config.num_experts,)
    assert stats.expert_usage.shape == (mob_layer.config.num_experts,)
    assert stats.expert_performance.shape == (mob_layer.config.num_experts,)
    assert stats.coupling_metrics is None


def test_forward_records_detached_finite_router_z_loss(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    stats = mob_layer.last_stats
    assert isinstance(stats, MoBStats)
    assert torch.isfinite(stats.router_z_loss)
    assert stats.router_z_loss >= 0.0
    assert not stats.router_z_loss.requires_grad
    assert not stats.confidence_logits.requires_grad
    assert not stats.confidences.requires_grad


def test_router_z_loss_matches_logsumexp_formula_with_default_weight(tiny_config):
    mob = MixtureOfBidders(tiny_config)
    logits = torch.tensor(
        [
            [[0.25, -1.5], [2.0, 0.5]],
            [[-0.75, 1.25], [0.0, -0.25]],
        ],
        dtype=torch.float16,
    )

    assert tiny_config.confidence_z_loss_weight == pytest.approx(0.0001)
    expected = (
        torch.logsumexp(logits.float(), dim=-1).square().mean()
        * tiny_config.confidence_z_loss_weight
    )

    assert torch.allclose(mob._compute_router_z_loss(logits), expected)


def test_mob_stats_does_not_support_mapping_style_access(mob_layer, random_hidden_states):
    mob_layer(random_hidden_states)

    stats = mob_layer.last_stats
    assert isinstance(stats, MoBStats)
    assert stats.expert_wealth.shape == (mob_layer.config.num_experts,)

    with pytest.raises(TypeError):
        contains(stats, "expert_wealth")
    with pytest.raises(TypeError):
        stats["expert_wealth"]
    with pytest.raises(AttributeError):
        stats.keys()


def test_save_load_roundtrip(tmp_path, tiny_config):
    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    mob.train()
    x = torch.randn(1, 8, 32)
    mob(x)
    per_token_loss = torch.randn(1, 8).abs()
    mob.update_wealth_from_loss(per_token_loss)

    original_wealth = mob.expert_wealth.clone()
    original_ema = mob.expert_performance_ema.clone()
    original_baseline = mob.expert_baseline_loss.clone()

    model = FakeModel(mob)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(model, save_path)

    mob2 = MixtureOfBidders(tiny_config)
    model2 = FakeModel(mob2)
    load_mob_state(model2, save_path)

    assert torch.allclose(mob2.expert_wealth, original_wealth, atol=1e-5)
    assert torch.allclose(mob2.expert_performance_ema, original_ema, atol=1e-5)
    assert torch.allclose(mob2.expert_baseline_loss, original_baseline, atol=1e-5)


def test_load_state_strict_mismatch(tmp_path, tiny_config):
    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    model = FakeModel(mob)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(model, save_path)

    different_config = MoBConfig(
        num_experts=4,
        top_k=1,
        hidden_dim=32,
        intermediate_dim=64,
        adapter_rank=4,
        adapter_alpha=4.0,
        use_shared_base=True,
    )
    mob2 = MixtureOfBidders(different_config)
    model2 = FakeModel(mob2)

    with pytest.raises(ValueError, match="Expert count mismatch"):
        load_mob_state(model2, save_path, strict=True)


def test_tracking_records_history(mob_layer, random_hidden_states):
    mob_layer.start_tracking()
    mob_layer(random_hidden_states)
    mob_layer(random_hidden_states)

    history = mob_layer.get_wealth_history()
    assert len(history) >= 2
    mob_layer.stop_tracking()


def test_training_and_eval_produce_same_output(tiny_config):
    mob = MixtureOfBidders(tiny_config)
    hidden = torch.randn(1, 4, tiny_config.hidden_dim)

    mob.train()
    train_out = mob(hidden, update_wealth=False)

    mob.eval()
    eval_out = mob(hidden, update_wealth=False)

    assert torch.allclose(train_out, eval_out, atol=1e-5)


def test_sparse_forward_skips_unselected_experts(tiny_config):
    mob = MixtureOfBidders(tiny_config)
    mob.eval()
    mob.expert_usage_count.zero_()
    hidden = torch.randn(1, 1, tiny_config.hidden_dim)
    mob(hidden, update_wealth=True)
    nonzero_experts = (mob.expert_usage_count > 0).sum().item()
    assert nonzero_experts == tiny_config.top_k


def test_apply_mob_replaces_mlp(tiny_config):
    class FakeFFN(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class FakeTransformerLayer(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim):
            super().__init__()
            self.mlp = FakeFFN(hidden_dim, intermediate_dim)

    class FakeTransformer(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim, num_layers):
            super().__init__()
            self.layers = nn.ModuleList(
                [FakeTransformerLayer(hidden_dim, intermediate_dim) for _ in range(num_layers)]
            )

    class FakeModel(nn.Module):
        def __init__(self, hidden_dim, intermediate_dim, num_layers):
            super().__init__()
            self.model = FakeTransformer(hidden_dim, intermediate_dim, num_layers)

    hd, inter = tiny_config.hidden_dim, tiny_config.intermediate_dim
    fake_model = FakeModel(hd, inter, num_layers=8)

    target_layer = 5
    apply_mob_to_model(fake_model, tiny_config, layers_to_modify=[target_layer])

    replaced = fake_model.model.layers[target_layer].mlp
    assert isinstance(replaced, MixtureOfBidders)

    untouched = fake_model.model.layers[0].mlp
    assert not isinstance(untouched, MixtureOfBidders)


def test_restored_wealth_is_clamped_to_the_configured_bounds(tmp_path, tiny_config, caplog):
    """A checkpoint is the fourth writer to expert_wealth, and the only external one.

    The three update paths clamp; `load_mob_state` did not. The auction divides each
    winner's price by its own wealth, so a restored zero or negative entry reaches
    that division rather than the boundary validation in `MoBConfig` — and a
    negative one small enough to sit inside the payment-negativity tolerance would
    reach that division instead of an invariant; `auction.py` records the window and
    the price it produces.
    """

    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(FakeModel(mob), save_path)

    # A checkpoint that a truncated write, a hand edit, or an older config could
    # plausibly produce: one bankrupt expert, one above the ceiling.
    state = torch.load(save_path, weights_only=True)
    state["layer_0"]["wealth"] = [-1e-4, tiny_config.max_wealth * 10]
    torch.save(state, save_path)

    restored = MixtureOfBidders(tiny_config)
    with caplog.at_level(logging.WARNING, logger="mob.utils"):
        load_mob_state(FakeModel(restored), save_path)

    assert (restored.expert_wealth >= tiny_config.min_wealth).all()
    assert (restored.expert_wealth <= tiny_config.max_wealth).all()
    assert any("repaired" in message for message in caplog.messages), (
        "every other rejection in load_mob_state logs; this one must too"
    )


def test_restored_wealth_repairs_non_finite_entries(tmp_path, tiny_config, caplog):
    """`clamp_` passes NaN through, so a diverged run's checkpoint would survive it.

    That matters most where the guard is thinnest: under `-O` the auction's
    finiteness assert is compiled out, so a NaN wealth would reach the bid silently.
    A non-finite entry is reset to `initial_wealth` rather than to a bound, because
    its true value is unknown rather than extreme.
    """

    class FakeModel(nn.Module):
        def __init__(self, mob):
            super().__init__()
            self.mob = mob

    mob = MixtureOfBidders(tiny_config)
    save_path = str(tmp_path / "mob_state.pt")
    save_mob_state(FakeModel(mob), save_path)

    state = torch.load(save_path, weights_only=True)
    state["layer_0"]["wealth"] = [float("nan"), float("inf")]
    torch.save(state, save_path)

    restored = MixtureOfBidders(tiny_config)
    with caplog.at_level(logging.WARNING, logger="mob.utils"):
        load_mob_state(FakeModel(restored), save_path)

    assert torch.isfinite(restored.expert_wealth).all()
    assert restored.expert_wealth[0].item() == pytest.approx(tiny_config.initial_wealth)
    assert restored.expert_wealth[1].item() == pytest.approx(tiny_config.max_wealth)
    assert any("repaired" in message for message in caplog.messages), (
        "a silent repair is the one thing this function does not do elsewhere"
    )


def _default_shaped_config(hidden_dim: int = 64, **overrides) -> MoBConfig:
    """Default expert count, top_k and wealth band; small enough to run on CPU.

    The gate statistics below are properties of the *report spread*, which
    ``ConfidenceHead``'s initialisation makes insensitive to ``hidden_dim`` by
    construction -- ``xavier_uniform_(gain=0.1)`` gives a logit standard deviation
    proportional to ``sqrt(h) * 1/sqrt(h)``. Measured at this test's own seed, the
    proportional share's top-1 median is 0.518 at the 64 used here and 0.520 at the
    production 4096; across seeds 0-5 the two stay inside [0.517, 0.523].

    ``hidden_dim`` is a named parameter rather than an override so that claim can
    be re-measured from this helper rather than only asserted in its docstring.
    """
    return MoBConfig(
        num_experts=8,
        top_k=2,
        hidden_dim=hidden_dim,
        intermediate_dim=128,
        adapter_rank=8,
        adapter_alpha=8.0,
        **overrides,
    )


@pytest.mark.parametrize("routing_share", ["uniform", "proportional"])
def test_default_gate_mixes_more_than_one_expert(routing_share):
    """``top_k=2`` has to buy two experts' worth of mixing, not one expert and a rounding error.

    Both numbers are the acceptance criteria of #11 stated directly: a median top-1
    weight under 0.9 with fewer than a tenth of tokens above 0.99, and an effective
    expert count above 1.5. Measured at this test's own seed: 0.500/0%/2.000 under
    the uniform share and 0.518/0%/1.997 under the proportional one, against
    0.9896/49.6%/1.275 for the raw-bid gate this replaced -- and 1.000/100%/1.000
    once wealth has spread across the configured band. The raw gate's effective
    expert count runs 1.22-1.28 across seeds 0-4; the other two figures are stable
    to the digits shown.
    """
    torch.manual_seed(0)
    mob = MixtureOfBidders(_default_shaped_config(routing_share=routing_share))
    mob.train()
    mob(torch.randn(4, 64, 64), update_wealth=False)

    routing = mob.last_stats.routing
    assert routing.top1_median.item() < 0.9
    assert routing.top1_saturated_fraction.item() < 0.1
    assert routing.effective_experts.item() > 1.5


def test_routing_temperature_reaches_the_gate_through_the_config():
    """The config field has to arrive at the auctioneer, not just exist.

    Deleting the ``temperature=config.routing_temperature`` line in ``core.py``
    leaves every other test in this suite green -- the plumbing is a single
    assignment and nothing else observes it. This asserts on *realised* routing
    weights rather than on ``mob.auctioneer.temperature``, so it fails if the value
    stops reaching the gate as well as if it stops being stored.

    Measured at this seed: mean top-1 weight 0.582 at ``tau=0.25`` against 0.521 at
    the default 1.0 and 0.505 at 4.0.
    """
    sharpness = []
    for temperature in (0.25, 1.0, 4.0):
        torch.manual_seed(0)
        mob = MixtureOfBidders(
            _default_shaped_config(routing_share="proportional", routing_temperature=temperature)
        )
        mob.train()
        mob(torch.randn(4, 64, mob.config.hidden_dim), update_wealth=False)
        sharpness.append(mob.last_stats.routing.top1_mean.item())

    sharp, plain, flat = sharpness
    assert sharp > plain > flat, (
        f"routing_temperature did not reach the gate: top-1 means {sharpness}"
    )
    # A margin far outside seed noise, so this fails on an unwired config rather
    # than merely on a differently-seeded one.
    assert sharp - plain > 0.02


def test_routing_diagnostics_reach_the_aggregate_statistics():
    """The statistic that would have surfaced #11 has to be visible without a debugger.

    ``get_mob_statistics`` is the surface #5 reads, so the gate's realised sharpness
    belongs in it beside the wealth figures it is meant to be interpreted against.
    """
    config = _default_shaped_config(routing_share="proportional")
    model = _LayerListModel([MixtureOfBidders(config) for _ in range(2)])

    assert "routing_effective_experts" not in get_mob_statistics(model), (
        "a gate statistic must not be reported before every layer has produced one"
    )

    hidden = torch.randn(2, 16, config.hidden_dim)
    for layer in model.layers:
        layer.eval()
        layer(hidden, update_wealth=False)

    statistics = get_mob_statistics(model)
    assert statistics["routing_effective_experts"].item() > 1.5
    assert statistics["routing_top1_mean"].item() < 0.9
    assert statistics["routing_top1_saturated_fraction"].item() < 0.1
    assert len(statistics["layer_routing_top1_median"]) == len(model.layers)


def test_mean_payment_reaches_the_aggregate_statistics_under_the_auction():
    """#7's ``auction/mean_payment`` needs a real number to log, not an absent key.

    #9 was a broken VCG computation that returned identically zero payments and
    went unnoticed because nothing surfaced the number. This is the statistic
    that would have made it visible on day one -- a flat line at zero rather
    than a metric nobody was looking at.
    """
    torch.manual_seed(0)
    config = _default_shaped_config()  # default router: the auction, VCG payments on
    model = _LayerListModel([MixtureOfBidders(config) for _ in range(2)])

    assert "mean_payment" not in get_mob_statistics(model), (
        "no forward has run yet, so there is no payment to report"
    )

    hidden = torch.randn(2, 16, config.hidden_dim)
    for layer in model.layers:
        layer.eval()
        layer(hidden, update_wealth=False)

    statistics = get_mob_statistics(model)
    assert statistics["mean_payment"].item() > 0.0


def test_mean_payment_absent_without_an_economy():
    """The softmax control arm has no auction, so there is no payment to report --
    absent, not a misleading zero. See MoBConfig.has_economy."""
    config = _default_shaped_config(router="softmax")
    model = _LayerListModel([MixtureOfBidders(config) for _ in range(2)])

    hidden = torch.randn(2, 16, config.hidden_dim)
    for layer in model.layers:
        layer.eval()
        layer(hidden, update_wealth=False)

    assert "mean_payment" not in get_mob_statistics(model)


# A nonconstant routing objective. Without genuine competence differences the
# economy has nothing to specialise on, wealth stays flat, and a stationarity test
# passes because nothing moved.
_STATIONARITY_COMPETENCE = torch.tensor([0.9, 0.7, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1])


def _wealth_gini(wealth: torch.Tensor) -> float:
    ordered = torch.sort(wealth)[0]
    n = len(ordered)
    index = torch.arange(1, n + 1, dtype=ordered.dtype)
    return ((2 * (index * ordered).sum()) / (n * ordered.sum()) - (n + 1) / n).abs().item()


@pytest.mark.slow
def test_gate_sharpness_is_stationary_across_a_training_run():
    """The confound #11 exists to remove, checked over a run rather than a batch.

    A single batch cannot see it. The defect was that gate sharpness tracked the
    *absolute* wealth scale, so the failure appears as a slow drift in every number
    read through the gate -- including the Phase 1 coupling ablation, where it would
    be indistinguishable from an effect of the variable under test.

    Note what actually drives the raw gate's drift *here*. Mean wealth has already
    settled by the first probe -- 22.6 at step 500 and 23.1 at step 5000, against a
    52-point fall from ``initial_wealth`` before either -- so this run does not catch
    the raw gate by moving the scale. It catches it because the raw gate multiplies
    the confidence heads' own report drift by that standing scale, and the
    log-domain gate divides it out.

    Measured over five seeds, step 500 against step 5000: this gate moves the top-1
    median by at most 0.0085 and the effective expert count by at most 0.0017 --
    two-fifths and one-sixth of the respective thresholds asserted below -- while the
    raw-bid gate it replaced moves them by up to 0.153 and 0.164 and takes its
    saturated fraction from 0.031 to 0.188. Read those as headroom rather than as
    tight bounds: a 5000-step feedback loop amplifies gate perturbations of order
    1e-7 into per-seed drift differences of order 1e-3, so the exact figures move
    whenever the gate's arithmetic does. That the raw-bid gate is not scale
    invariant at all is established exactly, and far more cheaply, by
    ``test_routing_weights_are_invariant_to_a_uniform_wealth_rescale``; this test is
    here for the part only a run can show.
    """
    torch.manual_seed(0)
    mob = MixtureOfBidders(_default_shaped_config(routing_share="proportional"))
    mob.train()

    probes: dict[int, tuple[float, float, float, float]] = {}
    for step in range(1, 5001):
        mob(torch.randn(2, 16, mob.config.hidden_dim))
        selected = mob._cached_selected_experts
        quality = _STATIONARITY_COMPETENCE[selected].mean(dim=-1)
        mob.update_wealth_from_loss(((2.0 - quality) + 0.05 * torch.randn(2, 16)).abs())

        if step in (500, 5000):
            routing = mob.last_stats.routing
            probes[step] = (
                routing.top1_median.item(),
                routing.effective_experts.item(),
                routing.top1_saturated_fraction.item(),
                _wealth_gini(mob.expert_wealth),
            )

    early, late = probes[500], probes[5000]

    assert abs(late[3] - early[3]) > 0.005, (
        "the economy must have moved between the probes, or this proves nothing"
    )
    assert abs(late[0] - early[0]) < 0.02, (
        f"top-1 median drifted from {early[0]:.4f} to {late[0]:.4f}"
    )
    assert abs(late[1] - early[1]) < 0.01, (
        f"effective expert count drifted from {early[1]:.4f} to {late[1]:.4f}"
    )
    assert early[2] == late[2] == 0.0
