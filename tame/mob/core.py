import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..coupling import CouplingMetrics, SteeringCoupling, SteeringCouplingConfig
except ImportError:
    if __package__ != "mob":
        raise
    from coupling import CouplingMetrics, SteeringCoupling, SteeringCouplingConfig

from .auction import (
    ROUTING_SHARE_PROPORTIONAL,
    AuctionOutcome,
    RoutingDiagnostics,
    VCGAuctioneer,
    routing_diagnostics,
)
from .experts import ConfidenceHead, Expert, LightweightExpert
from .mob_config import MoBConfig
from .softmax_router import SoftmaxRouter
from .wealth import ValueSummary, WealthUpdateMixin, realised_values

logger = logging.getLogger(__name__)


def _is_recomputing() -> bool:
    """Whether this forward is a gradient-checkpoint recompute, not the real one.

    ``torch.utils.checkpoint`` drops a checkpointed region's activations and runs
    the region's forward a second time, inside the backward, to rebuild them. The
    autograd engine reports a current graph task only while a backward is
    executing, so a forward that finds one is that second run. The private
    accessor is the one the checkpoint implementation itself keys its recompute
    bookkeeping on.
    """
    return torch._C._current_graph_task_id() != -1  # pyright: ignore[reportAttributeAccessIssue]


def _keep(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@contextmanager
def _saved_outside_checkpointing() -> Iterator[None]:
    """Keep the routing path's saved tensors out of gradient checkpointing.

    The value objective and the router z-loss are backwarded on their own, after
    the language-modelling backward has run and the economy has settled -- they
    cannot join the first backward, because the value they regress onto is the
    gradient that backward delivers. Under non-reentrant checkpointing every tensor
    saved inside the decoder layer is a placeholder that recomputes the *whole*
    layer on unpack, so that second backward would cost an attention pass per MoB
    layer to recover one detached activation. ``saved_tensors_hooks`` nests and the
    innermost pair wins, so an identity pair here saves the routing path's few
    activations eagerly. The cost is one ``(batch, seq, hidden)`` activation per
    layer held until the auxiliary backward.
    """
    with torch.autograd.graph.saved_tensors_hooks(_keep, _keep):
        yield


@dataclass(frozen=True)
class MoBStats:
    confidence_logits: torch.Tensor
    confidences: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    expert_wealth: torch.Tensor
    expert_usage: torch.Tensor
    expert_performance: torch.Tensor
    router_z_loss: torch.Tensor
    routing: RoutingDiagnostics
    coupling_metrics: CouplingMetrics | None = None
    # None under a gate with no economy (softmax, dense): there is no payment to
    # report, as opposed to a payment that happens to be zero. See #9.
    mean_payment: torch.Tensor | None = None


# Accumulated economic state. A transfer is of order 1e-2 credits against a
# wealth of order 1e2, which is below bfloat16's resolution there (0.5 at 83.5):
# measured on Qwen3-1.7B under bf16 over 120 steps, the wealth vector never moved
# and its standard deviation read exactly 0.0. These stay float32 whatever dtype
# the rest of the layer runs in; the auction reads them into the bid dtype itself.
LEDGER_BUFFERS = (
    "expert_wealth",
    "expert_usage_count",
    "expert_baseline_loss",
    "expert_performance_ema",
)


class MixtureOfBidders(WealthUpdateMixin, nn.Module):
    def _apply(self, fn, recurse=True):  # type: ignore[override]
        """Move the layer as usual, then put the ledgers back in float32.

        ``to(dtype)`` converts every floating buffer along with the parameters,
        and nothing in the public API distinguishes a buffer that holds
        activations-scale state from one that holds a running total. Recasting
        after the fact is the one place that distinction can be made.
        """
        module = super()._apply(fn, recurse)
        for name in LEDGER_BUFFERS:
            ledger = getattr(self, name)
            if ledger.dtype != torch.float32:
                setattr(self, name, ledger.float())
        return module

    def __init__(self, config: MoBConfig):
        super().__init__()
        self.config = config
        self.use_shared_base = config.use_shared_base

        if self.use_shared_base:
            self.base_gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.base_up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.base_down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

            self.experts = nn.ModuleList(
                [
                    LightweightExpert(
                        config.hidden_dim,
                        config.intermediate_dim,
                        rank=config.adapter_rank,
                        alpha=config.adapter_alpha,
                    )
                    for _ in range(config.num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [
                    Expert(config.hidden_dim, config.intermediate_dim)
                    for _ in range(config.num_experts)
                ]
            )

        self.confidence_heads = nn.ModuleList(
            [ConfidenceHead(config.hidden_dim, expert_id=i) for i in range(config.num_experts)]
        )

        # One attribute for whichever gate this arm runs, so nothing downstream
        # branches on which one it is except the code that has to.
        self.gate: nn.Module = (
            VCGAuctioneer(
                config.num_experts,
                config.top_k,
                differentiable=config.use_differentiable_routing,
                routing_share=config.routing_share,
                temperature=config.routing_temperature,
                exploration_rate=config.exploration_rate,
            )
            if config.has_economy
            else SoftmaxRouter(config.num_experts, config.top_k)
        )

        self.register_buffer(
            "expert_wealth",
            torch.full((config.num_experts,), config.initial_wealth),
        )

        self.register_buffer(
            "expert_usage_count",
            torch.zeros(config.num_experts),
        )

        self.register_buffer(
            "expert_baseline_loss",
            torch.ones(config.num_experts),
        )

        self.register_buffer(
            "expert_performance_ema",
            torch.zeros(config.num_experts),
        )

        self.last_stats: MoBStats | None = None

        self.wealth_history: list[list[float]] = []
        self._track_wealth: bool = False

        self._cached_selected_experts: torch.Tensor | None = None
        self._cached_routing_weights: torch.Tensor | None = None
        self._cached_confidences: torch.Tensor | None = None
        # Kept attached to the graph so update_wealth_from_loss can build each
        # expert's value objective; the detached copy above is what the wealth
        # arithmetic reads. Cleared as soon as the objective is built.
        self._live_confidences: torch.Tensor | None = None
        self._cached_payments: torch.Tensor | None = None
        self._cached_rebates: torch.Tensor | None = None
        self._cached_explored: torch.Tensor | None = None
        # Written by the hook the loss backward fires at this layer's output: the
        # value every winner realised on every token. Read by the wealth update,
        # which therefore has to run after the backward.
        self._cached_values: torch.Tensor | None = None
        self._loss_feedback_pending: bool = False
        self._cached_calibration_loss: torch.Tensor | None = None
        self._cached_router_z_loss: torch.Tensor | None = None
        self.last_value_summary: ValueSummary | None = None
        self.last_realised_values: torch.Tensor | None = None
        self._warned: set[str] = set()
        self._last_coupling_metrics: CouplingMetrics | None = None
        # Held-out evaluation reads the model without paying it. See
        # ``mob.utils.frozen_economy``, which is the only thing that sets this.
        self._economy_frozen: bool = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        update_wealth: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the MoB layer."""
        # Under gradient checkpointing this forward runs a second time inside the
        # backward, to rebuild the activations the first run dropped. That pass
        # must reproduce the first one's tensors and otherwise leave no trace: the
        # caches below belong to the real forward, the usage counts would double,
        # and an auction re-run against wealth the economy had moved in between is
        # what raised a checkpoint metadata mismatch at step 0 of every 8-expert
        # run while the wealth update sat between forward and backward.
        recomputing = _is_recomputing()

        if not recomputing:
            # Every cache belongs to the forward pass that produced it. The value
            # objective holds a live graph, so a stale one is not a harmless
            # constant the way the old detached calibration loss was -- a training
            # step that forwards without settling would backward through a graph
            # the previous step already freed.
            self._cached_calibration_loss = None
            self._live_confidences = None
            self._cached_values = None

        # The routing path observes the representation; it does not reshape it. Every
        # head reads the same hidden states, so without this detach each expert's
        # private value objective backpropagates into the backbone that every other
        # expert reads -- a shared auxiliary loss, which is the central planner the
        # auction exists to replace. Detaching the input rather than the coupling
        # output keeps SteeringCoupling.projection trainable.
        routing_hidden_states = hidden_states.detach()

        # The auxiliary objectives -- each head's value objective and the router
        # z-loss -- are backwarded on their own after the economy has settled, so
        # they need a graph the language-modelling backward does not free. Under
        # the uniform auction share the LM loss cannot reach the heads at all, so
        # the routing reports carry no graph and the one graph built is theirs.
        # Under a gate that is differentiable in the reports -- the softmax control
        # arm, the proportional baseline share -- the LM backward runs through the
        # routing reports, and the auxiliary objectives read a second, separate
        # pass over the same heads: one more linear per head, no more. The second
        # pass cannot be skipped on those arms even though the first carries a
        # graph: the LM backward frees that graph, and the z-loss is backwarded
        # after it.
        needs_auxiliary = self.training and torch.is_grad_enabled() and not recomputing
        live_confidences: torch.Tensor | None = None
        with _saved_outside_checkpointing():
            with torch.set_grad_enabled(torch.is_grad_enabled() and self._gate_trains_heads()):
                confidence_logits, confidences, coupling_metrics = self._report(
                    routing_hidden_states
                )
            if needs_auxiliary:
                live_logits, live_confidences, _ = self._report(routing_hidden_states)
                router_z_loss = self._compute_router_z_loss(live_logits)
            else:
                router_z_loss = self._compute_router_z_loss(confidence_logits).detach()
        if not recomputing:
            self._cached_router_z_loss = router_z_loss

        outcome = self._route(confidence_logits, confidences)
        selected_experts = outcome.selected_experts
        routing_weights = outcome.routing_weights

        # A frozen economy still routes and still computes an output -- it is an
        # evaluation, not an ablation -- but it must not move any state the next
        # training step reads. That is wealth, the usage counts the exploration
        # bonus reads, and the loss-feedback cache.
        update_wealth = update_wealth and not self._economy_frozen and not recomputing

        expects_feedback = (
            self.training
            and self.config.use_loss_feedback
            and self._economy_live()
            and not recomputing
        )
        collect_contributions = expects_feedback and torch.is_grad_enabled()

        output = torch.zeros_like(hidden_states)
        contributions: torch.Tensor | None = None
        if self.training:
            output, contributions = self._forward_training(
                hidden_states,
                output,
                selected_experts,
                routing_weights,
                update_wealth,
                collect_contributions,
            )
        else:
            output = self._forward_inference(
                hidden_states, output, selected_experts, routing_weights, update_wealth
            )

        if output.dtype == torch.bfloat16 or output.dtype == torch.float16:
            output = torch.nan_to_num(output, nan=0.0, posinf=65000.0, neginf=-65000.0)

        if collect_contributions:
            assert contributions is not None and live_confidences is not None
            self._cache_loss_feedback(outcome, confidences, live_confidences)
            self._register_value_hook(output, contributions)
        elif expects_feedback:
            self._warn_once(
                "no_gradient",
                "training forward ran with gradients disabled, so the loss backward "
                "cannot reach this layer and no value can be realised; loss feedback "
                "is skipped for this step. Reentrant gradient checkpointing runs its "
                "first pass this way and is not supported",
            )

        # The loss path settles in update_wealth_from_loss. The other two paths
        # are fallbacks for when no loss reaches the layer, and settle here.
        if update_wealth and self._economy_live() and not self.config.use_loss_feedback:
            if self.config.use_local_quality:
                self._update_wealth_local_quality(
                    selected_experts,
                    routing_weights,
                    confidences,
                    outcome.payments,
                    outcome.rebates,
                    output,
                )
            elif self.training:
                self._update_wealth_participation(
                    selected_experts,
                    routing_weights,
                    confidences,
                    outcome.payments,
                    outcome.rebates,
                )

        if not recomputing:
            self.last_stats = MoBStats(
                confidence_logits=confidence_logits.detach(),
                confidences=confidences.detach(),
                selected_experts=selected_experts.detach(),
                routing_weights=routing_weights.detach(),
                expert_wealth=self.expert_wealth.detach().clone(),
                expert_usage=self.expert_usage_count.detach().clone(),
                expert_performance=self.expert_performance_ema.detach().clone(),
                router_z_loss=router_z_loss.detach(),
                # The statistic that would have surfaced a gate saturating on the
                # wealth scale, so it is recorded on every step rather than reached
                # for after a result looks wrong. Left as device tensors; the
                # training loop syncs.
                routing=routing_diagnostics(routing_weights),
                coupling_metrics=coupling_metrics,
                mean_payment=(
                    outcome.payments.detach().mean() if outcome.payments is not None else None
                ),
            )
            self._last_coupling_metrics = coupling_metrics

            # A frozen forward contributes no row: the wealth history is a training
            # trace, and an evaluation pass interleaved into it would read as the
            # economy having done something on a step where it did not.
            if self._track_wealth and not self._economy_frozen:
                self.wealth_history.append(self.expert_wealth.cpu().tolist())

        return output

    def _report(
        self, routing_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, CouplingMetrics | None]:
        """Every head's logit and report for every token.

        Must match ConfidenceHead.forward. The report is a loss-reduction estimate,
        not a probability: a sigmoid here caps it at 1.0 while the reward it is
        trained to predict is unbounded, so "win when report > price" and "profit
        when value > price" stop being the same threshold. forward_logits is used
        rather than forward only so the z-loss can read the pre-activation logits.
        """
        confidence_hidden_states = routing_hidden_states
        coupling_metrics: CouplingMetrics | None = None
        coupling = self._get_coupling()
        if coupling is not None:
            confidence_hidden_states = coupling(routing_hidden_states)
            coupling_metrics = coupling.last_metrics

        confidence_logits = torch.stack(
            [
                cast(ConfidenceHead, confidence_head)
                .forward_logits(confidence_hidden_states)
                .squeeze(-1)
                for confidence_head in self.confidence_heads
            ],
            dim=-1,
        )
        return confidence_logits, F.softplus(confidence_logits), coupling_metrics

    def _gate_trains_heads(self) -> bool:
        """Whether the language-modelling loss reaches the heads through the gate.

        The softmax control arm and the proportional baseline share are
        differentiable in the reports, so the LM loss trains their heads -- the
        documented asymmetry those baselines exist to isolate. The uniform auction
        share is a constant, and its reports are computed without a graph so that
        no route, however indirect, lets the global loss into a head.
        """
        if not self.config.has_economy:
            return True
        return self.config.routing_share == ROUTING_SHARE_PROPORTIONAL

    def _cache_loss_feedback(
        self, outcome: AuctionOutcome, confidences: torch.Tensor, live_confidences: torch.Tensor
    ) -> None:
        self._cached_selected_experts = outcome.selected_experts.detach()
        self._cached_routing_weights = outcome.routing_weights.detach()
        self._cached_confidences = confidences.detach()
        self._live_confidences = live_confidences
        self._cached_payments = outcome.payments.detach() if outcome.payments is not None else None
        self._cached_rebates = outcome.rebates.detach() if outcome.rebates is not None else None
        self._cached_explored = outcome.explored
        self._loss_feedback_pending = True

    def _register_value_hook(self, output: torch.Tensor, contributions: torch.Tensor) -> None:
        """Arrange for the loss backward to tell this layer what each winner was worth.

        The hook fires with the gradient of whatever loss was backwarded, at this
        layer's output. Against each winner's contribution that is the first-order
        change in loss the winner caused -- see ``realised_values``. The
        contributions are held only by this closure, so they are released the
        moment the backward has read them -- and until then they are held in
        full, ``(batch, seq, top_k, hidden)`` in the model dtype per layer, which
        gradient checkpointing cannot drop because nothing saved them: about 16 MB
        per layer at batch 2, sequence 1024, hidden 2048, top-2 in bf16.
        """
        if not output.requires_grad:
            self._warn_once(
                "frozen",
                "no trainable parameter reaches this layer's output, so the loss "
                "gradient never arrives here and no expert can realise a value; the "
                "economy will not move",
            )
            return

        def capture(output_gradient: torch.Tensor) -> None:
            with torch.no_grad():
                self._cached_values = realised_values(contributions, output_gradient)

        output.register_hook(capture)

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message)

    def _route(self, confidence_logits: torch.Tensor, confidences: torch.Tensor) -> AuctionOutcome:
        """Turn reports into an allocation, by whichever gate this arm configures.

        The auction bids ``softplus(logits) x wealth``; the #12 control arm gates on
        the logits themselves. The two take different inputs because they mean
        different things by a report -- a bid is a value in loss-reduction units,
        a router logit is an unnormalised score -- so this dispatches rather than
        forcing one signature onto both.
        """
        if self.config.has_economy:
            return cast(VCGAuctioneer, self.gate)(confidences, self.expert_wealth)
        return cast(SoftmaxRouter, self.gate)(confidence_logits)

    def _economy_live(self) -> bool:
        """Whether this forward may move economic state.

        Two independent reasons it may not: the arm has no economy at all
        (``router="softmax"``), or a held-out evaluation has frozen it.
        """
        return self.config.has_economy and not self._economy_frozen

    def attach_coupling(
        self,
        steering_direction: torch.Tensor,
        config: SteeringCouplingConfig | None = None,
    ) -> SteeringCoupling:
        coupling_config = config or SteeringCouplingConfig(hidden_dim=self.config.hidden_dim)
        if coupling_config.hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"Coupling hidden_dim {coupling_config.hidden_dim} does not match "
                f"MoB hidden_dim {self.config.hidden_dim}"
            )

        if hasattr(self, "coupling"):
            self.detach_coupling()

        coupling = SteeringCoupling(coupling_config, steering_direction)
        reference_parameter = next(self.confidence_heads.parameters())
        coupling.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        self.add_module("coupling", coupling)
        self._last_coupling_metrics = None
        return coupling

    def detach_coupling(self) -> None:
        if hasattr(self, "coupling"):
            delattr(self, "coupling")
        self._last_coupling_metrics = None
        self.last_stats = None

    def get_router_z_loss(self) -> torch.Tensor:
        if self._cached_router_z_loss is None:
            return torch.tensor(0.0, device=self.expert_wealth.device)
        return self._cached_router_z_loss

    def set_coupling_step(self, step: int) -> "MixtureOfBidders":
        if step < 0:
            raise ValueError("coupling step must be non-negative")

        coupling = self._get_coupling()
        if coupling is not None:
            coupling.set_coupling_step(step)
        return self

    def _get_coupling(self) -> SteeringCoupling | None:
        return cast(SteeringCoupling | None, getattr(self, "coupling", None))

    def routing_parameters(self) -> list[nn.Parameter]:
        """Parameters trained by nothing but the value objective, at the heads' rate.

        The confidence heads, and the coupling's receptor when one is attached: it
        shifts what the heads see, is zero-initialised like them, and at the
        backbone's learning rate would never leave its initialisation either.
        """
        parameters = list(self.confidence_heads.parameters())
        coupling = self._get_coupling()
        if coupling is not None:
            parameters.extend(coupling.parameters())
        return parameters

    def _compute_router_z_loss(self, confidence_logits: torch.Tensor) -> torch.Tensor:
        log_z = torch.logsumexp(confidence_logits.float(), dim=-1)
        return log_z.square().mean() * self.config.confidence_z_loss_weight

    def _expert_forward(
        self, expert_idx: int, expert_input: torch.Tensor, with_contribution: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one expert on its tokens, optionally beside its contribution.

        The contribution is what the expert's output differs from the tissue
        default by, and it is what the economy prices. Under the shared base the
        default is the base FFN, so the contribution is exactly what the adapters
        add -- small, which is what makes the first-order value estimate accurate.
        A full expert has no base to fall back to: the empty slot is its reference,
        so its whole output is its contribution, a far larger perturbation for
        which the same estimate is correspondingly coarser.
        """
        if self.use_shared_base:
            expert = cast(LightweightExpert, self.experts[expert_idx])
            if with_contribution:
                expert_output, reference = expert.forward_with_reference(
                    expert_input, self.base_gate_proj, self.base_up_proj, self.base_down_proj
                )
                return expert_output, expert_output.detach() - reference
            expert_output = expert(
                expert_input, self.base_gate_proj, self.base_up_proj, self.base_down_proj
            )
            return expert_output, None

        expert_output = self.experts[expert_idx](expert_input)
        return expert_output, expert_output.detach() if with_contribution else None

    def _forward_training(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        update_wealth: bool,
        collect_contributions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_output = output.reshape(-1, hidden_dim)
        flat_contributions = (
            torch.zeros(
                flat_hidden.shape[0],
                self.config.top_k,
                hidden_dim,
                device=output.device,
                dtype=output.dtype,
            )
            if collect_contributions
            else None
        )

        for k in range(self.config.top_k):
            flat_expert_indices = selected_experts[:, :, k].reshape(-1)
            flat_weights = routing_weights[:, :, k].reshape(-1)

            for expert_idx in range(self.config.num_experts):
                mask = flat_expert_indices == expert_idx
                if not mask.any():
                    continue

                expert_output, contribution = self._expert_forward(
                    expert_idx, flat_hidden[mask], collect_contributions
                )
                weighted = expert_output * flat_weights[mask].unsqueeze(-1)
                token_indices = mask.nonzero(as_tuple=False).squeeze(-1)
                flat_output.index_add_(0, token_indices, weighted)

                if flat_contributions is not None:
                    assert contribution is not None
                    flat_contributions[token_indices, k] = contribution.to(flat_contributions.dtype)

                if update_wealth:
                    self.expert_usage_count[expert_idx] += mask.sum().float()

        contributions = (
            None
            if flat_contributions is None
            else flat_contributions.reshape(batch_size, seq_len, self.config.top_k, hidden_dim)
        )
        return flat_output.reshape_as(hidden_states), contributions

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        update_wealth: bool,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]

        for k in range(self.config.top_k):
            expert_indices = selected_experts[:, :, k]
            weights = routing_weights[:, :, k : k + 1]

            for expert_idx in range(self.config.num_experts):
                mask = expert_indices == expert_idx
                if not mask.any():
                    continue

                expert_output, _ = self._expert_forward(expert_idx, hidden_states[mask], False)

                weight_vals = weights.squeeze(-1)[mask]
                weighted_expert_output = expert_output * weight_vals.unsqueeze(-1)

                mask_indices = mask.nonzero(as_tuple=False)
                flat_indices = mask_indices[:, 0] * seq_len + mask_indices[:, 1]
                output_flat = output.view(-1, hidden_dim)
                output_flat.index_add_(0, flat_indices, weighted_expert_output)

                if update_wealth:
                    self.expert_usage_count[expert_idx] += mask.sum().float()

        return output

    def start_tracking(self):
        self._track_wealth = True
        self.wealth_history = []

    def stop_tracking(self):
        self._track_wealth = False

    def get_wealth_history(self) -> list[list[float]]:
        return self.wealth_history.copy()

    def reset_tracking(self):
        """Reset the wealth history without disabling tracking."""
        self.wealth_history = []

    @classmethod
    def from_pretrained_ffn(
        cls,
        ffn_module: nn.Module,
        config: MoBConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MixtureOfBidders":
        """
        Initialize MoB by upcycling from a pretrained FFN.

        Args:
            ffn_module: The original FFN module (e.g., from Mistral)
            config: MoB configuration
            device: Target device (auto-detected from ffn_module if None)
            dtype: Target dtype (auto-detected from ffn_module if None)

        Returns:
            Initialized MoB module with weights copied from FFN
        """
        if device is None or dtype is None:
            if hasattr(ffn_module, "gate_proj"):
                ref_param = cast(nn.Linear, ffn_module.gate_proj).weight
            elif hasattr(ffn_module, "up_proj"):
                ref_param = cast(nn.Linear, ffn_module.up_proj).weight
            else:
                ref_param = next(ffn_module.parameters())

            if device is None:
                device = ref_param.device
                if device.type == "meta":
                    device = (
                        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                    )
                    logger.warning(
                        f"Detected meta device from lazy loading, forcing device={device}"
                    )
            if dtype is None:
                dtype = ref_param.dtype

        mode_str = "shared-base" if config.use_shared_base else "full-expert"
        logger.info(f"Creating MoB ({mode_str}) on device={device}, dtype={dtype}")

        mob = cls(config)

        with torch.no_grad():
            if config.use_shared_base:
                if hasattr(ffn_module, "gate_proj"):
                    ffn_gate = cast(nn.Linear, ffn_module.gate_proj)
                    ffn_up = cast(nn.Linear, ffn_module.up_proj)
                    ffn_down = cast(nn.Linear, ffn_module.down_proj)
                    mob.base_gate_proj.weight.copy_(ffn_gate.weight.cpu())
                    mob.base_up_proj.weight.copy_(ffn_up.weight.cpu())
                    mob.base_down_proj.weight.copy_(ffn_down.weight.cpu())

                for i, expert_mod in enumerate(mob.experts):
                    lw = cast(LightweightExpert, expert_mod)
                    lw.gate_adapter_A.weight.add_(
                        torch.randn_like(lw.gate_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
                    lw.up_adapter_A.weight.add_(
                        torch.randn_like(lw.up_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
                    lw.down_adapter_A.weight.add_(
                        torch.randn_like(lw.down_adapter_A.weight) * config.jitter_std * (i + 1)
                    )
            else:
                ffn_gate_w: torch.Tensor | None = None
                ffn_up_w: torch.Tensor | None = None
                ffn_down_w: torch.Tensor | None = None
                if hasattr(ffn_module, "gate_proj"):
                    ffn_gate_w = cast(nn.Linear, ffn_module.gate_proj).weight.cpu()
                    ffn_up_w = cast(nn.Linear, ffn_module.up_proj).weight.cpu()
                    ffn_down_w = cast(nn.Linear, ffn_module.down_proj).weight.cpu()

                for expert_mod in mob.experts:
                    exp = cast(Expert, expert_mod)
                    if ffn_gate_w is not None and ffn_up_w is not None and ffn_down_w is not None:
                        exp.gate_proj.weight.copy_(ffn_gate_w)
                        exp.up_proj.weight.copy_(ffn_up_w)
                        exp.down_proj.weight.copy_(ffn_down_w)

                    exp.gate_proj.weight.add_(
                        torch.randn_like(exp.gate_proj.weight) * config.jitter_std
                    )
                    exp.up_proj.weight.add_(
                        torch.randn_like(exp.up_proj.weight) * config.jitter_std
                    )
                    exp.down_proj.weight.add_(
                        torch.randn_like(exp.down_proj.weight) * config.jitter_std
                    )

        mob = mob.to(device=device, dtype=dtype)

        logger.info(
            f"Upcycled FFN to MoB with {config.num_experts} experts, "
            f"top-k={config.top_k}, mode={mode_str}"
        )
        return mob


def apply_mob_to_model(
    model: nn.Module,
    mob_config: MoBConfig,
    layers_to_modify: list[int] | None = None,
) -> nn.Module:
    layers: nn.ModuleList | None = None
    if hasattr(model, "model"):
        model_inner = cast(nn.Module, model.model)
        if hasattr(model_inner, "layers"):
            layers = cast(nn.ModuleList, model_inner.layers)
    if layers is None and hasattr(model, "layers"):
        layers = cast(nn.ModuleList, model.layers)
    if layers is None:
        raise ValueError("Cannot find transformer layers in model")

    num_layers = len(layers)
    if layers_to_modify is None:
        layers_to_modify = list(range(4, num_layers - 4))

    for layer_idx in layers_to_modify:
        layer = layers[layer_idx]

        ffn_attr: str | None = None
        if hasattr(layer, "mlp"):
            ffn_attr = "mlp"
        elif hasattr(layer, "feed_forward"):
            ffn_attr = "feed_forward"
        else:
            logger.warning(f"Layer {layer_idx}: Cannot find FFN module, skipping")
            continue

        ffn = cast(nn.Module, getattr(layer, ffn_attr))
        mob = MixtureOfBidders.from_pretrained_ffn(ffn, mob_config)
        setattr(layer, ffn_attr, mob)

        logger.info(f"Layer {layer_idx}: Replaced FFN with MoB")

    return model
