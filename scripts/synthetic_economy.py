"""A planted-competence economy for measuring the mechanism without a language model.

The #15 measurement problem: to ask whether the economy tracks competence, the
competence has to be *known*, and it has to be shuffled away from expert index,
because ``ConfidenceHead`` seeds its bias monotone in index and a competence vector
that is also monotone in index cannot be told apart from that initialisation --
which is exactly how the correlations recorded on the #10 branch turned out to be
artefacts.

The fixture is a shared-base MoB layer whose experts have their adapters planted
rather than trained. Every expert's gate and up adapters are zero, so its hidden
activation is the base's; its down adapter is ``competence x M @ R``, a fixed
rank-``r`` correction scaled by that expert's competence. The per-token target is
the base output plus the full correction, so an expert of competence ``c`` closes a
fraction ``c`` of the gap on every token it holds, the loss is ``(1 - mean winner
competence)^2 |T|^2``, and the value of holding a slot is monotone in competence by
construction. Nothing else is trained but the confidence heads, which is the point:
this isolates the economy from adapter learning.

``positive_fraction`` flips the sign of the correction on a fraction of tokens the
heads cannot predict from the input, so an expert realises negative value on some
of the tokens it holds. That is the fixture on which a report trained onto the mean
of realised value and one trained onto its positive part *differ* -- the
acceptance criterion for unbiased reports.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from mob import LightweightExpert, MixtureOfBidders, MoBConfig  # noqa: E402

# Eight experts, top-2: the configuration every synthetic measurement in the
# project has used, so numbers here are comparable with the sweep and the
# stationarity run.
DEFAULT_COMPETENCE = torch.tensor([0.9, 0.7, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1])

BASE_CONFIG = MoBConfig(
    num_experts=8,
    top_k=2,
    hidden_dim=64,
    intermediate_dim=128,
    adapter_rank=8,
    adapter_alpha=8.0,
)

# Standard deviation of the planted correction's entries. Sets the scale of every
# value and price in the fixture; it is chosen so that a full correction moves the
# per-token loss by order one, which is the scale the reward constants were
# derived on.
CORRECTION_STD = 0.35


def shuffled(competence: torch.Tensor, seed: int) -> torch.Tensor:
    """A permutation of ``competence`` that is not monotone in expert index.

    Seeded from its own generator so the shuffle is fixed by ``seed`` alone and
    does not disturb the run's other draws. A permutation that happens to come
    back sorted either way would reproduce the initialisation artefact the
    fixture exists to escape, so it is redrawn.
    """
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(competence.numel(), generator=generator)
        candidate = competence[order]
        if not _monotone(candidate):
            return candidate


def _monotone(values: torch.Tensor) -> bool:
    steps = values[1:] - values[:-1]
    return bool((steps >= 0).all() or (steps <= 0).all())


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return float("nan")
    return float((a * b).sum() / denominator)


@dataclass(frozen=True)
class StepRecord:
    """What one settled step looked like, for the run to accumulate."""

    loss: float
    selected_experts: torch.Tensor
    realised_values: torch.Tensor
    # The exact counterfactual for every held slot; None unless asked for, since
    # it costs a forward per winning expert and only the accuracy check reads it.
    exact_values: torch.Tensor | None
    mean_realised_value: float
    mean_report: float
    mean_price: float
    mean_surplus: float


@dataclass(frozen=True)
class RunSummary:
    """A run's economy, with the market read at its steady state.

    The ``final_*`` figures average the last ``window`` steps only. The whole-run
    means fold in the transient during which the heads are still learning what
    their experts are worth, and the question #15 asks -- is winning profitable
    for a competent expert -- is about the market once the reports are calibrated.
    Wealth and win share are read over the whole run, as the issue measured them.
    """

    competence: torch.Tensor
    wealth: torch.Tensor
    win_share: torch.Tensor
    mean_realised_value: float
    mean_report: float
    mean_price: float
    mean_surplus: float
    final_realised_value: float
    final_report: float
    final_price: float
    final_surplus: float

    @property
    def wealth_vs_win_share(self) -> float:
        return pearson(self.wealth, self.win_share)

    @property
    def wealth_vs_competence(self) -> float:
        return pearson(self.wealth, self.competence)

    @property
    def wealth_vs_index(self) -> float:
        return pearson(self.wealth, torch.arange(self.wealth.numel()))


class SyntheticEconomy:
    """One MoB layer with planted competence, driven step by step."""

    def __init__(
        self,
        competence: torch.Tensor,
        seed: int,
        config: MoBConfig = BASE_CONFIG,
        batch_size: int = 2,
        seq_len: int = 16,
        positive_fraction: float = 1.0,
        head_learning_rate: float = 1e-2,
    ):
        torch.manual_seed(seed)
        self.competence = competence.float()
        self.config = replace(config, num_experts=competence.numel())
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.positive_fraction = positive_fraction

        self.mob = MixtureOfBidders(self.config)
        self.mob.train()
        self._plant(self.competence)

        self.optimizer = torch.optim.Adam(
            self.mob.confidence_heads.parameters(), lr=head_learning_rate
        )
        self.generator = torch.Generator().manual_seed(seed)

    def _plant(self, competence: torch.Tensor) -> None:
        config = self.config
        rank = config.adapter_rank
        # Shared across experts so competence is the *only* thing that differs.
        shared_a = torch.randn(rank, config.intermediate_dim) / math.sqrt(config.intermediate_dim)
        shared_m = torch.randn(config.hidden_dim, rank) * CORRECTION_STD / math.sqrt(rank)
        with torch.no_grad():
            for expert_competence, module in zip(
                competence.tolist(), self.mob.experts, strict=True
            ):
                expert = cast(LightweightExpert, module)
                expert.gate_adapter_B.weight.zero_()
                expert.up_adapter_B.weight.zero_()
                expert.down_adapter_A.weight.copy_(shared_a)
                expert.down_adapter_B.weight.copy_(expert_competence * shared_m)
        self._correction = (shared_m @ shared_a) * cast(
            LightweightExpert, self.mob.experts[0]
        ).scaling

    def _base(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_hidden = F.silu(self.mob.base_gate_proj(x)) * self.mob.base_up_proj(x)
        return self.mob.base_down_proj(base_hidden), base_hidden

    def _draw(self) -> tuple[torch.Tensor, torch.Tensor]:
        """An input batch and its target: the base output plus the signed correction."""
        x = torch.randn(
            self.batch_size, self.seq_len, self.config.hidden_dim, generator=self.generator
        )
        with torch.no_grad():
            base_out, base_hidden = self._base(x)
            correction = base_hidden @ self._correction.T
            positive = torch.rand(self.batch_size, self.seq_len, generator=self.generator)
            sign = torch.where(positive < self.positive_fraction, 1.0, -1.0).unsqueeze(-1)
            target = base_out + sign * correction
        return x, target

    def exact_values(
        self, x: torch.Tensor, output: torch.Tensor, target: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        """The counterfactual the gradient estimate approximates, computed exactly.

        For each winner slot: the per-token loss with that expert's contribution
        replaced by the base, minus the per-token loss as realised, per unit share.
        """
        k = self.config.top_k
        with torch.no_grad():
            per_token = ((output - target) ** 2).sum(-1)
            exact = torch.zeros(*selected.shape, dtype=torch.float32)
            for slot in range(k):
                for expert_idx in range(self.config.num_experts):
                    mask = selected[:, :, slot] == expert_idx
                    if not mask.any():
                        continue
                    expert = cast(LightweightExpert, self.mob.experts[expert_idx])
                    held, reference = expert.forward_with_reference(
                        x[mask],
                        self.mob.base_gate_proj,
                        self.mob.base_up_proj,
                        self.mob.base_down_proj,
                    )
                    without = output[mask] - (held - reference) / k
                    counterfactual = ((without - target[mask]) ** 2).sum(-1)
                    exact[:, :, slot][mask] = (counterfactual - per_token[mask]) * k
        return exact

    def step(self, with_exact_values: bool = False) -> StepRecord:
        x, target = self._draw()
        output = self.mob(x)
        per_token = ((output - target) ** 2).sum(-1)
        loss = per_token.mean()
        loss.backward()

        self.mob.update_wealth_from_loss(
            per_token.detach(), loss_gradient_scale=float(per_token.numel())
        )
        auxiliary = self.mob.get_confidence_calibration_loss() + self.mob.get_router_z_loss()
        auxiliary.backward()
        self.optimizer.step()
        self.mob.zero_grad(set_to_none=True)

        stats = self.mob.last_stats
        summary = self.mob.last_value_summary
        realised = self.mob.last_realised_values
        assert stats is not None and summary is not None and realised is not None
        return StepRecord(
            loss=loss.item(),
            selected_experts=stats.selected_experts,
            realised_values=realised,
            exact_values=(
                self.exact_values(x, output.detach(), target, stats.selected_experts)
                if with_exact_values
                else None
            ),
            mean_realised_value=summary.mean_realised_value.item(),
            mean_report=summary.mean_report.item(),
            mean_price=summary.mean_price.item(),
            mean_surplus=summary.mean_surplus.item(),
        )

    def run(self, steps: int, window: int = 50) -> RunSummary:
        if not 0 < window <= steps:
            raise ValueError(f"window must lie in (0, steps], got {window} for {steps} steps")
        wins = torch.zeros(self.config.num_experts)
        totals = {"value": 0.0, "report": 0.0, "price": 0.0, "surplus": 0.0}
        final = dict(totals)
        for step in range(steps):
            record = self.step()
            wins += torch.bincount(
                record.selected_experts.flatten(), minlength=self.config.num_experts
            ).float()
            for sums in (totals, final) if step >= steps - window else (totals,):
                sums["value"] += record.mean_realised_value
                sums["report"] += record.mean_report
                sums["price"] += record.mean_price
                sums["surplus"] += record.mean_surplus
        return RunSummary(
            competence=self.competence,
            wealth=self.mob.expert_wealth.detach().clone(),
            win_share=wins / wins.sum(),
            mean_realised_value=totals["value"] / steps,
            mean_report=totals["report"] / steps,
            mean_price=totals["price"] / steps,
            mean_surplus=totals["surplus"] / steps,
            final_realised_value=final["value"] / window,
            final_report=final["report"] / window,
            final_price=final["price"] / window,
            final_surplus=final["surplus"] / window,
        )
