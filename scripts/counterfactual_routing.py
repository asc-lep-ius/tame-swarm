"""Is routing load-bearing on this body? A counterfactual read on #39's checkpoints (#58).

On what fraction of held-out probe tokens does the *routing decision* change the
next-token probability by more than the read's own floor, and by how much on
those tokens? An instrument reading, not a verdict: it says whether routing is
load-bearing enough on this body for a cell-scale signature to show in an
aggregate readout, and it hands #57 the tokens where it is.

**Exploratory, and computed after #39's sign was known.** Nothing here is a row
of #39, nothing enters README ``#stakes-dial-cell``, and every figure carries
the checkpoint path, arm, seed, step, K, noise scale and this script's SHA.
Under preregistration section 8 a GPU run with no power row is exploratory
whatever its issue says; this one has none and says so.

**The two lines, fixed in #58 before the first forward pass.** A fragile token is
one where the best alternative route's log-probability of the realised token
exceeds the executed route's by more than the floor. A fragile fraction below
**5%** on the ``value`` arm at the reference dose reads "routing is not
load-bearing enough here for an aggregate cell-scale readout", and opens #48;
the literature's 6.9% (arXiv 2605.07260) was measured on bodies with 8 to 64
experts against this one's 4 with 2 slots. The second line is the **adapter
footprint**: held-out loss over the probe with every MoB adapter zeroed minus
with them on, which is an upper bound on any organism-scale effect at this
configuration, since no allocation change can move the organism further than
removing the cells altogether. A footprint below **0.002** -- the pooled seed
spread of held-out loss at #25's budget -- is the second way #48 opens.

**The floor is read first and it is the replicate pair's.** ``seed0`` and
``seed0-replicate`` are bitwise the same arm under ``strict``, so any per-token
difference between their reads is the read's own numerical noise::

    # 1. the floor, on every replicate pair, pooled over the six groups
    uv run python scripts/counterfactual_routing.py \\
        --checkpoint .../r1/value/runs/seed0/checkpoint-2000 \\
        --floor-against .../r1/value/runs/seed0-replicate/checkpoint-2000 \\
        --out .../r1/value/floor.json
    # 2. every group, with the pooled floor passed in
    uv run python scripts/counterfactual_routing.py \\
        --group .../r1/value --alternatives 32 --subset 0.1 \\
        --adapter-footprint --floor <pooled> \\
        --out .../r1/value/counterfactual_routing.json

The group form writes ``run_seeds.py``'s per-seed summary shape with each
checkpoint's own ``ArmFingerprint`` copied in, so ``compare_runs.py`` tables the
readings across arms unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tame"))

from evaluation import HeldOutSplit, evaluate  # noqa: E402
from mob import mob_layers_by_index  # noqa: E402
from mob.auction import VCGAuctioneer  # noqa: E402
from mob.utils import frozen_economy  # noqa: E402
from parity import code_identity  # noqa: E402

# #58's two lines, written into the issue before any forward pass ran.
FRAGILE_FRACTION_LINE = 0.05
FOOTPRINT_LINE = 0.002
DEFAULT_ALTERNATIVES = 32
# The Gumbel scale the alternatives are drawn at. One is the plain Gumbel-top-k
# sample from the bid distribution -- the routing the gate itself would produce
# if its reports were one standard Gumbel noisier -- so the alternatives are
# equal-compute routes the body could plausibly have taken rather than adversarial
# ones. It is recorded in every figure because the fragile fraction moves with it.
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_SUBSET = 0.1
DEFAULT_PROBE_TOKENS = 4096
CHECKPOINT_GLOB = "checkpoint-*"


@dataclass(frozen=True)
class ProbeRead:
    """What one forward over the probe says about every scoreable token."""

    log_probs: torch.Tensor
    token_ids: torch.Tensor
    rerouted: torch.Tensor | None = None

    @property
    def loss(self) -> float:
        return float(-self.log_probs.mean())

    @property
    def tokens(self) -> int:
        return int(self.log_probs.numel())


def probe_batches(
    split: HeldOutSplit, batch_size: int, probe_tokens: int
) -> list[dict[str, torch.Tensor]]:
    """The run's own held-out split, truncated to the probe's token budget.

    The probe is the split the arm evaluated on, cut at ``probe_tokens`` so the
    read costs what #58 priced it at. Truncation is by whole batches, so the
    token count is reported rather than assumed.
    """
    batches: list[dict[str, torch.Tensor]] = []
    seen = 0
    for batch in split.batches(batch_size):
        batches.append(batch)
        seen += int(batch["attention_mask"][..., 1:].sum())
        if seen >= probe_tokens:
            break
    return batches


def read_probe(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    wrappers: dict[int, GumbelTopKRoute] | None = None,
) -> ProbeRead:
    """The log-probability of the token that actually followed, per scoreable position.

    The same shift and the same mask as ``evaluation._batch_loss``, so the mean
    of what this returns is the arm's own ``eval/loss`` over these batches and
    the reproduction check is a comparison of like with like.
    """
    log_probs: list[torch.Tensor] = []
    token_ids: list[torch.Tensor] = []
    rerouted: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), frozen_economy(model):
            for batch in batches:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                shift_logits = outputs.logits[..., :-1, :].float()
                shift_labels = input_ids[..., 1:]
                shift_mask = attention_mask[..., 1:] == 1
                token_log_probs = (
                    torch.log_softmax(shift_logits, dim=-1)
                    .gather(-1, shift_labels.unsqueeze(-1))
                    .squeeze(-1)
                )
                log_probs.append(token_log_probs[shift_mask].detach().cpu())
                token_ids.append(shift_labels[shift_mask].detach().cpu())
                if wrappers:
                    # The route at position t is what produced the prediction of
                    # token t+1, so the mask shifts with the labels.
                    moved = torch.zeros_like(shift_mask)
                    for wrapper in wrappers.values():
                        assert wrapper.rerouted is not None
                        moved |= wrapper.rerouted[..., :-1]
                    rerouted.append(moved[shift_mask].detach().cpu())
    finally:
        model.train(was_training)
    return ProbeRead(
        torch.cat(log_probs),
        torch.cat(token_ids),
        torch.cat(rerouted) if rerouted else None,
    )


_TINY = torch.finfo(torch.float32).tiny


class GumbelTopKRoute(torch.nn.Module):
    """The arm's own gate, with Gumbel noise on the log-bids before the top-k.

    Equal compute: the same ``top_k`` slots, a different set of experts in them.
    The share each winner takes is recomputed by the gate's own rule rather than
    assumed uniform, so a proportional-share arm is not silently re-weighted.

    ``subset`` leaves a fraction of tokens on their executed route, which is how
    the upstream confound is bounded: a rerouted token changes the stream every
    later token sees, so a read where every token is rerouted at once measures
    that too. At 10% the two estimates must agree within the floor, or the
    confound is the finding.
    """

    def __init__(
        self,
        gate: VCGAuctioneer,
        generator: torch.Generator,
        scale: float,
        subset: float | None,
    ):
        super().__init__()
        self.gate = gate
        self.generator = generator
        self.scale = scale
        self.subset = subset
        self.rerouted: torch.Tensor | None = None

    def _gumbel(self, like: torch.Tensor) -> torch.Tensor:
        """A standard Gumbel per expert per token, from this read's own generator.

        Written in three named steps rather than as one nested expression: as
        one, ``-torch.log(u).clamp_min(tiny)`` binds the clamp to the log and
        the minus to the clamp, so the exponential comes back negative and the
        second log returns NaN for every element -- and ``topk`` over a row of
        NaN returns a *fixed* index pair, which reads as a perturbed route that
        never varies. That was the first version of this function, and it made
        32 draws into one constant route; ``tests/test_counterfactual_routing.py``
        pins both halves of it now.
        """
        uniform = torch.rand(
            like.shape, generator=self.generator, device=self.generator.device
        ).clamp_min(_TINY)
        exponential = (-torch.log(uniform)).clamp_min(_TINY)
        return (-torch.log(exponential)).to(like.device)

    def forward(
        self,
        confidences: torch.Tensor,
        wealth: torch.Tensor,
        staleness: torch.Tensor | None = None,
    ):
        outcome = self.gate(confidences, wealth, staleness=staleness)
        bids = confidences * wealth.detach().to(confidences.dtype).unsqueeze(0).unsqueeze(0)
        scores = torch.log(bids.float().clamp_min(_TINY))
        selected = torch.topk(scores + self.scale * self._gumbel(bids), self.gate.top_k, dim=-1)
        selected = selected.indices
        if self.subset is not None:
            drawn = torch.rand(
                bids.shape[:-1], generator=self.generator, device=self.generator.device
            ).to(bids.device)
            selected = torch.where(
                (drawn < self.subset).unsqueeze(-1), selected, outcome.selected_experts
            )
        # Which tokens this layer actually moved, as a set rather than a draw:
        # a Gumbel that does not reorder the top-k leaves the route alone, and a
        # token nobody moved is one the read has nothing to say about. Recorded
        # on the wrapper because only the caller knows how to fold the layers
        # together.
        self.rerouted = (selected != outcome.selected_experts).any(dim=-1)
        weights = self.gate._compute_routing_weights(
            bids, torch.gather(bids, -1, selected), selected
        )
        return outcome._replace(selected_experts=selected, routing_weights=weights)


@contextmanager
def alternative_routes(
    model: torch.nn.Module, seed: int, scale: float, subset: float | None, device: torch.device
) -> Iterator[dict[int, GumbelTopKRoute]]:
    """Every converted layer routes a Gumbel-top-k alternative for the duration.

    Yields the wrappers, whose ``rerouted`` mask is what the caller reads after
    the forward: a token whose route no layer moved is one this draw says
    nothing about, and counting it as evidence either way is what made the
    subset read uninterpretable.
    """
    layers = mob_layers_by_index(model)
    generator = torch.Generator(device="cpu" if device.type == "cpu" else device)
    generator.manual_seed(seed)
    original = {index: layer.gate for index, layer in layers.items()}
    wrappers: dict[int, GumbelTopKRoute] = {}
    try:
        for index, layer in layers.items():
            gate = original[index]
            if not isinstance(gate, VCGAuctioneer):
                raise TypeError(
                    f"layer {index} is gated by {type(gate).__name__}, not the auction; a "
                    "counterfactual route is defined against the arm's own bids"
                )
            wrappers[index] = GumbelTopKRoute(gate, generator, scale, subset)
            layer.gate = wrappers[index]
        yield wrappers
    finally:
        for index, layer in layers.items():
            layer.gate = original[index]


@contextmanager
def adapters_zeroed(model: torch.nn.Module) -> Iterator[None]:
    """Every MoB expert's contribution set to zero, and restored on the way out.

    Zeroing the ``B`` factor of each adapter is what removes the cell: the
    contribution is ``B A h`` at a scaling, so the experts fall back to the
    shared base FFN they were upcycled from and the layer computes what the
    unconverted model would have.
    """
    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for layer in mob_layers_by_index(model).values():
                for expert in layer.experts:
                    for name in ("gate_adapter_B", "up_adapter_B", "down_adapter_B"):
                        weight = getattr(expert, name).weight
                        saved.append((weight, weight.detach().clone()))
                        weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for weight, value in saved:
                weight.copy_(value)


def counterfactual_gaps(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    executed: ProbeRead,
    alternatives: int,
    scale: float,
    subset: float | None,
    seed: int,
) -> dict[str, Any]:
    """Executed against the best of ``alternatives`` equal-compute routes, per token."""
    best = executed.log_probs.clone()
    better = torch.zeros_like(executed.log_probs)
    moved_better = torch.zeros(())
    moved_total = torch.zeros(())
    still_better = torch.zeros(())
    still_total = torch.zeros(())
    for draw in range(alternatives):
        with alternative_routes(model, seed + draw, scale, subset, device) as wrappers:
            alternative = read_probe(model, batches, device, wrappers)
        if alternative.tokens != executed.tokens:
            raise ValueError("an alternative route read a different number of tokens")
        best = torch.maximum(best, alternative.log_probs)
        improved = alternative.log_probs > executed.log_probs
        better += improved.float()
        assert alternative.rerouted is not None
        moved = alternative.rerouted
        moved_better += float((improved & moved).sum())
        moved_total += float(moved.sum())
        still_better += float((improved & ~moved).sum())
        still_total += float((~moved).sum())
    gap = best - executed.log_probs
    return {
        "best_minus_executed": gap,
        "alternatives_better_than_executed": better / alternatives,
        # The confound, separated rather than bounded: among draw-token pairs
        # whose own route moved, how often the change helped -- against the same
        # rate on the pairs whose route did not move, where only the stream
        # upstream of them changed. The difference is the route's own effect and
        # the second number is its control.
        "moved_better": float(moved_better / moved_total) if float(moved_total) else float("nan"),
        "unmoved_better": float(still_better / still_total) if float(still_total) else float("nan"),
        "moved_fraction": float(moved_total) / (alternatives * executed.tokens),
    }


def read_checkpoint(
    checkpoint: Path,
    probe_tokens: int,
    batch_size: int,
    device_name: str,
) -> tuple[torch.nn.Module, list[dict[str, torch.Tensor]], HeldOutSplit, dict[str, Any]]:
    """Rebuild the arm this checkpoint was trained as, restore it, and load its probe.

    The config travels in ``training_state.pt``, so the arm is rebuilt from what
    it recorded rather than from flags a reader has to get right. The goal fields
    are deliberately *not* re-calibrated: a field prices realised value and never
    reaches the forward, and the recorded ``eval/loss`` this function reproduces
    is what proves that rather than an argument about it.
    """
    from train import TAMETrainer, TrainingConfig, restore_checkpoint  # noqa: PLC0415

    # The checkpoints are frozen and this read is the reason to say so: the
    # trainer opens an MLflow run on construction and, left to itself, writes
    # ``mlruns/`` into ``output_dir`` -- which here is #39's run directory. The
    # store is pointed at a scratch path before the first trainer exists, so a
    # read leaves the run directory exactly as it found it.
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI", f"file:{Path(tempfile.gettempdir()) / 'tame-counterfactual-mlruns'}"
    )
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    recorded = dict(state["config"])
    run_dir = checkpoint.parent
    recorded.update(
        output_dir=str(run_dir),
        device=device_name,
        goal_fields=(),
        goal_doses=(),
        gradient_checkpointing=False,
    )
    known = {field for field in TrainingConfig.__dataclass_fields__}
    config = TrainingConfig(**{key: value for key, value in recorded.items() if key in known})
    trainer = TAMETrainer(config)
    trainer.setup()
    restore_checkpoint(trainer.model, checkpoint)
    split = HeldOutSplit.load(run_dir / "held_out_split.pt")
    batches = probe_batches(split, batch_size, probe_tokens)
    provenance = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "global_step": state.get("global_step"),
        "arm_fingerprint": state.get("arm_fingerprint"),
        "eval_split": split.fingerprint,
        "recorded_eval_loss": next(
            (
                entry["eval/loss"]
                for entry in reversed(state.get("eval_history") or [])
                if "eval/loss" in entry
            ),
            None,
        ),
    }
    return trainer.model, batches, split, provenance


def verify_recorded_loss(
    model: torch.nn.Module,
    split: HeldOutSplit,
    batch_size: int,
    device: torch.device,
    provenance: dict[str, Any],
    tolerance: float,
) -> None:
    """The restored arm must reproduce the loss the run recorded, before anything else.

    #58's first acceptance criterion and the first trap it lists: a ``--use_lora``
    checkpoint held no MoB modules at all before #29, and a read on one would
    have measured an upcycled body wearing a trained wealth vector. This is the
    check that the arm on the card is the arm that was trained, and it runs
    through ``evaluation.evaluate`` rather than through this script's own
    per-token read. Two reasons, and the second is the one that bit: the whole
    held-out split is what ``eval/loss`` was computed on, where the probe is a
    prefix of it; and the recorded number is a bfloat16 cross-entropy, where the
    read below takes log-softmax in float32 for the per-token comparison it
    makes. The two differ in the fourth decimal on this body -- 2.617482 against
    2.618158 -- which is precision, not a different arm, but a check that cannot
    tell those apart is not a check.
    """
    recorded = provenance.get("recorded_eval_loss")
    reproduced = evaluate(model, split, batch_size, device).loss
    provenance["reproduced_eval_loss"] = reproduced
    if recorded is None:
        raise ValueError(
            f"{provenance['checkpoint']} records no eval/loss to reproduce; the read refuses "
            "rather than measure an arm it cannot show was restored"
        )
    if abs(reproduced - recorded) > tolerance:
        raise ValueError(
            f"the restored arm reads {reproduced:.6f} on the held-out split where the run "
            f"recorded {recorded:.6f} (tolerance {tolerance}); it is not the arm that trained"
        )


def read_one(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    floor: float | None,
) -> dict[str, Any]:
    """One checkpoint: the reproduction check, the counterfactual read, the footprint."""
    model, batches, split, provenance = read_checkpoint(
        checkpoint, args.probe_tokens, args.batch_size, args.device
    )
    verify_recorded_loss(model, split, args.batch_size, device, provenance, args.loss_tolerance)
    executed = read_probe(model, batches, device)
    reading: dict[str, Any] = {
        **provenance,
        "probe_tokens": executed.tokens,
        "probe_batches": len(batches),
        "probe_loss": executed.loss,
        "alternatives": args.alternatives,
        "noise_scale": args.noise_scale,
        "subset": args.subset,
        "script_sha": code_identity()[0],
        "script_dirty": code_identity()[1],
    }
    read = counterfactual_gaps(
        model, batches, device, executed, args.alternatives, args.noise_scale, None, args.seed
    )
    gap = read["best_minus_executed"]
    reading["counterfactual/mean_gap"] = float(gap.mean())
    reading["counterfactual/alternatives_better"] = float(
        read["alternatives_better_than_executed"].mean()
    )
    reading["counterfactual/moved_better"] = read["moved_better"]
    reading["counterfactual/unmoved_better"] = read["unmoved_better"]
    reading["counterfactual/own_route_effect"] = read["moved_better"] - read["unmoved_better"]
    reading["counterfactual/moved_fraction"] = read["moved_fraction"]
    if floor is not None:
        fragile = gap > floor
        reading.update(
            {
                "floor": floor,
                "counterfactual/fragile_fraction": float(fragile.float().mean()),
                "counterfactual/best_route_gap_on_fragile": float(gap[fragile].mean())
                if bool(fragile.any())
                else 0.0,
                "counterfactual/gap_p50": float(gap.median()),
                "counterfactual/gap_p95": float(gap.quantile(0.95)),
                "fragile_token_index": torch.nonzero(fragile).flatten().tolist(),
                "fragile_token_ids": executed.token_ids[fragile].tolist(),
            }
        )
    if args.subset is not None:
        # The upstream confound, bounded in the same model load: rerouting every
        # token at once measures the reroutes *and* the stream they change, so
        # the same read on a tenth of tokens is what says which it was. The two
        # must agree within the floor, or the disagreement is the finding.
        bounded = counterfactual_gaps(
            model,
            batches,
            device,
            executed,
            args.alternatives,
            args.noise_scale,
            args.subset,
            args.seed,
        )
        reading["counterfactual/mean_gap_subset"] = float(bounded["best_minus_executed"].mean())
        reading["counterfactual/moved_better_subset"] = bounded["moved_better"]
        reading["counterfactual/unmoved_better_subset"] = bounded["unmoved_better"]
        if floor is not None:
            subset_fragile = float((bounded["best_minus_executed"] > floor).float().mean())
            reading["counterfactual/fragile_fraction_subset"] = subset_fragile
            reading["counterfactual/subset_disagreement"] = abs(
                subset_fragile - reading["counterfactual/fragile_fraction"]
            )
    if args.floor_against is not None:
        against_model, against_batches, _, against = read_checkpoint(
            Path(args.floor_against), args.probe_tokens, args.batch_size, args.device
        )
        replicate = read_probe(against_model, against_batches, device)
        difference = (replicate.log_probs - executed.log_probs).abs()
        reading["floor_pair"] = {
            "against": against["checkpoint"],
            "max_abs_difference": float(difference.max()),
            "std_difference": float(difference.std()),
            "mean_abs_difference": float(difference.mean()),
        }
    if args.adapter_footprint:
        with adapters_zeroed(model):
            without = read_probe(model, batches, device)
        reading["footprint/loss_without_adapters"] = without.loss
        reading["footprint/loss_with_adapters"] = executed.loss
        reading["footprint/adapter_footprint"] = without.loss - executed.loss
        if floor is not None:
            fragile = gap > floor
            if bool(fragile.any()):
                reading["footprint/on_fragile_tokens"] = float(
                    (executed.log_probs[fragile] - without.log_probs[fragile]).mean()
                )
    return reading


def summarise(readings: dict[str, dict[str, Any]], group: Path | None) -> dict[str, Any]:
    """``run_seeds.py``'s summary shape, so ``compare_runs.py`` tables it unchanged."""
    per_seed = {
        seed: {
            key: value
            for key, value in reading.items()
            if isinstance(value, (int, float)) and "/" in key
        }
        for seed, reading in readings.items()
    }
    fingerprints = {
        seed: reading["arm_fingerprint"]
        for seed, reading in readings.items()
        if reading.get("arm_fingerprint")
    }
    return {
        "arm": next(
            (
                reading["arm_fingerprint"].get("persistence_coupling", "unknown")
                for reading in readings.values()
                if reading.get("arm_fingerprint")
            ),
            "unknown",
        ),
        "router": "mob",
        "group": str(group) if group else None,
        "seeds": sorted(per_seed, key=str),
        "primary": None,
        "per_seed": per_seed,
        "fingerprints": fingerprints,
        "stats": {
            metric: {
                "mean": sum(values) / len(values),
                "std": float(torch.tensor(values).std()) if len(values) > 1 else float("nan"),
                "n": len(values),
            }
            for metric, values in (
                (
                    metric,
                    [row[metric] for row in per_seed.values() if metric in row],
                )
                for metric in sorted({key for row in per_seed.values() for key in row})
            )
        },
        "replicate_seed": None,
        "replicate": None,
        "replication_std": None,
        "replication_error": (
            "exploratory read, not a run: the floor is the replicate pair's per-token spread "
            "(#58), recorded per reading rather than as a seed spread"
        ),
        "floor_recorded_at": None,
        "readings": readings,
    }


def checkpoints_of(group: Path, include_replicates: bool = False) -> dict[str, Path]:
    """Every seed's last checkpoint under a ``run_seeds.py`` group, keyed by seed.

    The replicate is left out by default: it is seed 0 again, it is what the
    floor pass reads, and a group summary that carried it would table one seed
    twice as though it were two.
    """
    found: dict[str, Path] = {}
    for run in sorted((group / "runs").glob("seed*")):
        if not include_replicates and run.name.endswith("-replicate"):
            continue
        checkpoints = sorted(
            run.glob(CHECKPOINT_GLOB), key=lambda path: int(path.name.split("-")[-1])
        )
        if checkpoints:
            found[run.name.replace("seed", "")] = checkpoints[-1]
    if not found:
        raise FileNotFoundError(f"no checkpoints under {group}/runs/seed*/")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="One checkpoint directory")
    parser.add_argument("--group", type=Path, default=None, help="A run_seeds.py group directory")
    parser.add_argument("--alternatives", type=int, default=DEFAULT_ALTERNATIVES)
    parser.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help=(
            "Reroute this fraction of tokens only, bounding the upstream confound "
            f"(#58 fixes it at {DEFAULT_SUBSET}; unset reroutes every token)"
        ),
    )
    parser.add_argument("--adapter-footprint", action="store_true")
    parser.add_argument("--floor", type=float, default=None, help="The pooled replicate-pair floor")
    parser.add_argument(
        "--floor-against",
        type=str,
        default=None,
        help="A bitwise-identical checkpoint; its per-token spread is this read's own floor",
    )
    parser.add_argument(
        "--loss-tolerance",
        type=float,
        default=1e-5,
        help="How far the restored arm may read from the eval/loss the run recorded",
    )
    parser.add_argument("--probe-tokens", type=int, default=DEFAULT_PROBE_TOKENS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="The alternatives' own RNG seed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.group is None):
        parser.error("pass exactly one of --checkpoint and --group")
    device = torch.device(args.device)

    checkpoints = (
        {"0": args.checkpoint} if args.checkpoint is not None else checkpoints_of(args.group)
    )
    readings: dict[str, dict[str, Any]] = {}
    for seed, checkpoint in checkpoints.items():
        print(f"reading {checkpoint}", flush=True)
        reading = read_one(checkpoint, args, device, args.floor)
        readings[seed] = reading
        recorded = reading.get("recorded_eval_loss")
        print(
            f"  probe {reading['probe_tokens']} tokens, loss {reading['probe_loss']:.5f}"
            + (f" (the run recorded {recorded:.5f} over the whole split)" if recorded else "")
        )
        if "counterfactual/fragile_fraction" in reading:
            fraction = reading["counterfactual/fragile_fraction"]
            below = "below: #48 opens" if fraction < FRAGILE_FRACTION_LINE else "at or above"
            print(
                f"  fragile fraction {fraction:.4f} against the "
                f"{FRAGILE_FRACTION_LINE:.0%} line -> {below}"
            )
        if "footprint/adapter_footprint" in reading:
            footprint = reading["footprint/adapter_footprint"]
            print(
                f"  adapter footprint {footprint:+.5f} nats against the {FOOTPRINT_LINE} line "
                f"-> {'below: #48 opens' if footprint < FOOTPRINT_LINE else 'at or above'}"
            )
        if "floor_pair" in reading:
            print(f"  floor pair: {reading['floor_pair']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summarise(readings, args.group), indent=2))
    print(f"\nrecord: {args.out}")


if __name__ == "__main__":
    main()
