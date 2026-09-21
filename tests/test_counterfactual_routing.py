"""The counterfactual routing read, on a body small enough to run on a CPU (#58).

The read itself is 24 checkpoints of Qwen3-1.7B and about two GPU-hours. What a
test can pin is every part of it that is not the model: that an alternative route
is a different equal-compute route and not a different amount of compute, that
the gate comes back afterwards, that zeroing the cells is reversible, and that a
subset read leaves most tokens on the route they executed.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from counterfactual_routing import (  # noqa: E402
    FOOTPRINT_LINE,
    FRAGILE_FRACTION_LINE,
    ProbeRead,
    adapters_zeroed,
    alternative_routes,
    counterfactual_gaps,
    read_probe,
)
from counterfactual_summary import checkpoints_of, summarise  # noqa: E402

from mob import apply_mob_to_model, mob_layers_by_index  # noqa: E402
from mob.softmax_router import SoftmaxRouter  # noqa: E402

from .conftest import build_tiny_causal_lm  # noqa: E402

BLOCKS = [1, 3]
DEVICE = torch.device("cpu")


@pytest.fixture
def body(tiny_mob_config):
    """A converted body whose cells contribute something.

    An upcycled conversion has every adapter's ``B`` factor at zero, so its
    experts are the shared base and the questions this script asks -- does the
    route matter, do the cells matter -- have the answer "no" by construction.
    The adapters are moved off that initialisation the way a training step would,
    so the read is exercised on a body that has cells at all.
    """
    torch.manual_seed(0)
    model = apply_mob_to_model(build_tiny_causal_lm(4), tiny_mob_config, BLOCKS)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in mob_layers_by_index(model).values():
            for expert in layer.experts:
                for name in ("gate_adapter_B", "up_adapter_B", "down_adapter_B"):
                    weight = getattr(expert, name).weight
                    weight.add_(torch.randn(weight.shape, generator=generator))
    return model.eval()


@pytest.fixture
def batches():
    generator = torch.Generator().manual_seed(3)
    return [
        {
            "input_ids": torch.randint(0, 64, (2, 16), generator=generator),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        for _ in range(2)
    ]


def _split_of(batches):
    """The batches as a HeldOutSplit, so the check runs through evaluation.evaluate."""
    from evaluation import HeldOutSplit

    return HeldOutSplit(
        input_ids=torch.cat([batch["input_ids"] for batch in batches]),
        attention_mask=torch.cat([batch["attention_mask"] for batch in batches]),
        fingerprint="test",
        source="validation",
        dataset="test",
    )


def win_share(body, batches):
    """What the gate did, through the trace the layer keeps outside training."""
    layer = mob_layers_by_index(body)[BLOCKS[0]]
    trace = layer.enable_routing_trace(maxlen=1024)
    try:
        read_probe(body, batches, DEVICE)
        return torch.tensor(trace.summary().win_share)
    finally:
        layer.disable_routing_trace()


def test_the_lines_are_the_ones_the_issue_fixed_before_the_read():
    """#48's entry criteria, in the code rather than in a reader's memory."""
    assert FRAGILE_FRACTION_LINE == 0.05
    assert FOOTPRINT_LINE == 0.002


def test_the_read_is_the_log_probability_of_the_token_that_followed(body, batches):
    read = read_probe(body, batches, DEVICE)

    assert read.tokens == 2 * 2 * 15
    assert read.loss == pytest.approx(float(-read.log_probs.mean()))
    # The same forward twice is the same read: the floor this measures against
    # is a numerical one, and it has to be zero before a checkpoint pair can
    # say otherwise.
    assert torch.equal(read.log_probs, read_probe(body, batches, DEVICE).log_probs)


def test_two_draws_are_two_routes_and_the_noise_is_finite():
    """The defect this test exists for: 32 alternatives that were all one route.

    ``-torch.log(u).clamp_min(tiny)`` binds the clamp to the log and the minus
    to the clamp, so the exponential came back negative, the second log
    returned NaN for every element, and ``topk`` over a row of NaN returns a
    *fixed* index pair. Every draw was the same constant route, the read still
    produced plausible-looking gaps, and nothing caught it -- the routes did
    differ from the executed one, which is all the first version of this test
    asserted.
    """
    from counterfactual_routing import GumbelTopKRoute

    from mob.auction import VCGAuctioneer

    gate = VCGAuctioneer(num_experts=4, top_k=2)
    torch.manual_seed(0)
    confidences = torch.rand(2, 5, 4) + 0.1
    wealth = torch.tensor([75.0, 120.0, 60.0, 90.0])

    draws = []
    for seed in (0, 1, 2):
        generator = torch.Generator()
        generator.manual_seed(seed)
        route = GumbelTopKRoute(gate, generator, 1.0, None)
        assert torch.isfinite(route._gumbel(confidences)).all()
        draws.append(route(confidences, wealth).selected_experts)

    assert not torch.equal(draws[0], draws[1])
    assert not torch.equal(draws[1], draws[2])
    # A Gumbel at scale one is a perturbation of the bid order, not a
    # randomisation of it: most tokens keep the route they executed.
    executed = gate(confidences, wealth).selected_experts
    kept = (draws[0] == executed).all(dim=-1).float().mean()
    assert 0.2 < float(kept) < 1.0


def test_an_alternative_route_is_a_different_route_at_the_same_compute(body, batches):
    executed = win_share(body, batches)

    with alternative_routes(body, seed=1, scale=1.0, subset=None, device=DEVICE):
        alternative = win_share(body, batches)
    with alternative_routes(body, seed=2, scale=1.0, subset=None, device=DEVICE):
        other = win_share(body, batches)

    assert not torch.equal(alternative, executed)
    assert not torch.equal(alternative, other)
    # Equal compute is the point: the same number of slots, other experts in
    # them, so the shares still sum to top_k.
    assert float(alternative.sum()) == pytest.approx(float(executed.sum()))
    assert torch.equal(win_share(body, batches), executed)


def test_a_subset_read_reroutes_the_fraction_of_tokens_it_says(body, batches):
    """`--subset 0.1` means a tenth of tokens, not a tenth of layer-token pairs.

    The first version drew the mask independently inside each converted layer,
    so a token was left alone only with probability (1 - subset) ** layers: on
    the body's sixteen converted layers `--subset 0.1` rerouted 61% of tokens
    and the stream-only control was 39% of them rather than 90%. One mask per
    draw, shared across the layers, is what makes the flag mean what it says.
    """
    executed = read_probe(body, batches, DEVICE)

    tenth = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=8, scale=1.0, subset=0.1, seed=0
    )
    everything = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=8, scale=1.0, subset=None, seed=0
    )

    assert tenth["moved_fraction"] < everything["moved_fraction"]
    # Two converted layers here, so a per-layer draw would read about 0.19.
    assert tenth["moved_fraction"] == pytest.approx(0.1, abs=0.06)


def test_the_gate_comes_back_even_when_the_read_raises(body, batches):
    original = {index: layer.gate for index, layer in mob_layers_by_index(body).items()}

    with (
        pytest.raises(RuntimeError),
        alternative_routes(body, seed=0, scale=1.0, subset=None, device=DEVICE),
    ):
        raise RuntimeError("the read failed halfway")

    assert {index: layer.gate for index, layer in mob_layers_by_index(body).items()} == original


def test_a_gate_that_is_not_an_auction_is_refused_rather_than_perturbed(body):
    """A counterfactual route is defined against the arm's own bids."""
    layer = mob_layers_by_index(body)[BLOCKS[0]]
    layer.gate = SoftmaxRouter(layer.config.num_experts, layer.config.top_k)

    with (
        pytest.raises(TypeError, match="not the auction"),
        alternative_routes(body, seed=0, scale=1.0, subset=None, device=DEVICE),
    ):
        pass


def test_zeroing_the_cells_is_reversible_and_changes_what_the_body_says(body, batches):
    before = read_probe(body, batches, DEVICE)

    with adapters_zeroed(body):
        without = read_probe(body, batches, DEVICE)

    after = read_probe(body, batches, DEVICE)
    assert not torch.equal(without.log_probs, before.log_probs)
    assert torch.equal(after.log_probs, before.log_probs)


def test_the_gap_is_never_negative_and_counts_how_many_alternatives_beat_the_route(body, batches):
    executed = read_probe(body, batches, DEVICE)

    read = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=4, scale=1.0, subset=None, seed=0
    )

    assert bool((read["best_minus_executed"] >= 0).all())
    assert read["alternatives_better_than_executed"].max() <= 1.0
    assert read["best_minus_executed"].numel() == executed.tokens


def test_the_summary_is_the_shape_compare_runs_already_reads(tmp_path):
    readings = {
        "0": {
            "counterfactual/fragile_fraction": 0.03,
            "footprint/adapter_footprint": 0.001,
            "probe_loss": 3.1,
            "arm_fingerprint": {"persistence_coupling": "value", "seed": 0},
            "checkpoint": "somewhere/checkpoint-2000",
        },
        "1": {
            "counterfactual/fragile_fraction": 0.05,
            "footprint/adapter_footprint": 0.003,
            "probe_loss": 3.2,
            "arm_fingerprint": {"persistence_coupling": "value", "seed": 1},
            "checkpoint": "elsewhere/checkpoint-2000",
        },
    }

    summary = summarise(readings, tmp_path)

    assert summary["arm"] == "value"
    assert summary["per_seed"]["0"] == {
        "counterfactual/fragile_fraction": 0.03,
        "footprint/adapter_footprint": 0.001,
    }
    assert summary["stats"]["counterfactual/fragile_fraction"]["mean"] == pytest.approx(0.04)
    assert summary["stats"]["counterfactual/fragile_fraction"]["n"] == 2
    assert set(summary["fingerprints"]) == {"0", "1"}
    # Not a run, and the summary says so where a reader looks for a floor.
    assert summary["replication_std"] is None
    assert "exploratory read" in summary["replication_error"]


def test_the_last_checkpoint_of_every_seed_is_the_one_read(tmp_path):
    for seed in ("seed0", "seed1", "seed0-replicate"):
        for step in (250, 2000):
            (tmp_path / "runs" / seed / f"checkpoint-{step}").mkdir(parents=True)

    found = checkpoints_of(tmp_path)

    # The replicate is seed 0 again and is what the floor pass reads; a group
    # summary carrying it would table one seed twice as though it were two.
    assert set(found) == {"0", "1"}
    assert set(checkpoints_of(tmp_path, include_replicates=True)) == {"0", "1", "0-replicate"}
    assert all(path.name == "checkpoint-2000" for path in found.values())
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        checkpoints_of(tmp_path / "nowhere")


def test_a_checkpoint_that_does_not_reproduce_its_recorded_loss_is_refused(body, batches):
    """#58's first acceptance criterion, and the trap #29 closed.

    A `--use_lora` checkpoint written before #29 restores an upcycled body
    wearing a trained wealth vector and reports success; the only thing that
    catches it is the loss the run recorded.
    """
    from counterfactual_routing import verify_recorded_loss

    from evaluation import evaluate

    split = _split_of(batches)
    reproduced = evaluate(body, split, 2, DEVICE).loss
    provenance = {"checkpoint": "somewhere", "recorded_eval_loss": reproduced}

    verify_recorded_loss(body, split, 2, DEVICE, provenance, tolerance=1e-5)
    assert provenance["reproduced_eval_loss"] == pytest.approx(reproduced)

    with pytest.raises(ValueError, match="not the arm that trained"):
        verify_recorded_loss(
            body, split, 2, DEVICE, {"checkpoint": "x", "recorded_eval_loss": 1.0}, tolerance=1e-5
        )
    with pytest.raises(ValueError, match="records no eval/loss"):
        verify_recorded_loss(body, split, 2, DEVICE, {"checkpoint": "x"}, tolerance=1e-5)


def test_the_own_route_effect_is_separated_from_the_stream_it_changes(body, batches):
    """What the mask buys: a token whose own route moved, against one whose did not.

    Rerouting every token at once moves the stream every later token sees, so a
    gap at token t is t's own route *and* everything upstream of it. Counting
    the two populations apart -- draw-token pairs whose route moved, and pairs
    where only the stream did -- is what makes the difference between them the
    route's own effect, with the second number as its control.
    """
    executed = read_probe(body, batches, DEVICE)

    read = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=6, scale=1.0, subset=None, seed=0
    )

    assert 0.0 < read["moved_fraction"] < 1.0
    assert read["moved_pairs"] + read["unmoved_pairs"] == pytest.approx(6 * executed.tokens)
    assert 0.0 <= (read["moved_better"] or 0.0) <= 1.0
    # The magnitude rides beside the rate, because a symmetric perturbation
    # gives a rate near one half whatever its size.
    assert (read["moved_swing"] or 0.0) > 0.0
    # The mask has to be the read's own, per draw: a token counted as moved in
    # a draw that left it alone would put the upstream effect into the route's.
    with alternative_routes(body, seed=0, scale=1.0, subset=None, device=DEVICE) as wrappers:
        drawn = read_probe(body, batches, DEVICE, wrappers)
    assert drawn.rerouted is not None
    assert drawn.rerouted.shape == executed.log_probs.shape
    assert read_probe(body, batches, DEVICE).rerouted is None


def test_a_rate_with_too_few_pairs_behind_it_is_not_reported_as_a_rate(body, batches):
    """Under a full reroute the unmoved population is empty, and an empty rate is not one.

    The recorded every-token read had `moved_fraction` exactly 1.0, so its
    `unmoved_better` stood on nothing and was tabled anyway. `None` is what a
    population under `MIN_POPULATION` now produces, so `compare_runs.py` cannot
    read it as a number.
    """
    from counterfactual_routing import MIN_POPULATION

    executed = read_probe(body, batches, DEVICE)

    read = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=2, scale=8.0, subset=None, seed=0
    )

    assert read["unmoved_pairs"] < MIN_POPULATION
    assert read["unmoved_better"] is None
    assert read["unmoved_swing"] is None


def test_a_read_with_no_alternatives_is_the_floor_pass_and_does_not_divide_by_zero(body, batches):
    """`--alternatives 0` is step 1 of #58's own procedure; it must still run."""
    executed = read_probe(body, batches, DEVICE)

    read = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=0, scale=1.0, subset=None, seed=0
    )

    assert math.isnan(read["moved_fraction"])
    assert math.isnan(read["confident_better"])
    assert bool((read["best_minus_executed"] == 0).all())


def test_the_confident_token_guardrail_is_pinned_by_direction_and_denominator(body, batches):
    """#58's second guardrail carries the conclusion, so its definition is pinned.

    Selecting the bottom quartile, dropping the `alternatives` factor from the
    denominator, or counting a tie as an improvement would each leave every
    other test in this file green -- and this instrument has published a wrong
    number twice already, both times caught after the fact.
    """
    executed = read_probe(body, batches, DEVICE)

    # A noise scale small enough to reorder no top-k: no alternative differs
    # from the executed route, so nothing can beat it and a tie is not a win.
    quiet = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=4, scale=1e-9, subset=None, seed=0
    )
    assert quiet["moved_fraction"] == 0.0
    assert quiet["confident_better"] == 0.0

    loud = counterfactual_gaps(
        body, batches, DEVICE, executed, alternatives=4, scale=1.0, subset=None, seed=0
    )
    assert 0.0 < loud["confident_better"] < 1.0

    # The denominator is draws x confident tokens, so a read where every
    # alternative beats the executed route on every token reads exactly 1.0.
    hopeless = ProbeRead(executed.log_probs - 100.0, executed.token_ids, executed.rerouted)
    certain = counterfactual_gaps(
        body, batches, DEVICE, hopeless, alternatives=4, scale=1.0, subset=None, seed=0
    )
    assert certain["confident_better"] == pytest.approx(1.0)
