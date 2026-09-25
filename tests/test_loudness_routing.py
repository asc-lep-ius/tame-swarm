"""The fixture port of #58's counterfactual-route read reads what it claims (#64)."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import loudness_routing as port  # noqa: E402
from counterfactual_routing import GumbelTopKRoute  # noqa: E402
from measure_ledger_stability import QUALITY  # noqa: E402

from mob import PERSISTENCE_DECOUPLED, PERSISTENCE_VALUE  # noqa: E402

torch.set_num_threads(1)
SHORT = 60


def _job(arm: str = PERSISTENCE_VALUE, seed: int = 0) -> port.Job:
    return port.Job(
        QUALITY, arm, seed, 8, 1.0, port.RECORDED_RATE, False, alternatives=3, probe_batches=4
    )


def test_the_executed_read_is_reproducible_and_moves_nothing():
    """Two reads of a settled economy agree to the bit and leave its ledger where it was."""
    economy, guardrail = port.settle(_job(), SHORT)
    probe = port.draw_probe(economy, 4)
    wealth = economy.mob.expert_wealth.clone()

    first = port.read(economy, probe)
    second = port.read(economy, probe)

    assert torch.equal(first.losses, second.losses)
    assert torch.equal(first.routes, second.routes)
    assert torch.equal(economy.mob.expert_wealth, wealth)
    assert economy.mob.training, "the read must leave the layer as it found it"
    assert first.tokens == 4 * 2 * 16
    assert set(guardrail) == {
        "guardrail/ceiling_occupancy",
        "guardrail/floor_occupancy",
        "guardrail/cell_steps",
        "guardrail/r_wealth_competence",
    }


def test_the_routes_compare_as_sets():
    """A top-k that lists the same cells in another order is the same route."""
    economy, _ = port.settle(_job(), SHORT)
    probe = port.draw_probe(economy, 2)
    executed = port.read(economy, probe)

    assert torch.equal(executed.routes, executed.routes.sort(dim=-1).values)


def test_an_alternative_that_moves_no_token_reads_the_executed_loss():
    """The fixture has no stream: a token whose route a draw left alone reads the executed loss."""
    economy, _ = port.settle(_job(), SHORT)
    probe = port.draw_probe(economy, 4)
    executed = port.read(economy, probe)
    generator = torch.Generator().manual_seed(7)
    wrapper = GumbelTopKRoute(economy.mob.gate, generator, 1.0, subset=None)
    with port.routed_by(economy, wrapper):
        alternative = port.read(economy, probe)
    moved = (alternative.routes != executed.routes).any(dim=-1)

    assert bool((~moved).any()) and bool(moved.any())
    # Up to the order the same cells' outputs were summed in, never further.
    assert torch.allclose(alternative.losses[~moved], executed.losses[~moved], rtol=1e-5, atol=1e-7)
    assert not torch.allclose(alternative.losses[moved], executed.losses[moved], rtol=1e-3)

    read = port.alternatives_read(economy, probe, executed, 3, 1.0, 7)
    assert read["gap"].min() >= 0.0
    assert 0.0 <= read["moved_fraction"] <= 1.0
    assert read["confident_tokens"] == pytest.approx(0.25 * executed.tokens, abs=1)
    # The conditional rate counts only the confident pairs a draw moved, so
    # the diluted rate is never above it.
    assert read["confident_better"] <= read["confident_moved_better"] + 1e-9


def test_the_floor_is_the_pair_of_gumbel_seeds():
    a = torch.tensor([0.0, 0.1, 0.2, 0.3])
    b = torch.tensor([0.0, 0.1, 0.2, 0.0])

    floor = port.floor_of(a, b)

    assert floor["floor_max"] == pytest.approx(0.3)
    assert floor["floor_mean"] == pytest.approx(0.075)
    # torch's quantile interpolates: the 95th percentile of [0, 0, 0, 0.3] is 0.255.
    assert floor["floor"] == pytest.approx(0.255)


def test_read_one_records_the_reading_and_its_identity():
    reading = port.read_one(_job(), SHORT)

    assert reading["probe_tokens"] == 128
    assert 0.0 <= reading["counterfactual/fragile_fraction"] <= 1.0
    assert reading["counterfactual/gap_on_fragile"] >= reading["floor"] or (
        reading["counterfactual/fragile_fraction"] == 0.0
    )
    assert reading["footprint/adapter_footprint"] > 0.0, "removing the cells must cost the fixture"
    assert reading["fingerprint"]["reward_scale"] == port.RECORDED_RATE
    assert reading["fingerprint"]["reward_scale_derived"] is False
    assert reading["fingerprint"]["contribution_scale"] == 1.0
    assert "code_sha" in reading and "probe_hash" in reading


def test_two_arms_of_one_seed_read_the_same_probe_tokens():
    live, _ = port.settle(_job(PERSISTENCE_VALUE), SHORT)
    pinned, _ = port.settle(_job(PERSISTENCE_DECOUPLED), SHORT)

    assert port.probe_hash(port.draw_probe(live, 2)) == port.probe_hash(port.draw_probe(pinned, 2))


def test_the_scale_contrast_pairs_by_seed_and_refuses_a_different_probe():
    def reading(value: float, probe: str) -> dict:
        return {port.PRIMARY: value, "probe_hash": probe}

    grouped = {
        (PERSISTENCE_VALUE, 8, 1.0): {0: reading(0.1, "a"), 1: reading(0.2, "b")},
        (PERSISTENCE_VALUE, 8, 2.0): {0: reading(0.3, "a"), 1: reading(0.5, "b")},
    }

    contrast = port.scale_contrast(grouped, PERSISTENCE_VALUE, 8, 2.0, port.PRIMARY)

    assert contrast is not None
    assert contrast["mean"] == pytest.approx(0.25)
    assert contrast["per_seed"] == {"0": pytest.approx(0.2), "1": pytest.approx(0.3)}
    grouped[(PERSISTENCE_VALUE, 8, 2.0)][1]["probe_hash"] = "c"
    with pytest.raises(AssertionError, match="probe tokens"):
        port.scale_contrast(grouped, PERSISTENCE_VALUE, 8, 2.0, port.PRIMARY)


def test_the_swing_to_contrast_correlation_recovers_a_planted_line():
    """Six cells whose swing is a line in the contrast: r = 1, both intervals at 1, p at 0."""
    contrasts = {
        (4, 1.0): -1.0,
        (4, 2.0): -2.0,
        (4, 4.0): -3.0,
        (8, 1.0): -0.5,
        (8, 2.0): -1.5,
        (8, 4.0): -2.5,
    }
    grouped = {
        (PERSISTENCE_VALUE, cells, scale): {
            seed: {port.PRIMARY: -dz + 0.001 * seed} for seed in range(3)
        }
        for (cells, scale), dz in contrasts.items()
    }

    read = port.swing_contrast_correlation(grouped, contrasts, resamples=200)

    assert read["cells"] == 6
    assert read["r"] == pytest.approx(-1.0, abs=1e-3)
    assert read["interval"][0] == pytest.approx(-1.0, abs=1e-2)
    assert read["fisher_interval"][1] < -0.9
    assert read["permutation_p"] <= 0.01
    # Too few cells to correlate at all comes back as nan rather than a number.
    few = port.swing_contrast_correlation(dict(list(grouped.items())[:2]), contrasts)
    assert math.isnan(few["r"])
