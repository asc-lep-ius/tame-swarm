"""#62's decomposition pinned where it is true by construction, and where it is not.

The seniority fraction has two readings that must come out of the arithmetic
rather than the fixture: 1.0 when the ledger is so spread that no report can
overturn it, and 0.0 on a token whose report does. The identity check is what
says the two terms are the gate the auction ran, and the ``decoupled`` control's
pinned ledger is what says the wealth term reads constant where it should.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_decomposition import (  # noqa: E402
    DIFFERENTIATED,
    QUALITY,
    SENIORITY_LINE,
    floor_between,
    read_run,
)
from synthetic_economy import BASE_CONFIG  # noqa: E402

from mob import PERSISTENCE_DECOUPLED, PERSISTENCE_VALUE  # noqa: E402
from mob.gate_decomposition import decompose, merge, wealth_top_k  # noqa: E402

TOP_K = 2


def gate_of(confidences: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
    """The winner set the auction would sell, computed the way the auction does."""
    return torch.topk(confidences * wealth.unsqueeze(0).unsqueeze(0), TOP_K, dim=-1).indices


def test_the_two_terms_are_the_gate_and_the_fractions_sum_to_one():
    torch.manual_seed(0)
    confidences = torch.nn.functional.softplus(torch.randn(3, 5, 8))
    wealth = torch.tensor([15.0, 40.0, 75.0, 120.0, 300.0, 500.0, 749.0, 750.0])
    reading = decompose(confidences, wealth, gate_of(confidences, wealth), TOP_K)
    assert reading.tokens == 15
    assert reading.identity_gap < 8 * torch.finfo(torch.float32).eps
    assert reading.degenerate_fraction == 0.0
    assert math.isclose(
        reading.confidence_fraction + reading.wealth_fraction + reading.cross_fraction,
        1.0,
        abs_tol=1e-9,
    )
    assert reading.wealth_term_spread == pytest.approx(math.log(750.0 / 15.0))


def test_a_spread_ledger_makes_seniority_total_by_construction():
    """Two cells at the ceiling and six at the floor: no report in range can overturn it."""
    torch.manual_seed(1)
    confidences = torch.nn.functional.softplus(torch.randn(2, 16, 8))
    wealth = torch.tensor([15.0, 15.0, 750.0, 15.0, 15.0, 750.0, 15.0, 15.0])
    reading = decompose(confidences, wealth, gate_of(confidences, wealth), TOP_K)
    assert reading.undecidable_fraction == 0.0
    assert reading.seniority_fraction == 1.0
    assert reading.sold_seniority_fraction == 1.0
    assert reading.wealth_fraction > reading.confidence_fraction


def test_a_flat_ledger_lets_the_report_decide_and_seniority_reads_zero():
    """Wealth one float apart still ranks, and a report that inverts it wins the token."""
    wealth = torch.tensor([100.0, 100.0 + 1e-3, 100.0 + 2e-3, 100.0 + 3e-3])
    confidences = torch.tensor([[[4.0, 3.0, 1.0, 1.0]]])
    selected = gate_of(confidences, wealth)
    reading = decompose(confidences, wealth, selected, TOP_K)
    assert reading.undecidable_fraction == 0.0
    assert reading.seniority_fraction == 0.0
    assert set(selected.flatten().tolist()) == {0, 1}
    assert reading.confidence_fraction > 0.99


def test_a_pinned_ledger_is_undecidable_and_its_wealth_term_reads_constant():
    """The ``decoupled`` control: every cell at the same wealth, nothing for the ledger to rank."""
    confidences = torch.nn.functional.softplus(torch.randn(2, 4, 8))
    wealth = torch.full((8,), 75.0)
    reading = decompose(confidences, wealth, gate_of(confidences, wealth), TOP_K)
    assert reading.undecidable_fraction == 1.0
    assert math.isnan(reading.seniority_fraction)
    assert reading.wealth_term_spread == 0.0
    assert reading.wealth_fraction == 0.0
    assert reading.confidence_fraction == pytest.approx(1.0)


def test_wealth_top_k_is_undecidable_exactly_when_the_kth_rank_ties():
    _, decidable = wealth_top_k(torch.tensor([5.0, 3.0, 3.0, 1.0]), 2)
    assert not decidable
    top, decidable = wealth_top_k(torch.tensor([5.0, 3.0, 2.0, 1.0]), 2)
    assert decidable
    assert set(top.tolist()) == {0, 1}
    _, decidable = wealth_top_k(torch.tensor([1.0, 1.0]), 2)
    assert decidable


def test_a_mask_drops_padding_and_merge_weights_by_tokens():
    torch.manual_seed(2)
    confidences = torch.nn.functional.softplus(torch.randn(1, 6, 8))
    wealth = torch.tensor([15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 750.0])
    mask = torch.tensor([[True, True, True, False, False, False]])
    kept = decompose(confidences, wealth, gate_of(confidences, wealth), TOP_K, mask=mask)
    assert kept.tokens == 3
    a = decompose(confidences[:, :2], wealth, gate_of(confidences[:, :2], wealth), TOP_K)
    b = decompose(confidences[:, 2:3], wealth, gate_of(confidences[:, 2:3], wealth), TOP_K)
    merged = merge([a, b])
    assert merged.tokens == 3
    assert merged.seniority_fraction == pytest.approx(kept.seniority_fraction)
    assert merged.confidence_fraction == pytest.approx(kept.confidence_fraction)


def test_merge_skips_undecidable_parts_when_weighting_seniority():
    torch.manual_seed(3)
    confidences = torch.nn.functional.softplus(torch.randn(1, 4, 8))
    spread = torch.tensor([15.0, 15.0, 750.0, 15.0, 15.0, 750.0, 15.0, 15.0])
    pinned = torch.full((8,), 75.0)
    ranked = decompose(confidences, spread, gate_of(confidences, spread), TOP_K)
    flat = decompose(confidences, pinned, gate_of(confidences, pinned), TOP_K)
    merged = merge([ranked, flat])
    assert merged.seniority_fraction == 1.0
    assert merged.undecidable_fraction == 0.5


def test_decompose_refuses_an_empty_token_set():
    confidences = torch.ones(1, 2, 4)
    with pytest.raises(ValueError, match="no tokens"):
        decompose(
            confidences,
            torch.ones(4),
            torch.zeros(1, 2, 2, dtype=torch.long),
            2,
            mask=torch.zeros(1, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize("fixture", [QUALITY, DIFFERENTIATED])
def test_a_short_fixture_run_passes_its_identity_and_guardrail_checks(fixture: str):
    reading = read_run(fixture, PERSISTENCE_VALUE, seed=0, steps=24, tail=8, horizon=8)
    tail = reading["tail"]
    assert tail["tokens"] == 8 * 2 * 16
    assert tail["identity_gap"] < 8 * torch.finfo(torch.float32).eps
    assert 0.0 <= tail["undecidable_fraction"] <= 1.0
    assert len(reading["course"]) == 3
    assert reading["fingerprint"]["persistence_coupling"] == PERSISTENCE_VALUE
    assert reading["fingerprint"]["num_experts"] == BASE_CONFIG.num_experts
    guardrail = reading["guardrail"]
    assert guardrail["guardrail/cell_steps"] == 8 * BASE_CONFIG.num_experts
    assert 0.0 <= guardrail["guardrail/ceiling_occupancy"] <= 1.0


def test_the_decoupled_arm_reads_a_constant_wealth_term_on_the_fixture():
    reading = read_run(QUALITY, PERSISTENCE_DECOUPLED, seed=0, steps=12, tail=4, horizon=4)
    assert reading["tail"]["wealth_term_spread"] == 0.0
    assert reading["tail"]["undecidable_fraction"] == 1.0


def test_the_fixture_is_its_own_replicate_so_the_floor_is_zero():
    first = read_run(QUALITY, PERSISTENCE_VALUE, seed=0, steps=12, tail=4, horizon=4)
    second = read_run(QUALITY, PERSISTENCE_VALUE, seed=0, steps=12, tail=4, horizon=4)
    floor = floor_between(first, second)
    assert floor["seniority_fraction"] == 0.0 or math.isnan(floor["seniority_fraction"])
    assert floor["confidence_fraction"] == 0.0


def test_the_line_is_the_one_the_issue_fixed_before_the_read():
    assert SENIORITY_LINE == 0.90
