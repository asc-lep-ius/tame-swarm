"""The blockade hooks do what they say, and the readouts read what they claim (#63).

Four things a blockade read depends on and none of which a diff shows: that
the output block leaves a contribution of *exactly* zero while touching neither
the head nor the ledger; that the ledger pin holds through a settlement rather
than being overwritten by it; that the gate block silences a bid on one class of
tokens and nothing else; and that the probe pairs -- two economies from
one seed walk the same steps bitwise, so an unblocked twin is a control and not
a different run. The readouts are then checked on hand-built shares where the
answer is arithmetic.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))

import blockade as driver  # noqa: E402
from synthetic_economy import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_COMPETENCE,
    REDUNDANT_COMPETENCE,
    DifferentiatedEconomy,
    SyntheticEconomy,
    shuffled,
)

from individuation import (  # noqa: E402
    dominant_cell,
    gains,
    half_life,
    paired_t,
    planted_statistic,
    predicted_substitute,
    reconvergence_step,
    returned,
    type_shares,
    uptake,
    winners,
)
from mob import PERSISTENCE_DECOUPLED, LightweightExpert  # noqa: E402

torch.set_num_threads(1)
SETTLE = 40
WINDOW = 10


def economy(seed: int = 0, arm: str = "value") -> SyntheticEconomy:
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    return SyntheticEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)


def head_state(economy: SyntheticEconomy) -> list[torch.Tensor]:
    return [p.detach().clone() for p in economy.mob.confidence_heads.parameters()]


def test_importing_the_driver_leaves_the_cuda_device_list_alone():
    """pytest imports every test module at collection.

    An import that hid the card hid it from the GPU suite in the same process
    (pipelines 546 and 549); hiding it is what running the script does.
    """
    probe = (
        "import os, sys; "
        f"sys.path.insert(0, {str(Path(__file__).parent.parent / 'scripts')!r}); "
        f"sys.path.insert(0, {str(Path(__file__).parent.parent / 'tame')!r}); "
        "import blockade; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES'))); "
        "blockade.hide_the_card(); print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"
    )
    # In a subprocess, never in this one: hiding the card here would hide it
    # from every later test this worker runs, which is the bug being pinned.
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    out = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True, env=env
    )
    assert out.stdout.split() == ["None", "''"], out.stdout


def test_output_block_zeroes_the_contribution_exactly_and_touches_nothing_else():
    eco = economy()
    for _ in range(SETTLE):
        eco.step()
    cell = 0
    heads, wealth = head_state(eco), eco.mob.expert_wealth.clone()
    expert = eco.mob.experts[cell]
    assert isinstance(expert, LightweightExpert)
    planted = expert.down_adapter_B.weight.detach().clone()
    assert planted.abs().sum() > 0

    eco.block_output(cell)
    x = torch.randn(4, eco.config.hidden_dim)
    held, reference = expert.forward_with_reference(
        x, eco.mob.base_gate_proj, eco.mob.base_up_proj, eco.mob.base_down_proj
    )
    assert torch.equal(held, reference), "a blocked cell's output must be the base's, bitwise"
    assert all(torch.equal(a, b) for a, b in zip(heads, head_state(eco), strict=True))
    assert torch.equal(wealth, eco.mob.expert_wealth)

    eco.release_output(cell)
    assert torch.equal(expert.down_adapter_B.weight.detach(), planted)


def test_ledger_pin_holds_through_settlement_and_leaves_the_bid_alone():
    eco = economy()
    for _ in range(SETTLE):
        eco.step()
    cell = int(eco.mob.expert_wealth.argmax())
    assert float(eco.mob.expert_wealth[cell]) > eco.config.min_wealth
    heads = head_state(eco)

    eco.pin_wealth(cell)
    assert float(eco.mob.expert_wealth[cell]) == eco.config.min_wealth
    assert all(torch.equal(a, b) for a, b in zip(heads, head_state(eco), strict=True))
    for _ in range(WINDOW):
        record = eco.step()
        assert float(record.selected_experts.numel()) > 0
        assert float(eco.mob.expert_wealth[cell]) == eco.config.min_wealth
    eco.release_wealth(cell)
    for _ in range(WINDOW):
        eco.step()
    # Released, the ledger moves again: the pin was a hold, not a rewrite.
    assert float(eco.mob.expert_wealth[cell]) != eco.config.min_wealth


def test_two_economies_from_one_seed_walk_the_same_steps_bitwise():
    """The pairing every reading rests on: the control is the same tokens, unblocked.

    Run one after the other, never interleaved: the exploration draw reads the
    global stream, which a constructor reseeds, so two economies stepped in
    turn share one stream and diverge -- which is why ``read`` settles one
    economy at a time and never two.
    """
    a = economy(3)
    first = [a.step() for _ in range(SETTLE)]
    b = economy(3)
    second = [b.step() for _ in range(SETTLE)]
    for ra, rb in zip(first, second, strict=True):
        assert ra.loss == rb.loss
        assert torch.equal(ra.selected_experts, rb.selected_experts)
    assert torch.equal(a.mob.expert_wealth, b.mob.expert_wealth)


def test_observing_does_not_move_the_economy():
    """The probe never pays: counting a window leaves the trajectory the plain run's."""
    plain = economy(1)
    losses = [plain.step().loss for _ in range(SETTLE)]
    observed = economy(1)
    window = driver.observe(observed, SETTLE)
    assert window.loss_by_type[0] == pytest.approx(losses)
    assert torch.equal(plain.mob.expert_wealth, observed.mob.expert_wealth)


def test_the_ledger_blockade_does_not_reach_the_allocation_on_decoupled():
    """Under ``decoupled`` the gate reads a pinned snapshot: pinning the ledger changes no route."""
    control = economy(2, PERSISTENCE_DECOUPLED)
    for _ in range(SETTLE):
        control.step()
    routes = [control.step().selected_experts for _ in range(WINDOW)]
    pinned = economy(2, PERSISTENCE_DECOUPLED)
    for _ in range(SETTLE):
        pinned.step()
    pinned.pin_wealth(0)
    for route in routes:
        assert torch.equal(pinned.step().selected_experts, route)


def differentiated(seed: int = 0, arm: str = "value") -> DifferentiatedEconomy:
    config = replace(BASE_CONFIG, persistence_coupling=arm)
    return DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, seed), seed=seed, config=config)


def on_type_share(eco: DifferentiatedEconomy, cell: int, cell_type: int, steps: int) -> float:
    held = 0
    of_type = 0
    for _ in range(steps):
        selected = eco.step().selected_experts
        assert eco.last_types is not None
        mask = eco.last_types == cell_type
        held += int((selected[mask] == cell).sum())
        of_type += int(mask.sum())
    return held / of_type


def test_gate_block_silences_the_cell_on_its_type_and_touches_nothing_else():
    """Blockade (iii): the bid on one class of tokens is zero; head, ledger and output stay."""
    eco = differentiated()
    for _ in range(SETTLE):
        eco.step()
    cell = int(eco.mob.expert_wealth.argmax())
    cell_type = int(eco.expert_types[cell])
    heads, wealth = head_state(eco), eco.mob.expert_wealth.clone()
    expert = eco.mob.experts[cell]
    assert isinstance(expert, LightweightExpert)
    planted = expert.down_adapter_B.weight.detach().clone()

    eco.block_bids(cell, cell_type)
    with pytest.raises(ValueError):
        eco.block_bids(cell, (cell_type + 1) % eco.num_types)
    assert all(torch.equal(a, b) for a, b in zip(heads, head_state(eco), strict=True))
    assert torch.equal(wealth, eco.mob.expert_wealth)
    assert torch.equal(expert.down_adapter_B.weight.detach(), planted)
    # Only the exploration gift can hand a silenced cell a slot on its own type.
    assert on_type_share(eco, cell, cell_type, WINDOW) <= eco.config.exploration_rate * 2
    eco.release_bids(cell)
    assert not eco.mob.gate._forward_pre_hooks, "a released gate carries no hook"


def test_gate_block_with_no_class_silences_the_cell_on_every_token():
    eco = economy()
    for _ in range(SETTLE):
        eco.step()
    cell = int(eco.mob.expert_wealth.argmax())
    eco.block_bids(cell)
    held = sum(int((eco.step().selected_experts == cell).sum()) for _ in range(WINDOW))
    assert held <= eco.config.exploration_rate * 2 * WINDOW * eco.batch_size * eco.seq_len
    with pytest.raises(ValueError):
        eco.block_bids((cell + 1) % eco.config.num_experts, token_type=0)


def test_the_gate_blockade_reaches_the_allocation_on_decoupled():
    """Unlike the ledger pin, silencing the bid moves the route under a pinned snapshot."""
    control = economy(2, PERSISTENCE_DECOUPLED)
    for _ in range(SETTLE):
        control.step()
    routes = [control.step().selected_experts for _ in range(WINDOW)]
    silenced = economy(2, PERSISTENCE_DECOUPLED)
    for _ in range(SETTLE):
        silenced.step()
    cell = int(routes[0].flatten()[0])
    silenced.block_bids(cell)
    assert any(not torch.equal(silenced.step().selected_experts, route) for route in routes)


def test_on_type_counts_split_the_slots_by_the_tokens_type():
    eco = DifferentiatedEconomy(shuffled(DEFAULT_COMPETENCE, 0), seed=0)
    window = driver.observe(eco, WINDOW)
    total_slots = WINDOW * eco.batch_size * eco.seq_len * eco.config.top_k
    assert float(window.wins_by_type.sum()) == total_slots
    assert window.own_type_wins(eco.expert_types).shape == (eco.config.num_experts,)


# --- the readouts, on hand-built shares -------------------------------------------------

PRE = torch.tensor([0.5, 0.48, 0.01, 0.005, 0.005, 0.0, 0.0, 0.0])
COMPETENCE = torch.tensor([0.9, 0.7, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1])
ON_TYPE = torch.ones(8, dtype=torch.bool)


def test_dominant_cell_is_the_one_delivering_the_most_correction():
    wins = torch.tensor([100.0, 100.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert dominant_cell(wins, COMPETENCE) == 0
    # A more competent cell that is shut out does not dominate.
    assert dominant_cell(torch.tensor([0.0, 100.0, 90.0, 0, 0, 0, 0, 0.0]), COMPETENCE) == 1


def test_uptake_is_the_fraction_of_the_blocked_share_released():
    inside = torch.tensor([0.1, 0.48, 0.41, 0.005, 0.005, 0.0, 0.0, 0.0])
    assert uptake(PRE, inside, 0) == pytest.approx(0.8)
    assert uptake(PRE, PRE, 0) == 0.0
    assert uptake(PRE, torch.tensor([0.6, 0.38, 0.01, 0.005, 0.005, 0, 0, 0.0]), 0) < 0
    with pytest.raises(ValueError):
        uptake(PRE, inside, 7)


def test_predicted_substitute_is_the_best_cell_not_already_winning():
    assert winners(PRE, 2) == [0, 1]
    assert predicted_substitute(COMPETENCE, ON_TYPE, PRE, blocked=0, top_k=2) == 2
    # Two cells a type, one blocked and one already winning: nothing to predict.
    two = torch.tensor([True, True, False, False, False, False, False, False])
    assert predicted_substitute(COMPETENCE, two, PRE, blocked=0, top_k=2) is None


def test_planted_statistic_reads_positive_when_the_share_went_by_competence():
    right = gains(PRE, torch.tensor([0.1, 0.48, 0.41, 0.005, 0.005, 0.0, 0.0, 0.0]))
    scattered = gains(PRE, torch.tensor([0.1, 0.48, 0.09, 0.085, 0.085, 0.08, 0.08, 0.0]))
    wrong = gains(PRE, torch.tensor([0.1, 0.48, 0.01, 0.005, 0.005, 0.0, 0.0, 0.4]))
    assert planted_statistic(right, ON_TYPE, PRE, 0, 2, 2) == pytest.approx(0.4)
    scattered_statistic = planted_statistic(scattered, ON_TYPE, PRE, 0, 2, 2)
    assert scattered_statistic is not None and abs(scattered_statistic) < 0.02
    assert planted_statistic(wrong, ON_TYPE, PRE, 0, 2, 2) < 0
    two = torch.tensor([True, True, True, False, False, False, False, False])
    assert planted_statistic(right, two, PRE, 0, 2, 2) is None


def test_returned_and_half_life():
    assert returned(PRE, torch.tensor([0.4, 0.5, 0.1, 0, 0, 0, 0, 0.0]), 0) == pytest.approx(0.8)
    assert half_life([0.5, 0.5, 0.0, 0.0, 0.1, 0.1], before=0.5, final=0.1) == 4
    assert half_life([0.5, 0.5, 0.5], before=0.5, final=0.5) is None


def test_reconvergence_step_reads_the_trailing_mean():
    losses = [1.0] * 10 + [0.2] * 10
    assert reconvergence_step(losses, target=0.2, factor=1.1, trailing=5) == 15
    assert reconvergence_step([1.0] * 10, target=0.2, factor=1.1, trailing=5) is None
    with pytest.raises(ValueError):
        reconvergence_step(losses, 0.2, 1.1, 0)


def test_type_shares_and_paired_t():
    assert type_shares(torch.tensor([3.0, 1.0])).tolist() == [0.75, 0.25]
    assert type_shares(torch.zeros(2)).tolist() == [0.0, 0.0]
    test = paired_t([0.4, 0.5, 0.6, 0.5, 0.45, 0.55])
    assert test.n == 6 and test.mean == pytest.approx(0.5) and test.p < 0.001
    flat = paired_t([0.3, 0.3, 0.3])
    assert flat.sd == 0.0 and flat.p == 0.0


def test_load_stage_reads_the_arm_files_and_not_its_own_summary(tmp_path):
    """A second summarise pass finds the SUMMARY.json the first one wrote, and must skip it."""
    (tmp_path / "value_none.json").write_text('{"readings": {"0": {"uptake": 0.0}}}')
    (tmp_path / "SUMMARY.json").write_text('{"stage": "earlier"}')
    assert driver.load_stage(tmp_path) == {("value", "none"): {"0": {"uptake": 0.0}}}


def test_read_pairs_its_control_and_refuses_an_unpaired_one(monkeypatch):
    monkeypatch.setattr(driver, "SETTLE_STEPS", 120)
    blocked = driver.read(driver.Job(driver.QUALITY, "value", 0, driver.LEDGER, WINDOW))
    control = driver.read(driver.Job(driver.QUALITY, "value", 0, driver.NONE, WINDOW))
    assert blocked["pre_shares"] == control["pre_shares"]
    assert blocked["blocked_cell"] == control["blocked_cell"]
    # The wealthiest cell outside the pre-block winner set, never the blocked cell.
    seated = set(winners(torch.tensor(blocked["pre_shares"]), 2)) | {blocked["blocked_cell"]}
    assert blocked["next_by_wealth"] not in seated
    assert blocked["wealth_hit"] == (blocked["largest_gainer"] == blocked["next_by_wealth"])
    assert len(blocked["wealth_at_settle"]) == 8
    assert blocked["next_by_bid"] not in seated
    assert blocked["bid_hit"] == (blocked["largest_gainer"] == blocked["next_by_bid"])
    assert len(blocked["pre_mean_report"]) == 8 and min(blocked["pre_mean_report"]) >= 0.0
    # The pinned cell bids at the floor from the first blocked step, so it loses
    # nearly everything inside the window: the ledger blockade substitutes at once.
    assert blocked["uptake"] > 0.9
    deltas = driver.paired_against_control({"0": blocked}, {"0": control}, "uptake")
    assert deltas == {"0": pytest.approx(blocked["uptake"] - control["uptake"])}
    other = driver.read(driver.Job(driver.QUALITY, "value", 1, driver.NONE, WINDOW))
    with pytest.raises(AssertionError):
        driver.paired_against_control({"0": blocked}, {"0": other}, "uptake")


# --- stage 5: the redundancy fixture and the born-without target ----------------------------


def test_the_redundancy_fixture_is_the_quality_fixture_with_a_second_top_cell():
    """Two cells of equal top competence, and otherwise the recorded vector."""
    twins = (REDUNDANT_COMPETENCE.max() == REDUNDANT_COMPETENCE).nonzero().flatten().tolist()
    assert len(twins) == 2
    differs = (REDUNDANT_COMPETENCE != DEFAULT_COMPETENCE).nonzero().flatten().tolist()
    assert len(differs) == 1
    assert float(DEFAULT_COMPETENCE[differs[0]]) == pytest.approx(0.3)

    built = driver.build(driver.REDUNDANCY, "value", 0, 1.0)
    assert isinstance(built, SyntheticEconomy) and not isinstance(built, DifferentiatedEconomy)
    assert torch.equal(built.competence, shuffled(REDUNDANT_COMPETENCE, 0))
    assert torch.equal(driver.expert_types(built), torch.zeros(8, dtype=torch.long))
    with pytest.raises(ValueError, match="differentiated"):
        driver.build(driver.REDUNDANCY, "value", 0, 2.0)


def test_the_predicted_substitute_on_the_redundancy_fixture_is_the_shut_out_twin():
    """With one twin seated beside the 0.7, the best cell not already winning is the other twin."""
    competence = REDUNDANT_COMPETENCE
    seated, other = 0, 6
    assert float(competence[seated]) == float(competence[other]) == pytest.approx(0.9)
    pre = torch.zeros(8)
    pre[seated], pre[1] = 0.5, 0.5
    on_type = torch.ones(8, dtype=torch.bool)

    assert predicted_substitute(competence, on_type, pre, seated, 2) == other


def test_the_target_is_the_arms_own_and_a_row_blocking_another_cell_is_refused():
    rows = {
        "0": {"blocked_cell": 1, "inside_on_type_loss": 0.011},
        "1": {"blocked_cell": 2, "inside_on_type_loss": 0.030},
    }
    targets = {
        "0": {"blocked_cell": 1, "born_without_on_type_loss": 0.010},
        "1": {"blocked_cell": 2, "born_without_on_type_loss": 0.020},
    }

    read = driver.against_target(rows, targets)

    assert read["reached_target"] == 1
    assert read["gap_to_target"] == {"0": pytest.approx(0.001), "1": pytest.approx(0.010)}
    assert read["mean_gap"] == pytest.approx(0.0055)
    with pytest.raises(AssertionError, match="blocked cell"):
        driver.against_target(rows, {**targets, "1": {**targets["1"], "blocked_cell": 3}})


def test_summarise_reads_the_targets_when_they_exist_and_says_so(tmp_path):
    def reading(blocked: int, loss: float, uptake: float) -> dict:
        return {
            "pre_shares": [0.5, 0.5, 0.0],
            "blocked_cell": blocked,
            "uptake": uptake,
            "wealth_hit": True,
            "bid_hit": False,
            "predicted_hit": False,
            "inside_on_type_loss": loss,
            "returned": 1.0,
            "guardrail/ceiling_occupancy": 0.25,
            "guardrail/floor_occupancy": 0.75,
            "guardrail/r_wealth_competence": 0.8,
        }

    stage = tmp_path / "arms_redundancy-fixture_2seeds"
    stage.mkdir()
    for arm in driver.ARMS:
        for blockade, loss in ((driver.NONE, 0.010), (driver.LEDGER, 0.012)):
            rows = {"0": reading(1, loss, 0.0 if blockade == driver.NONE else 0.9)}
            rows["1"] = reading(1, loss + 0.001, rows["0"]["uptake"])
            (stage / f"{arm}_{blockade}.json").write_text(json.dumps({"readings": rows}))
    targets = {
        arm: {seed: {"blocked_cell": 1, "born_without_on_type_loss": 0.0115} for seed in ("0", "1")}
        for arm in driver.ARMS
    }

    assert driver.stage_fixture(stage) == driver.REDUNDANCY
    without = driver.summarise_stage(stage)
    assert without["targets_read"] is False
    assert "against_target" not in without["against_control"]["ledger/value/inside_on_type_loss"]

    summary = driver.summarise_stage(stage, targets)
    assert summary["targets_read"] is True
    row = summary["against_control"]["ledger/value/inside_on_type_loss"]
    # 0.012 and 0.013 against 1.1 x 0.0115 = 0.01265: one seed reaches the target.
    assert row["against_target"]["reached_target"] == 1
    taken = summary["against_control"]["ledger/value/uptake"]
    assert "against_target" not in taken
    assert taken["substitute"] == {
        "predicted_hits": 0,
        "next_by_wealth_hits": 2,
        "next_by_bid_hits": 0,
    }


def test_summarise_gives_legacy_readings_no_substitute_block(tmp_path):
    """The recorded 1x stages predate the candidate fields and must still summarise."""

    def reading(loss: float, uptake: float) -> dict:
        return {
            "pre_shares": [0.5, 0.5, 0.0],
            "blocked_cell": 1,
            "uptake": uptake,
            "inside_on_type_loss": loss,
            "returned": 1.0,
            "guardrail/ceiling_occupancy": 0.25,
            "guardrail/floor_occupancy": 0.75,
            "guardrail/r_wealth_competence": 0.8,
        }

    stage = tmp_path / "arms_quality-fixture_2seeds"
    stage.mkdir()
    for arm in driver.ARMS:
        for blockade, loss in ((driver.NONE, 0.010), (driver.LEDGER, 0.012)):
            rows = {seed: reading(loss, 0.0 if blockade == driver.NONE else 0.9) for seed in "01"}
            (stage / f"{arm}_{blockade}.json").write_text(json.dumps({"readings": rows}))

    summary = driver.summarise_stage(stage)

    assert summary["targets_read"] is False
    assert "substitute" not in summary["against_control"]["ledger/value/uptake"]


def test_the_target_is_read_without_the_cell_the_window_stage_blocks(monkeypatch):
    """The operator's criterion, measured: the same cell, the same born-without loss."""
    monkeypatch.setattr(driver, "SETTLE_STEPS", 120)
    monkeypatch.setattr(driver, "RECONVERGENCE_CAP", 20)

    window = driver.measure_window(driver.QUALITY, 0, 1.0)
    target = driver.measure_target(driver.TargetJob(driver.QUALITY, "value", 0))

    assert target["blocked_cell"] == window["blocked_cell"]
    assert target["born_without_on_type_loss"] == window["born_without_on_type_loss"]
    assert target["pre_on_type_loss"] == window["pre_on_type_loss"]
