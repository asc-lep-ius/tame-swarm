"""The welfare floor (#38's third property): at the steady state no expert's share
of the slots falls below what the exploration slot alone hands it, and the state
that fails it is a rate of zero."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from economy_damage import BASE_CONFIG, steady, window  # noqa: E402
from synthetic_economy import DEFAULT_COMPETENCE, shuffled  # noqa: E402

READ_WINDOW = 300


def _floor(config) -> float:
    """#16's closed form: the share the exploration slot hands each shut-out expert
    when the shut-out are equally stale, which at the steady state they are."""
    return config.exploration_rate / config.top_k / (config.num_experts - config.top_k)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_expert_holds_less_than_the_exploration_floor(seed):
    """#16 measured 0.0017 as the share six of eight experts *hold*; #38 makes it
    the share none can fall below, because the staleness draw only moves gifts
    toward the more starved and never removes the lottery. Read over 300 steady
    steps (19,200 slots, ~32 gifts per shut-out expert), and against half the
    closed form, since the gifts are a Poisson count and #16 measured
    0.0009-0.0046 across bands and seeds on shorter windows."""
    economy, _ = steady(shuffled(DEFAULT_COMPETENCE, seed), seed)
    _, share = window(economy, READ_WINDOW)

    least = share.min().item()
    assert least >= 0.5 * _floor(economy.config), (least, _floor(economy.config))


def test_with_no_exploration_an_expert_falls_below_the_floor():
    """The inert state: at a rate of zero nothing hands a shut-out expert a token,
    and its share is whatever the auction happens to give it -- one slot in
    19,200 here, a tenth of what the shipped rate guarantees."""
    config = replace(BASE_CONFIG, exploration_rate=0.0)
    economy, _ = steady(shuffled(DEFAULT_COMPETENCE, 0), 0, config)
    _, share = window(economy, READ_WINDOW)

    assert share.min().item() < 0.5 * _floor(BASE_CONFIG)
