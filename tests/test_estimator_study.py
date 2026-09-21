"""The two properties #57's discriminator rests on, each failing on the code before it.

The setpoint step's second reading was published as "excludes zero" when it
could not have read anything else, and the guard that would have caught it is
one line of arithmetic: imposing a curve's *own* fit is the same fit, so the
excess is exactly zero, and every other imposition can only be worse. Once that
is true, "excludes zero" on the raw excess is a statement about the estimator.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from allocation_shift import setpoint_step_shift  # noqa: E402
from estimator_study import (  # noqa: E402
    fit_first_order_lag,
    imposed_lag_residual,
    within_control_excess,
)

from mob.ledger import PERSISTENCE_DECOUPLED  # noqa: E402

# A first-order approach to 0.4 with a little noise: the shape a step response
# has on this fixture, at the length the recorded stages read.
CURVE = [0.4 - 0.25 * (0.94**step) + 0.004 * ((step % 5) - 2) for step in range(40)]


def test_a_curve_imposed_on_its_own_fit_leaves_exactly_its_own_residual():
    """The identity the whole discriminator rests on, and the one nothing tested.

    ``imposed_lag_residual`` fixes the intercept and tau and refits only the
    amplitude, which is a strict subfamily of the ``(intercept, slope, tau)``
    family ``fit_first_order_lag`` minimises over the same grid. So the excess
    is zero at the curve's own fit and non-negative everywhere else -- which is
    why "excludes zero" on it was arithmetic rather than a reading.
    """
    asymptote, tau, residual = fit_first_order_lag(CURVE)

    assert imposed_lag_residual(CURVE, asymptote, tau) == pytest.approx(residual, abs=1e-12)
    for other_tau in (tau * 2, tau / 2):
        assert imposed_lag_residual(CURVE, asymptote, other_tau) >= residual
    assert imposed_lag_residual(CURVE, asymptote + 0.1, tau) >= residual


def test_a_control_arm_of_one_seed_is_refused_rather_than_reverting():
    """One seed rotates onto itself, so its null is exactly zero and the contrast reverts."""
    asymptote, tau, residual = fit_first_order_lag(CURVE)
    one = {
        PERSISTENCE_DECOUPLED: {
            "0": {
                "signature1/step_asymptote": asymptote,
                "signature1/step_tau": tau,
                "signature1/step_residual": residual,
            }
        }
    }

    with pytest.raises(ValueError, match="no neighbour to be read against"):
        within_control_excess(one, {PERSISTENCE_DECOUPLED: {"0": CURVE}})


def test_the_within_run_readout_names_the_column_a_run_is_missing():
    """It renamed the step columns before checking them, so it named one the run has.

    Every one of #39's recorded groups takes this path -- they carry
    `routing/win_share_e*` and never took a setpoint step -- and the message
    told the reader their run had no win-share column at all.
    """
    dose_only = {"eval/loss": 2.6, "routing/win_share_e0": 0.5, "routing/win_share_e1": 0.5}

    with pytest.raises(ValueError, match="win_share_pre_step_e"):
        setpoint_step_shift(dose_only)
