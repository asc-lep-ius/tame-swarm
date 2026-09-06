"""Every ``TrainingConfig`` field must be read by something.

The scanner lives in ``tests/config_reads.py`` and its rationale with it; this
module holds the ``TrainingConfig`` checks and the construction-time validation.
``MoBConfig`` and ``SteeringConfig`` are covered by ``tests/test_no_silent_noops.py``.
"""

import ast
from dataclasses import fields

import pytest

from train import TrainingConfig

from .config_reads import attribute_reads, read_names


@pytest.mark.parametrize("field", [field.name for field in fields(TrainingConfig)])
def test_training_config_field_is_read(field):
    assert field in read_names(), (
        f"TrainingConfig.{field} is declared and never read. A field that looks "
        "like it steers the experiment and does not is the defect this test "
        "exists to prevent -- wire it up or delete it."
    )


def test_the_check_can_fail(tmp_path):
    """A test that cannot fail is not a test.

    Re-runs the scanner over a module that declares a field and never reads it --
    the exact shape of the ``eval_steps`` defect -- and requires it to come back
    unread. Without this, a scanner that returned every identifier in the package
    would pass the parametrised checks above and detect nothing.
    """
    source = ast.parse(
        "class Trainer:\n"
        "    def run(self, config):\n"
        "        self.declared_and_never_read = 1\n"
        "        config.only_written = 2\n"
        "        return config.actually_read\n"
    )

    names = attribute_reads(source)

    assert "actually_read" in names
    assert "declared_and_never_read" not in names
    assert "only_written" not in names, "an assignment is not a read"
    assert "eval_steps" in read_names()


def test_an_unknown_router_is_rejected_at_construction():
    """argparse guards the CLI; the comparison harness builds configs in code.

    Without this the value survives to ``ARM_ROUTERS[router]`` and surfaces as a
    bare KeyError minutes into a run, after the model is loaded.
    """
    with pytest.raises(ValueError, match="Unsupported router"):
        TrainingConfig(router="mob2")


@pytest.mark.parametrize(
    "field",
    [
        "max_steps",
        "gradient_accumulation_steps",
        "eval_steps",
        "save_steps",
        "log_frequency",
        "probe_tokens",
        "held_out_sequences",
        "wealth_update_frequency",
    ],
)
def test_cadence_and_size_fields_must_be_positive(field):
    """Zero is the dangerous value: ``step % 0`` raises mid-run, not at startup."""
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        TrainingConfig(**{field: 0})
