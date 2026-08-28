"""Every ``TrainingConfig`` field must be read by something.

``eval_steps = 500`` was declared in ``TrainingConfig`` and never read anywhere:
grep returned exactly one hit, the declaration itself. For the life of the field
the training loop looked like it evaluated every 500 steps and did not, and every
number produced in that time was a training-batch statistic wearing a held-out
name. That is a *class* of defect -- configuration that appears to steer an
experiment and does not -- and it is cheap enough to make impossible that there is
no reason to rely on noticing it again.

The check is static rather than a runtime trace: a trace would only cover the code
paths one smoke run happens to take, so a field read exclusively inside, say, the
LoRA branch would look dead. Only reads through a config object count --
``config.<field>`` or ``<anything>.config.<field>`` -- and deliberately *not*
``args.<field>``: parsing a CLI flag into a namespace is not the same as the
configuration steering anything, and counting it would let a flag that goes
nowhere satisfy the check.
"""

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from train import TrainingConfig

PACKAGE_ROOT = Path(__file__).parent.parent / "tame"

# Local names that hold a config object across the package.
CONFIG_HOLDERS = frozenset({"config", "cfg"})


def _attribute_reads(tree: ast.AST) -> set[str]:
    """Attribute names read off anything that plausibly holds a config."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue

        base = node.value
        holds_config = (isinstance(base, ast.Attribute) and base.attr == "config") or (
            isinstance(base, ast.Name) and base.id in CONFIG_HOLDERS
        )
        if holds_config:
            names.add(node.attr)
    return names


def _read_names() -> set[str]:
    names: set[str] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names |= _attribute_reads(tree)
    return names


@pytest.mark.parametrize("field", [field.name for field in fields(TrainingConfig)])
def test_training_config_field_is_read(field):
    assert field in _read_names(), (
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
        "        return config.actually_read\n"
    )

    names = _attribute_reads(source)

    assert "actually_read" in names
    assert "declared_and_never_read" not in names
    assert "eval_steps" in _read_names()


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
