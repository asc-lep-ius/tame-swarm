"""Which config fields the package actually reads: the scanner behind the no-silent-no-op tests.

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
``config.<field>``, ``<anything>.config.<field>`` or one of the other holder
names below -- and deliberately *not* ``args.<field>``: parsing a CLI flag into a
namespace is not the same as the configuration steering anything, and counting it
would let a flag that goes nowhere satisfy the check. Only *loads* count: a field
that is assigned and never read is exactly the defect.

The match is by attribute name across the whole package, not per config class, so
a name two configs share (``hidden_dim``; ``kp``, ``ki``, ``kd`` and
``derivative_filter_alpha`` between ``SteeringConfig`` and ``PIDConfig``) is
satisfied by a read of either. Resolving the holder's class statically would need
type inference the check does not have; the collision is named here so nobody
mistakes the per-class parametrisation for a per-class guarantee.
"""

import ast
from functools import cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent / "tame"

# Local names that hold a config object across the package. ``mob_config`` and
# ``steering_config`` are how the app and the routes name the two configs they
# hold side by side; ``template`` is the pristine SteeringConfig every goal
# install starts from.
CONFIG_HOLDERS = frozenset({"config", "cfg", "mob_config", "steering_config", "template"})


def attribute_reads(tree: ast.AST) -> set[str]:
    """Attribute names read off anything that plausibly holds a config."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            continue

        base = node.value
        holds_config = (isinstance(base, ast.Attribute) and base.attr in CONFIG_HOLDERS) or (
            isinstance(base, ast.Name) and base.id in CONFIG_HOLDERS
        )
        if holds_config:
            names.add(node.attr)
    return names


@cache
def read_names() -> frozenset[str]:
    """Every attribute read off a config holder anywhere under ``tame/``, parsed once."""
    names: set[str] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names |= attribute_reads(tree)
    return frozenset(names)
