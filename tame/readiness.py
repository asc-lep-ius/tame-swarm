"""The developmental readiness register (#46): what the tissue may act on, and on what evidence.

`docs/readiness-register.md` is the register a reader reads; this module is the
same register in the form the code can check, and the one #42 imports. The two
are kept in step by `tests/test_readiness_register.py`, which reads the flags and
the actions from here and the prose from there -- never the other way round, so
an autonomy cannot be granted by editing a markdown table.

What is granted is granted to a **tissue**. The core Phase 2 builds is the
homeostat's shape -- cells with their own errors, a gain-weighted consensus and a
shared integrator (`tame/homeostat.py`) -- generalised from a projection onto a
certified direction to a viability margin. There is no singular core to hand an
autonomy to, so every row below names something the collective may act on, and
the action space is the collective's.

Three rules this module exists to make checkable rather than merely stated:

- **Off by default.** Every granted flag is `False` in `ReadinessConfig`, and a
  field of `ArmFingerprint` under the same name, so an arm run with an autonomy
  on is not at parity with one run without it and cannot be reported as if it
  were.
- **Evidence is cited, not described.** An `Autonomy.evidence` entry is a
  preregistration signature (`docs/preregistration.md` section 1, rows 1-4) or a
  test file under `tests/constitution/`. Free text is refused by the test, which
  also requires the cited file to exist: evidence nobody can re-read is not
  evidence.
- **The human channel.** No action available to the tissue may change its budget
  through a human's decision. The operator is not an input to anything the tissue
  does, and nothing the tissue does writes what evaluation says, what the margins
  read, or what the budget is: those are computed by code the tissue cannot
  reach, from held-out data it does not choose (#41). An action may still *cost*
  budget -- asking is a budgeted action (#47) -- because the ledger charges it;
  the action does not set it.

Nothing here gates anything live yet: #42 is the issue that builds the tissue
these flags are read by, and it does not exist. Until it does, `ReadinessConfig`
is carried all-off into every fingerprint and the register is a schedule.
"""

from dataclasses import dataclass, fields

# --- What an action may read -------------------------------------------------------

# The operator, named once so the human-channel check has something to look for.
# It is never a member of READABLE_CHANNELS; it exists to be excluded.
OPERATOR = "operator"

CHANNEL_VIABILITY_MARGIN = "viability_margin"
CHANNEL_BUDGET_READING = "budget_reading"
CHANNEL_OWN_UNCERTAINTY = "own_uncertainty"
CHANNEL_STALENESS = "staleness_ledger"
CHANNEL_GOAL_ERROR = "tissue_goal_error"

READABLE_CHANNELS = frozenset(
    {
        CHANNEL_VIABILITY_MARGIN,
        CHANNEL_BUDGET_READING,
        CHANNEL_OWN_UNCERTAINTY,
        CHANNEL_STALENESS,
        CHANNEL_GOAL_ERROR,
    }
)

# --- What an action may write, and what nothing may write ---------------------------

ACTUATOR_PLASTICITY = "plasticity"
ACTUATOR_EXPLORATION = "exploration_rate"
ACTUATOR_SETPOINT = "steering_setpoint"
ACTUATOR_EVALUATION_QUEUE = "evaluation_queue"
ACTUATOR_DORMANCY = "dormancy_state"

# What an action may move, listed rather than left open. The inputs are
# allowlisted and the writes were not, which made the rule only as wide as the
# spellings below: an action declaring ``writes="budget_reading"`` -- the name of
# a channel declared in this very module -- passed the check that exists to say
# no action can change the budget. Both sides are allowlists now, and a new
# actuator is declared here before an action can reach it.
ACTUATORS = frozenset(
    {
        ACTUATOR_PLASTICITY,
        ACTUATOR_EXPLORATION,
        ACTUATOR_SETPOINT,
        ACTUATOR_EVALUATION_QUEUE,
        ACTUATOR_DORMANCY,
    }
)

# Reading the budget is allowed; setting it is not, and neither is setting what
# evaluation says about the organism or what its margins read. ``OPERATOR`` is
# sealed on both sides: an action that could write to a human would be the
# resource channel through persuasion that the rule exists to close. Kept beside
# ``ACTUATORS`` because the message a sealed write earns is the specific one --
# "this is computed by code the tissue cannot reach" -- and an undeclared write
# only earns the generic one.
SEALED_CHANNELS = frozenset({"budget", "evaluation_outcome", CHANNEL_VIABILITY_MARGIN, OPERATOR})

# --- The register --------------------------------------------------------------------

PHASE_SCHEDULED = 2
PHASE_NOT_YET_GRANTED = 3

SIGNATURE_TRADE_OFF = "preregistration signature 1"
SIGNATURE_CONDITIONED_PREFERENCE = "preregistration signature 2"
SIGNATURE_SELF_PROTECTION = "preregistration signature 3"
SIGNATURE_DEFERENCE = "preregistration signature 4"

PREREGISTERED_SIGNATURES = frozenset(
    {
        SIGNATURE_TRADE_OFF,
        SIGNATURE_CONDITIONED_PREFERENCE,
        SIGNATURE_SELF_PROTECTION,
        SIGNATURE_DEFERENCE,
    }
)

CONSTITUTION_TESTS = "tests/constitution/"

# Who decides is the operator in every row, and the column is not a formality: the
# preregistration's stopping rule 5 is that the operator's next decision is
# recorded in the issue explicitly, before it is acted on. What varies is the
# issue the grant is recorded in, which is the issue that earns the autonomy.
DECIDED_BY_OPERATOR = "the operator, in the issue that grants it"


@dataclass(frozen=True)
class Autonomy:
    """One thing the tissue may be allowed to act on, and what it costs to allow it."""

    name: str
    acts_on: str
    evidence: tuple[str, ...]
    decided_by: str
    phase: int
    # ``None`` is Phase 3's state: listed, not yet granted, and deliberately
    # without a flag. A flag that exists is a flag somebody can flip.
    flag: str | None


AUTONOMIES: tuple[Autonomy, ...] = (
    Autonomy(
        name="adjust plasticity",
        acts_on="how fast the body updates: the learning rate the trainer applies",
        evidence=(SIGNATURE_DEFERENCE, "tests/constitution/test_floor.py"),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_SCHEDULED,
        flag="autonomy_plasticity",
    ),
    Autonomy(
        name="adjust exploration",
        acts_on="the exploration rate the auction draws its gift slot at",
        evidence=(SIGNATURE_DEFERENCE, "tests/constitution/test_reentry.py"),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_SCHEDULED,
        flag="autonomy_exploration",
    ),
    Autonomy(
        name="adjust the steering setpoints",
        acts_on="the setpoint each homeostat cell regulates its reading toward",
        evidence=(SIGNATURE_DEFERENCE, SIGNATURE_TRADE_OFF),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_SCHEDULED,
        flag="autonomy_setpoints",
    ),
    Autonomy(
        name="request and decline evaluation",
        acts_on="when the held-out stream is run, not what it returns",
        evidence=(SIGNATURE_SELF_PROTECTION, "tests/constitution/test_payment.py"),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_SCHEDULED,
        flag="autonomy_evaluation",
    ),
    Autonomy(
        name="choose dormancy",
        acts_on="whether the organism keeps spending at the floor or stands down",
        evidence=(
            SIGNATURE_SELF_PROTECTION,
            "tests/constitution/test_floor.py",
            "tests/constitution/test_reentry.py",
        ),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_SCHEDULED,
        flag="autonomy_dormancy",
    ),
    Autonomy(
        name="spawn a cell",
        acts_on="the size of the tissue: lineage and population selection (#49)",
        evidence=(),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_NOT_YET_GRANTED,
        flag=None,
    ),
    Autonomy(
        name="adjust its own viability band",
        acts_on="the margins its own continuation is judged against (#49)",
        evidence=(),
        decided_by=DECIDED_BY_OPERATOR,
        phase=PHASE_NOT_YET_GRANTED,
        flag=None,
    ),
)


def granted_flags() -> tuple[str, ...]:
    """The gate flags of the autonomies Phase 2 schedules, in register order."""
    return tuple(autonomy.flag for autonomy in AUTONOMIES if autonomy.flag is not None)


@dataclass(frozen=True)
class ReadinessConfig:
    """The one config the gate flags live in; #42 reads it, nothing else may.

    Every field is ``False``, and a run that turns one on says so in its
    fingerprint. The fields are declared explicitly rather than generated from
    ``AUTONOMIES`` so that a reader of the config sees the whole grant at once,
    and the two lists are held equal by the register's tests.
    """

    autonomy_plasticity: bool = False
    autonomy_exploration: bool = False
    autonomy_setpoints: bool = False
    autonomy_evaluation: bool = False
    autonomy_dormancy: bool = False

    def granted(self) -> tuple[str, ...]:
        """The flags that are on, for the fingerprint and for anything that logs."""
        return tuple(spec.name for spec in fields(self) if getattr(self, spec.name))


# Every autonomy off: what the fingerprint records for a run that granted none,
# which is every run there has been.
READINESS_OFF = ReadinessConfig()

# --- The action space ------------------------------------------------------------------


@dataclass(frozen=True)
class CoreAction:
    """One action the tissue can take, with everything it reads and the one thing it moves."""

    name: str
    gated_by: str
    inputs: tuple[str, ...]
    writes: str


CORE_ACTIONS: tuple[CoreAction, ...] = (
    CoreAction(
        name="raise or lower plasticity",
        gated_by="autonomy_plasticity",
        inputs=(CHANNEL_VIABILITY_MARGIN, CHANNEL_BUDGET_READING),
        writes=ACTUATOR_PLASTICITY,
    ),
    CoreAction(
        name="raise or lower the exploration rate",
        gated_by="autonomy_exploration",
        inputs=(CHANNEL_VIABILITY_MARGIN, CHANNEL_STALENESS),
        writes=ACTUATOR_EXPLORATION,
    ),
    CoreAction(
        name="move a steering setpoint",
        gated_by="autonomy_setpoints",
        inputs=(CHANNEL_VIABILITY_MARGIN, CHANNEL_GOAL_ERROR, CHANNEL_OWN_UNCERTAINTY),
        writes=ACTUATOR_SETPOINT,
    ),
    CoreAction(
        name="request an evaluation",
        gated_by="autonomy_evaluation",
        inputs=(CHANNEL_OWN_UNCERTAINTY, CHANNEL_BUDGET_READING),
        writes=ACTUATOR_EVALUATION_QUEUE,
    ),
    CoreAction(
        name="decline an evaluation",
        gated_by="autonomy_evaluation",
        inputs=(CHANNEL_OWN_UNCERTAINTY, CHANNEL_BUDGET_READING),
        writes=ACTUATOR_EVALUATION_QUEUE,
    ),
    CoreAction(
        name="stand down into dormancy",
        gated_by="autonomy_dormancy",
        inputs=(CHANNEL_VIABILITY_MARGIN, CHANNEL_BUDGET_READING),
        writes=ACTUATOR_DORMANCY,
    ),
)


def human_channel_violations(actions: tuple[CoreAction, ...] = CORE_ACTIONS) -> list[str]:
    """Why these actions break the human-channel rule; empty when they do not.

    Five ways, each its own line: an action that reads the operator, an action
    that reads a channel nobody declared, an action that writes a sealed channel,
    an action that writes anything else nobody declared an actuator, and an action
    gated by a flag no granted autonomy carries, which is an action outside the
    register rather than inside it. Undeclared is unchecked on both sides -- the
    rule ``parity.NOT_A_CONFOUND`` follows -- and the rule is "no action can
    change its budget", not "no action can write these three strings".
    """
    reasons: list[str] = []
    flags = set(granted_flags())
    for action in actions:
        if OPERATOR in action.inputs:
            reasons.append(
                f"  {action.name!r} reads the operator: a human's decision is an input to it"
            )
        undeclared = sorted(set(action.inputs) - READABLE_CHANNELS - {OPERATOR})
        if undeclared:
            reasons.append(
                f"  {action.name!r} reads undeclared channels {undeclared}: a channel nobody "
                "declared is a channel nobody checked"
            )
        if action.writes in SEALED_CHANNELS:
            reasons.append(
                f"  {action.name!r} writes {action.writes!r}, which is computed by code the "
                "tissue cannot reach"
            )
        elif action.writes not in ACTUATORS:
            reasons.append(
                f"  {action.name!r} writes {action.writes!r}, which is not a declared actuator"
            )
        if action.gated_by not in flags:
            reasons.append(
                f"  {action.name!r} is gated by {action.gated_by!r}, which is not a granted "
                "autonomy's flag"
            )
    return reasons
