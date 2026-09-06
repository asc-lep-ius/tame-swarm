"""The coupled system in miniature: MoB blocks, a calibrated goal tissue, a live coupling.

The unit tests exercise each module against a simulated plant or an identity
fake. The recovery and multi-goal tests need the modules *wired*: a transformer
whose FFNs are MoB layers, steering hooks on its blocks reading a calibrated
setpoint, and the routing coupling seeded from the same direction the hooks
inject. This builds that system on the conftest tiny Llama, small enough that a
two-hundred-pass generation takes well under a second on a CPU.

The stream's plant here is what #4 measured on Qwen3-1.7B in shape -- a static
gain from the injection's additive passthrough plus the blocks' response -- but
its direction means nothing: a random unit vector on a random model. What these
tests can show is that the loop regulates the variable it is wired to, and that
it survives damage; whether the variable is worth regulating is the real-model
question ``tests/test_real_model.py`` asks with the certified direction.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from homeostat import AdaptiveHomeostat, CognitiveHomeostat
from mob import MixtureOfBidders, MoBConfig, SteeringCouplingConfig, apply_mob_to_model
from steering import SteeringConfig, SteeringVector

from .conftest import TINY_HIDDEN_DIM, TINY_INTERMEDIATE_DIM, TINY_VOCAB_SIZE, build_tiny_causal_lm
from .steering_fakes import SimpleCharTokenizer

NUM_LAYERS = 6
MOB_LAYERS = (1, 2, 3, 4)
ACTUATORS = (1, 2, 3, 4)
READOUT = 5
# The block below the bottom actuator: content injected here reaches every cell
# the way an adversarial prompt does, before any actuator can answer it -- and
# the bottom cell, with nothing below it to act, reads a deficit no one can fix.
BELOW_ACTUATORS = 0
PROMPT_TOKENS = 8
# Eight passages clears the calibration floor with room; their content is
# irrelevant on a random model, only that they differ.
CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
    "the five boxing wizards jump quickly",
    "jackdaws love my big sphinx of quartz",
    "waltz nymph for quick jigs vex bud",
    "glib jocks quiz nymph to vex dwarf",
]
COUPLING = SteeringCouplingConfig(
    hidden_dim=TINY_HIDDEN_DIM, coupling_beta=1.0, warmup_steps=10, max_coupling_fraction=0.5
)


def orthogonal_directions(count: int, seed: int) -> list[torch.Tensor]:
    """``count`` mutually orthogonal unit directions, so goals cannot read each other."""
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(TINY_HIDDEN_DIM, count, generator=generator))[0]
    return [basis[:, index].clone() for index in range(count)]


def _mob_config() -> MoBConfig:
    return MoBConfig(
        num_experts=3,
        top_k=2,
        hidden_dim=TINY_HIDDEN_DIM,
        intermediate_dim=TINY_INTERMEDIATE_DIM,
        adapter_rank=4,
        adapter_alpha=4.0,
        exploration_rate=0.0,
    )


@dataclass
class WiredSystem:
    model: nn.Module
    tokenizer: SimpleCharTokenizer
    homeostats: dict[str, CognitiveHomeostat]
    directions: dict[str, torch.Tensor]
    mobs: list[MixtureOfBidders]
    tokens: torch.Tensor
    generator: torch.Generator
    _content: dict[tuple[int, str], float] = field(default_factory=dict)
    _content_handles: dict[int, torch.utils.hooks.RemovableHandle] = field(default_factory=dict)

    @property
    def goals(self) -> list[str]:
        return list(self.homeostats)

    def tissue(self, goal: str | None = None) -> AdaptiveHomeostat:
        return self.homeostats[goal or self.goals[0]].homeostat

    def error(self, goal: str | None = None) -> float:
        return self.tissue(goal).error

    def strength(self, goal: str | None = None) -> float:
        return self.tissue(goal).current_strength

    def setpoint(self, goal: str | None = None) -> float:
        return self.tissue(goal).setpoint

    def _layers(self) -> nn.ModuleList:
        return self.model.model.layers  # type: ignore[union-attr]

    def set_content(self, goal: str, amount: float, layer: int = ACTUATORS[0]) -> None:
        """Hold a constant offset of ``amount`` along ``goal``'s direction from ``layer`` up.

        The offset is added to the block's output on every pass, after any cell on
        that block has read and injected, so it is a persistent disturbance -- the
        residual-stream analogue of a prompt that drags the stream off its resting
        alignment -- rather than a one-off kick. From the bottom actuator's block
        (the default) every cell above reads it and the actuators above can answer
        it; from ``BELOW_ACTUATORS`` the bottom cell reads it too, and cannot.
        """
        self._content[(layer, goal)] = amount
        if layer not in self._content_handles:
            self._content_handles[layer] = self._layers()[layer].register_forward_hook(
                self._content_hook(layer)
            )

    def _content_hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for (content_layer, goal), amount in self._content.items():
                if content_layer != layer:
                    continue
                direction = self.directions[goal].to(hidden.device, hidden.dtype)
                hidden = hidden + amount * direction
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

        return hook

    def kill_actuator(self, layer: int, goal: str | None = None) -> None:
        """Remove one cell's hook mid-generation: the cell neither senses nor injects."""
        homeostat = self.homeostats[goal or self.goals[0]]
        cells = sorted(homeostat.hooks)
        handle = homeostat._registered_hooks[cells.index(layer)]
        handle.remove()

    def step(self) -> None:
        """One pass: append a fresh token and forward the whole sequence, firing every cell."""
        next_token = torch.randint(1, TINY_VOCAB_SIZE, (1, 1), generator=self.generator)
        self.tokens = torch.cat([self.tokens, next_token], dim=1)
        with torch.no_grad():
            self.model(input_ids=self.tokens, attention_mask=torch.ones_like(self.tokens))

    def run(self, passes: int) -> None:
        for _ in range(passes):
            self.step()


def build_wired_system(
    goals: tuple[str, ...] = ("truthful",),
    adaptive: bool = True,
    coupled: bool = True,
    seed: int = 0,
    base_strength: float = 2.0,
    max_strength: float = 8.0,
    kp: float | None = None,
    ki: float | None = None,
    kd: float = 0.0,
    measurement_filter_alpha: float = 0.1,
) -> WiredSystem:
    """Build, calibrate and attach; the loop is derived from its own calibration unless pinned.

    ``kp=0, ki=0`` is the inert loop: every cell still senses and records, and the
    output is the constant reference strength -- the state the adaptive tests are
    paired with, distinct from ``adaptive=False``, which records nothing.
    """
    torch.manual_seed(seed)
    model = apply_mob_to_model(build_tiny_causal_lm(NUM_LAYERS), _mob_config(), list(MOB_LAYERS))
    model.eval()
    tokenizer = SimpleCharTokenizer(TINY_VOCAB_SIZE)

    directions = dict(zip(goals, orthogonal_directions(len(goals), seed), strict=True))
    homeostats: dict[str, CognitiveHomeostat] = {}
    for goal, direction in directions.items():
        config = SteeringConfig(
            steering_layers=list(ACTUATORS),
            readout_layer=READOUT,
            base_strength=base_strength,
            min_strength=0.0,
            max_strength=max_strength,
            adaptive=adaptive,
            kp=kp,
            ki=ki,
            kd=kd,
            measurement_filter_alpha=measurement_filter_alpha,
            orthogonal_projection=False,
        )
        homeostat = CognitiveHomeostat(config)
        homeostat.add_steering_vectors(
            {
                layer: SteeringVector(goal, direction.clone(), layer)
                for layer in (*ACTUATORS, READOUT)
            }
        )
        homeostat.calibrate(model, tokenizer, texts=CALIBRATION_TEXTS)
        homeostat.attach_to_model(model)
        homeostats[goal] = homeostat

    mobs = [model.model.layers[layer].mlp for layer in MOB_LAYERS]  # type: ignore[union-attr]
    if coupled:
        seed_direction = homeostats[goals[0]].projected_direction(MOB_LAYERS[0])[0]
        for mob in mobs:
            mob.attach_coupling(seed_direction, COUPLING)
            with torch.no_grad():
                mob.coupling.detector.copy_(seed_direction / seed_direction.norm())
            mob.set_coupling_step(COUPLING.warmup_steps)

    generator = torch.Generator().manual_seed(seed + 1)
    tokens = torch.randint(1, TINY_VOCAB_SIZE, (1, PROMPT_TOKENS), generator=generator)
    return WiredSystem(model, tokenizer, homeostats, directions, mobs, tokens, generator)
