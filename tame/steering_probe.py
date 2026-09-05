"""Read the residual stream along a direction while injecting a schedule: the plant probe.

The plant characterisation and the layer sweep both need the same instrument: a
teacher-forced pass over a fixed token sequence that records, at chosen layers,
the projection of every position onto a direction, while injecting a
per-position strength schedule at other (or the same) layers. Because attention
is causal, one pass with a schedule that switches on at position ``p`` is
exactly the incremental decode with a step at ``p`` -- the step response of the
plant in a single forward. Readings are taken *before* any injection at the same
layer, which is what a homeostat cell reads at runtime.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def chat_prompt(tokenizer, question: str, **chat_kwargs) -> str:
    """``question`` as a user turn in the served chat format, or verbatim without a template."""
    if not getattr(tokenizer, "chat_template", None):
        return question
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )


def greedy_forced_sequences(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    questions: Sequence[str],
    tokens: int,
    **chat_kwargs,
) -> list[tuple[torch.Tensor, int]]:
    """Unsteered greedy continuations to teacher-force against, with each prompt's length."""
    out = []
    for question in questions:
        ids = tokenizer(chat_prompt(tokenizer, question, **chat_kwargs), return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            generated = model.generate(**ids, max_new_tokens=tokens, do_sample=False)  # pyright: ignore[reportCallIssue] # HF stubs
        out.append((generated, int(ids["input_ids"].shape[1])))
    return out


class ProjectionProbe:
    """Per-position projections onto per-layer directions, with an optional injection schedule."""

    def __init__(self, model: nn.Module, directions: dict[int, torch.Tensor]):
        self.model = model
        self.directions = directions
        self._records: dict[int, torch.Tensor] = {}
        self._inject: dict[int, torch.Tensor] = {}
        self._inject_directions: dict[int, torch.Tensor] = {}

    def _layers(self) -> nn.ModuleList:
        inner = getattr(self.model, "model", self.model)
        return getattr(inner, "layers")  # noqa: B009  # model internals absent on nn.Module stubs

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer in self.directions:
                direction = self.directions[layer].to(hidden.device, hidden.dtype)
                self._records[layer] = (hidden[0] @ direction).detach().float().cpu()
            if layer not in self._inject:
                return output
            schedule = self._inject[layer].to(hidden.device, hidden.dtype)
            direction = self._inject_directions[layer].to(hidden.device, hidden.dtype)
            hidden = hidden + schedule[None, :, None] * direction
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        inject: dict[int, torch.Tensor] | None = None,
        inject_directions: dict[int, torch.Tensor] | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """One teacher-forced pass: projections per recorded layer, log-prob of each next token.

        ``inject`` maps a layer to a per-position strength schedule; the direction
        injected there is ``inject_directions[layer]``, defaulting to the recorded one.
        """
        self._records = {}
        self._inject = dict(inject or {})
        resolved = {
            layer: (inject_directions or {}).get(layer, self.directions.get(layer))
            for layer in self._inject
        }
        missing = [layer for layer, direction in resolved.items() if direction is None]
        if missing:
            raise ValueError(f"no direction to inject at layers {missing}")
        self._inject_directions = {
            layer: direction for layer, direction in resolved.items() if direction is not None
        }

        layers = self._layers()
        handles = [
            layers[layer].register_forward_hook(self._hook(layer))
            for layer in sorted(set(self.directions) | set(self._inject))
        ]
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits
        finally:
            for handle in handles:
                handle.remove()

        log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
        token_log_probs = log_probs.gather(1, input_ids[0, 1:, None])[:, 0].detach().cpu()
        return dict(self._records), token_log_probs


def step_schedule(length: int, start: int, strength: float) -> torch.Tensor:
    """A strength schedule that is zero before ``start`` and ``strength`` from it on."""
    schedule = torch.zeros(length)
    schedule[start:] = strength
    return schedule
