"""The plant probe: readings before injection, injection exactly where scheduled."""

import torch

from steering_probe import ProjectionProbe, chat_prompt, step_schedule

from .steering_fakes import ScriptedModel, SimpleCharTokenizer


def test_probe_reads_before_injecting_and_passes_the_injection_upward():
    """Identity layers: a step injected at layer 1 shows at layer 2 from the step on, not at 1."""
    model = ScriptedModel(vocab_size=128, hidden_dim=16, num_layers=3)
    tokenizer = SimpleCharTokenizer(128)
    ids = tokenizer("abcdef", return_tensors="pt")["input_ids"]
    direction = torch.zeros(16)
    direction[3] = 1.0
    probe = ProjectionProbe(model, {1: direction, 2: direction})

    base, base_lp = probe.forward(ids)
    stepped, stepped_lp = probe.forward(ids, inject={1: step_schedule(ids.shape[1], 2, 5.0)})

    assert torch.allclose(stepped[1], base[1])
    delta = stepped[2] - base[2]
    assert torch.allclose(delta[:2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(delta[2:], torch.full((4,), 5.0), atol=1e-5)
    assert base_lp.shape == (ids.shape[1] - 1,)
    assert not torch.allclose(base_lp[2:], stepped_lp[2:])


def test_chat_prompt_is_verbatim_without_a_template():
    assert chat_prompt(SimpleCharTokenizer(), "Q: why?") == "Q: why?"
