"""Greedy-generation outcome check: stopping, answer matching, and the deltas."""

import torch

from contrastive_data import ContrastivePair
from outcome_check import contains_answer, greedy_continue, measure_outcome
from steering import SteeringConfig

from .steering_fakes import ScriptedModel, SimpleCharTokenizer

DEVICE = torch.device("cpu")
VOCAB = 128


def test_contains_answer_matches_whole_words_after_normalisation():
    assert contains_answer("The ball costs 5 cents.", " 5 cents")
    assert contains_answer("It is DAY 47, clearly.", " Day 47")
    assert not contains_answer("It is 170.", " 70")
    assert not contains_answer("anything", "   ")


def test_greedy_continue_stops_at_the_token_budget():
    model, tokenizer = ScriptedModel(vocab_size=VOCAB, hidden_dim=16), SimpleCharTokenizer(VOCAB)
    text, count = greedy_continue(model, tokenizer, "Q: hi\nA:", DEVICE, max_new_tokens=5)
    assert count == 5
    assert len(text) == 5


def test_greedy_continue_stops_at_a_stop_string_and_excludes_it():
    class Scripted(torch.nn.Module):
        """Emits 'ab' then the stop string, regardless of input."""

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            self.script = [ord(c) for c in "ab\nQ:"]
            self.step = 0

        def forward(self, input_ids, **_):
            logits = torch.full((1, input_ids.shape[1], VOCAB), -10.0)
            logits[0, -1, self.script[min(self.step, len(self.script) - 1)]] = 10.0
            self.step += 1

            class Out:
                pass

            out = Out()
            out.logits = logits
            return out

    text, count = greedy_continue(Scripted(), SimpleCharTokenizer(VOCAB), "x", DEVICE, 16)
    assert text == "ab"
    assert count == 2


def test_measure_outcome_reports_length_and_accuracy_deltas():
    model = ScriptedModel(vocab_size=VOCAB, hidden_dim=16, seed=3)
    tokenizer = SimpleCharTokenizer(VOCAB)
    pairs = [
        ContrastivePair(prompt="Q: a?\nA:", positive_completion=" 5", negative_completion=" 7"),
        ContrastivePair(prompt="Q: b?\nA:", positive_completion=" 9", negative_completion=" 1"),
    ]
    # Steer hard toward the stop character so steered generations end at once. The
    # hook normalises the direction, so the push comes from the strength.
    direction = model.token_readout(ord("\n"))
    directions = {0: direction / direction.norm()}
    config = SteeringConfig(
        steering_layers=[0],
        base_strength=40.0,
        max_strength=40.0,
        adaptive=False,
        orthogonal_projection=False,
    )
    result = measure_outcome(
        model,
        tokenizer,
        directions,
        pairs,
        config,
        DEVICE,
        "probe",
        max_new_tokens=6,
        stop_strings=("\n",),
    )
    assert result.num_questions == 2
    assert result.baseline_length == 6.0
    assert result.steered_length == 0.0
    assert result.length_delta == -6.0
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert result.accuracy_delta == result.steered_accuracy - result.baseline_accuracy


def test_eos_ids_accepts_int_list_or_nothing():
    from outcome_check import _eos_ids

    class Tok:
        eos_token_id = [7, 9]
        pad_token_id = 0

    assert _eos_ids(Tok()) == {7, 9}
    Tok.eos_token_id = 3
    assert _eos_ids(Tok()) == {3}
    Tok.eos_token_id = None
    assert _eos_ids(Tok()) == set(), "pad alone is not an end-of-text signal"
