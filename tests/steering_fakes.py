"""Deterministic model/tokenizer stubs for the behavioural-steering tests.

Real steering extraction and validation read activations at a specific token and
score teacher-forced log-probs. Testing that machinery needs a model whose
activations and logits are a *known* function of the input, not a randomly
initialised transformer whose behaviour is unpredictable. These stubs make the
position read and the logit shift exactly predictable, so a test can assert the
mechanism rather than hope an effect appears.
"""

import torch
import torch.nn as nn


class _Batch(dict):
    def to(self, _device):
        return self


class SimpleCharTokenizer:
    """One character, one token. No padding, so completion spans are real.

    ``id = ord(char) % vocab_size``. The point is that ``tokenizer(prompt)`` and
    ``tokenizer(prompt + completion)`` return different lengths with the prompt as
    a clean prefix -- which is exactly what the completion-position logic slices
    against, and exactly what the padding-to-fixed-length fixture tokenizer cannot
    provide.
    """

    all_special_tokens = ["<pad>"]
    pad_token_id = 0
    eos_token_id = None

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

    def decode(self, ids) -> str:
        # Exact inverse only when vocab_size covers ASCII (>=128); enough for the
        # greedy-decode tests, which use a 128-token vocabulary.
        return "".join(chr(int(i)) for i in ids)

    def __call__(self, text, return_tensors=None, max_length=None, truncation=False, **_):
        if isinstance(text, list):
            text = text[0]
        ids = [ord(char) % self.vocab_size for char in text]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return _Batch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


class _IdentityBlock(nn.Module):
    # Returns a 2-tuple like a real decoder layer (hidden, plus extras), so a
    # SteeringHook attached here returns a tuple rather than a bare tensor -- the
    # contract the model's own ``layer(hidden)[0]`` consumption depends on.
    def forward(self, hidden_states, **_):
        return (hidden_states, None)


class MonotonicModel(nn.Module):
    """Layer activation at position p encodes the token id at p in component 0.

    The embedding is a ramp -- ``embed(id)[0] == id`` -- and the layers are
    identity, so the activation a forward hook reads at position p reveals which
    token sat there. A test can therefore assert *which position* the extractor
    read from, which is the whole claim of the completion-position change.
    """

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 8, num_layers: int = 2):
        super().__init__()
        self.model = nn.Module()
        embed = nn.Embedding(vocab_size, hidden_dim)
        with torch.no_grad():
            embed.weight.zero_()
            embed.weight[:, 0] = torch.arange(vocab_size, dtype=torch.float32)
        self.model.embed_tokens = embed
        self.model.layers = nn.ModuleList(_IdentityBlock() for _ in range(num_layers))

    def forward(self, input_ids, **_):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


class _Output:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class ScriptedModel(nn.Module):
    """Logits are a fixed linear readout of the last hidden state.

    ``logits = hidden @ unembed.T``. Because the layers are identity, a direction
    injected at a layer by a steering hook adds straight into ``hidden``, so a
    direction aligned with ``unembed[pos] - unembed[neg]`` provably raises the
    positive token's log-prob over the negative's -- letting a test assert that a
    constructed vector beats matched random directions, not merely that something
    moved.
    """

    def __init__(
        self, vocab_size: int = 32, hidden_dim: int = 16, num_layers: int = 2, seed: int = 0
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList(_IdentityBlock() for _ in range(num_layers))
        self.unembed = nn.Linear(hidden_dim, vocab_size, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(
                torch.randn(vocab_size, hidden_dim, generator=generator)
            )
            self.unembed.weight.copy_(torch.randn(vocab_size, hidden_dim, generator=generator))

    def forward(self, input_ids, **_):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return _Output(self.unembed(hidden))

    def token_readout(self, token_id: int) -> torch.Tensor:
        return self.unembed.weight[token_id].detach().clone()
