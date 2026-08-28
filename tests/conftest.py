import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))

from mob import MixtureOfBidders, MoBConfig

TINY_VOCAB_SIZE = 64
TINY_HIDDEN_DIM = 32
TINY_INTERMEDIATE_DIM = 64

TINY_CONFIG = MoBConfig(
    num_experts=2,
    top_k=1,
    hidden_dim=32,
    intermediate_dim=64,
    adapter_rank=4,
    adapter_alpha=4.0,
    use_shared_base=True,
    use_vcg_payments=True,
    use_differentiable_routing=True,
    use_loss_feedback=True,
    use_local_quality=True,
)


@pytest.fixture
def tiny_config():
    return MoBConfig(
        num_experts=2,
        top_k=1,
        hidden_dim=32,
        intermediate_dim=64,
        adapter_rank=4,
        adapter_alpha=4.0,
        use_shared_base=True,
        use_vcg_payments=True,
        use_differentiable_routing=True,
        use_loss_feedback=True,
        use_local_quality=True,
    )


@pytest.fixture
def mob_layer(tiny_config):
    layer = MixtureOfBidders(tiny_config)
    layer.eval()
    return layer


@pytest.fixture
def training_mob_layer(tiny_config):
    layer = MixtureOfBidders(tiny_config)
    layer.train()
    return layer


@pytest.fixture
def random_hidden_states():
    return torch.randn(1, 8, 32)


class FakeTokenizer:
    """The tokenizer surface the evaluation and probe paths actually use.

    A real tokenizer would pull a vocabulary over the network, which no test may
    depend on. This implements exactly the three things the code under test calls:
    encoding a batch of documents to padded tensors, mapping ids back to token
    strings, and reporting its special tokens. Token strings follow the
    SentencePiece leading-space convention so the category mapping is exercised
    rather than bypassed.
    """

    all_special_tokens = ["<pad>"]
    pad_token_id = 0

    def __call__(
        self,
        texts,
        truncation=True,
        max_length=16,
        padding="max_length",
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            # Deterministic in the text, so the same document always tokenises to
            # the same ids -- which is what the split fingerprint is asserting.
            ids = [1 + (ord(character) % (TINY_VOCAB_SIZE - 1)) for character in text][:max_length]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_ids_to_tokens(self, ids):
        pieces = []
        for token_id in ids:
            if token_id == self.pad_token_id:
                pieces.append("<pad>")
            elif token_id % 7 == 0:
                pieces.append("▁word")
            elif token_id % 7 == 1:
                pieces.append("ing")
            elif token_id % 7 == 2:
                pieces.append("▁42")
            elif token_id % 7 == 3:
                pieces.append(",")
            else:
                pieces.append("▁other")
        return pieces


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def build_tiny_causal_lm(num_layers: int = 4):
    """A randomly initialised Llama small enough to train on a CPU in a test."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=TINY_HIDDEN_DIM,
        intermediate_size=TINY_INTERMEDIATE_DIM,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


@pytest.fixture
def tiny_causal_lm():
    torch.manual_seed(0)
    return build_tiny_causal_lm()


@pytest.fixture
def tiny_mob_config():
    return MoBConfig(
        num_experts=3,
        top_k=2,
        hidden_dim=TINY_HIDDEN_DIM,
        intermediate_dim=TINY_INTERMEDIATE_DIM,
        adapter_rank=4,
        adapter_alpha=4.0,
    )


# A dataset with blank rows (about a third, as wikitext has) and a validation
# split, so both held-out paths are exercised without touching the network.
TRAIN_ROWS = [{"text": f"train document number {i}" if i % 3 else ""} for i in range(600)]
VALIDATION_ROWS = [{"text": f"validation document number {i}"} for i in range(40)]


def fake_load_dataset(*args, split="train", streaming=False):
    """Stands in for ``datasets.load_dataset``. 'splitless' has no validation split."""
    name = args[0]
    if split == "validation":
        if name == "splitless":
            raise ValueError("Unknown split 'validation'")
        return list(VALIDATION_ROWS)
    return iter(TRAIN_ROWS)


@pytest.fixture
def held_out_split(fake_tokenizer):
    from evaluation import build_held_out_split

    return build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 16, fake_load_dataset, num_sequences=8
    )


@pytest.fixture
def padded_held_out_split(fake_tokenizer):
    """A split that actually contains padding, which ``held_out_split`` does not.

    Every conftest document is longer than 16 characters, so at the default length
    ``FakeTokenizer`` truncates and the attention mask comes back all ones -- a
    fixture that cannot fail if padding is scored. At 48 the same documents leave
    roughly 40% of each row as pad, which is the regime a real corpus of short
    paragraphs sits in.
    """
    from evaluation import build_held_out_split

    return build_held_out_split(
        "wikitext", "wikitext-2-raw-v1", fake_tokenizer, 48, fake_load_dataset, num_sequences=8
    )
