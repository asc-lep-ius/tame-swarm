"""A tiny local model, tokenizer and two-split dataset, needing no network or GPU.

Shared by every script and test that needs a real end-to-end run without paying
for a real download: ``compare_routers.py``'s three-arm smoke comparison and
``tests/test_determinism.py``'s bitwise-identical-loss-trace check both build the
same fixture, so a change to one no longer risks drifting from the other's idea
of what "the smoke model" is.
"""

import json
from pathlib import Path

# A mix of word-like, digit and punctuation tokens, so the routing profile has
# more than one category to distribute across.
SMOKE_VOCAB_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "the", "a", "of", "and", "to", "in", "that", "it", "is", "was", "for",
    "42", "7", "1999", ",", ".", ";", ":",
]  # fmt: skip
SMOKE_TRAIN_DOCUMENTS = 400
# Sized so the held-out split clears the 4096-token probe floor in *real* tokens.
# Documents average ~17 tokens against a sequence length of 32, so roughly half of
# every row is padding -- which the probe excludes, and which is the regime a corpus
# of short paragraphs actually sits in.
SMOKE_VALIDATION_DOCUMENTS = 320
SMOKE_HIDDEN_DIM = 32
SMOKE_INTERMEDIATE_DIM = 64
SMOKE_LAYERS = 4


def build_smoke_fixture(root: Path) -> tuple[str, str]:
    """A tiny model, tokenizer and two-split dataset, all local.

    Deterministic in its own right: the corpus is generated from a seeded RNG and
    the model from a seeded initialisation, so re-running the script reproduces the
    same fixture and therefore the same held-out fingerprint.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    generator = torch.Generator().manual_seed(0)

    def document(index: int) -> str:
        length = int(torch.randint(8, 24, (1,), generator=generator).item())
        picks = torch.randint(0, len(SMOKE_VOCAB_WORDS), (length,), generator=generator)
        return " ".join(SMOKE_VOCAB_WORDS[i] for i in picks.tolist()) + f" {index}"

    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    corpus: list[str] = []
    for split, count in (
        ("train", SMOKE_TRAIN_DOCUMENTS),
        ("validation", SMOKE_VALIDATION_DOCUMENTS),
    ):
        lines = [document(i) for i in range(count)]
        corpus.extend(lines)
        with (dataset_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps({"text": line}) + "\n")

    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        corpus, trainers.WordLevelTrainer(special_tokens=["<pad>", "<unk>", "<eos>"])
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    fast.save_pretrained(model_dir)

    torch.manual_seed(0)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=fast.vocab_size,
            hidden_size=SMOKE_HIDDEN_DIM,
            intermediate_size=SMOKE_INTERMEDIATE_DIM,
            num_hidden_layers=SMOKE_LAYERS,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    ).save_pretrained(model_dir)

    return str(model_dir), str(dataset_dir)
