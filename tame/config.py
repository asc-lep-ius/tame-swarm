from typing import TypedDict


class ModelProfile(TypedDict):
    model_id: str
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    mob_layers_start: int
    mob_layers_end: int


MODEL_PROFILES: dict[str, ModelProfile] = {
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b-it",
        "hidden_dim": 2304,
        "intermediate_dim": 9216,
        "num_layers": 26,
        "mob_layers_start": 5,
        "mob_layers_end": 18,
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_dim": 3072,
        "intermediate_dim": 8192,
        "num_layers": 28,
        "mob_layers_start": 6,
        "mob_layers_end": 20,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_layers": 32,
        "mob_layers_start": 8,
        "mob_layers_end": 24,
    },
    # Apache-2.0, ungated on HuggingFace -- unlike the three profiles above,
    # which all require accepting a license before the weights download. That
    # makes it the only profile a CI runner or a fresh machine without an HF
    # token can actually pull, and it is the model every measurement in the
    # README was taken on (#13 noise floor, #3/#17 steering gates, #4 plant).
    "qwen3-1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "hidden_dim": 2048,
        "intermediate_dim": 6144,
        "num_layers": 28,
        "mob_layers_start": 6,
        "mob_layers_end": 22,
    },
}

# The profile the server and trainer default to. It is the model the steering
# certifications (contrastive_data.CERTIFIED) and the homeostat's plant were
# measured on; serving another profile gets those numbers with a warning.
ACTIVE_MODEL = "qwen3-1.7b"


def get_active_profile() -> ModelProfile:
    return MODEL_PROFILES[ACTIVE_MODEL]
