from config import MODEL_PROFILES, get_active_profile, ModelProfile


def test_active_profile_exists():
    profile = get_active_profile()
    assert profile["model_id"]
    assert profile["hidden_dim"] > 0
    assert profile["intermediate_dim"] > 0


def test_all_profiles_have_required_keys():
    required_keys = {"model_id", "hidden_dim", "intermediate_dim", "num_layers", "mob_layers_start", "mob_layers_end"}
    for name, profile in MODEL_PROFILES.items():
        missing = required_keys - set(profile.keys())
        assert not missing, f"Profile '{name}' missing keys: {missing}"


def test_mob_layer_range_valid():
    for name, profile in MODEL_PROFILES.items():
        assert profile["mob_layers_start"] < profile["mob_layers_end"], (
            f"Profile '{name}': mob_layers_start must be < mob_layers_end"
        )
        assert profile["mob_layers_end"] < profile["num_layers"], (
            f"Profile '{name}': mob_layers_end must be < num_layers"
        )
