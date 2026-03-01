from mob import MoBConfig


def test_default_values_match_expected():
    cfg = MoBConfig()
    assert cfg.num_experts == 8
    assert cfg.top_k == 2
    assert cfg.hidden_dim == 4096
    assert cfg.intermediate_dim == 14336
    assert cfg.initial_wealth == 75.0
    assert cfg.wealth_decay == 0.997
    assert cfg.min_wealth == 15.0
    assert cfg.max_wealth == 750.0
    assert cfg.use_vcg_payments is True
    assert cfg.use_shared_base is True
    assert cfg.adapter_rank == 64
    assert cfg.adapter_alpha == 16.0
    assert cfg.use_differentiable_routing is True


def test_config_is_mutable():
    cfg = MoBConfig(num_experts=4, top_k=1, hidden_dim=512, intermediate_dim=1024)
    assert cfg.num_experts == 4
    assert cfg.top_k == 1
    assert cfg.hidden_dim == 512
    assert cfg.intermediate_dim == 1024
