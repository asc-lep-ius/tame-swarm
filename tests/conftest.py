import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tame"))

from mob import MoBConfig, MixtureOfBidders


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
