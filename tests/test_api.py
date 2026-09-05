from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import TAMEApplication
from mob import MoBConfig
from steering import SteeringConfig


@pytest.fixture
def mock_tame_app():
    tame = MagicMock(spec=TAMEApplication)
    tame.model_id = "test-model"
    tame.homeostat = None
    tame.mob_config = MoBConfig(num_experts=2, top_k=1, hidden_dim=32, intermediate_dim=64)
    tame.steering_config = SteeringConfig()
    return tame


@pytest.fixture
def client(mock_tame_app):
    from routes import router

    app = FastAPI()
    app.include_router(router)
    app.state.tame = mock_tame_app
    return TestClient(app)


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "alive"
    assert body["model_id"] == "test-model"
    assert "mob_active" in body
    assert "steering_active" in body


def test_swarm_status_returns_200(client, mock_tame_app):
    mock_model = MagicMock()
    mock_model.model.layers = []
    mock_tame_app.model = mock_model

    resp = client.get("/swarm/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "num_experts" in body
    assert "expert_wealth" in body


def test_homeostasis_status_disabled(client, mock_tame_app):
    mock_tame_app.homeostat = None

    resp = client.get("/homeostasis/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "disabled"


def test_generate_request_validation(client):
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422


@pytest.fixture
def loaded_homeostat():
    import torch

    from homeostat import CognitiveHomeostat
    from steering import SteeringVector

    config = SteeringConfig(steering_layers=[1, 2], base_strength=2.0, max_strength=4.0)
    homeostat = CognitiveHomeostat(config)
    for layer in (1, 2):
        homeostat.add_steering_vector(layer, SteeringVector("truthful", torch.randn(8), layer))
    return homeostat, config


def test_homeostasis_status_exposes_the_loop_terms(client, mock_tame_app, loaded_homeostat):
    homeostat, config = loaded_homeostat
    mock_tame_app.homeostat = homeostat
    mock_tame_app.steering_config = config

    body = client.get("/homeostasis/status").json()

    assert body["status"] == "active"
    assert body["config"]["goal"] == "truthful"
    assert body["config"]["steering_layers"] == [1, 2]
    pid = body["pid"]
    assert set(pid) >= {"p_term", "i_term", "d_term", "output", "integral_saturated", "kp", "ki"}
    assert pid["goal"] == "truthful"
    assert pid["calibrated"] is False


def test_gains_endpoint_applies_and_validates(client, mock_tame_app, loaded_homeostat):
    homeostat, config = loaded_homeostat
    mock_tame_app.homeostat = homeostat

    resp = client.put("/steering/gains", json={"kp": 0.25, "ki": 0.05})
    assert resp.status_code == 200
    assert resp.json()["kp"] == 0.25
    assert resp.json()["ki"] == 0.05
    assert homeostat.homeostat.gains() == (0.25, 0.05)

    assert client.put("/steering/gains", json={"adaptive": True}).status_code == 200
    assert config.adaptive is True
    assert client.put("/steering/gains", json={"adaptive": False}).status_code == 200
    assert config.adaptive is False

    assert client.put("/steering/gains", json={"kp": -1.0}).status_code == 422
    # kd above the noise bound (the strength band) is refused by the loop itself.
    assert client.put("/steering/gains", json={"kd": 100.0}).status_code == 422
    assert client.put("/steering/gains", json={"goal": "safe", "kp": 0.1}).status_code == 404


def test_gains_endpoint_without_steering(client, mock_tame_app):
    mock_tame_app.homeostat = None
    assert client.put("/steering/gains", json={"kp": 0.1}).status_code == 400


def test_steering_update_installs_the_goal_and_reports_the_band(
    client, mock_tame_app, loaded_homeostat
):
    from unittest.mock import MagicMock

    homeostat, config = loaded_homeostat
    mock_tame_app.homeostat = homeostat
    mock_tame_app.steering_config = config
    extraction = MagicMock(source="builtin", pair_format="completion", certified=True)
    mock_tame_app.install_goal = MagicMock(return_value=extraction)

    resp = client.post("/steering/update", params={"goal": "safe", "strength": 3.0, "kp": 0.2})

    assert resp.status_code == 200
    mock_tame_app.install_goal.assert_called_once_with("safe", strength=3.0)
    body = resp.json()
    assert body["layers"] == [1, 2]
    assert body["strength_band"] == [0.0, 4.0]
    assert body["pid"]["kp"] == 0.2


def test_generate_response_carries_the_loop_history_and_pid_status(loaded_homeostat):
    """The stats the homeostat returns must fit the response model, or /generate 500s."""
    from models import GenerateResponse

    homeostat, _ = loaded_homeostat
    direction = homeostat.steering_vectors[1].vector
    for _ in range(3):
        homeostat.homeostat.compute_strength(direction.view(1, 1, -1), direction)
    stats = homeostat.get_alignment_stats()
    assert stats["alignment_history"] and isinstance(stats["pid"], dict)

    response = GenerateResponse(response="x", usage={"input_tokens": 1}, homeostasis=stats)
    assert response.homeostasis is not None
    assert len(response.homeostasis["strength_history"]) == 3


def test_steering_update_rejects_bad_strengths(client, mock_tame_app, loaded_homeostat):
    from unittest.mock import MagicMock

    homeostat, _ = loaded_homeostat
    mock_tame_app.homeostat = homeostat
    assert client.post("/steering/update", params={"strength": -5}).status_code == 422
    assert client.post("/steering/update", params={"strength": 0}).status_code == 422

    mock_tame_app.install_goal = MagicMock(side_effect=ValueError("outside the certified band"))
    resp = client.post("/steering/update", params={"goal": "truthful", "strength": 40})
    assert resp.status_code == 422
    assert "band" in resp.json()["detail"]


def test_install_goal_reattaches_the_old_loop_when_the_new_one_fails(monkeypatch):
    import torch

    import app as app_module
    from homeostat import CognitiveHomeostat
    from steering import SteeringVector

    from .steering_fakes import MonotonicModel, SimpleCharTokenizer

    model = MonotonicModel(vocab_size=32, hidden_dim=8, num_layers=4)
    config = SteeringConfig(steering_layers=[1, 2], orthogonal_projection=False)
    homeostat = CognitiveHomeostat(config)
    for layer in (1, 2):
        homeostat.add_steering_vector(layer, SteeringVector("truthful", torch.randn(8), layer))
    homeostat.attach_to_model(model)
    tame = app_module.TAMEApplication(
        model=model,  # pyright: ignore[reportArgumentType]
        tokenizer=SimpleCharTokenizer(),  # pyright: ignore[reportArgumentType]
        homeostat=homeostat,
        mob_config=MoBConfig(num_experts=2, top_k=1, hidden_dim=8, intermediate_dim=16),
        steering_config=config,
        model_id="fake",
        steering_template=config,
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("no such goal")

    monkeypatch.setattr(app_module, "build_homeostat", explode)
    with pytest.raises(RuntimeError):
        tame.install_goal("safe")

    assert tame.homeostat is homeostat
    assert len(homeostat._registered_hooks) == 2
    homeostat.detach_from_model()


def test_build_homeostat_rejects_a_strength_outside_the_certified_band():
    """The band check runs before the model is touched, so no model is needed to test it."""
    from app import build_homeostat

    template = SteeringConfig()
    with pytest.raises(ValueError, match="certified band"):
        build_homeostat(None, None, template, "truthful", strength=40)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="positive"):
        build_homeostat(None, None, template, "truthful", strength=-1)  # pyright: ignore[reportArgumentType]


def test_install_goal_starts_every_install_from_the_pristine_template(monkeypatch):
    """Gains pinned on the served config must not leak into the next goal's derivation."""
    from unittest.mock import MagicMock

    import app as app_module
    from homeostat import CognitiveHomeostat

    seen = {}

    def spy(model, tokenizer, template, goal, model_id=None, strength=None):
        seen["template"] = template
        config = SteeringConfig()
        return CognitiveHomeostat(config), MagicMock(), config

    monkeypatch.setattr(app_module, "build_homeostat", spy)
    template = SteeringConfig()
    served = SteeringConfig(kp=9.0, ki=0.5, steering_layers=[13, 16])
    tame = app_module.TAMEApplication(
        model=MagicMock(),
        tokenizer=MagicMock(),
        homeostat=None,
        mob_config=MoBConfig(num_experts=2, top_k=1, hidden_dim=8, intermediate_dim=16),
        steering_config=served,
        model_id="fake",
        steering_template=template,
    )
    tame.install_goal("safe")

    assert seen["template"] is template
    assert seen["template"].kp is None
