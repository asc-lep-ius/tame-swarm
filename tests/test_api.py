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
