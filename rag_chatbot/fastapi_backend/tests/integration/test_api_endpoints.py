import pytest
from fastapi.testclient import TestClient
from ...main import app


def test_root_endpoint():
    """Test the root endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """Test the health endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_config_endpoint():
    """Test the config endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "streaming_enabled" in data


def test_health_check_endpoint():
    """Test the health check endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data


def test_status_endpoint():
    """Test the status endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data