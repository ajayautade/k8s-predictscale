# ============================================
# K8s PredictScale - API Unit Tests
# ============================================

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from src.api.main import app
    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "K8s PredictScale"
        assert "version" in data

class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "prometheus" in data
        assert "version" in data


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_prometheus_format(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        # Prometheus text format
        assert "predictscale" in response.text or "python_info" in response.text


class TestPredictionEndpoints:
    def test_predictions_returns_404_when_none(self, client):
        response = client.get("/api/v1/predictions")
        # No predictions yet, should return 404
        assert response.status_code in (404, 503)

    def test_prediction_history(self, client):
        response = client.get("/api/v1/predictions/history")
        assert response.status_code in (200, 503)


class TestScalingEndpoints:
    def test_scaling_events_empty(self, client):
        response = client.get("/api/v1/scaling/events")
        assert response.status_code in (200, 503)

    def test_dry_run_simulation(self, client):
        response = client.post(
            "/api/v1/scaling/dry-run",
            json={
                "predicted_peak": 3.5,
                "confidence": 0.9,
                "current_replicas": 2,
            },
        )
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "decision" in data

    def test_scaling_config_get(self, client):
        response = client.get("/api/v1/scaling/config")
        assert response.status_code in (200, 503)


class TestModelEndpoints:
    def test_model_status(self, client):
        response = client.get("/api/v1/model/status")
        assert response.status_code in (200, 503)
