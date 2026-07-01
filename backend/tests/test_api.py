"""
Integration tests for Aegis-MM FastAPI Backend API.
Verifies REST endpoints and WebSocket real-time streaming telemetry.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Aegis-MM" in data["project"]


def test_analyze_frame_endpoint(client):
    payload = {"session_id": "test_session_101", "inject_anomaly": "gaze_drift"}
    response = client.post("/analyze/frame", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_101"
    assert "overall_risk_score" in data
    assert "ai_fluency_score" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "stages" in data


def test_websocket_streaming_endpoint(client):
    with client.websocket_connect("/ws/stream/ws_test_session") as websocket:
        websocket.send_text('{"action": "stream", "anomaly": "tts_audio"}')
        data = websocket.receive_json()
        assert data["session_id"] == "ws_test_session"
        assert "overall_risk_score" in data
