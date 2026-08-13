"""Unit tests for the `/health` endpoint."""

from fastapi.testclient import TestClient


def test_health_endpoint_returns_ok_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
