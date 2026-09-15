"""Unit tests for the `/health` endpoint."""

import json
import uuid

from fastapi.testclient import TestClient

from credit_risk.api.dependencies import get_readiness_service
from credit_risk.exceptions import ReadinessError
from credit_risk.main import app
from credit_risk.services.readiness_service import ReadinessStatus


def test_health_endpoint_returns_ok_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "upstream-request-42"})

    assert response.headers["X-Request-ID"] == "upstream-request-42"


def test_invalid_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "unsafe request id"})

    assert uuid.UUID(response.headers["X-Request-ID"])


def test_readiness_reports_checked_dependencies(client: TestClient) -> None:
    class ReadyService:
        def check(self) -> ReadinessStatus:
            return ReadinessStatus("selected-model", "selected-v1")

    app.dependency_overrides[get_readiness_service] = ReadyService
    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"database": "ready", "model": "ready"},
        "model": {"name": "selected-model", "version": "selected-v1"},
    }


def test_readiness_failure_is_generic_and_correlated(client: TestClient) -> None:
    class UnreadyService:
        def check(self) -> ReadinessStatus:
            raise ReadinessError("Service dependencies are unavailable.")

    app.dependency_overrides[get_readiness_service] = UnreadyService
    try:
        response = client.get("/ready", headers={"X-Request-ID": "readiness-check-1"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Service dependencies are unavailable."}
    assert response.headers["X-Request-ID"] == "readiness-check-1"


def test_unexpected_failure_is_generic_and_correlated(capsys) -> None:
    class BrokenService:
        def check(self) -> ReadinessStatus:
            raise RuntimeError("sensitive internal detail")

    app.dependency_overrides[get_readiness_service] = BrokenService
    try:
        with TestClient(app) as client:
            response = client.get("/ready", headers={"X-Request-ID": "unexpected-failure-1"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert response.headers["X-Request-ID"] == "unexpected-failure-1"
    assert "sensitive internal detail" not in response.text
    output = capsys.readouterr()
    assert "sensitive internal detail" not in output.out + output.err
    events = [json.loads(line) for line in output.out.splitlines() if line.startswith("{")]
    failures = [event for event in events if event.get("event") == "request_failed"]
    assert len(failures) == 1
    assert failures[0]["request_id"] == "unexpected-failure-1"
    assert failures[0]["exception_type"] == "RuntimeError"
