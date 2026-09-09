"""Shared pytest fixtures.

Per CODESTYLE.md §11: unit tests have no live database dependencies, so `client` is safe
to use against endpoints that do not touch the database or a model
artifact (currently `/health`). Endpoints that do belong under
`tests/integration/` against a dedicated test database.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from credit_risk.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A FastAPI TestClient bound to the application, without DB overrides."""
    with TestClient(app) as test_client:
        yield test_client
