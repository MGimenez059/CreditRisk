"""Shared pytest fixtures.

Per CODESTYLE.md §11: unit tests have no I/O, so `client` here is only safe
to use against endpoints that do not touch the database or a model
artifact (currently `/health`). Endpoints that do belong under
`tests/integration/` against a dedicated test database.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# NOTE: as of starlette 1.6, TestClient emits a deprecation warning pointing
# at an `httpx2` replacement package. Tracked for a future dependency bump;
# not addressed here to avoid pulling in an unfamiliar transport dependency
# during the Phase 1 skeleton.
from credit_risk.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A FastAPI TestClient bound to the application, without DB overrides."""
    with TestClient(app) as test_client:
        yield test_client
