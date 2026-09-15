"""Unit tests for readiness dependency failure mapping."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from credit_risk.exceptions import ReadinessError
from credit_risk.services.readiness_service import ReadinessService


def test_database_probe_failure_is_reported_as_unready() -> None:
    session = Mock(spec=Session)
    session.execute.side_effect = SQLAlchemyError("database detail")

    with pytest.raises(ReadinessError, match="Service dependencies are unavailable") as raised:
        ReadinessService(session, Path("unused.joblib")).check()

    assert isinstance(raised.value.__cause__, SQLAlchemyError)


def test_missing_artifact_is_reported_as_unready(tmp_path: Path) -> None:
    session = Mock(spec=Session)
    session.execute.return_value.scalar_one.return_value = 1

    with pytest.raises(ReadinessError, match="Service dependencies are unavailable"):
        ReadinessService(session, tmp_path / "missing.joblib").check()
