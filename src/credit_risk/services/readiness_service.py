"""Readiness checks for dependencies required to serve a prediction."""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from credit_risk.exceptions import CreditRiskError, ReadinessError
from credit_risk.repositories.readiness import check_serving_schema
from credit_risk.services.model_service import load_serving_artifact


@dataclass(frozen=True)
class ReadinessStatus:
    """Identity of the model verified by the readiness check."""

    model_name: str
    model_version: str


class ReadinessService:
    """Verify the database connection and configured serving artifact without writing."""

    def __init__(self, session: Session, model_path: Path) -> None:
        self._session = session
        self._model_path = model_path

    def check(self) -> ReadinessStatus:
        """Return the checked model identity when all serving dependencies are available."""
        try:
            check_serving_schema(self._session)
            loaded = load_serving_artifact(self._model_path)
        except (SQLAlchemyError, CreditRiskError) as err:
            raise ReadinessError("Service dependencies are unavailable.") from err
        return ReadinessStatus(
            model_name=loaded.artifact.metadata.name,
            model_version=loaded.artifact.metadata.version,
        )
