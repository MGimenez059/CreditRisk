"""SQLAlchemy implementation of `ModelRepositoryProtocol`."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk.db.models.model import ModelMetadata


class SQLAlchemyModelRepository:
    """Persists and retrieves `ModelMetadata` rows via a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a request-scoped session.

        Args:
            session: An active SQLAlchemy session, typically obtained from
                `credit_risk.db.session.get_db`.
        """
        self._session = session

    def get_active(self) -> ModelMetadata | None:
        """Return the currently active model's metadata, or None if unset."""
        statement = select(ModelMetadata).where(ModelMetadata.is_active.is_(True))
        return self._session.execute(statement).scalar_one_or_none()

    def get_by_name_version(self, name: str, version: str) -> ModelMetadata | None:
        """Return the metadata row for a specific model name and version, if registered."""
        statement = select(ModelMetadata).where(
            ModelMetadata.name == name, ModelMetadata.version == version
        )
        return self._session.execute(statement).scalar_one_or_none()

    def add(self, model: ModelMetadata) -> ModelMetadata:
        """Persist metadata for a newly trained model artifact."""
        self._session.add(model)
        self._session.flush()
        return model
