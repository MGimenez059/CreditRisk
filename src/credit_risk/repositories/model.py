"""SQLAlchemy implementation of `ModelRepositoryProtocol`."""

from sqlalchemy import select, text, update
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

    def lock_registry(self) -> None:
        """Serialize registration and activation until the enclosing transaction ends."""
        self._session.execute(text("SELECT pg_advisory_xact_lock(734621)"))

    def activate(self, model: ModelMetadata) -> None:
        """Promote the configured artifact atomically while holding the registry lock."""
        self._session.execute(
            update(ModelMetadata).where(ModelMetadata.id != model.id).values(is_active=False)
        )
        model.is_active = True
        self._session.flush()
