"""FastAPI dependency providers.

Route handlers depend only on the functions in this module, never on
`credit_risk.db.session` or a repository class directly — this is the single
place where the service graph is wired together.
"""

from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from credit_risk.config.settings import Settings, get_settings
from credit_risk.db.session import get_db
from credit_risk.repositories.model import SQLAlchemyModelRepository
from credit_risk.repositories.prediction import SQLAlchemyPredictionRepository
from credit_risk.services.model_service import ModelService
from credit_risk.services.prediction_service import PredictionService
from credit_risk.services.readiness_service import ReadinessService

DbSession = Annotated[Session, Depends(get_db)]


def get_prediction_repository(session: DbSession) -> SQLAlchemyPredictionRepository:
    """Provide a request-scoped prediction repository."""
    return SQLAlchemyPredictionRepository(session)


def get_model_repository(session: DbSession) -> SQLAlchemyModelRepository:
    """Provide a request-scoped model registry repository."""
    return SQLAlchemyModelRepository(session)


def get_prediction_service(
    session: DbSession,
    settings: Annotated[Settings, Depends(get_settings)],
    prediction_repository: Annotated[
        SQLAlchemyPredictionRepository, Depends(get_prediction_repository)
    ],
    model_repository: Annotated[SQLAlchemyModelRepository, Depends(get_model_repository)],
) -> PredictionService:
    """Provide a request-scoped `PredictionService`."""
    return PredictionService(
        session=session,
        settings=settings,
        prediction_repository=prediction_repository,
        model_repository=model_repository,
    )


def get_model_service(
    session: DbSession,
    repository: Annotated[SQLAlchemyModelRepository, Depends(get_model_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelService:
    """Provide the configured model service with a request-scoped transaction."""
    return ModelService(session, repository, Path(settings.model_path))


def get_readiness_service(
    session: DbSession,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadinessService:
    """Provide dependency checks using the request-scoped database session."""
    return ReadinessService(session, Path(settings.model_path))
