"""FastAPI dependency providers.

Route handlers depend only on the functions in this module, never on
`credit_risk.db.session` or a repository class directly — this is the single
place where the service graph is wired together.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from credit_risk.db.session import get_db
from credit_risk.repositories.model import SQLAlchemyModelRepository
from credit_risk.repositories.prediction import SQLAlchemyPredictionRepository
from credit_risk.services.prediction_service import PredictionService

DbSession = Annotated[Session, Depends(get_db)]


def get_prediction_repository(session: DbSession) -> SQLAlchemyPredictionRepository:
    """Provide a request-scoped prediction repository."""
    return SQLAlchemyPredictionRepository(session)


def get_model_repository(session: DbSession) -> SQLAlchemyModelRepository:
    """Provide a request-scoped model registry repository."""
    return SQLAlchemyModelRepository(session)


def get_prediction_service(
    prediction_repository: Annotated[
        SQLAlchemyPredictionRepository, Depends(get_prediction_repository)
    ],
    model_repository: Annotated[SQLAlchemyModelRepository, Depends(get_model_repository)],
) -> PredictionService:
    """Provide a request-scoped `PredictionService`."""
    return PredictionService(
        prediction_repository=prediction_repository,
        model_repository=model_repository,
    )
