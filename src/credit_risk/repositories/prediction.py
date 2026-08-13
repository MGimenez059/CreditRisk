"""SQLAlchemy implementation of `PredictionRepositoryProtocol`.

Raw SQL strings are forbidden outside this module, per CODESTYLE.md §13; all
queries here use SQLAlchemy Core/ORM constructs.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk.db.models.prediction import Prediction


class SQLAlchemyPredictionRepository:
    """Persists and retrieves `Prediction` rows via a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a request-scoped session.

        Args:
            session: An active SQLAlchemy session, typically obtained from
                `credit_risk.db.session.get_db`.
        """
        self._session = session

    def add(self, prediction: Prediction) -> Prediction:
        """Persist a new prediction and return it with server-generated fields set.

        Note:
            This method stages the row and flushes it; committing the
            transaction remains the caller's (service layer's)
            responsibility, per CODESTYLE.md §13.
        """
        self._session.add(prediction)
        self._session.flush()
        return prediction

    def get_by_id(self, prediction_id: uuid.UUID) -> Prediction | None:
        """Return the prediction with the given id, or None if it does not exist."""
        statement = select(Prediction).where(Prediction.id == prediction_id)
        return self._session.execute(statement).scalar_one_or_none()
