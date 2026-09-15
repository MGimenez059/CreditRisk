"""Read-only database probes for the serving schema."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk.db.models.model import ModelMetadata
from credit_risk.db.models.prediction import Prediction


def check_serving_schema(session: Session) -> None:
    """Verify required tables and mapped columns without fetching applicant records."""
    for entity in (ModelMetadata, Prediction):
        session.execute(select(entity).limit(0)).close()
