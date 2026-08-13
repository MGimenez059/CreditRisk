"""SQLAlchemy ORM models. Never imported outside the db/repositories layers.

Every model is imported here so that `Base.metadata` is fully populated
before Alembic autogenerate or `Base.metadata.create_all` runs, and so that
string-based `relationship()` forward references resolve correctly.
"""

from credit_risk.db.models.credit_history import CreditHistory
from credit_risk.db.models.customer import Customer
from credit_risk.db.models.loan import Loan
from credit_risk.db.models.model import ModelMetadata
from credit_risk.db.models.prediction import Prediction

__all__ = [
    "CreditHistory",
    "Customer",
    "Loan",
    "ModelMetadata",
    "Prediction",
]
