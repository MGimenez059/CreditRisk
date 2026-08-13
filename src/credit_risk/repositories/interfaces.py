"""Repository interfaces, defined as `Protocol`s per CODESTYLE.md §6.

Services depend on these protocols, not on concrete SQLAlchemy classes, so
they can be exercised in unit tests with in-memory fakes instead of a real
database.
"""

import uuid
from typing import Protocol

from credit_risk.db.models.customer import Customer
from credit_risk.db.models.loan import Loan
from credit_risk.db.models.model import ModelMetadata
from credit_risk.db.models.prediction import Prediction


class PredictionRepositoryProtocol(Protocol):
    """Persists and retrieves model predictions."""

    def add(self, prediction: Prediction) -> Prediction:
        """Persist a new prediction and return it with server-generated fields set."""
        ...

    def get_by_id(self, prediction_id: uuid.UUID) -> Prediction | None:
        """Return the prediction with the given id, or None if it does not exist."""
        ...


class ModelRepositoryProtocol(Protocol):
    """Persists and retrieves trained-model metadata (the model registry)."""

    def get_active(self) -> ModelMetadata | None:
        """Return the currently active model's metadata, or None if unset."""
        ...

    def get_by_name_version(self, name: str, version: str) -> ModelMetadata | None:
        """Return the metadata row for a specific model name and version, if registered."""
        ...

    def add(self, model: ModelMetadata) -> ModelMetadata:
        """Persist metadata for a newly trained model artifact."""
        ...


class CustomerRepositoryProtocol(Protocol):
    """Persists and retrieves customers.

    Not yet used by any route — `api/routes/customers.py` is a roadmap
    Phase 6 placeholder. Defined now so the Phase 1 repository skeleton
    matches SPECS.md §4's file structure exactly.
    """

    def add(self, customer: Customer) -> Customer:
        """Persist a new customer and return it with server-generated fields set."""
        ...

    def get_by_id(self, customer_id: uuid.UUID) -> Customer | None:
        """Return the customer with the given id, or None if it does not exist."""
        ...


class LoanRepositoryProtocol(Protocol):
    """Persists and retrieves loans. See `CustomerRepositoryProtocol` note."""

    def add(self, loan: Loan) -> Loan:
        """Persist a new loan and return it with server-generated fields set."""
        ...

    def get_by_id(self, loan_id: uuid.UUID) -> Loan | None:
        """Return the loan with the given id, or None if it does not exist."""
        ...
