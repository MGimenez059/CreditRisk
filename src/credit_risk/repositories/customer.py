"""SQLAlchemy implementation of `CustomerRepositoryProtocol`.

Not yet used by any route — see `api/routes/customers.py`.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk.db.models.customer import Customer


class SQLAlchemyCustomerRepository:
    """Persists and retrieves `Customer` rows via a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a request-scoped session.

        Args:
            session: An active SQLAlchemy session, typically obtained from
                `credit_risk.db.session.get_db`.
        """
        self._session = session

    def add(self, customer: Customer) -> Customer:
        """Persist a new customer and return it with server-generated fields set."""
        self._session.add(customer)
        self._session.flush()
        return customer

    def get_by_id(self, customer_id: uuid.UUID) -> Customer | None:
        """Return the customer with the given id, or None if it does not exist."""
        statement = select(Customer).where(Customer.id == customer_id)
        return self._session.execute(statement).scalar_one_or_none()
