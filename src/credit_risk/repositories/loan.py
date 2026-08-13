"""SQLAlchemy implementation of `LoanRepositoryProtocol`.

Not yet used by any route — see `api/routes/customers.py`.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk.db.models.loan import Loan


class SQLAlchemyLoanRepository:
    """Persists and retrieves `Loan` rows via a SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a request-scoped session.

        Args:
            session: An active SQLAlchemy session, typically obtained from
                `credit_risk.db.session.get_db`.
        """
        self._session = session

    def add(self, loan: Loan) -> Loan:
        """Persist a new loan and return it with server-generated fields set."""
        self._session.add(loan)
        self._session.flush()
        return loan

    def get_by_id(self, loan_id: uuid.UUID) -> Loan | None:
        """Return the loan with the given id, or None if it does not exist."""
        statement = select(Loan).where(Loan.id == loan_id)
        return self._session.execute(statement).scalar_one_or_none()
