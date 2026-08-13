"""Pydantic schemas for the loan resource.

Not yet used by any route — see `api/routes/customers.py`. Field names
here match the API's domain vocabulary (`loan_amount`, `loan_intent`), the
same naming SPECS.md §20 uses for `PredictionRequest` — translation to the
canonical DB column names (`amount`, `purpose`) is a repository-layer
concern. See `db/models/loan.py`.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

LoanIntent = Literal[
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
]
LoanGrade = Literal["A", "B", "C", "D", "E", "F", "G"]


class LoanCreate(BaseModel):
    """Fields required to register a new loan application."""

    customer_id: UUID
    loan_amount: float = Field(..., gt=0)
    interest_rate: float = Field(..., ge=0)
    term_months: int | None = Field(default=None, gt=0)
    loan_intent: LoanIntent
    loan_grade: LoanGrade | None = None


class LoanRead(LoanCreate):
    """A persisted loan, as returned by the API."""

    id: UUID
