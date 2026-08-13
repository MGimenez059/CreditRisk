"""ORM model for the `loans` table.

Column names follow SPECS.md §6 (Canonical Data Model) exactly: `amount`,
`purpose`, and `grade` are the canonical DB column names, distinct from the
API's `loan_amount` / `loan_intent` field names in
`schemas/prediction.py` (SPECS.md §20). Translating between the two is a
repository/service-layer concern, not a naming accident.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Enum, ForeignKey, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from credit_risk.db.models.enums import LoanGrade, LoanIntent

if TYPE_CHECKING:
    from credit_risk.db.models.customer import Customer


class Loan(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single loan application submitted by a customer.

    Note:
        `loan_status` is the historical outcome (0 = no default, 1 =
        default) for loans ingested from the training dataset. It is the
        model's training target — per SPECS.md §7, it must never be read
        back as a model input feature. It is nullable because a live loan
        application (as opposed to a historical training record) has no
        known outcome yet.
    """

    __tablename__ = "loans"
    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_loans_amount_positive"),
        CheckConstraint("interest_rate >= 0", name="ck_loans_interest_rate_non_negative"),
        CheckConstraint("loan_status IN (0, 1)", name="ck_loans_status_binary"),
    )

    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    interest_rate: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    term_months: Mapped[int | None] = mapped_column(Integer, nullable=True)
    purpose: Mapped[LoanIntent] = mapped_column(
        Enum(LoanIntent, name="loan_intent_enum"),
        nullable=False,
    )
    grade: Mapped[LoanGrade | None] = mapped_column(
        Enum(LoanGrade, name="loan_grade_enum"),
        nullable=True,
    )
    loan_status: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Not in SPECS.md §6's canonical Loan table; kept for parity with the
    # Phase 0 source dataset's `loan_percent_income` column. May be dropped
    # in favor of a Phase 3 derived feature computed on the fly instead of
    # stored.
    loan_percent_income: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="loans")
