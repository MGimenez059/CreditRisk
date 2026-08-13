"""ORM model for the `customers` table."""

from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Enum, Integer, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from credit_risk.db.models.enums import HomeOwnership

if TYPE_CHECKING:
    from credit_risk.db.models.credit_history import CreditHistory
    from credit_risk.db.models.loan import Loan


class Customer(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A loan applicant.

    Represents the person-level attributes of a credit risk assessment,
    independent of any specific loan application. One customer may have
    multiple loans over time.
    """

    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint("age > 0 AND age < 120", name="ck_customers_age_range"),
        CheckConstraint("income >= 0", name="ck_customers_income_non_negative"),
    )

    age: Mapped[int] = mapped_column(Integer, nullable=False)
    income: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    home_ownership: Mapped[HomeOwnership] = mapped_column(
        Enum(HomeOwnership, name="home_ownership_enum"),
        nullable=False,
    )
    employment_years: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)

    loans: Mapped[list["Loan"]] = relationship(back_populates="customer")
    credit_history: Mapped["CreditHistory | None"] = relationship(
        back_populates="customer",
        uselist=False,
    )
