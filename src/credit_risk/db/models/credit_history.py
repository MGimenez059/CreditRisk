"""ORM model for the `credit_histories` table.

Field set follows SPECS.md §6 (Canonical Data Model) exactly.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from credit_risk.db.models.customer import Customer


class CreditHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Optional customer history scaffold; previous_defaults encodes the source 0/1 flag."""

    __tablename__ = "credit_histories"

    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    credit_history_years: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_defaults: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    customer: Mapped["Customer"] = relationship(back_populates="credit_history")
