"""ORM model for the `credit_histories` table.

Field set follows SPECS.md §6 (Canonical Data Model) exactly.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from credit_risk.db.models.customer import Customer


class CreditHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Bureau-reported credit history attached to a customer.

    Note:
        `late_payments`, `credit_utilization`, and `active_credit_lines` are
        nullable: they are part of the public API contract (see
        `docs/data_dictionary.md`) but are not present in the Phase 0
        training dataset (Kaggle `laotse/credit-risk-dataset`) and must be
        backfilled from a richer data source, or engineered as proxies,
        before the model can consume them. See `docs/model_card.md` for the
        current feature set actually used at inference time.

        `previous_defaults` is SPECS.md §6's `INTEGER` count, not a boolean
        flag. The Phase 0 source dataset only provides a boolean
        (`cb_person_default_on_file`, Y/N), so ingestion maps it to `0` or
        `1` until a richer source provides an actual count — see
        `docs/data_dictionary.md`.
    """

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
    late_payments: Mapped[int | None] = mapped_column(Integer, nullable=True)
    credit_utilization: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    active_credit_lines: Mapped[int | None] = mapped_column(Integer, nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="credit_history")
