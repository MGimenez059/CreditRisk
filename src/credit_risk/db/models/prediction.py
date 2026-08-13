"""ORM model for the `predictions` table.

Column set follows SPECS.md §6 (Canonical Data Model): `customer_id`
(nullable, for ad-hoc predictions with no persisted customer) and `model_id`
(a normalized FK into the `models` registry table) replace the denormalized
`loan_id` / `model_name` / `model_version` columns from an earlier draft of
this schema.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum, Float, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from credit_risk.db.models.enums import RiskLevel

if TYPE_CHECKING:
    from credit_risk.db.models.customer import Customer
    from credit_risk.db.models.model import ModelMetadata


class Prediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A persisted model prediction, traceable to the exact model used.

    Note:
        `request_id`, `latency_ms`, and `explanation` are not in SPECS.md
        §6's canonical Prediction table, but are required to log what
        SPECS.md §33 (Logging) mandates — request ID, latency, and (per
        §19/§25) reproducible historical explanations — so they are kept
        here as legitimate additions rather than removed for a literal
        column-for-column match.
    """

    __tablename__ = "predictions"

    customer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    default_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[RiskLevel] = mapped_column(
        Enum(RiskLevel, name="risk_level_enum"),
        nullable=False,
    )
    prediction_version: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(10, 3), nullable=False)
    explanation: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)

    customer: Mapped["Customer | None"] = relationship()
    model: Mapped["ModelMetadata"] = relationship()
