"""ORM model for the `predictions` table.

Column set follows SPECS.md §6 (Canonical Data Model): `customer_id`
(nullable, for ad-hoc predictions with no persisted customer) and `model_id`
(a normalized FK into the `models` registry table) replace the denormalized
`loan_id` / `model_name` / `model_version` columns from an earlier draft of
this schema.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Enum, Float, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from credit_risk.db.models.enums import RiskLevel

if TYPE_CHECKING:
    from credit_risk.db.models.customer import Customer
    from credit_risk.db.models.model import ModelMetadata


class Prediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Anonymous prediction linked to its model, with its complete stored explanation."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "default_probability >= 0 AND default_probability <= 1",
            name="ck_predictions_probability",
        ),
        CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="ck_predictions_score"),
        CheckConstraint("latency_ms >= 0", name="ck_predictions_latency"),
    )

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
    explanation: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)

    customer: Mapped["Customer | None"] = relationship()
    model: Mapped["ModelMetadata"] = relationship()
