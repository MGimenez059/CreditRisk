"""Registered artifact identity, immutable sidecar metadata and activation state."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ModelMetadata(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Immutable artifact identity and metadata, plus transactional activation state."""

    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_models_name_version"),
        Index("uq_models_active", "is_active", unique=True, postgresql_where=text("is_active")),
    )

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    training_dataset: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    pr_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
