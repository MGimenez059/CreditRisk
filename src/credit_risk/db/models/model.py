"""ORM model for the `models` table: the trained-model registry.

Column set follows SPECS.md §6 (Canonical Data Model): `training_dataset`,
`roc_auc`, `pr_auc`, `f1`, `brier_score`, and `artifact_path` are named and
typed exactly as specified there. `feature_version` is not in §6's
simplified diagram but is kept because §25 (Model Versioning) explicitly
requires "feature version" as part of every deployed model's metadata —
the prose requirement takes precedence over the abbreviated diagram.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from credit_risk.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ModelMetadata(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Metadata for a single trained model artifact (the `models` table).

    Persisted alongside every artifact written by `scripts/train_model.py`.
    Exactly one row should have `is_active = True` at a time; enforcing
    that invariant is a service-layer concern (`services/`), not a database
    constraint, since promoting a new active model requires deactivating
    the previous one atomically.
    """

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_models_name_version"),)

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    training_dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    pr_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
