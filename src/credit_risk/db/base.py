"""Declarative base and shared mixins for all SQLAlchemy ORM models."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Return the current UTC timestamp with timezone info attached."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model in the application."""


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key column, per CODESTYLE.md §13."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Adds `created_at` (all tables) and `updated_at` (mutable tables)."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )
