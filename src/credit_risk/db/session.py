"""SQLAlchemy engine and session factory.

A single module-level engine is created from :func:`get_settings`. FastAPI
routes obtain a session exclusively through :func:`get_db`, never by
importing ``SessionLocal`` directly, so the dependency can be overridden in
tests.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from credit_risk.config.settings import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, echo=settings.database_echo, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session scoped to a single request.

    The session is always closed on exit, even if the request raised.
    Transaction commit/rollback is the caller's (service layer's)
    responsibility, per CODESTYLE.md §13.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
