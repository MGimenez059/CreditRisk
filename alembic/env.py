"""Alembic migration environment.

The connection string is read from `credit_risk.config.settings.Settings`
rather than `alembic.ini`, so `.env` remains the single source of truth for
the database URL, per CODESTYLE.md §16.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from credit_risk.config.settings import get_settings
from credit_risk.db.base import Base
from credit_risk.db.models import *  # noqa: F403 -- registers every model on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

config.set_main_option("sqlalchemy.url", get_settings().database_url)


def run_migrations_offline() -> None:
    """Run migrations without a live database connection, emitting raw SQL."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
