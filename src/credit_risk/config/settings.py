"""Centralized application configuration.

All configuration is read through the single :class:`Settings` object below.
No other module should call ``os.environ.get`` directly — see CODESTYLE.md
§16. Defaults are safe for local development only; production values must
always be supplied via the environment.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables and `.env`.

    Attributes:
        app_env: Deployment environment. Affects logging verbosity and
            debug-only behavior; must never gate security controls.
        log_level: Minimum level emitted by the structured logger.
        api_v1_prefix: URL prefix mounted for all versioned API routes.
        cors_allowed_origins: Origins allowed to call the API from a browser.
        database_url: SQLAlchemy connection string for PostgreSQL.
        database_echo: Whether SQLAlchemy logs every emitted SQL statement.
        model_path: Filesystem path to the serialized inference pipeline.
        model_name: Registered name of the active model, echoed in responses.
        model_version: Semantic version of the active model artifact.
        max_batch_size: Upper bound on records accepted per batch request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    api_v1_prefix: str = "/api/v1"
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: str = "postgresql+psycopg://creditrisk:creditrisk@localhost:5432/creditrisk"
    database_echo: bool = False

    model_path: str = "models/credit_risk_xgboost_v1.joblib"
    model_name: str = "credit-risk-xgboost"
    model_version: str = "1.0.0"

    max_batch_size: int = 500


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance.

    Cached with ``lru_cache`` so environment variables are parsed once per
    process. Tests that need different configuration should override this
    dependency rather than mutate the cached instance.
    """
    return Settings()
