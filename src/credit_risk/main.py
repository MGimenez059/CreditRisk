"""FastAPI application entrypoint.

Run locally with `uvicorn credit_risk.main:app --reload`, or via
`docker compose up --build` — see README.md.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from credit_risk.api.exception_handlers import register_exception_handlers
from credit_risk.api.routes import health, models, predictions
from credit_risk.config.settings import get_settings
from credit_risk.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log application startup and shutdown lifecycle events."""
    logger.info("application_startup", app_env=settings.app_env)
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="CreditRisk",
    description="Estimates the probability of loan default and explains why.",
    version=settings.model_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(models.router, prefix=settings.api_v1_prefix)
app.include_router(predictions.router, prefix=settings.api_v1_prefix)
