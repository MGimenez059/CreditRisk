"""Liveness and serving-readiness endpoints."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from credit_risk.api.dependencies import get_readiness_service
from credit_risk.services.readiness_service import ReadinessService

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Trivial liveness payload."""

    status: str = "ok"


class ReadinessChecks(BaseModel):
    """Serving dependencies checked by the readiness endpoint."""

    database: Literal["ready"] = "ready"
    model: Literal["ready"] = "ready"


class ReadinessModel(BaseModel):
    """Identity of the verified serving model."""

    name: str
    version: str


class ReadinessResponse(BaseModel):
    """Successful readiness result."""

    status: Literal["ready"] = "ready"
    checks: ReadinessChecks = ReadinessChecks()
    model: ReadinessModel


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Report whether the service process is up and accepting requests."""
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse)
def get_readiness(
    service: Annotated[ReadinessService, Depends(get_readiness_service)],
) -> ReadinessResponse:
    """Verify the database and configured model needed to serve predictions."""
    checked = service.check()
    return ReadinessResponse(
        model=ReadinessModel(name=checked.model_name, version=checked.model_version)
    )
