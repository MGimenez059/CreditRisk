"""Liveness endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Trivial liveness payload."""

    status: str = "ok"


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Report whether the service process is up and accepting requests."""
    return HealthResponse()
