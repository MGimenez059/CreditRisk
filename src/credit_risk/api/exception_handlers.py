"""Translates domain exceptions into HTTP responses.

Registered once on the FastAPI app in `main.py`. Route handlers never catch
these exceptions themselves, per CODESTYLE.md §9.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from credit_risk.exceptions import (
    BatchSizeExceededError,
    CreditRiskError,
    EntityNotFoundError,
    ExplanationError,
    InvalidFeatureSchemaError,
    ModelLoadError,
    ModelNotFoundError,
    ReadinessError,
)

_STATUS_BY_EXCEPTION: dict[type[CreditRiskError], int] = {
    EntityNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelLoadError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    InvalidFeatureSchemaError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    BatchSizeExceededError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ExplanationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ReadinessError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the centralized `CreditRiskError` handler to the FastAPI app."""

    @app.exception_handler(CreditRiskError)
    async def handle_credit_risk_error(request: Request, exc: CreditRiskError) -> JSONResponse:
        status_code = _STATUS_BY_EXCEPTION.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
        request.state.failure_type = type(exc).__name__
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.exception_handler(SQLAlchemyError)
    async def handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        request.state.failure_type = type(exc).__name__
        return JSONResponse(
            status_code=503, content={"detail": "Prediction storage is unavailable."}
        )
