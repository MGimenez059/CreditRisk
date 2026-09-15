"""Request correlation and request-wide structured logging."""

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _request_id(request: Request) -> str:
    """Reuse a bounded safe caller ID, or generate a new UUID."""
    candidate = request.headers.get(REQUEST_ID_HEADER)
    if candidate is not None and _SAFE_REQUEST_ID.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


def register_request_context_middleware(app: FastAPI) -> None:
    """Add one correlation ID and one completion/failure event to every request."""

    @app.middleware("http")
    async def correlate_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        structlog.contextvars.clear_contextvars()
        request_id = _request_id(request)
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        logger.info("request_started", method=request.method, path=request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:
            # Exception messages and tracebacks can contain input data or credentials.
            # Handle here so the server does not log the same exception unsanitized.
            request.state.failure_type = type(exc).__name__
            response = JSONResponse(status_code=500, content={"detail": "Internal server error."})
        response.headers[REQUEST_ID_HEADER] = request_id
        fields = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }
        failure_type = getattr(request.state, "failure_type", None)
        if failure_type is not None:
            fields["exception_type"] = failure_type
        if response.status_code >= 500:
            logger.error("request_failed", **fields)
        elif response.status_code >= 400:
            logger.warning("request_failed", **fields)
        else:
            logger.info("request_completed", **fields)
        structlog.contextvars.clear_contextvars()
        return response
