"""Structured logging setup, per CODESTYLE.md §10.

Called once at process startup from `main.py`. Application code never calls
`print()`; it obtains a logger via `structlog.get_logger(__name__)`.
"""

import logging

import structlog


def configure_logging(log_level: str) -> None:
    """Configure `structlog` to emit JSON with standard-library log records.

    Args:
        log_level: Minimum level to emit (e.g. "INFO"), from `Settings.log_level`.
    """
    logging.basicConfig(format="%(message)s", level=log_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
