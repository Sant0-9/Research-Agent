"""Structured logging configuration using structlog.

Provides consistent, queryable logs across all services.
Supports both human-readable (development) and JSON (production) output.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    service_name: str = "research-agent",
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output JSON logs (for production). Otherwise, colored console.
        service_name: Name of the service for log context
    """
    # Set the log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Common processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # Production: JSON output for log aggregation
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored, human-readable output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Log initialization
    logger = get_logger()
    logger.info(
        "Logging initialized",
        service=service_name,
        level=level,
        json_logs=json_logs,
    )


def get_logger(name: str | None = None, **initial_context: Any) -> Any:
    """Get a structured logger instance.

    Args:
        name: Optional logger name for context
        **initial_context: Additional context to bind to all log messages

    Returns:
        A bound structlog logger instance
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**context: Any) -> None:
    """Bind context variables that will be included in all subsequent log messages.

    Useful for adding request-specific context like request_id, user_id, etc.

    Args:
        **context: Key-value pairs to add to log context
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """Clear all context variables.

    Should be called at the end of request processing.
    """
    structlog.contextvars.clear_contextvars()
