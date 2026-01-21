"""Research Agent API - Main entry point.

FastAPI application for the deep research AI system.
Handles research queries and generates publication-ready LaTeX papers.
"""

import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import get_settings
from src.utils.errors import RateLimitError, ResearchAgentError
from src.utils.logging import bind_context, clear_context, get_logger, setup_logging

# Version info
__version__ = "0.1.0"


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        json_logs=settings.log_json,
        service_name="research-agent",
    )

    logger = get_logger()
    logger.info(
        "Starting Research Agent API",
        version=__version__,
        environment=settings.environment,
    )

    # Validate required services are configured
    logger.info("Configuration validated", brain_url=settings.brain_service_url)

    yield

    # Shutdown
    logger.info("Shutting down Research Agent API")


# =============================================================================
# Application Factory
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Research Agent API",
        description="Deep research AI system for generating publication-ready papers",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Add exception handlers
    app.add_exception_handler(ResearchAgentError, research_agent_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    # Add middleware
    app.middleware("http")(request_middleware)

    return app


# =============================================================================
# Middleware
# =============================================================================

# Simple in-memory rate limiter (replace with Redis in production)
_rate_limit_store: dict[str, list[float]] = {}


async def request_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Request middleware for logging, rate limiting, and context."""
    settings = get_settings()
    logger = get_logger()

    # Generate request ID
    request_id = str(uuid4())[:8]
    bind_context(request_id=request_id)

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting (skip for health checks)
    if request.url.path not in ("/health", "/ready"):
        current_time = time.time()
        window_start = current_time - settings.rate_limit_window_seconds

        # Clean old entries and check rate limit
        if client_ip not in _rate_limit_store:
            _rate_limit_store[client_ip] = []

        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip] if t > window_start
        ]

        if len(_rate_limit_store[client_ip]) >= settings.rate_limit_requests:
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            clear_context()
            raise RateLimitError(
                "Rate limit exceeded. Please try again later.",
                details={"retry_after_seconds": settings.rate_limit_window_seconds},
            )

        _rate_limit_store[client_ip].append(current_time)

    # Log request
    start_time = time.time()
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client_ip=client_ip,
    )

    # Process request
    response = await call_next(request)

    # Log response
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    # Clear context
    clear_context()

    return response


# =============================================================================
# Exception Handlers
# =============================================================================


async def research_agent_error_handler(
    _request: Request,
    exc: ResearchAgentError,
) -> JSONResponse:
    """Handle ResearchAgentError exceptions."""
    logger = get_logger()
    logger.error(
        "Request failed",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def generic_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger = get_logger()
    logger.exception("Unexpected error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {},
        },
    )


# =============================================================================
# Create Application
# =============================================================================

app = create_app()


# =============================================================================
# Health Check Endpoints
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness check response model."""

    status: str
    checks: dict[str, Any]


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Basic health check endpoint.

    Returns 200 if the service is running.
    Used by container orchestration for liveness checks.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.environment,
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check() -> ReadyResponse:
    """Readiness check endpoint.

    Verifies that all required services are available.
    Used by container orchestration for readiness checks.
    """
    settings = get_settings()
    checks: dict[str, Any] = {
        "api": True,
        "brain_configured": bool(settings.brain_service_url),
        "openai_configured": bool(settings.openai_api_key.get_secret_value()),
        "tavily_configured": bool(settings.tavily_api_key.get_secret_value()),
    }

    # TODO: Add actual connectivity checks to brain service

    all_ready = all(checks.values())

    return ReadyResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the application using uvicorn."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
