# Research Agent Dockerfile
# Multi-stage build for smaller production images

# =============================================================================
# Base stage - Common dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # pip configuration
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # For LaTeX PDF compilation (optional, remove if not needed)
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-bibtex-extra \
    biber \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# =============================================================================
# Dependencies stage - Install Python packages
# =============================================================================
FROM base AS dependencies

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better caching
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir .

# =============================================================================
# Development stage - For local development with hot reload
# =============================================================================
FROM dependencies AS development

# Install dev dependencies
RUN pip install --no-cache-dir ".[dev]"

# Copy source code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup tests/ ./tests/

# Create outputs directory
RUN mkdir -p outputs && chown appuser:appgroup outputs

USER appuser

# Run with hot reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production stage - Minimal image for deployment
# =============================================================================
FROM base AS production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy source code
COPY --chown=appuser:appgroup src/ ./src/

# Create outputs directory
RUN mkdir -p outputs && chown appuser:appgroup outputs

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with production settings
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
