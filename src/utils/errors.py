"""Custom exceptions for the research agent.

All exceptions inherit from ResearchAgentError for consistent handling.
Each exception includes an error code for API responses.
"""

from typing import Any


class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""

    error_code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional additional details for debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(ResearchAgentError):
    """Raised when there's a configuration problem."""

    error_code = "CONFIGURATION_ERROR"
    status_code = 500


# =============================================================================
# Brain Service Errors
# =============================================================================


class BrainServiceError(ResearchAgentError):
    """Base error for brain service issues."""

    error_code = "BRAIN_SERVICE_ERROR"
    status_code = 502


class BrainConnectionError(BrainServiceError):
    """Raised when brain service is unreachable."""

    error_code = "BRAIN_CONNECTION_ERROR"


class BrainTimeoutError(BrainServiceError):
    """Raised when brain service request times out."""

    error_code = "BRAIN_TIMEOUT_ERROR"
    status_code = 504


class BrainInferenceError(BrainServiceError):
    """Raised when brain inference fails."""

    error_code = "BRAIN_INFERENCE_ERROR"


# =============================================================================
# Worker Errors
# =============================================================================


class WorkerError(ResearchAgentError):
    """Base error for worker issues."""

    error_code = "WORKER_ERROR"
    status_code = 502


class SearchError(WorkerError):
    """Raised when search operation fails."""

    error_code = "SEARCH_ERROR"


class ArxivError(WorkerError):
    """Raised when ArXiv API fails."""

    error_code = "ARXIV_ERROR"


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(ResearchAgentError):
    """Raised when input validation fails."""

    error_code = "VALIDATION_ERROR"
    status_code = 400


class QueryTooLongError(ValidationError):
    """Raised when research query exceeds maximum length."""

    error_code = "QUERY_TOO_LONG"


class InvalidDomainError(ValidationError):
    """Raised when an invalid research domain is specified."""

    error_code = "INVALID_DOMAIN"


# =============================================================================
# Rate Limiting Errors
# =============================================================================


class RateLimitError(ResearchAgentError):
    """Raised when rate limit is exceeded."""

    error_code = "RATE_LIMIT_EXCEEDED"
    status_code = 429


# =============================================================================
# Output Errors
# =============================================================================


class OutputError(ResearchAgentError):
    """Base error for output generation issues."""

    error_code = "OUTPUT_ERROR"
    status_code = 500


class LaTeXError(OutputError):
    """Raised when LaTeX generation or compilation fails."""

    error_code = "LATEX_ERROR"


class CitationError(OutputError):
    """Raised when citation processing fails."""

    error_code = "CITATION_ERROR"


# =============================================================================
# Workflow Errors
# =============================================================================


class WorkflowError(ResearchAgentError):
    """Base error for workflow/pipeline issues."""

    error_code = "WORKFLOW_ERROR"
    status_code = 500


class WorkflowTimeoutError(WorkflowError):
    """Raised when workflow exceeds time limit."""

    error_code = "WORKFLOW_TIMEOUT"
    status_code = 504


class MaxIterationsError(WorkflowError):
    """Raised when workflow exceeds maximum iterations."""

    error_code = "MAX_ITERATIONS_EXCEEDED"
