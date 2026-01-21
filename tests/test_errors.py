"""Tests for custom error classes."""

import pytest

from src.utils.errors import (
    ArxivError,
    BrainConnectionError,
    BrainInferenceError,
    BrainServiceError,
    BrainTimeoutError,
    CitationError,
    ConfigurationError,
    InvalidDomainError,
    LaTeXError,
    MaxIterationsError,
    OutputError,
    QueryTooLongError,
    RateLimitError,
    ResearchAgentError,
    SearchError,
    ValidationError,
    WorkerError,
    WorkflowError,
    WorkflowTimeoutError,
)


class TestResearchAgentError:
    """Test cases for base ResearchAgentError."""

    def test_base_error_attributes(self):
        """Test that base error has correct attributes."""
        error = ResearchAgentError("Test error", details={"key": "value"})

        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert error.error_code == "INTERNAL_ERROR"
        assert error.status_code == 500

    def test_to_dict(self):
        """Test to_dict method."""
        error = ResearchAgentError("Test error", details={"key": "value"})
        result = error.to_dict()

        assert result["error"] == "INTERNAL_ERROR"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}

    def test_empty_details_default(self):
        """Test that details defaults to empty dict."""
        error = ResearchAgentError("Test error")

        assert error.details == {}


class TestBrainServiceErrors:
    """Test cases for brain service errors."""

    def test_brain_service_error(self):
        """Test BrainServiceError attributes."""
        error = BrainServiceError("Brain failed")

        assert error.error_code == "BRAIN_SERVICE_ERROR"
        assert error.status_code == 502

    def test_brain_connection_error(self):
        """Test BrainConnectionError attributes."""
        error = BrainConnectionError("Connection refused")

        assert error.error_code == "BRAIN_CONNECTION_ERROR"
        assert error.status_code == 502

    def test_brain_timeout_error(self):
        """Test BrainTimeoutError attributes."""
        error = BrainTimeoutError("Request timed out")

        assert error.error_code == "BRAIN_TIMEOUT_ERROR"
        assert error.status_code == 504

    def test_brain_inference_error(self):
        """Test BrainInferenceError attributes."""
        error = BrainInferenceError("Inference failed")

        assert error.error_code == "BRAIN_INFERENCE_ERROR"
        assert error.status_code == 502


class TestWorkerErrors:
    """Test cases for worker errors."""

    def test_worker_error(self):
        """Test WorkerError attributes."""
        error = WorkerError("Worker failed")

        assert error.error_code == "WORKER_ERROR"
        assert error.status_code == 502

    def test_search_error(self):
        """Test SearchError attributes."""
        error = SearchError("Search failed")

        assert error.error_code == "SEARCH_ERROR"

    def test_arxiv_error(self):
        """Test ArxivError attributes."""
        error = ArxivError("ArXiv API failed")

        assert error.error_code == "ARXIV_ERROR"


class TestValidationErrors:
    """Test cases for validation errors."""

    def test_validation_error(self):
        """Test ValidationError attributes."""
        error = ValidationError("Invalid input")

        assert error.error_code == "VALIDATION_ERROR"
        assert error.status_code == 400

    def test_query_too_long_error(self):
        """Test QueryTooLongError attributes."""
        error = QueryTooLongError("Query exceeds 10000 characters")

        assert error.error_code == "QUERY_TOO_LONG"
        assert error.status_code == 400

    def test_invalid_domain_error(self):
        """Test InvalidDomainError attributes."""
        error = InvalidDomainError("Unknown domain: biology")

        assert error.error_code == "INVALID_DOMAIN"
        assert error.status_code == 400


class TestOutputErrors:
    """Test cases for output errors."""

    def test_output_error(self):
        """Test OutputError attributes."""
        error = OutputError("Output generation failed")

        assert error.error_code == "OUTPUT_ERROR"
        assert error.status_code == 500

    def test_latex_error(self):
        """Test LaTeXError attributes."""
        error = LaTeXError("LaTeX compilation failed")

        assert error.error_code == "LATEX_ERROR"

    def test_citation_error(self):
        """Test CitationError attributes."""
        error = CitationError("Invalid citation format")

        assert error.error_code == "CITATION_ERROR"


class TestWorkflowErrors:
    """Test cases for workflow errors."""

    def test_workflow_error(self):
        """Test WorkflowError attributes."""
        error = WorkflowError("Workflow failed")

        assert error.error_code == "WORKFLOW_ERROR"
        assert error.status_code == 500

    def test_workflow_timeout_error(self):
        """Test WorkflowTimeoutError attributes."""
        error = WorkflowTimeoutError("Workflow timed out after 30 minutes")

        assert error.error_code == "WORKFLOW_TIMEOUT"
        assert error.status_code == 504

    def test_max_iterations_error(self):
        """Test MaxIterationsError attributes."""
        error = MaxIterationsError("Exceeded maximum of 10 iterations")

        assert error.error_code == "MAX_ITERATIONS_EXCEEDED"


class TestRateLimitError:
    """Test cases for rate limit error."""

    def test_rate_limit_error(self):
        """Test RateLimitError attributes."""
        error = RateLimitError(
            "Rate limit exceeded",
            details={"retry_after_seconds": 60},
        )

        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.status_code == 429
        assert error.details["retry_after_seconds"] == 60


class TestErrorInheritance:
    """Test that error inheritance works correctly."""

    def test_brain_errors_inherit_from_research_agent_error(self):
        """Test brain errors inherit from ResearchAgentError."""
        error = BrainConnectionError("Test")

        assert isinstance(error, ResearchAgentError)
        assert isinstance(error, BrainServiceError)

    def test_worker_errors_inherit_from_research_agent_error(self):
        """Test worker errors inherit from ResearchAgentError."""
        error = SearchError("Test")

        assert isinstance(error, ResearchAgentError)
        assert isinstance(error, WorkerError)

    def test_validation_errors_inherit_from_research_agent_error(self):
        """Test validation errors inherit from ResearchAgentError."""
        error = QueryTooLongError("Test")

        assert isinstance(error, ResearchAgentError)
        assert isinstance(error, ValidationError)
