"""Pytest configuration and fixtures for research agent tests."""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "development"
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"
os.environ["TAVILY_API_KEY"] = "tvly-test-key-for-testing"
os.environ["BRAIN_API_KEY"] = "test-brain-key"
os.environ["BRAIN_SERVICE_URL"] = "http://localhost:8001"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LOG_JSON"] = "false"


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Use asyncio as the async backend for tests."""
    return "asyncio"


@pytest.fixture(scope="module")
def test_settings() -> Generator[Any, None, None]:
    """Provide test settings with mocked secrets."""
    from src.config import Settings

    settings = Settings(
        environment="development",
        openai_api_key="sk-test-key-for-testing",  # type: ignore[arg-type]
        tavily_api_key="tvly-test-key-for-testing",  # type: ignore[arg-type]
        brain_api_key="test-brain-key",  # type: ignore[arg-type]
        brain_service_url="http://localhost:8001",
        log_level="DEBUG",
        log_json=False,
    )
    yield settings


@pytest.fixture(scope="module")
def app() -> Generator[Any, None, None]:
    """Create a FastAPI application for testing."""
    from src.main import create_app

    test_app = create_app()
    yield test_app


@pytest.fixture(scope="module")
def client(app: Any) -> Generator[TestClient, None, None]:
    """Create a synchronous test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app: Any) -> AsyncGenerator[AsyncClient, None]:
    """Create an asynchronous test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def mock_brain_response() -> dict[str, Any]:
    """Mock response from brain service."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the brain.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_search_results() -> list[dict[str, Any]]:
    """Mock search results from Tavily."""
    return [
        {
            "title": "Test Article 1",
            "url": "https://example.com/article1",
            "content": "This is the content of test article 1.",
            "score": 0.95,
        },
        {
            "title": "Test Article 2",
            "url": "https://example.com/article2",
            "content": "This is the content of test article 2.",
            "score": 0.85,
        },
    ]


@pytest.fixture
def mock_arxiv_paper() -> dict[str, Any]:
    """Mock paper from ArXiv."""
    return {
        "arxiv_id": "2401.12345",
        "title": "Test Paper: A Study of Testing",
        "authors": ["Author One", "Author Two"],
        "abstract": "This is a test abstract for a mock paper.",
        "categories": ["cs.AI", "cs.LG"],
        "published": "2024-01-15",
        "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
    }


@pytest.fixture
def sample_research_query() -> str:
    """Sample research query for testing."""
    return "What are the latest advances in quantum computing error correction?"


@pytest.fixture
def sample_domains() -> list[str]:
    """Sample research domains."""
    return ["quantum_physics", "ml"]
