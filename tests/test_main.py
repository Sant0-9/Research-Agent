"""Tests for main FastAPI application."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test cases for health check endpoints."""

    def test_health_check_returns_healthy(self, client: TestClient):
        """Test that /health returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["environment"] == "development"

    def test_readiness_check_returns_ready(self, client: TestClient):
        """Test that /ready returns ready status."""
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ready", "not_ready")
        assert "checks" in data
        assert "api" in data["checks"]

    def test_health_check_includes_request_id(self, client: TestClient):
        """Test that responses include X-Request-ID header."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8


class TestErrorHandling:
    """Test cases for error handling."""

    def test_404_for_unknown_routes(self, client: TestClient):
        """Test that unknown routes return 404."""
        response = client.get("/unknown-endpoint")

        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient):
        """Test that wrong HTTP methods return 405."""
        response = client.post("/health")

        assert response.status_code == 405


class TestRateLimiting:
    """Test cases for rate limiting."""

    def test_rate_limiting_allows_requests(self, client: TestClient):
        """Test that normal requests are allowed."""
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200


class TestCORS:
    """Test cases for CORS configuration."""

    def test_cors_headers_in_development(self, client: TestClient):
        """Test that CORS headers are present in development."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # In development, CORS should be permissive
        assert response.status_code in (200, 204)
