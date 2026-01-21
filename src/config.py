"""Application configuration using pydantic-settings.

All configuration is loaded from environment variables.
Secrets are validated on startup - the app will fail fast if required vars are missing.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # API Keys (Required)
    openai_api_key: SecretStr = Field(..., description="OpenAI API key for GPT-4o-mini")
    tavily_api_key: SecretStr = Field(..., description="Tavily API key for web search")

    # Brain Service
    brain_service_url: str = Field(
        default="http://localhost:8001",
        description="URL of the vLLM brain service",
    )
    brain_api_key: SecretStr = Field(..., description="API key for brain service auth")
    brain_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    brain_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    brain_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=128000, ge=1024)

    # Rate Limiting
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    # Output
    output_dir: Path = Path("outputs")
    compile_pdf: bool = True

    # Workers
    max_search_workers: int = Field(default=5, ge=1, le=10)
    api_timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Workflow
    max_search_iterations: int = Field(default=3, ge=1, le=10)

    # Optional: Observability
    otel_exporter_otlp_endpoint: str | None = None
    metrics_port: int = 9090

    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_output_dir_exists(cls, v: str | Path) -> Path:
        """Ensure output directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    This will raise a validation error if required environment variables are missing.
    """
    return Settings()
