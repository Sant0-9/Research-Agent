"""Brain service client for vLLM-hosted DeepSeek-R1-Distill model.

Uses the OpenAI Python SDK with vLLM's OpenAI-compatible API.
Implements async streaming, retry logic, and proper error handling.

Key DeepSeek-R1-Distill best practices (from research):
- NO system prompts - put all instructions in user message
- NO few-shot prompting - degrades performance
- Temperature 0.5-0.7 (0.6 optimal), top_p 0.95
- Model uses <think> tags for reasoning output
"""

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from src.config import get_settings
from src.utils.errors import (
    BrainConnectionError,
    BrainInferenceError,
    BrainTimeoutError,
)
from src.utils.logging import get_logger


@dataclass
class BrainResponse:
    """Response from the brain service."""

    content: str
    thinking: str | None  # Content inside <think> tags
    tokens_used: int
    model: str
    finish_reason: str | None


@dataclass
class StreamChunk:
    """A single chunk from streaming response."""

    content: str
    is_thinking: bool  # True if this chunk is inside <think> tags
    finish_reason: str | None = None


class BrainClient:
    """Async client for the vLLM brain service.

    Uses OpenAI SDK with vLLM's OpenAI-compatible API endpoint.
    Handles streaming, retries, and DeepSeek-R1 specific patterns.
    """

    def __init__(self) -> None:
        """Initialize the brain client."""
        self._settings = get_settings()
        self._logger = get_logger(__name__)

        # Configure OpenAI client to point to vLLM server
        self._client = AsyncOpenAI(
            api_key=self._settings.brain_api_key.get_secret_value(),
            base_url=f"{self._settings.brain_service_url}/v1",
            timeout=httpx.Timeout(
                timeout=float(self._settings.api_timeout_seconds),
                connect=10.0,
            ),
            max_retries=0,  # We handle retries ourselves for better control
        )

        self._model = self._settings.brain_model_name

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        force_thinking: bool = False,
    ) -> BrainResponse:
        """Generate a response from the brain.

        Args:
            prompt: The user prompt (NO system prompt for R1-Distill).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (default from config: 0.6).
            top_p: Top-p sampling (default from config: 0.95).
            stop: Stop sequences.
            force_thinking: If True, prepend <think> to force reasoning mode.

        Returns:
            BrainResponse with content, thinking, and usage info.

        Raises:
            BrainConnectionError: If connection to vLLM server fails.
            BrainTimeoutError: If request times out.
            BrainInferenceError: If inference fails.
        """
        # Apply defaults from config
        temperature = temperature if temperature is not None else self._settings.brain_temperature
        top_p = top_p if top_p is not None else self._settings.brain_top_p

        # For R1-Distill: No system message, all instructions in user message
        messages = [{"role": "user", "content": prompt}]

        # Force thinking mode if requested
        prefill = None
        if force_thinking:
            prefill = "<think>\n"

        return await self._generate_with_retry(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            prefill=prefill,
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        force_thinking: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response from the brain.

        Args:
            prompt: The user prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            stop: Stop sequences.
            force_thinking: If True, prepend <think> to force reasoning mode.

        Yields:
            StreamChunk objects with content and thinking state.

        Raises:
            BrainConnectionError: If connection fails.
            BrainTimeoutError: If request times out.
            BrainInferenceError: If inference fails.
        """
        temperature = temperature if temperature is not None else self._settings.brain_temperature
        top_p = top_p if top_p is not None else self._settings.brain_top_p

        messages = [{"role": "user", "content": prompt}]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        if stop:
            kwargs["stop"] = stop

        # Force thinking with assistant prefill
        if force_thinking:
            messages.append({"role": "assistant", "content": "<think>\n"})

        try:
            stream = await self._client.chat.completions.create(**kwargs)

            # Track whether we're inside <think> tags
            in_thinking = force_thinking
            buffer = ""

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    buffer += content

                    # Check for <think> tag transitions
                    if "<think>" in buffer and not in_thinking:
                        in_thinking = True
                    if "</think>" in buffer and in_thinking:
                        in_thinking = False

                    yield StreamChunk(
                        content=content,
                        is_thinking=in_thinking,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
                elif chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        content="",
                        is_thinking=False,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except APIConnectionError as e:
            self._logger.error("Brain connection failed", error=str(e))
            raise BrainConnectionError(
                f"Failed to connect to brain service: {e}",
                details={"url": self._settings.brain_service_url},
            ) from e
        except APITimeoutError as e:
            self._logger.error("Brain request timed out", error=str(e))
            raise BrainTimeoutError(
                f"Brain request timed out after {self._settings.api_timeout_seconds}s",
            ) from e
        except Exception as e:
            self._logger.error("Brain inference failed", error=str(e))
            raise BrainInferenceError(f"Brain inference failed: {e}") from e

    async def _generate_with_retry(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        prefill: str | None,
    ) -> BrainResponse:
        """Execute generation with retry logic.

        Implements exponential backoff for transient failures.
        """
        last_error: Exception | None = None
        max_retries = self._settings.max_retries

        for attempt in range(max_retries + 1):
            try:
                return await self._execute_generation(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    prefill=prefill,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    self._logger.warning(
                        "Brain request failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_seconds=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._logger.error(
                        "Brain request failed after all retries",
                        attempts=max_retries + 1,
                        error=str(e),
                    )

        # Convert to appropriate error type
        # Check APITimeoutError first as it may inherit from APIConnectionError
        if isinstance(last_error, APITimeoutError):
            raise BrainTimeoutError(
                f"Brain request timed out after {max_retries + 1} attempts",
            ) from last_error
        elif isinstance(last_error, APIConnectionError):
            raise BrainConnectionError(
                f"Failed to connect to brain service after {max_retries + 1} attempts",
                details={"url": self._settings.brain_service_url},
            ) from last_error
        else:
            raise BrainInferenceError(
                f"Brain inference failed: {last_error}",
            ) from last_error

    async def _execute_generation(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        prefill: str | None,
    ) -> BrainResponse:
        """Execute a single generation request."""
        # Build request
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages.copy(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop:
            kwargs["stop"] = stop

        # Add assistant prefill for forced thinking
        if prefill:
            kwargs["messages"].append({"role": "assistant", "content": prefill})

        try:
            response = await self._client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""

            # If we used prefill, prepend it to the response
            if prefill:
                content = prefill + content

            # Extract thinking content from <think> tags
            thinking = self._extract_thinking(content)

            # Calculate tokens
            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.total_tokens

            return BrainResponse(
                content=content,
                thinking=thinking,
                tokens_used=tokens_used,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
            )

        except APIConnectionError:
            raise
        except APITimeoutError:
            raise
        except RateLimitError:
            raise
        except Exception as e:
            self._logger.error("Brain inference failed", error=str(e))
            raise BrainInferenceError(f"Brain inference failed: {e}") from e

    def _extract_thinking(self, content: str) -> str | None:
        """Extract content from <think> tags.

        DeepSeek-R1 models use <think>...</think> tags to show reasoning.
        """
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return "\n".join(match.strip() for match in matches)
        return None

    async def health_check(self) -> bool:
        """Check if the brain service is healthy.

        Returns:
            True if service is reachable and responding.
        """
        try:
            # Use models endpoint to verify vLLM is running
            models = await self._client.models.list()
            return len(models.data) > 0
        except Exception as e:
            self._logger.warning("Brain health check failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the client connections."""
        await self._client.close()


# Convenience function for one-off generations
async def generate_brain_response(
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = None,
    force_thinking: bool = False,
) -> BrainResponse:
    """Generate a one-off response from the brain.

    Creates a temporary client for simple use cases.
    For repeated calls, use BrainClient directly.
    """
    client = BrainClient()
    try:
        return await client.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            force_thinking=force_thinking,
        )
    finally:
        await client.close()
