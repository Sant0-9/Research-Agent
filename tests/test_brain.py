"""Tests for brain service client and prompts."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain import (
    ANALYZE_PAPER,
    ANALYZE_SOURCES,
    RESEARCH_PLAN,
    SYNTHESIZE_FINDINGS,
    WRITE_SECTION,
    BrainClient,
    BrainResponse,
    PromptTemplate,
    get_all_prompts,
)


class TestPromptTemplate:
    """Test cases for PromptTemplate class."""

    def test_template_format_basic(self):
        """Test basic template formatting."""
        template = PromptTemplate(
            template="Hello {name}, your query is: {query}",
            description="Test template",
        )
        result = template.format(name="User", query="What is AI?")

        assert result == "Hello User, your query is: What is AI?"

    def test_template_format_with_multiline(self):
        """Test template formatting with multiline content."""
        template = PromptTemplate(
            template="Topic: {topic}\n\nContent:\n{content}",
            description="Multiline template",
        )
        result = template.format(topic="Testing", content="Line 1\nLine 2")

        assert "Topic: Testing" in result
        assert "Line 1\nLine 2" in result

    def test_template_has_description(self):
        """Test that templates have descriptions."""
        template = PromptTemplate(
            template="Test",
            description="A test template",
        )

        assert template.description == "A test template"


class TestBuiltInPrompts:
    """Test cases for built-in prompt templates."""

    def test_research_plan_prompt_format(self):
        """Test RESEARCH_PLAN prompt formatting."""
        result = RESEARCH_PLAN.format(
            query="What is quantum entanglement?",
            domains="Quantum Physics",
        )

        assert "What is quantum entanglement?" in result
        assert "Quantum Physics" in result
        assert "KEY QUESTIONS" in result
        assert "SEARCH STRATEGY" in result

    def test_analyze_sources_prompt_format(self):
        """Test ANALYZE_SOURCES prompt formatting."""
        result = ANALYZE_SOURCES.format(
            topic="Machine Learning",
            sources="Source 1: Introduction to ML\nSource 2: Deep Learning basics",
        )

        assert "Machine Learning" in result
        assert "Source 1" in result
        assert "RELEVANCE" in result
        assert "SYNTHESIS" in result

    def test_analyze_paper_prompt_format(self):
        """Test ANALYZE_PAPER prompt formatting."""
        result = ANALYZE_PAPER.format(
            title="Test Paper",
            authors="Author One, Author Two",
            abstract="This is a test abstract.",
            content="Full paper content here.",
            research_topic="AI Safety",
        )

        assert "Test Paper" in result
        assert "Author One" in result
        assert "AI Safety" in result
        assert "MAIN CONTRIBUTIONS" in result

    def test_synthesize_findings_prompt_format(self):
        """Test SYNTHESIZE_FINDINGS prompt formatting."""
        result = SYNTHESIZE_FINDINGS.format(
            query="How does attention work?",
            analyzed_sources="Source analysis here",
            key_findings="Key finding 1\nKey finding 2",
        )

        assert "How does attention work?" in result
        assert "INTRODUCTION" in result
        assert "IMPLICATIONS" in result

    def test_write_section_prompt_format(self):
        """Test WRITE_SECTION prompt formatting."""
        result = WRITE_SECTION.format(
            topic="Neural Networks",
            section_name="Introduction",
            outline="1. History\n2. Basics",
            sources="Relevant sources here",
            length_guidance="2-3 paragraphs",
        )

        assert "Neural Networks" in result
        assert "Introduction" in result
        assert "2-3 paragraphs" in result

    def test_get_all_prompts_returns_dict(self):
        """Test that get_all_prompts returns all prompts."""
        prompts = get_all_prompts()

        assert isinstance(prompts, dict)
        assert "research_plan" in prompts
        assert "analyze_sources" in prompts
        assert "synthesize_findings" in prompts
        assert len(prompts) >= 10  # We defined at least 10 prompts

    def test_all_prompts_are_prompt_templates(self):
        """Test that all prompts are PromptTemplate instances."""
        prompts = get_all_prompts()

        for name, prompt in prompts.items():
            assert isinstance(prompt, PromptTemplate), f"{name} is not a PromptTemplate"
            assert prompt.template, f"{name} has empty template"
            assert prompt.description, f"{name} has empty description"


class TestBrainResponse:
    """Test cases for BrainResponse dataclass."""

    def test_brain_response_creation(self):
        """Test creating a BrainResponse."""
        response = BrainResponse(
            content="Test content",
            thinking="Test thinking",
            tokens_used=100,
            model="test-model",
            finish_reason="stop",
        )

        assert response.content == "Test content"
        assert response.thinking == "Test thinking"
        assert response.tokens_used == 100
        assert response.model == "test-model"
        assert response.finish_reason == "stop"

    def test_brain_response_without_thinking(self):
        """Test BrainResponse with no thinking content."""
        response = BrainResponse(
            content="Direct answer",
            thinking=None,
            tokens_used=50,
            model="test-model",
            finish_reason="stop",
        )

        assert response.thinking is None


class TestBrainClient:
    """Test cases for BrainClient class."""

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create a mock OpenAI API response."""
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a test response."
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.total_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

        return mock_response

    @pytest.fixture
    def mock_openai_response_with_thinking(self) -> MagicMock:
        """Create a mock response with <think> tags."""
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "<think>\nLet me analyze this...\nStep 1: Consider the options.\n</think>\n"
            "The answer is 42."
        )
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.total_tokens = 100

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

        return mock_response

    @pytest.mark.asyncio
    async def test_brain_client_generate(self, mock_openai_response: MagicMock):
        """Test basic generation."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            response = await client.generate("What is 2 + 2?")

            assert response.content == "This is a test response."
            assert response.tokens_used == 50
            assert response.finish_reason == "stop"

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_client_extracts_thinking(
        self, mock_openai_response_with_thinking: MagicMock
    ):
        """Test that thinking content is extracted from <think> tags."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response_with_thinking
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            response = await client.generate("What is the meaning of life?")

            assert response.thinking is not None
            assert "Let me analyze this" in response.thinking
            assert "The answer is 42" in response.content

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_client_no_system_prompt(self, mock_openai_response: MagicMock):
        """Test that no system prompt is sent (R1-Distill requirement)."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            await client.generate("Test prompt")

            # Verify the call
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]

            # Should only have user message, no system message
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_client_uses_correct_temperature(
        self, mock_openai_response: MagicMock
    ):
        """Test that default temperature is used from settings."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            await client.generate("Test prompt")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs

            # Default temperature should be 0.6
            assert call_kwargs["temperature"] == 0.6

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_client_custom_temperature(
        self, mock_openai_response: MagicMock
    ):
        """Test overriding temperature."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            await client.generate("Test prompt", temperature=0.3)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.3

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_client_health_check_success(self):
        """Test health check when service is healthy."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_models = MagicMock()
            mock_models.data = [MagicMock()]  # At least one model

            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            is_healthy = await client.health_check()

            assert is_healthy is True

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_client_health_check_failure(self):
        """Test health check when service is unhealthy."""
        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()
            is_healthy = await client.health_check()

            assert is_healthy is False

            await client.close()


class TestBrainClientErrors:
    """Test cases for BrainClient error handling."""

    @pytest.mark.asyncio
    async def test_brain_connection_error(self):
        """Test handling of connection errors."""
        from openai import APIConnectionError

        from src.utils.errors import BrainConnectionError

        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError(request=MagicMock())
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()

            with pytest.raises(BrainConnectionError):
                await client.generate("Test prompt")

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_timeout_error(self):
        """Test handling of timeout errors."""
        from openai import APITimeoutError

        from src.utils.errors import BrainTimeoutError

        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=APITimeoutError(request=MagicMock())
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()

            with pytest.raises(BrainTimeoutError):
                await client.generate("Test prompt")

            await client.close()

    @pytest.mark.asyncio
    async def test_brain_inference_error(self):
        """Test handling of general inference errors."""
        from src.utils.errors import BrainInferenceError

        with patch("src.brain.client.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=ValueError("Unexpected error")
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            client = BrainClient()

            with pytest.raises(BrainInferenceError):
                await client.generate("Test prompt")

            await client.close()


class TestThinkingExtraction:
    """Test cases for <think> tag extraction."""

    def test_extract_single_think_block(self):
        """Test extracting a single <think> block."""
        from src.brain.client import BrainClient

        client = BrainClient.__new__(BrainClient)

        content = "<think>This is my reasoning.</think>\nFinal answer."
        result = client._extract_thinking(content)

        assert result == "This is my reasoning."

    def test_extract_multiple_think_blocks(self):
        """Test extracting multiple <think> blocks."""
        from src.brain.client import BrainClient

        client = BrainClient.__new__(BrainClient)

        content = (
            "<think>First thought.</think>\n"
            "Some text.\n"
            "<think>Second thought.</think>\n"
            "Final answer."
        )
        result = client._extract_thinking(content)

        assert "First thought" in result
        assert "Second thought" in result

    def test_extract_multiline_think_block(self):
        """Test extracting multiline <think> content."""
        from src.brain.client import BrainClient

        client = BrainClient.__new__(BrainClient)

        content = "<think>\nLine 1\nLine 2\nLine 3\n</think>\nAnswer."
        result = client._extract_thinking(content)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_extract_no_think_block(self):
        """Test when there are no <think> blocks."""
        from src.brain.client import BrainClient

        client = BrainClient.__new__(BrainClient)

        content = "Just a direct answer with no thinking."
        result = client._extract_thinking(content)

        assert result is None
