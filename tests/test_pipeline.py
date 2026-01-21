"""Tests for the research pipeline.

Tests state schema, node functions, graph routing,
and workflow execution.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.graph import (
    build_research_graph,
    get_graph_ascii,
    get_graph_mermaid,
    route_after_search,
    route_after_write,
    should_continue_research,
)
from src.pipeline.nodes import (
    _estimate_brain_cost,
    _format_sources_for_analysis,
    _generate_title,
    _parse_analysis_response,
    _parse_research_plan,
    _parse_search_queries,
)
from src.pipeline.state import (
    ResearchDomain,
    Source,
    WorkflowStatus,
    create_initial_state,
    state_to_summary,
)
from src.pipeline.workflow import (
    ResearchWorkflow,
    StreamUpdate,
    WorkflowResult,
)

# =============================================================================
# State Tests
# =============================================================================


class TestResearchState:
    """Tests for ResearchState schema and utilities."""

    def test_create_initial_state(self) -> None:
        """Test creating initial state with defaults."""
        state = create_initial_state(
            query="What is quantum entanglement?",
        )

        assert state["query"] == "What is quantum entanglement?"
        assert state["domains"] == [ResearchDomain.GENERAL.value]
        assert state["status"] == WorkflowStatus.PENDING.value
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 3
        assert state["cost_usd"] == 0.0
        assert state["tokens_used"] == 0
        assert len(state["thread_id"]) > 0

    def test_create_initial_state_with_domains(self) -> None:
        """Test creating initial state with specific domains."""
        state = create_initial_state(
            query="LLM scaling laws",
            domains=[ResearchDomain.AI_ML.value, ResearchDomain.GENERAL.value],
            max_iterations=5,
        )

        assert state["domains"] == ["ai_ml", "general"]
        assert state["max_iterations"] == 5

    def test_create_initial_state_with_thread_id(self) -> None:
        """Test creating initial state with custom thread ID."""
        state = create_initial_state(
            query="Test query",
            thread_id="custom-thread-123",
        )

        assert state["thread_id"] == "custom-thread-123"

    def test_state_to_summary(self) -> None:
        """Test converting state to summary dict."""
        state = create_initial_state(
            query="Test query",
            thread_id="test-123",
        )
        state["sources"] = [{"id": "1"}, {"id": "2"}]
        state["sections"] = {"intro": "content"}
        state["synthesis"] = "Some synthesis"
        state["quality_score"] = 4.5
        state["cost_usd"] = 0.05
        state["tokens_used"] = 5000

        summary = state_to_summary(state)

        assert summary["thread_id"] == "test-123"
        assert summary["query"] == "Test query"
        assert summary["status"] == "pending"
        assert summary["sources_count"] == 2
        assert summary["sections_count"] == 1
        assert summary["has_synthesis"] is True
        assert summary["quality_score"] == 4.5
        assert summary["cost_usd"] == 0.05
        assert summary["tokens_used"] == 5000


class TestResearchDomain:
    """Tests for ResearchDomain enum."""

    def test_domain_values(self) -> None:
        """Test domain enum values."""
        assert ResearchDomain.AI_ML.value == "ai_ml"
        assert ResearchDomain.QUANTUM_PHYSICS.value == "quantum_physics"
        assert ResearchDomain.ASTROPHYSICS.value == "astrophysics"
        assert ResearchDomain.GENERAL.value == "general"


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.PLANNING.value == "planning"
        assert WorkflowStatus.SEARCHING.value == "searching"
        assert WorkflowStatus.ANALYZING.value == "analyzing"
        assert WorkflowStatus.SYNTHESIZING.value == "synthesizing"
        assert WorkflowStatus.WRITING.value == "writing"
        assert WorkflowStatus.REVIEWING.value == "reviewing"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"


class TestSource:
    """Tests for Source dataclass."""

    def test_source_to_dict(self) -> None:
        """Test Source.to_dict() method."""
        source = Source(
            id="test-1",
            title="Test Paper",
            url="https://example.com",
            content="Some content",
            source_type="arxiv",
            relevance_score=0.85,
            authors=["Author One", "Author Two"],
            published_date="2026-01-01",
            bibtex="@article{test2026}",
        )

        d = source.to_dict()

        assert d["id"] == "test-1"
        assert d["title"] == "Test Paper"
        assert d["relevance_score"] == 0.85
        assert d["authors"] == ["Author One", "Author Two"]


# =============================================================================
# Graph Routing Tests
# =============================================================================


class TestConditionalRouting:
    """Tests for conditional edge functions."""

    def test_should_continue_research_with_gaps(self) -> None:
        """Test routing to generate_queries when gaps exist."""
        state = create_initial_state("test")
        state["needs_more_research"] = True
        state["iteration_count"] = 1
        state["max_iterations"] = 3

        result = should_continue_research(state)
        assert result == "generate_queries"

    def test_should_continue_research_max_iterations(self) -> None:
        """Test routing to synthesize when max iterations reached."""
        state = create_initial_state("test")
        state["needs_more_research"] = True
        state["iteration_count"] = 3
        state["max_iterations"] = 3

        result = should_continue_research(state)
        assert result == "synthesize"

    def test_should_continue_research_no_gaps(self) -> None:
        """Test routing to synthesize when no gaps."""
        state = create_initial_state("test")
        state["needs_more_research"] = False
        state["iteration_count"] = 1

        result = should_continue_research(state)
        assert result == "synthesize"

    def test_should_continue_research_failed(self) -> None:
        """Test routing to end when failed."""
        state = create_initial_state("test")
        state["status"] = WorkflowStatus.FAILED.value

        result = should_continue_research(state)
        assert result == "end"

    def test_route_after_search_success(self) -> None:
        """Test routing after successful search."""
        state = create_initial_state("test")
        state["status"] = WorkflowStatus.ANALYZING.value

        result = route_after_search(state)
        assert result == "analyze"

    def test_route_after_search_failed(self) -> None:
        """Test routing after failed search."""
        state = create_initial_state("test")
        state["status"] = WorkflowStatus.FAILED.value

        result = route_after_search(state)
        assert result == "end"

    def test_route_after_write_success(self) -> None:
        """Test routing after successful write."""
        state = create_initial_state("test")
        state["status"] = WorkflowStatus.REVIEWING.value

        result = route_after_write(state)
        assert result == "review"

    def test_route_after_write_failed(self) -> None:
        """Test routing after failed write."""
        state = create_initial_state("test")
        state["status"] = WorkflowStatus.FAILED.value

        result = route_after_write(state)
        assert result == "end"


class TestGraphBuilder:
    """Tests for graph building functions."""

    def test_build_research_graph(self) -> None:
        """Test that graph is built correctly."""
        graph = build_research_graph()

        # Verify nodes exist
        assert "plan" in graph.nodes
        assert "search" in graph.nodes
        assert "analyze" in graph.nodes
        assert "generate_queries" in graph.nodes
        assert "synthesize" in graph.nodes
        assert "write" in graph.nodes
        assert "review" in graph.nodes

    def test_get_graph_mermaid(self) -> None:
        """Test Mermaid diagram generation."""
        mermaid = get_graph_mermaid()

        assert "graph TD" in mermaid
        assert "plan" in mermaid
        assert "search" in mermaid
        assert "analyze" in mermaid
        assert "synthesize" in mermaid

    def test_get_graph_ascii(self) -> None:
        """Test ASCII diagram generation."""
        ascii_art = get_graph_ascii()

        assert "Research Workflow Graph" in ascii_art
        assert "plan" in ascii_art
        assert "search" in ascii_art
        assert "analyze" in ascii_art


# =============================================================================
# Node Parsing Tests
# =============================================================================


class TestNodeParsing:
    """Tests for node parsing functions."""

    def test_parse_research_plan_with_questions(self) -> None:
        """Test parsing plan with key questions."""
        content = """
KEY QUESTIONS:
1. What is the current state of research?
2. What are the main approaches?
3. What are the limitations?

SEARCH STRATEGY:
- Search for "current research" on web
- Look for "approaches" on arxiv

METHODOLOGY:
We will analyze sources systematically.

EXPECTED STRUCTURE:
1. Introduction
2. Literature Review
3. Analysis
4. Conclusion
"""
        plan = _parse_research_plan(content, ["ai_ml"])

        assert len(plan["key_questions"]) == 3
        assert "What is the current state of research?" in plan["key_questions"]
        assert len(plan["search_queries"]) >= 2
        assert "systematically" in plan["methodology"]
        assert "Introduction" in plan["expected_structure"]
        assert plan["domains"] == ["ai_ml"]

    def test_parse_research_plan_with_thinking_tags(self) -> None:
        """Test that thinking tags are removed."""
        content = """
<think>
Let me think about this...
I need to create a good plan.
</think>

KEY QUESTIONS:
1. Test question

SEARCH STRATEGY:
- Test search
"""
        plan = _parse_research_plan(content, [])

        assert "Let me think" not in str(plan)
        assert "Test question" in plan["key_questions"]

    def test_parse_analysis_response(self) -> None:
        """Test parsing analysis response."""
        content = """
SYNTHESIS:
The sources agree on several key points. There is a clear consensus on the methodology.

GAPS:
- Missing recent 2026 developments
- No coverage of alternative approaches

KEY INSIGHTS:
1. Important finding one
2. Important finding two

CONFLICTS:
Some sources disagree about the timeline.
"""
        analysis = _parse_analysis_response(content)

        assert "consensus" in analysis["synthesis"]
        assert len(analysis["gaps"]) == 2
        assert "Missing recent 2026 developments" in analysis["gaps"]
        assert len(analysis["key_findings"]) == 2
        assert "disagree" in analysis.get("conflicts", "")

    def test_parse_search_queries(self) -> None:
        """Test parsing search queries from brain response."""
        content = """
Here are the queries:

QUERY: machine learning scaling laws 2026
TARGET: arxiv

QUERY: latest transformer architectures
TARGET: web
"""
        queries = _parse_search_queries(content)

        assert len(queries) >= 2
        assert queries[0]["query"] == "machine learning scaling laws 2026"
        assert queries[0]["executed"] is False

    def test_format_sources_for_analysis(self) -> None:
        """Test formatting sources for analysis prompt."""
        sources = [
            {
                "title": "Test Paper One",
                "source_type": "arxiv",
                "url": "https://arxiv.org/abs/123",
                "content": "This is the content of paper one.",
            },
            {
                "title": "Web Article",
                "source_type": "web",
                "url": "https://example.com",
                "content": "This is web content.",
            },
        ]

        formatted = _format_sources_for_analysis(sources)

        assert "Source 1:" in formatted
        assert "Source 2:" in formatted
        assert "Test Paper One" in formatted
        assert "arxiv" in formatted
        assert "Web Article" in formatted

    def test_generate_title(self) -> None:
        """Test title generation from query."""
        query = "what are the effects of scaling laws on LLM performance?"
        title = _generate_title(query, [])

        # Note: capitalize() converts "LLM" to "Llm"
        assert title == "What Are the Effects of Scaling Laws on Llm Performance"

    def test_estimate_brain_cost(self) -> None:
        """Test brain cost estimation."""
        cost = _estimate_brain_cost(10000)

        # 10000 tokens * $0.001 / 1K = $0.01
        assert cost == pytest.approx(0.01, rel=0.01)


# =============================================================================
# Workflow Tests
# =============================================================================


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_workflow_result_to_dict(self) -> None:
        """Test WorkflowResult.to_dict() method."""
        result = WorkflowResult(
            thread_id="test-123",
            status="completed",
            title="Test Title",
            abstract="Test abstract",
            sections={"intro": "content"},
            citations=[{"id": "1"}],
            quality_score=4.5,
            cost_usd=0.123456789,
            tokens_used=5000,
            errors=[],
            started_at="2026-01-21T10:00:00",
            completed_at="2026-01-21T10:05:00",
        )

        d = result.to_dict()

        assert d["thread_id"] == "test-123"
        assert d["status"] == "completed"
        assert d["quality_score"] == 4.5
        assert d["cost_usd"] == 0.123457  # Rounded to 6 decimals
        assert d["tokens_used"] == 5000


class TestStreamUpdate:
    """Tests for StreamUpdate dataclass."""

    def test_stream_update_creation(self) -> None:
        """Test StreamUpdate creation."""
        update = StreamUpdate(
            node="search",
            status="running",
            message="Searching sources...",
            progress=0.3,
            data={"iteration": 1},
        )

        assert update.node == "search"
        assert update.status == "running"
        assert update.progress == 0.3
        assert update.data["iteration"] == 1


class TestResearchWorkflow:
    """Tests for ResearchWorkflow class."""

    def test_workflow_initialization(self) -> None:
        """Test workflow initialization."""
        workflow = ResearchWorkflow(use_checkpointing=False)

        assert workflow._use_checkpointing is False
        assert workflow._checkpointer is None

    def test_workflow_initialization_with_checkpointing(self) -> None:
        """Test workflow initialization with checkpointing."""
        workflow = ResearchWorkflow(use_checkpointing=True)

        assert workflow._use_checkpointing is True

    def test_state_to_result(self) -> None:
        """Test _state_to_result conversion."""
        workflow = ResearchWorkflow(use_checkpointing=False)

        state = create_initial_state("test query")
        state["status"] = "completed"
        state["title"] = "Test Title"
        state["abstract"] = "Test abstract"
        state["sections"] = {"intro": "content"}
        state["citations"] = [{"id": "1"}]
        state["quality_score"] = 4.0
        state["cost_usd"] = 0.5
        state["tokens_used"] = 10000
        state["completed_at"] = "2026-01-21T10:00:00"

        result = workflow._state_to_result(state)

        assert result.status == "completed"
        assert result.title == "Test Title"
        assert result.quality_score == 4.0
        assert result.cost_usd == 0.5

    def test_get_node_message(self) -> None:
        """Test _get_node_message for different nodes."""
        workflow = ResearchWorkflow(use_checkpointing=False)

        assert "plan" in workflow._get_node_message("plan", {}).lower()
        assert "search" in workflow._get_node_message("search", {"iteration_count": 2}).lower()
        assert "synthesiz" in workflow._get_node_message("synthesize", {}).lower()

    def test_node_progress_values(self) -> None:
        """Test that node progress values are properly ordered."""
        progress = ResearchWorkflow.NODE_PROGRESS

        assert progress["plan"] < progress["search"]
        assert progress["search"] < progress["analyze"]
        assert progress["synthesize"] < progress["write"]
        assert progress["write"] < progress["review"]


# =============================================================================
# Integration-Style Tests (with mocks)
# =============================================================================


class TestNodeFunctions:
    """Tests for node functions with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_plan_node_success(self) -> None:
        """Test plan_node with mocked brain."""
        from src.pipeline.nodes import plan_node

        state = create_initial_state("What is quantum computing?")

        mock_response = MagicMock()
        mock_response.content = """
<think>Let me plan this research.</think>

KEY QUESTIONS:
1. What is quantum computing?
2. How does it work?

SEARCH STRATEGY:
- Search for quantum computing basics
- Search for quantum algorithms

METHODOLOGY:
Systematic analysis of sources.

EXPECTED STRUCTURE:
1. Introduction
2. Background
3. Analysis
4. Conclusion
"""
        mock_response.tokens_used = 1000

        with patch("src.pipeline.nodes.BrainClient") as MockBrain:
            mock_brain = AsyncMock()
            mock_brain.generate.return_value = mock_response
            mock_brain.close = AsyncMock()
            MockBrain.return_value = mock_brain

            result = await plan_node(state)

        assert result["status"] == "searching"
        assert result["research_plan"] is not None
        assert len(result["search_queries"]) > 0
        assert result["last_node"] == "plan"
        assert result["tokens_used"] > 0

    @pytest.mark.asyncio
    async def test_plan_node_failure(self) -> None:
        """Test plan_node handles errors gracefully."""
        from src.pipeline.nodes import plan_node

        state = create_initial_state("Test query")

        with patch("src.pipeline.nodes.BrainClient") as MockBrain:
            mock_brain = AsyncMock()
            mock_brain.generate.side_effect = Exception("Brain error")
            mock_brain.close = AsyncMock()
            MockBrain.return_value = mock_brain

            result = await plan_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "Planning failed" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_search_node_no_queries(self) -> None:
        """Test search_node with no pending queries."""
        from src.pipeline.nodes import search_node

        state = create_initial_state("Test query")
        state["search_queries"] = [
            {"query": "test", "target": "web", "priority": "high", "executed": True},
        ]

        result = await search_node(state)

        # Should still transition to analyzing
        assert result["status"] == "analyzing"
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_analyze_node_no_sources(self) -> None:
        """Test analyze_node with no sources."""
        from src.pipeline.nodes import analyze_node

        state = create_initial_state("Test query")
        state["sources"] = []
        state["arxiv_papers"] = []

        result = await analyze_node(state)

        assert result["needs_more_research"] is True
        # knowledge_gaps is a list, check if any element contains the substring
        assert any("No sources found" in gap for gap in result["knowledge_gaps"])

    @pytest.mark.asyncio
    async def test_synthesize_node_success(self) -> None:
        """Test synthesize_node with mocked brain."""
        from src.pipeline.nodes import synthesize_node

        state = create_initial_state("Test query")
        state["sources"] = [{"title": "Test", "content": "Content"}]
        state["key_findings"] = ["Finding 1", "Finding 2"]
        state["research_plan"] = {
            "expected_structure": ["Intro", "Methods", "Results"],
        }

        mock_response = MagicMock()
        mock_response.content = "This is the synthesized research narrative."
        mock_response.tokens_used = 2000

        with patch("src.pipeline.nodes.BrainClient") as MockBrain:
            mock_brain = AsyncMock()
            mock_brain.generate.return_value = mock_response
            mock_brain.close = AsyncMock()
            MockBrain.return_value = mock_brain

            result = await synthesize_node(state)

        assert result["status"] == "writing"
        assert result["synthesis"] is not None
        assert result["paper_outline"] is not None
        assert result["title"] is not None

    @pytest.mark.asyncio
    async def test_review_node_success(self) -> None:
        """Test review_node with mocked brain."""
        from src.pipeline.nodes import review_node

        state = create_initial_state("Test query")
        state["title"] = "Test Paper"
        state["sections"] = {"intro": "Introduction content"}
        state["sources"] = [{"title": "Source", "content": "Content"}]

        mock_response = MagicMock()
        mock_response.content = """
The paper is well-structured.

QUALITY: 4.5

Recommendations:
- Add more citations
"""
        mock_response.tokens_used = 1500

        with patch("src.pipeline.nodes.BrainClient") as MockBrain:
            mock_brain = AsyncMock()
            mock_brain.generate.return_value = mock_response
            mock_brain.close = AsyncMock()
            MockBrain.return_value = mock_brain

            result = await review_node(state)

        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert result["review_feedback"] is not None
        assert result["quality_score"] == 4.5


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestAPIEndpoints:
    """Tests for research API endpoints."""

    @pytest.fixture
    def client(self) -> Any:
        """Create test client."""
        from fastapi.testclient import TestClient

        from src.main import app

        return TestClient(app)

    def test_get_workflow_graph(self, client: Any) -> None:
        """Test GET /research/graph endpoint."""
        response = client.get("/research/graph")

        assert response.status_code == 200
        data = response.json()
        assert "ascii_diagram" in data
        assert "node_descriptions" in data
        assert "plan" in data["node_descriptions"]
        assert "search" in data["node_descriptions"]
