"""Research workflow state schema for LangGraph.

Defines the state structure that flows through the research pipeline.
Uses TypedDict with Annotated types for proper state management.

Key design decisions:
- State is explicit and typed
- Use operator.add for list fields to enable additive updates
- Immutable workflow status via enum
- Cost tracking built into state
"""

import operator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, TypedDict


class WorkflowStatus(str, Enum):
    """Status of the research workflow."""

    PENDING = "pending"
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    WRITING = "writing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDomain(str, Enum):
    """Supported research domains."""

    AI_ML = "ai_ml"
    QUANTUM_PHYSICS = "quantum_physics"
    ASTROPHYSICS = "astrophysics"
    GENERAL = "general"


@dataclass
class Source:
    """A source used in research."""

    id: str
    title: str
    url: str
    content: str
    source_type: str  # "web", "arxiv", "paper"
    relevance_score: float = 0.0
    authors: list[str] = field(default_factory=list)
    published_date: str | None = None
    bibtex: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source_type": self.source_type,
            "relevance_score": self.relevance_score,
            "authors": self.authors,
            "published_date": self.published_date,
            "bibtex": self.bibtex,
            "metadata": self.metadata,
        }


@dataclass
class SearchQuery:
    """A search query to execute."""

    query: str
    target: str  # "web", "arxiv", "both"
    priority: str  # "high", "medium", "low"
    rationale: str = ""
    executed: bool = False


@dataclass
class PaperSection:
    """A section of the research paper."""

    name: str
    content: str
    order: int
    word_count: int = 0


@dataclass
class ResearchPlan:
    """The research plan created by the brain."""

    key_questions: list[str]
    search_queries: list[SearchQuery]
    methodology: str
    expected_structure: list[str]
    domains: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_questions": self.key_questions,
            "search_queries": [
                {
                    "query": q.query,
                    "target": q.target,
                    "priority": q.priority,
                    "rationale": q.rationale,
                }
                for q in self.search_queries
            ],
            "methodology": self.methodology,
            "expected_structure": self.expected_structure,
            "domains": self.domains,
        }


@dataclass
class AnalysisResult:
    """Result of source analysis."""

    synthesis: str
    knowledge_gaps: list[str]
    key_findings: list[str]
    source_evaluations: dict[str, dict[str, Any]]  # source_id -> evaluation


@dataclass
class CostSummary:
    """Cost tracking for the workflow."""

    brain_cost_usd: float = 0.0
    worker_cost_usd: float = 0.0
    search_cost_usd: float = 0.0
    total_tokens: int = 0
    api_calls: int = 0

    @property
    def total_cost_usd(self) -> float:
        """Calculate total cost."""
        return self.brain_cost_usd + self.worker_cost_usd + self.search_cost_usd


# =============================================================================
# LangGraph State Definition
# =============================================================================


class ResearchState(TypedDict, total=False):
    """State schema for the research workflow.

    This TypedDict defines the shape of state that flows through the graph.
    Fields use Annotated with operator.add for list fields to support
    additive updates (new items are appended, not replaced).

    Workflow flow:
    1. PLAN: Brain creates research plan from query
    2. SEARCH: Workers execute search queries (parallel)
    3. ANALYZE: Brain analyzes gathered sources
    4. (conditional) If gaps exist and iterations < max, loop to SEARCH
    5. SYNTHESIZE: Brain synthesizes findings
    6. WRITE: Workers draft paper sections
    7. REVIEW: Brain reviews and refines
    8. OUTPUT: Generate final LaTeX
    """

    # === Input (set once at start) ===
    query: str
    domains: list[str]
    thread_id: str

    # === Planning ===
    research_plan: dict[str, Any] | None
    search_queries: Annotated[list[dict[str, Any]], operator.add]

    # === Search Results ===
    sources: Annotated[list[dict[str, Any]], operator.add]
    arxiv_papers: Annotated[list[dict[str, Any]], operator.add]

    # === Analysis ===
    analysis: dict[str, Any] | None
    knowledge_gaps: list[str]
    key_findings: Annotated[list[str], operator.add]

    # === Synthesis ===
    synthesis: str | None
    paper_outline: list[str]

    # === Writing ===
    sections: dict[str, str]  # section_name -> content
    citations: Annotated[list[dict[str, Any]], operator.add]
    abstract: str | None
    title: str | None

    # === Review ===
    review_feedback: str | None
    quality_score: float | None

    # === Control Flow ===
    status: str
    iteration_count: int
    max_iterations: int
    needs_more_research: bool

    # === Error Handling ===
    errors: Annotated[list[str], operator.add]

    # === Cost Tracking ===
    cost_usd: float
    tokens_used: int

    # === Metadata ===
    started_at: str
    completed_at: str | None
    last_node: str | None


def create_initial_state(
    query: str,
    domains: list[str] | None = None,
    thread_id: str | None = None,
    max_iterations: int = 3,
) -> ResearchState:
    """Create the initial state for a research workflow.

    Args:
        query: The research query/question.
        domains: Research domains to focus on.
        thread_id: Unique thread ID for checkpointing.
        max_iterations: Maximum search iterations.

    Returns:
        Initial ResearchState with all fields initialized.
    """
    import uuid

    if domains is None:
        domains = [ResearchDomain.GENERAL.value]

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    return ResearchState(
        # Input
        query=query,
        domains=domains,
        thread_id=thread_id,
        # Planning
        research_plan=None,
        search_queries=[],
        # Search
        sources=[],
        arxiv_papers=[],
        # Analysis
        analysis=None,
        knowledge_gaps=[],
        key_findings=[],
        # Synthesis
        synthesis=None,
        paper_outline=[],
        # Writing
        sections={},
        citations=[],
        abstract=None,
        title=None,
        # Review
        review_feedback=None,
        quality_score=None,
        # Control
        status=WorkflowStatus.PENDING.value,
        iteration_count=0,
        max_iterations=max_iterations,
        needs_more_research=False,
        # Errors
        errors=[],
        # Cost
        cost_usd=0.0,
        tokens_used=0,
        # Metadata
        started_at=datetime.now().isoformat(),
        completed_at=None,
        last_node=None,
    )


def state_to_summary(state: ResearchState) -> dict[str, Any]:
    """Convert state to a summary for API responses.

    Returns a simplified view of the state suitable for API responses.
    """
    return {
        "thread_id": state.get("thread_id"),
        "query": state.get("query"),
        "status": state.get("status"),
        "iteration_count": state.get("iteration_count"),
        "sources_count": len(state.get("sources", [])),
        "sections_count": len(state.get("sections", {})),
        "has_synthesis": state.get("synthesis") is not None,
        "quality_score": state.get("quality_score"),
        "cost_usd": state.get("cost_usd", 0.0),
        "tokens_used": state.get("tokens_used", 0),
        "errors": state.get("errors", []),
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
    }
