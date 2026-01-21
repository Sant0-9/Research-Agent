"""Workflow execution interface for the research pipeline.

Provides high-level APIs for:
- Starting research workflows
- Checking workflow status
- Resuming from checkpoints
- Streaming workflow updates

Uses PostgreSQL checkpointing for durability.
"""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from src.config import get_settings
from src.pipeline.graph import create_research_workflow
from src.pipeline.state import (
    ResearchDomain,
    ResearchState,
    WorkflowStatus,
    create_initial_state,
    state_to_summary,
)
from src.utils.errors import WorkflowError, WorkflowTimeoutError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowResult:
    """Result of a completed workflow."""

    thread_id: str
    status: str
    title: str | None
    abstract: str | None
    sections: dict[str, str]
    citations: list[dict[str, Any]]
    quality_score: float | None
    cost_usd: float
    tokens_used: int
    errors: list[str]
    started_at: str
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "status": self.status,
            "title": self.title,
            "abstract": self.abstract,
            "sections": self.sections,
            "citations": self.citations,
            "quality_score": self.quality_score,
            "cost_usd": round(self.cost_usd, 6),
            "tokens_used": self.tokens_used,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class StreamUpdate:
    """A streaming update from the workflow."""

    node: str
    status: str
    message: str
    progress: float  # 0.0 to 1.0
    data: dict[str, Any] | None = None


class ResearchWorkflow:
    """High-level interface for running research workflows.

    Supports both synchronous and streaming execution modes.
    Uses PostgreSQL checkpointing when available for durability.
    """

    # Progress estimates for each node
    NODE_PROGRESS: ClassVar[dict[str, float]] = {
        "plan": 0.1,
        "search": 0.3,
        "analyze": 0.5,
        "generate_queries": 0.4,
        "synthesize": 0.7,
        "write": 0.85,
        "review": 0.95,
    }

    def __init__(self, use_checkpointing: bool = False) -> None:
        """Initialize the workflow interface.

        Args:
            use_checkpointing: Whether to use PostgreSQL checkpointing.
                             Requires database connection.
        """
        self._settings = get_settings()
        self._use_checkpointing = use_checkpointing
        self._checkpointer: Any = None

    async def _get_checkpointer(self) -> Any:
        """Get or create the PostgreSQL checkpointer."""
        if not self._use_checkpointing:
            return None

        if self._checkpointer is None:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                # Build connection string from settings
                # This assumes DATABASE_URL env var or similar
                db_url = self._settings.__dict__.get(
                    "database_url",
                    "postgresql://postgres:postgres@localhost:5432/research_agent",
                )

                self._checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
                # Setup tables if needed (idempotent)
                await self._checkpointer.setup()

            except ImportError:
                logger.warning("langgraph-checkpoint-postgres not installed, disabling checkpointing")
                self._use_checkpointing = False
                return None
            except Exception as e:
                logger.warning("Failed to initialize checkpointer", error=str(e))
                self._use_checkpointing = False
                return None

        return self._checkpointer

    async def run(
        self,
        query: str,
        domains: list[str] | None = None,
        max_iterations: int = 3,
        timeout_seconds: int = 600,
        thread_id: str | None = None,
    ) -> WorkflowResult:
        """Run a complete research workflow.

        Args:
            query: The research question/topic.
            domains: Research domains to focus on.
            max_iterations: Maximum search iterations.
            timeout_seconds: Timeout for the entire workflow.
            thread_id: Optional thread ID for tracking.

        Returns:
            WorkflowResult with the completed research.

        Raises:
            WorkflowError: If the workflow fails.
            WorkflowTimeoutError: If the workflow times out.
        """
        logger.info("Starting research workflow", query=query, domains=domains)

        # Validate domains
        if domains:
            valid_domains = [d.value for d in ResearchDomain]
            domains = [d for d in domains if d in valid_domains]
        if not domains:
            domains = [ResearchDomain.GENERAL.value]

        # Generate thread ID
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Create initial state
        initial_state = create_initial_state(
            query=query,
            domains=domains,
            thread_id=thread_id,
            max_iterations=max_iterations,
        )

        try:
            # Get checkpointer if available
            checkpointer = await self._get_checkpointer()

            # Create and compile the workflow
            workflow = create_research_workflow(checkpointer)

            # Run with timeout
            config = {"configurable": {"thread_id": thread_id}}

            async def run_workflow() -> ResearchState:
                result = await workflow.ainvoke(initial_state, config)
                return result  # type: ignore[no-any-return]

            final_state = await asyncio.wait_for(
                run_workflow(),
                timeout=timeout_seconds,
            )

            return self._state_to_result(final_state)

        except TimeoutError as e:
            logger.error("Workflow timed out", thread_id=thread_id, timeout=timeout_seconds)
            raise WorkflowTimeoutError(
                f"Research workflow timed out after {timeout_seconds} seconds",
                details={"thread_id": thread_id, "query": query},
            ) from e
        except Exception as e:
            logger.error("Workflow failed", thread_id=thread_id, error=str(e))
            raise WorkflowError(
                f"Research workflow failed: {e}",
                details={"thread_id": thread_id, "query": query},
            ) from e

    async def stream(
        self,
        query: str,
        domains: list[str] | None = None,
        max_iterations: int = 3,
        thread_id: str | None = None,
    ) -> AsyncIterator[StreamUpdate]:
        """Stream workflow updates as they occur.

        Yields updates for each node transition, allowing real-time
        progress tracking.

        Args:
            query: The research question/topic.
            domains: Research domains to focus on.
            max_iterations: Maximum search iterations.
            thread_id: Optional thread ID for tracking.

        Yields:
            StreamUpdate objects with progress information.

        Raises:
            WorkflowError: If the workflow fails.
        """
        logger.info("Starting streaming research workflow", query=query)

        # Validate domains
        if domains:
            valid_domains = [d.value for d in ResearchDomain]
            domains = [d for d in domains if d in valid_domains]
        if not domains:
            domains = [ResearchDomain.GENERAL.value]

        # Generate thread ID
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Create initial state
        initial_state = create_initial_state(
            query=query,
            domains=domains,
            thread_id=thread_id,
            max_iterations=max_iterations,
        )

        yield StreamUpdate(
            node="start",
            status="started",
            message="Research workflow started",
            progress=0.0,
            data={"thread_id": thread_id, "query": query},
        )

        try:
            # Get checkpointer if available
            checkpointer = await self._get_checkpointer()

            # Create and compile the workflow
            workflow = create_research_workflow(checkpointer)

            config = {"configurable": {"thread_id": thread_id}}

            # Stream node updates
            async for event in workflow.astream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    progress = self.NODE_PROGRESS.get(node_name, 0.5)
                    status = node_output.get("status", "running")

                    yield StreamUpdate(
                        node=node_name,
                        status=status,
                        message=self._get_node_message(node_name, node_output),
                        progress=progress,
                        data={
                            "last_node": node_name,
                            "cost_usd": node_output.get("cost_usd", 0.0),
                            "tokens_used": node_output.get("tokens_used", 0),
                        },
                    )

            # Final update
            yield StreamUpdate(
                node="end",
                status="completed",
                message="Research workflow completed",
                progress=1.0,
                data={"thread_id": thread_id},
            )

        except Exception as e:
            logger.error("Streaming workflow failed", thread_id=thread_id, error=str(e))
            yield StreamUpdate(
                node="error",
                status="failed",
                message=f"Workflow failed: {e}",
                progress=0.0,
                data={"error": str(e)},
            )
            raise WorkflowError(
                f"Research workflow failed: {e}",
                details={"thread_id": thread_id},
            ) from e

    async def get_status(self, thread_id: str) -> dict[str, Any] | None:
        """Get the status of a workflow by thread ID.

        Requires checkpointing to be enabled.

        Args:
            thread_id: The workflow thread ID.

        Returns:
            Status dictionary or None if not found.
        """
        if not self._use_checkpointing:
            logger.warning("Cannot get status without checkpointing enabled")
            return None

        try:
            checkpointer = await self._get_checkpointer()
            if checkpointer is None:
                return None

            config = {"configurable": {"thread_id": thread_id}}
            state = await checkpointer.aget(config)

            if state is None:
                return None

            return state_to_summary(state)

        except Exception as e:
            logger.error("Failed to get workflow status", thread_id=thread_id, error=str(e))
            return None

    async def resume(
        self,
        thread_id: str,
        timeout_seconds: int = 600,
    ) -> WorkflowResult:
        """Resume a workflow from a checkpoint.

        Requires checkpointing to be enabled.

        Args:
            thread_id: The workflow thread ID to resume.
            timeout_seconds: Timeout for the resumed workflow.

        Returns:
            WorkflowResult with the completed research.

        Raises:
            WorkflowError: If resume fails or thread not found.
        """
        if not self._use_checkpointing:
            raise WorkflowError("Cannot resume without checkpointing enabled")

        logger.info("Resuming workflow", thread_id=thread_id)

        try:
            checkpointer = await self._get_checkpointer()
            if checkpointer is None:
                raise WorkflowError("Checkpointer not available")

            # Get current state
            config = {"configurable": {"thread_id": thread_id}}
            current_state = await checkpointer.aget(config)

            if current_state is None:
                raise WorkflowError(f"No checkpoint found for thread {thread_id}")

            # Create workflow and continue
            workflow = create_research_workflow(checkpointer)

            async def run_workflow() -> ResearchState:
                result = await workflow.ainvoke(None, config)
                return result  # type: ignore[no-any-return]

            final_state = await asyncio.wait_for(
                run_workflow(),
                timeout=timeout_seconds,
            )

            return self._state_to_result(final_state)

        except TimeoutError as e:
            raise WorkflowTimeoutError(
                f"Resumed workflow timed out after {timeout_seconds} seconds",
                details={"thread_id": thread_id},
            ) from e
        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Failed to resume workflow: {e}",
                details={"thread_id": thread_id},
            ) from e

    def _state_to_result(self, state: ResearchState) -> WorkflowResult:
        """Convert final state to WorkflowResult."""
        return WorkflowResult(
            thread_id=state.get("thread_id", ""),
            status=state.get("status", WorkflowStatus.COMPLETED.value),
            title=state.get("title"),
            abstract=state.get("abstract"),
            sections=state.get("sections", {}),
            citations=state.get("citations", []),
            quality_score=state.get("quality_score"),
            cost_usd=state.get("cost_usd", 0.0),
            tokens_used=state.get("tokens_used", 0),
            errors=state.get("errors", []),
            started_at=state.get("started_at", datetime.now().isoformat()),
            completed_at=state.get("completed_at"),
        )

    def _get_node_message(self, node_name: str, output: dict[str, Any]) -> str:
        """Generate a human-readable message for a node update."""
        messages = {
            "plan": "Creating research plan...",
            "search": f"Searching sources (iteration {output.get('iteration_count', 1)})...",
            "analyze": "Analyzing gathered sources...",
            "generate_queries": "Generating additional search queries...",
            "synthesize": "Synthesizing research findings...",
            "write": "Writing paper sections...",
            "review": "Reviewing paper quality...",
        }
        return messages.get(node_name, f"Processing {node_name}...")


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_research(
    query: str,
    domains: list[str] | None = None,
    max_iterations: int = 3,
    timeout_seconds: int = 600,
) -> WorkflowResult:
    """Run a research workflow (convenience function).

    Creates a temporary workflow instance for one-off executions.

    Args:
        query: The research question/topic.
        domains: Research domains to focus on.
        max_iterations: Maximum search iterations.
        timeout_seconds: Timeout for the workflow.

    Returns:
        WorkflowResult with the completed research.
    """
    workflow = ResearchWorkflow(use_checkpointing=False)
    return await workflow.run(
        query=query,
        domains=domains,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )


@asynccontextmanager
async def research_workflow_context(
    use_checkpointing: bool = True,
) -> AsyncIterator[ResearchWorkflow]:
    """Context manager for research workflow with cleanup.

    Args:
        use_checkpointing: Whether to use PostgreSQL checkpointing.

    Yields:
        ResearchWorkflow instance.
    """
    workflow = ResearchWorkflow(use_checkpointing=use_checkpointing)
    try:
        yield workflow
    finally:
        # Cleanup if needed
        if workflow._checkpointer is not None:
            with contextlib.suppress(Exception):
                await workflow._checkpointer.close()
