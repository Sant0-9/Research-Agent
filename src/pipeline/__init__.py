"""Research pipeline package.

Provides the LangGraph-based research workflow orchestration.

Main components:
- ResearchState: State schema for the workflow
- build_research_graph: Creates the StateGraph
- ResearchWorkflow: High-level workflow execution interface
- run_research: Convenience function for one-off research

Example usage:
    from src.pipeline import run_research, ResearchWorkflow

    # One-off research
    result = await run_research("What is quantum entanglement?")

    # With workflow instance for more control
    workflow = ResearchWorkflow(use_checkpointing=True)
    result = await workflow.run("Effects of LLM scaling laws")
    async for update in workflow.stream("Latest advances in AI"):
        print(f"{update.node}: {update.message}")
"""

from src.pipeline.graph import (
    build_research_graph,
    create_research_workflow,
    get_graph_ascii,
    get_graph_mermaid,
)
from src.pipeline.state import (
    AnalysisResult,
    CostSummary,
    PaperSection,
    ResearchDomain,
    ResearchPlan,
    ResearchState,
    SearchQuery,
    Source,
    WorkflowStatus,
    create_initial_state,
    state_to_summary,
)
from src.pipeline.workflow import (
    ResearchWorkflow,
    StreamUpdate,
    WorkflowResult,
    research_workflow_context,
    run_research,
)

__all__ = [
    "AnalysisResult",
    "CostSummary",
    "PaperSection",
    "ResearchDomain",
    "ResearchPlan",
    "ResearchState",
    "ResearchWorkflow",
    "SearchQuery",
    "Source",
    "StreamUpdate",
    "WorkflowResult",
    "WorkflowStatus",
    "build_research_graph",
    "create_initial_state",
    "create_research_workflow",
    "get_graph_ascii",
    "get_graph_mermaid",
    "research_workflow_context",
    "run_research",
    "state_to_summary",
]
