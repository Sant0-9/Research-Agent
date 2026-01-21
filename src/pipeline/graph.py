"""LangGraph StateGraph definition for the research workflow.

Defines the workflow graph with nodes, edges, and conditional routing.

Workflow Structure:
    START -> plan -> search -> analyze -> [conditional]
                                            |
                                            v
                          (needs more research?) ----yes----> generate_queries -> search
                                            |
                                            no
                                            |
                                            v
                                        synthesize -> write -> review -> END
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.pipeline.nodes import (
    analyze_node,
    generate_additional_queries_node,
    plan_node,
    review_node,
    search_node,
    synthesize_node,
    write_node,
)
from src.pipeline.state import ResearchState, WorkflowStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Conditional Edge Functions
# =============================================================================


def should_continue_research(
    state: ResearchState,
) -> Literal["generate_queries", "synthesize", "end"]:
    """Determine if more research is needed after analysis.

    Routes to:
    - generate_queries: If knowledge gaps exist and under max iterations
    - synthesize: If analysis is complete and ready for synthesis
    - end: If workflow has failed

    Args:
        state: Current workflow state.

    Returns:
        Next node name to route to.
    """
    # Check for failure
    if state.get("status") == WorkflowStatus.FAILED.value:
        logger.info("Routing to end due to failure")
        return "end"

    # Check if more research is needed
    needs_more = state.get("needs_more_research", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if needs_more and iteration_count < max_iterations:
        logger.info(
            "Routing to generate_queries",
            iteration=iteration_count,
            max_iterations=max_iterations,
        )
        return "generate_queries"

    logger.info("Routing to synthesize", iteration=iteration_count)
    return "synthesize"


def check_status_after_node(
    state: ResearchState,
) -> Literal["continue", "end"]:
    """Check if workflow should continue or end after any node.

    Args:
        state: Current workflow state.

    Returns:
        "continue" to proceed, "end" to terminate.
    """
    if state.get("status") == WorkflowStatus.FAILED.value:
        return "end"
    return "continue"


def route_after_search(
    state: ResearchState,
) -> Literal["analyze", "end"]:
    """Route after search node.

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    if state.get("status") == WorkflowStatus.FAILED.value:
        return "end"
    return "analyze"


def route_after_write(
    state: ResearchState,
) -> Literal["review", "end"]:
    """Route after write node.

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    if state.get("status") == WorkflowStatus.FAILED.value:
        return "end"
    return "review"


# =============================================================================
# Graph Builder
# =============================================================================


def build_research_graph() -> StateGraph:
    """Build the research workflow StateGraph.

    Creates a graph with the following structure:

    [START] -> plan -> search -> analyze -+-> synthesize -> write -> review -> [END]
                          ^               |
                          |               v (if needs_more_research)
                          +-- generate_queries

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Initialize the graph with our state schema
    graph = StateGraph(ResearchState)

    # ==========================================================================
    # Add Nodes
    # ==========================================================================

    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("generate_queries", generate_additional_queries_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    # ==========================================================================
    # Add Edges
    # ==========================================================================

    # Start -> Plan
    graph.add_edge(START, "plan")

    # Plan -> Search (always)
    graph.add_edge("plan", "search")

    # Search -> Analyze (with failure check)
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "analyze": "analyze",
            "end": END,
        },
    )

    # Analyze -> Conditional routing
    graph.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "generate_queries": "generate_queries",
            "synthesize": "synthesize",
            "end": END,
        },
    )

    # Generate Queries -> Search (loop back)
    graph.add_edge("generate_queries", "search")

    # Synthesize -> Write (always)
    graph.add_edge("synthesize", "write")

    # Write -> Review (with failure check)
    graph.add_conditional_edges(
        "write",
        route_after_write,
        {
            "review": "review",
            "end": END,
        },
    )

    # Review -> End
    graph.add_edge("review", END)

    return graph


def create_research_workflow(checkpointer: Any | None = None) -> Any:
    """Create a compiled research workflow.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                     Use AsyncPostgresSaver for production.

    Returns:
        Compiled graph ready for invocation.
    """
    graph = build_research_graph()

    # Compile with optional checkpointer
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()


# =============================================================================
# Graph Visualization
# =============================================================================


def get_graph_mermaid() -> str:
    """Get a Mermaid diagram representation of the graph.

    Returns:
        Mermaid diagram string for visualization.
    """
    return """
graph TD
    START((Start)) --> plan[Plan Research]
    plan --> search[Execute Searches]
    search --> analyze[Analyze Sources]
    analyze -->|needs more research| generate_queries[Generate Queries]
    generate_queries --> search
    analyze -->|ready| synthesize[Synthesize Findings]
    synthesize --> write[Write Sections]
    write --> review[Review Paper]
    review --> END((End))

    style plan fill:#e1f5fe
    style search fill:#fff3e0
    style analyze fill:#f3e5f5
    style generate_queries fill:#fff3e0
    style synthesize fill:#e8f5e9
    style write fill:#fce4ec
    style review fill:#e0f2f1
"""


def get_graph_ascii() -> str:
    """Get an ASCII representation of the workflow.

    Returns:
        ASCII art showing the workflow structure.
    """
    return """
    Research Workflow Graph
    =======================

    [START]
        |
        v
    +--------+
    |  plan  |  <- Brain creates research plan
    +--------+
        |
        v
    +--------+
    | search |  <- Workers search web + ArXiv
    +--------+
        |
        v
    +---------+
    | analyze |  <- Brain analyzes sources
    +---------+
        |
        +---------------+
        |               |
        v               v
    (needs more?)   +-----------+
        |           | synthesize|  <- Brain synthesizes
        v           +-----------+
    +----------+         |
    | generate |         v
    | queries  |    +-------+
    +----------+    | write |  <- Workers write sections
        |           +-------+
        v               |
    [loop to search]    v
                    +--------+
                    | review |  <- Brain reviews quality
                    +--------+
                        |
                        v
                      [END]
    """
