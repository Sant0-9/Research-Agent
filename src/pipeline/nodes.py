"""Node functions for the research workflow graph.

Each node receives the current state, performs its operation,
and returns a dictionary of state updates.

Nodes:
- plan_node: Creates research plan using Brain
- search_node: Executes searches using Workers (parallel)
- analyze_node: Analyzes sources using Brain
- synthesize_node: Synthesizes findings using Brain
- write_node: Drafts paper sections using Workers
- review_node: Reviews paper using Brain
"""

import asyncio
import re
from datetime import datetime
from typing import Any

from src.brain.client import BrainClient
from src.brain.prompts import (
    ANALYZE_SOURCES,
    GENERATE_SEARCH_QUERIES,
    RESEARCH_PLAN,
    REVIEW_PAPER,
    SYNTHESIZE_FINDINGS,
)
from src.config import get_settings
from src.pipeline.state import ResearchState, WorkflowStatus
from src.utils.logging import get_logger
from src.workers.arxiv import ArxivWorker
from src.workers.search import SearchWorker
from src.workers.writer import WriterWorker

logger = get_logger(__name__)


# =============================================================================
# Plan Node
# =============================================================================


async def plan_node(state: ResearchState) -> dict[str, Any]:
    """Create a research plan from the query.

    Uses the Brain to analyze the query and generate:
    - Key questions to answer
    - Search queries to execute
    - Research methodology
    - Expected paper structure

    Args:
        state: Current workflow state.

    Returns:
        State updates with research_plan and search_queries.
    """
    logger.info("Planning research", query=state["query"])

    try:
        brain = BrainClient()

        # Format the planning prompt
        prompt = RESEARCH_PLAN.format(
            query=state["query"],
            domains=", ".join(state.get("domains", ["general"])),
        )

        # Generate research plan
        response = await brain.generate(
            prompt,
            max_tokens=4096,
            force_thinking=True,
        )

        await brain.close()

        # Parse the response into structured plan
        plan = _parse_research_plan(response.content, state.get("domains", []))

        # Convert search queries to state format
        search_queries = [
            {
                "query": q.query if hasattr(q, "query") else q["query"],
                "target": q.target if hasattr(q, "target") else q["target"],
                "priority": q.priority if hasattr(q, "priority") else q["priority"],
                "rationale": q.rationale if hasattr(q, "rationale") else q.get("rationale", ""),
                "executed": False,
            }
            for q in plan.get("search_queries", [])
        ]

        return {
            "research_plan": plan,
            "search_queries": search_queries,
            "status": WorkflowStatus.SEARCHING.value,
            "cost_usd": state.get("cost_usd", 0.0) + _estimate_brain_cost(response.tokens_used),
            "tokens_used": state.get("tokens_used", 0) + response.tokens_used,
            "last_node": "plan",
        }

    except Exception as e:
        logger.error("Planning failed", error=str(e))
        return {
            "status": WorkflowStatus.FAILED.value,
            "errors": [f"Planning failed: {e}"],
            "last_node": "plan",
        }


def _parse_research_plan(content: str, domains: list[str]) -> dict[str, Any]:
    """Parse brain response into structured research plan.

    Extracts key sections from the brain's response.
    Uses robust parsing that handles various output formats.
    """
    # Remove thinking tags if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Default structure
    plan: dict[str, Any] = {
        "key_questions": [],
        "search_queries": [],
        "methodology": "",
        "expected_structure": [
            "Abstract",
            "Introduction",
            "Background",
            "Methods",
            "Results",
            "Discussion",
            "Conclusion",
        ],
        "domains": domains,
        "raw_response": content,
    }

    # Extract key questions
    questions_match = re.search(
        r"(?:KEY QUESTIONS|QUESTIONS)[\s:]*\n((?:[-*\d.]+[^\n]+\n?)+)",
        content,
        re.IGNORECASE,
    )
    if questions_match:
        questions = re.findall(r"[-*\d.]+\s*(.+)", questions_match.group(1))
        plan["key_questions"] = [q.strip() for q in questions if q.strip()]

    # Extract search queries
    search_match = re.search(
        r"(?:SEARCH STRATEGY|SEARCH QUERIES)[\s:]*\n((?:[-*\d.]+[^\n]+\n?)+)",
        content,
        re.IGNORECASE,
    )
    if search_match:
        queries = re.findall(r"[-*\d.]+\s*(.+)", search_match.group(1))
        for q in queries:
            q = q.strip()
            if q:
                # Determine target based on query content
                target = "both"
                if "arxiv" in q.lower():
                    target = "arxiv"
                elif "web" in q.lower() or "documentation" in q.lower():
                    target = "web"

                plan["search_queries"].append({
                    "query": q,
                    "target": target,
                    "priority": "high",
                    "rationale": "",
                })

    # Extract methodology
    method_match = re.search(
        r"(?:METHODOLOGY|APPROACH)[\s:]*\n(.+?)(?=\n[A-Z]{2,}|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if method_match:
        plan["methodology"] = method_match.group(1).strip()

    # Extract expected structure
    structure_match = re.search(
        r"(?:EXPECTED STRUCTURE|PAPER STRUCTURE|SECTIONS)[\s:]*\n((?:[-*\d.]+[^\n]+\n?)+)",
        content,
        re.IGNORECASE,
    )
    if structure_match:
        sections = re.findall(r"[-*\d.]+\s*(.+)", structure_match.group(1))
        if sections:
            plan["expected_structure"] = [s.strip() for s in sections if s.strip()]

    # If no search queries were extracted, generate default ones
    if not plan["search_queries"]:
        # Generate generic queries based on the original research query
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            # Check if it looks like a search query
            is_valid_line = len(line) > 10 and not line.startswith("#")
            has_search_keyword = any(
                kw in line.lower()
                for kw in ["search", "find", "look for", "query"]
            )
            if is_valid_line and has_search_keyword:
                plan["search_queries"].append({
                    "query": line[:200],  # Limit length
                    "target": "both",
                    "priority": "medium",
                    "rationale": "Extracted from plan",
                })

    return plan


# =============================================================================
# Search Node
# =============================================================================


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Execute search queries using Workers.

    Runs web searches and ArXiv searches in parallel.
    Aggregates results and tracks costs.

    Args:
        state: Current workflow state.

    Returns:
        State updates with sources and arxiv_papers.
    """
    logger.info(
        "Executing searches",
        query_count=len(state.get("search_queries", [])),
        iteration=state.get("iteration_count", 0),
    )

    settings = get_settings()
    search_worker = SearchWorker()
    arxiv_worker = ArxivWorker()

    sources: list[dict[str, Any]] = []
    arxiv_papers: list[dict[str, Any]] = []
    total_cost = 0.0
    errors: list[str] = []

    # Get pending search queries
    queries = [
        q for q in state.get("search_queries", [])
        if not q.get("executed", False)
    ]

    # Limit parallel searches
    max_parallel = min(settings.max_search_workers, len(queries))

    try:
        # Separate queries by target
        web_queries = [q for q in queries if q.get("target") in ("web", "both")]
        arxiv_queries = [q for q in queries if q.get("target") in ("arxiv", "both")]

        # Execute web searches
        if web_queries:
            web_tasks = []
            for q in web_queries[:max_parallel]:
                web_tasks.append(_execute_web_search(search_worker, q["query"]))

            web_results = await asyncio.gather(*web_tasks, return_exceptions=True)

            for i, result in enumerate(web_results):
                if isinstance(result, Exception):
                    errors.append(f"Web search failed: {result}")
                    logger.warning("Web search failed", query=web_queries[i]["query"], error=str(result))
                elif isinstance(result, dict):
                    sources.extend(result.get("sources", []))
                    total_cost += result.get("cost", 0.0)

        # Execute ArXiv searches
        if arxiv_queries:
            arxiv_tasks = []
            for q in arxiv_queries[:max_parallel]:
                arxiv_tasks.append(_execute_arxiv_search(arxiv_worker, q["query"]))

            arxiv_results = await asyncio.gather(*arxiv_tasks, return_exceptions=True)

            for i, result in enumerate(arxiv_results):
                if isinstance(result, Exception):
                    errors.append(f"ArXiv search failed: {result}")
                    logger.warning("ArXiv search failed", query=arxiv_queries[i]["query"], error=str(result))
                elif isinstance(result, dict):
                    arxiv_papers.extend(result.get("papers", []))

        # Mark queries as executed
        updated_queries = []
        for q in state.get("search_queries", []):
            q_copy = q.copy()
            q_copy["executed"] = True
            updated_queries.append(q_copy)

        return {
            "sources": sources,
            "arxiv_papers": arxiv_papers,
            "search_queries": updated_queries,
            "status": WorkflowStatus.ANALYZING.value,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "cost_usd": state.get("cost_usd", 0.0) + total_cost,
            "errors": errors if errors else [],
            "last_node": "search",
        }

    except Exception as e:
        logger.error("Search node failed", error=str(e))
        return {
            "status": WorkflowStatus.FAILED.value,
            "errors": [f"Search failed: {e}"],
            "last_node": "search",
        }


async def _execute_web_search(
    worker: SearchWorker,
    query: str,
) -> dict[str, Any]:
    """Execute a single web search."""
    try:
        response = await worker.search(
            query,
            max_results=5,
            search_depth="advanced",
            min_score=0.5,
        )

        sources = []
        for result in response.results:
            sources.append({
                "id": f"web_{hash(result.url) % 10000}",
                "title": result.title,
                "url": result.url,
                "content": result.content,
                "source_type": "web",
                "relevance_score": result.score,
            })

        return {
            "sources": sources,
            "cost": SearchWorker.COST_PER_SEARCH,
        }

    except Exception as e:
        raise RuntimeError(f"Web search failed for '{query}': {e}") from e


async def _execute_arxiv_search(
    worker: ArxivWorker,
    query: str,
) -> dict[str, Any]:
    """Execute a single ArXiv search."""
    try:
        response = await worker.search(
            query,
            max_results=5,
            sort_by="relevance",
        )

        papers = []
        for paper in response.papers:
            papers.append({
                "id": f"arxiv_{paper.arxiv_id}",
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "url": paper.entry_url,
                "content": paper.abstract,
                "source_type": "arxiv",
                "authors": paper.authors,
                "published_date": paper.published.isoformat(),
                "categories": paper.categories,
                "bibtex": paper.to_bibtex(),
            })

        return {
            "papers": papers,
            "cost": 0.0,  # ArXiv is free
        }

    except Exception as e:
        raise RuntimeError(f"ArXiv search failed for '{query}': {e}") from e


# =============================================================================
# Analyze Node
# =============================================================================


async def analyze_node(state: ResearchState) -> dict[str, Any]:
    """Analyze gathered sources using Brain.

    Evaluates source quality, extracts key findings,
    identifies knowledge gaps.

    Args:
        state: Current workflow state.

    Returns:
        State updates with analysis, key_findings, knowledge_gaps.
    """
    logger.info(
        "Analyzing sources",
        source_count=len(state.get("sources", [])),
        arxiv_count=len(state.get("arxiv_papers", [])),
    )

    # Combine all sources
    all_sources = state.get("sources", []) + state.get("arxiv_papers", [])

    if not all_sources:
        logger.warning("No sources to analyze")
        return {
            "analysis": {"synthesis": "", "evaluations": {}},
            "knowledge_gaps": ["No sources found - need to search again"],
            "needs_more_research": True,
            "status": WorkflowStatus.SEARCHING.value,
            "last_node": "analyze",
        }

    try:
        brain = BrainClient()

        # Format sources for analysis
        sources_text = _format_sources_for_analysis(all_sources[:20])  # Limit to prevent context overflow

        prompt = ANALYZE_SOURCES.format(
            topic=state["query"],
            sources=sources_text,
        )

        response = await brain.generate(
            prompt,
            max_tokens=8192,
            force_thinking=True,
        )

        await brain.close()

        # Parse analysis results
        analysis = _parse_analysis_response(response.content)

        # Determine if more research is needed
        has_significant_gaps = len(analysis.get("gaps", [])) > 2
        below_max_iterations = state.get("iteration_count", 0) < state.get("max_iterations", 3)
        needs_more = has_significant_gaps and below_max_iterations

        next_status = (
            WorkflowStatus.SEARCHING.value
            if needs_more
            else WorkflowStatus.SYNTHESIZING.value
        )

        return {
            "analysis": analysis,
            "key_findings": analysis.get("key_findings", []),
            "knowledge_gaps": analysis.get("gaps", []),
            "needs_more_research": needs_more,
            "status": next_status,
            "cost_usd": state.get("cost_usd", 0.0) + _estimate_brain_cost(response.tokens_used),
            "tokens_used": state.get("tokens_used", 0) + response.tokens_used,
            "last_node": "analyze",
        }

    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        return {
            "status": WorkflowStatus.FAILED.value,
            "errors": [f"Analysis failed: {e}"],
            "last_node": "analyze",
        }


def _format_sources_for_analysis(sources: list[dict[str, Any]]) -> str:
    """Format sources for the analysis prompt."""
    parts = []
    for i, source in enumerate(sources, 1):
        content = source.get("content", "")[:2000]  # Limit content length
        parts.append(f"""
Source {i}:
Title: {source.get('title', 'Unknown')}
Type: {source.get('source_type', 'unknown')}
URL: {source.get('url', '')}

Content:
{content}
""")
    return "\n---\n".join(parts)


def _parse_analysis_response(content: str) -> dict[str, Any]:
    """Parse brain analysis response into structured data."""
    # Remove thinking tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    analysis: dict[str, Any] = {
        "synthesis": "",
        "gaps": [],
        "conflicts": [],
        "key_findings": [],
        "evaluations": {},
    }

    # Extract synthesis
    synthesis_match = re.search(
        r"(?:SYNTHESIS|SUMMARY)[\s:]*\n(.+?)(?=\n[A-Z]{2,}|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if synthesis_match:
        analysis["synthesis"] = synthesis_match.group(1).strip()

    # Extract gaps
    gaps_match = re.search(
        r"(?:GAPS|MISSING|NOT COVERED)[\s:]*\n((?:[-*\d.]+[^\n]+\n?)+)",
        content,
        re.IGNORECASE,
    )
    if gaps_match:
        gaps = re.findall(r"[-*\d.]+\s*(.+)", gaps_match.group(1))
        analysis["gaps"] = [g.strip() for g in gaps if g.strip()]

    # Extract key findings
    findings_match = re.search(
        r"(?:KEY INSIGHTS|KEY TAKEAWAYS|MAIN TAKEAWAYS|FINDINGS)[\s:]*\n((?:[-*\d.]+[^\n]+\n?)+)",
        content,
        re.IGNORECASE,
    )
    if findings_match:
        findings = re.findall(r"[-*\d.]+\s*(.+)", findings_match.group(1))
        analysis["key_findings"] = [f.strip() for f in findings if f.strip()]

    # Extract conflicts
    conflicts_match = re.search(
        r"(?:CONFLICTS|CONTRADICTORY)[\s:]*\n(.+?)(?=\n[A-Z]{2,}|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if conflicts_match:
        analysis["conflicts"] = conflicts_match.group(1).strip()

    return analysis


# =============================================================================
# Synthesize Node
# =============================================================================


async def synthesize_node(state: ResearchState) -> dict[str, Any]:
    """Synthesize findings into a coherent narrative.

    Uses Brain to combine all analyzed information
    into a structured research narrative.

    Args:
        state: Current workflow state.

    Returns:
        State updates with synthesis and paper_outline.
    """
    logger.info("Synthesizing findings")

    try:
        brain = BrainClient()

        # Format analyzed sources
        sources_text = _format_sources_for_analysis(
            state.get("sources", [])[:10] + state.get("arxiv_papers", [])[:10]
        )

        # Format key findings
        findings_text = "\n".join(
            f"- {f}" for f in state.get("key_findings", [])
        )

        prompt = SYNTHESIZE_FINDINGS.format(
            query=state["query"],
            analyzed_sources=sources_text,
            key_findings=findings_text or "See source analysis above.",
        )

        response = await brain.generate(
            prompt,
            max_tokens=8192,
            force_thinking=True,
        )

        await brain.close()

        # Extract synthesis content (remove thinking)
        synthesis = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)

        # Generate paper outline from plan or default
        plan = state.get("research_plan", {})
        paper_outline = plan.get("expected_structure", []) if plan else [
            "Abstract",
            "Introduction",
            "Background",
            "Methodology",
            "Results",
            "Discussion",
            "Conclusion",
        ]

        # Generate title if not set
        title = _generate_title(state["query"], state.get("key_findings", []))

        return {
            "synthesis": synthesis.strip(),
            "paper_outline": paper_outline,
            "title": title,
            "status": WorkflowStatus.WRITING.value,
            "cost_usd": state.get("cost_usd", 0.0) + _estimate_brain_cost(response.tokens_used),
            "tokens_used": state.get("tokens_used", 0) + response.tokens_used,
            "last_node": "synthesize",
        }

    except Exception as e:
        logger.error("Synthesis failed", error=str(e))
        return {
            "status": WorkflowStatus.FAILED.value,
            "errors": [f"Synthesis failed: {e}"],
            "last_node": "synthesize",
        }


def _generate_title(query: str, _key_findings: list[str]) -> str:
    """Generate a paper title from the query and findings."""
    # Clean up the query to make it title-like
    title = query.strip()

    # Remove question marks and make it statement-like
    if title.endswith("?"):
        title = title[:-1]

    # Capitalize appropriately
    words = title.split()
    minor_words = {"a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "in", "of"}

    title_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in minor_words:
            title_words.append(word.capitalize())
        else:
            title_words.append(word.lower())

    return " ".join(title_words)


# =============================================================================
# Write Node
# =============================================================================


async def write_node(state: ResearchState) -> dict[str, Any]:
    """Draft paper sections using Workers.

    Uses WriterWorker to generate LaTeX content for each section.

    Args:
        state: Current workflow state.

    Returns:
        State updates with sections and citations.
    """
    logger.info("Writing paper sections", outline=state.get("paper_outline", []))

    writer = WriterWorker()
    sections: dict[str, str] = {}
    citations: list[dict[str, Any]] = []
    total_cost = 0.0
    total_tokens = 0

    try:
        # Format sources for citation
        sources_text = _format_sources_for_writing(
            state.get("sources", []) + state.get("arxiv_papers", [])
        )

        # Write each section
        outline = state.get("paper_outline", [])

        for section_name in outline:
            if section_name.lower() == "abstract":
                # Abstract handled separately
                continue

            try:
                synthesis_text = state.get("synthesis") or ""
                result = await writer.write_section(
                    topic=state["query"],
                    section_name=section_name,
                    outline=synthesis_text[:3000],
                    sources=sources_text,
                    max_tokens=2000,
                )

                sections[section_name] = result.content
                total_cost += writer.calculate_cost(
                    result.metadata.get("input_tokens", 0),
                    result.metadata.get("output_tokens", 0),
                )
                total_tokens += result.tokens_used

            except Exception as e:
                logger.warning(f"Failed to write section {section_name}", error=str(e))
                sections[section_name] = f"[Section generation failed: {e}]"

        # Write abstract
        try:
            title_text = state.get("title") or "Research Paper"
            synthesis_summary = state.get("synthesis") or ""
            abstract_result = await writer.write_abstract(
                title=title_text,
                content_summary=synthesis_summary[:2000],
            )
            abstract = abstract_result.content
            total_cost += writer.calculate_cost(
                abstract_result.metadata.get("input_tokens", 0),
                abstract_result.metadata.get("output_tokens", 0),
            )
            total_tokens += abstract_result.tokens_used
        except Exception as e:
            logger.warning("Failed to write abstract", error=str(e))
            abstract = None

        # Collect citations from ArXiv papers
        for paper in state.get("arxiv_papers", []):
            if paper.get("bibtex"):
                citations.append({
                    "id": paper.get("id"),
                    "bibtex": paper.get("bibtex"),
                    "title": paper.get("title"),
                })

        await writer.close()

        return {
            "sections": sections,
            "citations": citations,
            "abstract": abstract,
            "status": WorkflowStatus.REVIEWING.value,
            "cost_usd": state.get("cost_usd", 0.0) + total_cost,
            "tokens_used": state.get("tokens_used", 0) + total_tokens,
            "last_node": "write",
        }

    except Exception as e:
        logger.error("Writing failed", error=str(e))
        return {
            "status": WorkflowStatus.FAILED.value,
            "errors": [f"Writing failed: {e}"],
            "last_node": "write",
        }


def _format_sources_for_writing(sources: list[dict[str, Any]]) -> str:
    """Format sources for the writing prompt with citation info."""
    parts = []
    for source in sources[:15]:
        authors = source.get("authors", ["Unknown"])
        if isinstance(authors, list):
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
        else:
            author_str = str(authors)

        parts.append(f"""
[{source.get('id', 'unknown')}]
Title: {source.get('title', 'Unknown')}
Authors: {author_str}
Key content: {source.get('content', '')[:500]}
""")

    return "\n---\n".join(parts)


# =============================================================================
# Review Node
# =============================================================================


async def review_node(state: ResearchState) -> dict[str, Any]:
    """Review the generated paper for quality.

    Uses Brain to review the paper for:
    - Factual accuracy
    - Logical coherence
    - Citation correctness
    - Writing quality

    Args:
        state: Current workflow state.

    Returns:
        State updates with review_feedback and quality_score.
    """
    logger.info("Reviewing paper")

    try:
        brain = BrainClient()

        # Assemble paper content
        paper_content = _assemble_paper_content(state)

        # Format sources
        sources_text = _format_sources_for_analysis(
            state.get("sources", [])[:10] + state.get("arxiv_papers", [])[:10]
        )

        prompt = REVIEW_PAPER.format(
            paper_content=paper_content,
            sources=sources_text,
        )

        response = await brain.generate(
            prompt,
            max_tokens=4096,
            force_thinking=True,
        )

        await brain.close()

        # Parse review response
        review_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)

        # Extract quality score
        score_match = re.search(r"(?:quality|score|rating)[\s:]*(\d+(?:\.\d+)?)", review_content, re.IGNORECASE)
        quality_score = float(score_match.group(1)) if score_match else 3.0

        return {
            "review_feedback": review_content.strip(),
            "quality_score": min(quality_score, 5.0),  # Cap at 5
            "status": WorkflowStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "cost_usd": state.get("cost_usd", 0.0) + _estimate_brain_cost(response.tokens_used),
            "tokens_used": state.get("tokens_used", 0) + response.tokens_used,
            "last_node": "review",
        }

    except Exception as e:
        logger.error("Review failed", error=str(e))
        # Don't fail the whole workflow for review failure
        return {
            "review_feedback": f"Review could not be completed: {e}",
            "quality_score": None,
            "status": WorkflowStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "last_node": "review",
        }


def _assemble_paper_content(state: ResearchState) -> str:
    """Assemble all sections into paper content."""
    parts = []

    if state.get("title"):
        parts.append(f"TITLE: {state['title']}")

    if state.get("abstract"):
        parts.append(f"\nABSTRACT:\n{state['abstract']}")

    for section_name, content in state.get("sections", {}).items():
        parts.append(f"\n{section_name.upper()}:\n{content}")

    return "\n\n".join(parts)


# =============================================================================
# Additional Search Queries Node
# =============================================================================


async def generate_additional_queries_node(state: ResearchState) -> dict[str, Any]:
    """Generate additional search queries based on knowledge gaps.

    Called when analyze_node identifies gaps and more iterations are needed.

    Args:
        state: Current workflow state.

    Returns:
        State updates with new search_queries.
    """
    logger.info("Generating additional search queries", gaps=state.get("knowledge_gaps", []))

    gaps = state.get("knowledge_gaps", [])
    if not gaps:
        return {
            "status": WorkflowStatus.SYNTHESIZING.value,
            "needs_more_research": False,
            "last_node": "generate_queries",
        }

    try:
        brain = BrainClient()

        gaps_text = "\n".join(f"- {gap}" for gap in gaps)

        prompt = GENERATE_SEARCH_QUERIES.format(
            topic=state["query"],
            domain=", ".join(state.get("domains", ["general"])),
            gaps=gaps_text,
        )

        response = await brain.generate(
            prompt,
            max_tokens=2048,
        )

        await brain.close()

        # Parse new queries
        new_queries = _parse_search_queries(response.content)

        return {
            "search_queries": new_queries,
            "status": WorkflowStatus.SEARCHING.value,
            "cost_usd": state.get("cost_usd", 0.0) + _estimate_brain_cost(response.tokens_used),
            "tokens_used": state.get("tokens_used", 0) + response.tokens_used,
            "last_node": "generate_queries",
        }

    except Exception as e:
        logger.error("Query generation failed", error=str(e))
        # Continue to synthesis even if query generation fails
        return {
            "status": WorkflowStatus.SYNTHESIZING.value,
            "needs_more_research": False,
            "errors": [f"Query generation failed: {e}"],
            "last_node": "generate_queries",
        }


def _parse_search_queries(content: str) -> list[dict[str, Any]]:
    """Parse search queries from brain response."""
    queries = []

    # Remove thinking tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Look for QUERY: lines
    query_matches = re.findall(
        r"QUERY[\s:]*([^\n]+)",
        content,
        re.IGNORECASE,
    )

    for query_text in query_matches:
        query_text = query_text.strip()
        if query_text:
            # Determine target
            target = "both"
            target_match = re.search(r"TARGET[\s:]*(\w+)", content, re.IGNORECASE)
            if target_match:
                t = target_match.group(1).lower()
                if t in ("web", "arxiv", "both"):
                    target = t

            queries.append({
                "query": query_text[:200],
                "target": target,
                "priority": "medium",
                "rationale": "Generated to fill knowledge gap",
                "executed": False,
            })

    # Fallback: look for bullet points
    if not queries:
        bullets = re.findall(r"[-*\d.]+\s*(.+)", content)
        for bullet in bullets[:5]:
            if len(bullet) > 10:
                queries.append({
                    "query": bullet.strip()[:200],
                    "target": "both",
                    "priority": "medium",
                    "rationale": "Generated to fill knowledge gap",
                    "executed": False,
                })

    return queries


# =============================================================================
# Utility Functions
# =============================================================================


def _estimate_brain_cost(tokens: int) -> float:
    """Estimate cost for brain (vLLM) usage.

    Since vLLM is self-hosted, this is mainly for tracking.
    We use a nominal rate for comparison purposes.
    """
    # Nominal rate: ~$0.001 per 1K tokens (accounting for compute costs)
    return (tokens / 1000) * 0.001
