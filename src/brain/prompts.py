"""Prompt templates for the DeepSeek-R1-Distill brain model.

CRITICAL: DeepSeek-R1-Distill models do NOT use system prompts.
All instructions must be in the user message.
Do NOT use few-shot examples - they degrade performance.

The model naturally uses <think>...</think> tags for reasoning.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution."""

    template: str
    description: str

    def format(self, **kwargs: Any) -> str:
        """Format the template with the given variables."""
        return self.template.format(**kwargs)


# =============================================================================
# Research Planning Prompts
# =============================================================================

RESEARCH_PLAN = PromptTemplate(
    template="""You are a research scientist. Your task is to create a comprehensive research plan.

RESEARCH QUERY: {query}

DOMAIN(S): {domains}

Create a detailed research plan that includes:

1. KEY QUESTIONS: What specific questions need to be answered?
2. SEARCH STRATEGY: What search queries should be used? (list 5-10 specific queries)
3. SOURCES TO EXPLORE:
   - Academic sources (ArXiv, journals)
   - Web sources (documentation, blogs, papers)
4. METHODOLOGY: How should the gathered information be analyzed?
5. EXPECTED STRUCTURE: What sections should the final paper have?

Think through this carefully. Consider what information is essential vs. supplementary.

Output your plan in a structured format.""",
    description="Creates a research plan from a user query",
)


REFINE_PLAN = PromptTemplate(
    template="""You are refining a research plan based on initial findings.

ORIGINAL QUERY: {query}

CURRENT PLAN:
{current_plan}

INITIAL FINDINGS SUMMARY:
{findings_summary}

Based on these initial findings:
1. What aspects of the original plan are well-covered?
2. What knowledge gaps remain?
3. What additional searches are needed?
4. Should the research direction be adjusted?

Provide an updated plan focusing on the gaps.""",
    description="Refines a research plan based on initial findings",
)


# =============================================================================
# Analysis Prompts
# =============================================================================

ANALYZE_SOURCES = PromptTemplate(
    template="""You are analyzing research sources for a paper.

RESEARCH TOPIC: {topic}

SOURCES TO ANALYZE:
{sources}

For each source, evaluate:
1. RELEVANCE: How relevant is this to the research topic? (High/Medium/Low)
2. QUALITY: Is this a reliable source? (Academic paper, documentation, blog, etc.)
3. KEY INSIGHTS: What are the main takeaways?
4. LIMITATIONS: Any caveats or limitations in the source?

Then provide:
- SYNTHESIS: How do these sources connect and complement each other?
- GAPS: What important topics are NOT covered by these sources?
- CONFLICTS: Any contradictory information between sources?

Think step by step through each source before synthesizing.""",
    description="Analyzes and evaluates research sources",
)


ANALYZE_PAPER = PromptTemplate(
    template="""You are analyzing an academic paper for a research project.

PAPER TITLE: {title}
AUTHORS: {authors}
ABSTRACT: {abstract}

FULL CONTENT:
{content}

Provide a detailed analysis:

1. MAIN CONTRIBUTIONS: What are the key contributions of this paper?
2. METHODOLOGY: What approach/methods did the authors use?
3. RESULTS: What are the main findings?
4. LIMITATIONS: What limitations do the authors acknowledge or you identify?
5. RELEVANCE TO RESEARCH: How does this connect to: {research_topic}
6. QUOTABLE SECTIONS: Key quotes that could be cited (with context)
7. CITATION QUALITY: Rate 1-5 how essential this paper is to cite

Be thorough and precise. Extract specific technical details.""",
    description="Deep analysis of an academic paper",
)


# =============================================================================
# Synthesis Prompts
# =============================================================================

SYNTHESIZE_FINDINGS = PromptTemplate(
    template="""You are synthesizing research findings into a coherent narrative.

RESEARCH QUESTION: {query}

ANALYZED SOURCES:
{analyzed_sources}

KEY FINDINGS:
{key_findings}

Your task is to synthesize these findings into a coherent research narrative:

1. INTRODUCTION: Frame the research question and its importance
2. BACKGROUND: What prior work is relevant?
3. MAIN FINDINGS: What have we learned from the sources?
4. ANALYSIS: What patterns, trends, or insights emerge?
5. IMPLICATIONS: What are the implications of these findings?
6. FUTURE DIRECTIONS: What questions remain open?

Focus on:
- Logical flow and coherence
- Proper attribution (note which source each claim comes from)
- Identifying consensus vs. debate in the field
- Technical accuracy

Think through the connections between sources carefully.""",
    description="Synthesizes research findings into a narrative",
)


# =============================================================================
# Writing Prompts
# =============================================================================

WRITE_SECTION = PromptTemplate(
    template="""You are writing a section of a research paper.

PAPER TOPIC: {topic}

SECTION: {section_name}

SECTION OUTLINE:
{outline}

RELEVANT SOURCES AND FINDINGS:
{sources}

Write this section following academic writing conventions:
- Clear, precise language
- Logical flow of ideas
- Proper citations (use [Author, Year] format - citations will be converted to BibTeX later)
- Technical accuracy
- Appropriate depth for the target audience

The section should be {length_guidance}.

Write the section now. Use LaTeX formatting where appropriate (equations, etc.).""",
    description="Writes a section of the research paper",
)


WRITE_ABSTRACT = PromptTemplate(
    template="""You are writing an abstract for a research paper.

TITLE: {title}

PAPER CONTENT SUMMARY:
{content_summary}

Write a concise abstract (150-250 words) that:
1. States the research question/problem
2. Briefly describes the methodology
3. Summarizes key findings
4. States the main conclusions and implications

The abstract should stand alone and be understandable without the full paper.""",
    description="Writes the paper abstract",
)


WRITE_CONCLUSION = PromptTemplate(
    template="""You are writing the conclusion of a research paper.

RESEARCH QUESTION: {query}

MAIN FINDINGS:
{main_findings}

KEY CONTRIBUTIONS:
{contributions}

Write a conclusion section that:
1. Restates the research question and its importance
2. Summarizes the main findings
3. Discusses implications and significance
4. Acknowledges limitations
5. Suggests future research directions

The conclusion should be {length_guidance}.""",
    description="Writes the paper conclusion",
)


# =============================================================================
# Quality Review Prompts
# =============================================================================

REVIEW_PAPER = PromptTemplate(
    template="""You are reviewing a research paper for quality and accuracy.

PAPER CONTENT:
{paper_content}

SOURCES USED:
{sources}

Review the paper for:

1. FACTUAL ACCURACY:
   - Are claims properly supported by sources?
   - Any statements that seem unsupported or potentially incorrect?
   - Are citations used appropriately?

2. LOGICAL COHERENCE:
   - Does the argument flow logically?
   - Are there gaps in reasoning?
   - Are conclusions supported by the evidence?

3. COMPLETENESS:
   - Are all important aspects of the topic covered?
   - Any obvious omissions?

4. TECHNICAL ACCURACY:
   - Are technical terms used correctly?
   - Are equations/formulas correct?
   - Any mathematical or logical errors?

5. WRITING QUALITY:
   - Is the writing clear and precise?
   - Any grammatical issues?
   - Is the tone appropriate for academic writing?

Provide specific feedback with line references where possible.
Rate overall quality: 1-5 (5 being publication-ready).""",
    description="Reviews paper for quality and accuracy",
)


FACT_CHECK = PromptTemplate(
    template="""You are fact-checking claims in a research paper.

CLAIMS TO VERIFY:
{claims}

AVAILABLE SOURCES:
{sources}

For each claim:
1. Is it supported by the provided sources? (Yes/No/Partial)
2. If yes, which source(s) support it?
3. If no or partial, what evidence would be needed?
4. Is the claim accurately representing the source, or is it misinterpreted?

Be rigorous. Unsupported claims should be flagged for removal or citation.""",
    description="Fact-checks claims against sources",
)


# =============================================================================
# Search Query Generation
# =============================================================================

GENERATE_SEARCH_QUERIES = PromptTemplate(
    template="""You are generating search queries for a research project.

RESEARCH TOPIC: {topic}

DOMAIN: {domain}

CURRENT KNOWLEDGE GAPS:
{gaps}

Generate 5-10 specific search queries that would help fill these gaps.

For each query, specify:
- QUERY: The search query text
- TARGET: web | arxiv | both
- PRIORITY: high | medium | low
- RATIONALE: Why this query is important

Focus on queries that will find:
- Authoritative sources
- Recent developments (2024-2026)
- Technical details
- Different perspectives on the topic""",
    description="Generates targeted search queries",
)


# =============================================================================
# Citation Generation
# =============================================================================

GENERATE_BIBTEX = PromptTemplate(
    template="""You are generating a BibTeX entry for a source.

SOURCE INFORMATION:
Title: {title}
Authors: {authors}
URL: {url}
Publication Date: {date}
Source Type: {source_type}

Additional metadata:
{metadata}

Generate a properly formatted BibTeX entry.
Use an appropriate entry type (@article, @inproceedings, @misc, @book, etc.)
Create a sensible citation key (AuthorYear format, e.g., smith2025).

Output only the BibTeX entry, nothing else.""",
    description="Generates BibTeX entries from source info",
)


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_prompts() -> dict[str, PromptTemplate]:
    """Get all available prompt templates."""
    return {
        "research_plan": RESEARCH_PLAN,
        "refine_plan": REFINE_PLAN,
        "analyze_sources": ANALYZE_SOURCES,
        "analyze_paper": ANALYZE_PAPER,
        "synthesize_findings": SYNTHESIZE_FINDINGS,
        "write_section": WRITE_SECTION,
        "write_abstract": WRITE_ABSTRACT,
        "write_conclusion": WRITE_CONCLUSION,
        "review_paper": REVIEW_PAPER,
        "fact_check": FACT_CHECK,
        "generate_search_queries": GENERATE_SEARCH_QUERIES,
        "generate_bibtex": GENERATE_BIBTEX,
    }
