"""Brain service client and prompt templates.

The brain is the reasoning core of the research agent,
powered by DeepSeek-R1-Distill-Qwen-14B via vLLM.
"""

from src.brain.client import (
    BrainClient,
    BrainResponse,
    StreamChunk,
    generate_brain_response,
)
from src.brain.prompts import (
    ANALYZE_PAPER,
    ANALYZE_SOURCES,
    FACT_CHECK,
    GENERATE_BIBTEX,
    GENERATE_SEARCH_QUERIES,
    REFINE_PLAN,
    RESEARCH_PLAN,
    REVIEW_PAPER,
    SYNTHESIZE_FINDINGS,
    WRITE_ABSTRACT,
    WRITE_CONCLUSION,
    WRITE_SECTION,
    PromptTemplate,
    get_all_prompts,
)

__all__ = [
    "ANALYZE_PAPER",
    "ANALYZE_SOURCES",
    "FACT_CHECK",
    "GENERATE_BIBTEX",
    "GENERATE_SEARCH_QUERIES",
    "REFINE_PLAN",
    "RESEARCH_PLAN",
    "REVIEW_PAPER",
    "SYNTHESIZE_FINDINGS",
    "WRITE_ABSTRACT",
    "WRITE_CONCLUSION",
    "WRITE_SECTION",
    "BrainClient",
    "BrainResponse",
    "PromptTemplate",
    "StreamChunk",
    "generate_brain_response",
    "get_all_prompts",
]
