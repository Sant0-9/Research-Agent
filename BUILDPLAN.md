# BUILDPLAN.md - Comprehensive Build Plan

## Project: Deep Research AI System

**Created**: 2026-01-20
**Scope**: Nevever MVP. Full comprehensive execution with short-term memory only (no self-evolution, no long-term memory)

---

## Executive Summary

Build a secure, production-quality research agent that can:
1. Accept research queries (AI/ML, Quantum Physics, Astrophysics)
2. Search and gather information from web and ArXiv
3. Synthesize findings using a self-hosted reasoning model
4. Output publication-ready LaTeX papers with citations

**Deferred to v2:** Long-term memory, self-evolution, fine-tuning

---

## Architecture Overview (MVP)

```
USER
  |
  v
+------------------+
|   NEXT.JS UI     |  <-- Next.js 15 + Aceternity UI + Tailwind + Motion
|   (Minimal,      |      Premium typography, subtle animations
|    Premium)      |      Dark theme, fast & responsive
+------------------+
  |
  v
+------------------+
|   FastAPI        |  <-- Backend API (Python ML ecosystem)
+------------------+
  |
  v
+------------------+
|   ORCHESTRATOR   |  <-- LangGraph workflow
|   (Pipeline)     |
+------------------+
  |
  +---> BRAIN (DeepSeek-R1-Distill-14B, self-hosted vLLM)
  |       - Planning
  |       - Analysis
  |       - Synthesis
  |       - Quality review
  |
  +---> WORKERS (GPT-4o-mini API)
  |       - Web search (Tavily/SerpAPI)
  |       - ArXiv search
  |       - Data extraction
  |       - LaTeX drafting
  |
  +---> SHORT-TERM MEMORY
          - 128K context window
          - In-memory state during research
          - No persistence between sessions (MVP)
  |
  v
+------------------+
|   OUTPUT         |
|   - LaTeX paper  |
|   - BibTeX file  |
|   - Research log |
+------------------+
```

---

## Phase Breakdown

### PHASE 1: Foundation & Infrastructure
**Goal:** Project setup, dev environment, basic services running

### PHASE 2: Brain Service
**Goal:** Self-hosted DeepSeek-R1-Distill-14B with vLLM

### PHASE 3: Worker Services
**Goal:** GPT-4o-mini workers for search and extraction

### PHASE 4: Orchestration Pipeline
**Goal:** LangGraph workflow connecting brain and workers

### PHASE 5: LaTeX Output
**Goal:** Generate publication-ready papers with citations

### PHASE 6: Security Hardening & Polish
**Goal:** Production-ready, secure, documented

---

## PHASE 1: Foundation & Infrastructure

### 1.1 Project Structure

```
research-agent/
├── CLAUDE.md
├── BUILDPLAN.md
├── RESEARCH_REPORT.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── docker-compose.yml
├── docker-compose.dev.yml
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration management
│   │
│   ├── brain/                  # Brain service client
│   │   ├── __init__.py
│   │   ├── client.py           # vLLM client
│   │   └── prompts.py          # Prompt templates
│   │
│   ├── workers/                # Worker services
│   │   ├── __init__.py
│   │   ├── base.py             # Base worker class
│   │   ├── search.py           # Web search worker
│   │   ├── arxiv.py            # ArXiv worker
│   │   └── writer.py           # LaTeX writer worker
│   │
│   ├── pipeline/               # LangGraph orchestration
│   │   ├── __init__.py
│   │   ├── state.py            # State schema
│   │   ├── nodes.py            # Node functions
│   │   ├── graph.py            # Graph definition
│   │   └── workflow.py         # Workflow execution
│   │
│   ├── output/                 # Output generation
│   │   ├── __init__.py
│   │   ├── latex.py            # LaTeX generation
│   │   ├── bibtex.py           # BibTeX management
│   │   └── templates/          # LaTeX templates
│   │
│   ├── security/               # Security utilities
│   │   ├── __init__.py
│   │   ├── validation.py       # Input validation
│   │   ├── sanitization.py     # Output sanitization
│   │   └── rate_limit.py       # Rate limiting
│   │
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── logging.py          # Structured logging
│       └── errors.py           # Custom exceptions
│
├── services/
│   └── brain/                  # Brain service (vLLM)
│       ├── Dockerfile
│       └── config.yaml
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_brain.py
│   ├── test_workers.py
│   ├── test_pipeline.py
│   └── test_security.py
│
├── scripts/
│   ├── setup.sh
│   ├── run_local.sh
│   └── test.sh
│
└── outputs/                    # Generated papers (gitignored)
    └── .gitkeep
```

### 1.2 Dependencies to Research

Before implementing, research current versions and best practices for:

| Library | Purpose | Questions to Answer |
|---------|---------|---------------------|
| FastAPI | API server | Latest version? Async patterns? |
| LangGraph | Orchestration | Latest API? State management patterns? |
| vLLM | Inference | Version compatible with DeepSeek? Deployment patterns? |
| httpx | Async HTTP | Best practices for retries, timeouts? |
| Pydantic | Validation | v2 patterns? Settings management? |
| Tavily/SerpAPI | Web search | API limits? Pricing? Best option for 2026? |
| arxiv | ArXiv API | Rate limits? Bulk download patterns? |

### 1.3 Security Checklist (Phase 1)

- [ ] `.env.example` with all required vars (no real values)
- [ ] `.gitignore` includes `.env`, `outputs/`, `__pycache__/`, etc.
- [ ] No hardcoded API keys anywhere
- [ ] Config loaded from environment only
- [ ] Secrets validation on startup (fail fast if missing)

### 1.4 Deliverables

- [ ] Working directory structure
- [ ] `pyproject.toml` with pinned dependencies
- [ ] `docker-compose.yml` skeleton
- [ ] `.env.example` with all required variables
- [ ] Basic FastAPI app that starts
- [ ] Health check endpoint (`/health`)
- [ ] Structured logging configured

---

## PHASE 2: Brain Service

### 2.1 Objectives

- Self-hosted DeepSeek-R1-Distill-Qwen-14B
- vLLM server with OpenAI-compatible API
- Prompt templates for research tasks
- Streaming response support

### 2.2 Research Questions

Before building:
1. What's the latest vLLM version and its DeepSeek support?
2. Optimal quantization for RTX 4090 (Q4_K_M vs AWQ vs GPTQ)?
3. vLLM configuration for 128K context?
4. Best practices for prompt engineering with R1-Distill models?

### 2.3 Components

**Brain Client (`src/brain/client.py`):**
- Async client for vLLM server
- Streaming support
- Timeout handling
- Retry logic with exponential backoff
- Token counting

**Prompt Templates (`src/brain/prompts.py`):**
- Research planning prompt
- Analysis prompt
- Synthesis prompt
- Quality review prompt
- Each with clear system message and structure

**Brain Service (`services/brain/`):**
- Dockerfile with CUDA support
- vLLM configuration
- Model download script
- Health check

### 2.4 Security Checklist (Phase 2)

- [ ] vLLM server not exposed to internet (internal network only)
- [ ] API key required for brain service access
- [ ] Input length limits enforced
- [ ] No arbitrary code execution in prompts
- [ ] Timeout on all inference calls

### 2.5 Deliverables

- [ ] vLLM server running with DeepSeek model
- [ ] Brain client with all CRUD operations
- [ ] Prompt templates tested and working
- [ ] Streaming responses functional
- [ ] Unit tests for brain client
- [ ] Documentation for brain service setup

---

## PHASE 3: Worker Services

### 3.1 Objectives

- GPT-4o-mini workers for specialized tasks
- Web search capability
- ArXiv paper search and download
- Parallel execution support

### 3.2 Research Questions

Before building:
1. Tavily vs SerpAPI vs other search APIs in 2026?
2. ArXiv API best practices and rate limits?
3. Best patterns for parallel async workers?
4. GPT-4o-mini function calling patterns?

### 3.3 Components

**Base Worker (`src/workers/base.py`):**
- Abstract base class
- Common retry logic
- Error handling
- Cost tracking

**Search Worker (`src/workers/search.py`):**
- Web search via Tavily/SerpAPI
- Result parsing and cleaning
- Source validation
- Deduplication

**ArXiv Worker (`src/workers/arxiv.py`):**
- Paper search by keywords
- Paper search by category
- Abstract extraction
- PDF download (if needed)
- BibTeX entry generation

**Writer Worker (`src/workers/writer.py`):**
- Section drafting
- LaTeX formatting assistance
- Citation formatting

### 3.4 Security Checklist (Phase 3)

- [ ] API keys stored securely (env vars only)
- [ ] Response validation (don't trust external APIs)
- [ ] URL validation before fetching
- [ ] No execution of content from external sources
- [ ] Rate limiting on outbound requests
- [ ] Timeout on all external calls

### 3.5 Deliverables

- [ ] Search worker with web search capability
- [ ] ArXiv worker with paper retrieval
- [ ] Writer worker for LaTeX assistance
- [ ] All workers tested with mocks
- [ ] Cost tracking per worker call
- [ ] Error handling for API failures

---

## PHASE 4: Orchestration Pipeline

### 4.1 Objectives

- LangGraph workflow for research process
- State management across nodes
- Conditional branching (iterate if needed)
- Error recovery

### 4.2 Research Questions

Before building:
1. LangGraph latest patterns for complex workflows?
2. Checkpointing for long-running tasks?
3. Best practices for state schema design?
4. How to handle partial failures?

### 4.3 Research Workflow

```
START
  |
  v
[PLAN] Brain plans research approach
  |
  v
[SEARCH] Workers gather information (parallel)
  |     - Web search
  |     - ArXiv search
  v
[ANALYZE] Brain analyzes gathered information
  |
  +---> Need more info? ---> [SEARCH] (loop max 3x)
  |
  v
[SYNTHESIZE] Brain synthesizes findings
  |
  v
[WRITE] Workers draft sections + Brain reviews
  |
  v
[OUTPUT] Generate LaTeX + BibTeX
  |
  v
END
```

### 4.4 State Schema

```python
class ResearchState(TypedDict):
    # Input
    query: str
    domains: list[str]  # ["quantum_physics", "ml"]

    # Planning
    research_plan: dict | None
    search_queries: list[str]

    # Gathering
    search_results: list[dict]
    arxiv_papers: list[dict]
    sources: list[dict]

    # Analysis
    analysis: dict | None
    knowledge_gaps: list[str]
    iteration_count: int

    # Synthesis
    synthesis: dict | None
    paper_outline: dict | None

    # Output
    sections: dict[str, str]  # section_name -> content
    citations: list[dict]

    # Meta
    errors: list[str]
    cost_usd: float
    tokens_used: int
```

### 4.5 Security Checklist (Phase 4)

- [ ] State cannot be tampered with externally
- [ ] Iteration limits enforced (prevent infinite loops)
- [ ] Timeout on entire workflow
- [ ] No sensitive data logged in state
- [ ] Graceful handling of partial failures

### 4.6 Deliverables

- [ ] LangGraph workflow definition
- [ ] All node functions implemented
- [ ] State management working
- [ ] Conditional branching tested
- [ ] End-to-end workflow test
- [ ] Cost tracking across workflow

---

## PHASE 5: LaTeX Output

### 5.1 Objectives

- Generate compilable LaTeX papers
- Proper citation management
- Multiple paper formats (article, report)
- Quality validation

### 5.2 Research Questions

Before building:
1. Best LaTeX templates for research papers?
2. BibTeX generation best practices?
3. LaTeX validation libraries in Python?
4. How to handle math equations from LLM output?

### 5.3 Components

**LaTeX Generator (`src/output/latex.py`):**
- Template rendering (Jinja2)
- Section assembly
- Math equation handling
- Figure placeholders

**BibTeX Manager (`src/output/bibtex.py`):**
- Entry creation from URLs
- Entry creation from ArXiv
- Duplicate detection
- Key generation

**Templates (`src/output/templates/`):**
- `article.tex` - Standard article
- `report.tex` - Technical report
- Configurable metadata (title, author, date)

### 5.4 Security Checklist (Phase 5)

- [ ] No LaTeX injection (sanitize all content)
- [ ] Safe file path handling (no path traversal)
- [ ] Output directory restricted
- [ ] No shell execution in LaTeX compilation
- [ ] Validate generated LaTeX before saving

### 5.5 Deliverables

- [ ] LaTeX template system
- [ ] BibTeX generation
- [ ] Paper compilation (pdflatex)
- [ ] Output validation
- [ ] Sample papers generated
- [ ] Templates documented

---

## PHASE 6: Security Hardening & Polish

### 6.1 Objectives

- Production-ready security
- Comprehensive documentation
- Easy setup and deployment
- Demo-ready

### 6.2 Security Audit Checklist

**Input Validation:**
- [ ] All API inputs validated with Pydantic
- [ ] Query length limits
- [ ] Allowed characters whitelist
- [ ] No SQL/command injection vectors

**Authentication & Authorization:**
- [ ] API key authentication
- [ ] Rate limiting per key
- [ ] Request logging

**Network Security:**
- [ ] Internal services not exposed
- [ ] HTTPS for external calls
- [ ] Firewall rules documented

**Data Security:**
- [ ] No PII logging
- [ ] Secure secret management
- [ ] Output sanitization

**Dependency Security:**
- [ ] All dependencies pinned
- [ ] No known vulnerabilities (pip-audit)
- [ ] Minimal dependency footprint

### 6.3 Documentation Checklist

- [ ] README.md with full setup guide
- [ ] API documentation (OpenAPI)
- [ ] Configuration reference
- [ ] Troubleshooting guide
- [ ] Example queries and outputs

### 6.4 Scripts & Automation

- [ ] `scripts/setup.sh` - One-command setup
- [ ] `scripts/run.sh` - Start all services
- [ ] `scripts/test.sh` - Run all tests
- [ ] `scripts/demo.sh` - Run demo queries

### 6.5 Deliverables

- [ ] Security audit passed
- [ ] All documentation complete
- [ ] Setup tested on clean machine
- [ ] Demo queries working
- [ ] Performance benchmarks documented

---

## Questions Before Starting

Before I begin implementation, I need to confirm:

1. **Development Environment:**
   - Will development happen locally or on Brev.dev?
   - Do you have a GPU available locally for testing?
   - Which OS are you developing on?

2. **API Keys:**
   - Do you have OpenAI API access (for GPT-4o-mini)?
   - Do you have Tavily or SerpAPI access for web search?
   - Any preference between search providers?

3. **Priorities:**
   - Which domains to focus on first? (AI/ML, Quantum, Astro)
   - Preference for CLI interface vs web UI vs both?
   - Any specific paper format requirements?

4. **Scope Confirmation:**
   - Confirm: No long-term memory in MVP (papers not remembered between sessions)
   - Confirm: No fine-tuning in MVP (use base DeepSeek model)
   - Confirm: No self-evolution in MVP

5. **Output:**
   - Where should generated papers be saved?
   - Any specific LaTeX template preferences?
   - Do you need PDF compilation or just .tex files?

---

## Success Criteria

The MVP is complete when:

1. **Functional:**
   - Can accept a research query via API
   - Searches web and ArXiv for relevant information
   - Synthesizes findings into coherent analysis
   - Outputs compilable LaTeX paper with citations

2. **Quality:**
   - Generated papers are coherent and well-structured
   - Citations are accurate and properly formatted
   - No hallucinated references

3. **Security:**
   - No known vulnerabilities
   - All inputs validated
   - Secrets properly managed
   - Rate limiting in place

4. **Operational:**
   - One-command setup works
   - Documentation is complete
   - Tests pass
   - Demo runs successfully

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Foundation | 2-3 days | 2-3 days |
| Phase 2: Brain Service | 3-4 days | 5-7 days |
| Phase 3: Workers | 3-4 days | 8-11 days |
| Phase 4: Pipeline | 4-5 days | 12-16 days |
| Phase 5: LaTeX Output | 2-3 days | 14-19 days |
| Phase 6: Security & Polish | 3-4 days | 17-23 days |

**Total: ~3-4 weeks for MVP**

*Note: These are estimates. Actual time depends on research findings and unforeseen issues.*

---

## Next Steps

1. User answers questions above
2. I research Phase 1 dependencies and best practices
3. We review and confirm the approach
4. Begin Phase 1 implementation

**Waiting for user input before proceeding.**
