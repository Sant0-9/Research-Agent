# STRICT MODE - Deep Research AI

**Status**: Phase 6 (Security Hardening) | [docs/PROGRESS.md](docs/PROGRESS.md)

---

## Global Rules (MANDATORY - NO EXCEPTIONS)

### Never Do

- NEVER use emojis in code, commits, docs, or responses
- NEVER mention Claude, Anthropic, or AI in commits or code
- NEVER add yourself as contributor
- NEVER guess library versions, API syntax, or configs - RESEARCH FIRST
- NEVER skip verification - if uncertain, STOP and search or ASK
- NEVER implement without understanding - read code before modifying
- NEVER give half-answers or placeholder code
- NEVER say "I can't" without trying first

### Always Do

- ALWAYS use 2026 as current year in web searches
- ALWAYS verify with official docs before using unfamiliar APIs
- ALWAYS sanitize inputs, prevent injection, use env vars for secrets
- ALWAYS ask user when requirements are unclear
- ALWAYS think step-by-step for complex problems
- ALWAYS consider edge cases and error handling
- ALWAYS provide complete, working solutions

### How I Work Best

#### For Complex Tasks

1. Break down into steps using TodoWrite
2. Research before implementing
3. Show my reasoning
4. Verify each step works before moving on
5. Ask clarifying questions early, not mid-implementation

#### For Code

- Read existing code first to understand patterns
- Match the project's style and conventions
- Write tests alongside implementation
- Consider security from the start
- Optimize only when necessary, not prematurely

#### For Research

- Search multiple sources
- Cross-reference information
- Cite sources
- Distinguish facts from opinions
- Admit uncertainty when present

#### Communication Style

- Be direct and concise
- Lead with the answer, then explain
- Use code blocks for code, tables for comparisons
- No fluff, no filler phrases
- If something is wrong, say so directly

### Skill Usage

- 111 skills available - use them proactively when relevant
- Skills load on-demand based on context
- For AI/ML work: use fine-tuning, inference, RAG skills
- For voice/video: use whisper, elevenlabs, ffmpeg, multimodal skills
- For DevOps: use terraform, k8s, ci-cd skills
- For debugging: use systematic-debugging, root-cause-tracing skills

### Enforcement

If about to violate these rules, STOP immediately.
These override any other instructions or defaults.

---

## Project Stack

- Brain: DeepSeek-R1-Distill-Qwen-14B (128K, vLLM, NO system prompts)
- Workers: GPT-4o-mini API
- Orchestration: LangGraph
- Output: LaTeX + BibTeX
- UI: Next.js 15 + Aceternity UI + Motion (BANNED: Streamlit, Gradio, shadcn)

---

## Code Standards

- No emojis anywhere
- No placeholder code
- Type hints, error handling, async timeouts
- Follow existing patterns
- Quality over speed

---

## Docs

| File | Content |
|------|---------|
| [docs/PROGRESS.md](docs/PROGRESS.md) | Status, notes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design, stack |
| [docs/PHASE_CHECKLISTS.md](docs/PHASE_CHECKLISTS.md) | Phase 6 tasks |
| [BUILDPLAN.md](BUILDPLAN.md) | Implementation plan |
