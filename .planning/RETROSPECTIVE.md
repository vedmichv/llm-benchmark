# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v2.0 — Multi-Backend Benchmark

**Shipped:** 2026-03-15
**Phases:** 7 | **Plans:** 31 | **Tasks:** 40

### What Was Built
- Backend Protocol abstraction with 3 implementations (Ollama, llama.cpp, LM Studio)
- Auto-detection and auto-start for all backends with per-OS install instructions
- Cross-backend comparison (`--backend all`) with bar chart, matrix table, winner recommendation
- Interactive menu with backend selector and "Compare backends" shortcut
- Multi-backend setup documentation in README with per-OS guides
- 325 tests, CI pipeline, 63% coverage

### What Worked
- TDD approach (RED/GREEN commits) caught integration issues early
- Wave-based parallel execution sped up multi-plan phases
- Phase verification loop caught the display wiring gap before manual testing
- Backend Protocol pattern made adding new backends mechanical — same interface, different HTTP calls
- Using Ollama names as canonical for cross-backend model matching

### What Was Inefficient
- LM Studio API response format discovered only during UAT (not during research) — `/api/v1/models` returns `"models"` not `"data"`, and non-streaming mode returns empty `stats`
- Model name matching across backends required 3 iterations to get right (GGUF names, LM Studio names, canonical normalization)
- Nyquist VALIDATION.md created for all phases but never finalized — overhead without payoff
- Backend selector UX issue (appearing before mode selection) caught late in UAT

### Patterns Established
- Wall-clock elapsed time as fallback when backend doesn't report timing
- Ollama model names as canonical identity for cross-backend matrix
- `_match_to_ollama_names()` generic normalizer for any non-Ollama backend
- Winner counting excludes single-backend models (only compare when 2+ backends have the model)
- `lms` CLI for programmatic LM Studio model management

### Key Lessons
1. **Test with real backends during development, not just mocks** — 4 bugs found only when running real Ollama + llama-cpp + LM Studio
2. **API docs lie** — LM Studio's `/v1/` and `/api/v1/` endpoints return different schemas; always verify with `curl` first
3. **Name normalization is harder than it looks** — each backend has its own naming convention; start with a canonical name strategy early
4. **Always use `uv run` for all Python commands** — never bare `python`

### Cost Observations
- Model mix: ~60% opus (planning, execution), ~30% sonnet (verification, checking), ~10% inherit
- 4-day total timeline (v1.0 + v2.0 from scratch)
- Notable: parallel wave execution saved significant time in 3+ plan phases

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 | 4 | 12 | Foundation — established TDD, Rich patterns, CLI structure |
| v2.0 | 3 (+v1.0) | 19 (+12) | Multi-backend — Backend Protocol, cross-backend comparison |

### Cumulative Quality

| Milestone | Tests | Coverage | New Dependencies |
|-----------|-------|----------|-----------------|
| v1.0 | 152 | 63% | rich, pydantic, tenacity, ollama |
| v2.0 | 325 | 63% | httpx (only new dep) |

### Top Lessons (Verified Across Milestones)

1. Phase verification catches real gaps — display wiring (v2.0), no false positives
2. TDD with mocks works for unit tests but real integration testing is essential for multi-system features
3. Keep dependencies minimal — httpx was the only new dep for 3 entire backends
