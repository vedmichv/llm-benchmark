---
phase: 05-backend-abstraction
plan: 03
subsystem: api
tags: [backend-abstraction, threadpool, refactor, protocol-pattern, ollama]

# Dependency graph
requires:
  - phase: 05-01
    provides: Backend Protocol, BackendResponse, BackendError, StreamResult, OllamaBackend, create_backend()
  - phase: 05-02
    provides: Backend-aware runner, preflight, system, exporters with seconds-based timing
provides:
  - Backend-agnostic concurrent benchmarking via ThreadPoolExecutor (no asyncio)
  - Backend-agnostic parameter sweep with dict options (no ollama.Options)
  - Backend-agnostic model recommendations via backend.list_models()
  - Full CLI wiring creating backend once and passing to all modules
  - Zero ollama imports outside backends/ in entire codebase and tests
affects: [06-lm-studio, 06-llama-cpp]

# Tech tracking
tech-stack:
  added: []
  patterns: [ThreadPoolExecutor replacing asyncio for concurrent requests, duck-typing for backend-specific methods like get_model_layers]

key-files:
  created: []
  modified:
    - llm_benchmark/concurrent.py
    - llm_benchmark/sweep.py
    - llm_benchmark/recommend.py
    - llm_benchmark/menu.py
    - llm_benchmark/cli.py
    - llm_benchmark/backends/ollama.py
    - tests/conftest.py
    - tests/test_concurrent.py
    - tests/test_sweep.py
    - tests/test_menu.py

key-decisions:
  - "ThreadPoolExecutor replaces asyncio for concurrent requests -- simpler, Backend.chat() is sync"
  - "get_model_layers uses duck-typing (hasattr check) to keep Backend Protocol clean of Ollama-specific methods"
  - "Removed isinstance backward-compat checks in menu/recommend -- models are always dicts now"

patterns-established:
  - "Duck-typing for backend-specific methods: hasattr(backend, 'method') before calling"
  - "All entry paths (run, recommend, menu, info) create backend via create_backend() at top level"

requirements-completed: [BACK-04, BACK-05]

# Metrics
duration: 6min
completed: 2026-03-14
---

# Phase 5 Plan 3: Final Backend Migration Summary

**Completed Backend abstraction: ThreadPoolExecutor concurrent mode, dict-based sweep, backend-agnostic recommend/menu/cli -- zero ollama leaks, 170 tests pass**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-14T12:16:28Z
- **Completed:** 2026-03-14T12:22:41Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments
- concurrent.py migrated from asyncio/AsyncClient to ThreadPoolExecutor with backend.chat()
- sweep.py migrated from ollama.chat/Options to backend.chat() with dict options
- recommend.py migrated from lazy `import ollama` to backend.list_models() for model re-fetch
- menu.py and cli.py fully wired with backend parameter threading through all paths
- OllamaBackend gained get_model_layers() for sweep layer detection (duck-typed)
- All 170 tests pass with updated fixtures using mock_backend and BackendResponse
- Zero `import ollama`, `OllamaResponse`, or `_ns_to_sec` outside backends/

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor concurrent, sweep, recommend, menu, cli to use Backend** - `165451d` (refactor)
2. **Task 2: Update all test files -- fixtures, mocks, and assertions** - `c552aa3` (test)

## Files Created/Modified
- `llm_benchmark/concurrent.py` - ThreadPoolExecutor-based concurrent benchmarking, backend.chat() calls
- `llm_benchmark/sweep.py` - Backend-agnostic parameter sweep with dict options
- `llm_benchmark/recommend.py` - Backend.list_models() for model re-fetch after pull
- `llm_benchmark/menu.py` - Accepts backend param, dict-only model access (no isinstance checks)
- `llm_benchmark/cli.py` - All paths (run, recommend, menu, info) create and pass backend
- `llm_benchmark/backends/ollama.py` - Added get_model_layers() method for sweep
- `tests/conftest.py` - Added mock_backend, cached_backend_response fixtures; removed mock_async_client
- `tests/test_concurrent.py` - ThreadPoolExecutor tests with mock_backend
- `tests/test_sweep.py` - Backend method mocks instead of ollama SDK mocks
- `tests/test_menu.py` - Dict models, backend parameter, updated mock signatures

## Decisions Made
- ThreadPoolExecutor replaces asyncio.gather for concurrent requests. Backend.chat() is synchronous, so ThreadPoolExecutor is the natural fit. Simpler code, no async/await complexity.
- get_model_layers() added to OllamaBackend but NOT to Backend Protocol. It is an Ollama-specific concept. Sweep uses duck-typing (hasattr check) and gracefully degrades when unavailable.
- Removed isinstance backward-compat checks in menu.py and recommend.py. Models are always dicts from backend.list_models() now -- no need for dual-path code.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Entire codebase is backend-agnostic: zero ollama imports outside backends/ollama.py
- Backend Protocol fully wired through all modules
- Ready for Phase 6: adding LM Studio and llama.cpp backend implementations
- All 170 tests pass with Backend abstraction

---
*Phase: 05-backend-abstraction*
*Completed: 2026-03-14*
