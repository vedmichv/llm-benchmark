---
phase: 05-backend-abstraction
plan: 01
subsystem: api
tags: [protocol, pydantic, ollama, abstraction, factory-pattern]

# Dependency graph
requires: []
provides:
  - Backend Protocol with 9 methods (@runtime_checkable)
  - BackendResponse Pydantic model (seconds-based timing)
  - BackendError exception with retryable flag
  - StreamResult wrapper for streaming responses
  - OllamaBackend implementation wrapping ollama SDK
  - create_backend() factory function
affects: [05-02, 05-03, 06-lm-studio, 06-llama-cpp]

# Tech tracking
tech-stack:
  added: []
  patterns: [Backend Protocol pattern, factory function with lazy import, nanosecond-to-seconds conversion internalized in backend]

key-files:
  created:
    - llm_benchmark/backends/__init__.py
    - llm_benchmark/backends/ollama.py
    - tests/test_backends.py
  modified: []

key-decisions:
  - "Import ollama exception classes directly to avoid mock interference in tests"
  - "StreamResult uses finalize callable for deferred BackendResponse construction"
  - "BackendResponse has all timing fields as float seconds with sensible defaults (0.0)"

patterns-established:
  - "Backend Protocol: all backends implement 9 methods via typing.Protocol"
  - "Error wrapping: native SDK errors wrapped into BackendError with retryable flag"
  - "Timing normalization: each backend converts native units to seconds internally"

requirements-completed: [BACK-01, BACK-02, BACK-03]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 5 Plan 1: Backend Protocol and OllamaBackend Summary

**Backend Protocol with 9 methods, BackendResponse model in seconds, OllamaBackend wrapping ollama SDK with error mapping and streaming support**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T11:56:44Z
- **Completed:** 2026-03-14T11:59:28Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Backend Protocol (@runtime_checkable) with chat, list_models, unload_model, warmup, detect_context_window, get_model_size, check_connectivity, name, version
- BackendResponse Pydantic model storing all timing in seconds (float), not nanoseconds
- OllamaBackend converts nanoseconds to seconds, detects prompt caching, wraps errors with retryable flags
- StreamResult provides chunks iterator for streaming display and deferred response for timing data
- create_backend("ollama") factory with lazy import; "unknown" raises ValueError
- 20 comprehensive tests covering all behaviors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create backends subpackage with Protocol, models, and factory** (TDD)
   - RED: `fa32836` (test: add failing tests for backend abstraction layer)
   - GREEN: `8b912a7` (feat: implement backend abstraction with Protocol, models, and OllamaBackend)

## Files Created/Modified
- `llm_benchmark/backends/__init__.py` - Backend Protocol, BackendResponse, BackendError, StreamResult, create_backend()
- `llm_benchmark/backends/ollama.py` - OllamaBackend implementation wrapping ollama SDK
- `tests/test_backends.py` - 20 tests covering protocol compliance, models, factory, error wrapping, streaming

## Decisions Made
- Imported ollama exception classes directly (`from ollama import RequestError`) to avoid mock interference when patching the ollama module in tests
- StreamResult uses a finalize callable pattern for deferred BackendResponse construction after chunk iteration
- All BackendResponse timing fields default to 0.0 for flexibility across backends

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed mock interference with exception catching**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** Patching `llm_benchmark.backends.ollama.ollama` caused `except ollama.RequestError` to reference mock objects instead of real exception classes, raising TypeError
- **Fix:** Imported exception classes directly (`from ollama import RequestError as _OllamaRequestError`) so they are bound at import time, not affected by module-level mocking
- **Files modified:** llm_benchmark/backends/ollama.py
- **Verification:** All 20 tests pass including error wrapping tests
- **Committed in:** 8b912a7 (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix for testability. No scope creep.

## Issues Encountered
None beyond the mock interference issue documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend Protocol and OllamaBackend ready for integration into runner, preflight, and CLI
- All 172 existing tests still pass (zero regression)
- Ready for plan 05-02 (module refactoring to use Backend)

---
*Phase: 05-backend-abstraction*
*Completed: 2026-03-14*
