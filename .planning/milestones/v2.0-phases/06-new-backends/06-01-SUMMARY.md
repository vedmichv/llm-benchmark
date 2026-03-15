---
phase: 06-new-backends
plan: 01
subsystem: backends
tags: [httpx, llama-cpp, lm-studio, backend-protocol, streaming, sse]

# Dependency graph
requires:
  - phase: 05-backend-abstraction
    provides: "Backend Protocol, BackendResponse, BackendError, StreamResult, create_backend()"
provides:
  - "LlamaCppBackend class with ms-to-seconds timing conversion"
  - "LMStudioBackend class with tokens_per_second-derived timing"
  - "httpx as explicit project dependency"
  - "Shared httpx mock response factory in conftest.py"
affects: [06-02-detection, 06-03-cli-integration, 06-04-preflight, 06-05-exporters]

# Tech tracking
tech-stack:
  added: [httpx]
  patterns: [httpx-client-per-backend, sse-line-parsing, ms-to-seconds-conversion, tps-derived-duration]

key-files:
  created:
    - llm_benchmark/backends/llamacpp.py
    - llm_benchmark/backends/lmstudio.py
    - tests/test_llamacpp.py
    - tests/test_lmstudio.py
  modified:
    - tests/conftest.py
    - pyproject.toml

key-decisions:
  - "httpx.Client with base_url per backend instance (not shared)"
  - "SSE streaming parsed via iter_lines with data: prefix stripping"
  - "LM Studio eval_duration derived from eval_count / tokens_per_second"
  - "llama.cpp total_duration computed as prompt_ms + predicted_ms (no separate field)"

patterns-established:
  - "Backend httpx pattern: Client(base_url, timeout=600) stored as self._client"
  - "SSE streaming: stream context manager with iter_lines, json.loads per data: line"
  - "Mock factory: make_httpx_response() in conftest for all httpx-based backend tests"

requirements-completed: [BEND-01, BEND-02]

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 6 Plan 01: Backend Core Classes Summary

**LlamaCppBackend and LMStudioBackend with httpx HTTP clients, ms/tps timing conversion, SSE streaming, and 50 unit tests via mocked httpx**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T16:50:11Z
- **Completed:** 2026-03-14T16:55:14Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 6

## Accomplishments
- LlamaCppBackend satisfies Backend Protocol with millisecond-to-seconds timing conversion from llama-server timings object
- LMStudioBackend satisfies Backend Protocol with eval_duration derived from tokens_per_second in stats object
- Both backends use httpx.Client for HTTP communication with SSE streaming support
- Both backends wrap httpx exceptions (ConnectError, TimeoutException, HTTPStatusError) in BackendError with retryable flag
- 50 new tests covering protocol compliance, chat timing, streaming, connectivity, error handling
- All 245 tests pass (170 existing + 75 new including conftest fixtures)

## Task Commits

Each task was committed atomically (TDD flow):

1. **Task 1 RED: Failing tests** - `5ff2eaf` (test)
2. **Task 1 GREEN: Implementation** - `74f2d95` (feat)

## Files Created/Modified
- `llm_benchmark/backends/llamacpp.py` - LlamaCppBackend: chat, streaming, list_models, unload, warmup, detect_context_window, check_connectivity (286 lines)
- `llm_benchmark/backends/lmstudio.py` - LMStudioBackend: chat, streaming, list_models, unload, warmup, detect_context_window, check_connectivity (266 lines)
- `tests/test_llamacpp.py` - 25 tests covering all Backend Protocol methods and error handling (351 lines)
- `tests/test_lmstudio.py` - 25 tests covering all Backend Protocol methods and error handling (332 lines)
- `tests/conftest.py` - Added make_httpx_response factory, llamacpp_chat_response and lmstudio_chat_response fixtures
- `pyproject.toml` - Added httpx>=0.28.1 as explicit dependency

## Decisions Made
- Used httpx.Client with base_url per backend instance rather than a shared client -- follows existing OllamaBackend isolation pattern
- SSE streaming parsed via iter_lines with manual "data:" prefix parsing rather than an SSE library -- keeps dependencies minimal
- LM Studio eval_duration derived as eval_count / tokens_per_second -- best available approximation given LM Studio does not provide separate timing fields
- llama.cpp total_duration computed as sum of prompt_ms + predicted_ms since llama-server does not provide a separate total in the timings object

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mock httpx.Response factory missing request attribute**
- **Found during:** Task 1 GREEN (running tests)
- **Issue:** httpx.Response.raise_for_status() requires a request attribute to be set; our factory-created responses lacked it
- **Fix:** Added `request=httpx.Request("GET", "http://test")` to the factory; also fixed json/text parameter conflict
- **Files modified:** tests/conftest.py
- **Verification:** All 50 new tests pass
- **Committed in:** 74f2d95 (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for test infrastructure correctness. No scope creep.

## Issues Encountered
None beyond the mock factory fix documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both backend classes ready for detection/auto-start integration (Plan 02)
- create_backend() factory in __init__.py needs updating to support new backends (Plan 02 or 03)
- LM Studio stats field names are MEDIUM confidence -- should validate against real LM Studio server when available

---
*Phase: 06-new-backends*
*Completed: 2026-03-14*
