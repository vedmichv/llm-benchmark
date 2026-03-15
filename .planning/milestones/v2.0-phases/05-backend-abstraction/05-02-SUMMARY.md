---
phase: 05-backend-abstraction
plan: 02
subsystem: api
tags: [backend-abstraction, refactor, protocol-pattern, ollama, seconds-timing]

# Dependency graph
requires:
  - phase: 05-01
    provides: Backend Protocol, BackendResponse, BackendError, StreamResult, OllamaBackend, create_backend()
provides:
  - Backend-aware runner (all functions accept Backend parameter)
  - Backend-aware preflight checks (check_backend_connectivity, dict-based models)
  - Backend-aware system info (backend_name/backend_version fields)
  - Backend-aware exporters (seconds-based BackendResponse, no _ns_to_sec)
  - CLI integration creating and threading backend through all modules
affects: [05-03, 06-lm-studio, 06-llama-cpp]

# Tech tracking
tech-stack:
  added: []
  patterns: [Backend parameter threading through function signatures, dict-based model lists from backend.list_models()]

key-files:
  created: []
  modified:
    - llm_benchmark/models.py
    - llm_benchmark/runner.py
    - llm_benchmark/preflight.py
    - llm_benchmark/system.py
    - llm_benchmark/exporters.py
    - llm_benchmark/cli.py
    - llm_benchmark/compare.py
    - llm_benchmark/concurrent.py
    - llm_benchmark/sweep.py
    - llm_benchmark/menu.py
    - llm_benchmark/recommend.py
    - tests/conftest.py
    - tests/test_models.py
    - tests/test_runner.py
    - tests/test_exporters.py
    - tests/test_system.py
    - tests/test_preflight.py
    - tests/test_analyze.py

key-decisions:
  - "preflight returns list[dict] instead of list[OllamaModel] for backend-agnostic model lists"
  - "run_preflight_checks accepts optional backend param with auto-creation fallback for backward compat"
  - "get_system_info/format_system_summary accept optional backend param for backward compat (info subcommand)"
  - "concurrent.py and sweep.py keep local _ns_to_sec for raw ollama async responses until full Backend migration"

patterns-established:
  - "Backend parameter threading: all runner/preflight functions take Backend as first parameter"
  - "Dict-based model access: models from backend.list_models() accessed via m['model'], m['size']"
  - "Backward compatibility: optional backend param with fallback to create_backend()"

requirements-completed: [BACK-04]

# Metrics
duration: 12min
completed: 2026-03-14
---

# Phase 5 Plan 2: Core Pipeline Backend Migration Summary

**Migrated runner, preflight, system, and exporters to Backend abstraction with seconds-based timing and dict-based model lists**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-14T12:01:44Z
- **Completed:** 2026-03-14T12:13:18Z
- **Tasks:** 2
- **Files modified:** 18

## Accomplishments
- Removed OllamaResponse and _ns_to_sec from models.py; BenchmarkResult now uses BackendResponse
- All runner.py functions accept Backend parameter; retry logic uses BackendError.retryable
- SystemInfo uses backend_name/backend_version instead of ollama_version
- All exporters use BackendResponse seconds directly -- no nanosecond conversion
- Preflight uses backend.check_connectivity() and backend.list_models() returning dicts
- CLI creates backend via create_backend() and threads it through all modules
- All 170 tests pass with updated test fixtures using BackendResponse

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor models.py** - `b6018f6` (refactor)
2. **Task 2: Refactor runner, preflight, system, exporters** - `42b976c` (refactor)

## Files Created/Modified
- `llm_benchmark/models.py` - Removed OllamaResponse, _ns_to_sec; BenchmarkResult uses BackendResponse; SystemInfo has backend_name/backend_version
- `llm_benchmark/runner.py` - All functions accept Backend param; uses BackendError for retries; seconds-based compute_averages
- `llm_benchmark/preflight.py` - check_backend_connectivity(backend); check_available_models(backend) returns list[dict]
- `llm_benchmark/system.py` - get_system_info(backend) and format_system_summary(backend) with optional backend param
- `llm_benchmark/exporters.py` - Removed _ns_to_sec import; all duration fields read directly from BackendResponse seconds
- `llm_benchmark/cli.py` - Creates backend, passes to preflight/runner/system; handles dict-based models
- `llm_benchmark/compare.py` - Backward compat for both ollama_version and backend_name/backend_version in JSON
- `llm_benchmark/concurrent.py` - Fixed imports; local _ns_to_sec for raw async ollama responses
- `llm_benchmark/sweep.py` - Fixed imports; creates temporary backend for warmup/unload
- `llm_benchmark/menu.py` - Handles dict-based models from preflight
- `llm_benchmark/recommend.py` - Handles dict-based models from preflight
- `tests/conftest.py` - New sample_backend_response fixture; updated sample_benchmark_result
- `tests/test_models.py` - Updated for BackendResponse; test SystemInfo backend fields
- `tests/test_runner.py` - Updated all tests for Backend parameter; mock backend helper
- `tests/test_exporters.py` - Updated to use BackendResponse; SystemInfo with backend fields
- `tests/test_system.py` - Tests backend_name/backend_version fields
- `tests/test_preflight.py` - Tests use mock backend; dict-based models
- `tests/test_analyze.py` - Updated fixture for backend_name/backend_version

## Decisions Made
- Preflight returns `list[dict]` from `backend.list_models()` instead of SDK model objects. This makes the interface backend-agnostic. Callers access `m['model']` and `m['size']`.
- `run_preflight_checks` accepts optional `backend` param with `create_backend()` fallback, so existing callers (interactive menu, recommend) work without changes.
- `get_system_info()` and `format_system_summary()` accept optional `backend` param. Without backend, falls back to "ollama" + `get_ollama_version()` for `info` subcommand backward compatibility.
- `concurrent.py` and `sweep.py` retain local `_ns_to_sec` and raw `ollama` SDK usage. These modules use the async client (concurrent) and `ollama.Options` (sweep) which haven't been abstracted yet. Full Backend migration will happen in a future plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed concurrent.py and sweep.py import errors**
- **Found during:** Task 2 (test execution)
- **Issue:** concurrent.py and sweep.py imported OllamaResponse and _ns_to_sec from models.py which were removed in Task 1
- **Fix:** Updated concurrent.py to build BackendResponse directly from raw ollama async responses with local _ns_to_sec. Updated sweep.py to use local _ns_to_sec and create temporary backend for warmup/unload.
- **Files modified:** llm_benchmark/concurrent.py, llm_benchmark/sweep.py
- **Verification:** All 170 tests pass
- **Committed in:** 42b976c

**2. [Rule 3 - Blocking] Updated cli.py, menu.py, recommend.py for dict-based models**
- **Found during:** Task 2 (integration with refactored preflight)
- **Issue:** cli.py accessed `model.model` on objects from preflight, but preflight now returns dicts. menu.py and recommend.py had same issue.
- **Fix:** Updated all callers to use `model['model']` and `model['size']` dict access, with isinstance checks for backward compat in menu/recommend.
- **Files modified:** llm_benchmark/cli.py, llm_benchmark/menu.py, llm_benchmark/recommend.py
- **Verification:** All 170 tests pass
- **Committed in:** 42b976c

**3. [Rule 3 - Blocking] Updated compare.py for backward compatibility**
- **Found during:** Task 2 (integration review)
- **Issue:** compare.py referenced sys_info.get('ollama_version') which no longer exists in new JSON exports
- **Fix:** Updated to check both backend_name/backend_version (new) and ollama_version (old) for backward compat with existing JSON result files.
- **Files modified:** llm_benchmark/compare.py
- **Verification:** All 170 tests pass
- **Committed in:** 42b976c

---

**Total deviations:** 3 auto-fixed (3 blocking)
**Impact on plan:** All fixes were necessary to maintain importability and correct function. No scope creep -- concurrent/sweep retain raw ollama usage for now.

## Issues Encountered
None beyond the blocking import/access fixes documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All core pipeline modules (runner, preflight, system, exporters) use Backend abstraction
- Zero `import ollama` in runner.py, preflight.py, system.py, exporters.py
- Zero `OllamaResponse` references outside backends/ollama.py
- Zero `_ns_to_sec` in core modules (only in backends/ollama.py, concurrent.py, sweep.py)
- Ready for plan 05-03 (CLI wiring and final integration)
- concurrent.py and sweep.py still use raw ollama SDK -- full Backend migration deferred

---
*Phase: 05-backend-abstraction*
*Completed: 2026-03-14*
