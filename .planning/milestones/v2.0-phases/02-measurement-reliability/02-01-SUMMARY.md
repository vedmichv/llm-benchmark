---
phase: 02-measurement-reliability
plan: 01
subsystem: runner
tags: [warmup, retry, tenacity, exponential-backoff, ollama]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: runner.py with run_single_benchmark and benchmark_model
provides:
  - warmup_model() function for pre-loading models before measurement
  - Tenacity-based retry with exponential backoff in run_single_benchmark
  - --skip-warmup and --max-retries CLI flags
affects: [02-measurement-reliability]

# Tech tracking
tech-stack:
  added: [tenacity]
  patterns: [dynamic-tenacity-retryer, warmup-before-measurement]

key-files:
  created: []
  modified:
    - llm_benchmark/config.py
    - llm_benchmark/runner.py
    - llm_benchmark/cli.py
    - tests/test_runner.py
    - tests/test_cli.py

key-decisions:
  - "Import ollama RequestError/ResponseError at module level to avoid mock interference in _is_retryable"
  - "Retry wraps outside timeout so each attempt gets full timeout budget"
  - "Dynamic tenacity retryer (not decorator) since max_retries is runtime config"

patterns-established:
  - "Dynamic tenacity retry: build retryer at call time for runtime-configurable retry count"
  - "Warmup pattern: short prompt to pre-load model, excluded from measurements"

requirements-completed: [BENCH-01, BENCH-02]

# Metrics
duration: 5min
completed: 2026-03-12
---

# Phase 02 Plan 01: Warmup & Retry Summary

**warmup_model() pre-loads models before benchmarking, tenacity-based retry with exponential backoff for transient Ollama failures**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-12T20:35:23Z
- **Completed:** 2026-03-12T20:40:14Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- warmup_model() sends short prompt to pre-load model, prints status, handles failures gracefully
- run_single_benchmark() retries transient errors (ConnectionError, TimeoutError, 5xx) with exponential backoff
- Non-retryable errors (4xx ResponseError) fail immediately without retry
- --skip-warmup and --max-retries CLI flags parsed and passed through to runner

## Task Commits

Each task was committed atomically:

1. **Task 1: Add warmup and retry logic to runner with tests**
   - `1f7f273` (test) - RED: failing tests for warmup and retry
   - `d2b56f8` (feat) - GREEN: implementation passes all tests
2. **Task 2: Add --skip-warmup and --max-retries CLI flags with tests**
   - `0bc5a98` (feat) - RED+GREEN combined: CLI flags and tests

_Note: TDD tasks have test-first commits followed by implementation commits._

## Files Created/Modified
- `llm_benchmark/config.py` - Added DEFAULT_MAX_RETRIES=3 and DEFAULT_WARMUP_PROMPT="Hello" constants
- `llm_benchmark/runner.py` - Added warmup_model(), _is_retryable(), retry logic in run_single_benchmark, warmup integration in benchmark_model
- `llm_benchmark/cli.py` - Added --skip-warmup and --max-retries flags, pass through to benchmark_model
- `tests/test_runner.py` - TestWarmupModel, TestRetryLogic, TestBenchmarkModelWarmup test classes (14 new tests)
- `tests/test_cli.py` - TestSkipWarmup, TestMaxRetries test classes (5 new tests)

## Decisions Made
- Imported ollama.RequestError and ollama.ResponseError at module level (as _RequestError, _ResponseError) to avoid isinstance() failures when ollama module is mocked in tests
- Used dynamic tenacity retryer (not @retry decorator) since max_retries is a runtime parameter
- Retry wraps outside run_with_timeout so each retry attempt gets a fresh full timeout budget

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed isinstance() failure with mocked ollama module**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** When tests mock `llm_benchmark.runner.ollama`, `ollama.RequestError` becomes a MagicMock, causing `isinstance()` to raise TypeError
- **Fix:** Import RequestError/ResponseError at module level as _RequestError/_ResponseError before mock can intercept
- **Files modified:** llm_benchmark/runner.py
- **Verification:** All 23 runner tests pass
- **Committed in:** d2b56f8

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for testability. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Warmup and retry infrastructure complete
- Ready for Plan 02 (statistical outlier detection / measurement quality)
- All 61 tests pass across entire test suite

---
*Phase: 02-measurement-reliability*
*Completed: 2026-03-12*

## Self-Check: PASSED
