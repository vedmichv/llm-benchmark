---
phase: 03-advanced-benchmarking
plan: 01
subsystem: benchmarking
tags: [asyncio, ollama, concurrent, async-client, pydantic]

# Dependency graph
requires:
  - phase: 02-measurement-reliability
    provides: "BenchmarkResult, OllamaResponse models, runner.py warmup/unload"
provides:
  - "concurrent.py module with run_concurrent_batch and benchmark_model_concurrent"
  - "ConcurrentBatchResult, SweepConfigResult, SweepModelResult Pydantic models"
  - "DEFAULT_CONCURRENT, SWEEP_NUM_CTX, SWEEP_PROMPT config constants"
  - "auto_detect_concurrency resource-based default"
affects: [03-02, 03-04, cli]

# Tech tracking
tech-stack:
  added: [pytest-asyncio]
  patterns: [asyncio.gather for parallel requests, AsyncClient context manager, per-request try/except isolation]

key-files:
  created:
    - llm_benchmark/concurrent.py
    - tests/test_concurrent.py
  modified:
    - llm_benchmark/models.py
    - llm_benchmark/config.py
    - tests/conftest.py
    - pyproject.toml

key-decisions:
  - "AsyncClient used as async context manager for proper connection cleanup"
  - "Per-request try/except in _single_request rather than gather return_exceptions for cleaner error handling"
  - "aggregate_throughput_ts = sum(eval_count) / wall_time (not mean of rates)"
  - "auto_detect_concurrency thresholds: 8 (VRAM>=16), 4 (RAM>=32), 2 (default)"

patterns-established:
  - "Async concurrent pattern: asyncio.gather with per-coroutine error handling"
  - "SDK response conversion: hasattr model_dump pattern for ollama ChatResponse"

requirements-completed: [BENCH-03, BENCH-04]

# Metrics
duration: 7min
completed: 2026-03-13
---

# Phase 03 Plan 01: Concurrent Benchmarking Summary

**Async concurrent benchmarking module firing N parallel requests via asyncio.gather with wall-time aggregate throughput measurement and per-request failure isolation**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-13T08:37:10Z
- **Completed:** 2026-03-13T08:44:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- ConcurrentBatchResult, SweepConfigResult, SweepModelResult Pydantic models for structured concurrent/sweep results
- concurrent.py module with run_concurrent_batch (N parallel async requests) and benchmark_model_concurrent (full orchestration)
- Aggregate throughput correctly computed as sum(tokens)/wall_time, not mean of rates
- Failed requests isolated via per-request try/except -- other requests continue unaffected
- auto_detect_concurrency returns sensible 2/4/8 defaults based on RAM and VRAM
- 18 tests covering model validation, config constants, concurrency, failure isolation, throughput calculation

## Task Commits

Each task was committed atomically:

1. **Task 1: Data models and config constants** - `1656ae1` (feat)
2. **Task 2: Concurrent benchmarking module** - `451ec62` (feat)

_Note: TDD tasks -- tests written first (RED), then implementation (GREEN)_

## Files Created/Modified
- `llm_benchmark/concurrent.py` - Async concurrent benchmarking orchestration (run_concurrent_batch, benchmark_model_concurrent, auto_detect_concurrency)
- `llm_benchmark/models.py` - Added ConcurrentBatchResult, SweepConfigResult, SweepModelResult
- `llm_benchmark/config.py` - Added DEFAULT_CONCURRENT, SWEEP_NUM_CTX, SWEEP_PROMPT constants
- `tests/test_concurrent.py` - 18 tests for concurrent module
- `tests/conftest.py` - Added sample_benchmark_result and mock_async_client fixtures
- `pyproject.toml` - Added pytest-asyncio dev dependency

## Decisions Made
- Used AsyncClient as async context manager for proper connection cleanup
- Per-request try/except in _single_request rather than gather(return_exceptions=True) for cleaner BenchmarkResult construction
- aggregate_throughput_ts = sum(eval_count) / wall_time_s (correct total throughput, not mean of rates)
- auto_detect_concurrency thresholds: 8 for VRAM >= 16GB, 4 for RAM >= 32GB, 2 otherwise

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- concurrent.py ready for CLI integration (plan 03-04)
- ConcurrentBatchResult model available for exporters
- SweepConfigResult/SweepModelResult ready for parameter sweep plan (03-02)

---
*Phase: 03-advanced-benchmarking*
*Completed: 2026-03-13*
