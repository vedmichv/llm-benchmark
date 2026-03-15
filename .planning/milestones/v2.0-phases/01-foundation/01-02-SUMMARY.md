---
phase: 01-foundation
plan: 02
subsystem: benchmark-engine
tags: [ollama, threading, timeout, system-info, json-export, csv-export, markdown-export, rich]

# Dependency graph
requires:
  - phase: 01-foundation-01
    provides: "Pydantic data models (BenchmarkResult, OllamaResponse, ModelSummary, SystemInfo), config.py console singleton, compute_averages()"
provides:
  - "Benchmark runner with threading-based timeouts and model offloading"
  - "Cross-platform system info detection (CPU, RAM, GPU) without psutil"
  - "JSON, CSV, Markdown result exporters with timestamped filenames"
  - "Results comparison module with rich tables"
  - "compute_averages() in runner.py with prompt-cache exclusion"
affects: [01-03, 02-01]

# Tech tracking
tech-stack:
  added: []
  patterns: [threading-timeout, keep-alive-0-offloading, correct-rate-averaging-with-cache-exclusion]

key-files:
  created:
    - llm_benchmark/runner.py
    - llm_benchmark/system.py
    - llm_benchmark/exporters.py
    - llm_benchmark/compare.py
    - tests/test_runner.py
    - tests/test_system.py
  modified: []

key-decisions:
  - "Runner compute_averages() excludes prompt_cached results from prompt_eval calculations (extends STAB-04)"
  - "Apple Silicon GPU detection returns integrated GPU label (no separate VRAM since unified memory)"
  - "Exporters use flat _result_to_dict helper for JSON serialization of BenchmarkResult"

patterns-established:
  - "Threading timeout: Thread(daemon=True) + join(timeout) + is_alive() check for cross-platform timeouts"
  - "Model offloading: ollama.generate(model=name, prompt='', keep_alive=0) -- no sudo, no subprocess"
  - "Exporter pattern: _ensure_dir + _timestamp + write_text/csv.writer for all export formats"

requirements-completed: [STAB-01, STAB-04, STAB-05, STAB-06]

# Metrics
duration: 8min
completed: 2026-03-12
---

# Phase 1 Plan 02: Core Engine Summary

**Benchmark runner with threading timeouts (STAB-06), keep_alive=0 offloading (STAB-05), correct averaging with cache exclusion (STAB-04), cross-platform system info (STAB-01), and JSON/CSV/Markdown exporters**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-12T19:44:21Z
- **Completed:** 2026-03-12T19:52:01Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Benchmark runner with run_single_benchmark(), streaming support, and TimeoutError handling
- Threading-based timeout (run_with_timeout) -- no signal.SIGALRM, works cross-platform
- Model offloading via ollama.generate(keep_alive=0) -- no sudo needed
- compute_averages() with correct sum_tokens/sum_time and prompt-cache exclusion
- Cross-platform system info: CPU (sysctl/proc/wmic), RAM, GPU (nvidia-smi/Apple Silicon), Ollama version
- Three export formats (JSON, CSV, Markdown) with timestamped filenames and auto-created output dirs
- Compare module with rich Table output migrated from compare_results.py
- 16 tests passing across test_runner.py and test_system.py

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for runner** - `907d398` (test)
2. **Task 1 GREEN: Implement runner.py** - `4c1d182` (feat)
3. **Task 2: system.py, exporters.py, compare.py, test_system.py** - `162bf3c` (feat)

_Note: Task 1 followed TDD with RED and GREEN commits._

## Files Created/Modified
- `llm_benchmark/runner.py` - Benchmark execution, timeout wrapper, offloading, averaging
- `llm_benchmark/system.py` - Cross-platform hardware/system info collection
- `llm_benchmark/exporters.py` - JSON, CSV, Markdown result writers
- `llm_benchmark/compare.py` - Results comparison with rich tables
- `tests/test_runner.py` - 9 tests: averaging, caching, timeout, offloading, benchmark success/failure
- `tests/test_system.py` - 7 tests: system info fields, format_system_summary

## Decisions Made
- Runner's compute_averages() extends STAB-04 by also excluding prompt_cached results from prompt_eval calculations (the models.py version doesn't do this)
- Apple Silicon GPU detection returns "{chip} (integrated GPU)" since there's no separate VRAM to report
- Exporters use a shared _result_to_dict() helper for consistent JSON serialization
- Docstring wording avoids mentioning "SIGALRM" by name (test asserts no occurrence in source)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed SIGALRM test false positive from docstring**
- **Found during:** Task 1 GREEN (runner.py implementation)
- **Issue:** Docstring contained "SIGALRM" string which triggered test_timeout_no_sigalrm assertion
- **Fix:** Reworded docstring to avoid the literal string while preserving intent
- **Files modified:** llm_benchmark/runner.py
- **Verification:** test_timeout_no_sigalrm passes
- **Committed in:** 4c1d182 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial wording fix. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Runner, system info, exporters, and compare modules all importable and tested
- Ready for Plan 01-03 (CLI + preflight checks) which will wire these modules together
- benchmark_model() ready to be called from CLI handler

## Self-Check: PASSED

All 6 created files verified. All 3 commit hashes verified (907d398, 4c1d182, 162bf3c).

---
*Phase: 01-foundation*
*Completed: 2026-03-12*
