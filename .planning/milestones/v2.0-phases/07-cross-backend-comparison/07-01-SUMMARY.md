---
phase: 07-cross-backend-comparison
plan: 01
subsystem: comparison
tags: [pydantic, rich-table, bar-chart, json-export, markdown-export, gguf-matching]

# Dependency graph
requires:
  - phase: 06-new-backends
    provides: Backend Protocol, create_backend(), detect_backends(), benchmark_model()
  - phase: 05-backend-abstraction
    provides: Backend Protocol, BackendResponse, runner infrastructure
provides:
  - ComparisonResult and BackendModelResult Pydantic models
  - run_comparison() orchestration across multiple backends
  - render_comparison_bar_chart() for single-model display
  - render_comparison_matrix() Rich table for multi-model display
  - export_comparison_json() and export_comparison_markdown() for reports
  - match_gguf_to_ollama_name() for GGUF-to-Ollama name matching
affects: [07-02-cli-backend-all, 07-03-menu-and-docs]

# Tech tracking
tech-stack:
  added: []
  patterns: [comparison-orchestration, cross-backend-matrix-display]

key-files:
  created:
    - llm_benchmark/comparison.py
    - tests/test_comparison.py
  modified: []

key-decisions:
  - "Module-level imports for create_backend, benchmark_model, etc. (no circular deps, enables clean test mocking)"
  - "ComparisonResult uses flat results list with BackendModelResult entries (not nested dict)"
  - "Winner metric is avg_response_ts (eval_count/eval_duration) per user decision"

patterns-established:
  - "Comparison orchestration: sequential backend loop with preflight + benchmark + unload per backend"
  - "Matrix display: Rich Table with dynamic backend columns, star on per-row winner, overall summary line"

requirements-completed: [COMP-01, COMP-02, COMP-03, COMP-04]

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 7 Plan 01: Comparison Module Summary

**Cross-backend comparison engine with sequential orchestration, bar chart and matrix display, JSON/Markdown export, and GGUF name matching**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-14T17:50:19Z
- **Completed:** 2026-03-14T17:54:40Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Created self-contained comparison.py module with all cross-backend comparison logic
- Pydantic data models (ComparisonResult, BackendModelResult) for structured comparison data
- Orchestration function that loops backends sequentially, runs preflight + benchmarks, computes winners
- Display functions following existing display.py patterns (bar chart + Rich matrix table)
- JSON and Markdown export functions following existing exporters.py patterns
- GGUF-to-Ollama fuzzy name matching for llama-cpp backend support

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `63730a0` (test)
2. **Task 1 GREEN: Implementation** - `602bccc` (feat)

_Note: TDD task with RED and GREEN commits._

## Files Created/Modified
- `llm_benchmark/comparison.py` - Comparison orchestration, display, and export (348 lines)
- `tests/test_comparison.py` - 10 unit tests covering orchestration, display, export, GGUF matching

## Decisions Made
- Used module-level imports for create_backend, benchmark_model, run_preflight_checks, unload_model since no circular dependency exists -- enables clean mock patching in tests
- ComparisonResult uses a flat list of BackendModelResult instead of nested dict for Pydantic serialization simplicity
- Winner metric is avg_response_ts (response tokens per second) per CONTEXT.md decision

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- comparison.py module ready for CLI integration (07-02: --backend all)
- comparison.py module ready for menu integration (07-02 or 07-03: option 5)
- All display and export functions ready to be wired into existing CLI handler

## Self-Check: PASSED

- FOUND: llm_benchmark/comparison.py
- FOUND: tests/test_comparison.py
- FOUND: commit 63730a0 (RED)
- FOUND: commit 602bccc (GREEN)
- All 318 tests pass (full suite)

---
*Phase: 07-cross-backend-comparison*
*Completed: 2026-03-14*
