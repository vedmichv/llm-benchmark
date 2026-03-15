---
phase: 06-new-backends
plan: 04
subsystem: exporters, system
tags: [export, backend-aware, filenames, system-info, detection]

# Dependency graph
requires:
  - phase: 06-01
    provides: LlamaCppBackend and LMStudioBackend with .name property
  - phase: 06-02
    provides: detect_backends() and BackendStatus for inventory display
provides:
  - Backend-aware export filenames (benchmark_{backend}_{timestamp}.ext)
  - JSON metadata with backend_name and backend_version
  - Markdown report headers with backend label
  - System summary with full backend detection inventory
  - Backend inventory display for info command
  - Known-issues hint table for common backend errors
affects: [06-05, cli, exporters]

# Tech tracking
tech-stack:
  added: []
  patterns: [_filename helper for backend-aware naming, lazy-import detect_backends in system.py]

key-files:
  created: []
  modified:
    - llm_benchmark/exporters.py
    - llm_benchmark/runner.py
    - llm_benchmark/system.py
    - llm_benchmark/cli.py
    - tests/test_exporters.py
    - tests/test_system.py

key-decisions:
  - "_filename() helper centralizes backend-aware filename generation for all export functions"
  - "detect_backends() lazy-imported in format_system_summary to avoid circular deps"
  - "KNOWN_ISSUES dict maps (backend_name, error_substring) to hints"

patterns-established:
  - "_filename(prefix, ext, system_info) pattern for all export filename generation"
  - "get_known_issue_hint() for backend-specific error messages"

requirements-completed: [CLI-03, CLI-04]

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 6 Plan 4: Backend-Aware Exports and System Inventory Summary

**Backend name in export filenames, JSON metadata, Markdown headers; system summary with color-coded backend detection inventory**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T16:57:40Z
- **Completed:** 2026-03-14T17:02:42Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Export filenames now follow `{prefix}_{backend}_{timestamp}.{ext}` pattern across all modes (standard, concurrent, sweep)
- JSON output includes backend_name and backend_version in system_info metadata
- Markdown reports show backend name in title and header line
- format_system_summary() displays all detected backends with running/installed/not-found color coding
- Info command shows full backend inventory with install instructions for missing backends
- Known-issues hint table provides actionable suggestions on common backend failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Add backend name to export filenames and metadata** - `15e1d1e` (feat)
2. **Task 2: Enhance system summary with backend inventory** - `dd82bd0` (feat)

## Files Created/Modified
- `llm_benchmark/exporters.py` - Added _filename() helper, backend-aware filenames for all 9 export functions, backend label in Markdown headers
- `llm_benchmark/runner.py` - Added KNOWN_ISSUES hint table, get_known_issue_hint(), failure summary count
- `llm_benchmark/system.py` - Added backend detection section to format_system_summary(), new get_backend_inventory() function
- `llm_benchmark/cli.py` - Updated _handle_info() to display backend inventory
- `tests/test_exporters.py` - 12 new tests for filenames, metadata, Markdown headers, known-issues hints
- `tests/test_system.py` - 7 new tests for backend detection scenarios and inventory display

## Decisions Made
- Centralized filename generation in `_filename()` helper to avoid duplication across 9 export functions
- Used lazy import for `detect_backends()` in `format_system_summary()` to avoid circular dependencies
- KNOWN_ISSUES dict uses (backend_name, error_substring) tuple keys for efficient pattern matching
- Concurrent exports renamed from `benchmark_*.json` to `concurrent_*.json` for clarity

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend-aware exports fully functional for all 3 modes (standard, concurrent, sweep)
- System inventory ready for Plan 05 integration testing
- 291 tests pass with zero regressions

---
*Phase: 06-new-backends*
*Completed: 2026-03-14*
