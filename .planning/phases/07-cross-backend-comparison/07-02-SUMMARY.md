---
phase: 07-cross-backend-comparison
plan: 02
subsystem: cli
tags: [argparse, interactive-menu, backend-comparison, cli-integration]

# Dependency graph
requires:
  - phase: 07-cross-backend-comparison
    provides: comparison.py with run_comparison, export functions, ComparisonResult model
  - phase: 06-new-backends
    provides: detect_backends(), get_install_instructions(), BackendStatus
provides:
  - "--backend all" CLI option triggering cross-backend comparison
  - Menu option 5 "Compare backends" in interactive menu
  - _mode_compare() with install hints for single-backend case
affects: [07-03-readme-docs]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy-import-comparison-in-cli, menu-mode-with-backend-detection]

key-files:
  created: []
  modified:
    - llm_benchmark/cli.py
    - llm_benchmark/menu.py
    - tests/test_cli.py
    - tests/test_menu.py

key-decisions:
  - "Comparison imports are lazy (inside --backend all branch) to avoid loading comparison.py for normal runs"
  - "Menu option 5 always returns backend='all' even with single backend -- lets run_comparison handle fallback"
  - "_mode_compare() shows install hints for missing backends when only 1 detected"

patterns-established:
  - "CLI comparison branch: detect backends, filter running, call run_comparison, export results"
  - "Menu compare mode: detect + hint + return Namespace with backend='all'"

requirements-completed: [COMP-01, COMP-05]

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 7 Plan 02: CLI and Menu Integration Summary

**--backend all CLI flag and menu option 5 wiring comparison.py into both user entry points with install hints**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-14T18:02:40Z
- **Completed:** 2026-03-14T18:10:55Z
- **Tasks:** 2 (both TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Added "all" to --backend choices and comparison branch in _handle_run
- Added menu option 5 "Compare backends" with backend detection and install hints
- Full TDD coverage: 6 new tests (3 CLI, 3 menu) all passing
- Full suite green: 324 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing CLI tests** - `190387c` (test)
2. **Task 1 GREEN: --backend all implementation** - `2eb3ea2` (feat)
3. **Task 2 RED: Failing menu tests** - `eb0cafc` (test)
4. **Task 2 GREEN: Menu option 5 implementation** - `4c17e66` (feat)

_Note: TDD tasks with RED and GREEN commits._

## Files Created/Modified
- `llm_benchmark/cli.py` - Added "all" to --backend choices, comparison branch in _handle_run
- `llm_benchmark/menu.py` - Added option 5, _mode_compare() with backend detection and install hints
- `tests/test_cli.py` - 3 new tests for --backend all parsing, comparison branch, error handling
- `tests/test_menu.py` - 3 new tests for option 5, single-backend hints, modes 1-4 regression

## Decisions Made
- Comparison imports are lazy (inside the --backend all branch) to avoid loading comparison.py for normal single-backend runs
- Menu option 5 always returns backend='all' Namespace even with single backend -- run_comparison handles the single-backend fallback gracefully
- _mode_compare() shows platform-specific install hints for each missing backend when only 1 is detected

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CLI and menu fully wired to comparison.py
- --backend all triggers full comparison flow (detect, benchmark, export)
- Menu option 5 provides guided experience with install hints
- Ready for 07-03 README documentation

## Self-Check: PASSED

- FOUND: llm_benchmark/cli.py
- FOUND: llm_benchmark/menu.py
- FOUND: tests/test_cli.py
- FOUND: tests/test_menu.py
- FOUND: commit 190387c (Task 1 RED)
- FOUND: commit 2eb3ea2 (Task 1 GREEN)
- FOUND: commit eb0cafc (Task 2 RED)
- FOUND: commit 4c17e66 (Task 2 GREEN)
- All 324 tests pass (full suite)

---
*Phase: 07-cross-backend-comparison*
*Completed: 2026-03-14*
