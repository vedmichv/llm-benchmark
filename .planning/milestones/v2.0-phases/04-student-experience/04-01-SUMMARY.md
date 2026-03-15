---
phase: 04-student-experience
plan: 01
subsystem: ui
tags: [interactive-menu, bar-chart, unicode, rich, cli-ux]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "CLI framework, preflight checks, config module"
  - phase: 03-advanced-benchmarking
    provides: "Concurrent and sweep benchmark modes in cli.py"
provides:
  - "Interactive menu (menu.py) returning argparse.Namespace for 4 modes"
  - "Unicode bar chart display (display.py) with Rich and plain-text variants"
  - "No-args CLI path triggering interactive menu"
  - "Bar chart rendering after standard, concurrent, and sweep benchmarks"
  - "KeyboardInterrupt handling for partial result export"
affects: [04-student-experience]

# Tech tracking
tech-stack:
  added: []
  patterns: [interactive-input-with-eof-handling, unicode-bar-chart, lazy-import-for-menu]

key-files:
  created:
    - llm_benchmark/menu.py
    - llm_benchmark/display.py
  modified:
    - llm_benchmark/cli.py

key-decisions:
  - "Menu uses input() loop with EOFError/KeyboardInterrupt handling for clean exit"
  - "Quick test mode sorts models by size and picks smallest for ~30s run"
  - "Bar chart uses Unicode block characters (full/empty) for universal terminal support"
  - "render_text_bar_chart returns plain string without Rich markup for Markdown embedding"
  - "Concurrent bar chart uses average aggregate_throughput_ts across batches per model"

patterns-established:
  - "Interactive input pattern: _prompt_choice/_prompt_int with try/except for EOF"
  - "No-args detection before parse_args for menu fallback"

requirements-completed: [UX-01, UX-02, UX-03, UX-05]

# Metrics
duration: 3min
completed: 2026-03-13
---

# Phase 4 Plan 01: Interactive Menu & Bar Chart Summary

**Interactive numbered menu for no-args invocation with Unicode bar chart rankings after all benchmark modes**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-13T12:31:08Z
- **Completed:** 2026-03-13T12:34:29Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created menu.py with 4 modes: quick test, standard, full, and custom (user picks prompt set, runs, models)
- Created display.py with render_bar_chart (Rich) and render_text_bar_chart (plain text) using Unicode block characters
- Wired no-args detection in cli.py main() to launch interactive menu with preflight checks
- Added bar chart display after standard, concurrent, and sweep benchmark modes
- Added KeyboardInterrupt handling around standard benchmark loop for partial result saving

## Task Commits

Each task was committed atomically:

1. **Task 1: Create menu.py and display.py modules** - `2e2f32d` (feat)
2. **Task 2: Wire menu and bar chart into cli.py** - `3a5f1cf` (feat)

## Files Created/Modified
- `llm_benchmark/menu.py` - Interactive menu with 4 modes returning argparse.Namespace
- `llm_benchmark/display.py` - Unicode bar chart rendering (Rich + plain text)
- `llm_benchmark/cli.py` - No-args menu path, bar chart after all benchmark modes, KeyboardInterrupt handling

## Decisions Made
- Menu uses input() with try/except for EOFError/KeyboardInterrupt for clean terminal exit
- Quick test mode sorts models by m.size ascending, picks smallest, uses single short prompt
- Bar chart uses BAR_FULL/BAR_EMPTY Unicode blocks with proportional width relative to max rate
- Concurrent bar chart averages aggregate_throughput_ts across batches per model
- Sweep bar chart extracts response_ts from best_config per model

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Interactive menu and bar chart ready for student use
- All 118 existing tests pass with no regressions

---
*Phase: 04-student-experience*
*Completed: 2026-03-13*

## Self-Check: PASSED
