---
phase: 06-new-backends
plan: 05
subsystem: ui
tags: [interactive-menu, backend-selection, gguf, failure-summary, rich-table]

requires:
  - phase: 06-02
    provides: detect_backends() and BackendStatus for backend discovery
  - phase: 06-03
    provides: create_backend() factory and backend-aware preflight
  - phase: 06-04
    provides: KNOWN_ISSUES dict for failure hint display
provides:
  - select_backend_interactive() for no-args menu backend selection
  - GGUF file scanner and metadata extraction for llama-cpp
  - Failure summary table replacing interactive error prompts
  - Backend field propagated through menu Namespace to _handle_run
affects: [07-docs]

tech-stack:
  added: []
  patterns: [struct-based GGUF header parsing, auto-skip-with-summary error handling]

key-files:
  created: []
  modified:
    - llm_benchmark/menu.py
    - llm_benchmark/cli.py
    - tests/test_menu.py

key-decisions:
  - "select_backend_interactive() is separate from run_interactive_menu() -- keeps menu signature stable"
  - "GGUF metadata extraction uses simple struct parser reading general.name key"
  - "Failure summary auto-skips failed models and shows Rich table at end instead of per-model prompt"

patterns-established:
  - "Backend selection before preflight in no-args path: select -> create -> preflight -> menu"
  - "Error handling pattern: auto-skip + collect failures + summary table with known-issue hints"

requirements-completed: [CLI-02, CLI-05]

duration: 3min
completed: 2026-03-14
---

# Phase 6 Plan 5: Interactive Menu Backend Selection Summary

**Backend selection menu with GGUF scanner, auto-select for single backends, and failure summary table replacing per-model error prompts**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T17:05:28Z
- **Completed:** 2026-03-14T17:08:28Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Interactive backend selection with status display (running/installed/not found) shown before mode selection
- GGUF file scanner with metadata extraction for llama-cpp model selection
- Auto-select behavior when only one backend detected, with informational note about alternatives
- Failure summary table with known-issue hints replaces interactive "Continue?" prompt

## Task Commits

Each task was committed atomically:

1. **Task 1: Add backend selection to interactive menu** - `ca7672d` (feat)

**Plan metadata:** [pending] (docs: complete plan)

## Files Created/Modified
- `llm_benchmark/menu.py` - Added select_backend_interactive(), scan_gguf_files(), extract_gguf_model_name(), _select_gguf_model(); updated _build_namespace with backend/model_path fields; updated mode functions to pass backend through
- `llm_benchmark/cli.py` - Added _print_failure_summary(); updated no-args path to call select_backend_interactive() before preflight; replaced interactive error prompt with auto-skip + failure summary
- `tests/test_menu.py` - Added 17 new tests: backend selection (4), GGUF scanning (5), build namespace backend field (3), backend propagation (1), failure summary (2), filename cleaning (1), mode propagation (1)

## Decisions Made
- Kept run_interactive_menu() signature accepting backend and models -- added select_backend_interactive() as a separate function called in cli.py before create_backend()
- GGUF metadata extraction reads the GGUF header struct directly (magic, version, kv pairs) looking for general.name key, with fallback to cleaned filename
- Failure summary uses Rich Table with model name, truncated error, and known-issue hint columns
- Auto-skip on error (no interactive "Continue?" prompt) -- all failures collected and displayed as summary at end

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 6 (New Backends) is now complete -- all 5 plans executed
- Interactive menu flow: backend selection -> preflight -> system summary -> model downloads -> mode selection
- Ready for Phase 7 (docs/comparison features)

---
*Phase: 06-new-backends*
*Completed: 2026-03-14*
