---
phase: 01-foundation
plan: 03
subsystem: cli-entrypoints
tags: [argparse, preflight, rich, cross-platform, cleanup]

# Dependency graph
requires:
  - phase: 01-foundation-01
    provides: "Package scaffold, Pydantic models, config.py console singleton, prompts.py"
  - phase: 01-foundation-02
    provides: "Runner (benchmark_model, unload_model), system.py, exporters.py, compare.py"
provides:
  - "CLI entry point with run/compare/info subcommands via argparse"
  - "Pre-flight checks: Ollama connectivity (blocking), model availability (blocking), RAM warnings (advisory)"
  - "__main__.py global exception handler with --debug support"
  - "Thin run.py convenience wrapper (6 lines)"
  - "Old files removed (9 files, -2386 lines)"
affects: [02-01]

# Tech tracking
tech-stack:
  added: []
  patterns: [preflight-chain-pattern, cli-subcommand-dispatch, lazy-imports-in-handlers]

key-files:
  created:
    - llm_benchmark/preflight.py
    - llm_benchmark/cli.py
    - llm_benchmark/__main__.py
    - tests/test_preflight.py
    - tests/test_cli.py
  modified:
    - run.py
    - CLAUDE.md

key-decisions:
  - "Preflight _get_system_ram_gb() duplicates logic from system.py to avoid circular import; acceptable since it's a private helper"
  - "CLI handlers use lazy imports (inside function bodies) to keep startup fast and avoid circular imports"
  - "run.py uses subprocess.call to delegate to python -m llm_benchmark (no direct import)"

patterns-established:
  - "Preflight chain: connectivity (blocking) -> models (blocking) -> RAM (warning only)"
  - "CLI dispatch: _HANDLERS dict maps subcommand name to handler function"
  - "Lazy imports in CLI handlers: heavy modules imported inside function bodies"

requirements-completed: [STAB-02, STAB-03]

# Metrics
duration: 5min
completed: 2026-03-12
---

# Phase 1 Plan 03: CLI & Preflight Summary

**Argparse CLI with run/compare/info subcommands, Ollama connectivity and RAM pre-flight checks, old file cleanup (9 files removed)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-12T19:55:35Z
- **Completed:** 2026-03-12T20:00:18Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments
- Pre-flight checks with platform-specific Ollama guidance (STAB-02) and RAM warnings (STAB-03)
- CLI with run/compare/info subcommands, --debug, --verbose, --skip-checks, --prompt-set, --prompts flags
- Global exception handler in __main__.py (friendly errors by default, stack traces with --debug)
- Removed 9 old files (-2386 lines): benchmark.py, extended_benchmark.py, compare_results.py, run.sh/bat/ps1, setup_passwordless_sudo.sh, requirements.txt, test_ollama.py
- CLAUDE.md updated to reflect new package structure
- 42 total tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for preflight checks** - `ea554cb` (test)
2. **Task 1 GREEN: Implement preflight.py** - `c69e950` (feat)
3. **Task 2: CLI, entry points, old file cleanup, CLAUDE.md** - `b08d121` (feat)

_Note: Task 1 followed TDD with RED and GREEN commits._

## Files Created/Modified
- `llm_benchmark/preflight.py` - Ollama connectivity, model availability, RAM warning checks
- `llm_benchmark/cli.py` - Argparse CLI with run/compare/info subcommands
- `llm_benchmark/__main__.py` - Global exception handler entry point
- `tests/test_preflight.py` - 10 tests for preflight checks
- `tests/test_cli.py` - 7 tests for CLI parsing and flags
- `run.py` - Thin convenience wrapper (6 lines)
- `CLAUDE.md` - Updated to reflect new package structure and commands

## Decisions Made
- Preflight's _get_system_ram_gb() duplicates RAM detection from system.py to keep preflight self-contained and avoid circular imports
- CLI handlers use lazy imports inside function bodies for fast startup and to avoid circular import chains
- run.py delegates via subprocess.call rather than direct import to keep it truly thin

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 Foundation complete: all 3 plans executed
- Full CLI tool operational: `python -m llm_benchmark run` is the single entry point
- All stability requirements (STAB-01 through STAB-06) and quality requirements (QUAL-01, QUAL-02, QUAL-05) addressed
- Ready for Phase 2 (UX & Output)

## Self-Check: PASSED

All 7 created/modified files verified. All 3 commit hashes verified (ea554cb, c69e950, b08d121).

---
*Phase: 01-foundation*
*Completed: 2026-03-12*
