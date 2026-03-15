---
phase: 06-new-backends
plan: 03
subsystem: cli
tags: [cli, preflight, factory-pattern, backend-selection, argparse]

# Dependency graph
requires:
  - phase: 06-01
    provides: "LlamaCppBackend and LMStudioBackend classes"
  - phase: 06-02
    provides: "detect_backends(), auto_start_backend(), get_install_instructions()"
provides:
  - "create_backend() factory supporting ollama, llama-cpp, lm-studio with host/port"
  - "--backend, --port, --model-path CLI flags on run and info subcommands"
  - "Backend-generic preflight: install check via detection module, backend-specific messages"
  - "Auto-start prompt when backend installed but not running"
affects: [06-04-runner, 06-05-exporters]

# Tech tracking
tech-stack:
  added: []
  patterns: [factory-with-kwargs, backend-aware-preflight, module-level-detection-imports]

key-files:
  created: []
  modified:
    - llm_benchmark/backends/__init__.py
    - llm_benchmark/cli.py
    - llm_benchmark/preflight.py
    - tests/test_cli.py
    - tests/test_preflight.py

key-decisions:
  - "create_backend() passes host/port as kwargs only when provided (preserves backend defaults)"
  - "model_path stored as _model_path attribute on backend instance for preflight access"
  - "check_ollama_installed() kept as backward-compatible alias delegating to check_backend_installed"
  - "Detection module imports at module level in preflight.py for clean test mocking"

patterns-established:
  - "Backend selection: --backend flag with choices list, default ollama, zero behavior change for existing users"
  - "Preflight chain: install (blocking) -> connectivity (blocking) -> models (blocking) -> RAM (advisory)"

requirements-completed: [BEND-05, CLI-01, PLAT-01]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 6 Plan 03: CLI & Preflight Integration Summary

**Extended create_backend() factory to three backends, added --backend/--port/--model-path CLI flags, and generalized preflight with detection-based install checks and backend-specific messages**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T16:57:48Z
- **Completed:** 2026-03-14T17:01:27Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- create_backend() supports ollama, llama-cpp, lm-studio with optional host/port parameters
- CLI --backend flag accepted by run and info subcommands with correct backend creation
- Preflight install check uses detection module for any backend (not hardcoded Ollama)
- Backend-specific error messages in connectivity check and model hints
- Auto-start prompt when backend installed but not running
- 284 total tests pass (38 new tests added)

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend create_backend() and add CLI flags** - `b4cf7f0` (feat)
2. **Task 2: Generalize preflight checks for any backend** - `8d93b43` (feat)

## Files Created/Modified
- `llm_benchmark/backends/__init__.py` - Extended create_backend() with llama-cpp and lm-studio branches, host/port kwargs
- `llm_benchmark/cli.py` - Added --backend, --port, --model-path flags; updated description; backend-aware _handle_run and _handle_info
- `llm_benchmark/preflight.py` - Replaced check_ollama_installed with check_backend_installed using detection module; backend-specific messages throughout
- `tests/test_cli.py` - 18 new tests for backend flags and create_backend factory
- `tests/test_preflight.py` - 26 tests rewritten for backend-generic preflight (install, connectivity, model hints)

## Decisions Made
- create_backend() builds kwargs dict and passes only non-None host/port to backend constructors, preserving their defaults
- model_path stored as _model_path attribute on backend instance rather than passing through factory (keeps factory signature clean)
- check_ollama_installed() preserved as backward-compatible alias for any code calling it directly
- Detection module (detect_backends, auto_start_backend, get_install_instructions) imported at module level in preflight.py for clean test mocking per project convention

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CLI integration complete: `python -m llm_benchmark run --backend llama-cpp --model-path /path/to/model.gguf` parses correctly
- Preflight chain works for all three backends with appropriate messages
- Ready for runner integration (Plan 04) and export format updates (Plan 05)

---
*Phase: 06-new-backends*
*Completed: 2026-03-14*
