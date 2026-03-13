---
phase: quick
plan: 2
subsystem: preflight
tags: [shutil, subprocess, interactive-install, ollama]

requires:
  - phase: 01-foundation
    provides: preflight.py with connectivity/model/RAM checks
provides:
  - check_ollama_installed() function with interactive install flow
  - Ollama binary detection via shutil.which before any API calls
affects: [preflight, cli-startup]

tech-stack:
  added: []
  patterns: [interactive-install-prompt, platform-specific-commands]

key-files:
  created: []
  modified:
    - llm_benchmark/preflight.py
    - tests/test_preflight.py

key-decisions:
  - "Use shutil.which for binary detection (stdlib, cross-platform)"
  - "try/except EOFError+KeyboardInterrupt on input() treats interrupts as decline"
  - "Post-install verification re-checks shutil.which to confirm success"

patterns-established:
  - "Installation check pattern: detect binary -> offer install -> verify"

requirements-completed: [quick-2]

duration: 2min
completed: 2026-03-13
---

# Quick Task 2: Add Ollama Installation Check Summary

**shutil.which-based Ollama binary detection with platform-specific interactive auto-install before preflight checks**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-13T15:25:50Z
- **Completed:** 2026-03-13T15:27:52Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Added check_ollama_installed() that detects Ollama binary via shutil.which
- Platform-specific install commands: curl for Darwin/Linux, irm for Windows
- Interactive y/N prompt with auto-install via subprocess and post-install verification
- Integrated as step 0 in run_preflight_checks, before connectivity check
- 7 new tests covering all scenarios, all 17 preflight tests passing

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: Failing tests** - `32a9ef1` (test)
2. **Task 1 GREEN: Implementation** - `f48ddfe` (feat)

## Files Created/Modified
- `llm_benchmark/preflight.py` - Added check_ollama_installed() and integrated into run_preflight_checks
- `tests/test_preflight.py` - Added TestOllamaInstallation class (7 tests), updated existing tests to mock new function

## Decisions Made
- Used shutil.which for binary detection (stdlib, cross-platform, no external deps)
- try/except on input() for EOFError/KeyboardInterrupt treats interrupts as decline (safe default)
- Post-install verification re-checks shutil.which to confirm binary is on PATH

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ollama installation check fully integrated into preflight flow
- Students without Ollama get helpful prompt instead of cryptic connection error

---
*Quick task: 2*
*Completed: 2026-03-13*
