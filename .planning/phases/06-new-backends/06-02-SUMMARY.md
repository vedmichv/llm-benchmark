---
phase: 06-new-backends
plan: 02
subsystem: infra
tags: [detection, auto-start, socket, subprocess, shutil, httpx]

requires:
  - phase: 05-backend-abstraction
    provides: BackendError exception class, Backend Protocol
provides:
  - BackendStatus dataclass for installed/running state
  - detect_backends() for all three backends
  - auto_start_backend() with health polling
  - get_install_instructions() per-OS commands
affects: [06-03, 06-04, 06-05]

tech-stack:
  added: [httpx]
  patterns: [shutil.which for binary detection, socket.connect_ex for port probing, subprocess.Popen for server management]

key-files:
  created:
    - llm_benchmark/backends/detection.py
    - tests/test_detection.py
  modified: []

key-decisions:
  - "httpx imported at module level (not locally) to enable clean test mocking"
  - "detect_backends skips port probe if binary not installed (running=False when not installed)"

patterns-established:
  - "Backend detection triple: (name, binary, default_port) in _BACKENDS constant"
  - "Health endpoint registry: _HEALTH_ENDPOINTS dict for per-backend polling"

requirements-completed: [BEND-03, BEND-04, PLAT-02, PLAT-03]

duration: 2min
completed: 2026-03-14
---

# Phase 6 Plan 2: Backend Detection Summary

**Backend detection with shutil.which + socket probing, auto-start via subprocess.Popen with health polling, and per-OS install instructions for ollama/llama-cpp/lm-studio**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T16:50:23Z
- **Completed:** 2026-03-14T16:52:55Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- BackendStatus dataclass representing installed (binary found) vs running (port open) state
- detect_backends() checks all three backends with port_overrides support
- auto_start_backend() starts servers, polls health endpoints with configurable timeout, captures logs
- Platform-specific install instructions for macOS, Linux, Windows across all backends
- 25 new tests with full mock coverage (no real binary/port checks)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for backend detection** - `60bf95d` (test)
2. **Task 1 GREEN: Implement detection module** - `c0cadda` (feat)

_TDD task with RED and GREEN commits._

## Files Created/Modified
- `llm_benchmark/backends/detection.py` - Backend detection, auto-start, port probing, install instructions (236 lines)
- `tests/test_detection.py` - 25 unit tests covering detection, auto-start, and install instructions

## Decisions Made
- Imported httpx at module level instead of locally inside auto_start_backend -- enables clean `@patch` in tests
- detect_backends() returns running=False when backend is not installed (skips port probe) -- avoids misleading state

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] httpx import moved to module level**
- **Found during:** Task 1 GREEN phase
- **Issue:** Plan specified "Import httpx locally inside the function" but this prevents `@patch("llm_benchmark.backends.detection.httpx")` from working since the attribute doesn't exist on the module
- **Fix:** Moved `import httpx` to module level
- **Files modified:** llm_benchmark/backends/detection.py
- **Verification:** All 25 tests pass
- **Committed in:** c0cadda

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minimal -- import location change for testability. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- detection.py provides BackendStatus, detect_backends(), auto_start_backend(), get_install_instructions()
- Ready for Plan 03 (llama.cpp backend) and Plan 04 (LM Studio backend) to use detection
- Ready for CLI integration to validate --backend flag against detected backends

## Self-Check: PASSED

- detection.py: FOUND
- test_detection.py: FOUND
- Commit 60bf95d (RED): FOUND
- Commit c0cadda (GREEN): FOUND
- 25 new tests passing, 195 total tests passing

---
*Phase: 06-new-backends*
*Completed: 2026-03-14*
