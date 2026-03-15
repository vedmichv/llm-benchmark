---
phase: 01-foundation
plan: 01
subsystem: package-structure
tags: [pydantic, rich, pyproject-toml, hatchling, python-packaging]

# Dependency graph
requires:
  - phase: none
    provides: "First plan -- no prior dependencies"
provides:
  - "Python package scaffold (llm_benchmark/) importable"
  - "pyproject.toml with all dependencies and build config"
  - "Pydantic data models: Message, OllamaResponse, BenchmarkResult, ModelSummary, SystemInfo"
  - "Rich console singleton via config.get_console()"
  - "Debug flag via config.is_debug()/set_debug()"
  - "Prompt sets (small/medium/large) via prompts.PROMPT_SETS"
  - "compute_averages() with correct total_tokens/total_time averaging (STAB-04)"
affects: [01-02, 01-03, 02-01]

# Tech tracking
tech-stack:
  added: [rich, tenacity, hatchling]
  patterns: [console-singleton, model-validator-no-side-effects, correct-rate-averaging]

key-files:
  created:
    - pyproject.toml
    - llm_benchmark/__init__.py
    - llm_benchmark/config.py
    - llm_benchmark/models.py
    - llm_benchmark/prompts.py
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_package.py
    - tests/test_models.py
  modified: []

key-decisions:
  - "Used model_validator(mode='before') instead of field_validator to handle prompt_cached flag alongside prompt_eval_count normalization"
  - "Migrated all 11 prompts from extended_benchmark.py large set (source had 11, not 7 as plan parenthetical stated)"
  - "Used uv sync for dependency management with .venv in project root"

patterns-established:
  - "Console singleton: import get_console from config.py, never instantiate Console directly"
  - "No print side effects in Pydantic validators -- use flags instead"
  - "Correct averaging: compute_averages() sums tokens/time, not mean of rates"

requirements-completed: [QUAL-01, QUAL-02, QUAL-05]

# Metrics
duration: 4min
completed: 2026-03-12
---

# Phase 1 Plan 01: Package Scaffold Summary

**Python package with Pydantic data models, Rich console singleton, prompt sets, and correct STAB-04 averaging via compute_averages()**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-12T19:35:24Z
- **Completed:** 2026-03-12T19:39:52Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Python package scaffold with pyproject.toml (hatchling, Python >=3.12, 4 runtime deps)
- Pydantic models for all data contracts: Message, OllamaResponse, BenchmarkResult, ModelSummary, SystemInfo
- OllamaResponse handles prompt caching (eval_count=-1) silently with prompt_cached flag -- no print side effects
- compute_averages() implements correct total_tokens/total_time averaging (STAB-04 fix)
- Prompt sets migrated from extended_benchmark.py (small=3, medium=5, large=11)
- Test suite with 9 passing tests covering imports, structure, all models, and averaging

## Task Commits

Each task was committed atomically:

1. **Task 1: Package scaffold and pyproject.toml** - `0e76eca` (feat)
2. **Task 2 RED: Failing tests for data models** - `c84f4f4` (test)
3. **Task 2 GREEN: Implement Pydantic data models** - `c38c9de` (feat)

_Note: Task 2 followed TDD with RED and GREEN commits._

## Files Created/Modified
- `pyproject.toml` - Package metadata, dependencies, build config, entry point
- `llm_benchmark/__init__.py` - Package init with __version__ = "2.0.0"
- `llm_benchmark/config.py` - Rich Console singleton, debug flag, constants
- `llm_benchmark/models.py` - All Pydantic data models and compute_averages()
- `llm_benchmark/prompts.py` - PROMPT_SETS dict and get_prompts() helper
- `tests/__init__.py` - Test package marker
- `tests/conftest.py` - Shared fixtures (sample_ollama_response_dict, cached variant)
- `tests/test_package.py` - Import and structure validation (STAB-01, QUAL-02)
- `tests/test_models.py` - Model validation, caching, averaging tests (STAB-04)

## Decisions Made
- Used `model_validator(mode="before")` instead of `field_validator` to handle prompt_cached flag alongside prompt_eval_count normalization in a single pass
- Migrated all 11 prompts from extended_benchmark.py large set (plan parenthetical said 7, but source had 11)
- Used `uv sync --extra test` for test dependency installation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `rich` not available on system Python; resolved by using `uv sync` to create venv with all dependencies
- pytest needed `uv sync --extra test` to install into the venv

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Package scaffold complete, all data contracts defined
- config.py, models.py, and prompts.py ready for import by downstream plans (01-02, 01-03)
- Test infrastructure established with conftest.py fixtures

## Self-Check: PASSED

All 9 created files verified. All 3 commit hashes verified (0e76eca, c84f4f4, c38c9de).

---
*Phase: 01-foundation*
*Completed: 2026-03-12*
