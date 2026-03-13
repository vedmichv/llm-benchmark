---
phase: quick
plan: 1
subsystem: interactive-menu
tags: [recommend, menu, ollama, ram-detection]
dependency_graph:
  requires: []
  provides: [model-recommender, ram-tier-filtering]
  affects: [menu.py, interactive-flow]
tech_stack:
  added: []
  patterns: [tiered-recommendation, lazy-import, passthrough-pattern]
key_files:
  created:
    - llm_benchmark/recommend.py
    - tests/test_recommend.py
  modified:
    - llm_benchmark/menu.py
    - tests/test_menu.py
decisions:
  - Import _get_ram_gb from system.py (shared RAM detection, avoids third copy)
  - Use plain input() for recommend prompts (consistent with menu pattern)
  - Mock offer_model_downloads in menu tests to isolate input sequences
metrics:
  duration: 3min
  completed: 2026-03-13
---

# Quick Task 1: Add Model Recommender to Interactive Menu Summary

Tiered model recommender with RAM-based filtering using 4 tiers (small/medium/large/xl) and interactive download UI via ollama pull.

## What Was Done

### Task 1: Create recommend.py with tiered model recommendations (TDD)
**Commit:** `e13cb8c`

Created `llm_benchmark/recommend.py` with:
- `TIERED_MODELS` list: 13 curated models across 4 tiers (small, medium, large, xl)
- `RAM_TIER_THRESHOLDS`: small=0GB, medium=16GB, large=36GB, xl=64GB
- `get_recommended_models(ram_gb)`: filters models by eligible RAM tiers
- `filter_already_installed(recommended, installed_names)`: removes already-pulled models
- `offer_model_downloads(installed_models)`: interactive UI with grouped tier display, number selection, and `ollama pull` execution

Created `tests/test_recommend.py` with 10 tests covering all tier thresholds and filtering edge cases.

### Task 2: Wire recommend into interactive menu flow
**Commit:** `c101e43`

Inserted `offer_model_downloads(models)` call in `run_interactive_menu()` after system summary display and before mode selection menu. Used lazy import (consistent with project pattern). Updated 6 existing menu tests to mock the recommender as a passthrough to avoid input sequence interference.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed menu test input sequence offset**
- **Found during:** Task 2
- **Issue:** Adding `offer_model_downloads` introduced an extra `input()` call that consumed one of the mocked inputs in existing menu tests, causing test_custom_mode to fail.
- **Fix:** Added `@patch("llm_benchmark.recommend.offer_model_downloads", side_effect=lambda m: m)` to all 6 menu test methods to isolate the recommender from input mocking.
- **Files modified:** tests/test_menu.py
- **Commit:** c101e43

## Verification

- All 145 tests pass (`uv run pytest tests/ -x -q`)
- Module imports cleanly (`from llm_benchmark.recommend import get_recommended_models, offer_model_downloads`)
- Menu flow: header -> system info -> recommend prompt -> mode selection (1-4)

## Self-Check: PASSED
