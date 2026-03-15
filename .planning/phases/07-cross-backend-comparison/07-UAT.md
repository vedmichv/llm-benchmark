---
status: complete
phase: 07-cross-backend-comparison
source: [07-01-SUMMARY.md, 07-02-SUMMARY.md, 07-03-SUMMARY.md, 07-04-SUMMARY.md]
started: 2026-03-14T20:30:00Z
updated: 2026-03-15T15:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. --backend all CLI option
expected: Run `uv run python -m llm_benchmark run --backend all`. The tool detects all running backends, benchmarks models on each sequentially, displays comparison results in the terminal, and saves comparison report files.
result: pass

### 2. Single-model comparison bar chart
expected: When only one model is common across backends, a horizontal bar chart appears showing each backend's throughput (t/s) with a star on the fastest.
result: pass

### 3. Multi-model matrix table
expected: When multiple models are benchmarked, a Rich matrix table displays models as rows, backends as columns with t/s values, winner starred per row, and "Fastest backend: X (N/M models)" summary at the bottom.
result: pass

### 4. Comparison export files
expected: After comparison, JSON and Markdown files are saved to results/ directory. JSON contains structured comparison data. Markdown contains a human-readable comparison report.
result: pass

### 5. Menu option 5 ("Compare backends")
expected: Run `uv run python -m llm_benchmark` (interactive mode). Menu shows option 5 "Compare backends". Selecting it auto-runs comparison with medium prompts and 2 runs per prompt.
result: issue
reported: "Backend selector appears before mode selection — user has to pick a single backend before seeing option 5 (Compare backends). Confusing because compare mode uses all backends anyway."
severity: minor

### 6. Single-backend install hints
expected: With only Ollama running (no llama.cpp or LM Studio), selecting menu option 5 shows install hints for the missing backends instead of erroring.
result: pass

### 7. README multi-backend setup
expected: README.md contains a "Multi-Backend Setup" section after Quick Start with per-OS install guides (macOS, Windows, Linux) covering Ollama, llama.cpp, and LM Studio.
result: pass

### 8. README comparison example
expected: README.md contains a "Cross-Backend Comparison" section with CLI usage examples and a crafted-but-realistic example output showing llama.cpp ~1.5x faster than Ollama on Apple Silicon.
result: pass

## Summary

total: 8
passed: 7
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Menu option 5 auto-uses all detected backends with no backend selection step"
  status: failed
  reason: "User reported: Backend selector appears before mode selection — user has to pick a single backend before seeing option 5 (Compare backends). Confusing because compare mode uses all backends anyway."
  severity: minor
  test: 5
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
