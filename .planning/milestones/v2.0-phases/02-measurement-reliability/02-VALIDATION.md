---
phase: 2
slug: measurement-reliability
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | pyproject.toml (existing) |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | BENCH-01 | unit | `uv run pytest tests/test_runner.py::TestWarmupModel -x` | No -- Wave 0 | pending |
| 02-01-02 | 01 | 1 | BENCH-01 | unit | `uv run pytest tests/test_cli.py::TestSkipWarmup -x` | No -- Wave 0 | pending |
| 02-02-01 | 02 | 1 | BENCH-02 | unit | `uv run pytest tests/test_runner.py::TestRetryLogic -x` | No -- Wave 0 | pending |
| 02-02-02 | 02 | 1 | BENCH-02 | unit | `uv run pytest tests/test_runner.py::TestRetryLogic -x` | No -- Wave 0 | pending |
| 02-03-01 | 03 | 1 | BENCH-07 | unit | `uv run pytest tests/test_runner.py::TestCacheVisibility -x` | No -- Wave 0 | pending |
| 02-03-02 | 03 | 1 | BENCH-07 | unit | `uv run pytest tests/test_exporters.py::TestCsvCacheColumn -x` | No -- Wave 0 | pending |
| 02-03-03 | 03 | 1 | BENCH-07 | unit | `uv run pytest tests/test_exporters.py::TestMarkdownCacheIndicator -x` | No -- Wave 0 | pending |
| 02-04-01 | 04 | 1 | UX-04 | unit | `uv run pytest tests/test_exporters.py -x` | No -- Wave 0 | pending |
| 02-04-02 | 04 | 1 | UX-04 | unit | `uv run pytest tests/test_exporters.py::TestResultsGitignore -x` | No -- Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_runner.py` — add TestWarmupModel, TestRetryLogic, TestCacheVisibility classes
- [ ] `tests/test_cli.py` — add tests for --skip-warmup, --max-retries flags
- [ ] `tests/test_exporters.py` — add TestCsvCacheColumn, TestMarkdownCacheIndicator, TestResultsGitignore classes

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Warmup "Warming up..." display | BENCH-01 | Visual terminal output | Run `python -m llm_benchmark run` and verify warmup message appears before benchmarking |
| [cached] one-liner explanation | BENCH-07 | Requires actual prompt caching from Ollama | Run repeated benchmarks with same prompt, verify one-liner appears once |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
