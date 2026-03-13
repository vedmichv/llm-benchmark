---
phase: 3
slug: advanced-benchmarking
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 + pytest-asyncio |
| **Config file** | pyproject.toml (existing) |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~8 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 8 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | BENCH-03 | unit | `uv run pytest tests/test_concurrent.py -x` | No -- Wave 0 | pending |
| 03-01-02 | 01 | 1 | BENCH-04 | unit | `uv run pytest tests/test_concurrent.py::test_aggregate_throughput -x` | No -- Wave 0 | pending |
| 03-02-01 | 02 | 1 | BENCH-05 | unit | `uv run pytest tests/test_sweep.py -x` | No -- Wave 0 | pending |
| 03-02-02 | 02 | 1 | BENCH-06 | unit | `uv run pytest tests/test_sweep.py::test_best_config_selection -x` | No -- Wave 0 | pending |
| 03-03-01 | 03 | 2 | ANLZ-01 | unit | `uv run pytest tests/test_analyze.py::test_sort_by_metric -x` | No -- Wave 0 | pending |
| 03-03-02 | 03 | 2 | ANLZ-02 | unit | `uv run pytest tests/test_analyze.py::test_top_n_filter -x` | No -- Wave 0 | pending |
| 03-03-03 | 03 | 2 | ANLZ-03 | unit | `uv run pytest tests/test_cli.py::test_compare_enhanced -x` | No -- Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `pytest-asyncio` dev dependency: `uv add --dev pytest-asyncio`
- [ ] `tests/test_concurrent.py` — covers BENCH-03, BENCH-04 (mock AsyncClient)
- [ ] `tests/test_sweep.py` — covers BENCH-05, BENCH-06 (mock ollama.show + chat)
- [ ] `tests/test_analyze.py` — covers ANLZ-01, ANLZ-02 (sample JSON result files)
- [ ] `tests/conftest.py` — add fixtures for mock AsyncClient, sample sweep modelinfo, sample result files

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Concurrent requests actually run in parallel against Ollama | BENCH-03 | Requires live Ollama server | Run `python -m llm_benchmark run --concurrent 4` and verify wall time < 4x sequential time |
| Sweep detects real model layer count | BENCH-05 | Requires actual model metadata | Run `python -m llm_benchmark run --sweep` and verify num_gpu values match model architecture |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 8s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
