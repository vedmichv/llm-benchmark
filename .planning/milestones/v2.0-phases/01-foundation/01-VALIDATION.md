---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | none — Wave 0 installs |
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
| 01-01-01 | 01 | 0 | QUAL-02 | unit | `uv run pytest tests/test_package.py -x` | -- W0 | pending |
| 01-01-02 | 01 | 0 | QUAL-05 | unit | `uv run pytest tests/test_package.py -x` | -- W0 | pending |
| 01-02-01 | 02 | 1 | STAB-01 | unit | `uv run pytest tests/test_package.py::test_imports -x` | -- W0 | pending |
| 01-02-02 | 02 | 1 | QUAL-01 | smoke | `uv run python -m llm_benchmark --help` | -- W0 | pending |
| 01-02-03 | 02 | 1 | STAB-02 | unit (mocked) | `uv run pytest tests/test_preflight.py::test_ollama_unreachable -x` | -- W0 | pending |
| 01-02-04 | 02 | 1 | STAB-03 | unit (mocked) | `uv run pytest tests/test_preflight.py::test_ram_warning -x` | -- W0 | pending |
| 01-02-05 | 02 | 1 | STAB-04 | unit | `uv run pytest tests/test_runner.py::test_correct_averaging -x` | -- W0 | pending |
| 01-02-06 | 02 | 1 | STAB-05 | unit (mocked) | `uv run pytest tests/test_runner.py::test_offload_model -x` | -- W0 | pending |
| 01-02-07 | 02 | 1 | STAB-06 | unit | `uv run pytest tests/test_runner.py::test_timeout -x` | -- W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` — add `[project.optional-dependencies]` with pytest
- [ ] `tests/` directory — does not exist yet
- [ ] `tests/conftest.py` — shared fixtures (mock ollama client, mock system info)
- [ ] `tests/test_package.py` — import tests, structure validation
- [ ] `tests/test_preflight.py` — connectivity and RAM check tests
- [ ] `tests/test_runner.py` — averaging, timeout, offloading tests
- [ ] `tests/test_cli.py` — subcommand parsing tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Python >=3.12 enforced | QUAL-05 | Build metadata, not runtime | Verify `requires-python = ">= 3.12"` in pyproject.toml |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
