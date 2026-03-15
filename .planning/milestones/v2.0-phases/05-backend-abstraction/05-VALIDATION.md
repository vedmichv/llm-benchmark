---
phase: 5
slug: backend-abstraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via uv) |
| **Config file** | pyproject.toml |
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
| 05-01-01 | 01 | 1 | BACK-01 | unit | `uv run pytest tests/test_backends.py -x` | No -- Wave 0 | pending |
| 05-01-02 | 01 | 1 | BACK-02 | unit | `uv run pytest tests/test_backends.py::test_ollama_backend_converts_ns -x` | No -- Wave 0 | pending |
| 05-01-03 | 01 | 1 | BACK-03 | unit | `uv run pytest tests/test_backends.py -x` | No -- Wave 0 | pending |
| 05-02-01 | 02 | 2 | BACK-04 | unit | `uv run pytest tests/test_runner.py -x` | Yes -- needs update | pending |
| 05-02-02 | 02 | 2 | BACK-05 | integration | `uv run pytest tests/ -v` | Yes -- needs fixture updates | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_backends.py` -- stubs for BACK-01, BACK-02, BACK-03 (Backend protocol compliance, OllamaBackend, BackendResponse, BackendError, StreamResult)
- [ ] `tests/conftest.py` -- update fixtures from OllamaResponse to BackendResponse (covers BACK-05)
- [ ] No framework install needed -- pytest already configured

*Existing infrastructure covers framework requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Identical CLI output to v1.0 | BACK-03/BACK-05 | Requires visual comparison of full benchmark run output | Run `python -m llm_benchmark run` against a local model and compare output format with v1.0 baseline |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
