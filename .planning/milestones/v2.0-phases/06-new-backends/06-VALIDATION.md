---
phase: 6
slug: new-backends
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0 |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | BEND-01 | unit | `uv run pytest tests/test_llamacpp.py -x` | No -- Wave 0 | pending |
| 06-01-02 | 01 | 1 | BEND-02 | unit | `uv run pytest tests/test_lmstudio.py -x` | No -- Wave 0 | pending |
| 06-02-01 | 02 | 2 | BEND-03 | unit | `uv run pytest tests/test_detection.py -x` | No -- Wave 0 | pending |
| 06-02-02 | 02 | 2 | BEND-04 | unit | `uv run pytest tests/test_detection.py::test_auto_start -x` | No -- Wave 0 | pending |
| 06-02-03 | 02 | 2 | BEND-05 | unit | `uv run pytest tests/test_preflight.py -x` | Partial | pending |
| 06-02-04 | 02 | 2 | PLAT-02 | unit | `uv run pytest tests/test_detection.py::test_llamacpp_detection -x` | No -- Wave 0 | pending |
| 06-02-05 | 02 | 2 | PLAT-03 | unit | `uv run pytest tests/test_detection.py::test_lmstudio_detection -x` | No -- Wave 0 | pending |
| 06-03-01 | 03 | 3 | CLI-01 | unit | `uv run pytest tests/test_cli.py -x` | Partial | pending |
| 06-03-02 | 03 | 3 | CLI-02 | unit | `uv run pytest tests/test_menu.py -x` | Partial | pending |
| 06-03-03 | 03 | 3 | CLI-03 | unit | `uv run pytest tests/test_exporters.py -x` | Partial | pending |
| 06-03-04 | 03 | 3 | CLI-04 | unit | `uv run pytest tests/test_system.py -x` | Partial | pending |
| 06-03-05 | 03 | 3 | CLI-05 | unit | `uv run pytest tests/test_menu.py -x` | Partial | pending |
| 06-XX-XX | -- | -- | PLAT-01 | manual | N/A | N/A | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_llamacpp.py` -- stubs for BEND-01: LlamaCppBackend protocol compliance, timing conversion, error wrapping
- [ ] `tests/test_lmstudio.py` -- stubs for BEND-02: LMStudioBackend protocol compliance, usage parsing, model management
- [ ] `tests/test_detection.py` -- stubs for BEND-03, BEND-04, PLAT-02, PLAT-03: backend detection, port probing, auto-start
- [ ] Update `tests/conftest.py` -- add fixtures for mock llama-cpp and lm-studio HTTP responses
- [ ] `uv add httpx` -- make httpx explicit dependency (currently transitive only)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cross-platform backend operation | PLAT-01 | Requires actual backends on macOS/Windows/Linux | Install each backend, run `python -m llm_benchmark run --backend <name>`, verify results |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
