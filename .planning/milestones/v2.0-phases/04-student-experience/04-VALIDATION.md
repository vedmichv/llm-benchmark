---
phase: 4
slug: student-experience
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0 |
| **Config file** | pyproject.toml (implicit pytest discovery) |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -x -q --cov=llm_benchmark --cov-fail-under=60` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -x -q --cov=llm_benchmark --cov-fail-under=60`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | UX-01 | unit | `uv run pytest tests/test_menu.py -x` | No -- Wave 0 | pending |
| 04-01-02 | 01 | 1 | UX-02 | unit | `uv run pytest tests/test_menu.py::TestQuickTest -x` | No -- Wave 0 | pending |
| 04-02-01 | 02 | 1 | UX-03 | unit | `uv run pytest tests/test_display.py -x` | No -- Wave 0 | pending |
| 04-02-02 | 02 | 1 | UX-05 | unit | `uv run pytest tests/test_display.py::TestRecommendation -x` | No -- Wave 0 | pending |
| 04-03-01 | 03 | 2 | UX-06 | unit | `uv run pytest tests/test_exporters.py::TestEnhancedMarkdown -x` | No -- Wave 0 | pending |
| 04-04-01 | 04 | 2 | QUAL-03 | integration | `uv run pytest tests/ --cov=llm_benchmark --cov-fail-under=60` | Partial | pending |
| 04-04-02 | 04 | 2 | QUAL-04 | smoke | manual verify CI run | No -- Wave 0 | pending |

*Status: pending · green · red · flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_menu.py` — stubs for UX-01, UX-02 (menu logic, quick test selection)
- [ ] `tests/test_display.py` — stubs for UX-03, UX-05 (bar chart rendering, recommendation)
- [ ] Enhanced tests in `tests/test_exporters.py` — stubs for UX-06 (rankings in Markdown)
- [ ] `pyproject.toml` updates — ruff + pytest-cov in dev deps

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Quick test completes in ~30s | UX-02 | Timing depends on hardware/model | Run `python -m llm_benchmark` → select Quick test → verify completes within ~30s |
| CI runs on push | QUAL-04 | Requires GitHub infrastructure | Push to branch → verify Actions run passes |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
