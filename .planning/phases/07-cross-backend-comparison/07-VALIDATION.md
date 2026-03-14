---
phase: 7
slug: cross-backend-comparison
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
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
| 7-01-01 | 01 | 1 | COMP-01 | unit | `uv run pytest tests/test_comparison.py::test_run_comparison_sequential -x` | ❌ W0 | ⬜ pending |
| 7-01-02 | 01 | 1 | COMP-02 | unit | `uv run pytest tests/test_comparison.py::test_single_model_bar_chart -x` | ❌ W0 | ⬜ pending |
| 7-01-03 | 01 | 1 | COMP-03 | unit | `uv run pytest tests/test_comparison.py::test_matrix_table -x` | ❌ W0 | ⬜ pending |
| 7-01-04 | 01 | 1 | COMP-04 | unit | `uv run pytest tests/test_comparison.py::test_winner_recommendation -x` | ❌ W0 | ⬜ pending |
| 7-02-01 | 02 | 1 | COMP-05 | unit | `uv run pytest tests/test_menu.py::test_mode_compare -x` | ❌ W0 | ⬜ pending |
| 7-02-02 | 02 | 1 | COMP-01 | unit | `uv run pytest tests/test_cli.py::test_backend_all_choice -x` | ❌ W0 | ⬜ pending |
| 7-03-01 | 03 | 2 | DOC-01 | manual | Visual inspection of README.md | N/A | ⬜ pending |
| 7-03-02 | 03 | 2 | DOC-02 | manual | Visual inspection of README.md | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_comparison.py` — stubs for COMP-01, COMP-02, COMP-03, COMP-04
- [ ] Update `tests/test_menu.py` — stubs for COMP-05 (option 5)
- [ ] Update `tests/test_cli.py` — stubs for `--backend all` parsing

*Existing infrastructure (pytest, conftest.py fixtures) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README multi-backend quick start section | DOC-01 | Documentation content requires human review | Verify new "Multi-Backend Setup" section exists after Quick Start with per-OS install guides for all 3 backends |
| README cross-backend comparison example | DOC-02 | Documentation content requires human review | Verify "Cross-Backend Comparison" section exists with CLI usage and realistic example output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
