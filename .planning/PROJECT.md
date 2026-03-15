# LLM Benchmark

## What This Is

A Python CLI tool for students on a DevOps/Infrastructure course to benchmark LLM performance (tokens/sec) on their local hardware. Supports three inference backends (Ollama, llama.cpp, LM Studio) with cross-backend comparison showing which runtime is fastest. Interactive menu, visual bar charts, shareable Markdown reports.

## Core Value

A student clones the repo, runs one command, and gets a clear answer: "Here's what your machine can do with local LLMs, here's the best model/config for your setup, and here's the fastest backend."

## Current State

**Shipped:** v2.0 Multi-Backend Benchmark (2026-03-15)
**Code:** 11,639 LOC Python + 5,416 LOC tests
**Tests:** 325 passing, 63% coverage, CI pipeline
**Next:** Planning next milestone

## Requirements

### Validated

- ✓ Benchmark models via Ollama API with streaming — v1.0 Phase 1
- ✓ Prompt eval + response + total tokens/sec metrics — v1.0 Phase 1
- ✓ System info collection (CPU, RAM, GPU) — v1.0 Phase 1
- ✓ Model offloading between benchmarks (keep_alive=0) — v1.0 Phase 1
- ✓ Multiple export formats (Markdown, JSON, CSV) — v1.0 Phase 1
- ✓ Predefined prompt sets (small/medium/large) — v1.0 Phase 1
- ✓ Cross-platform stability (threading timeout, no sudo) — v1.0 Phase 1
- ✓ Retry with exponential backoff (tenacity) — v1.0 Phase 2
- ✓ Warmup runs before measurements — v1.0 Phase 2
- ✓ Concurrent benchmark mode (async, aggregate throughput) — v1.0 Phase 3
- ✓ Parameter sweep (num_ctx, num_gpu) — v1.0 Phase 3
- ✓ Enhanced results analysis (sort, filter, rankings) — v1.0 Phase 3
- ✓ Interactive CLI menu (no CLI flags needed) — v1.0 Phase 4
- ✓ Quick verification mode (~30s sanity check) — v1.0 Phase 4
- ✓ Visual ranked results (Unicode bar chart) — v1.0 Phase 4
- ✓ Shareable Markdown report with rankings — v1.0 Phase 4
- ✓ Unit tests with mocked Ollama (152 tests, 63% coverage) — v1.0 Phase 4
- ✓ GitHub Actions CI (ruff + pytest) — v1.0 Phase 4
- ✓ RAM-based model recommender — v1.0 Quick Task
- ✓ Ollama auto-install check — v1.0 Quick Task
- ✓ Auto-detect context window per model — v1.0 post-release
- ✓ Backend abstraction layer (Backend Protocol + 3 implementations) — v2.0 Phase 5
- ✓ llama.cpp backend via native /completion API — v2.0 Phase 6
- ✓ LM Studio backend via native /v1/chat/completions API — v2.0 Phase 6
- ✓ Auto-detect running backends (shutil.which + socket probe) — v2.0 Phase 6
- ✓ Auto-start backends if installed but not running — v2.0 Phase 6
- ✓ `--backend` CLI flag (ollama, llama-cpp, lm-studio, all) — v2.0 Phase 6
- ✓ Cross-backend comparison with bar chart and matrix table — v2.0 Phase 7
- ✓ "Fastest backend" recommendation per model — v2.0 Phase 7
- ✓ Menu option 5 "Compare backends" with install hints — v2.0 Phase 7
- ✓ Multi-backend setup guides in README (per-OS) — v2.0 Phase 7

### Active

(None — planning next milestone)

### Out of Scope

- nginx round-robin load balancing — too complex for course context
- dialog TUI — adds unnecessary dependency, CLI menu is enough
- 30+ environment variables — too overwhelming, use argparse CLI flags
- Cloud API backends (OpenAI, Anthropic) — focus on local inference
- Custom llama.cpp compilation optimization — students use Homebrew or prebuilt
- Docker packaging — students need to understand local setup
- llama-cpp-python binding — tool talks to llama-server via HTTP
- Auto-download models for all backends — potential v3 feature

## Context

- v1.0 shipped 2026-03-13: 4 phases, 12 plans, 152 tests
- v2.0 shipped 2026-03-15: 7 phases, 31 plans, 325 tests
- Tech stack: Python 3.12+, uv, Rich, Pydantic, httpx, tenacity
- 3 backends: Ollama (default), llama.cpp (fastest on Apple Silicon), LM Studio (GUI-friendly)
- llama-cpp typically 1.3x faster than Ollama on Apple Silicon (confirmed in UAT: 243 vs 190 t/s)
- LM Studio comparable to Ollama speed (~190 t/s for same model)

## Constraints

- **Stack**: Python 3.12+, uv, minimal new dependencies (httpx for HTTP backends)
- **Simplicity**: Students are not Python experts; tool must be self-explanatory
- **Cross-platform**: Must work on Windows, macOS, and Linux
- **No sudo**: All operations must work without elevated privileges
- **Backwards-compatible**: `--backend ollama` is default, existing workflows unchanged

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Ollama API instead of llama.cpp | Simpler setup for students | ✓ Good (v1.0), expanded in v2.0 |
| Replace signal.SIGALRM with threading | Works on Windows | ✓ Good |
| Replace sudo pkill with keep_alive=0 API | No sudo required | ✓ Good |
| Package structure (llm_benchmark/) | Clean imports, testable | ✓ Good |
| Rich Console for all output | Consistent formatting | ✓ Good |
| Native APIs (not OpenAI-compat) for backends | OpenAI endpoint strips timing metrics | ✓ Good — confirmed, LM Studio /v1/ has empty stats |
| httpx for HTTP backends | async-ready, stdlib-like API | ✓ Good |
| Backend protocol (not base class) | Lightweight, Pythonic, runtime_checkable | ✓ Good |
| Sequential backend comparison (not parallel) | Backends share GPU resources, parallel corrupts timing | ✓ Good |
| Ollama names as canonical for cross-backend matrix | Consistent model identity across backends | ✓ Good |
| Wall-clock fallback for LM Studio timing | LM Studio non-streaming returns empty stats | ✓ Good |

---
*Last updated: 2026-03-15 after v2.0 milestone completion*
