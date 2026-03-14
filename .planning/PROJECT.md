# LLM Benchmark

## What This Is

A Python CLI tool for students on a DevOps/Infrastructure course to benchmark LLM performance (tokens/sec) on their local hardware. Supports multiple inference backends (Ollama, llama.cpp, LM Studio) so students can compare not just models but also which runtime is fastest on their hardware. Interactive menu, visual results, shareable reports.

## Core Value

A student clones the repo, runs one command, and gets a clear answer: "Here's what your machine can do with local LLMs, here's the best model/config for your setup."

## Current Milestone: v2.0 Multi-Backend Benchmark

**Goal:** Add llama.cpp and LM Studio as alternative backends alongside Ollama, with cross-backend comparison mode showing which runtime is fastest for each model.

**Target features:**
- Backend abstraction layer (protocol + 3 implementations: Ollama, llama.cpp, LM Studio)
- `--backend` CLI flag to select inference runtime
- Auto-detect running backends
- Cross-backend comparison: same model on all backends, side-by-side report
- Setup guides for each backend (students need to know how to install)

## Requirements

### Validated (v1.0 shipped)

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

### Active (v2.0)

- [ ] Backend abstraction layer (protocol/ABC for inference backends)
- [ ] llama.cpp backend via native /completion API
- [ ] LM Studio backend via native /api/v0/ API
- [ ] `--backend` CLI flag (ollama, llama-cpp, lm-studio)
- [ ] Auto-detect running backends
- [ ] Cross-backend comparison mode (same model, all backends, comparison report)
- [ ] Setup documentation for llama.cpp and LM Studio
- [ ] Backend-aware model management (GGUF paths for llama.cpp, JIT for LM Studio)

### Out of Scope

- nginx round-robin load balancing — too complex for course context
- dialog TUI — adds unnecessary dependency, CLI menu is enough
- 30+ environment variables — too overwhelming, use argparse CLI flags
- Cloud API backends (OpenAI, Anthropic) — focus on local inference
- Custom llama.cpp compilation optimization — students use Homebrew or prebuilt

## Context

- v1.0 complete: 4 phases, 12 plans, 152 tests, 2 quick tasks
- Known Ollama bug with Qwen 3.5 MoE models (#14579, #14662) — motivation for llama.cpp alternative
- llama.cpp typically 1.5-2x faster than Ollama on Apple Silicon (same model)
- Research confirmed: all 3 backends have native APIs with server-side timing metrics
- Each backend uses different native endpoint (NOT OpenAI-compat — that strips timing data)
- llama.cpp: native `/completion` with `timings.predicted_per_second`
- LM Studio: native `/api/v0/chat/completions` with `stats.tokens_per_second`
- Ollama: native `/api/chat` with `eval_count`/`eval_duration` (current)

## Constraints

- **Stack**: Python 3.12+, uv, minimal new dependencies (httpx for HTTP backends)
- **Simplicity**: Students are not Python experts; tool must be self-explanatory
- **Cross-platform**: Must work on Windows, macOS, and Linux
- **No sudo**: All operations must work without elevated privileges
- **Backwards-compatible**: `--backend ollama` is default, existing workflows unchanged

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Ollama API instead of llama.cpp | Simpler setup for students | ✓ Good (v1.0), expanding in v2.0 |
| Replace signal.SIGALRM with threading | Works on Windows | ✓ Good |
| Replace sudo pkill with keep_alive=0 API | No sudo required | ✓ Good |
| Package structure (llm_benchmark/) | Clean imports, testable | ✓ Good |
| Rich Console for all output | Consistent formatting | ✓ Good |
| Native APIs (not OpenAI-compat) for backends | OpenAI endpoint strips timing metrics | — v2.0 |
| httpx for HTTP backends | async-ready, stdlib-like API | — v2.0 |
| Backend protocol (not base class) | Lightweight, Pythonic | — v2.0 |

---
*Last updated: 2026-03-14 after v2.0 milestone start*
