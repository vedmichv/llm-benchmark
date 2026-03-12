# LLM Benchmark v2

## What This Is

A Python CLI tool for students on a DevOps/Infrastructure course to benchmark LLM performance (tokens/sec) on their local hardware via Ollama. Students run the tool, compare models, understand what their GPU/CPU/RAM can handle, find optimal configurations, and generate shareable reports.

## Core Value

A student clones the repo, runs one command, and gets a clear answer: "Here's what your machine can do with local LLMs, here's the best model/config for your setup."

## Requirements

### Validated

- Benchmark models via Ollama API with streaming — existing
- Prompt eval + response + total tokens/sec metrics — existing
- System info collection (CPU, RAM, GPU) — existing
- Model offloading between benchmarks — existing (fixed: now uses API instead of sudo pkill)
- Multiple export formats (Markdown, JSON, CSV) — existing
- Predefined prompt sets (small/medium/large) — existing
- Cross-platform launcher (run.py) — existing

### Active

- [ ] Cross-platform stability (Windows/macOS/Linux without crashes)
- [ ] Concurrent benchmark mode (parallel requests, aggregate throughput)
- [ ] Parameter sweep (auto-explore num_ctx, num_gpu, temperature)
- [ ] Warmup runs before measurements
- [ ] Retry with exponential backoff for transient errors
- [ ] Enhanced results analysis (sort, filter, top-N)
- [ ] Interactive CLI menu for students (no CLI flags needed)
- [ ] Quick verification mode (~30 sec sanity check)
- [ ] Shareable HTML/Markdown report with system info + rankings
- [ ] Results directory structure (stop cluttering project root)
- [ ] Unit tests with mocked Ollama
- [ ] GitHub Actions CI (lint + tests)
- [ ] Pre-flight hardware check (RAM/GPU warnings before running)
- [ ] Visual ranked results at end of benchmark (bar chart in terminal)
- [ ] Code consolidation (single benchmark.py, not two files)

### Out of Scope

- llama.cpp server direct support — Ollama abstracts this, keep it simple for students
- nginx round-robin load balancing — too complex for course context
- dialog TUI — adds unnecessary dependency, CLI menu is enough
- 30+ environment variables — too overwhelming, use argparse CLI flags

## Context

- Existing codebase at `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark` with working benchmark tool
- External reference repo: `alexziskind1/llama-throughput-lab` with concurrent tests, parameter sweeps, retry logic, analyze tool
- Target: students on DevOps/Infrastructure course who need to understand how local hardware affects AI inference
- Students use various platforms (Windows+WSL, macOS, Linux)
- Phase 1 bug fixes already applied: signal.SIGALRM -> threading, sudo pkill -> API, macOS system info, bare excepts fixed

## Constraints

- **Stack**: Python 3.6+, Ollama SDK, Pydantic — keep minimal dependencies
- **Simplicity**: Students are not Python experts; tool must be self-explanatory
- **Cross-platform**: Must work on Windows, macOS, and Linux without platform-specific hacks
- **No sudo**: All operations must work without elevated privileges
- **Ollama-based**: All model interaction goes through Ollama API, not raw llama.cpp

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Ollama API instead of llama.cpp | Simpler setup for students, one `ollama serve` command | — Pending |
| Replace signal.SIGALRM with threading | Works on Windows (no SIGALRM) | ✓ Good |
| Replace sudo pkill with keep_alive=0 API | No sudo required, cross-platform | ✓ Good |
| Consolidate to single benchmark.py | Two files confuse students | — Pending |
| Adopt concurrent testing from external repo | Shows real-world throughput under load | — Pending |
| Adopt parameter sweep from external repo | Helps students find optimal config | — Pending |

---
*Last updated: 2026-03-12 after initialization*
