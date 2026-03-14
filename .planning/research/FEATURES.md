# Feature Landscape: Multi-Backend Benchmark Support

**Domain:** Cross-backend LLM benchmarking (Ollama + llama.cpp + LM Studio)
**Researched:** 2026-03-14
**Scope:** NEW features only -- existing v1.0 features (interactive menu, bar charts, model recommender, concurrent mode, sweep mode, exports, 152 tests) are already shipped.

## Table Stakes

Features users expect from a multi-backend benchmark tool. Missing any of these = the multi-backend story feels broken.

| Feature | Why Expected | Complexity | Depends On | Notes |
|---------|--------------|------------|------------|-------|
| `--backend` CLI flag | Users need to select which runtime to benchmark. Without it, multi-backend is unusable | Low | Backend protocol | `--backend ollama` (default), `--backend llama-cpp`, `--backend lm-studio`. Default to `ollama` for backward compat |
| Backend abstraction protocol | All existing runner code is hardcoded to `ollama` Python SDK. Need a common interface so runner.py works with any backend | Med | None (foundation) | Python Protocol class with methods: `chat()`, `list_models()`, `load_model()`, `unload_model()`, `health_check()`, `get_system_info()`. Each backend implements this |
| Ollama backend (wrap existing) | Existing code must still work unchanged as the default backend | Med | Backend protocol | Wrap current `ollama.chat()`, `ollama.list()`, `ollama.generate(keep_alive=0)` into the protocol. Minimal behavioral change |
| llama.cpp backend via `/completion` | llama.cpp is 1.5-2x faster than Ollama on Apple Silicon. Students want to see this for themselves | High | Backend protocol | Native `/completion` endpoint (NOT OpenAI-compat -- that strips timings). Response includes `timings.predicted_per_second`, `timings.prompt_per_second`, `timings.predicted_n`, `timings.prompt_n`, `timings.predicted_ms`, `timings.prompt_ms` |
| LM Studio backend via `/api/v1/chat` | LM Studio has a large student userbase (GUI app, easy model management). Natural companion backend | High | Backend protocol | Native `/api/v1/chat` endpoint returns `stats.tokens_per_second`. Model management: `GET /api/v1/models`, `POST /api/v1/models/load`, `POST /api/v1/models/unload` |
| Normalized timing metrics | Each backend reports metrics differently. Users need comparable numbers | Med | All backends | Normalize to common fields: `prompt_eval_tokens`, `prompt_eval_seconds`, `response_tokens`, `response_seconds`, `tokens_per_second`. Backend-specific raw data preserved in response |
| Auto-detect running backends | Students should not need to know which port each backend runs on. The tool should find what is running | Med | Backend protocol | Ollama: `http://localhost:11434/api/tags`. llama.cpp: `http://localhost:8080/health`. LM Studio: `http://localhost:1234/api/v1/models`. Check each, report what is available |
| Backend-aware preflight checks | Current preflight assumes Ollama. Must work for any backend | Med | Backend protocol, auto-detect | Check connectivity to selected backend. List available models. RAM warnings still apply. Show backend-specific install instructions on failure |
| Backend label in exports | Results must clearly show which backend was used. Otherwise comparison data is meaningless | Low | Backend protocol | Add `backend` field to JSON/CSV/Markdown exports. Include in `SystemInfo` model |
| Setup documentation per backend | Students need to know how to install and start llama.cpp server and LM Studio. Ollama docs already exist | Low | None | Concise "Getting started with llama.cpp" and "Getting started with LM Studio" in docs or `--help` output. Not code -- just instructions |

## Differentiators

Features that make this tool stand out. Not expected in every multi-backend tool but high value for students comparing runtimes.

| Feature | Value Proposition | Complexity | Depends On | Notes |
|---------|-------------------|------------|------------|-------|
| Cross-backend comparison mode | "Same model, all backends, side-by-side" -- the killer feature. Students see that llama.cpp is 1.5x faster than Ollama on their exact hardware | High | All 3 backends, normalized metrics | Run same model on each detected backend. Produce comparison table with delta percentages. Reuse existing `compare.py` pattern but cross-backend instead of cross-file |
| Cross-backend comparison report | Visual side-by-side in Markdown/terminal showing Ollama vs llama.cpp vs LM Studio | Med | Cross-backend comparison | Extend existing bar chart and Markdown exporter. Show grouped bars or table with backend columns. Highlight winner per model |
| Model name mapping | "llama3.2:1b" in Ollama is a GGUF file at a specific path in llama.cpp. Students should not manually figure out GGUF file paths | High | All 3 backends | Map Ollama model names to GGUF file paths. Ollama stores GGUFs in `~/.ollama/models/blobs/`. LM Studio stores in `~/.cache/lm-studio/models/`. Provide `--model-path` override for llama.cpp |
| `--backend all` shorthand | Benchmark on every running backend in one command. Ultimate convenience | Med | Auto-detect, cross-backend comparison | Detect running backends, benchmark each, produce combined comparison. Simple orchestration of existing per-backend runs |
| Backend health dashboard | `llm_benchmark info` shows status of all backends: running/stopped, version, loaded models, port | Low | Auto-detect | Extend existing `info` subcommand. Check each backend endpoint, report status. Helpful for debugging "why is my benchmark failing" |
| Warmup-aware backend switching | When comparing backends, unload from one before loading in another to avoid memory contention | Med | Backend protocol | Ollama: `keep_alive=0`. llama.cpp: managed by server (single model). LM Studio: `POST /api/v1/models/unload`. Sequence: unload from backend A, warmup on backend B, measure |
| Backend-specific parameter sweep | Sweep `num_ctx` and `num_gpu` on each backend. llama.cpp uses `-c` and `-ngl`, LM Studio uses load parameters | High | Sweep mode, all backends | Map existing sweep dimensions to backend-specific parameters. llama.cpp: `n_predict`, context size via server restart or slot config. LM Studio: context length in load request |

## Anti-Features

Features to explicitly NOT build for v2.0. Each represents a complexity trap.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| llama.cpp server lifecycle management | Starting/stopping/building llama.cpp server is OS-specific, requires knowing binary paths, GPU flags. Students should start it separately | Document how to start llama.cpp server. Detect if running. Do NOT attempt to `subprocess.Popen()` the server |
| LM Studio process management | LM Studio is a GUI app. Launching/controlling it programmatically is fragile and OS-specific | Document "open LM Studio, load a model, enable API". Detect if running |
| OpenAI-compatible endpoint usage | `/v1/chat/completions` on llama.cpp and LM Studio strips native timing fields. You get `usage.completion_tokens` but NOT `timings.predicted_per_second` or `stats.tokens_per_second` | Always use native endpoints: llama.cpp `/completion`, LM Studio `/api/v1/chat`, Ollama `/api/chat` |
| Custom llama.cpp compilation | Building llama.cpp with CUDA/Metal/ROCm flags is expert-level. Students use Homebrew (`brew install llama.cpp`) or prebuilt binaries | Document `brew install llama.cpp` for macOS, prebuilt releases for Linux/Windows. Assume server binary exists |
| GGUF model downloading | Downloading models from HuggingFace requires knowing quantization formats, file sizes, model variants. Ollama and LM Studio handle this via their own UIs | Direct students to use `ollama pull` or LM Studio GUI for model acquisition. For llama.cpp, use models already downloaded by Ollama or LM Studio |
| Backend-specific concurrent mode | Running concurrent benchmarks across backends simultaneously would require complex cross-process coordination | Keep concurrent mode per-backend. Cross-backend comparison runs sequentially (one backend at a time) |
| Cloud API backends (OpenAI, Anthropic, etc.) | Adds API key management, rate limiting, cost tracking, network latency variables. Completely different from local inference benchmarking | Stay focused on local inference. Cloud APIs are a different product |
| Model quality/accuracy comparison across backends | Same GGUF on different backends should produce similar (but not identical) output. Comparing quality is a different tool | Note in docs that outputs may differ slightly. Benchmark throughput only |

## Feature Dependencies

```
Backend protocol -----------------> Ollama backend (wrap existing)
                                --> llama.cpp backend
                                --> LM Studio backend

Ollama backend ------------------> All existing features continue working
                                   (backward compatibility)

All 3 backends ------------------> Auto-detect running backends
                                --> --backend CLI flag
                                --> Backend-aware preflight
                                --> Normalized timing metrics

Normalized timing metrics -------> Cross-backend comparison mode
                                --> Cross-backend comparison report
                                --> Backend label in exports

Auto-detect running backends ----> --backend all shorthand
                                --> Backend health dashboard

Cross-backend comparison mode ---> Cross-backend comparison report

Model name mapping --------------> Cross-backend comparison mode
                                   (need same model on multiple backends)
```

## Backend API Reference (for implementation)

### Ollama (existing, wrap into protocol)
- **Chat:** `POST /api/chat` -- returns `eval_count`, `eval_duration` (ns), `prompt_eval_count`, `prompt_eval_duration` (ns), `total_duration` (ns), `load_duration` (ns)
- **List models:** `GET /api/tags` (or `ollama.list()`)
- **Model info:** `POST /api/show` -- returns model metadata including context length
- **Unload:** `ollama.generate(model=name, prompt="", keep_alive=0)`
- **Health:** `GET /api/tags` (200 = running)
- **Default port:** 11434
- **Confidence:** HIGH (current codebase uses these directly)

### llama.cpp server
- **Chat:** `POST /completion` -- returns `timings.predicted_per_second`, `timings.prompt_per_second`, `timings.predicted_n`, `timings.prompt_n`, `timings.predicted_ms`, `timings.prompt_ms`
- **List models:** `GET /models` (router mode only; single-model mode has no list)
- **Load/unload:** `POST /models/load`, `POST /models/unload` (router mode only)
- **Health:** `GET /health` -- returns `{"status": "ok"}` or 503 during loading
- **Slots:** `GET /slots` -- per-slot processing state and metrics
- **Default port:** 8080
- **Key difference:** Single-model server by default. Router mode (launch without `-m`) supports multiple models
- **Confidence:** HIGH (verified from official README)

### LM Studio
- **Chat:** `POST /api/v1/chat` -- returns `stats.tokens_per_second` (and speculative decoding stats if enabled)
- **List models:** `GET /api/v1/models` -- returns model identifiers, capabilities, load status
- **Load:** `POST /api/v1/models/load` -- accepts GPU offload, context length, flash attention, TTL parameters
- **Unload:** `POST /api/v1/models/unload` -- requires model identifier
- **Health:** `GET /api/v1/models` (200 = running)
- **Default port:** 1234
- **Key difference:** GUI-first app with API as secondary. Models managed through GUI. API provides programmatic access to loaded models
- **Confidence:** MEDIUM (from LM Studio docs; `stats` object fields beyond `tokens_per_second` not fully documented. May need to verify exact response schema by hitting the endpoint)

## Timing Metric Normalization Map

| Normalized Field | Ollama | llama.cpp | LM Studio |
|-----------------|--------|-----------|-----------|
| `prompt_eval_tokens` | `prompt_eval_count` | `timings.prompt_n` | Need to verify (possibly in `usage` or `stats`) |
| `prompt_eval_seconds` | `prompt_eval_duration / 1e9` | `timings.prompt_ms / 1000` | Need to verify |
| `response_tokens` | `eval_count` | `timings.predicted_n` | Need to verify (possibly `usage.completion_tokens`) |
| `response_seconds` | `eval_duration / 1e9` | `timings.predicted_ms / 1000` | Need to compute from `stats.tokens_per_second` and token count |
| `response_tokens_per_second` | Computed: `eval_count / (eval_duration/1e9)` | `timings.predicted_per_second` (native) | `stats.tokens_per_second` (native) |
| `prompt_tokens_per_second` | Computed: `prompt_eval_count / (prompt_eval_duration/1e9)` | `timings.prompt_per_second` (native) | Need to verify |
| `total_duration_seconds` | `total_duration / 1e9` | Computed: `(prompt_ms + predicted_ms) / 1000` | Need to compute |
| `load_duration_seconds` | `load_duration / 1e9` | Not in response (separate `/health` status) | Not in response |

**Key observation:** llama.cpp and LM Studio provide t/s natively. Ollama provides raw token counts and nanosecond durations, from which t/s is computed. The normalization layer must handle both directions: raw -> computed and native -> stored.

**Gap flag:** LM Studio's `stats` object is not fully documented. The `tokens_per_second` field is confirmed, but prompt eval metrics and token counts need verification by hitting an actual LM Studio endpoint. This is a LOW confidence area that needs phase-specific research during implementation.

## MVP Recommendation

Build in this priority order based on dependencies and student impact:

**Phase 1 -- Backend Abstraction (foundation):**
1. Backend protocol (Python Protocol class with type hints)
2. Ollama backend (wrap existing code, zero behavior change)
3. Normalized response model (extend `BenchmarkResult` with `backend` field)
4. `--backend` CLI flag with default `ollama`
5. Backend label in exports

**Rationale:** After phase 1, existing functionality works identically. All tests pass. The protocol is proven with one real implementation.

**Phase 2 -- New Backends:**
6. llama.cpp backend via httpx + `/completion` endpoint
7. LM Studio backend via httpx + `/api/v1/chat` endpoint
8. Auto-detect running backends
9. Backend-aware preflight checks
10. Setup documentation for each backend

**Rationale:** httpx is the only new dependency. Each backend is an independent implementation of the protocol. Auto-detect ties them together.

**Phase 3 -- Cross-Backend Comparison:**
11. Model name mapping (Ollama name -> GGUF path resolution)
12. Cross-backend comparison mode (`--compare-backends` or `--backend all`)
13. Cross-backend comparison report (terminal + Markdown)
14. Backend health dashboard in `info` subcommand

**Rationale:** This is the "wow" feature but requires both backends working reliably first. Model name mapping is the hardest sub-problem (Ollama blob storage is not user-friendly).

**Defer to later:**
- Backend-specific parameter sweep (high complexity, niche use case)
- `--backend all` shorthand (nice-to-have, Phase 3 comparison mode covers the need)
- Warmup-aware backend switching (optimization, not correctness)

## Sources

- Current codebase: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/` (runner.py, models.py, cli.py, exporters.py, compare.py)
- Project context: `.planning/PROJECT.md` (v2.0 milestone definition)
- llama.cpp server API: `https://github.com/ggml-org/llama.cpp/tools/server/README.md` -- Confidence: HIGH
- LM Studio REST API: `https://lmstudio.ai/docs/api` and `/docs/developer/rest` -- Confidence: MEDIUM (stats object partially documented)
- Ollama API: Current codebase uses `ollama` Python SDK -- Confidence: HIGH
- GPU benchmark patterns: `https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference` -- Confidence: HIGH (real benchmark tool using llama.cpp)
- Prior v1.0 research: `.planning/research/` (previous FEATURES.md)
