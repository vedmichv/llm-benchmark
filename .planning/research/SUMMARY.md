# Project Research Summary

**Project:** LLM Benchmark v2.0 — Multi-Backend Benchmark
**Domain:** Cross-backend local LLM throughput benchmarking (Ollama + llama.cpp + LM Studio)
**Researched:** 2026-03-14
**Confidence:** MEDIUM-HIGH

## Executive Summary

LLM Benchmark v2.0 extends an existing, well-tested Ollama-only tool to support two additional local inference runtimes: the llama.cpp HTTP server and LM Studio. The primary value proposition is enabling students to compare tokens/second across runtimes on their own hardware — motivated by a known Ollama bug where Qwen 3.5 MoE models hang indefinitely, while llama.cpp handles them correctly and at 5-7x higher throughput. The recommended approach is a backend abstraction layer using Python `typing.Protocol`, with one adapter class per backend, a single normalized response model, and `httpx` as the only new dependency.

The central architectural challenge is that Ollama, llama.cpp, and LM Studio each report timing metrics in different units and formats (nanoseconds, milliseconds, or pre-computed tokens/second) with different field names. Every comparison becomes meaningless unless all backends normalize into the same canonical shape before results reach any downstream code. The `BackendResponse` Pydantic model — storing all durations in seconds — is the foundation the entire multi-backend implementation depends on, and it must be designed first.

The key risk is scope creep and abstraction leakage. Research identifies five critical pitfalls: timing unit mismatches, model identity divergence across backends, accidental use of OpenAI-compatible endpoints (which strip timing data), non-fatal preflight handling across multiple backend states, and llama.cpp's single-model-per-server constraint. All five are avoidable if the backend protocol is designed to explicitly declare capabilities (multi-model vs single-model) and if each adapter is fully responsible for its own normalization. The tool should not manage backend server lifecycles — students start their own servers, and the tool detects what is running.

## Key Findings

### Recommended Stack

The existing stack (Ollama SDK, Pydantic, Rich, tenacity, argparse, pytest) requires only one new dependency: `httpx >= 0.28`. httpx provides both sync and async HTTP clients in one package, covers both new backends without separate SDKs, and has no C extension compilation issues. The Ollama backend continues using the Ollama Python SDK — do not replace it with raw httpx. All other extension work uses stdlib (`typing.Protocol`) or already-present packages.

**Core technologies:**
- `httpx >= 0.28`: HTTP client for llama.cpp `/completion` and LM Studio `/api/v1/chat` — only new runtime dependency; sync + async in one package, no C extensions
- `typing.Protocol`: backend interface contract — lighter than ABC, no forced inheritance, easy to mock in tests
- `pydantic >= 2.9` (already present): `BackendResponse` normalized model and new `BackendModel` — no version bump needed
- `ollama >= 0.6` (already present): Ollama backend continues using the SDK directly, not replaced with httpx
- `tenacity >= 9.0` (already present): wrap httpx calls with the same retry pattern as existing Ollama calls

### Expected Features

**Must have (table stakes):**
- `--backend` CLI flag (`ollama` / `llama-cpp` / `lm-studio`) — without it, multi-backend is unusable
- Backend protocol (`typing.Protocol`) with `chat()`, `list_models()`, `is_available()`, `unload_model()` — foundation for all other work
- Ollama backend adapter wrapping existing code — preserves 100% backward compatibility for default users
- llama.cpp backend via httpx posting to `/completion` — extracts `timings` object for native t/s metrics
- LM Studio backend via httpx posting to `/api/v1/chat` — extracts `stats.tokens_per_second`
- Normalized `BackendResponse` model — all durations in seconds, consistent field names
- Auto-detect running backends by probing default ports — students should not need to know ports
- Backend-aware preflight checks — per-backend health checks, non-fatal for optional backends
- `backend` field in all JSON/CSV/Markdown exports — results without backend labels are unauditable
- Setup documentation per backend — students need instructions for installing and starting llama.cpp and LM Studio

**Should have (differentiators):**
- Cross-backend comparison mode — same model, all backends, side-by-side — the killer feature
- Cross-backend comparison report — terminal table and Markdown showing delta percentages with winner highlighted
- Backend health dashboard in `info` subcommand — shows running/stopped, port, loaded models per backend
- `--backend auto` / `--backend all` shorthand — detect running backends, benchmark each, produce combined report
- Explicit model mapping flags (`--llama-cpp-model`, `--lm-studio-model`) — fuzzy matching is unreliable across backend naming conventions

**Defer to v3+:**
- Backend-specific parameter sweep across backends (high complexity, niche use case)
- Concurrent mode for non-Ollama backends (httpx async is possible but adds coordination complexity)
- Cloud API backends (different product: rate limiting, cost, network latency variables)
- llama.cpp server lifecycle management via subprocess (fragile, OS-specific, out of scope)

### Architecture Approach

The recommended architecture introduces a `backends/` sub-package with `base.py` (Protocol + `BackendResponse` + `BackendModel`), one file per adapter (`ollama.py`, `llamacpp.py`, `lmstudio.py`), and a registry in `__init__.py` for lazy import and auto-detection. Existing modules (`runner.py`, `concurrent.py`, `preflight.py`, `cli.py`, `exporters.py`, `system.py`) are modified to accept a `Backend` instance rather than calling the Ollama SDK directly. Modules with no backend coupling (`prompts.py`, `display.py`, `analyze.py`, `compare.py`, `recommend.py`) are untouched. The critical rule: runner.py must never contain `if backend.name == "ollama"` branches — all backend-specific logic stays inside the adapter.

**Major components:**
1. `backends/base.py` — `Backend` Protocol + `BackendResponse` (canonical seconds-based timings) + `BackendModel` — foundation everything depends on
2. `backends/ollama.py` — wraps existing `ollama` SDK calls, converts nanoseconds to seconds, preserves all current behavior including prompt cache detection
3. `backends/llamacpp.py` — httpx client for llama.cpp `/completion`, converts milliseconds to seconds, handles single-model-per-server constraint
4. `backends/lmstudio.py` — httpx client for LM Studio `/api/v1/chat`, derives durations from `stats.tokens_per_second`, handles explicit model load/unload
5. `backends/__init__.py` — registry with `get_backend(name)` (lazy import) and `detect_backends()` (parallel port probing)
6. `runner.py` (modified) — accepts `Backend` parameter, calls `backend.chat()`, no more direct `ollama.chat()` calls
7. `preflight.py` (modified) — per-backend health checks, non-fatal for optional backends, 3-second connection timeout

### Critical Pitfalls

1. **Timing unit mismatch** — Ollama reports nanoseconds, llama.cpp reports milliseconds, LM Studio reports tokens/second directly. Passing llama.cpp ms values through `_ns_to_sec()` (divides by 1e9) produces throughput numbers 1,000,000x too high. Prevention: `_ns_to_sec()` lives only inside `OllamaBackend`; `BackendResponse` stores everything in seconds; each adapter owns its own unit conversion.

2. **Using OpenAI-compatible endpoints** — Both llama.cpp and LM Studio expose `/v1/chat/completions` but these strip all native timing fields, returning only token counts. The tool's entire purpose breaks silently. Prevention: llama.cpp must use `/completion`; LM Studio must use `/api/v1/chat`. Add integration tests asserting timing fields are present in every adapter response.

3. **llama.cpp single-model constraint** — llama.cpp loads exactly one model at server startup; there is no model switching via API. The existing multi-model benchmark loop will silently benchmark the wrong model or fail. Prevention: the Backend Protocol must declare model management capability (`SINGLE_MODEL` vs `MULTI_MODEL`); runner.py uses this to skip the iteration loop for single-model backends.

4. **All-or-nothing preflight failing multi-backend runs** — the current preflight exits fatally if Ollama is unavailable. With multiple backends, some may be running and others not. Prevention: per-backend health checks; `--backend all` skips unavailable backends rather than exiting; connection timeout set to 3 seconds; llama.cpp `/health` returning 503 means "model loading" not "dead" (retry for up to 10 seconds).

5. **OllamaResponse naming coupling** — `SystemInfo.ollama_version`, `BenchmarkResult.response: OllamaResponse | None`, and all exporters are structurally coupled to Ollama. Adding new backends while this coupling exists produces confusing results. Prevention: rename `OllamaResponse` to `BackendResponse` (or scope it inside the Ollama adapter), add `backend: str` to `BenchmarkResult`, change `ollama_version` to `backend_versions: dict[str, str]`. Stage this refactor carefully — it touches all 152 tests.

## Implications for Roadmap

Based on research, the build must proceed bottom-up: define the contract before implementations, prove the contract with the existing backend before adding new ones, and verify the tool remains fully functional at every step. Cross-backend comparison is a Phase 3 concern that depends on Phases 1 and 2 being solid.

### Phase 1: Backend Abstraction Foundation

**Rationale:** Every other phase depends on the normalized response model and Protocol interface. Defining these first sets the contract that all adapters must satisfy. This phase has zero behavior change for end users — it is purely additive (new files) plus careful model renames.
**Delivers:** `backends/` package skeleton, `BackendResponse` Pydantic model, `Backend` Protocol, `OllamaBackend` adapter wrapping existing code, `BenchmarkResult.backend` field, `OllamaResponse` renamed and scoped to Ollama adapter
**Features addressed:** Backend protocol, Ollama adapter, normalized timing metrics, backend label in exports
**Pitfalls avoided:** Timing unit mismatch (#1), OllamaResponse/SystemInfo coupling (#5)
**Success gate:** All 152 existing tests pass with `OllamaBackend` as the default before any new backend is added

### Phase 2: New Backend Adapters + Preflight Redesign

**Rationale:** httpx is the only new dependency. llama.cpp adapter comes first (simpler: single model, clear `timings` fields, well-documented); LM Studio second (more complex: model management API, partially documented `stats` object). Preflight must be redesigned in this phase because the existing all-or-nothing pattern breaks as soon as a second backend exists.
**Delivers:** `backends/llamacpp.py` (httpx + `/completion`), `backends/lmstudio.py` (httpx + `/api/v1/chat`), backend auto-detection by port probing, redesigned per-backend preflight with 3-second timeouts, `--backend` CLI flag, backend-specific startup instructions on failure, setup documentation
**Stack:** `httpx >= 0.28` added to `pyproject.toml`
**Features addressed:** llama.cpp backend, LM Studio backend, auto-detect running backends, backend-aware preflight, `--backend` CLI flag
**Pitfalls avoided:** OpenAI endpoint trap (#2), cascading preflight failures (#4), llama.cpp single-model constraint (#3), port conflicts

### Phase 3: Cross-Backend Comparison

**Rationale:** The killer feature but requires Phase 1 and 2 to be production-stable. Model name mapping must be explicit (no fuzzy matching). Cross-backend comparison builds on existing `compare.py` patterns with the addition of backend-labeled columns.
**Delivers:** `--backend all` orchestration, explicit model mapping via CLI flags, cross-backend comparison report (terminal + Markdown), backend health dashboard in `info` subcommand, result file backward compatibility (`format_version` field, `backend` defaults to `"ollama"` for v1 result files)
**Features addressed:** Cross-backend comparison mode, comparison report, backend health dashboard, `--backend all` shorthand, model name mapping
**Pitfalls avoided:** Model identity crisis (#2 from PITFALLS.md), context window mismatch, result file backward compatibility

### Phase Ordering Rationale

- Protocol-first ordering mirrors the dependency graph: `BackendResponse` is imported by all adapters; adapters are required by runner.py and preflight.py; runner + preflight changes are prerequisites for comparison features
- Phase 1 is purely additive (no behavior change) so the tool ships new infrastructure without regression risk to existing Ollama users
- Preflight redesign belongs in Phase 2 (not Phase 1) because the new per-backend logic is meaningless until at least one non-Ollama backend exists
- Phase 3 is deferred until both new backends are proven to avoid building comparison logic on unstable adapters

### Research Flags

Phases needing deeper research during planning:
- **Phase 2 (LM Studio adapter):** `stats` object fields are MEDIUM confidence. Validate against a running LM Studio server before finalizing normalization code. Specifically verify: are `usage.prompt_tokens` and `usage.completion_tokens` present? Is any prompt eval timing exposed beyond what can be derived from `tokens_per_second`?
- **Phase 2 (llama.cpp adapter):** `timings` object field names (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `predicted_per_second`) should be verified against a running `llama-server` instance before finalizing. Research notes the official docs were rate-limited and field names are MEDIUM confidence.
- **Phase 3 (prompt template handling):** llama.cpp `/completion` does not apply chat templates automatically. Decide whether this is documented-only or requires client-side template injection before building cross-backend comparison. Affects fairness of t/s comparisons.

Phases with standard patterns (skip deeper research):
- **Phase 1 (Protocol + OllamaBackend):** Python Protocol (PEP 544) is well-documented. Ollama SDK behavior is verified from the existing codebase. This is extraction and rename work, not novel design.
- **Phase 2 (httpx integration pattern):** httpx API is stable and well-documented. The integration pattern (create client in `__init__`, POST with timeout, parse JSON) is standard across both new adapters.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | One new dependency (httpx 0.28.1 verified on PyPI). All existing deps confirmed unchanged. |
| Features | HIGH | Table stakes and anti-features are clear. LM Studio stats object has one MEDIUM gap for prompt eval timing. |
| Architecture | HIGH (Ollama), MEDIUM (new backends) | Ollama adapter design verified from codebase analysis. llama.cpp and LM Studio timing field names need validation against real servers. |
| Pitfalls | HIGH | Critical pitfalls (#1, #2, #3, #5) derived from direct codebase reading — 100% certain. Pitfall #4 (preflight) from code + API docs. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **LM Studio prompt eval timing:** `stats.tokens_per_second` is confirmed, but whether the response includes prompt eval duration separately is unverified. If unavailable, `prompt_eval_duration_s` and `prompt_tokens_per_sec` in `BackendResponse` will be computed approximations. Decide and document how to handle this before Phase 2 implementation.
- **llama.cpp timings field names:** Verify `timings.prompt_ms`, `timings.predicted_ms`, `timings.prompt_n`, `timings.predicted_n` exist with these exact names. Plan one session of integration testing against a real llama-server at the start of Phase 2.
- **LM Studio chat template behavior:** Whether `/api/v1/chat` applies the model's chat template automatically needs verification. If it does not (unlike Ollama), Phase 3 comparisons are apples-to-oranges on prompt token counts.
- **Test fixture scope of Phase 1 refactor:** 152 tests reference `OllamaResponse` and nanosecond-based field patterns. The rename work in Phase 1 will require updating most test fixtures. Estimate this carefully — it is the highest-effort task in the phase.

## Sources

### Primary (HIGH confidence)
- Existing codebase (`runner.py`, `models.py`, `preflight.py`, `exporters.py`, `cli.py`) — coupling points, timing field usage, test count
- `github.com/ggml-org/llama.cpp/tools/server/README.md` — llama.cpp API endpoints, timings object structure, health endpoint, default port 8080
- `lmstudio.ai/docs/api/rest-api` — LM Studio native v1 API, `/api/v1/chat`, `stats` object with `tokens_per_second`, model management endpoints
- `pypi.org/project/httpx/` — httpx v0.28.1, Python compat, release date 2024-12-06
- `.planning/PROJECT.md` — constraints, scope, known Ollama bug (Qwen 3.5 MoE hang, issues #14579 and #14662)

### Secondary (MEDIUM confidence)
- LM Studio docs (main) — port 1234, model management, auth token pattern — accessed 2026-03-14; `stats` object sub-fields beyond `tokens_per_second` not exhaustively documented
- llama.cpp `/props` endpoint behavior — training data knowledge; verify field names against actual server before implementation

### Tertiary (LOW confidence)
- LM Studio prompt template application behavior — not verified from any source; requires runtime testing against real LM Studio instance
- llama.cpp compute backend detection via `/props` (Metal vs CPU on macOS) — training data inference; verify before implementing warmup or system info enhancements

---
*Research completed: 2026-03-14*
*Ready for roadmap: yes*
