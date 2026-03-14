# Technology Stack

**Project:** LLM Benchmark v2.0 -- Multi-Backend Support
**Researched:** 2026-03-14
**Mode:** Ecosystem research for llama.cpp, LM Studio backend integration
**Scope:** NEW dependencies and integration points only. Ollama SDK, Rich, Pydantic, tenacity, argparse, pytest, ruff are validated from v1.0 and not re-researched.

---

## Executive Summary

One new runtime dependency: **httpx >=0.28**. That is it. The previous STACK.md (v1.0) correctly said "do not add httpx" because Ollama SDK handled all HTTP. Now the project talks to two additional HTTP APIs (llama.cpp server, LM Studio server) that have no Python SDKs. httpx is the right tool. Everything else is stdlib or already present.

---

## New Runtime Dependency

### HTTP Client: httpx

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| httpx | >=0.28 | HTTP client for llama.cpp and LM Studio native REST APIs | Sync + async in one package, `requests`-like API, connection pooling, timeout objects, JSON helpers. Neither llama.cpp nor LM Studio has an official Python SDK. Both expose HTTP/JSON APIs. httpx is the lightest correct choice. |

**Version rationale:** 0.28.1 is the latest stable (released 2024-12-06). Supports Python >=3.8, so compatible with project's Python 3.12+ requirement. The 1.0 pre-release exists but is not stable -- pin to >=0.28 for now.

**Confidence: HIGH** -- version verified on PyPI.

**Why httpx over alternatives:**

| Alternative | Why Not |
|-------------|---------|
| `requests` | No async support. The project has async concurrent benchmarking mode -- httpx provides `AsyncClient` without a second dependency. |
| `aiohttp` | Async-only. Sync benchmark mode (the default) would need event loop boilerplate. httpx does both. |
| `urllib3` / stdlib `urllib` | Too low-level. No `response.json()`, no async, verbose timeout/error handling. |
| `llama-cpp-python` | This is a C++ binding that *embeds* llama.cpp as a library. The project talks to llama.cpp **server** via HTTP. The binding requires C++ compilation toolchain, violating the "students use Homebrew or prebuilt" constraint. |
| `lmstudio` SDK | LM Studio has a TypeScript SDK only. No maintained Python SDK exists. |

### No Other New Dependencies

Everything else needed is already present or in stdlib:

| What | Status | Notes |
|------|--------|-------|
| `typing.Protocol` | stdlib (3.8+) | Backend protocol interface. Zero dependencies. |
| `pydantic` | Already present (>=2.9) | Add new response models for llama.cpp and LM Studio timing data. No version bump needed. |
| `tenacity` | Already present (>=9.0) | Reuse retry logic for all backends. Wrap httpx calls identically to Ollama calls. |
| `rich` | Already present (>=14.0) | Backend selection UI, comparison tables. No changes. |
| `ollama` | Already present (>=0.6) | Ollama backend continues using the SDK. Do NOT replace with httpx. |

---

## Backend API Integration Details

### llama.cpp Server

| Aspect | Detail | Confidence |
|--------|--------|------------|
| Default port | `127.0.0.1:8080` | HIGH (official server README) |
| Health check | `GET /health` -- returns `{"status": "ok"}` (200) or 503 during model load | HIGH |
| Chat endpoint | `POST /v1/chat/completions` (OpenAI-compat but WITH `timings` object) | HIGH |
| Native completion | `POST /completion` (prompt-based, also has `timings`) | HIGH |
| Server start | `llama-server -m /path/to/model.gguf -c 2048` | HIGH |
| Model loading | ONE model at startup via `--model` / `-m` flag. No dynamic model swap. | HIGH |
| Unload model | Not possible via API. Requires server restart with different `--model`. | HIGH |

**Timings object in response:**

```json
{
  "timings": {
    "prompt_n": 42,
    "prompt_per_second": 312.5,
    "predicted_n": 128,
    "predicted_per_second": 45.2,
    "tokens_evaluated": 42,
    "cache_n": 0
  }
}
```

**Use `/v1/chat/completions`, not `/completion`.** Reason: `/v1/chat/completions` accepts chat messages (role/content), matching the Ollama and LM Studio request format. `/completion` takes raw text prompts -- a different interface shape. Using the chat endpoint keeps the backend protocol uniform. The timings object is included in both endpoints.

### LM Studio Server

| Aspect | Detail | Confidence |
|--------|--------|------------|
| Default port | `127.0.0.1:1234` | HIGH (official docs) |
| Native chat | `POST /api/v1/chat` (v1 REST API, released in LM Studio 0.4.0) | HIGH |
| Model list | `GET /api/v1/models` | HIGH |
| Load model | `POST /api/v1/models/load` | HIGH |
| Unload model | `POST /api/v1/models/unload` | HIGH |
| Authentication | Optional Bearer token `Authorization: Bearer $LM_API_TOKEN` | HIGH |
| Context length | Configurable per-request | HIGH |
| Model identifiers | HuggingFace-style: `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` | HIGH |
| Server start | GUI or `lms server start --port 1234` | HIGH |

**Stats in response:**

```json
{
  "stats": {
    "tokens_per_second": 42.5,
    "draft_model": null,
    "total_draft_tokens_count": 0,
    "accepted_draft_tokens_count": 0,
    "rejected_draft_tokens_count": 0
  }
}
```

**Use `/api/v1/chat`, NOT `/v1/chat/completions`.** The OpenAI-compatible endpoint (`/v1/chat/completions`) strips the `stats` object with `tokens_per_second`. The native v1 endpoint preserves performance metrics. This was already identified in PROJECT.md.

### Ollama (existing -- no changes)

| Aspect | Detail |
|--------|--------|
| Default port | `127.0.0.1:11434` |
| Interface | `ollama` Python SDK (wraps httpx internally) |
| Chat | `ollama.chat()` / `ollama.AsyncClient().chat()` |
| Timing data | `eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration` (nanoseconds) |
| Unload | `ollama.generate(model=name, prompt="", keep_alive=0)` |

---

## Timing Field Normalization Map

Each backend reports performance metrics differently. The backend abstraction layer must normalize into a common shape.

| Metric | Ollama | llama.cpp | LM Studio |
|--------|--------|-----------|-----------|
| Response tokens/sec | Compute: `eval_count / (eval_duration / 1e9)` | Direct: `timings.predicted_per_second` | Direct: `stats.tokens_per_second` |
| Response token count | `eval_count` | `timings.predicted_n` | `usage.completion_tokens` |
| Prompt token count | `prompt_eval_count` | `timings.prompt_n` | `usage.prompt_tokens` |
| Prompt eval speed | Compute: `prompt_eval_count / (prompt_eval_duration / 1e9)` | Direct: `timings.prompt_per_second` (if present) | Not directly reported -- derive from wall clock |
| Total duration | `total_duration` (nanoseconds) | Compute from wall clock | Compute from wall clock |
| Prompt cache signal | `prompt_eval_count == -1` | `timings.cache_n > 0` | Unknown -- needs runtime testing (LOW confidence) |

**Normalization target model (Pydantic):**

```python
class BenchmarkTimings(BaseModel):
    """Backend-agnostic timing results from a single benchmark run."""
    response_tokens: int          # tokens generated
    response_seconds: float       # time to generate
    response_tokens_per_sec: float  # pre-computed or derived
    prompt_tokens: int            # tokens in prompt
    prompt_seconds: float         # time to eval prompt
    prompt_tokens_per_sec: float  # pre-computed or derived
    total_duration_seconds: float # wall-clock total
    prompt_cached: bool = False   # prompt was served from cache
```

Each backend adapter converts its native response into this shape. The existing `OllamaResponse` model stays for Ollama-specific internals, but `BenchmarkResult` gets refactored to use `BenchmarkTimings` instead of `OllamaResponse` directly.

---

## GGUF Model Path Handling (llama.cpp)

**Design decision: The benchmark tool does NOT manage GGUF files or the llama.cpp server lifecycle.**

| What the tool DOES | What the tool does NOT do |
|---------------------|--------------------------|
| Detect running llama.cpp server at configured host:port | Download GGUF files |
| Query `/health` to confirm readiness | Start/stop the llama.cpp server |
| Query model info from server (name of loaded model) | Swap models (requires server restart) |
| Benchmark whatever model is loaded | Manage GGUF file paths |
| Report model name in results | Validate GGUF quantization |

**Rationale:**
1. Students install llama.cpp via Homebrew (`brew install llama.cpp`) or prebuilt binaries
2. Students start the server manually: `llama-server -m ~/models/llama-3.2-1b.Q4_K_M.gguf`
3. GGUF files live in arbitrary locations -- no standard registry like Ollama
4. Managing file paths, downloads, and server restarts adds massive complexity for minimal gain
5. The setup guide will document how to download and run models

**Cross-backend comparison implication:** For "same model on all backends", the student must manually ensure the same model is loaded in each backend. The tool can warn if model names don't appear to match but cannot enforce this (model naming differs across backends: `llama3.2:1b` in Ollama vs `llama-3.2-1b.Q4_K_M.gguf` in llama.cpp vs `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` in LM Studio).

---

## Backend Auto-Detection

Use httpx to probe known ports in parallel:

```python
async def detect_backends() -> list[str]:
    """Probe default ports to find running backends."""
    checks = {
        "ollama": ("http://localhost:11434/api/tags", 200),
        "llama-cpp": ("http://localhost:8080/health", 200),
        "lm-studio": ("http://localhost:1234/api/v1/models", 200),
    }
    # ... parallel httpx.AsyncClient.get() with short timeout
```

The Ollama check can also use `ollama.list()` (existing code), but httpx provides a uniform mechanism. For auto-detection specifically, httpx is cleaner than mixing the Ollama SDK with raw HTTP for the other two.

---

## Backend Protocol Design

Use `typing.Protocol` (not ABC) because:
- No forced inheritance -- backends can be plain classes
- Easier to mock in tests (any object satisfying the shape works)
- More Pythonic for structural subtyping
- The Ollama backend wraps the existing SDK; others use httpx. Different internals, same interface.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def default_port(self) -> int: ...

    def is_available(self, host: str, port: int) -> bool:
        """Check if the backend server is reachable."""
        ...

    def list_models(self, host: str, port: int) -> list[str]:
        """Return names of available/loaded models."""
        ...

    def chat(
        self,
        model: str,
        prompt: str,
        host: str,
        port: int,
        num_ctx: int | None = None,
    ) -> BenchmarkTimings:
        """Send a prompt and return normalized timing results."""
        ...

    def unload_model(self, model: str, host: str, port: int) -> bool:
        """Release model from memory. Return False if not supported."""
        ...
```

---

## Configuration Constants to Add

```python
# config.py additions
DEFAULT_BACKEND: str = "ollama"
BACKEND_PORTS: dict[str, int] = {
    "ollama": 11434,
    "llama-cpp": 8080,
    "lm-studio": 1234,
}
BACKEND_HEALTH_ENDPOINTS: dict[str, str] = {
    "ollama": "/api/tags",
    "llama-cpp": "/health",
    "lm-studio": "/api/v1/models",
}
```

---

## Updated pyproject.toml Dependencies

```toml
dependencies = [
    "ollama>=0.6",
    "pydantic>=2.9",
    "rich>=14.0",
    "tenacity>=9.0",
    "httpx>=0.28",       # NEW: HTTP client for llama.cpp and LM Studio APIs
]
```

**Total new runtime dependencies: 1** (httpx). httpx is pure-Python, well-maintained (22k GitHub stars), and has no C extension compilation issues on any platform.

---

## What NOT to Change

| Component | Keep As-Is | Reason |
|-----------|-----------|--------|
| `ollama` SDK for Ollama backend | Yes | Works, tested, has AsyncClient. Do not replace with raw httpx. |
| `argparse` CLI | Yes | Add `--backend` flag, do not migrate to click. |
| `threading` timeout | Yes | Reuse for httpx-based backends too. |
| `tenacity` retry | Yes | Wrap httpx calls with same retry pattern as Ollama calls. |
| Test framework (pytest) | Yes | Mock httpx responses with `httpx.MockTransport` or `respx` for new backends. |

---

## Sources

| Source | What | Confidence |
|--------|------|------------|
| github.com/ggml-org/llama.cpp/tools/server/README.md | llama.cpp server API, timings object, /health endpoint, default port | HIGH (official repo) |
| lmstudio.ai/docs/api/rest-api | LM Studio native v1 API, /api/v1/chat, stats object | HIGH (official docs) |
| lmstudio.ai/docs (main) | LM Studio port, model management endpoints, auth | HIGH (official docs) |
| pypi.org/project/httpx/ | httpx v0.28.1, Python compat, release date | HIGH (PyPI) |
| Existing codebase (runner.py, models.py, config.py, preflight.py) | Current architecture, integration points | HIGH (source code) |
| PROJECT.md | Constraints, decisions, scope | HIGH (project definition) |

---

*Researched: 2026-03-14. Supersedes v1.0 STACK.md for the multi-backend milestone.*
