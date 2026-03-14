# Architecture: Multi-Backend Integration

**Domain:** LLM benchmarking tool -- adding llama.cpp and LM Studio backends
**Researched:** 2026-03-14
**Confidence:** HIGH (codebase analysis) / MEDIUM (llama.cpp/LM Studio API details from docs + PROJECT.md findings)

## Current Architecture Snapshot

The existing codebase is a flat Python package with these key modules:

```
llm_benchmark/
    cli.py          # Argparse + dispatch to modes (standard, concurrent, sweep)
    runner.py       # Benchmark execution -- directly calls ollama.chat()
    concurrent.py   # Async concurrent mode -- directly calls ollama.AsyncClient
    sweep.py        # Parameter sweep -- delegates to runner.py
    models.py       # Pydantic models: OllamaResponse, BenchmarkResult, ModelSummary
    preflight.py    # Ollama connectivity, model list, RAM checks
    system.py       # Hardware detection, Ollama version
    exporters.py    # JSON/CSV/Markdown writers
    menu.py         # Interactive mode selection
    config.py       # Constants, Rich console singleton
    prompts.py      # Prompt sets
    display.py      # Bar chart rendering
    analyze.py      # Result analysis
    compare.py      # Result comparison
    recommend.py    # Model recommendations
```

**Critical observation:** Ollama is deeply coupled into runner.py (imports `ollama` directly, calls `ollama.chat()` inline), concurrent.py (uses `ollama.AsyncClient`), preflight.py (calls `ollama.list()`, `ollama.show()`), system.py (gets `ollama --version`), and sweep.py (delegates to runner). There is no abstraction layer -- Ollama SDK calls are scattered across 5+ modules.

## Recommended Architecture for Multi-Backend

### Design: Backend Protocol + Registry

Use a Python `Protocol` (PEP 544) rather than an ABC. This is lighter weight, doesn't require inheritance, and matches the project's constraint of keeping things simple for students.

```python
# backends/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Backend(Protocol):
    """Contract every inference backend must satisfy."""

    name: str  # "ollama", "llama-cpp", "lm-studio"

    def is_available(self) -> bool:
        """Check if this backend server is reachable."""
        ...

    def list_models(self) -> list[BackendModel]:
        """Return models available on this backend."""
        ...

    def chat(
        self,
        model: str,
        prompt: str,
        options: dict | None = None,
    ) -> BackendResponse:
        """Send prompt, return normalized response with timing metrics."""
        ...

    def unload_model(self, model: str) -> bool:
        """Release model from memory. Return True on success."""
        ...
```

**Why Protocol over ABC:** The existing OllamaClient code works without inheriting from anything. Protocol lets us type-check structurally (duck typing with teeth) while keeping implementations simple standalone classes. Students can understand `class OllamaBackend:` without needing to grok inheritance hierarchies.

### New Normalized Response Model

The key challenge: each backend reports timing in different formats and units. The solution is a `BackendResponse` Pydantic model that normalizes everything to seconds and token counts.

```python
# backends/base.py
class BackendResponse(BaseModel):
    """Normalized response from any backend. All durations in seconds."""

    backend: str                    # "ollama" | "llama-cpp" | "lm-studio"
    model: str                      # Model identifier (backend-specific)
    content: str                    # Generated text
    prompt_eval_count: int          # Input tokens processed
    prompt_eval_duration_s: float   # Seconds to process prompt
    eval_count: int                 # Output tokens generated
    eval_duration_s: float          # Seconds to generate output
    total_duration_s: float         # Total request duration in seconds
    load_duration_s: float = 0.0    # Model load time in seconds
    prompt_cached: bool = False     # Was prompt evaluation skipped?
```

**This is the central design decision.** Every backend adapter's job is to translate its native response into this normalized structure. Downstream code (runner.py, compute_averages, exporters) works exclusively with `BackendResponse` -- never with backend-specific data.

### Timing Normalization Map

| Field | Ollama | llama.cpp | LM Studio |
|-------|--------|-----------|-----------|
| **Source endpoint** | `/api/chat` | `/completion` | `/api/v1/chat` |
| **prompt_eval_count** | `prompt_eval_count` (int) | `timings.prompt_n` (int) | Parsed from `usage.prompt_tokens` |
| **prompt_eval_duration** | `prompt_eval_duration` (nanoseconds) | `timings.prompt_ms` (milliseconds) | Computed: `prompt_tokens / timings.prompt_per_second` |
| **eval_count** | `eval_count` (int) | `timings.predicted_n` (int) | Parsed from `usage.completion_tokens` |
| **eval_duration** | `eval_duration` (nanoseconds) | `timings.predicted_ms` (milliseconds) | Computed: `completion_tokens / stats.tokens_per_second` |
| **total_duration** | `total_duration` (nanoseconds) | Sum of prompt_ms + predicted_ms | Wall clock measured by client |
| **load_duration** | `load_duration` (nanoseconds) | Not reported (model pre-loaded) | Not reported |
| **prompt_cached** | `prompt_eval_count == -1` | `timings.prompt_n == 0` | Not applicable |
| **Rate fields** | Not provided (computed) | `timings.predicted_per_second` | `stats.tokens_per_second` |

**Normalization strategy per backend:**

- **Ollama:** Divide nanosecond values by 1,000,000,000 to get seconds. This is the existing `_ns_to_sec()` function.
- **llama.cpp:** Divide millisecond values by 1,000 to get seconds. When rate fields are available (e.g., `predicted_per_second`), cross-validate with `predicted_n / (predicted_ms / 1000)`.
- **LM Studio:** `stats.tokens_per_second` is a direct rate. Compute duration as `eval_count / tokens_per_second` for the normalized duration field. If `usage.prompt_tokens` and total time are available, derive prompt duration by subtraction.

**Confidence note:** The llama.cpp `timings` object structure (prompt_n, prompt_ms, predicted_n, predicted_ms, predicted_per_second) is based on PROJECT.md research findings and training data. The LM Studio `stats.tokens_per_second` field was verified via official docs. MEDIUM confidence on exact field names -- validate against running servers during implementation.

### Package Layout Changes

```
llm_benchmark/
    backends/               # NEW package
        __init__.py         # Registry: get_backend(), list_backends(), detect_backends()
        base.py             # Backend Protocol, BackendResponse, BackendModel
        ollama.py           # OllamaBackend (extract from runner.py + preflight.py)
        llamacpp.py         # LlamaCppBackend (new, uses httpx)
        lmstudio.py         # LMStudioBackend (new, uses httpx)
    models.py               # MODIFIED: BenchmarkResult.response becomes BackendResponse
    runner.py               # MODIFIED: accepts Backend instead of calling ollama directly
    concurrent.py           # MODIFIED: accepts Backend
    sweep.py                # MODIFIED: accepts Backend
    preflight.py            # MODIFIED: delegates to Backend.is_available/list_models
    system.py               # MODIFIED: backend-aware version detection
    cli.py                  # MODIFIED: --backend flag, auto-detect
    menu.py                 # MODIFIED: backend selection in interactive mode
    exporters.py            # MODIFIED: include backend name in output
    config.py               # MODIFIED: add DEFAULT_BACKEND, backend port constants
    # Unchanged: prompts.py, display.py, analyze.py, compare.py, recommend.py
```

### Component Boundaries (New + Modified)

| Component | Status | Responsibility | Communicates With |
|-----------|--------|---------------|-------------------|
| `backends/__init__.py` | NEW | Registry: resolve backend by name, auto-detect | cli.py, menu.py |
| `backends/base.py` | NEW | Protocol definition, BackendResponse model | All backend implementations |
| `backends/ollama.py` | NEW (extracted) | Ollama SDK wrapper, timing normalization (ns -> s) | runner.py via Protocol |
| `backends/llamacpp.py` | NEW | HTTP client for llama.cpp /completion, timing normalization (ms -> s) | runner.py via Protocol |
| `backends/lmstudio.py` | NEW | HTTP client for LM Studio /api/v1/chat, timing normalization | runner.py via Protocol |
| `runner.py` | MODIFIED | Receives `Backend` instance, calls `backend.chat()` | backends via Protocol |
| `models.py` | MODIFIED | BenchmarkResult uses BackendResponse instead of OllamaResponse | All modules |
| `preflight.py` | MODIFIED | Delegates connectivity/model checks to backend | backends |
| `cli.py` | MODIFIED | `--backend` flag, dispatches to correct backend | backends registry |
| `exporters.py` | MODIFIED | Includes backend name in reports | No new deps |
| `system.py` | MODIFIED | Reports versions for active backend | backends |

### New Data Flow

```
CLI Input (--backend=llama-cpp)
    |
    v
cli.py: resolve backend
    |
    +---> backends/__init__.py: get_backend("llama-cpp") -> LlamaCppBackend()
    |
    v
preflight.py: backend.is_available()? backend.list_models()?
    |
    v
runner.py: benchmark_model(backend=backend, model_name=..., ...)
    |
    +---> backend.chat(model, prompt) -> BackendResponse (normalized, in seconds)
    |
    +---> BenchmarkResult(response=backend_response)
    |
    v
compute_averages(results)  # Works unchanged -- uses seconds-based fields
    |
    v
exporters: export_json/csv/markdown  # Add backend field to output
```

## Integration Points: What Changes in Existing Code

### 1. runner.py -- The Biggest Change

**Current state:** `run_single_benchmark()` calls `ollama.chat()` directly (line 312) and constructs `OllamaResponse` via `OllamaResponse.model_validate(raw_response)` (line 359).

**Required change:** Accept a `Backend` parameter and call `backend.chat()` instead. The `_parse_response` inner function is replaced by the backend's own normalization.

```python
# BEFORE (runner.py line 308-337)
def _run_benchmark() -> dict:
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options=options,
    )
    return response

# AFTER
def _run_benchmark() -> BackendResponse:
    return backend.chat(
        model=model_name,
        prompt=prompt,
        options=options,
    )
```

**Impact:** The `OllamaResponse` Pydantic model becomes internal to `backends/ollama.py`. `BenchmarkResult.response` changes type from `OllamaResponse | None` to `BackendResponse | None`.

**Functions that need the `backend` parameter threaded through:**
- `run_single_benchmark()` -- core change
- `benchmark_model()` -- passes backend to run_single_benchmark
- `warmup_model()` -- calls backend.chat with short prompt
- `unload_model()` -- calls backend.unload_model
- `detect_num_ctx()` -- Ollama-specific, may not apply to other backends

### 2. models.py -- Response Type Change

**Current:** `BenchmarkResult.response: OllamaResponse | None`
**After:** `BenchmarkResult.response: BackendResponse | None`

The `_ns_to_sec()` helper becomes unnecessary in models.py since `BackendResponse` stores durations in seconds. However, `_ns_to_sec` is used in compute_averages and exporters, so it should be kept as-is until all callers are migrated.

**Migration path:** Add `BackendResponse` to models.py (or import from backends.base). Keep `OllamaResponse` temporarily for backward compatibility. `BenchmarkResult.response` becomes a Union type during transition, then drops `OllamaResponse` once all backends are implemented.

**Better approach:** Since `BackendResponse` stores values in seconds, add computed properties that match the current field access patterns:

```python
class BackendResponse(BaseModel):
    eval_count: int
    eval_duration_s: float
    # ...

    @property
    def eval_duration(self) -> int:
        """Nanoseconds, for backward compatibility with existing code."""
        return int(self.eval_duration_s * 1_000_000_000)
```

This avoids a massive find-and-replace across runner.py, compute_averages, and exporters. Remove the shim properties once all callers are updated to use `_s` fields.

### 3. preflight.py -- Backend-Aware Checks

**Current:** Hardcoded to `ollama.list()`, `ollama.show()`, `check_ollama_installed()`.

**After:** Delegate to the selected backend:

```python
def run_preflight_checks(backend: Backend, skip_models=None, skip_checks=False):
    # 1. Check if backend server is reachable
    if not backend.is_available():
        console.print(f"[red]Cannot connect to {backend.name}[/red]")
        _print_setup_instructions(backend.name)
        sys.exit(1)

    # 2. Get available models
    models = backend.list_models()
    # ... filter, RAM checks, etc.
```

Ollama-specific checks (install binary, `ollama serve` instructions) move into `backends/ollama.py` as helper methods.

### 4. concurrent.py -- Backend-Aware Async

**Current:** Uses `ollama.AsyncClient()` directly.

**Challenge:** llama.cpp and LM Studio don't have Python SDK async clients. They use HTTP APIs.

**Solution:** The Backend Protocol should include an `async_chat()` method, or concurrent.py should use `httpx.AsyncClient` directly through the backend. Simpler approach: add an optional `async_chat` method to the protocol. If not implemented, concurrent mode falls back to `ThreadPoolExecutor` with sync `backend.chat()`.

```python
# In concurrent.py
async def _single_request(backend, model, prompt, ...):
    if hasattr(backend, 'async_chat'):
        return await backend.async_chat(model, prompt)
    else:
        # Fallback: run sync chat in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, backend.chat, model, prompt)
```

### 5. cli.py -- New --backend Flag

Add `--backend` to the run parser:

```python
run_parser.add_argument(
    "--backend",
    choices=["ollama", "llama-cpp", "lm-studio", "auto"],
    default="auto",
    help="Inference backend (default: auto-detect)",
)
```

The `auto` option checks which backends are running and uses the first available (preference: ollama > lm-studio > llama-cpp for backward compat).

### 6. exporters.py -- Add Backend Field

Minimal change: include `backend` in JSON output and CSV/Markdown headers. The `_result_to_dict` function adds `"backend": result.response.backend` to the dict.

### 7. system.py -- Backend Version Detection

**Current:** Hardcoded `get_ollama_version()` and `SystemInfo.ollama_version`.

**After:** `SystemInfo` gains a `backend_versions: dict[str, str]` field. Or simpler: rename `ollama_version` to `backend_version` and populate from the active backend.

## Model Naming Across Backends

This is a significant pain point. Each backend uses different model identifiers:

| Aspect | Ollama | llama.cpp | LM Studio |
|--------|--------|-----------|-----------|
| **Model ID format** | `llama3.2:1b` | GGUF file path or loaded model name | `llama-3.2-1b` or download slug |
| **Listing** | `ollama list` / API | Files in model directory | `GET /api/v1/models` |
| **Loading** | Automatic on first use | Must load via `/load` or `--model` flag | Auto-loads or via `/api/v1/load` |
| **Same model check** | Exact name match | Filename-based | Slug-based |

**Strategy: Canonical model names with backend-specific resolution.**

```python
class BackendModel(BaseModel):
    """A model available on a specific backend."""
    name: str           # Display name: "llama3.2:1b"
    backend_id: str     # Backend-specific ID: path, slug, or name
    backend: str        # "ollama", "llama-cpp", "lm-studio"
    size_bytes: int | None = None
    parameter_count: str | None = None  # "1B", "7B", etc.
```

For cross-backend comparison, users must specify which model to compare. The tool cannot automatically map "llama3.2:1b" (Ollama) to "/models/llama-3.2-1B-Q4_K_M.gguf" (llama.cpp). Instead:

1. **Each backend lists its models independently.** The `list_models()` return includes the display name.
2. **Cross-backend comparison is user-directed.** User selects which models to compare. The report shows model names with backend labels.
3. **Optional: fuzzy matching helper.** A utility that suggests matches based on substrings (e.g., "llama3.2" appears in both backend model lists). This is a convenience, not a requirement.

## Backend Implementation Details

### OllamaBackend (Extract from Existing Code)

- Wraps the `ollama` Python SDK (already a dependency)
- `chat()`: calls `ollama.chat()`, converts nanosecond fields to seconds
- `list_models()`: calls `ollama.list()`
- `unload_model()`: calls `ollama.generate(keep_alive=0)`
- `is_available()`: calls `ollama.list()` in try/except
- `async_chat()`: uses `ollama.AsyncClient` (for concurrent mode)
- Preserves all existing Ollama-specific behavior (prompt caching detection, num_ctx auto-detect)

### LlamaCppBackend (New, HTTP via httpx)

- Talks to `llama-server` (the llama.cpp HTTP server) at `http://localhost:8080` by default
- `chat()`: POST to `/completion` with `{"prompt": prompt, "n_predict": -1}`
- Native timing in `timings` object: `prompt_ms`, `predicted_ms`, `prompt_n`, `predicted_n`, `predicted_per_second`, `prompt_per_second`
- Normalization: `ms / 1000.0` for durations
- `list_models()`: GET `/props` returns the currently loaded model. llama.cpp loads one model at a time, so this returns a single-item list
- `unload_model()`: Not needed (single model server). Return True.
- `is_available()`: GET `/health` returns `{"status": "ok"}`
- No async client needed initially -- httpx supports async natively

**llama.cpp quirk:** The server loads exactly one model at startup (`--model path.gguf`). There's no model switching. For multi-model benchmarks, the user must restart the server with a different model, or the tool could automate this via subprocess.

### LMStudioBackend (New, HTTP via httpx)

- Talks to LM Studio's built-in server at `http://localhost:1234` by default
- `chat()`: POST to `/api/v1/chat` with messages format
- Response includes `stats.tokens_per_second` for generation rate
- Normalization: derive `eval_duration_s = eval_count / stats.tokens_per_second`
- `list_models()`: GET `/api/v1/models` returns loaded and available models
- `unload_model()`: POST `/api/v1/unload` with model identifier
- `is_available()`: GET `/api/v1/models` in try/except

**LM Studio quirk:** It can load multiple models simultaneously (unlike llama.cpp). Model management is richer but the API is also more complex.

### Backend Registry

```python
# backends/__init__.py
_BACKENDS = {
    "ollama": ("llm_benchmark.backends.ollama", "OllamaBackend"),
    "llama-cpp": ("llm_benchmark.backends.llamacpp", "LlamaCppBackend"),
    "lm-studio": ("llm_benchmark.backends.lmstudio", "LMStudioBackend"),
}

def get_backend(name: str, **kwargs) -> Backend:
    """Lazy-import and instantiate a backend by name."""
    module_path, class_name = _BACKENDS[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)

def detect_backends() -> list[str]:
    """Probe all known backends and return names of reachable ones."""
    available = []
    for name in _BACKENDS:
        try:
            backend = get_backend(name)
            if backend.is_available():
                available.append(name)
        except Exception:
            pass
    return available
```

Lazy imports prevent `import httpx` from failing when httpx isn't installed (it's only needed for llama.cpp and LM Studio backends).

## Anti-Patterns to Avoid

### Anti-Pattern: OpenAI-Compatible Endpoints

Both llama.cpp and LM Studio offer OpenAI-compatible `/v1/chat/completions` endpoints. Do NOT use them. The PROJECT.md explicitly states: "OpenAI endpoint strips timing data." The native endpoints return server-side timing metrics; the OpenAI-compat endpoints return only token counts without timing breakdown.

### Anti-Pattern: Backend-Specific Code in runner.py

Runner should never contain `if backend.name == "ollama": ...` branches. All backend-specific logic (timing normalization, model listing, connection handling) stays inside the backend class. Runner only calls Protocol methods.

### Anti-Pattern: Forcing All Backends to Support All Features

llama.cpp doesn't support multi-model serving. LM Studio doesn't report nanosecond-level timing. Don't fake missing data or add unsupported features. Instead, document what each backend supports and handle gracefully:

```python
class LlamaCppBackend:
    def list_models(self) -> list[BackendModel]:
        # Only returns the one currently loaded model
        props = httpx.get(f"{self.base_url}/props").json()
        return [BackendModel(
            name=props.get("model_alias", "unknown"),
            backend_id=props.get("model_path", "unknown"),
            backend="llama-cpp",
        )]
```

### Anti-Pattern: Shared HTTP Client Configuration

Don't create a generic `HTTPBackend` base class that both llama.cpp and LM Studio inherit from. Their APIs are different enough that shared HTTP plumbing creates more confusion than it saves. Each backend manages its own httpx client.

## Dependency Impact

### New Dependencies

| Package | Purpose | When Needed |
|---------|---------|-------------|
| `httpx` | HTTP client for llama.cpp and LM Studio | Only when those backends are used |

**Installation strategy:** Make httpx an optional dependency:

```toml
# pyproject.toml
dependencies = [
    "ollama>=0.6",
    "pydantic>=2.9",
    "rich>=14.0",
    "tenacity>=9.0",
]

[project.optional-dependencies]
all-backends = ["httpx>=0.27"]
```

If a user tries `--backend llama-cpp` without httpx installed, the lazy import fails with a clear error: "httpx is required for llama-cpp backend. Install with: pip install llm-benchmark[all-backends]"

**Alternative (simpler):** Just add httpx to core dependencies. It's lightweight (no C extensions), well-maintained, and students won't need to think about optional deps. Given the project's student audience, this is the better choice.

## Suggested Build Order

Build from the bottom up, keeping the tool functional at every step.

### Step 1: Backend Protocol + BackendResponse Model
**New files:** `backends/__init__.py`, `backends/base.py`
**Why first:** Defines the contract. Everything else depends on this.
**Risk:** Low -- pure type definitions, no behavior changes.

### Step 2: OllamaBackend (Extract)
**New file:** `backends/ollama.py`
**Modified:** Nothing yet -- OllamaBackend exists alongside current code.
**Why second:** Proves the Protocol design works with known behavior. Write this as a wrapper around existing ollama SDK calls, test that it produces correct BackendResponse values by comparing against current OllamaResponse parsing.
**Risk:** Low -- additive only.

### Step 3: Wire runner.py to Use Backend
**Modified:** `runner.py` (accept `backend` param), `models.py` (BackendResponse type)
**Why third:** This is the critical integration point. Once runner.py works with OllamaBackend, the architecture is proven. All existing tests must still pass with Ollama as default backend.
**Risk:** MEDIUM -- touches the core benchmark loop. Must preserve backward compatibility.

### Step 4: Wire preflight.py, cli.py, menu.py
**Modified:** `preflight.py`, `cli.py`, `menu.py`, `system.py`
**Why fourth:** Thread the backend through the full CLI flow. `--backend ollama` works end-to-end.
**Risk:** Low-medium -- plumbing changes, but behavior unchanged for default backend.

### Step 5: LlamaCppBackend
**New file:** `backends/llamacpp.py`
**Modified:** `config.py` (add LLAMACPP_DEFAULT_PORT)
**Why fifth:** Second backend proves the abstraction. Requires a running llama-server for integration testing.
**Risk:** MEDIUM -- new HTTP client code, timing normalization to validate.

### Step 6: LMStudioBackend
**New file:** `backends/lmstudio.py`
**Why sixth:** Third backend, same pattern. Requires LM Studio running for integration testing.
**Risk:** Low (pattern established by step 5).

### Step 7: Auto-Detection + Cross-Backend Comparison
**Modified:** `backends/__init__.py` (detect_backends), `cli.py`, `exporters.py`
**Why last:** Requires all backends to exist. Cross-backend comparison is a feature on top of the abstraction.
**Risk:** Low -- uses established backend protocol.

## Key Architectural Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Protocol over ABC | Lightweight, no inheritance needed, Pythonic |
| BackendResponse normalizes to seconds | Avoids unit confusion; all downstream code works in one unit |
| Lazy backend imports | Don't break `--backend ollama` if httpx isn't installed |
| httpx as core dependency | Simpler for students than optional deps |
| Native APIs only (not OpenAI-compat) | OpenAI endpoints strip timing metrics |
| One backend class per file | Clear ownership, easy to find code |
| Backend-specific quirks stay in backend class | Runner never branches on backend.name |
| llama.cpp = single model server | Don't try to automate model switching; document the limitation |
| Cross-backend comparison is user-directed | No automatic model name mapping across backends |

## Sources

- Direct codebase analysis of llm_benchmark/ package (HIGH confidence)
- PROJECT.md research findings on API endpoints and timing fields (MEDIUM-HIGH confidence)
- LM Studio official docs at lmstudio.ai/docs/api (MEDIUM confidence -- verified stats.tokens_per_second field)
- llama.cpp server API: timings object structure from PROJECT.md + training data (MEDIUM confidence -- field names need validation against running server)
- Python Protocol (PEP 544) documentation (HIGH confidence)

---

*Architecture research: 2026-03-14*
