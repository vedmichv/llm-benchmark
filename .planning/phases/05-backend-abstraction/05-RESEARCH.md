# Phase 5: Backend Abstraction - Research

**Researched:** 2026-03-14
**Domain:** Python protocol-based abstraction layer, refactoring Ollama-coupled codebase
**Confidence:** HIGH

## Summary

Phase 5 is a pure refactoring phase: extract all direct `ollama` library calls behind a `typing.Protocol`-based `Backend` interface, implement `OllamaBackend` as the sole adapter, and ensure zero user-visible change. The codebase currently has **12 direct `ollama.*` call sites** across 5 modules (`runner.py`, `preflight.py`, `concurrent.py`, `sweep.py`, `recommend.py`) plus `_ns_to_sec()` in `models.py` used by 4 modules for nanosecond conversion. All 152 existing tests must continue to pass.

The refactor surface is well-defined. The `ollama` Python SDK uses synchronous `ollama.chat()`, `ollama.list()`, `ollama.show()`, `ollama.generate()` calls and an async `ollama.AsyncClient` for concurrent mode. These map cleanly to the 7-method Backend protocol decided by the user. The key technical challenge is the concurrent mode, which uses `ollama.AsyncClient` directly -- this needs an async-compatible path in the Backend protocol or the OllamaBackend.

**Primary recommendation:** Create `llm_benchmark/backends/` subpackage with Protocol + OllamaBackend, replace all `ollama.*` calls module-by-module starting with models (BackendResponse), then runner, then preflight/sweep/concurrent, updating tests in parallel.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full protocol with 7 methods: `chat()`, `list_models()`, `unload_model()`, `warmup()`, `detect_context_window()`, `get_model_size()`, `check_connectivity()`
- `typing.Protocol` (not ABC)
- `chat(model, messages, stream=False, options=None)` -- stream is a parameter, not separate methods
- `options` is a generic dict -- backends interpret what they understand
- Each `chat()` returns a `BackendResponse` object
- Unified `BackendError` exception with `retryable: bool` flag
- Runner only catches `BackendError`, never native Ollama/httpx errors
- OllamaBackend wraps `ollama.RequestError`, `ollama.ResponseError`, `ConnectionError` into `BackendError`
- Factory function `create_backend(name='ollama')` returns a Backend instance
- CLI creates backend once, passes to runner/preflight/menu
- No global singleton -- explicit dependency passing for testability
- BackendResponse replaces OllamaResponse entirely -- no coexistence, OllamaResponse removed from models.py
- All timing fields normalized to seconds (OllamaBackend converts ns internally)
- Includes `content: str` for generated text
- Universal `prompt_cached: bool` flag
- `BenchmarkResult.response` type changes from `OllamaResponse` to `BackendResponse`
- `chat(stream=True)` returns a `StreamResult` wrapper object with `.chunks` iterator and `.response` BackendResponse
- All 3 benchmark modes abstracted: standard, concurrent, sweep
- Preflight becomes backend-aware
- `SystemInfo.ollama_version` renamed to `backend_name` + `backend_version`
- New `llm_benchmark/backends/` subpackage
- Mock at Backend level in tests
- No `ollama` imports in test files after refactor
- New `test_backends.py` for Backend protocol compliance

### Claude's Discretion
- BackendResponse field selection (which fields universal vs optional)
- Internal OllamaBackend parsing implementation
- Exact StreamResult implementation details
- compute_averages() adaptation to BackendResponse seconds-based fields
- _ns_to_sec() removal/internalization into OllamaBackend

### Deferred Ideas (OUT OF SCOPE)
- Backend identifier in export filenames -- Phase 6
- Backend-specific sweep parameters -- future
- Backend auto-detection (shutil.which) -- Phase 6
- Backend auto-start -- Phase 6
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BACK-01 | Backend Protocol defines chat(), list_models(), unload_model() with normalized BackendResponse | Protocol design pattern, field mapping analysis, StreamResult design |
| BACK-02 | BackendResponse normalizes all timing data to seconds regardless of backend | Field mapping from OllamaResponse (ns) to seconds, _ns_to_sec internalization |
| BACK-03 | OllamaBackend wraps existing ollama.chat() code with zero behavior change | 12 call site inventory, error wrapping into BackendError, async client handling |
| BACK-04 | Runner accepts Backend instance instead of calling ollama directly | Dependency injection pattern, function signature changes, compute_averages adaptation |
| BACK-05 | All existing tests pass after refactor with no Ollama-specific type leaks | Test fixture migration plan, mock target changes, 152 test inventory |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| typing | stdlib | Protocol definition | Already decided: typing.Protocol for structural subtyping |
| pydantic | 2.x (existing) | BackendResponse model | Already used for OllamaResponse, consistent pattern |
| ollama | existing | OllamaBackend implementation | Already a dependency, SDK stays inside backend only |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tenacity | existing | Retry logic in runner | Stays in runner, catches BackendError instead of native exceptions |

### No New Dependencies
This phase adds zero new dependencies. All work uses existing stdlib `typing.Protocol` and project dependencies.

## Architecture Patterns

### Recommended Package Structure
```
llm_benchmark/
  backends/
    __init__.py          # Backend Protocol, BackendResponse, BackendError, StreamResult, create_backend()
    ollama.py            # OllamaBackend implementation
  models.py              # Remove OllamaResponse and _ns_to_sec; keep BenchmarkResult, ModelSummary, etc.
  runner.py              # Accept Backend param; no direct ollama imports
  preflight.py           # Accept Backend param; no direct ollama imports
  concurrent.py          # Accept Backend param; no direct ollama imports
  sweep.py               # Accept Backend param; no direct ollama imports
  recommend.py           # Accept Backend param for re-listing models
  system.py              # Use backend.name + backend.version instead of get_ollama_version()
  cli.py                 # Create backend once via create_backend(), pass everywhere
  menu.py                # Receive backend as parameter
```

### Pattern 1: typing.Protocol for Backend Interface
**What:** Structural subtyping -- any class with the right methods satisfies the protocol without inheritance.
**When to use:** When you want to decouple interface from implementation without ABC overhead.
**Example:**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Backend(Protocol):
    """Protocol that all benchmark backends must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict | None = None,
    ) -> "BackendResponse | StreamResult": ...

    def list_models(self) -> list[dict]: ...

    def unload_model(self, model: str) -> bool: ...

    def warmup(self, model: str, timeout: int) -> bool: ...

    def detect_context_window(self, model: str) -> int: ...

    def get_model_size(self, model: str) -> float | None: ...

    def check_connectivity(self) -> bool: ...
```

### Pattern 2: BackendResponse as Pydantic Model with Seconds
**What:** Normalized response model where all timing is in seconds (float), not backend-specific units.
**When to use:** Every `chat()` return wraps raw backend data into this.
**Example:**
```python
class BackendResponse(BaseModel):
    """Normalized response from any backend. All times in seconds."""
    model: str
    content: str
    done: bool

    # Token counts
    prompt_eval_count: int = 0
    eval_count: int = 0

    # Timing in seconds (float)
    total_duration: float = 0.0
    load_duration: float = 0.0
    prompt_eval_duration: float = 0.0
    eval_duration: float = 0.0

    # Flags
    prompt_cached: bool = False
```

### Pattern 3: BackendError with Retryable Flag
**What:** Single exception type wrapping all backend-specific errors.
**When to use:** Runner catches only BackendError, checks `.retryable` for retry decisions.
**Example:**
```python
class BackendError(Exception):
    """Unified error from any backend."""
    def __init__(self, message: str, retryable: bool = False, original: Exception | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.original = original
```

### Pattern 4: StreamResult Wrapper
**What:** Object returned by `chat(stream=True)` with iterable chunks and deferred response.
**When to use:** Verbose mode streaming display.
**Example:**
```python
class StreamResult:
    """Wrapper for streaming chat responses."""
    def __init__(self, chunks: Iterator[str], finalize: Callable[[], BackendResponse]):
        self._chunks = chunks
        self._finalize = finalize
        self._response: BackendResponse | None = None

    @property
    def chunks(self) -> Iterator[str]:
        return self._chunks

    @property
    def response(self) -> BackendResponse:
        if self._response is None:
            self._response = self._finalize()
        return self._response
```

### Pattern 5: Factory Function
**What:** Simple factory to create backend instances.
**Example:**
```python
def create_backend(name: str = "ollama") -> Backend:
    if name == "ollama":
        from llm_benchmark.backends.ollama import OllamaBackend
        return OllamaBackend()
    raise ValueError(f"Unknown backend: {name}")
```

### Anti-Patterns to Avoid
- **Leaking backend types:** Runner/tests must never import `ollama` directly or reference `OllamaResponse`
- **Partial abstraction:** Don't leave some calls going through Backend and others calling ollama directly
- **Global backend singleton:** User explicitly decided against this -- pass as parameter
- **Keeping OllamaResponse alongside BackendResponse:** User decided OllamaResponse is removed entirely

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Response validation | Manual dict parsing | Pydantic BackendResponse | Consistent with existing pattern, type safety |
| Retry logic | Custom retry loop | Existing tenacity in runner | Already works, just change exception filter |
| Nanosecond conversion | Keep _ns_to_sec in models | Internalize in OllamaBackend | BackendResponse is always seconds |

## Common Pitfalls

### Pitfall 1: Concurrent Mode AsyncClient
**What goes wrong:** The concurrent module uses `ollama.AsyncClient` directly, which doesn't fit the synchronous Backend protocol.
**Why it happens:** Backend.chat() is sync, but concurrent needs async.
**How to avoid:** OllamaBackend should expose an async chat method or the concurrent module should use `run_in_executor` to call sync Backend.chat() in threads. Alternatively, add an `async_chat()` method to the protocol. Given the existing pattern uses `asyncio.gather`, the cleanest approach is for OllamaBackend to have a private `_async_client` property and concurrent module to be aware it's using OllamaBackend's async capabilities. Consider: concurrent.py can call `backend.chat()` in a ThreadPoolExecutor instead of using AsyncClient, which would make it truly backend-agnostic.
**Warning signs:** Tests passing but concurrent mode silently broken.

### Pitfall 2: Export Functions Using _ns_to_sec
**What goes wrong:** `exporters.py` calls `_ns_to_sec(r.response.eval_duration)` etc. -- after refactor, BackendResponse stores seconds, so these would double-convert.
**Why it happens:** exporters.py directly imports and uses `_ns_to_sec` from `models.py` on response fields.
**How to avoid:** When BackendResponse stores seconds, exporters must use `r.response.eval_duration` directly (already in seconds). Remove all `_ns_to_sec` calls from exporters.
**Warning signs:** Duration values being 1e-9 of expected (nanoseconds of nanoseconds).

### Pitfall 3: compute_averages Using Nanosecond Fields
**What goes wrong:** `compute_averages()` in runner.py calls `_ns_to_sec(r.response.eval_duration)` -- with BackendResponse already in seconds, this would produce wrong values.
**Why it happens:** Existing code assumes nanosecond integers from OllamaResponse.
**How to avoid:** Update compute_averages to use response fields directly (they're already seconds in BackendResponse).
**Warning signs:** Throughput values being billions of tokens/second.

### Pitfall 4: Test Mock Targets Change
**What goes wrong:** Tests mock `llm_benchmark.runner.ollama` etc. After refactor, these mock targets no longer exist.
**Why it happens:** All ollama usage moves into `backends/ollama.py`.
**How to avoid:** Tests should mock at Backend level: `backend.chat()` returns `BackendResponse`. For OllamaBackend-specific tests, mock `ollama` inside `backends.ollama`.
**Warning signs:** Tests passing vacuously because mocks don't match real code paths.

### Pitfall 5: Streaming Response Timing Data
**What goes wrong:** In streaming mode, Ollama only returns full timing in the final chunk. The current code captures `last_chunk` and parses it.
**Why it happens:** Streaming iterates chunks for display, then needs full metrics.
**How to avoid:** StreamResult.response must capture the final chunk's timing data. OllamaBackend's streaming implementation should consume the ollama stream iterator, yield content strings, and build the BackendResponse from the final chunk.
**Warning signs:** StreamResult.response returning zeros for all timing fields.

### Pitfall 6: recommend.py Has Direct ollama Import
**What goes wrong:** `recommend.py` line 234 does `import ollama as ollama_client; ollama_client.list()` to re-fetch models after pulling.
**Why it happens:** Easy to miss -- it's a lazy import deep in the function.
**How to avoid:** recommend.py should receive Backend instance and call `backend.list_models()`.
**Warning signs:** ollama imports leaking through in production code.

### Pitfall 7: Sweep Module Uses ollama.Options
**What goes wrong:** `sweep.py` imports `from ollama import Options` and passes `Options(num_ctx=..., num_gpu=...)` to `ollama.chat()`.
**Why it happens:** ollama SDK has its own Options type.
**How to avoid:** Backend.chat() accepts a plain `dict` for options. OllamaBackend internally converts dict to ollama.Options if needed (or just passes dict -- ollama SDK accepts dicts too).
**Warning signs:** TypeErrors when sweep calls backend.chat() with wrong options type.

### Pitfall 8: SystemInfo.ollama_version Field Rename
**What goes wrong:** Renaming `ollama_version` to `backend_name`/`backend_version` in SystemInfo breaks exporters and display code that references `system_info.ollama_version`.
**Why it happens:** Field used in CSV headers, Markdown reports, info command display.
**How to avoid:** Rename field AND update all references: exporters.py CSV rows, system.py format_system_summary, cli.py _handle_info, test_system.py.
**Warning signs:** AttributeError on `system_info.ollama_version` at runtime.

## Code Examples

### Ollama Call Site Inventory (12 sites to refactor)

| Module | Line | Current Call | Backend Method |
|--------|------|-------------|----------------|
| runner.py | 102 | `ollama.chat(model=..., messages=...)` | `backend.warmup(model, timeout)` |
| runner.py | 137 | `ollama.list()` | `backend.list_models()` + size extraction |
| runner.py | 158 | `ollama.show(model_name)` | `backend.detect_context_window(model)` |
| runner.py | 220 | `ollama.generate(model=..., keep_alive=0)` | `backend.unload_model(model)` |
| runner.py | 312 | `ollama.chat(stream=True, ...)` | `backend.chat(model, msgs, stream=True)` |
| runner.py | 332 | `ollama.chat(...)` | `backend.chat(model, msgs, options=...)` |
| preflight.py | 130 | `ollama.list()` | `backend.check_connectivity()` |
| preflight.py | 160 | `ollama.list()` | `backend.list_models()` |
| concurrent.py | 68 | `client.chat(...)` (AsyncClient) | Thread-based `backend.chat()` or async variant |
| concurrent.py | 124 | `ollama.AsyncClient()` | Thread pool or backend async interface |
| sweep.py | 42 | `ollama.show(model_name)` | `backend.detect_context_window()` or similar |
| sweep.py | 93 | `ollama.chat(options=Options(...))` | `backend.chat(model, msgs, options={...})` |
| recommend.py | 237 | `ollama_client.list()` | `backend.list_models()` |

### BackendResponse Field Mapping

| BackendResponse field | OllamaResponse source | Conversion |
|-----------------------|----------------------|------------|
| model | model | direct |
| content | message.content | extract from message |
| done | done | direct |
| prompt_eval_count | prompt_eval_count | direct (after cache normalization) |
| eval_count | eval_count | direct |
| total_duration | total_duration | / 1_000_000_000 |
| load_duration | load_duration | / 1_000_000_000 |
| prompt_eval_duration | prompt_eval_duration | / 1_000_000_000 |
| eval_duration | eval_duration | / 1_000_000_000 |
| prompt_cached | prompt_cached (derived) | detect from prompt_eval_count == -1 |

### compute_averages Adaptation

Current code (nanoseconds):
```python
total_response_time = sum(_ns_to_sec(r.response.eval_duration) for r in successful)
```

After refactor (seconds):
```python
total_response_time = sum(r.response.eval_duration for r in successful)
```

All `_ns_to_sec()` calls removed from runner.py, exporters.py, concurrent.py, sweep.py. The function is deleted from models.py and internalized into OllamaBackend.

### Test Fixture Migration

Current conftest.py fixture:
```python
@pytest.fixture
def sample_ollama_response_dict():
    return {
        "total_duration": 5_000_000_000,  # nanoseconds
        "eval_duration": 4_000_000_000,
        ...
    }
```

After refactor:
```python
@pytest.fixture
def sample_backend_response():
    return BackendResponse(
        model="llama3.2:1b",
        content="The sky is blue because...",
        done=True,
        total_duration=5.0,  # seconds
        eval_duration=4.0,   # seconds
        prompt_eval_duration=0.2,
        load_duration=0.5,
        prompt_eval_count=15,
        eval_count=120,
    )
```

### Concurrent Mode Refactor Strategy

Current approach uses `ollama.AsyncClient`:
```python
async with ollama.AsyncClient() as client:
    tasks = [_single_request(client, ...) for i in range(n)]
    results = await asyncio.gather(*tasks)
```

Recommended refactor -- use ThreadPoolExecutor for backend-agnostic concurrency:
```python
import concurrent.futures

def _run_batch(backend: Backend, model: str, prompt: str, n: int) -> ...:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(backend.chat, model, messages, options=options)
            for _ in range(n)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

This eliminates the AsyncClient dependency and makes concurrent mode work with any backend. The Ollama server handles concurrent requests regardless of client-side async vs threads.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ABC base class | typing.Protocol | Python 3.8+ | No inheritance needed, structural subtyping |
| OllamaResponse with ns | BackendResponse with seconds | This phase | All downstream code uses seconds directly |
| ollama.AsyncClient for concurrency | ThreadPoolExecutor | This phase | Backend-agnostic concurrent benchmarking |
| _ns_to_sec scattered in 4 modules | Conversion internal to OllamaBackend | This phase | Single conversion point |

## Open Questions

1. **Concurrent mode: threads vs async?**
   - What we know: ThreadPoolExecutor is backend-agnostic and simpler. AsyncClient is Ollama-specific.
   - What's unclear: Performance difference for high concurrency (8+ workers).
   - Recommendation: Use ThreadPoolExecutor. For Ollama specifically, the server-side queuing dominates latency, not client-side async vs threads. This is Claude's discretion per CONTEXT.md.

2. **BackendResponse: created_at field?**
   - What we know: OllamaResponse has `created_at: datetime`. Other backends may not provide this.
   - What's unclear: Whether any downstream code uses created_at.
   - Recommendation: Omit from BackendResponse. Grep shows no usage of `created_at` outside OllamaResponse model definition itself.

3. **recommend.py model pulling**
   - What we know: recommend.py calls `subprocess.run(["ollama", "pull", model])` then `ollama.list()`.
   - What's unclear: Should model pulling be part of Backend protocol?
   - Recommendation: No. Pulling is an install/setup action, not a benchmark operation. Keep `subprocess.run(["ollama", "pull", ...])` in recommend.py but use `backend.list_models()` for the re-fetch. In Phase 6, recommend.py may need backend-specific pull commands.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via uv) |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-01 | Backend Protocol defines 7 methods, BackendResponse model | unit | `uv run pytest tests/test_backends.py -x` | No -- Wave 0 |
| BACK-02 | BackendResponse timing in seconds, OllamaBackend converts ns | unit | `uv run pytest tests/test_backends.py::test_ollama_backend_converts_ns -x` | No -- Wave 0 |
| BACK-03 | OllamaBackend wraps ollama calls, zero behavior change | unit + integration | `uv run pytest tests/test_backends.py -x` | No -- Wave 0 |
| BACK-04 | Runner uses Backend instance, no direct ollama calls | unit | `uv run pytest tests/test_runner.py -x` | Yes -- needs update |
| BACK-05 | All 152 existing tests pass | integration | `uv run pytest tests/ -v` | Yes -- needs fixture updates |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_backends.py` -- covers BACK-01, BACK-02, BACK-03 (Backend protocol compliance, OllamaBackend, BackendResponse, BackendError, StreamResult)
- [ ] `tests/conftest.py` -- update fixtures from OllamaResponse to BackendResponse (covers BACK-05)
- [ ] No framework install needed -- pytest already configured

## Sources

### Primary (HIGH confidence)
- Direct source code analysis of all 5 affected modules + 15 test files
- Python typing.Protocol documentation (stdlib, stable since 3.8)
- Pydantic v2 BaseModel documentation (already used in project)

### Secondary (MEDIUM confidence)
- ollama Python SDK behavior (verified from existing working code patterns in codebase)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all patterns already used in codebase
- Architecture: HIGH -- user decisions are extremely detailed, all integration points mapped from source
- Pitfalls: HIGH -- identified from direct code analysis of all 12 call sites and 152 tests
- Concurrent mode: MEDIUM -- ThreadPoolExecutor recommendation is sound but unverified against AsyncClient perf

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable -- refactoring existing codebase, no external dependency changes)
