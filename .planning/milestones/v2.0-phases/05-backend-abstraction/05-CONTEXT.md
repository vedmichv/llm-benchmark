# Phase 5: Backend Abstraction - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract a Backend protocol and OllamaBackend adapter so the entire codebase talks to a generic Backend instead of Ollama directly. Zero user-visible change: same output, same filenames, same behavior. All 152+ existing tests must pass (behavioral assertions unchanged; field names/units may update to match BackendResponse).

</domain>

<decisions>
## Implementation Decisions

### Protocol design
- Full protocol with 7 methods: `chat()`, `list_models()`, `unload_model()`, `warmup()`, `detect_context_window()`, `get_model_size()`, `check_connectivity()`
- `typing.Protocol` (not ABC) — already decided in PROJECT.md
- `chat(model, messages, stream=False, options=None)` — stream is a parameter, not separate methods
- `options` is a generic dict — backends interpret what they understand, ignore the rest (enables sweep mode with num_ctx/num_gpu)
- Each `chat()` returns a `BackendResponse` object — backend parses raw data internally, runner never sees raw responses

### Error handling
- Unified `BackendError` exception with `retryable: bool` flag
- Runner only catches `BackendError`, never native Ollama/httpx errors
- OllamaBackend wraps `ollama.RequestError`, `ollama.ResponseError`, `ConnectionError` into `BackendError`

### Backend instantiation
- Factory function `create_backend(name='ollama')` returns a Backend instance
- CLI creates backend once, passes to runner/preflight/menu
- No global singleton — explicit dependency passing for testability

### BackendResponse model
- Replaces `OllamaResponse` entirely — no coexistence, `OllamaResponse` removed from `models.py`
- All timing fields normalized to seconds (OllamaBackend converts ns internally)
- Includes `content: str` for generated text (needed for verbose display)
- Universal `prompt_cached: bool` flag (OllamaBackend detects from prompt_eval_count=-1)
- `BenchmarkResult.response` type changes from `OllamaResponse` to `BackendResponse`
- Specific field selection: Claude's discretion based on what all 3 backend APIs can provide

### Streaming
- `chat(stream=True)` returns a `StreamResult` wrapper object
- `StreamResult.chunks` — iterator of `str` content chunks for display
- `StreamResult.response` — `BackendResponse` with timing, available after iteration completes
- Runner consumes chunks for verbose display, then reads `.response` for metrics

### Abstraction scope
- All 3 benchmark modes abstracted: standard, concurrent, sweep
- Interactive menu uses Backend instance (Phase 5: always OllamaBackend)
- Preflight becomes backend-aware: Backend.check_connectivity() + Backend.list_models()
- `SystemInfo.ollama_version` renamed to `backend_name` + `backend_version`
- Export filenames stay unchanged in Phase 5 (zero-change constraint) — deferred to Phase 6

### Package layout
- New `llm_benchmark/backends/` subpackage
- `backends/__init__.py`: Backend protocol, BackendResponse, BackendError, StreamResult, create_backend()
- `backends/ollama.py`: OllamaBackend implementation
- Phase 6 adds `backends/llama_cpp.py`, `backends/lm_studio.py`

### Test strategy
- Mock at Backend level — tests mock `Backend.chat()` returning `BackendResponse`
- No `ollama` imports in test files after refactor
- All conftest.py fixtures updated to use `BackendResponse` (no OllamaResponse type leaks)
- BACK-05 interpreted as "no behavioral assertion changes" — field name/unit changes expected
- New `test_backends.py` for Backend protocol compliance, OllamaBackend, BackendResponse, BackendError
- CI pipeline updated if test paths change

### Claude's Discretion
- BackendResponse field selection (which fields universal vs optional)
- Internal OllamaBackend parsing implementation
- Exact StreamResult implementation details
- compute_averages() adaptation to BackendResponse seconds-based fields
- _ns_to_sec() removal/internalization into OllamaBackend

</decisions>

<specifics>
## Specific Ideas

- "Full protocol" preference — user wants every backend to implement all capabilities (warmup, detect_context_window, get_model_size) with sensible defaults, not just the minimum 3 methods
- "Zero visible change" is strict for output/filenames, but field name updates in test assertions are acceptable
- Backend metadata goes into JSON content in Phase 5 (not filenames) — filenames change in Phase 6

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_ns_to_sec()` in models.py: Will become internal to OllamaBackend (not exported)
- `run_with_timeout()` in runner.py: Stays in runner, wraps Backend.chat() calls
- `compute_averages()` in runner.py: Stays in runner, adapts to seconds-based BackendResponse
- Tenacity retry logic in runner.py: Stays, catches BackendError instead of native exceptions

### Established Patterns
- Rich Console singleton for output (`get_console()`) — backends should NOT print, only runner/CLI prints
- Pydantic models for validation — BackendResponse follows same pattern
- Lazy imports in cli.py — backends can follow same pattern

### Integration Points
- `cli.py:_handle_run()` — creates Backend, passes to preflight + runner
- `cli.py:main()` (no-args menu path) — creates Backend, passes to menu + runner
- `runner.py:run_single_benchmark()` — replaces ollama.chat() with backend.chat()
- `runner.py:warmup_model()` — replaces ollama.chat() with backend.warmup()
- `runner.py:unload_model()` — replaces ollama.generate() with backend.unload_model()
- `runner.py:detect_num_ctx()` — replaces ollama.show() with backend.detect_context_window()
- `runner.py:_get_model_size_gb()` — replaces ollama.list() with backend.get_model_size()
- `preflight.py:check_ollama_connectivity()` — becomes generic via backend.check_connectivity()
- `preflight.py:check_available_models()` — uses backend.list_models()
- `system.py:get_system_info()` — uses backend.name + backend.version for SystemInfo fields
- `concurrent.py` — uses runner functions (inherits Backend through runner)
- `sweep.py` — uses runner functions (inherits Backend through runner)

</code_context>

<deferred>
## Deferred Ideas

- Backend identifier in export filenames — Phase 6 (alongside --backend flag, CLI-03)
- Backend-specific sweep parameters (each backend declares what's sweepable) — future consideration
- Backend auto-detection (shutil.which) — Phase 6 (BEND-03)
- Backend auto-start — Phase 6 (BEND-04)

</deferred>

---

*Phase: 05-backend-abstraction*
*Context gathered: 2026-03-14*
