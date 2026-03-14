# Domain Pitfalls: Multi-Backend Integration

**Domain:** Adding llama.cpp and LM Studio backends to an Ollama-based LLM benchmark tool
**Researched:** 2026-03-14
**Focus:** Integration pitfalls when extending existing Ollama-only architecture

---

## Critical Pitfalls

Mistakes that cause incorrect benchmark comparisons, data corruption, or rewrites of the abstraction layer.

### Pitfall 1: Timing Metric Units Mismatch Across Backends

**What goes wrong:** Each backend reports timing in different units, and mixing them without normalization produces nonsensical cross-backend comparisons. Ollama reports durations in **nanoseconds** (integers), llama.cpp reports in **milliseconds** (floats) or pre-computed **tokens/second**, and LM Studio reports **tokens_per_second** directly in its `stats` object. If you feed llama.cpp millisecond values into the existing `_ns_to_sec()` function (which divides by 1,000,000,000), you get durations 1,000,000x too small, resulting in impossibly high throughput numbers.

**Why it happens:** The existing codebase has `_ns_to_sec()` in `models.py` and uses it everywhere: `runner.py` (compute_averages), `exporters.py` (all export functions). Developers wire up a new backend, pass its timing values through the same pipeline, and never notice the unit mismatch because the numbers look "plausible" at first glance (both are large integers).

**Consequences:** Cross-backend comparison reports show llama.cpp as 1,000,000x faster than Ollama. Students conclude their hardware is broken. Or worse, the numbers are off by a smaller factor (e.g., llama.cpp reports some timings in microseconds) and the error goes unnoticed, producing subtly wrong but believable comparisons.

**Specific fields to normalize:**

| Metric | Ollama | llama.cpp `/completion` | LM Studio `/api/v1/chat` |
|--------|--------|------------------------|--------------------------|
| Prompt eval time | `prompt_eval_duration` (ns) | `timings.prompt_ms` (ms) | Not directly exposed; compute from token counts |
| Generation time | `eval_duration` (ns) | `timings.predicted_ms` (ms) | Not directly exposed; use `stats.tokens_per_second` |
| Prompt tokens | `prompt_eval_count` | `timings.prompt_n` | `usage.prompt_tokens` |
| Generated tokens | `eval_count` | `timings.predicted_n` | `usage.completion_tokens` |
| Pre-computed t/s | Must compute manually | `timings.predicted_per_second` | `stats.tokens_per_second` |
| Total duration | `total_duration` (ns) | Compute from component timings | Not directly exposed |
| Load duration | `load_duration` (ns) | Not in response (separate endpoint) | Model load events in stream |

**Prevention:**
- Create a `NormalizedTimings` dataclass that all backends convert INTO, with fields in a single canonical unit (seconds as float). Each backend adapter is responsible for its own conversion.
- NEVER pass raw backend values through the existing `_ns_to_sec()`. That function should only exist in the Ollama adapter.
- Add a unit test that verifies: for the same model/prompt, cross-backend normalized timings are within the same order of magnitude (not a correctness test, a sanity test).
- If a backend provides pre-computed `tokens_per_second` but not raw durations, compute the duration from `token_count / tokens_per_second` for consistency. Do NOT mix pre-computed rates from one backend with computed rates from another.

**Detection:** Cross-backend comparison where one backend shows > 10x the throughput of another for the same model. This should trigger a warning.

**Phase:** Must be the FIRST thing designed in the backend abstraction layer. The normalized response model is the foundation everything else builds on.

---

### Pitfall 2: Model Identity Crisis -- Same Model, Different Names

**What goes wrong:** Cross-backend comparison requires matching "the same model" across backends, but each backend names models differently. Ollama uses registry-style names (`llama3.2:3b-instruct-q4_K_M`), llama.cpp uses bare GGUF filenames (`llama-3.2-3b-instruct-Q4_K_M.gguf`), and LM Studio uses a path-style identifier (`lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`). There is no universal model ID.

**Why it happens:** Each backend has its own model management. Ollama pulls from its own registry with its own naming scheme. llama.cpp loads raw GGUF files. LM Studio downloads from Hugging Face with publisher/repo/file hierarchy. A developer building cross-backend comparison will try to string-match these names and it will fail for all but the simplest cases.

**Specific naming examples for the SAME model:**

| Backend | Model identifier |
|---------|-----------------|
| Ollama | `qwen2.5:7b` or `qwen2.5:7b-instruct-q4_K_M` |
| llama.cpp | `qwen2.5-7b-instruct-q4_k_m.gguf` |
| LM Studio | `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf` |

**Consequences:**
- Cross-backend comparison fails to match models, showing separate entries instead of side-by-side.
- Worse: fuzzy matching incorrectly pairs different quantizations (Q4_K_M vs Q8_0) or different model sizes (7B vs 3B), making the comparison meaningless.
- Students manually specifying model names must remember three different naming conventions.

**Prevention:**
- Do NOT attempt automatic fuzzy matching for cross-backend comparison. It will be wrong more often than right.
- Use a **model mapping file** or **CLI flag** approach: `--model "qwen2.5:7b" --llama-cpp-model "path/to/qwen2.5-7b.gguf" --lm-studio-model "Qwen/..."`. Explicit is better than magic.
- For display purposes, extract a **canonical short name** from each backend's identifier (e.g., strip paths, strip `.gguf`, normalize separators) but ONLY for display, never for matching.
- Store the original backend-specific model identifier in results so comparisons are auditable.
- For the interactive menu: let students select a model from each backend's available list, then manually confirm the pairing.

**Detection:** Cross-backend comparison with zero matched models = mapping is broken. Warning: "No models matched across backends. Use --model-map to specify which models to compare."

**Phase:** Backend abstraction layer phase. Model identity is part of the protocol interface.

---

### Pitfall 3: Using OpenAI-Compatible Endpoints Instead of Native APIs

**What goes wrong:** Both llama.cpp and LM Studio expose OpenAI-compatible endpoints (`/v1/chat/completions`). It is tempting to use these because the request/response format is familiar and a single HTTP client could work for both. But these endpoints strip backend-specific timing metrics. The OpenAI response format has `usage.prompt_tokens` and `usage.completion_tokens` but NO duration/timing fields. You get token counts but cannot compute tokens/second.

**Why it happens:** Developer sees OpenAI-compatible endpoint, thinks "great, one client for everything," implements it, and only discovers the missing timing data when cross-backend comparison shows 0 t/s for everything except Ollama. This is explicitly documented in the project's own research (PROJECT.md line 73: "Each backend uses different native endpoint (NOT OpenAI-compat -- that strips timing data)").

**Consequences:** The core purpose of the tool (measuring tokens/second) is completely broken for non-Ollama backends. Results show token counts but no throughput. The entire backend integration is useless.

**Prevention:**
- Use ONLY native endpoints for each backend:
  - Ollama: `/api/chat` (current, correct)
  - llama.cpp: `/completion` (NOT `/v1/chat/completions`)
  - LM Studio: `/api/v1/chat` (NOT `/v1/chat/completions`)
- Each backend adapter builds its own HTTP request and parses its own response format. No shared request/response shape.
- Add an integration test per backend: assert that the response contains timing data, not just token counts.
- Document in the backend protocol: "Implementations MUST return timing durations, not just token counts."

**Detection:** Any benchmark result where `eval_duration` (or equivalent) is 0 or None for a non-Ollama backend = wrong endpoint.

**Phase:** First thing validated when implementing each backend adapter. Test with real server before building anything else.

---

### Pitfall 4: Backend Server Not Running -- Cascading Failures

**What goes wrong:** The existing preflight check (`preflight.py`) calls `ollama.list()` and exits if it fails. With multiple backends, a student might have Ollama running but not llama.cpp server, or vice versa. If the tool tries to benchmark on a dead backend, it either hangs (connection timeout), crashes, or produces confusing errors.

**Why it happens:** The current architecture assumes ONE backend (Ollama) and treats its absence as fatal (`sys.exit(1)`). With multiple backends, some might be available and others not. The existing "all-or-nothing" preflight pattern does not work.

**Specific backend connection details:**

| Backend | Default address | Health check | Failure mode |
|---------|----------------|--------------|--------------|
| Ollama | `http://localhost:11434` | `GET /api/tags` or `ollama.list()` | ConnectionError |
| llama.cpp | `http://localhost:8080` | `GET /health` | ConnectionError, or 503 if model loading |
| LM Studio | `http://localhost:1234` | `GET /api/v1/models` | ConnectionError |

**Consequences:**
- Tool hangs for 30+ seconds on connection timeout to dead backend.
- If `--backend all` is specified, one dead backend kills the entire run.
- Students do not know which backend failed or how to start it.
- llama.cpp has a unique state: `/health` returns 503 while model is loading (not dead, just not ready). Treating this as "not running" is wrong.

**Prevention:**
- Preflight checks must be **per-backend** and **non-fatal for optional backends**:
  - `--backend ollama`: Ollama must be running (fatal if not).
  - `--backend llama-cpp`: llama.cpp server must be running (fatal if not).
  - `--backend all`: Check each, report which are available, benchmark only available ones.
- Set connection timeout to 3 seconds (not the default 200s benchmark timeout).
- For llama.cpp: retry `/health` for up to 10 seconds if it returns 503 (model loading).
- Print backend-specific start instructions on failure:
  - Ollama: `ollama serve`
  - llama.cpp: `llama-server -m /path/to/model.gguf --port 8080`
  - LM Studio: "Open LM Studio and start the local server"
- Auto-detect running backends: probe all three default ports at startup, report what is available.

**Detection:** Backend health check returning non-200 or timing out. Should be detected in < 5 seconds, not 200 seconds.

**Phase:** Preflight checks phase. Must be redesigned before any backend runs benchmarks.

---

### Pitfall 5: llama.cpp Loads One Model Per Server Instance

**What goes wrong:** Ollama is a model manager -- it loads/unloads models on demand. You start Ollama once, then benchmark any model. llama.cpp server is fundamentally different: each server instance loads exactly ONE model at startup (`llama-server -m model.gguf`). To benchmark a different model, you must stop the server and restart with the new model file.

**Why it happens:** The existing codebase assumes a model management layer exists. `check_available_models()` lists models from `ollama.list()`. `unload_model()` calls `keep_alive=0`. `benchmark_model()` iterates over multiple models in a loop. None of this translates to llama.cpp.

**Consequences:**
- The tool tries to benchmark model B on a llama.cpp server that only has model A loaded. The request either fails or (worse) silently benchmarks model A and labels it as model B.
- Students expect to benchmark multiple GGUF files in one run and cannot.
- Attempting to restart llama.cpp server programmatically requires subprocess management, is fragile, and may require elevated privileges.

**Prevention:**
- The backend protocol MUST declare a `model_management` capability:
  - Ollama: `MULTI_MODEL` -- can load/unload models on demand.
  - llama.cpp: `SINGLE_MODEL` -- one model per server, must query which model is loaded.
  - LM Studio: `MULTI_MODEL` -- can load/unload via `/api/v1/models/load` and `/api/v1/models/unload`.
- For `SINGLE_MODEL` backends: query the loaded model at preflight, benchmark ONLY that model, skip the model iteration loop.
- Do NOT attempt to restart llama.cpp server programmatically. Instead, tell the student: "llama.cpp has model X loaded. To benchmark model Y, restart the server with `llama-server -m Y.gguf`."
- For cross-backend comparison: verify that all backends have the equivalent model loaded before starting.

**Detection:** llama.cpp `/health` or `/props` endpoint reports the loaded model. Compare against requested model name.

**Phase:** Backend abstraction layer. The protocol must account for this from the start or the entire multi-model loop breaks.

---

## Moderate Pitfalls

### Pitfall 6: Warmup Behavior Differs Across Backends

**What goes wrong:** The existing warmup (`warmup_model()` in `runner.py`) sends a short prompt via `ollama.chat()` to pre-load the model into GPU memory. For llama.cpp, the model is loaded at server startup, so warmup means something different (filling KV cache, JIT compilation on some platforms). For LM Studio, model loading can happen on first request or via explicit `/api/v1/models/load`.

**Prevention:**
- Each backend adapter should implement its own `warmup()` method:
  - Ollama: send short prompt (current behavior, correct).
  - llama.cpp: send short prompt to warm compute kernels. Model is already loaded.
  - LM Studio: call `/api/v1/models/load` explicitly if model not loaded, then send short prompt.
- The warmup method should be part of the backend protocol, not a shared function that calls backend-specific APIs.

**Phase:** Backend protocol design. Include `warmup()` in the protocol interface.

---

### Pitfall 7: Context Window Defaults Differ Silently

**What goes wrong:** The existing `detect_num_ctx()` queries Ollama's `ollama.show()` API to determine context size. llama.cpp uses whatever `--ctx-size` (default: 4096 historically, but changed to 0 = model's trained max in recent versions) was passed at server startup. LM Studio has per-request `context_length`. If backends use different effective context sizes, throughput comparisons are invalid -- larger context = more VRAM for KV cache = potentially slower generation.

**Prevention:**
- Each backend adapter must report its effective context size.
  - Ollama: `detect_num_ctx()` (existing, works).
  - llama.cpp: query `/props` endpoint which returns `default_generation_settings.n_ctx`.
  - LM Studio: query model info or set explicitly per request.
- For cross-backend comparison: either enforce the same `num_ctx` across all backends, or clearly label the context size in results so students know the comparison is apples-to-oranges.
- WARNING: Recent llama.cpp (post-2025) defaults to `--ctx-size 0` meaning "use model's full trained context." This can be 128K+ for models like Llama 3.1, consuming enormous VRAM. Students will OOM without understanding why.

**Detection:** If cross-backend context sizes differ by more than 2x, print a warning.

**Phase:** Backend protocol. Each adapter must expose effective context size.

---

### Pitfall 8: Prompt Format Differences Cause Unfair Comparisons

**What goes wrong:** Ollama's `/api/chat` applies the model's chat template automatically (system prompt, BOS/EOS tokens, role markers). llama.cpp's `/completion` endpoint is a raw completion endpoint -- it does NOT apply chat templates unless you use `/v1/chat/completions` (which strips timing data -- Pitfall 3). LM Studio's native `/api/v1/chat` does apply templates.

**Consequences:** The same user prompt sent to Ollama and llama.cpp may result in different actual token sequences. Ollama wraps it in `<|start|>system\nYou are a helpful assistant<|end|>\n<|start|>user\n{prompt}<|end|>\n<|start|>assistant\n` while llama.cpp receives it raw. This affects both prompt token count and generation behavior, making throughput numbers non-comparable.

**Prevention:**
- For llama.cpp `/completion`: manually apply the model's chat template before sending. llama.cpp server has a `/apply-template` endpoint (or the template can be set at server startup with `--chat-template`).
- Alternatively, use llama.cpp's `/v1/chat/completions` ONLY for the actual request (to get template application), but extract timing from a parallel call to `/completion` -- this is complex and fragile. Better: apply template client-side.
- Document this clearly: "For fair comparison, ensure all backends use the same prompt template."
- Store the actual tokenized prompt length in results so students can verify parity.

**Phase:** Backend adapter implementation. Each adapter's `run_benchmark()` must handle template application.

---

### Pitfall 9: Streaming vs Non-Streaming Response Format Differences

**What goes wrong:** The existing code has two paths: streaming (verbose mode) and non-streaming. For Ollama, the final streaming chunk contains timing data. For llama.cpp, streaming chunks have incremental timing in each chunk, and the final chunk has aggregate timings. For LM Studio, streaming returns Server-Sent Events with timing in the final `[DONE]`-adjacent message.

If the streaming implementation assumes Ollama's format (timing only in last chunk), it will miss timing data from llama.cpp's per-chunk updates or LM Studio's event format.

**Prevention:**
- For benchmarking, use non-streaming mode by default for all backends. Streaming adds latency measurement complexity (TTFT vs total throughput) and format differences.
- If verbose/streaming mode is needed, each backend adapter must implement its own stream parser.
- NEVER share a streaming parser across backends.

**Phase:** Backend adapter implementation. Start with non-streaming, add streaming per-backend later.

---

### Pitfall 10: httpx vs ollama SDK -- Two HTTP Client Patterns

**What goes wrong:** The existing codebase uses the `ollama` Python SDK which handles connection management, retries, and response parsing internally. New backends will use `httpx` for direct HTTP calls. Mixing two HTTP client libraries creates inconsistent timeout behavior, error types, and connection pool management.

**Specific differences:**

| Behavior | `ollama` SDK | `httpx` |
|----------|-------------|---------|
| Timeout | Thread-based wrapper (current `run_with_timeout`) | Native `timeout` parameter |
| Errors | `ollama.RequestError`, `ollama.ResponseError` | `httpx.ConnectError`, `httpx.TimeoutException` |
| Connection | Managed internally | Must create/close `httpx.Client` |
| Retry | Tenacity wrapper (current) | Must implement or wrap with tenacity |

**Prevention:**
- Each backend adapter handles its own HTTP client internally. The backend protocol returns normalized results, hiding the transport layer.
- Timeout management should be per-backend: Ollama uses the existing thread-based timeout (because the SDK does not support native timeouts well), httpx backends use `httpx.Client(timeout=...)`.
- Error handling: each adapter catches its own exceptions and converts to a common `BackendError` or returns a failed `BenchmarkResult`.
- The `_is_retryable()` function must be per-backend since error types differ.
- Connection lifecycle: `httpx.Client` should be created once per benchmark session (in adapter `__init__`) and closed after, not per-request.

**Phase:** Backend abstraction layer. Each adapter is self-contained.

---

### Pitfall 11: Qwen 3.5 MoE Hang -- The Motivating Bug

**What goes wrong:** This is a known Ollama bug (issues #14579, #14662) where Qwen 3.5 MoE models hang indefinitely during inference. The model loads but never produces output. The existing 200-second timeout eventually fires, but the student waits 3+ minutes per prompt for nothing, and the model is reported as "Timeout after 200s" rather than "known Ollama bug."

**Why this matters for multi-backend:** This is the PRIMARY motivation for adding llama.cpp support. Qwen 3.5 MoE works correctly and fast (5-7x faster than expected Ollama performance) in llama.cpp. If the multi-backend implementation does not handle this case well, the main value proposition is lost.

**Prevention:**
- When Ollama times out on a model that is known to hang (maintain a list of affected models, or detect the pattern: load succeeds but zero tokens generated), suggest: "This model may not work in Ollama. Try `--backend llama-cpp` instead."
- For cross-backend comparison: if Ollama fails/times out on a model, still show the llama.cpp result rather than omitting the entire model from comparison.
- Do NOT increase the timeout to "give Ollama more time" -- the hang is infinite. 200 seconds is already too long for a known hang. Consider a shorter initial timeout (30s) for the first few tokens, then extend.

**Detection:** Model loads successfully (warmup seems to work) but benchmark produces zero tokens. This is the hang pattern.

**Phase:** Early in backend integration. Test with Qwen 3.5 MoE as the primary validation case.

---

### Pitfall 12: SystemInfo Model Assumes Ollama Exists

**What goes wrong:** The `SystemInfo` Pydantic model has `ollama_version: str` as a required field. The `get_system_info()` function calls `get_ollama_version()` which runs `ollama --version`. If a student only uses llama.cpp (no Ollama installed), this field returns "Unknown" -- which is fine -- but the model and all exporters are structurally coupled to Ollama being the only backend.

**Specific coupling points in current code:**
- `SystemInfo.ollama_version` -- Ollama-specific field name
- `format_system_summary()` -- prints "Ollama {version}" unconditionally
- CSV exporter writes "Ollama" row in system info
- Markdown exporter does not mention backend at all
- JSON exporter does not record which backend produced each result
- `BenchmarkResult.response` is typed as `OllamaResponse | None` -- the response model is literally named `OllamaResponse`

**Prevention:**
- Rename `OllamaResponse` to `BackendResponse` (or create a new normalized model and keep `OllamaResponse` as an internal Ollama adapter type).
- Add `backend: str` field to `BenchmarkResult` so results record which backend produced them.
- Change `SystemInfo.ollama_version` to `backend_versions: dict[str, str]` (e.g., `{"ollama": "0.6.1", "llama_cpp": "b4567"}`).
- Update all exporters to include backend information per-result.
- This is a significant refactor. Plan it carefully or the diffs will be enormous and break all 152 tests.

**Phase:** Backend abstraction layer, but staged carefully. Rename models first, then update exporters, then tests. Do not attempt in one commit.

---

### Pitfall 13: Port Conflicts Between Backends

**What goes wrong:** Students run all three backends simultaneously for comparison. Default ports: Ollama 11434, llama.cpp 8080, LM Studio 1234. But llama.cpp's default port 8080 conflicts with many common services (web servers, proxies). If a student also runs a web dev server on 8080, llama.cpp silently fails to start or the benchmark tool connects to the wrong service.

**Prevention:**
- Support `--llama-cpp-host`, `--lm-studio-host` CLI flags for custom endpoints.
- Environment variables as fallback: `LLAMA_CPP_HOST`, `LM_STUDIO_HOST` (similar to Ollama's `OLLAMA_HOST`).
- Health check should verify the response is actually from the expected backend (e.g., llama.cpp `/health` returns a specific JSON shape, not a random web server's 200 OK).

**Phase:** CLI and configuration phase.

---

## Minor Pitfalls

### Pitfall 14: llama.cpp Server Build Variations

**What goes wrong:** llama.cpp server compiled with different backends (CUDA, Metal, Vulkan, CPU-only) reports different capabilities and performance. A student's Homebrew-installed `llama-server` might be CPU-only on macOS (missing Metal support), producing dramatically slower results that they attribute to the model rather than the build.

**Prevention:** Query `/props` or `/health` to detect compute backend. Display in system info: "llama.cpp server (Metal)" vs "llama.cpp server (CPU)". On macOS, warn if Metal is not detected.

**Phase:** System info and preflight phase.

---

### Pitfall 15: LM Studio Model Download and Load Latency

**What goes wrong:** LM Studio can download models on demand. If a model is not yet downloaded, the first API call may trigger a download, taking minutes. The benchmark timeout fires and reports the model as failed, when it was actually downloading.

**Prevention:** Check if model is loaded via `/api/v1/models` before benchmarking. If not loaded, call `/api/v1/models/load` explicitly and wait for the load event. If not downloaded, inform the student rather than timing out.

**Phase:** LM Studio backend adapter implementation.

---

### Pitfall 16: Result File Format Backward Compatibility

**What goes wrong:** Adding `backend` field to JSON results, changing `OllamaResponse` to a generic response type, and adding backend-specific timing fields breaks the `compare` and `analyze` subcommands for existing result files. Students who already have v1.0 results cannot compare them with v2.0 results.

**Prevention:**
- Design the JSON format to be additive: new fields are optional, old files parse correctly with defaults.
- `backend` field defaults to `"ollama"` when missing (backward compat).
- The compare subcommand should handle mixed-format files gracefully.
- Version the result format: add `"format_version": 2` to new files.

**Phase:** Exporter refactoring phase. Plan the JSON schema migration before writing code.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Severity | Mitigation |
|-------------|---------------|----------|------------|
| Backend protocol design | Timing unit mismatch (#1), model identity (#2), model management differences (#5) | CRITICAL | Design normalized response model first. Protocol must declare capabilities (multi-model vs single-model). |
| Ollama adapter refactoring | SystemInfo coupling (#12), OllamaResponse naming (#12), test breakage | CRITICAL | Rename types before adding new backends. Refactor in small PRs. Run tests after each rename. |
| llama.cpp adapter | Wrong endpoint (#3), prompt template missing (#8), server restart needed (#5), build variations (#14) | HIGH | Validate timing data in first integration test. Apply chat template client-side. Query `/props` for model and backend info. |
| LM Studio adapter | Wrong endpoint (#3), model download latency (#15), API version changes | MEDIUM | Verify `/api/v1/chat` returns stats object. Check model loaded state before benchmarking. |
| Preflight redesign | All-or-nothing exits (#4), backend not running, port conflicts (#13) | HIGH | Per-backend health checks, non-fatal for optional backends, 3-second connection timeout. |
| Cross-backend comparison | Model name mismatch (#2), context window differences (#7), prompt template differences (#8), timing normalization (#1) | CRITICAL | Explicit model mapping, enforce same num_ctx, normalize timings in adapter not in shared code. |
| Exporter updates | Backward compatibility (#16), missing backend info in results | MEDIUM | Additive JSON schema, version field, default backend to "ollama". |
| CLI changes | Port conflicts (#13), too many flags | LOW | Environment variable fallbacks, sensible defaults, interactive menu handles complexity. |

## Pre-Integration Checklist

Before writing any backend adapter code, verify:

- [ ] Normalized response model defined (canonical units: seconds, not ns/ms)
- [ ] Backend protocol includes: `health_check()`, `warmup()`, `run_benchmark()`, `list_models()`, `get_effective_context_size()`
- [ ] Protocol declares model management capability (SINGLE_MODEL vs MULTI_MODEL)
- [ ] `OllamaResponse` renamed or wrapped so new backends do not inherit Ollama-specific field names
- [ ] `BenchmarkResult` has `backend: str` field
- [ ] Each backend's native endpoint verified with real server (not just documented)
- [ ] Timing field units documented per backend with conversion formulas
- [ ] Existing 152 tests still pass after model renames

## Sources

- Direct codebase analysis of `runner.py`, `models.py`, `preflight.py`, `exporters.py`, `system.py`, `cli.py`
- PROJECT.md context: native API endpoints, known Ollama bugs, architecture decisions
- LM Studio REST API docs at lmstudio.ai/docs/api (MEDIUM confidence -- accessed 2026-03-14)
- llama.cpp server API: training data knowledge of `/completion`, `/health`, `/props` endpoints (LOW-MEDIUM confidence -- could not access current docs due to rate limiting; verify against actual llama.cpp server before implementation)
- Ollama API behavior: HIGH confidence (direct from working codebase and `ollama` SDK)
- Model naming conventions: HIGH confidence (observable from each tool's CLI/API)

**Confidence assessment:**
- Timing normalization pitfalls: HIGH (verified from codebase analysis of `_ns_to_sec()` usage and known API formats)
- Model identity pitfalls: HIGH (observable naming differences across backends)
- llama.cpp specific details (endpoint paths, response field names): MEDIUM (verify with actual server)
- LM Studio specific details (stats object fields): MEDIUM (docs accessed but field names should be verified)
- Architectural coupling pitfalls: HIGH (direct from codebase -- every coupling point identified by reading actual code)

---

*Pitfalls audit: 2026-03-14 -- Multi-backend integration focus*
