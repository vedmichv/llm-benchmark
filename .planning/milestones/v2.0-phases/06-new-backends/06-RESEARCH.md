# Phase 6: New Backends - Research

**Researched:** 2026-03-14
**Domain:** LLM backend integration (llama.cpp server, LM Studio), cross-platform detection, CLI integration
**Confidence:** HIGH

## Summary

Phase 6 implements two new backends (LlamaCppBackend and LMStudioBackend) following the existing Backend Protocol established in Phase 5. Both backends communicate via HTTP using httpx -- llama.cpp via its native `/completion` and `/v1/chat/completions` endpoints, LM Studio via its OpenAI-compatible `/v1/chat/completions` and native `/api/v1/` endpoints. The Backend Protocol, BackendResponse model, BackendError, StreamResult, and create_backend() factory are already in place; the new backends only need to implement the protocol interface and handle timing conversion.

The llama.cpp server (`llama-server`) provides a `timings` object in its OpenAI-compatible chat responses with millisecond-precision fields. LM Studio provides `tokens_per_second` in response metadata and standard OpenAI `usage` fields. Both require process/server lifecycle management -- llama-server serves one model at a time (restart per model), while LM Studio manages models via load/unload API endpoints.

**Primary recommendation:** Implement backends in three waves: (1) LlamaCppBackend + LMStudioBackend core classes, (2) detection/auto-start/server lifecycle management, (3) CLI integration + menu + exporters + system info updates.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Support both explicit `--model-path /path/to/file.gguf` (CLI) and directory scanning (menu mode) for llama.cpp
- Default scan directory: `~/.cache/huggingface/` for GGUF discovery
- Extract model name from GGUF metadata when available, fallback to cleaned filename
- Restart llama-server per model for multi-model runs -- unload_model() stops the server process
- Query LM Studio API (`/api/v1/models`) for model discovery
- Show full downloaded catalog and offer to load models (not just currently-loaded)
- Detection: binary check (`shutil.which`) + port probe (Ollama:11434, llama-server:8080, LM Studio:1234)
- Distinguish "installed" (binary found) from "running" (port responds)
- Auto-start: ask permission first -- "llama-server is installed but not running. Start it? (y/N)"
- For llama-cpp: collect model selection BEFORE starting server (server needs `--model` flag)
- Optional `--port` flag for custom port override per backend
- Startup timeout: 30 seconds with retries (poll every 1s)
- Backend selection appears BEFORE mode selection in menu
- When only Ollama detected: still show backend prompt with auto-select and note
- `--backend` flag accepts: `ollama`, `llama-cpp`, `lm-studio` (default: `ollama`)
- Backend name included in export filenames and JSON metadata
- System summary shows ALL detected backends with status
- `python -m llm_benchmark info` shows full backend inventory
- Model failure: skip model + log warning, continue (no interactive prompt per failure)
- Show failure summary at end with all skipped models and errors
- Known-issues hint table: hardcoded Python dict mapping (backend, error_pattern) to hint
- Backend server stdout/stderr captured to log file on auto-start
- Backend not installed: show platform-specific installation instructions
- Preflight checks run ONLY for the selected backend

### Claude's Discretion
- llama.cpp unload_model() implementation details (stop server process aligns with restart-per-model)
- GGUF metadata parsing approach
- LM Studio model loading API calls
- Known-issues dict initial entries and matching logic
- Server log rotation/cleanup policy

### Deferred Ideas (OUT OF SCOPE)
- `--backend all` (run all backends sequentially) -- Phase 7
- Cross-backend comparison matrix and reports -- Phase 7
- "Compare backends" as menu option 5 -- Phase 7
- Automatic model name matching across backends -- rejected
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BEND-01 | LlamaCppBackend connects to llama-server via httpx, reads native /completion timings | llama.cpp timings object documented: prompt_ms, predicted_ms, prompt_n, predicted_n, prompt_per_second, predicted_per_second |
| BEND-02 | LMStudioBackend connects to LM Studio via httpx, reads native /api/v1/ stats | LM Studio provides usage stats + tokens_per_second in response metadata |
| BEND-03 | Auto-detect installed backends by checking binary presence (shutil.which) | Binary names: `ollama`, `llama-server`, `lms`; pattern already used in check_ollama_installed() |
| BEND-04 | Auto-start backends if installed but not running | Start commands: `ollama serve`, `llama-server -m <model> --port <port>`, `lms server start`; subprocess management with health polling |
| BEND-05 | Backend-specific preflight checks (non-fatal: skip unavailable backends gracefully) | Generalize existing preflight chain; each backend gets connectivity + model availability checks |
| CLI-01 | `--backend` flag accepts ollama, llama-cpp, lm-studio (default: ollama) | Add to argparse run_parser; `--backend all` deferred to Phase 7 |
| CLI-02 | Interactive menu shows detected backends and lets user choose | Add backend selection step before mode selection in menu.py |
| CLI-03 | Backend name included in export filenames, JSON metadata, and Markdown reports | Modify exporters to accept backend_name parameter for filename generation |
| CLI-04 | System summary shows backend name and version | Extend format_system_summary() to show all detected backends |
| CLI-05 | Backend choice only prompted when >1 backend detected | Detection logic returns list; menu shows selection only when len > 1 (but per CONTEXT: always show with note) |
| PLAT-01 | All backends work on macOS, Windows, Linux | httpx is cross-platform; binary detection uses shutil.which (cross-platform); subprocess start commands vary per OS |
| PLAT-02 | llama.cpp install detection and auto-start per OS | Binary: `llama-server` (all platforms); install: brew (macOS), apt/snap (Linux), winget (Windows) |
| PLAT-03 | LM Studio install detection and auto-start per OS | Binary: `lms` CLI; install from lmstudio.ai; server start: `lms server start` (all platforms) |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpx | >=0.28 | HTTP client for llama-server and LM Studio API calls | Already available (transitive dep), async-capable, modern Python HTTP client |
| shutil.which | stdlib | Binary detection for backends | Cross-platform, already used for Ollama detection |
| subprocess | stdlib | Server process management (start/stop llama-server) | Cross-platform process control |
| socket | stdlib | Port probing for backend detection | Lightweight connectivity check without full HTTP request |
| pathlib | stdlib | GGUF file scanning in ~/.cache/huggingface/ | Already used throughout project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| struct | stdlib | GGUF metadata header parsing | Extract model name from GGUF file metadata |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx | requests | httpx already available, no new dependency needed |
| struct for GGUF | gguf-parser lib | Extra dependency; only need model name, not full parse |
| socket port probe | httpx health check | Socket is faster for "is port open?" check; httpx for actual API verification |

**Installation:**
```bash
# httpx is already a transitive dependency; add explicitly for clarity
uv add httpx
```

## Architecture Patterns

### Recommended Project Structure
```
llm_benchmark/
  backends/
    __init__.py          # Backend Protocol, BackendResponse, create_backend() [EXISTS]
    ollama.py            # OllamaBackend [EXISTS]
    llamacpp.py          # NEW: LlamaCppBackend
    lmstudio.py          # NEW: LMStudioBackend
    detection.py         # NEW: detect_backends(), auto-start logic, port probing
```

### Pattern 1: Backend Implementation Pattern
**What:** Each backend class implements the Backend Protocol using httpx for HTTP communication.
**When to use:** For every new backend.
**Example:**
```python
# Source: Derived from existing OllamaBackend pattern + llama.cpp server docs
class LlamaCppBackend:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self._base_url = f"http://{host}:{port}"
        self._client = httpx.Client(timeout=600.0)
        self._server_process: subprocess.Popen | None = None

    @property
    def name(self) -> str:
        return "llama-cpp"

    def chat(self, model, messages, stream=False, options=None):
        # Use /v1/chat/completions for chat format
        # Response includes timings object with ms-precision fields
        payload = {"model": model, "messages": messages, "stream": stream}
        resp = self._client.post(f"{self._base_url}/v1/chat/completions", json=payload)
        data = resp.json()
        timings = data.get("timings", {})
        return self._to_response(data, timings)

    def _to_response(self, data, timings):
        return BackendResponse(
            model=data.get("model", ""),
            content=data["choices"][0]["message"]["content"],
            done=True,
            prompt_eval_count=timings.get("prompt_n", 0),
            eval_count=timings.get("predicted_n", 0),
            prompt_eval_duration=timings.get("prompt_ms", 0) / 1000.0,  # ms -> sec
            eval_duration=timings.get("predicted_ms", 0) / 1000.0,
            total_duration=(timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) / 1000.0,
            load_duration=0.0,  # Not in per-request timings
        )
```

### Pattern 2: Server Lifecycle Management (llama.cpp)
**What:** Start/stop llama-server subprocess, poll /health for readiness.
**When to use:** llama-cpp backend auto-start and model switching.
**Example:**
```python
def _start_server(self, model_path: str, port: int = 8080) -> None:
    cmd = ["llama-server", "-m", model_path, "--port", str(port), "--host", "127.0.0.1"]
    self._server_process = subprocess.Popen(
        cmd, stdout=open("results/llama-server.log", "a"),
        stderr=subprocess.STDOUT,
    )
    # Poll /health until ready (30s timeout, 1s interval)
    for _ in range(30):
        try:
            resp = self._client.get(f"{self._base_url}/health")
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(1.0)
    raise BackendError("llama-server failed to start within 30s")

def unload_model(self, model: str) -> bool:
    if self._server_process:
        self._server_process.terminate()
        self._server_process.wait(timeout=10)
        self._server_process = None
    return True
```

### Pattern 3: Backend Detection
**What:** Detect installed vs running backends using binary check + port probe.
**When to use:** Backend selection menu, system info display.
**Example:**
```python
@dataclass
class BackendStatus:
    name: str
    installed: bool
    running: bool
    binary_path: str | None
    port: int

def detect_backends() -> list[BackendStatus]:
    backends = []
    checks = [
        ("ollama", "ollama", 11434),
        ("llama-cpp", "llama-server", 8080),
        ("lm-studio", "lms", 1234),
    ]
    for name, binary, port in checks:
        installed = shutil.which(binary) is not None
        running = _port_is_open("127.0.0.1", port)
        backends.append(BackendStatus(name, installed, running, shutil.which(binary), port))
    return backends
```

### Anti-Patterns to Avoid
- **Using OpenAI-compat endpoints for timing data:** The OpenAI-compatible `/v1/chat/completions` on llama.cpp DOES include timings in the response (in the `timings` object), so it CAN be used. However, be aware that the standard OpenAI `usage` field only has token counts, not timing. Always read from the `timings` object for performance data.
- **Keeping llama-server running between models:** llama-server loads one model at startup; switching models requires restart. Do not try to swap models on a running server.
- **Blocking on server startup without timeout:** Always implement health-check polling with a maximum timeout.
- **Importing httpx at module level in __init__.py:** Keep httpx imports inside backend classes to avoid ImportError for users who only use Ollama.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP client | Raw urllib/socket | httpx.Client | Connection pooling, timeouts, streaming, error handling |
| GGUF metadata | Full GGUF parser | Minimal struct-based header reader | Only need model name from metadata; GGUF format header is simple (magic + version + metadata count + kv pairs) |
| Process management | Complex daemon system | subprocess.Popen + terminate/wait | llama-server is a simple foreground process; no need for systemd/launchd integration |
| Port checking | HTTP request to check port | socket.connect_ex() | Much faster for "is anything listening on this port?" check |

**Key insight:** Both llama.cpp and LM Studio expose standard HTTP APIs. The complexity is in lifecycle management (starting/stopping servers, model switching), not in the API communication itself.

## Common Pitfalls

### Pitfall 1: llama.cpp Timing Fields Are in Milliseconds
**What goes wrong:** Assuming timings are in seconds (like final throughput) or nanoseconds (like Ollama).
**Why it happens:** llama.cpp `timings` object uses milliseconds for duration fields (`prompt_ms`, `predicted_ms`) but provides pre-calculated `predicted_per_second` and `prompt_per_second`.
**How to avoid:** Convert ms to seconds: `duration_sec = timings["predicted_ms"] / 1000.0`
**Warning signs:** Throughput numbers are 1000x too high or too low.

### Pitfall 2: llama-server Requires Model at Startup
**What goes wrong:** Trying to start llama-server without a model, or trying to switch models via API.
**Why it happens:** Unlike Ollama/LM Studio, llama-server loads exactly one model specified via `-m` flag at startup.
**How to avoid:** Collect model path first, then start server. For multi-model benchmarks, stop and restart server with each model.
**Warning signs:** Server fails to start, or model endpoints return errors.

### Pitfall 3: LM Studio Model State
**What goes wrong:** Sending inference requests when no model is loaded in LM Studio.
**Why it happens:** LM Studio may have models downloaded but not loaded into memory.
**How to avoid:** Check `/api/v1/models` or `/v1/models` for loaded status. If nothing loaded, use load API or prompt user.
**Warning signs:** 404 or model-not-found errors from inference endpoint.

### Pitfall 4: Port Conflicts
**What goes wrong:** Auto-starting a backend fails because another process already uses the default port.
**Why it happens:** User may have another service on port 8080 (common web dev port) or already running a llama-server.
**How to avoid:** Check port before auto-start. Support `--port` override. Provide clear error message.
**Warning signs:** "Address already in use" errors in server logs.

### Pitfall 5: Cross-Platform Binary Names
**What goes wrong:** Binary detection fails on some platforms.
**Why it happens:** On Windows, executables have `.exe` extension but `shutil.which` handles this. However, some tools install to non-standard locations.
**How to avoid:** `shutil.which()` already handles PATH search and Windows extensions. For LM Studio on macOS, the `lms` CLI may need to be installed separately via `lms bootstrap`.
**Warning signs:** Binary not found despite application being installed.

### Pitfall 6: Streaming Response Parsing
**What goes wrong:** Timing data not captured from streamed responses.
**Why it happens:** In SSE streaming mode, timing data only appears in the final chunk. For llama.cpp `/v1/chat/completions` with `stream: true`, the `timings` object is in the last SSE event.
**How to avoid:** Accumulate content from intermediate chunks, extract timings from the final `data: [DONE]`-adjacent chunk.
**Warning signs:** BackendResponse has zero timings after streaming.

## Code Examples

### llama.cpp /v1/chat/completions Response with Timings
```json
// Source: llama.cpp server README (ggml-org/llama.cpp, tools/server/README.md)
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "model": "model-alias",
  "timings": {
    "cache_n": 236,
    "prompt_n": 1,
    "prompt_ms": 30.958,
    "prompt_per_token_ms": 30.958,
    "prompt_per_second": 32.30,
    "predicted_n": 35,
    "predicted_ms": 661.064,
    "predicted_per_token_ms": 18.887,
    "predicted_per_second": 52.94
  },
  "usage": {
    "prompt_tokens": 237,
    "completion_tokens": 35,
    "total_tokens": 272
  }
}
```

### llama.cpp /health Endpoint
```python
# Source: llama.cpp server README
# GET /health (or /v1/health)
# Returns 200 {"status": "ok"} when ready
# Returns 503 {"error": {"code": 503, "message": "Loading model"}} during load
def check_connectivity(self) -> bool:
    try:
        resp = self._client.get(f"{self._base_url}/health", timeout=5.0)
        return resp.status_code == 200
    except httpx.ConnectError:
        return False
```

### LM Studio Model Management
```python
# Source: LM Studio REST API docs (lmstudio.ai/docs)
# List models: GET /api/v1/models
# Load model: POST /api/v1/models/load  {"model": "model-id"}
# Unload model: POST /api/v1/models/unload  {"model": "model-id"}
# Chat: POST /v1/chat/completions (OpenAI-compatible, includes usage)

def list_models(self) -> list[dict]:
    resp = self._client.get(f"{self._base_url}/api/v1/models")
    data = resp.json()
    return [
        {"model": m["id"], "size": 0}  # LM Studio doesn't report size in list
        for m in data.get("data", [])
    ]
```

### LM Studio Response Format
```python
# LM Studio /v1/chat/completions response includes standard OpenAI usage
# plus additional performance metadata
# {
#   "choices": [...],
#   "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N},
#   "stats": {"tokens_per_second": 45.2, ...}  # LM Studio extension
# }
def _to_response(self, data: dict) -> BackendResponse:
    usage = data.get("usage", {})
    stats = data.get("stats", {})
    content = data["choices"][0]["message"]["content"]
    eval_count = usage.get("completion_tokens", 0)
    tps = stats.get("tokens_per_second", 0)
    # Derive eval_duration from tokens_per_second
    eval_duration = eval_count / tps if tps > 0 else 0.0
    return BackendResponse(
        model=data.get("model", ""),
        content=content,
        done=True,
        prompt_eval_count=usage.get("prompt_tokens", 0),
        eval_count=eval_count,
        eval_duration=eval_duration,
        total_duration=eval_duration,  # Best estimate without separate total
        prompt_eval_duration=0.0,  # Not provided separately
        load_duration=0.0,
    )
```

### GGUF Model Name Extraction
```python
# GGUF file format: magic (4 bytes) + version (4 bytes LE) + metadata_count (8 bytes LE) + ...
# Metadata is key-value pairs; look for "general.name" key
import struct

def extract_gguf_model_name(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return None
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            # Parse KV pairs looking for "general.name"
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                value_type = struct.unpack("<I", f.read(4))[0]
                if key == "general.name" and value_type == 8:  # STRING type
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    return f.read(str_len).decode("utf-8", errors="replace")
                else:
                    _skip_gguf_value(f, value_type)  # Skip non-matching values
    except Exception:
        pass
    return None
```

### Backend Detection and Port Probing
```python
import socket

def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    finally:
        sock.close()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| llama.cpp separate examples/ dir | llama.cpp tools/server/ (renamed) | 2024-2025 | Binary name is `llama-server`, docs moved to tools/server/README.md |
| llama-server single-model only | llama-server `--models-dir` router mode | Recent (2025) | Server can now auto-load models from a directory; but single-model mode is simpler for benchmarking |
| LM Studio no CLI | LM Studio `lms` CLI tool | 2024 (v0.3+) | `lms server start`, `lms load`, etc. available for automation |
| OpenAI-compat only | Native timings in OAI-compat responses | llama.cpp 2024+ | `timings` object added to `/v1/chat/completions` responses, not just `/completion` |

**Deprecated/outdated:**
- `/completion` endpoint on llama.cpp: Still works but is NOT OpenAI-compatible. Prefer `/v1/chat/completions` which includes `timings` object and uses chat message format.
- `llama-server` used to be in `examples/server/`; now in `tools/server/`.

## Open Questions

1. **LM Studio `stats` object exact field names**
   - What we know: Response includes `tokens_per_second` and speculative decoding stats
   - What's unclear: Exact field names may vary by LM Studio version; `stats` vs nested in `usage`
   - Recommendation: Implement with defensive `.get()` access; validate against real LM Studio server during development; LOW confidence on exact field names

2. **LM Studio `/api/v1/models` vs `/v1/models` distinction**
   - What we know: `/v1/models` is OpenAI-compatible (lists loaded models); `/api/v1/models` is LM Studio native (lists all downloaded)
   - What's unclear: Whether `/api/v1/models` includes load status per model
   - Recommendation: Use `/api/v1/models` for full catalog discovery, `/v1/models` for currently-loaded check

3. **GGUF metadata parsing edge cases**
   - What we know: GGUF v2+ format has well-defined header with typed KV pairs
   - What's unclear: Some GGUF files may lack "general.name" metadata
   - Recommendation: Fallback to cleaned filename (strip `.gguf`, replace `-` with spaces)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BEND-01 | LlamaCppBackend chat + timing conversion | unit | `uv run pytest tests/test_llamacpp.py -x` | No -- Wave 0 |
| BEND-02 | LMStudioBackend chat + usage parsing | unit | `uv run pytest tests/test_lmstudio.py -x` | No -- Wave 0 |
| BEND-03 | detect_backends() binary + port check | unit | `uv run pytest tests/test_detection.py -x` | No -- Wave 0 |
| BEND-04 | auto-start server with health polling | unit | `uv run pytest tests/test_detection.py::test_auto_start -x` | No -- Wave 0 |
| BEND-05 | Backend-specific preflight checks | unit | `uv run pytest tests/test_preflight.py -x` | Partial (Ollama tests exist) |
| CLI-01 | --backend flag parsing | unit | `uv run pytest tests/test_cli.py -x` | Partial (parser tests exist) |
| CLI-02 | Menu backend selection | unit | `uv run pytest tests/test_menu.py -x` | Partial (menu tests exist) |
| CLI-03 | Backend name in exports | unit | `uv run pytest tests/test_exporters.py -x` | Partial (exporter tests exist) |
| CLI-04 | System summary with backend info | unit | `uv run pytest tests/test_system.py -x` | Partial (system tests exist) |
| CLI-05 | Backend prompt conditional display | unit | `uv run pytest tests/test_menu.py -x` | Partial |
| PLAT-01 | Cross-platform backend operation | manual-only | N/A | N/A -- requires actual backends |
| PLAT-02 | llama.cpp per-OS detection | unit | `uv run pytest tests/test_detection.py::test_llamacpp_detection -x` | No -- Wave 0 |
| PLAT-03 | LM Studio per-OS detection | unit | `uv run pytest tests/test_detection.py::test_lmstudio_detection -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_llamacpp.py` -- covers BEND-01: LlamaCppBackend protocol compliance, timing conversion, error wrapping
- [ ] `tests/test_lmstudio.py` -- covers BEND-02: LMStudioBackend protocol compliance, usage parsing, model management
- [ ] `tests/test_detection.py` -- covers BEND-03, BEND-04, PLAT-02, PLAT-03: backend detection, port probing, auto-start
- [ ] Update `tests/conftest.py` -- add fixtures for mock llama-cpp and lm-studio backends
- [ ] httpx explicit dependency: `uv add httpx` (currently transitive only)

## Sources

### Primary (HIGH confidence)
- llama.cpp server README (ggml-org/llama.cpp, tools/server/README.md via GitHub API) -- timings object fields, server flags, health endpoint, /v1/chat/completions format
- Existing OllamaBackend source code (llm_benchmark/backends/ollama.py) -- reference implementation pattern
- Existing Backend Protocol (llm_benchmark/backends/__init__.py) -- interface contract

### Secondary (MEDIUM confidence)
- LM Studio REST API docs (lmstudio.ai/docs/api/rest-api) -- endpoint list, model management, server CLI
- LM Studio OpenAI API docs (lmstudio.ai/docs/api/openai-api) -- chat completion format, usage stats

### Tertiary (LOW confidence)
- LM Studio `stats` object field names -- only from high-level docs; exact schema needs validation against real server
- LM Studio `/api/v1/models` response format -- inferred from docs; exact fields need validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- httpx already available, all stdlib tools well-known
- Architecture: HIGH -- follows established Backend Protocol pattern from Phase 5
- llama.cpp API: HIGH -- timings object verified from official README via GitHub API
- LM Studio API: MEDIUM -- endpoint list verified but exact response field names need validation
- Pitfalls: HIGH -- based on documented API behavior and common subprocess patterns
- GGUF parsing: MEDIUM -- format is well-documented but edge cases exist

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (llama.cpp server API is stable; LM Studio may update more frequently)
