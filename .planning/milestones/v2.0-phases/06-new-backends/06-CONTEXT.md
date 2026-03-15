# Phase 6: New Backends - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Add llama.cpp and LM Studio backends implementing the existing Backend Protocol, with CLI integration (`--backend` flag), auto-detection, auto-start, and cross-platform support. `--backend all` and cross-backend comparison are Phase 7 scope.

</domain>

<decisions>
## Implementation Decisions

### Model Management — llama.cpp
- Support both explicit `--model-path /path/to/file.gguf` (CLI) and directory scanning (menu mode)
- Default scan directory: `~/.cache/huggingface/` where huggingface-cli downloads GGUFs
- Extract model name from GGUF metadata when available, fallback to cleaned filename (strip .gguf extension)
- Restart llama-server per model for multi-model runs — unload_model() stops the server process, next benchmark starts it with the new model
- llama-server serves one model at a time; tool manages the server lifecycle transparently

### Model Management — LM Studio
- Query LM Studio API (`/api/v1/models`) for model discovery
- Show full downloaded catalog and offer to load models (not just currently-loaded)
- If nothing is loaded, tell student to load a model via LM Studio UI or let the tool load it via API

### Model Management — Cross-backend
- For Phase 7 cross-backend comparison: user maps models manually (e.g., `--models ollama:llama3:8b llama-cpp:path/to/llama3.gguf`)
- No automatic model name matching heuristics between backends

### Auto-detect & Auto-start
- Detection: binary check (`shutil.which`) + port probe (Ollama:11434, llama-server:8080, LM Studio:1234)
- Distinguish "installed" (binary found) from "running" (port responds)
- Auto-start: ask permission first — "llama-server is installed but not running. Start it? (y/N)"
- For llama-cpp: collect model selection BEFORE starting server (server needs `--model` flag)
- Optional `--port` flag for custom port override per backend
- Startup timeout: 30 seconds with retries (poll every 1s)

### Backend Selection UX — Menu
- Backend selection appears BEFORE mode selection (Quick/Standard/Full/Custom)
- When only Ollama detected: still show backend prompt with auto-select and note "Only Ollama detected" — makes students aware other backends exist
- When multiple backends detected: show numbered list with status (running/installed/not found)

### Backend Selection UX — CLI
- `--backend` flag accepts: `ollama`, `llama-cpp`, `lm-studio` (default: `ollama`)
- `--backend all` deferred to Phase 7
- Backend name always included in export filenames: `benchmark_ollama_2026-03-14.json`, `benchmark_llama-cpp_2026-03-14.json`
- Backend name in JSON metadata and Markdown reports

### Backend Selection UX — Info & Summary
- System summary (pre-benchmark) shows ALL detected backends with status (running/installed/not found)
- `python -m llm_benchmark info` shows full backend inventory with versions and status

### Error Handling & Graceful Degradation
- Model failure: skip model + log warning, continue with remaining models (no interactive prompt per failure)
- Show failure summary at end of benchmark run with all skipped models and their errors
- Known-issues hint table: hardcoded Python dict mapping (backend, error_pattern) to human-readable hint — e.g., "qwen3.5:3b failed on Ollama (timeout) — Known issue: try llama-cpp backend instead"
- When auto-starting backend: capture server stdout/stderr to log file (e.g., `results/llama-server.log`), show log path on errors
- Backend not installed: show platform-specific installation instructions (brew/apt/winget), matching existing Ollama install prompt pattern
- Preflight checks run ONLY for the selected backend (not all detected)

### Claude's Discretion
- llama.cpp unload_model() implementation details (stop server process aligns with restart-per-model)
- GGUF metadata parsing approach
- LM Studio model loading API calls
- Known-issues dict initial entries and matching logic
- Server log rotation/cleanup policy

</decisions>

<specifics>
## Specific Ideas

- "Ideally log collection during startup to diagnose model issues — like when we had the Qwen 3.5 problem on Ollama" — capture backend server logs to file for debugging
- Known-issues hints inspired by the real Qwen 3.5 MoE Ollama bug (#14579, #14662) — when a model fails on one backend, suggest trying another
- Student awareness: even Ollama-only users should see the backend selection to know alternatives exist

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Backend` Protocol (`backends/__init__.py`): Fully defined with chat(), list_models(), unload_model(), warmup(), detect_context_window(), get_model_size(), check_connectivity()
- `BackendResponse` model: Normalized timing in seconds — new backends convert their native formats internally
- `BackendError` exception: Unified error with `retryable` flag — new backends raise this
- `StreamResult` wrapper: Iterator-based streaming with deferred finalize
- `create_backend()` factory: Currently only "ollama" — extend with "llama-cpp" and "lm-studio" branches
- `OllamaBackend` (~300 lines): Reference implementation showing the full pattern to follow

### Established Patterns
- Backend detection: `shutil.which("ollama")` already used in `check_ollama_installed()` — extend for llama-server and lms
- Preflight chain: connectivity (blocking) → models (blocking) → RAM (advisory) — generalize for any backend
- CLI hardcodes `create_backend()` with no arguments — needs `--backend` arg passed through
- Menu builds `argparse.Namespace` — needs backend field added
- Timing conversion: Ollama ns→seconds in `_ns_to_sec()` — llama.cpp uses ms, LM Studio pre-computed

### Integration Points
- `cli.py:_build_parser()` — add `--backend` and `--port` and `--model-path` to run parser
- `cli.py:_handle_run()` — pass backend name to `create_backend()` instead of hardcoded default
- `cli.py` no-args path — pass backend to `run_interactive_menu()`
- `preflight.py:run_preflight_checks()` — generalize install check from Ollama-only to backend-aware
- `menu.py:run_interactive_menu()` — add backend selection step before mode selection
- `exporters.py` — include backend name in output filenames
- `system.py:format_system_summary()` — show all detected backends
- New files: `backends/llamacpp.py`, `backends/lmstudio.py`, `backends/detection.py`

</code_context>

<deferred>
## Deferred Ideas

- `--backend all` (run all backends sequentially) — Phase 7
- Cross-backend comparison matrix and reports — Phase 7
- "Compare backends" as menu option 5 — Phase 7
- Automatic model name matching across backends — rejected in favor of manual mapping

</deferred>

---

*Phase: 06-new-backends*
*Context gathered: 2026-03-14*
