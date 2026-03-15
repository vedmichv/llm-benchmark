# Phase 7: Cross-Backend Comparison - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Students can compare the same model across all detected backends (Ollama, llama.cpp, LM Studio) side-by-side and see which runtime is fastest on their hardware. Includes `--backend all` CLI mode, comparison reports, menu option 5, and README documentation for multi-backend setup.

Requirements: COMP-01, COMP-02, COMP-03, COMP-04, COMP-05, DOC-01, DOC-02

</domain>

<decisions>
## Implementation Decisions

### Comparison report layout
- Single-model view: side-by-side horizontal bar chart with backends as bars (reuse/extend render_bar_chart pattern)
- Multi-model view: Rich matrix table (models as rows, backends as columns, response t/s values, winner starred per row)
- Winner metric: response t/s (eval_count/eval_duration) -- what students care about most
- Missing backend/model combos show "--" in matrix (no error, no blocking)
- Overall recommendation: "Fastest backend: X (N/M models)" at bottom of matrix
- Export: Markdown + JSON comparison report alongside individual per-backend result files

### Execution flow for --backend all
- Sequential per backend: run ALL models on Backend A, then ALL on Backend B, etc.
- Auto-detect backends via detect_backends() -- only run on what's actually running
- If only 1 backend detected: warn and fall back to standard single-backend benchmark (don't block)
- llama-cpp GGUF selection: auto-scan ~/.cache/huggingface with scan_gguf_files(), match by model name similarity to Ollama names, skip if no match
- Each backend's results saved individually (normal export) PLUS unified comparison report
- All existing flags (--prompt-set, --runs-per-prompt, --concurrent, etc.) apply identically to each backend run

### Menu option 5 ("Compare backends")
- Always visible in menu (option 5) regardless of how many backends detected
- Auto-run comparison with medium prompts, 2 runs -- minimal friction
- If only 1 backend when selected: show "Install another backend to compare" with install hints
- Modes 1-4 keep existing backend selector when >1 detected (Phase 6 behavior unchanged)
- Mode 5 auto-uses all detected backends (no backend selection step)

### README documentation
- New "Multi-Backend Setup" section after existing Quick Start
- Per-OS guides: one-liner install + verify command for each backend (macOS/Windows/Linux)
- New "Cross-Backend Comparison" section with CLI usage and crafted-but-realistic example output
- Example shows llama.cpp ~1.5x faster than Ollama on Apple Silicon (realistic numbers)

### Claude's Discretion
- GGUF-to-Ollama model name matching algorithm details
- Comparison JSON schema structure
- Exact terminal formatting and spacing of comparison output
- How to handle --concurrent with --backend all (sequential backends each running concurrent internally, or error)
- llama-cpp single-model-per-server handling during sequential comparison runs

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `display.py`: `render_bar_chart()` and `render_text_bar_chart()` -- extend for backend comparison (backends as entries instead of models)
- `compare.py`: `compare_results()` -- different purpose (file-to-file comparison) but Rich Table patterns reusable
- `menu.py`: `scan_gguf_files()` and `extract_gguf_model_name()` -- reuse for auto-matching GGUFs to model names
- `menu.py`: `select_backend_interactive()` -- reference for backend detection UX patterns
- `exporters.py`: `_filename()` helper -- extend for comparison report filenames
- `backends/detection.py`: `detect_backends()` -- core detection for --backend all

### Established Patterns
- Rich Console singleton for all output (`get_console()`)
- Rich Table for tabular data (compare.py, sweep.py)
- Backend-aware filename generation via `_filename()` in exporters.py
- Sequential model runs with unload between (runner.py)
- Lazy imports for detection module to avoid circular deps

### Integration Points
- `cli.py`: Add "all" to --backend choices, add comparison handler
- `menu.py`: Add option 5 to `run_interactive_menu()`
- `runner.py`: Orchestrate multi-backend sequential runs
- `exporters.py`: New comparison exporter functions (Markdown + JSON)
- `display.py`: New comparison bar chart variant
- `README.md`: New sections for multi-backend setup and comparison

</code_context>

<specifics>
## Specific Ideas

- Bar chart for single-model comparison should look like existing model rankings but with backend names and a star on the fastest
- Matrix table should use "--" for missing combos, star on per-row winner
- "Fastest backend: llama.cpp (2/3 models)" summary line at bottom
- README example should show realistic Apple Silicon numbers (llama.cpp ~62 t/s vs Ollama ~45 t/s)
- Menu option 5 should auto-run with zero additional prompts when backends are detected

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-cross-backend-comparison*
*Context gathered: 2026-03-14*
