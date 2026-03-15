# Phase 2: Measurement Reliability - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Ensure benchmark numbers are trustworthy: warmup excludes cold-start overhead, transient failures retry automatically, prompt caching is visibly handled, and all result files go to results/ directory. Requirements: BENCH-01, BENCH-02, BENCH-07, UX-04.

</domain>

<decisions>
## Implementation Decisions

### Warmup behavior
- 1 warmup run per model, before all prompts (not per-prompt)
- Use a short fixed prompt (e.g., "Hello") just to load the model into GPU/RAM
- Show brief status line: "Warming up llama3.2:1b..." then "Ready"
- Add --skip-warmup CLI flag (default: warmup on)
- When --skip-warmup is used, show brief note: "Warmup skipped -- first run may include model load time"

### Retry strategy
- 3 retries by default with exponential backoff (using tenacity library)
- Add --max-retries N CLI flag (default: 3, set to 0 to disable)
- Retryable errors: connection errors, Ollama server errors, timeouts
- After retries exhausted: mark run as failed (success=False), continue to next prompt/run
- Show retry count during attempts: "Retry 1/3..."

### Prompt caching visibility
- Per-run indicator: show [cached] tag next to affected runs in terminal output
- First time caching is detected in a benchmark session, show one-liner explanation: "Prompt caching: Ollama reused the prompt from memory, so prompt eval time is 0."
- When ALL runs for a model are cached, show warning: "All runs cached -- prompt eval metrics unavailable for this model"
- All export formats (JSON, CSV, Markdown) include prompt caching indicators
- In verbose mode, cached runs still stream response text normally

### Results organization
- Flat results/ directory with timestamped filenames (current behavior)
- Add .gitignore inside results/ to prevent accidental commits of benchmark data
- Auto-create results/ on first run (exporters already do mkdir)
- No symlinks, no subdirectories, no .gitkeep

### Claude's Discretion
- Exact backoff timing for retries (e.g., 1s/2s/4s or similar)
- Exact warmup prompt text
- Which tenacity exception types map to retryable errors
- .gitignore patterns inside results/

</decisions>

<specifics>
## Specific Ideas

- Warmup should feel like a quick "loading..." step, not a full benchmark pass
- Students should always understand why prompt eval shows 0 -- the educational hint matters
- The tool should never crash mid-benchmark on a transient error; always recover or mark failed and continue

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_single_benchmark()` in runner.py: wrap with tenacity retry decorator
- `compute_averages()` in runner.py: already excludes prompt_cached from prompt eval calculations
- `OllamaResponse.handle_prompt_caching` in models.py: already detects caching and sets flag
- `_ensure_dir()` in exporters.py: already creates output directory
- `run_with_timeout()` in runner.py: timeout mechanism to integrate with retry logic

### Established Patterns
- Rich Console singleton for all terminal output (config.py)
- Lazy imports in CLI handlers for fast startup
- Constants in config.py (DEFAULT_TIMEOUT, DEFAULT_RUNS_PER_PROMPT)
- BenchmarkResult with success/error pattern for failure handling

### Integration Points
- `benchmark_model()` in runner.py: add warmup call before prompt loop
- `_build_parser()` in cli.py: add --skip-warmup and --max-retries flags
- `_handle_run()` in cli.py: pass new flags through to runner
- exporters: CSV and Markdown need prompt_cached column/indicator added

</code_context>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 02-measurement-reliability*
*Context gathered: 2026-03-12*
