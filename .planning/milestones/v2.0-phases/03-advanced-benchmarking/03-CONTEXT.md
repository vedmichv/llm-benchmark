# Phase 3: Advanced Benchmarking - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Add concurrent benchmarking (--concurrent N), parameter sweep (--sweep), and results analysis (analyze subcommand + enhanced compare). Students can stress-test models with parallel requests, auto-find optimal configurations, and inspect/compare results. Requirements: BENCH-03, BENCH-04, BENCH-05, BENCH-06, ANLZ-01, ANLZ-02, ANLZ-03.

</domain>

<decisions>
## Implementation Decisions

### Concurrent mode behavior
- Use asyncio + httpx AsyncClient for true parallel HTTP requests (not threads)
- Single warmup run per model before all concurrent requests (consistent with Phase 2 pattern)
- Failed concurrent requests marked as failed (success=False), other in-flight requests continue
- --concurrent N flag: when N is omitted, auto-detect based on available RAM/GPU
- All N workers send the SAME prompt simultaneously (measures true concurrent throughput)
- Aggregate throughput displayed as wall-time aggregate (total_tokens/wall_time) PLUS per-request average
- Combinable with --runs-per-prompt: each "run" fires N concurrent requests (e.g., --concurrent 4 --runs-per-prompt 3 = 3 rounds of 4 parallel)

### Parameter sweep design
- Sweep num_ctx (512, 1024, 2048, 4096) and num_gpu (0, 50% layers, 100% layers)
- Auto-detect model's total layer count from Ollama to determine num_gpu values
- Report as ranked Rich table of all configurations + highlighted "Best config" recommendation
- Use single short prompt per configuration (fast sweep, not full prompt set)
- Sweep applies to each model individually (per-model best config since optimal settings differ by model size)
- Show config counter: "Testing config 3/12: num_ctx=2048, num_gpu=16..."
- Sweep results saved to separate files: sweep_YYYYMMDD_HHMMSS.{json,csv,md} in results/

### Results analysis commands
- New 'analyze' subcommand: `llm_benchmark analyze results/file.json --sort response_ts --top 3`
- Sortable metrics: response_ts, total_ts, prompt_eval_ts, load_time
- Default sort order: descending (fastest first), --asc flag to reverse
- Single file only (use compare for multi-file)
- Model averages by default, --detail flag for per-run breakdown
- --top N filters by models (top N models by average throughput), not individual runs
- Terminal output only (no file export from analyze)

### Compare enhancements
- Add Unicode arrows (up/down) with color for visual diff indicators
- Add "winner" column to comparison tables
- Ensure compare works with concurrent and sweep result formats

### Output & reporting
- Concurrent terminal output: per-request live results as they complete ("Request 1/4: 234 tokens @ 32.1 t/s") then aggregate summary line
- Extended JSON/CSV/Markdown schema with 'mode' field ('standard', 'concurrent', 'sweep') and mode-specific data
- Sweep Markdown report: ranked config table with bold/green highlighted best row + "Recommended" label

### Claude's Discretion
- httpx vs ollama async client specifics
- Exact auto-detect heuristic for --concurrent N default
- num_ctx values to sweep (suggested 512/1024/2048/4096 but can adjust)
- Sweep prompt text choice
- analyze Rich table layout and column formatting
- Compare arrow placement and color scheme

</decisions>

<specifics>
## Specific Ideas

- Concurrent mode should feel like "stress testing" — students want to know "how fast can my hardware go when pushed?"
- Sweep is educational — students learn that configuration tuning matters, not just model choice
- The analyze subcommand is for quick inspection ("what was my fastest model?"), not deep analysis
- Compare should make it immediately obvious which run was faster without reading numbers carefully

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_single_benchmark()` in runner.py: core benchmark logic, needs async variant for concurrent mode
- `warmup_model()` in runner.py: single warmup before concurrent batch
- `compute_averages()` in runner.py: needs extension for concurrent aggregate metrics (wall-time)
- `compare.py`: existing Rich table comparison, extend with arrows/winner column
- `_result_to_dict()` in exporters.py: extend with mode-specific fields
- `_ensure_dir()` in exporters.py: reuse for results/ directory creation
- tenacity retry in runner.py: integrate with async concurrent requests

### Established Patterns
- Rich Console singleton for all terminal output (config.py)
- Lazy imports in CLI handlers for fast startup (cli.py)
- Constants in config.py (DEFAULT_TIMEOUT, DEFAULT_RUNS_PER_PROMPT, DEFAULT_MAX_RETRIES)
- BenchmarkResult with success/error pattern for failure handling
- Argparse subcommands with _build_parser() and _HANDLERS dict

### Integration Points
- `_build_parser()` in cli.py: add --concurrent, --sweep flags to run subparser; add analyze subparser
- `_handle_run()` in cli.py: route to concurrent or sweep mode based on flags
- `benchmark_model()` in runner.py: parallel entry point for concurrent mode
- `exporters.py`: add mode field and mode-specific sections to all three export formats
- `compare.py`: enhance with visual diff indicators

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-advanced-benchmarking*
*Context gathered: 2026-03-12*
