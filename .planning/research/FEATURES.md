# Feature Landscape

**Domain:** LLM benchmarking tool for local models via Ollama (student-facing)
**Researched:** 2026-03-12
**Reference:** Current codebase + `alexziskind1/llama-throughput-lab` external repo

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Status | Notes |
|---------|--------------|------------|--------|-------|
| Single-model throughput (t/s) | Core purpose of any benchmark tool | Low | EXISTS | `benchmark.py` and `extended_benchmark.py` both do this |
| Multi-model iteration | Users want to compare all installed models in one run | Low | EXISTS | Iterates `ollama list`, skip-list filtering |
| Prompt eval + response + total metrics | Three distinct throughput numbers are standard (input processing, generation, combined) | Low | EXISTS | Nanosecond precision from Ollama API |
| System info collection | Results without hardware context are meaningless for comparison | Med | EXISTS | CPU, RAM, GPU, Ollama version in `extended_benchmark.py` |
| Multiple export formats | JSON for programmatic use, Markdown for sharing, CSV for spreadsheets | Med | EXISTS | JSON, Markdown, CSV all implemented |
| Model offloading between benchmarks | Without unloading, cached model skews next model's metrics | Low | EXISTS | Uses `keep_alive=0` API call |
| Warmup runs | First inference includes model loading overhead, skewing results; every serious benchmark tool discards warmup | Med | NEEDED | llama-throughput-lab does 2 warmup requests before measurement. Standard practice in all benchmarking. Without this, first-run metrics are inflated by model load + KV cache allocation time |
| Retry with backoff | Ollama returns 5xx during model loading or under memory pressure; without retry, entire benchmark run fails partway through | Med | NEEDED | External repo retries on HTTP 500/502/503/504 and "Loading model" errors with linear backoff (`base_sleep_s * (attempt + 1)`). 8 retry attempts by default. Critical for reliability on machines with limited resources |
| Results directory structure | Result files cluttering project root is messy; students lose track of which file is which | Low | NEEDED | External repo uses `results/{subdir}/{prefix}_{timestamp}.csv` pattern. Simple `results/` directory with timestamped files |
| Cross-platform stability | Students use Windows+WSL, macOS, Linux; crashes on any platform = tool is broken | Med | NEEDED | SIGALRM already fixed to threading, sudo removed. Remaining: path separators, subprocess behavior differences |
| Pre-flight checks | Running a 30-minute benchmark only to fail because Ollama is not running is unacceptable | Low | NEEDED | Check Ollama is serving, sufficient RAM for target models, warn about GPU/VRAM limitations |
| Code consolidation (single entry point) | Two benchmark files (`benchmark.py` + `extended_benchmark.py`) confuse students about which to use | Med | NEEDED | Merge into single `benchmark.py` with the extended features |

## Differentiators

Features that set product apart. Not expected in every benchmark tool, but high value for student context.

| Feature | Value Proposition | Complexity | Status | Notes |
|---------|-------------------|------------|--------|-------|
| Concurrent benchmark mode | Shows how a model handles parallel requests -- real-world servers don't get one request at a time. Students learn that single-request t/s and multi-request aggregate throughput are very different numbers | High | NEEDED | External repo pattern: `ThreadPoolExecutor(max_workers=concurrency)` submitting N requests, measuring `total_tokens / elapsed_time`. Key parameters: concurrency level, total requests. Ollama supports parallel via its `--parallel` flag internally |
| Parameter sweep | Auto-explore `num_ctx`, `num_gpu`, `temperature` to find optimal config for student's hardware. Transforms tool from "measure one config" to "find best config" | High | NEEDED | External repo sweeps: instances x parallel x batch x ubatch x concurrency (5-dimensional grid). For Ollama context: sweep `num_ctx` (context window size affects memory/speed), `num_gpu` (GPU layer offloading), temperature. Track best throughput configuration |
| Interactive CLI menu | Students who are not CLI experts can navigate without memorizing flags | Med | NEEDED | Simple numbered menu: "1. Quick test, 2. Full benchmark, 3. Compare results". No TUI library needed -- just input() prompts |
| Quick verification mode | 30-second sanity check that confirms setup works before committing to a long benchmark | Low | NEEDED | Run smallest available model with 1 short prompt, report pass/fail |
| Shareable HTML/Markdown report | Students submit results for course assignments; needs to look professional with system info + rankings | Med | NEEDED | Already have Markdown export. Add ranked summary table at top, system info header. HTML is optional stretch goal |
| Visual ranked results (terminal) | Seeing a bar chart of model speeds in terminal makes results immediately actionable | Med | NEEDED | Simple ASCII bar chart. No matplotlib dependency. Print after benchmark completes |
| Results comparison tool | Compare benchmark runs across time or across machines; students want to see if hardware changes improved performance | Med | EXISTS (partial) | `compare_results.py` exists but is basic. Add: sort by metric, filter by model, highlight top-N |
| Predefined prompt sets (sized) | Students don't know what good benchmark prompts look like; sized sets (small/medium/large) let them pick based on available time | Low | EXISTS | Three sets already defined in `PROMPT_SETS` dict |

## Anti-Features

Features to explicitly NOT build. Each represents a complexity trap that would hurt the student experience.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Direct llama.cpp server management | Ollama already wraps llama.cpp. Managing server binaries, model paths, and build systems is a DevOps nightmare for students. The external repo (`llama-throughput-lab`) requires building llama.cpp from source, locating binaries, and setting 15+ environment variables -- that's expert-level tooling, not student-friendly | Stay on Ollama API exclusively. Ollama handles model downloading, server lifecycle, and GPU allocation |
| Nginx round-robin load balancing | External repo uses nginx to distribute across multiple llama.cpp instances. Requires nginx installation, dynamic config generation, process management. Adds massive complexity for a niche use case | If concurrent testing is needed, use Ollama's built-in parallel request handling (`OLLAMA_NUM_PARALLEL` env var) |
| Environment variable configuration | External repo uses 30+ env vars (`LLAMA_PROMPT`, `LLAMA_N_PREDICT`, `LLAMA_INSTANCES_LIST`, etc.) for configuration. Students will not read documentation to discover these | Use argparse CLI flags with clear `--help` output. Config file (YAML/TOML) as optional advanced feature |
| dialog/curses TUI | Adds ncurses dependency, breaks on Windows, complex to maintain | Simple input() menu with numbered choices works everywhere |
| Multi-machine distributed testing | Network-based benchmarking adds firewall, latency, authentication complexity | Single-machine focus. Students compare results via exported files |
| GPU memory profiling (nvidia-smi polling) | Polling nvidia-smi during inference adds timing overhead and is platform-specific | Report GPU info once at start (already done). Students can run nvidia-smi separately if curious |
| Real-time streaming dashboard | WebSocket server, browser UI, JavaScript -- massive scope increase | Terminal output with periodic progress updates is sufficient |
| Model quality evaluation (accuracy, perplexity) | Benchmarking throughput and evaluating output quality are fundamentally different tools. Quality eval requires ground truth datasets, scoring rubrics, and domain expertise | Stay focused on performance metrics (t/s, latency). Quality is orthogonal |

## Feature Dependencies

```
Cross-platform stability -----> All other features (foundation)
Code consolidation -----------> All new features (single codebase to extend)
Results directory structure --> Export formats (where files go)
                            --> Results comparison (where to find files)

Pre-flight checks ------------> Warmup runs (check before spending time)
                            --> Parameter sweep (validate before long sweep)

Warmup runs ------------------> Concurrent benchmark (warm before measuring)
                            --> Parameter sweep (warm before each config)

Retry with backoff -----------> Concurrent benchmark (parallel requests may hit transient errors)
                            --> Parameter sweep (many configs, some may fail)

Concurrent benchmark ---------> Parameter sweep (concurrency is one sweep dimension)

Quick verification mode ------> Interactive CLI menu (menu option 1)
```

## MVP Recommendation

Build in this priority order, based on dependencies and impact:

**Phase 1 -- Foundation (must happen first):**
1. Code consolidation (single `benchmark.py`) -- unblocks all subsequent work
2. Cross-platform stability fixes -- ensures nothing breaks on student machines
3. Results directory structure -- stop cluttering project root

**Phase 2 -- Reliability:**
4. Warmup runs -- immediate measurement accuracy improvement, low complexity
5. Retry with exponential backoff -- prevents partial benchmark failures
6. Pre-flight hardware check -- fails fast with actionable message

**Phase 3 -- Analysis and UX:**
7. Enhanced results analysis (sort, filter, top-N in comparison tool)
8. Visual ranked results (terminal bar chart)
9. Interactive CLI menu
10. Quick verification mode

**Phase 4 -- Advanced benchmarking:**
11. Concurrent benchmark mode -- significantly more complex, needs warmup + retry first
12. Parameter sweep -- most complex feature, needs concurrent mode as a dimension

**Defer indefinitely:**
- HTML reports (Markdown is sufficient for course submissions)
- Shareable report improvements beyond what Markdown provides

## Patterns Extracted from External Repo

Key implementation patterns from `llama-throughput-lab` that inform how features should work:

**Warmup pattern:** Send N throwaway requests (default: 2) with minimal token count (`n_predict=8`) before measurement begins. Purpose is to ensure model is loaded and KV cache is allocated.

**Retry pattern:** Linear backoff `sleep(base_sleep * (attempt + 1))` with 8 max attempts. Retry on HTTP 500/502/503/504 and "Loading model" string in error response. Non-retryable errors re-raise immediately.

**Concurrent pattern:** `ThreadPoolExecutor(max_workers=concurrency)` submitting `total_requests` futures. Measure wall-clock elapsed time and total tokens across all responses. Report aggregate throughput as `total_tokens / elapsed_time`. Track errors separately without failing the batch.

**Sweep pattern:** Nested loops over parameter dimensions. CSV output with one row per configuration. Track best configuration seen so far. `continue_on_error` flag to skip failed configurations rather than aborting entire sweep. Progress reporting to stderr.

**Results pattern:** Timestamped files in structured subdirectories. CSV for machine-readable sweep results. Print to stdout for immediate feedback, write to file for persistence.

## Sources

- Current codebase: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/` (benchmark.py, extended_benchmark.py, compare_results.py)
- External reference: `/Users/viktor/Documents/GitHub/llama-throughput-lab/` (full_sweep.py, test_llama_server_concurrent.py, test_llama_server_threads_sweep.py, llama_server_test_utils.py)
- Project requirements: `.planning/PROJECT.md`
- Architecture analysis: `.planning/codebase/ARCHITECTURE.md`
- Confidence: HIGH -- based on direct code analysis of both repos, not web sources
