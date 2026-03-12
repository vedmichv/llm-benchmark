# Architecture

**Analysis Date:** 2026-03-12

## Pattern Overview

**Overall:** Modular Python benchmark suite with layered responsibility separation

**Key Characteristics:**
- Single-responsibility modules for different benchmark scenarios
- Data validation via Pydantic models
- Streaming-first design for long-running operations
- Timeout and error recovery mechanisms built-in
- System information collection and reporting
- Cross-platform support (Linux, macOS, Windows)

## Layers

**CLI/Entry Points:**
- Purpose: Command-line interfaces for users
- Location: `run.py`, `benchmark.py`, `extended_benchmark.py`
- Contains: Argument parsing, dependency checking, launcher logic
- Depends on: Ollama SDK, pydantic, system utilities
- Used by: End users and CI/CD pipelines

**Core Benchmark Logic:**
- Purpose: Execute LLM inference and collect performance metrics
- Location: Functions in `benchmark.py` and `extended_benchmark.py` (`run_benchmark()`, `run_single_benchmark()`)
- Contains: Streaming chat calls, response validation, metric calculation
- Depends on: Ollama SDK, OllamaResponse Pydantic models
- Used by: Main orchestration layer

**Data Models & Validation:**
- Purpose: Type-safe representation of Ollama responses and benchmark results
- Location: Lines 14-200 in both `benchmark.py` and `extended_benchmark.py`
- Contains: `Message`, `OllamaResponse`, `BenchmarkResult`, `ModelBenchmarkSummary`, `SystemInfo` Pydantic models
- Depends on: pydantic library
- Used by: All layers for data passing and validation

**System Analysis:**
- Purpose: Collect hardware and environment information for benchmarking context
- Location: `collect_system_info()` in `extended_benchmark.py` (lines 220-340)
- Contains: CPU detection, RAM detection, GPU detection, Ollama version detection
- Depends on: subprocess, platform stdlib, nvidia-smi (optional)
- Used by: Reporting and result contextualization

**Output Generation:**
- Purpose: Serialize benchmark results in multiple formats
- Location: `save_results_to_markdown()`, `save_results_to_json()`, `save_results_to_csv()` in `extended_benchmark.py` (lines 675-868)
- Contains: Template rendering, file writing, format conversion
- Depends on: json, csv stdlib modules
- Used by: Main orchestration after benchmarking completes

**Utilities & Helpers:**
- Purpose: Cross-cutting concerns (timeouts, logging, resource management)
- Location: `extended_benchmark.py` (lines 61-436)
- Contains: Lock file management, signal handling, timeout wrappers, model unloading, step logging
- Depends on: threading, signal, os, subprocess
- Used by: All layers

## Data Flow

**Basic Benchmark Flow (benchmark.py):**

1. Parse CLI arguments (`argparse`)
2. Get available models from Ollama (`ollama.list()`)
3. Filter skip-list models
4. For each model:
   - For each prompt:
     - Run benchmark via `run_benchmark()` (streaming or non-streaming)
     - Parse response via `OllamaResponse.model_validate()`
     - Calculate throughput metrics via `inference_stats()`
   - Calculate averaged stats via `average_stats()`

**Extended Benchmark Flow (extended_benchmark.py):**

1. Validate prerequisites and create lock file
2. Collect system information via `collect_system_info()`
3. Parse configuration (models, prompts, runs-per-prompt)
4. Ensure no models running via `ensure_no_models_running()`
5. For each model:
   - For each prompt:
     - For each run (configurable runs-per-prompt):
       - Test model load via `test_model_load()`
       - Run single benchmark via `run_single_benchmark()` with timeout wrapper
       - Collect result in `BenchmarkResult`
     - Calculate model summary via `benchmark_model()`
   - (Optional) Offload model via `offload_model()`
6. Save aggregated results to Markdown/JSON/CSV
7. Clean up lock file via `atexit` handler

**State Management:**

- **In-flight data:** `List[OllamaResponse]` → `List[BenchmarkResult]` → `ModelBenchmarkSummary`
- **Persistence:** Results written to timestamped markdown/JSON/CSV files
- **Lock state:** Temp file `ollama_benchmark.lock` prevents concurrent runs
- **Model state:** Tracked via `ollama ps` command, unloaded via `ollama.generate(keep_alive=0)`

## Key Abstractions

**OllamaResponse Model:**
- Purpose: Safely parse and validate Ollama API responses with special handling for prompt caching
- Location: `extended_benchmark.py` lines 150-168
- Pattern: Pydantic BaseModel with field validator for `prompt_eval_count` edge case
- Handles: Nanosecond timing fields, message nesting, optional fields

**BenchmarkResult Model:**
- Purpose: Immutable record of a single benchmark run with calculated throughput metrics
- Location: `extended_benchmark.py` lines 171-185
- Pattern: Data class-like Pydantic model storing pre-computed metrics
- Contains: Input (model, prompt), output (tokens, throughput), timing, success flag

**ModelBenchmarkSummary Model:**
- Purpose: Aggregate statistics across all runs of a single model
- Location: `extended_benchmark.py` lines 188-200
- Pattern: Holds raw run list plus pre-computed averages
- Used by: Output generation to avoid recalculating averages

**SystemInfo Model:**
- Purpose: Immutable snapshot of test environment for reproducibility
- Location: `extended_benchmark.py` lines 203-217
- Pattern: Collected once at startup, included in all outputs
- Includes: GPU detection, CUDA version, CPU cores, OS details

## Entry Points

**run.py (Primary launcher):**
- Location: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/run.py`
- Triggers: User runs `python run.py [args]`
- Responsibilities:
  - Check Python version (3.8+)
  - Verify Ollama installed and running
  - Create/use virtual environment
  - Install dependencies
  - Delegate to extended_benchmark.py

**benchmark.py (Simple benchmark):**
- Location: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/benchmark.py`
- Triggers: User runs `python benchmark.py [args]`
- Responsibilities:
  - Quick benchmarking without system info or model offloading
  - Default prompts ("Why is the sky blue?", "Write a report...")
  - Outputs to stdout only (no file persistence)

**extended_benchmark.py (Full-featured benchmark):**
- Location: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/extended_benchmark.py`
- Triggers: User runs `python extended_benchmark.py [args]` or indirectly via `run.py`
- Responsibilities:
  - Complete benchmarking with system profiling
  - Model offloading/unloading between runs
  - Multi-format output (Markdown, JSON, CSV)
  - Concurrency prevention via lock file
  - Timeout management for long-running benchmarks

**compare_results.py (Analysis tool):**
- Location: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/compare_results.py`
- Triggers: User runs `python compare_results.py file1.json file2.json [...]`
- Responsibilities:
  - Load and parse JSON result files
  - Display side-by-side comparison tables
  - Calculate performance deltas and percentage changes
  - Support multiple result sets

## Error Handling

**Strategy:** Defensive layers with graceful degradation

**Patterns:**

- **Validation Errors:** Pydantic validators catch malformed Ollama responses, apply defaults (e.g., `prompt_eval_count = -1 → 0`)
- **Timeout Errors:** `run_with_timeout()` uses threading + join timeout to enforce per-benchmark limits (not platform-dependent SIGALRM)
- **Model Loading Failures:** `test_model_load()` catches timeout/error, retrieves Ollama logs, reports to user
- **Lock File Conflicts:** Check if process in lock file still running before failing; clean up stale locks
- **Signal Handling:** `signal_handler()` catches SIGINT/SIGTERM, cleans up lock file on Ctrl+C
- **Dependency Errors:** `run.py` gracefully continues without venv if `python3-venv` unavailable
- **CLI Errors:** ArgumentParser validates argument types; bad choices rejected at parse time
- **System Call Failures:** Fallback to "Unknown" for undetectable system properties (CPU model, GPU info, Ollama version)

## Cross-Cutting Concerns

**Logging:**
- Pattern: Print to stdout/stderr only (no dedicated logger)
- Timestamps: Applied in `log_step()` for extended_benchmark.py key milestones
- Levels: Info (✓), Warning (⚠), Error (✗), organized with `=====` dividers

**Validation:**
- Pattern: Pydantic BaseModel for all data structures (OllamaResponse, BenchmarkResult, etc.)
- Enforcement: Applied at parse time; field_validators handle domain-specific rules (prompt caching)

**Authentication:**
- Pattern: None explicit; relies on Ollama server availability
- Design: Ollama SDK handles connection details via environment or default localhost:11434

**Concurrency:**
- Pattern: Single-process-at-a-time enforcement via lock file at start of extended_benchmark.py
- Prevention: PID stored, checked if still alive before failing
- Cleanup: atexit + signal handlers ensure lock removal

---

*Architecture analysis: 2026-03-12*
