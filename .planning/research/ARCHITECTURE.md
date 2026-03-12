# Architecture Patterns

**Domain:** LLM benchmarking tool (Python CLI, Ollama-based)
**Researched:** 2026-03-12
**Confidence:** HIGH (based on direct codebase analysis + established Python patterns)

## Current State

The tool is split across two benchmark files (`benchmark.py` at 219 lines, `extended_benchmark.py` at 1114 lines) plus a comparison tool (`compare_results.py` at 223 lines) and a launcher (`run.py`). The extended benchmark file is a monolith containing: data models, system info collection, lock file management, timeout utilities, Ollama interaction, metric calculation, three output formatters, and CLI argument parsing -- all in one file. Code is duplicated between `benchmark.py` and `extended_benchmark.py` (Pydantic models, benchmark execution logic).

## Recommended Architecture

Refactor into a flat Python package with clear module boundaries. No deep nesting -- this is a CLI tool for students, not a framework.

```
llm_benchmark/
    __init__.py          # Package version, public API
    __main__.py          # Entry point: python -m llm_benchmark
    cli.py               # Argument parsing, interactive menu, CLI output
    models.py            # All Pydantic data models (single source of truth)
    runner.py            # Core benchmark execution (single + batch)
    ollama_client.py     # Ollama API wrapper (list models, chat, offload)
    system_info.py       # Hardware/software detection
    metrics.py           # Throughput calculation, averaging, validation
    exporters/
        __init__.py
        markdown.py      # Markdown report generation
        json.py          # JSON export
        csv.py           # CSV export
        html.py          # Future: HTML report
    prompts.py           # Prompt set definitions and loading
    results.py           # Result loading, comparison, analysis
    utils.py             # Timeout wrapper, lock file, logging helpers
run.py                   # Thin launcher (kept for backward compatibility)
benchmark.py             # Deprecated wrapper -> imports from llm_benchmark
```

### Component Boundaries

| Component | Responsibility | Receives From | Sends To |
|-----------|---------------|---------------|----------|
| `cli.py` | Parse args, orchestrate flow, display progress | User input | `runner`, `exporters`, `system_info` |
| `models.py` | Define all data structures | None (imported by all) | All modules |
| `runner.py` | Execute benchmarks (single run, model sweep) | `cli.py` | `models.BenchmarkResult` |
| `ollama_client.py` | All Ollama API calls (list, chat, offload, ps) | `runner.py`, `cli.py` | Raw API responses |
| `system_info.py` | Detect CPU, RAM, GPU, OS, Ollama version | `cli.py` | `models.SystemInfo` |
| `metrics.py` | Calculate t/s, averages, validate results | `runner.py` | `models.BenchmarkResult`, `models.ModelBenchmarkSummary` |
| `exporters/*` | Serialize results to output formats | `cli.py` | Files on disk |
| `prompts.py` | Provide prompt sets, load custom prompts | `cli.py` | `List[str]` |
| `results.py` | Load past results, compare runs | `cli.py` | Comparison tables |
| `utils.py` | Timeout, lock file, logging | Various | Various |

### Data Flow

```
User CLI Input
    |
    v
cli.py (parse args, select mode)
    |
    +---> system_info.py --> SystemInfo
    |
    +---> prompts.py --> List[str]
    |
    +---> ollama_client.py --> List[model_name]
    |
    v
runner.py (for each model, for each prompt, for each run)
    |
    +---> ollama_client.py.chat() --> raw response
    |
    +---> metrics.py.calculate() --> BenchmarkResult
    |
    v
runner.py collects List[BenchmarkResult]
    |
    +---> metrics.py.summarize() --> ModelBenchmarkSummary
    |
    v
cli.py receives List[ModelBenchmarkSummary]
    |
    +---> exporters/markdown.py --> .md file
    +---> exporters/json.py --> .json file
    +---> exporters/csv.py --> .csv file
    +---> stdout (progress + summary table)
```

**Key data types that flow through the system:**

1. `OllamaResponse` -- validated raw API response (only used inside `runner.py` and `metrics.py`)
2. `BenchmarkResult` -- one run's calculated metrics (the core data unit)
3. `ModelBenchmarkSummary` -- aggregated results per model (what exporters consume)
4. `SystemInfo` -- hardware snapshot (attached to exports)

## Patterns to Follow

### Pattern 1: Thin Ollama Wrapper

Isolate all Ollama SDK calls behind `ollama_client.py`. This enables mocking for tests, handles SDK version changes in one place, and centralizes error handling for connection failures.

**What:** Every `ollama.*` call goes through `ollama_client.py`. No other module imports `ollama` directly.

**Why:** The Ollama SDK is unversioned and its response format could change. Wrapping it means one file to update when the SDK changes. It also makes testing trivial -- mock one module instead of patching calls scattered across files.

```python
# ollama_client.py
import ollama
from typing import List, Optional, Iterator

class OllamaClient:
    """Wrapper around Ollama SDK for testability and error isolation."""

    def list_models(self) -> List[str]:
        """Get names of all downloaded models."""
        models = ollama.list().models
        return [m.model for m in models]

    def chat_stream(self, model: str, prompt: str) -> Iterator[dict]:
        """Stream a chat response. Yields chunks."""
        return ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

    def offload_model(self, model: str) -> None:
        """Unload a model from memory."""
        ollama.generate(model=model, prompt="", keep_alive=0)

    def is_running(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            ollama.list()
            return True
        except Exception:
            return False
```

### Pattern 2: Calculation Separated from Collection

**What:** `runner.py` collects raw responses. `metrics.py` does all math. Runner never computes t/s directly.

**Why:** The current code mixes response collection with metric calculation inside `run_single_benchmark()`. This makes it hard to add new metrics (like std deviation, percentiles) or change calculation logic without touching benchmark execution. Separation means you can add statistical analysis without modifying the runner.

```python
# metrics.py
from models import OllamaResponse, BenchmarkResult

def calculate_throughput(response: OllamaResponse, model: str, prompt: str) -> BenchmarkResult:
    """Pure function: raw response in, calculated metrics out."""
    prompt_eval_ts = (
        response.prompt_eval_count / _ns_to_sec(response.prompt_eval_duration)
        if response.prompt_eval_duration > 0 else 0
    )
    # ... rest of calculation
    return BenchmarkResult(...)

def summarize_model(results: List[BenchmarkResult]) -> ModelBenchmarkSummary:
    """Aggregate individual runs into a model summary."""
    successful = [r for r in results if r.success]
    # ... averaging logic
```

### Pattern 3: Exporters as Pluggable Writers

**What:** Each output format is an independent module with the same interface: `def export(summaries, system_info, output_path)`.

**Why:** Adding HTML reports or terminal bar charts requires zero changes to existing code. Just add a new file in `exporters/`.

```python
# exporters/markdown.py
from models import ModelBenchmarkSummary, SystemInfo
from typing import List, Optional

def export(
    summaries: List[ModelBenchmarkSummary],
    output_path: str,
    system_info: Optional[SystemInfo] = None
) -> None:
    """Write benchmark results as Markdown."""
    with open(output_path, 'w') as f:
        # ... formatting logic
```

### Pattern 4: Models as the Shared Contract

**What:** `models.py` is the only module that every other module imports. It defines all Pydantic models. No circular dependencies.

**Why:** Currently, data models are duplicated across files. A single `models.py` eliminates duplication and serves as the schema documentation for the entire tool.

```
models.py <--- imported by all
    ^              |
    |              v
  (no module imports models.py AND is imported by models.py)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: God Orchestrator

**What:** Putting all orchestration logic (model iteration, prompt iteration, run iteration, progress display, error handling) in `cli.py`.

**Why bad:** The current `main()` in `extended_benchmark.py` is 230 lines of orchestration. Moving it all to `cli.py` repeats the problem. Instead, `runner.py` should own the benchmark loop, and `cli.py` should call `runner.run_all(config)` and get results back.

**Instead:** `cli.py` calls `runner.run_all_models(models, prompts, config)` which returns `List[ModelBenchmarkSummary]`. The runner owns iteration; the CLI owns display.

### Anti-Pattern 2: Deep Package Nesting

**What:** Creating `llm_benchmark/core/engine/benchmark/runner.py` style hierarchies.

**Why bad:** This is a student-facing tool. Students need to understand and potentially modify the code. Deep nesting makes navigation confusing and imports verbose.

**Instead:** Flat package. Every module at the same level. Max one level of nesting (only `exporters/` because it groups related but independent formatters).

### Anti-Pattern 3: Abstract Base Classes for One Implementation

**What:** Creating `BaseBenchmarkRunner`, `BaseExporter`, `BaseMetricsCalculator` with single concrete implementations.

**Why bad:** Over-engineering. There is one runner (Ollama). There will not be a llama.cpp runner (out of scope per PROJECT.md). ABCs add complexity students must understand with no benefit.

**Instead:** Simple functions and classes. If a second implementation is ever needed, refactor then.

### Anti-Pattern 4: Event-Driven Progress Reporting

**What:** Building a pub/sub event system for progress updates, logging, and status display.

**Why bad:** Adds architectural complexity for something `print()` handles. Students should be able to read the code and understand it.

**Instead:** Pass an optional `verbose: bool` or `on_progress: Callable` callback to the runner. Keep it simple.

## Dependency Graph (Build Order)

The dependency graph determines which modules must exist before others can be built. Build from the bottom up.

```
Layer 0 (no internal deps):
    models.py
    prompts.py
    utils.py

Layer 1 (depends on Layer 0):
    ollama_client.py    (imports: models)
    system_info.py      (imports: models)
    metrics.py          (imports: models)

Layer 2 (depends on Layer 1):
    runner.py           (imports: models, ollama_client, metrics, utils)
    exporters/*         (imports: models)
    results.py          (imports: models)

Layer 3 (depends on Layer 2):
    cli.py              (imports: everything)
    __main__.py         (imports: cli)
```

### Suggested Build Order for Phases

1. **Phase: Extract Models + Utils** -- Create `models.py` (move all Pydantic models), `utils.py` (timeout, lock file, nanosec_to_sec), and `prompts.py` (prompt sets). These have zero internal dependencies. Both existing files can immediately import from them. This is the lowest-risk refactor.

2. **Phase: Extract Ollama Client + System Info** -- Create `ollama_client.py` (wrap all `ollama.*` calls) and `system_info.py` (move `collect_system_info()`). Requires `models.py` to exist. Enables mocked unit tests.

3. **Phase: Extract Metrics + Runner** -- Create `metrics.py` (calculation logic) and `runner.py` (benchmark execution loop). Requires `ollama_client.py` and `models.py`. This is the core refactor -- splitting execution from calculation.

4. **Phase: Extract Exporters + Results** -- Create `exporters/` package (move three save functions) and `results.py` (move comparison logic from `compare_results.py`). Requires only `models.py`.

5. **Phase: Create CLI + Entry Points** -- Create `cli.py` (argparse, orchestration), `__main__.py`. Deprecate `benchmark.py`. Update `run.py` to launch `python -m llm_benchmark`.

### Why This Order

- Layers 0 and 1 can be extracted without changing any behavior -- pure moves with import updates. Safe, testable, immediately reduces duplication.
- Runner extraction (Layer 2) is the riskiest refactor because it touches the core benchmark loop. By the time you reach it, models, client, and metrics are already extracted and tested.
- CLI comes last because it depends on everything else existing. Building it first would force placeholder imports.

## Scalability Considerations

| Concern | Current (1 file) | After Refactor (package) | Future (concurrent) |
|---------|-------------------|--------------------------|---------------------|
| Adding new metrics | Edit 1100-line file, find right section | Add to `metrics.py` (50-100 lines) | Same |
| Adding output format | Add function to monolith | Add file in `exporters/` | Same |
| Unit testing | Must mock inline Ollama calls | Mock `ollama_client.py` | Same |
| Concurrent benchmarks | Not possible (sequential loop) | `runner.py` loop is isolated | Replace loop with `asyncio`/`concurrent.futures` in runner |
| Parameter sweeps | Would add 200+ lines to monolith | New `sweep.py` module that imports `runner` | Parallelize sweep combinations |
| Interactive menu | Would bloat main() further | Add to `cli.py` as separate mode | Same |

## Key Architectural Decision: Package vs Flat Files

**Decision:** Python package (`llm_benchmark/`) with `__main__.py` entry point.

**Why not just split into flat files in the root?** The project already has `run.py`, `run.sh`, `run.bat`, `run.ps1`, `compare_results.py`, `test_ollama.py`, `setup_passwordless_sudo.sh` in the root. Adding 8+ more `.py` files to the root creates a mess. A package provides clear "this is the tool" vs "this is supporting infrastructure" separation.

**Student impact:** `python -m llm_benchmark` is simple. `run.py` continues to work as the "just run it" entry point. Students who want to understand the code see a clean package with descriptive filenames.

## Sources

- Direct analysis of `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/` codebase (HIGH confidence)
- Python packaging conventions from Python Packaging Authority documentation (HIGH confidence)
- Standard patterns for CLI tool architecture in Python ecosystem (HIGH confidence)

---

*Architecture research: 2026-03-12*
