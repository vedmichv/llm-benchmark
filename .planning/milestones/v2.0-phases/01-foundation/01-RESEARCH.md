# Phase 1: Foundation - Research

**Researched:** 2026-03-12
**Domain:** Python package restructuring, CLI tooling, pre-flight checks, cross-platform stability
**Confidence:** HIGH

## Summary

Phase 1 consolidates two parallel scripts (benchmark.py and extended_benchmark.py) into a proper Python package (`llm_benchmark/`) with subcommands, pre-flight checks for Ollama connectivity and hardware, and cross-platform error handling using `rich` for terminal output. The existing codebase already has ~1100 lines of working logic that needs to be restructured, not rewritten.

The primary technical challenge is not algorithmic but organizational: splitting the monolithic extended_benchmark.py into well-separated modules while preserving all existing behavior (system info collection, export formats, prompt sets, model offloading, threading-based timeouts). The secondary challenge is implementing pre-flight checks that provide actionable, platform-specific guidance when things are wrong.

**Primary recommendation:** Migrate extended_benchmark.py's logic into the new package structure module-by-module, using `rich` for all terminal output and `tenacity` for retry logic. Use `pyproject.toml` with hatchling as the build backend. The ollama SDK already provides `ConnectionError` with a helpful message -- wrap it with platform-specific start instructions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Migrate everything from extended_benchmark.py into new package structure; benchmark.py becomes redundant and is removed
- compare_results.py becomes a subcommand (`python -m llm_benchmark compare`)
- Remove setup_passwordless_sudo.sh (obsolete since offloading uses keep_alive=0 API)
- Keep run.py as a thin convenience wrapper; remove run.sh/run.bat/run.ps1
- Package layout: `llm_benchmark/` with `__init__.py`, `__main__.py`, `cli.py`, `runner.py`, `models.py`, `system.py`, `exporters.py`, `preflight.py`, `prompts.py`, `config.py`
- CLI uses explicit subcommands: `run`, `compare`, `info`
- `pyproject.toml` for modern Python packaging (replaces requirements.txt)
- Python 3.12+ minimum, `uv` as recommended package manager/runner
- Dependencies: ollama, pydantic, rich, tenacity
- Ollama connectivity check blocks execution with platform-specific fix instructions
- RAM/GPU check warns but continues (don't gatekeep)
- Pre-flight runs automatically on every `run` command; slow checks skippable with `--skip-checks`
- No models found: suggest pulling `ollama pull llama3.2:1b` and exit
- Compact one-line system summary before benchmark; full details via `info` subcommand
- Friendly + actionable error messages with emoji indicators
- No stack traces unless `--debug` flag
- Mid-benchmark model failure: ask user whether to continue (`Continue with remaining models? [Y/n]`)
- `--debug` for full stack traces, `--verbose` for response streaming (separate concerns)
- Use `rich` for colored/formatted terminal output
- English only

### Claude's Discretion
- Exact rich formatting choices (panels, tables, colors)
- Internal module boundaries (what helper functions go where)
- Pydantic model field naming and validation approach
- pyproject.toml build backend choice (hatchling, setuptools, etc.)
- Exact RAM estimation heuristic for model size warnings

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STAB-01 | Benchmark runs without crashes on Windows, macOS, and Linux | Package structure + cross-platform patterns from extended_benchmark.py; threading-based timeouts (no SIGALRM); rich handles terminal differences |
| STAB-02 | Tool checks Ollama connectivity before starting and shows actionable error if unreachable | ollama SDK raises `ConnectionError` with message; wrap with platform-specific start instructions in `preflight.py` |
| STAB-03 | Tool checks available RAM and GPU before benchmark and warns if resources insufficient | `ollama.list()` returns model `size` field; `ollama.ps()` returns `size_vram`; system RAM via platform-specific calls already in extended_benchmark.py |
| STAB-04 | Throughput averaging uses total_tokens/total_time (not arithmetic mean of rates) | Current extended_benchmark.py uses arithmetic mean of rates (bug); fix in `runner.py` to sum tokens / sum time |
| STAB-05 | Model offloading works without sudo via Ollama API (keep_alive=0) | Already implemented in extended_benchmark.py via `ollama.generate(model=name, prompt="", keep_alive=0)`; migrate to `runner.py` |
| STAB-06 | Timeouts work cross-platform via threading (no signal.SIGALRM) | Already implemented in extended_benchmark.py via `run_with_timeout()` using `threading.Thread`; migrate to `runner.py` |
| QUAL-01 | Single consolidated benchmark module | Core deliverable: merge benchmark.py + extended_benchmark.py into `llm_benchmark/` package |
| QUAL-02 | Python package structure (llm_benchmark/ with submodules) | Package layout defined in CONTEXT.md; pyproject.toml with hatchling |
| QUAL-05 | Python >=3.10 requirement (Pydantic 2.x + tenacity compatibility) | User locked Python 3.12+; tenacity 9.x requires >=3.10; pydantic 2.x requires >=3.9; all compatible |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ollama | 0.6.1 | Ollama API client (chat, list, generate, ps) | Official Python SDK; already used in codebase |
| pydantic | 2.12.x | Response validation, data models | Already used; required by ollama SDK (>=2.9) |
| rich | 14.3.x | Terminal formatting (colors, tables, panels, emoji) | De facto standard for Python CLI output; user decision |
| tenacity | 9.1.x | Retry with exponential backoff | Standard retry library; user decision; requires Python >=3.10 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hatchling | >=1.26 | Build backend for pyproject.toml | Package building only (build-system requires) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hatchling | setuptools | Hatchling is simpler config, no setup.cfg needed; setuptools more widely known but more verbose |
| argparse | click/typer | argparse is stdlib, no extra dep; click/typer add dependency for marginal gain at this scale |

**Installation (for users):**
```bash
uv run python -m llm_benchmark run
```

**Installation (for development):**
```bash
uv sync
```

## Architecture Patterns

### Recommended Project Structure
```
llm_benchmark/
├── __init__.py        # Package version, top-level imports
├── __main__.py        # Entry point: `python -m llm_benchmark`
├── cli.py             # argparse with subcommands (run, compare, info)
├── runner.py          # Benchmark execution, timeout wrapper, model offloading
├── models.py          # Pydantic models (Message, OllamaResponse, BenchmarkResult, ModelBenchmarkSummary, SystemInfo)
├── system.py          # Hardware/system info collection (CPU, RAM, GPU across platforms)
├── exporters.py       # JSON, CSV, Markdown output writers
├── preflight.py       # Ollama connectivity check, RAM/GPU warnings, model availability
├── prompts.py         # Prompt sets (small/medium/large)
├── config.py          # Defaults, constants (timeouts, default prompt set, etc.)
└── compare.py         # Results comparison logic (from compare_results.py)
pyproject.toml         # Package metadata + dependencies
run.py                 # Thin wrapper: `python run.py` -> `python -m llm_benchmark`
```

### Pattern 1: Global Exception Handler in `__main__.py`
**What:** Catch all exceptions at the top level; show friendly messages by default, stack traces with `--debug`.
**When to use:** Always -- this is the single entry point.
**Example:**
```python
# __main__.py
import sys
from llm_benchmark.cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(130)
    except Exception as e:
        # --debug flag would be parsed in cli.py and stored globally
        from llm_benchmark.config import is_debug
        if is_debug():
            raise
        from rich.console import Console
        console = Console(stderr=True)
        console.print(f"[red bold]Error:[/] {e}")
        console.print("[dim]Run with --debug for full traceback[/dim]")
        sys.exit(1)
```

### Pattern 2: Pre-flight Check Chain in `preflight.py`
**What:** Ordered sequence of checks that either block (Ollama connectivity, no models) or warn (RAM/GPU).
**When to use:** Automatically on every `run` command.
**Example:**
```python
# preflight.py
import ollama
from rich.console import Console

console = Console()

def check_ollama_connectivity() -> bool:
    """Returns True if Ollama is reachable. Exits with guidance if not."""
    try:
        ollama.list()
        return True
    except Exception:
        # ConnectionError from httpx underneath
        import platform
        os_name = platform.system()
        console.print("[red bold]Cannot connect to Ollama[/red bold]")
        console.print("")
        if os_name == "Darwin":
            console.print("  Start Ollama from your Applications folder, or run:")
            console.print("  [cyan]ollama serve[/cyan]")
        elif os_name == "Windows":
            console.print("  Start the Ollama application from the Start menu")
        else:
            console.print("  Start Ollama with:")
            console.print("  [cyan]ollama serve[/cyan]")
        console.print("")
        console.print("[dim]Download Ollama: https://ollama.com/download[/dim]")
        return False

def check_available_models() -> list:
    """Returns model list or exits with suggestion to pull a model."""
    response = ollama.list()
    models = response.models
    if not models:
        console.print("[yellow]No models found![/yellow]")
        console.print("  Pull a small model to get started:")
        console.print("  [cyan]ollama pull llama3.2:1b[/cyan]")
        return []
    return models

def check_ram_for_models(models, skip_models=None):
    """Warn if any model may exceed available RAM. Non-blocking."""
    # Model size from ollama.list() is disk size
    # Rough heuristic: GGUF in RAM ~ 1.0-1.2x disk size
    # Compare against system RAM
    pass
```

### Pattern 3: Rich Console as Singleton
**What:** Single `Console()` instance shared across modules for consistent output.
**When to use:** All terminal output.
**Example:**
```python
# config.py
from rich.console import Console

_console = Console()
_debug = False

def get_console() -> Console:
    return _console

def set_debug(enabled: bool):
    global _debug
    _debug = enabled

def is_debug() -> bool:
    return _debug
```

### Pattern 4: Subcommand-based CLI with argparse
**What:** Top-level parser with subparsers for `run`, `compare`, `info`.
**When to use:** CLI entry point.
**Example:**
```python
# cli.py
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llm_benchmark",
        description="Benchmark your Ollama models"
    )
    parser.add_argument("--debug", action="store_true", help="Show full stack traces")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--verbose", "-v", action="store_true")
    run_parser.add_argument("--skip-checks", action="store_true")
    run_parser.add_argument("--skip-models", nargs="*", default=[])
    run_parser.add_argument("--prompt-set", choices=["small", "medium", "large"], default="medium")
    # ... more args

    # compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument("files", nargs="+")

    # info subcommand
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args(argv)
    # dispatch to handler
```

### Anti-Patterns to Avoid
- **Printing directly with print():** Use `rich.console.Console` everywhere for consistent formatting. The existing code uses raw `print()` with manual ANSI codes (see `run.py` Colors class) -- replace entirely with rich.
- **Side-effect validators in Pydantic:** The current `OllamaResponse.validate_prompt_eval_count` prints a warning as a side effect. Move warning logic out of the model validator into the calling code.
- **Arithmetic mean of rates for averaging:** Current `benchmark_model()` averages `t/s` values. This is mathematically wrong -- use `total_tokens / total_time` instead (STAB-04).
- **Platform-specific signal handling:** `extended_benchmark.py` registers SIGINT/SIGTERM handlers globally. This is fragile -- use a `try/except KeyboardInterrupt` pattern instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Terminal colors/formatting | ANSI escape codes, custom Colors class | `rich` | Handles terminal capability detection, Windows support, NO_COLOR env var |
| Retry with backoff | Custom retry loops | `tenacity` | Handles jitter, max attempts, exception filtering, logging |
| CLI argument parsing | Custom arg handling | `argparse` (stdlib) | Already standard, subcommand support built in |
| Package metadata | setup.py / requirements.txt | `pyproject.toml` | PEP 621 standard, understood by uv/pip/hatch |
| Cross-platform RAM detection | Subprocess calls to sysctl/proc | Platform-specific code in `system.py` (keep existing pattern) | psutil would add a C extension dependency; the existing subprocess approach works and has no compile requirement |

**Key insight:** The existing codebase already solves most cross-platform problems (RAM detection, GPU detection, threading timeouts). The work is restructuring, not reimplementing. Don't introduce psutil -- it's a C extension that can cause install issues for students.

## Common Pitfalls

### Pitfall 1: Averaging Rates vs Averaging Totals
**What goes wrong:** Computing `mean(tokens_per_second)` across runs gives a different (wrong) result than `sum(tokens) / sum(time)`.
**Why it happens:** Rates are not additive. A slow run with many tokens biases the arithmetic mean upward.
**How to avoid:** Sum all tokens, sum all durations, divide once. This is STAB-04.
**Warning signs:** Average t/s that seems too high compared to individual runs.

### Pitfall 2: Ollama ConnectionError vs ResponseError
**What goes wrong:** Catching generic `Exception` instead of specific ollama errors, leading to confusing messages.
**Why it happens:** The ollama SDK raises `httpx.ConnectError` (wrapped as `ConnectionError` by ollama) when the server is unreachable, and `ollama.ResponseError` when the server responds with an error.
**How to avoid:** In preflight, catch `Exception` broadly (since the exact exception type depends on httpx internals). In benchmark runs, catch `ollama.ResponseError` specifically for model-level errors.
**Warning signs:** "Connection refused" errors showing up as model failures instead of connectivity issues.

### Pitfall 3: Windows Terminal Emoji Support
**What goes wrong:** Emoji characters render as boxes or garbled text on older Windows terminals.
**Why it happens:** Windows cmd.exe and older PowerShell versions don't support Unicode emoji well.
**How to avoid:** Rich handles this automatically when using its markup syntax (`:white_check_mark:` style). Alternatively, use simple ASCII indicators as fallback. Rich's `Console()` detects terminal capability.
**Warning signs:** Test on Windows cmd.exe, not just Windows Terminal.

### Pitfall 4: Model Size vs RAM Estimation
**What goes wrong:** The `size` field from `ollama.list()` is the on-disk compressed (GGUF) size, not the in-memory size.
**Why it happens:** GGUF models decompress when loaded. Memory usage depends on quantization level, context length, and batch size.
**How to avoid:** Use a conservative heuristic: memory ~= 1.1x disk size for Q4 quantization, ~1.2x for higher quantization. Show warning with estimated range, not exact number. Phrase as "may require approximately X GB RAM."
**Warning signs:** Users reporting warnings for models that work fine, or no warnings for models that OOM.

### Pitfall 5: Streaming Response Final Chunk
**What goes wrong:** The final chunk in a streaming response contains the timing/token count metadata. Missing it means no stats.
**Why it happens:** Iterating `for chunk in stream` and breaking early loses the final summary chunk.
**How to avoid:** Always consume the full stream. The existing code handles this by iterating to completion.
**Warning signs:** `total_duration`, `eval_count` etc. are 0 or missing.

### Pitfall 6: `__main__.py` vs `__init__.py` Import Order
**What goes wrong:** Circular imports when `__main__.py` imports from package modules that import from `__init__.py`.
**Why it happens:** When running `python -m llm_benchmark`, Python executes `__main__.py` with the package already partially initialized.
**How to avoid:** Keep `__init__.py` minimal (version string only). Have `__main__.py` import only `cli.main` and call it. All other imports happen inside function bodies or in leaf modules.
**Warning signs:** ImportError on startup that doesn't reproduce when importing modules individually.

## Code Examples

### pyproject.toml
```toml
# Source: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-benchmark"
version = "2.0.0"
description = "Measure token throughput of LLMs running via Ollama"
requires-python = ">= 3.12"
license = "MIT"
dependencies = [
    "ollama>=0.6",
    "pydantic>=2.9",
    "rich>=14.0",
    "tenacity>=9.0",
]

[project.scripts]
llm-benchmark = "llm_benchmark.cli:main"
```

### Correct Throughput Averaging (STAB-04)
```python
def compute_averages(results: list[BenchmarkResult]) -> dict:
    """Correct averaging: total_tokens / total_time, not mean of rates."""
    successful = [r for r in results if r.success]
    if not successful:
        return {}

    total_prompt_tokens = sum(r.prompt_tokens for r in successful)
    total_response_tokens = sum(r.response_tokens for r in successful)
    total_prompt_time = sum(r.prompt_eval_time for r in successful)
    total_response_time = sum(r.response_time for r in successful)

    return {
        "prompt_eval_ts": total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0,
        "response_ts": total_response_tokens / total_response_time if total_response_time > 0 else 0,
        "total_ts": (total_prompt_tokens + total_response_tokens) / (total_prompt_time + total_response_time)
            if (total_prompt_time + total_response_time) > 0 else 0,
    }
```

### Rich Console Output Pattern
```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# System summary (compact one-liner)
console.print(
    f":computer: [bold]{cpu}[/bold] | "
    f":floppy_disk: {ram_gb:.0f} GB RAM | "
    f":tv: {gpu_info} | "
    f":gear: Ollama {ollama_ver}"
)

# Error with guidance
console.print(Panel(
    "[red bold]Cannot connect to Ollama[/red bold]\n\n"
    "Start it with: [cyan]ollama serve[/cyan]",
    title="Pre-flight Check Failed",
    border_style="red"
))
```

### Model Offloading (keep_alive=0)
```python
# Source: existing extended_benchmark.py + ollama SDK docs
import ollama

def unload_model(model_name: str) -> bool:
    """Unload a model from memory. No sudo required."""
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        return True
    except Exception:
        return False
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| requirements.txt + setup.py | pyproject.toml (PEP 621) | 2022-2023 | Single config file, works with uv/pip/hatch |
| pip + venv manual setup | `uv` (fast, handles venv automatically) | 2024 | `uv run` handles everything in one command |
| ANSI escape codes for colors | `rich` library | Mature since 2020 | Terminal capability detection, Windows support, structured output |
| signal.SIGALRM for timeouts | threading.Thread with join(timeout) | Always for cross-platform | SIGALRM is Unix-only; threading works everywhere |
| setup_passwordless_sudo.sh for model offload | ollama.generate(keep_alive=0) | Ollama API addition ~2024 | No sudo, no shell scripts, cross-platform |

**Deprecated/outdated:**
- `setup.py` / `requirements.txt`: Replaced by `pyproject.toml`
- `run.sh` / `run.bat` / `run.ps1`: Replaced by `python run.py` or `uv run python -m llm_benchmark`
- `setup_passwordless_sudo.sh`: Replaced by keep_alive=0 API

## Open Questions

1. **Exact RAM heuristic threshold**
   - What we know: `ollama.list()` returns disk `size` per model. System RAM can be detected via platform calls.
   - What's unclear: The exact ratio of disk-to-memory size varies by quantization. No official formula from Ollama.
   - Recommendation: Use 1.2x disk size as conservative estimate. Warn if model estimated memory > 80% of available RAM. This is Claude's discretion per CONTEXT.md.

2. **Ollama SDK exception hierarchy**
   - What we know: It wraps httpx exceptions. `ConnectionError` for unreachable server, `ResponseError` for API errors.
   - What's unclear: Whether the exact exception types are stable public API or implementation detail.
   - Recommendation: Catch broadly in preflight (any Exception from `ollama.list()`), specifically in benchmarks (`ollama.ResponseError`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (standard for Python projects) |
| Config file | none -- see Wave 0 |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STAB-01 | Package imports without errors on all platforms | unit | `uv run pytest tests/test_package.py::test_imports -x` | -- Wave 0 |
| STAB-02 | Ollama connectivity check shows error when unreachable | unit (mocked) | `uv run pytest tests/test_preflight.py::test_ollama_unreachable -x` | -- Wave 0 |
| STAB-03 | RAM check warns for large models | unit (mocked) | `uv run pytest tests/test_preflight.py::test_ram_warning -x` | -- Wave 0 |
| STAB-04 | Throughput averaging uses total_tokens/total_time | unit | `uv run pytest tests/test_runner.py::test_correct_averaging -x` | -- Wave 0 |
| STAB-05 | Model offloading calls keep_alive=0 | unit (mocked) | `uv run pytest tests/test_runner.py::test_offload_model -x` | -- Wave 0 |
| STAB-06 | Timeout via threading works | unit | `uv run pytest tests/test_runner.py::test_timeout -x` | -- Wave 0 |
| QUAL-01 | Single benchmark module (no duplicate scripts) | smoke | `uv run python -m llm_benchmark --help` | -- Wave 0 |
| QUAL-02 | Package structure valid | unit | `uv run pytest tests/test_package.py::test_structure -x` | -- Wave 0 |
| QUAL-05 | Python version check in pyproject.toml | manual-only | Check `requires-python` in pyproject.toml | N/A |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `pyproject.toml` -- needs `[project.optional-dependencies.test]` with pytest
- [ ] `tests/` directory -- does not exist
- [ ] `tests/conftest.py` -- shared fixtures (mock ollama client, mock system info)
- [ ] `tests/test_package.py` -- import tests, structure validation
- [ ] `tests/test_preflight.py` -- connectivity and RAM check tests
- [ ] `tests/test_runner.py` -- averaging, timeout, offloading tests
- [ ] `tests/test_cli.py` -- subcommand parsing tests

## Sources

### Primary (HIGH confidence)
- PyPI ollama 0.6.1 -- version, dependencies (pydantic>=2.9, httpx>=0.27)
- PyPI pydantic 2.12.5 -- version, requires Python >=3.9
- PyPI rich 14.3.3 -- version, requires Python >=3.8
- PyPI tenacity 9.1.4 -- version, requires Python >=3.10
- ollama-python GitHub `_client.py` -- ConnectionError message, chat/generate/list/ps APIs, keep_alive parameter
- ollama-python GitHub `_types.py` -- ListResponse.Model.size, ProcessResponse.Model.size_vram fields
- packaging.python.org -- pyproject.toml format, build backends, entry points

### Secondary (MEDIUM confidence)
- Rich documentation (readthedocs) -- Console API, terminal detection, markup syntax
- ollama-python README -- streaming patterns, error handling

### Tertiary (LOW confidence)
- RAM-to-disk size ratio (1.1-1.2x) -- heuristic from community experience, no official source

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions verified from PyPI, compatibility confirmed
- Architecture: HIGH -- package layout locked by user; patterns derived from existing working code
- Pitfalls: HIGH -- most identified from reading actual codebase (STAB-04 bug confirmed in source)
- Validation: MEDIUM -- test framework choice is standard but no existing test infrastructure to build on

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable domain, no fast-moving dependencies)
