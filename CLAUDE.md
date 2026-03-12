# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Benchmark is a Python tool for measuring token throughput (tokens per second) of Large Language Models running via Ollama. It benchmarks models by sending prompts and collecting timing/performance metrics.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- [uv](https://docs.astral.sh/uv/) recommended for dependency management

## Setup

```bash
# Install dependencies (recommended)
uv sync

# Or with pip
pip install -e .

# Ensure Ollama server is running
ollama serve
```

## Running Benchmarks

```bash
# Basic usage with default prompts (medium set)
python -m llm_benchmark run

# Verbose mode (streams responses)
python -m llm_benchmark run --verbose

# Custom prompts
python -m llm_benchmark run --prompts "Why is the sky blue?" "Write a report on Nvidia"

# Skip specific models
python -m llm_benchmark run --skip-models llama2:latest model1

# Use small prompt set with 3 runs per prompt
python -m llm_benchmark run --prompt-set small --runs-per-prompt 3

# Compare results
python -m llm_benchmark compare results/run1.json results/run2.json

# Show system information
python -m llm_benchmark info

# Convenience launcher (equivalent to python -m llm_benchmark)
python run.py run
```

## Running Tests

```bash
# Quick test run
uv run pytest tests/ -x -q

# Verbose test output
uv run pytest tests/ -v
```

## Architecture

The project is structured as a Python package (`llm_benchmark/`) with subcommand-based CLI.

### Package Layout

```
llm_benchmark/
  __init__.py        # Package version (__version__ = "2.0.0")
  __main__.py        # Entry point: python -m llm_benchmark (exception handler)
  cli.py             # Argparse CLI with run/compare/info subcommands
  config.py          # Rich Console singleton, debug flag, constants
  models.py          # Pydantic data models (Message, OllamaResponse, BenchmarkResult, etc.)
  runner.py          # Benchmark execution, threading timeout, model offloading
  system.py          # Cross-platform hardware/system info (CPU, RAM, GPU)
  exporters.py       # JSON, CSV, Markdown result writers
  preflight.py       # Ollama connectivity check, model availability, RAM warnings
  prompts.py         # Prompt sets (small/medium/large)
  compare.py         # Results comparison with rich tables
tests/
  conftest.py        # Shared test fixtures
  test_models.py     # Pydantic model tests
  test_runner.py     # Runner and averaging tests
  test_system.py     # System info tests
  test_preflight.py  # Preflight check tests
  test_cli.py        # CLI parsing tests
  test_package.py    # Import and structure validation
pyproject.toml       # Package metadata, dependencies, build config
run.py               # Thin convenience wrapper
```

### Core Components

1. **CLI (`cli.py`)**: Argparse with subcommands: `run`, `compare`, `info`. Global `--debug` flag.

2. **Pre-flight Checks (`preflight.py`)**: Runs before every benchmark:
   - Ollama connectivity (blocking) -- shows platform-specific start instructions on failure
   - Model availability (blocking) -- suggests `ollama pull llama3.2:1b` if none
   - RAM warning (advisory) -- warns if model may exceed 80% RAM, but continues

3. **Runner (`runner.py`)**: Benchmark execution with threading-based timeouts (cross-platform), model offloading via keep_alive=0, and correct total_tokens/total_time averaging.

4. **System Info (`system.py`)**: Cross-platform CPU, RAM, GPU, OS detection without psutil.

5. **Exporters (`exporters.py`)**: JSON, CSV, Markdown output with timestamped filenames.

6. **Data Models (`models.py`)**: Pydantic models for OllamaResponse, BenchmarkResult, ModelSummary, SystemInfo. Handles prompt caching silently.

### Data Flow

```
CLI Args -> Pre-flight Checks -> System Summary -> For each model:
  For each prompt (x runs_per_prompt):
    Run Benchmark -> Parse Response -> Calculate Stats
  -> Average Stats (total_tokens / total_time)
  -> Unload Model
-> Export Results (JSON + CSV + Markdown)
```

### Key Timing Fields from Ollama

The tool processes these timing fields from Ollama responses:
- `total_duration`: Full request duration
- `load_duration`: Model loading time
- `prompt_eval_duration`: Time to process input prompt
- `prompt_eval_count`: Input tokens processed
- `eval_duration`: Time to generate response
- `eval_count`: Output tokens generated

## Development Notes

- Use `from llm_benchmark.config import get_console` for all terminal output (Rich Console singleton)
- Never use `print()` directly -- use the console singleton
- Pydantic validators must not have print side effects
- Throughput averaging: sum tokens / sum time (not mean of rates)
- Timeouts use threading (no signal.SIGALRM) for cross-platform support
- Model offloading uses `ollama.generate(keep_alive=0)` -- no sudo needed
