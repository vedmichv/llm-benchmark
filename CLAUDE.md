# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Benchmark is a Python tool for measuring token throughput (tokens per second) of Large Language Models running via Ollama. It benchmarks models by sending prompts and collecting timing/performance metrics.

## Prerequisites

- Python 3.6+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- Dependencies: `ollama` and `pydantic` (see requirements.txt)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama server is running
ollama serve
```

## Running Benchmarks

```bash
# Basic usage with default prompts
python benchmark.py

# Verbose mode (streams responses)
python benchmark.py --verbose

# Custom prompts
python benchmark.py --prompts "Why is the sky blue?" "Write a report on Nvidia"

# Skip specific models
python benchmark.py --skip-models llama2:latest model1

# Combined example
python benchmark.py --verbose --skip-models llama2:latest --prompts "Custom prompt 1" "Custom prompt 2"
```

## Architecture

This is a single-file Python script (`benchmark.py`) with a straightforward architecture:

### Core Components

1. **Pydantic Models** (lines 14-40):
   - `Message`: Represents a chat message (role + content)
   - `OllamaResponse`: Validates and structures Ollama API responses with timing/token metrics
   - Includes validation for prompt caching scenarios where prompt_eval_count may be -1

2. **Benchmark Flow**:
   - `get_benchmark_models()`: Queries Ollama for available models, filters by skip list
   - `run_benchmark()`: Executes single prompt against a model (streaming or non-streaming)
   - `inference_stats()`: Calculates and displays tokens/second metrics (prompt eval, response, total)
   - `average_stats()`: Aggregates metrics across multiple prompts for a model
   - `main()`: Orchestrates full benchmark run across all models and prompts

3. **Timing Calculations**:
   - Converts nanoseconds (from Ollama) to seconds
   - Calculates three throughput metrics:
     - Prompt evaluation speed (t/s)
     - Response generation speed (t/s)
     - Total throughput (t/s)

### Data Flow

```
CLI Args → Get Models from Ollama → For each model:
  For each prompt:
    Run Benchmark → Parse Response → Calculate Stats
  → Average Stats across prompts
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

- The script uses `ollama.chat()` with optional streaming
- Responses are validated through Pydantic models for type safety
- Prompt caching may cause `prompt_eval_count=-1`; this is handled with a warning
- All models from `ollama list` are benchmarked unless explicitly skipped
