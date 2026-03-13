# LLM Benchmark

Measure how fast Large Language Models run on **your** hardware. Uses [Ollama](https://ollama.com/) to benchmark token throughput (tokens/second) across models, so you know which one works best for your setup.

```
uv run python -m llm_benchmark
```

No CLI knowledge needed — an interactive menu guides you through everything.

```
LLM Benchmark

Apple M3 Max | 64 GB RAM | Apple M3 Max (integrated GPU) | Ollama 0.17.7

  Download recommended models? (y/N): y

  Small models (any hardware)
    1. llama3.2:1b  (1B) - Fast general-purpose chat
    2. qwen2.5:0.5b (0.5B) - Smallest available model
  Medium models (16 GB+ RAM)
    3. llama3.2:3b  (3B) - Balanced speed and quality
    4. gemma3:4b    (4B) - Google's efficient model

  Enter model numbers to download (e.g. 1,3,5) or 'all': 1,3

    1. Quick test (~30 seconds)
    2. Standard benchmark
    3. Full benchmark
    4. Custom

  Select mode [1-4]:
```

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    llm-benchmark                        │
│                                                         │
│  1. Preflight                                           │
│     ├─ Ollama installed? (offers auto-install if not)   │
│     ├─ Ollama running?                                  │
│     ├─ Models available? (recommends by your RAM)       │
│     └─ RAM check (warns if model may not fit)           │
│                                                         │
│  2. Benchmark                                           │
│     ├─ Send prompts to each model via Ollama API        │
│     ├─ Measure: prompt eval, response gen, total time   │
│     ├─ Multiple runs per prompt for stable averages     │
│     └─ Unload model between tests (fair comparison)     │
│                                                         │
│  3. Results                                             │
│     ├─ Terminal: ranked bar chart + recommendation      │
│     └─ Files: JSON + CSV + Markdown (in results/)      │
│                                                         │
│  Model Rankings                                         │
│                                                         │
│    qwen3:1.7b   ██████████████████████████████  212 t/s │
│    llama3.2:3b  ████████████████████            142 t/s │
│    gemma3:4b    ███████████████                 108 t/s │
│    mistral:7b   █████████████                    91 t/s │
│                                                         │
│    Best for your setup: qwen3:1.7b (212 t/s)           │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install prerequisites

**Python 3.12+** and **uv** (recommended):

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Ollama** — the tool will offer to install it automatically on first run. Or install manually:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (PowerShell)
irm https://ollama.com/install.ps1 | iex
```

### 2. Clone and install

```bash
git clone https://github.com/vedmichv/llm-benchmark.git
cd llm-benchmark
uv sync
```

### 3. Download recommended models

```bash
uv run python -m llm_benchmark --recommend
```

The tool detects your RAM (+ GPU VRAM on discrete GPUs), shows models that fit your hardware grouped by tier, and lets you pick which ones to download. Already installed models are marked `[installed]`.

### 4. Run benchmark

```bash
uv run python -m llm_benchmark
```

The interactive menu will:
- Check Ollama is installed and running
- Let you pick a benchmark mode (Quick test / Standard / Full / Custom)
- Show results with a ranked bar chart and recommendation

### 5. Check your results

After a benchmark completes, results are saved to `results/`:

```
results/
  benchmark_20260313_140317.json   # Machine-readable data
  benchmark_20260313_140317.csv    # For spreadsheets
  benchmark_20260313_140317.md     # Shareable report
```

The Markdown report includes system info, rankings with bar chart, and a recommendation — paste it into GitHub, Discord, or Slack.

## Benchmark Modes

### Interactive menu (no-args)

```bash
uv run python -m llm_benchmark
```

| Mode | Prompts | Runs | Time estimate | Best for |
|------|---------|------|---------------|----------|
| Quick test | 1 short | 1 | ~30 seconds | Verify everything works |
| Standard | 5 medium | 2 | ~10-20 min | Normal comparison |
| Full | 11 large | 3 | ~30-60 min | Thorough analysis |
| Custom | You pick | You pick | Varies | Specific needs |

### CLI flags (for power users)

```bash
# Standard benchmark (same as menu option 2)
uv run python -m llm_benchmark run

# Specific prompt set and runs
uv run python -m llm_benchmark run --prompt-set large --runs-per-prompt 3

# Skip certain models
uv run python -m llm_benchmark run --skip-models llama3.2:1b

# Custom prompts
uv run python -m llm_benchmark run --prompts "Explain Docker" "What is Kubernetes?"

# Verbose mode (stream responses)
uv run python -m llm_benchmark run --verbose
```

### Advanced modes

**Concurrent benchmarking** — test throughput under parallel load:

```bash
# Auto-detect optimal concurrency based on hardware
uv run python -m llm_benchmark run --concurrent

# Or specify worker count
uv run python -m llm_benchmark run --concurrent 4
```

**Parameter sweep** — find the best `num_ctx` / `num_gpu` config per model:

```bash
uv run python -m llm_benchmark run --sweep
```

### Other commands

```bash
# Compare two benchmark runs
uv run python -m llm_benchmark compare results/run1.json results/run2.json

# Show system information
uv run python -m llm_benchmark info

# Analyze and rank existing results
uv run python -m llm_benchmark analyze results/benchmark.json
```

## Understanding Results

### Key metrics

| Metric | What it means |
|--------|--------------|
| **Prompt Eval (t/s)** | Speed processing your input (higher = better) |
| **Response (t/s)** | Speed generating output — **the most important metric** |
| **Total (t/s)** | Combined throughput |

### Performance tiers

| Response t/s | Rating | Experience |
|-------------|--------|------------|
| 150+ | Excellent | Near-instant streaming |
| 100-150 | Good | Comfortable reading speed |
| 50-100 | Moderate | Noticeable delay |
| < 50 | Slow | Significant wait |

### Example Markdown report

```markdown
# LLM Benchmark Results

**Generated:** 2026-03-13 14:03:17 | **Models:** 4 | **Mode:** Standard

**System:** Apple M3 Max, 64.0 GB RAM, Apple M3 Max (integrated GPU), Darwin 25.3.0

## Rankings

  qwen3:1.7b   ██████████████████████████████   212.8 t/s
  llama3.2:3b  ████████████████████             142.1 t/s
  gemma3:4b    ███████████████                  108.3 t/s
  mistral:7b   █████████████                     91.7 t/s

  Best for your setup: qwen3:1.7b (212.8 t/s) -- fastest response generation

## Summary

| Model | Prompt Eval (t/s) | Response (t/s) | Total (t/s) |
|-------|-------------------|----------------|-------------|
| qwen3:1.7b | 7129.93 | 212.81 | 227.84 |
| llama3.2:3b | 5841.22 | 142.10 | 155.32 |
| ...
```

## Model Recommendations by RAM

The tool auto-recommends models based on your hardware. Here's the full tier list:

| RAM | Models you can comfortably run |
|-----|-------------------------------|
| Any | `llama3.2:1b`, `qwen2.5:0.5b`, `phi4-mini` (3.8B) |
| 16 GB+ | `llama3.2:3b`, `qwen2.5:7b`, `gemma3:4b`, `phi4` (14B) |
| 36 GB+ | `llama3.1:8b`, `qwen2.5:14b`, `gemma3:12b`, `phi4:14b` |
| 64 GB+ | `llama3.3:70b`, `qwen2.5:32b` |

Download models manually:

```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull gemma3:4b
# ... etc
```

Or let the interactive menu recommend and download them for you.

## Project Structure

```
llm_benchmark/
  cli.py          # CLI with run/compare/info/analyze subcommands
  menu.py         # Interactive mode selection menu
  display.py      # Terminal bar chart rendering
  recommend.py    # RAM-based model recommendations
  runner.py       # Benchmark execution with retry and timeout
  models.py       # Pydantic data models
  exporters.py    # JSON, CSV, Markdown report writers
  preflight.py    # Ollama install/connectivity checks
  system.py       # Cross-platform hardware detection
  concurrent.py   # Parallel benchmark mode
  sweep.py        # Parameter sweep mode
  compare.py      # Results comparison
  analyze.py      # Results analysis and ranking
  prompts.py      # Prompt sets (small/medium/large)
  config.py       # Rich Console singleton, constants
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest tests/ -x -q

# Run tests with coverage
uv run pytest tests/ -x -q --cov=llm_benchmark --cov-fail-under=60

# Lint
uv run ruff check llm_benchmark/ tests/
```

CI runs automatically on push to main and PRs (lint + compile check + tests).

## Troubleshooting

**Ollama not running:**
```bash
ollama serve    # macOS — run in a separate terminal
# Linux: sudo systemctl start ollama
```

**Model too slow / system freezing:**
Your model may be too large for available RAM. Try a smaller model:
```bash
ollama pull llama3.2:1b
uv run python -m llm_benchmark   # Quick test with smallest model
```

**Timeout errors:**
```bash
uv run python -m llm_benchmark run --timeout 600   # Increase to 10 min
```

## License

MIT — see [LICENSE](LICENSE) for details.
