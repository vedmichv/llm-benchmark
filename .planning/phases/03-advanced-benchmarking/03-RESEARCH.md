# Phase 3: Advanced Benchmarking - Research

**Researched:** 2026-03-13
**Domain:** Concurrent HTTP requests, parameter sweeping, results analysis (Python asyncio + Ollama API)
**Confidence:** HIGH

## Summary

Phase 3 adds three major capabilities: concurrent benchmarking (asyncio parallel requests to Ollama), parameter sweep (auto-explore num_ctx/num_gpu combinations), and results analysis (new `analyze` subcommand + enhanced `compare`). The existing codebase uses the synchronous `ollama` Python SDK; the same package provides `AsyncClient` with identical method signatures, making the async migration straightforward. httpx 0.28.1 is already installed as an ollama dependency.

The Ollama SDK's `Options` type natively supports `num_ctx` and `num_gpu` as typed fields, so parameter sweep configuration passes directly through the existing `chat()` call via the `options` parameter. Model layer count for computing num_gpu values is available through `ollama.show(model).modelinfo` -- the response contains architecture-specific keys like `*.block_count` that give the total layer count.

**Primary recommendation:** Use `ollama.AsyncClient` for concurrent mode (not raw httpx), add a new `concurrent.py` module for async orchestration, a new `sweep.py` module for parameter sweep logic, and a new `analyze.py` module for the analyze subcommand. Extend existing models/exporters with a `mode` discriminator field.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use asyncio + httpx AsyncClient for true parallel HTTP requests (not threads)
- Single warmup run per model before all concurrent requests
- Failed concurrent requests marked as failed (success=False), other in-flight requests continue
- --concurrent N flag: when N is omitted, auto-detect based on available RAM/GPU
- All N workers send the SAME prompt simultaneously
- Aggregate throughput: wall-time aggregate (total_tokens/wall_time) PLUS per-request average
- Combinable with --runs-per-prompt: each "run" fires N concurrent requests
- Sweep num_ctx (512, 1024, 2048, 4096) and num_gpu (0, 50% layers, 100% layers)
- Auto-detect model's total layer count from Ollama
- Report as ranked Rich table + highlighted "Best config" recommendation
- Single short prompt per configuration for sweep
- Sweep per-model (not global)
- Config counter display during sweep
- Sweep results saved to sweep_YYYYMMDD_HHMMSS.{json,csv,md}
- New 'analyze' subcommand: analyze results/file.json --sort response_ts --top 3
- Sortable metrics: response_ts, total_ts, prompt_eval_ts, load_time
- Default sort: descending, --asc flag to reverse
- Single file only for analyze
- Model averages by default, --detail flag for per-run breakdown
- --top N filters by models, not individual runs
- Terminal output only from analyze (no file export)
- Compare: add Unicode arrows with color, "winner" column
- Compare works with concurrent and sweep result formats
- Concurrent terminal output: per-request live results then aggregate summary
- Extended schema with 'mode' field ('standard', 'concurrent', 'sweep')
- Sweep Markdown: ranked config table with bold/green best row

### Claude's Discretion
- httpx vs ollama async client specifics
- Exact auto-detect heuristic for --concurrent N default
- num_ctx values to sweep (suggested 512/1024/2048/4096 but can adjust)
- Sweep prompt text choice
- analyze Rich table layout and column formatting
- Compare arrow placement and color scheme

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BENCH-03 | User can run concurrent benchmark with --concurrent N | AsyncClient.chat() supports parallel requests via asyncio.gather(); Options type supports all needed params |
| BENCH-04 | Concurrent mode reports aggregate throughput and per-request average | Wall-clock timing via asyncio event loop; per-request results from individual chat() responses |
| BENCH-05 | User can run parameter sweep with --sweep | Options(num_ctx=N, num_gpu=N) passes directly to chat(); ollama.show() provides layer count |
| BENCH-06 | Sweep reports best configuration with throughput numbers | Rich Table with sorting and style highlighting; sweep results as separate export files |
| ANLZ-01 | User can sort benchmark results by any metric | JSON result files already contain all metrics; sorted() with key function |
| ANLZ-02 | User can filter top-N results | Slice sorted list by model averages |
| ANLZ-03 | User can compare results side-by-side | Extend existing compare.py with arrows, winner column, mode-aware loading |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ollama (AsyncClient) | >=0.6 (installed) | Async parallel requests to Ollama | Already a dependency; AsyncClient mirrors sync API exactly |
| asyncio | stdlib | Event loop for concurrent execution | Standard Python async; no additional dependency |
| rich | >=14.0 (installed) | Tables, live display, styled terminal output | Already used throughout codebase |
| pydantic | >=2.9 (installed) | Data models for sweep/concurrent results | Already used for all data models |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | 0.28.1 (installed via ollama) | Already available if needed for raw HTTP | Not needed -- AsyncClient wraps httpx internally |
| time (stdlib) | - | Wall-clock measurement via time.perf_counter() | Measuring total wall time for concurrent batch |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ollama.AsyncClient | Raw httpx.AsyncClient | More control but must reimplement Ollama protocol; unnecessary |
| asyncio.gather | ThreadPoolExecutor | Threads work but asyncio is cleaner for I/O-bound HTTP; user locked asyncio |

**Installation:**
```bash
# No new dependencies needed -- all libraries already installed
uv sync
```

## Architecture Patterns

### Recommended New Module Structure
```
llm_benchmark/
  concurrent.py      # Async concurrent benchmarking (BENCH-03, BENCH-04)
  sweep.py           # Parameter sweep logic (BENCH-05, BENCH-06)
  analyze.py         # Analyze subcommand (ANLZ-01, ANLZ-02)
  # Modified existing:
  cli.py             # Add --concurrent, --sweep flags + analyze subcommand
  models.py          # Add ConcurrentBatchResult, SweepResult models
  exporters.py       # Add mode field, sweep export functions
  compare.py         # Add arrows, winner column, mode-aware loading
  config.py          # Add DEFAULT_CONCURRENT, SWEEP_NUM_CTX, SWEEP_PROMPT constants
```

### Pattern 1: AsyncClient for Concurrent Requests
**What:** Use ollama.AsyncClient within asyncio.run() to fire N parallel chat() calls
**When to use:** --concurrent N flag is provided
**Example:**
```python
# Verified: ollama.AsyncClient has identical API to sync Client
import asyncio
import time
from ollama import AsyncClient

async def run_concurrent_batch(
    model_name: str, prompt: str, n: int, options: dict | None = None
) -> tuple[list[dict], float]:
    """Fire N concurrent requests and measure wall time."""
    client = AsyncClient()
    start = time.perf_counter()

    async def single_request(request_id: int):
        try:
            response = await client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
            return (request_id, response, None)
        except Exception as exc:
            return (request_id, None, exc)

    tasks = [single_request(i) for i in range(n)]
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start
    return results, wall_time

# Called from sync code:
# results, wall_time = asyncio.run(run_concurrent_batch(...))
```

### Pattern 2: Parameter Sweep via Options
**What:** Pass num_ctx and num_gpu through Ollama Options to test configurations
**When to use:** --sweep flag is provided
**Example:**
```python
# Verified: Options accepts num_ctx: int and num_gpu: int
from ollama import Options

# Get layer count from model info
info = ollama.show(model_name)
# modelinfo contains keys like "llama.block_count" -> total layers
block_count = None
if info.modelinfo:
    for key, value in info.modelinfo.items():
        if "block_count" in key:
            block_count = int(value)
            break

# Build sweep configs
num_ctx_values = [512, 1024, 2048, 4096]
num_gpu_values = [0]  # CPU only
if block_count:
    num_gpu_values.append(block_count // 2)  # 50% layers
    num_gpu_values.append(block_count)        # 100% layers

# Run with specific config
response = ollama.chat(
    model=model_name,
    messages=[{"role": "user", "content": sweep_prompt}],
    options=Options(num_ctx=2048, num_gpu=16),
)
```

### Pattern 3: Analyze Subcommand with Sorted Rich Table
**What:** Load JSON results, sort by metric, display top-N in Rich table
**When to use:** `llm_benchmark analyze results/file.json --sort response_ts --top 3`
**Example:**
```python
from rich.table import Table

def analyze_results(filepath, sort_by="response_ts", top_n=None, ascending=False, detail=False):
    data = json.loads(Path(filepath).read_text())
    models = data.get("models", [])

    # Sort by average metric
    models.sort(
        key=lambda m: m["averages"].get(sort_by, 0),
        reverse=not ascending,
    )
    if top_n:
        models = models[:top_n]

    table = Table(title=f"Results: {Path(filepath).name}")
    # ... build table columns and rows
```

### Pattern 4: Mode-Aware Export Schema
**What:** Add 'mode' discriminator to exported JSON for format identification
**Example:**
```python
# Standard mode (existing)
{"generated": "...", "mode": "standard", "system_info": {...}, "models": [...]}

# Concurrent mode
{"generated": "...", "mode": "concurrent", "concurrent_workers": 4,
 "system_info": {...}, "models": [
   {"model": "...", "averages": {...}, "wall_time_s": 12.3,
    "aggregate_throughput_ts": 128.5, "runs": [...]}
 ]}

# Sweep mode
{"generated": "...", "mode": "sweep", "system_info": {...}, "sweeps": [
   {"model": "...", "best_config": {"num_ctx": 2048, "num_gpu": 16},
    "configs": [
      {"num_ctx": 512, "num_gpu": 0, "response_ts": 25.3, ...},
    ]}
 ]}
```

### Anti-Patterns to Avoid
- **Running asyncio.run() inside an existing event loop:** The CLI is sync, so asyncio.run() is safe at the top level. Never nest event loops.
- **Using asyncio.gather(return_exceptions=True) without checking:** Each result could be an exception -- always check type before processing.
- **Averaging rates instead of total_tokens/total_time:** The existing STAB-04 discipline must extend to concurrent aggregate metrics.
- **Blocking the event loop with synchronous ollama calls:** Always use AsyncClient inside async functions, never mix sync ollama.chat with asyncio.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async HTTP to Ollama | Raw httpx async requests | ollama.AsyncClient | Handles protocol, streaming, error types automatically |
| Parallel execution | threading.Thread pool | asyncio.gather() | Cleaner for I/O-bound work, user decision locked this |
| Model layer detection | Parse GGUF files | ollama.show() modelinfo | API provides block_count directly |
| Terminal tables | Manual string formatting | rich.Table | Already used everywhere in codebase |
| JSON schema validation | Manual dict checking | Pydantic models | Already the pattern; extend with new result types |
| Wall-clock timing | datetime math | time.perf_counter() | Monotonic, high-resolution, no timezone issues |

**Key insight:** The ollama SDK already provides everything needed for both async requests and model introspection. No new HTTP client code is needed.

## Common Pitfalls

### Pitfall 1: Ollama Queues Concurrent Requests
**What goes wrong:** Ollama may serialize concurrent requests internally (single-model inference is often sequential on GPU). Students might see N requests take N times as long as 1, with no actual throughput gain.
**Why it happens:** GPU inference is inherently sequential unless the model supports batched inference, which Ollama's default configuration may not enable.
**How to avoid:** This is expected behavior for many setups. Document it clearly in output: "Note: Ollama may queue requests. Aggregate throughput shows total work done per wall-clock second." The value is still measuring server-side queuing performance.
**Warning signs:** wall_time approximately equals N * single_request_time.

### Pitfall 2: num_gpu Mismatch with Hardware
**What goes wrong:** Setting num_gpu higher than available GPU layers (or on CPU-only machines) causes errors or silent fallback.
**Why it happens:** Not all machines have GPUs; Apple Silicon unified memory behaves differently from discrete GPUs.
**How to avoid:** Detect GPU availability from system.py before generating sweep configs. If no GPU detected, skip num_gpu variations (only test num_ctx). Wrap each sweep config in try/except.
**Warning signs:** Sweep errors on first num_gpu > 0 config.

### Pitfall 3: asyncio.run() Compatibility
**What goes wrong:** asyncio.run() fails if called from within an existing event loop (e.g., Jupyter notebooks, some test frameworks).
**Why it happens:** Python's asyncio restriction on nested event loops.
**How to avoid:** Always call asyncio.run() from synchronous CLI handler code only. For testing, use pytest-asyncio or manually manage the event loop.
**Warning signs:** "RuntimeError: This event loop is already running."

### Pitfall 4: Sweep Takes Too Long
**What goes wrong:** 4 num_ctx * 3 num_gpu = 12 configs per model, times warmup and inference per config. With 5 models, that's 60 benchmark runs.
**Why it happens:** Each config change may require model reload.
**How to avoid:** Use a single short prompt (fixed, not from prompt sets). Show progress counter. Consider skipping warmup for sweep (first config's run acts as warmup for subsequent configs with same model loaded). Allow --sweep-models to limit which models to sweep.
**Warning signs:** Sweep taking >10 minutes per model.

### Pitfall 5: Mode Field Breaks Backward Compatibility
**What goes wrong:** Adding 'mode' field to JSON output could break existing compare logic that expects the old format.
**Why it happens:** compare.py reads JSON directly without schema validation.
**How to avoid:** Default mode to "standard" when field is missing. compare.py should handle missing mode gracefully (treat as "standard").
**Warning signs:** compare crashes on old result files.

## Code Examples

### Verified: AsyncClient.chat() with Options
```python
# Source: ollama SDK inspection (installed version >=0.6)
# AsyncClient.chat signature matches sync Client.chat exactly
# Options supports: num_ctx, num_gpu, num_batch, num_thread, etc.

from ollama import AsyncClient, Options

async def benchmark_with_config(model: str, prompt: str, num_ctx: int, num_gpu: int):
    client = AsyncClient()
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=Options(num_ctx=num_ctx, num_gpu=num_gpu),
    )
    return response  # ChatResponse with same fields as sync version
```

### Verified: Getting Model Layer Count
```python
# Source: ollama SDK inspection -- ShowResponse.modelinfo is Mapping[str, Any]
# The modelinfo dict contains architecture-specific keys

import ollama

def get_model_layers(model_name: str) -> int | None:
    """Get total block/layer count from model metadata."""
    info = ollama.show(model_name)
    if not info.modelinfo:
        return None
    for key, value in info.modelinfo.items():
        if "block_count" in key:
            return int(value)
    return None
```

### Verified: Concurrent Gather with Error Handling
```python
import asyncio
import time

async def run_concurrent(model: str, prompt: str, n: int):
    client = AsyncClient()
    errors = []
    results = []

    async def worker(idx: int):
        try:
            resp = await client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return ("ok", idx, resp)
        except Exception as e:
            return ("error", idx, e)

    start = time.perf_counter()
    outcomes = await asyncio.gather(*[worker(i) for i in range(n)])
    wall_time = time.perf_counter() - start

    for status, idx, data in outcomes:
        if status == "ok":
            results.append(data)
        else:
            errors.append((idx, data))

    return results, errors, wall_time
```

### Auto-Detect Concurrent Workers Heuristic
```python
# Claude's discretion: heuristic for default --concurrent N
def auto_detect_concurrency(ram_gb: float, gpu_vram_gb: float | None) -> int:
    """Conservative default: 2 for low-RAM, 4 for normal, 8 for high-end."""
    if gpu_vram_gb and gpu_vram_gb >= 16:
        return 8
    if ram_gb >= 32:
        return 4
    return 2
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ollama Python SDK sync only | AsyncClient available | ollama >=0.4 | Native async support, no httpx needed directly |
| Manual HTTP to Ollama | SDK handles protocol | ollama >=0.2 | Simpler code, proper error types |
| Options as raw dict | Typed Options class | ollama >=0.3 | IDE completion, validation for num_ctx/num_gpu |

**Deprecated/outdated:**
- Raw httpx calls to Ollama: The ollama SDK wraps httpx internally; using AsyncClient is simpler and maintains compatibility.
- ollama.generate() for chat: Use ollama.chat() (already the pattern in this codebase).

## Open Questions

1. **Ollama Concurrent Request Behavior**
   - What we know: Ollama accepts multiple concurrent HTTP requests but may serialize GPU inference.
   - What's unclear: Whether Ollama implements request queuing, parallel batching, or rejection under load. This varies by Ollama version and model.
   - Recommendation: Test empirically during implementation. Document observed behavior. The wall-time aggregate metric is valid regardless.

2. **Model Reload Between Sweep Configs**
   - What we know: Changing num_ctx or num_gpu may require model reload (Ollama decides).
   - What's unclear: Whether Ollama automatically reloads when options change, or if we need to explicitly unload/reload.
   - Recommendation: Call unload_model() between config changes to ensure clean state. Accept the time cost -- correctness over speed.

3. **block_count Key Name Varies by Architecture**
   - What we know: modelinfo contains architecture-prefixed keys (e.g., "llama.block_count", "qwen2.block_count").
   - What's unclear: Whether ALL model architectures use "*block_count" or if some use different naming.
   - Recommendation: Search for any key containing "block_count" (the pattern shown above). Fall back gracefully if not found (skip num_gpu sweep, only test num_ctx).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | pyproject.toml (implicit) |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-03 | Concurrent benchmark runs N parallel requests | unit (mock AsyncClient) | `uv run pytest tests/test_concurrent.py -x` | No -- Wave 0 |
| BENCH-04 | Aggregate throughput = total_tokens/wall_time | unit | `uv run pytest tests/test_concurrent.py::test_aggregate_throughput -x` | No -- Wave 0 |
| BENCH-05 | Sweep explores num_ctx/num_gpu combinations | unit (mock ollama.show + chat) | `uv run pytest tests/test_sweep.py -x` | No -- Wave 0 |
| BENCH-06 | Sweep reports best config | unit | `uv run pytest tests/test_sweep.py::test_best_config_selection -x` | No -- Wave 0 |
| ANLZ-01 | Sort results by metric | unit | `uv run pytest tests/test_analyze.py::test_sort_by_metric -x` | No -- Wave 0 |
| ANLZ-02 | Filter top-N models | unit | `uv run pytest tests/test_analyze.py::test_top_n_filter -x` | No -- Wave 0 |
| ANLZ-03 | Compare side-by-side with arrows | unit | `uv run pytest tests/test_cli.py::test_compare_enhanced -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_concurrent.py` -- covers BENCH-03, BENCH-04
- [ ] `tests/test_sweep.py` -- covers BENCH-05, BENCH-06
- [ ] `tests/test_analyze.py` -- covers ANLZ-01, ANLZ-02
- [ ] pytest-asyncio dependency for testing async code: `uv add --dev pytest-asyncio`
- [ ] Fixtures in conftest.py for: mock AsyncClient, sample sweep modelinfo, sample JSON result files

## Sources

### Primary (HIGH confidence)
- ollama Python SDK (installed >=0.6) -- AsyncClient, Options, ShowResponse inspected directly via Python
- Project source code -- runner.py, cli.py, models.py, exporters.py, compare.py, config.py read in full
- pyproject.toml -- current dependencies and versions verified

### Secondary (MEDIUM confidence)
- Ollama API behavior under concurrent load -- based on SDK structure and general Ollama documentation patterns
- modelinfo block_count key naming -- inferred from common Ollama model metadata patterns

### Tertiary (LOW confidence)
- Exact Ollama queuing behavior under concurrent requests -- needs empirical testing (flagged as Open Question 1)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and inspected
- Architecture: HIGH -- patterns follow existing codebase conventions, AsyncClient API verified
- Pitfalls: MEDIUM -- Ollama concurrent behavior needs empirical validation during implementation

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable libraries, local project)
