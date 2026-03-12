# Technology Stack

**Project:** LLM Benchmark v2
**Researched:** 2026-03-12
**Mode:** Ecosystem research for concurrent benchmarking, parameter sweep, retry, and enhanced analysis

## Python Version Decision

**Recommendation: Python >= 3.10** (bump from stated 3.6+)

The project already requires Python 3.9+ in practice (Pydantic 2.x demands >=3.9, Ollama SDK demands >=3.8). Modern libraries like tenacity 9.x, tabulate 0.10, and click 8.3 all require >=3.10. Students on a current DevOps course will have Python 3.10+ available. Pinning to 3.6 limits the library choices for zero practical benefit.

**Confidence: HIGH** -- verified via PyPI requirements for all packages.

## Recommended Stack

### Core (Keep)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `ollama` | >=0.6.1 | LLM API communication | Already in use. Provides `AsyncClient` for concurrent benchmarking -- this is the key unlock for parallel requests. No need for httpx directly. |
| `pydantic` | >=2.12 | Response validation, config models | Already in use. Extend for benchmark config schemas (parameter sweep definitions, export settings). |

**Confidence: HIGH** -- versions verified on PyPI (2025-11-13 and 2025-11-26 respectively).

### New: Terminal UI and Progress

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `rich` | >=14.3 | Progress bars, tables, live display, colored output | The standard for Python terminal UI in 2025. Replaces manual print formatting. Provides `Progress` for multi-bar benchmark tracking, `Table` for ranked results, `Console` for styled output. One dependency covers progress bars, tables, and terminal bar charts. Cross-platform (Windows/macOS/Linux). |

**Why rich over alternatives:**
- `tqdm`: Only does progress bars. No tables, no styled output, no live display.
- `tabulate` + `tqdm` combo: Two dependencies to get what rich does alone, and tabulate 0.10 also requires 3.10+.
- `textual` (by the same author): Full TUI framework -- overkill. PROJECT.md explicitly says "dialog TUI" is out of scope. rich gives styled output without a full TUI.

**Confidence: HIGH** -- rich 14.3.3 verified on PyPI (2026-02-19). Cross-platform support confirmed.

### New: Concurrency

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `asyncio` | stdlib | Concurrent benchmark orchestration | Standard library. The Ollama SDK's `AsyncClient` is built on httpx async, which uses asyncio. Use `asyncio.gather()` or `asyncio.Semaphore` to run N concurrent requests against a model. No additional HTTP library needed. |

**Why asyncio over threading/multiprocessing:**
- Ollama SDK already provides `AsyncClient` -- using threading to wrap synchronous calls adds complexity for no gain.
- `concurrent.futures.ThreadPoolExecutor`: Viable but clunkier. You'd wrap synchronous `ollama.chat()` calls. AsyncClient is purpose-built for this.
- `multiprocessing`: Wrong tool. We're IO-bound (waiting on Ollama HTTP responses), not CPU-bound. Multiprocessing adds process overhead and complicates shared state.
- `httpx` directly: Unnecessary. Ollama SDK already wraps httpx. Going around the SDK means reimplementing response parsing.

**Confidence: HIGH** -- AsyncClient documented in Ollama SDK README with working examples.

### New: Retry Logic

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `tenacity` | >=9.1 | Retry with exponential backoff | The standard Python retry library. Decorator-based (`@retry`), configurable wait strategies (exponential, fixed, random jitter), stop conditions (max attempts, max time), and retry-on-exception filtering. Wraps both sync and async functions. Far better than hand-rolling retry loops. |

**Why tenacity over alternatives:**
- Hand-rolled retry loop: Error-prone, doesn't handle edge cases (jitter, async, logging retries). Students would need to maintain it.
- `backoff` library: Less popular, fewer features, last significant update older. tenacity is the community standard.
- `urllib3.util.retry`: HTTP-only, doesn't work with Ollama SDK's higher-level API.

**Note:** tenacity 9.x requires Python >=3.10. This is one reason to bump the minimum Python version.

**Confidence: HIGH** -- tenacity 9.1.4 verified on PyPI (2026-02-07).

### New: Data Analysis (optional, for enhanced results)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `statistics` | stdlib | Mean, median, stdev calculations | Standard library. Sufficient for benchmark aggregation. No need for numpy/pandas for this use case. |

**Why NOT numpy/pandas:**
- The project calculates averages, sorts results, and finds top-N models. This is basic arithmetic on small datasets (tens of rows, not millions).
- Adding numpy (25MB) or pandas (50MB+) for `mean()` and `sorted()` is absurd dependency bloat for a student tool.
- `statistics.mean()`, `statistics.stdev()`, and built-in `sorted()` cover every need.

**Confidence: HIGH** -- standard library, no version concern.

## Keep from Standard Library

| Module | Purpose | Notes |
|--------|---------|-------|
| `argparse` | CLI argument parsing | Sufficient for the CLI. click is nicer but adds a dependency for minimal gain when the CLI is simple. |
| `json` | Result serialization | Already in use. |
| `csv` | CSV export | Already in use. |
| `pathlib` | File paths | Already in use. Cross-platform. |
| `asyncio` | Async orchestration | New usage for concurrent benchmarks. |
| `statistics` | New: aggregation math | Mean, median, stdev for benchmark results. |
| `dataclasses` | New: lightweight config objects | For parameter sweep configs where Pydantic is overkill. |

## Do NOT Add

| Library | Why Not |
|---------|---------|
| `httpx` | Ollama SDK already wraps httpx internally. Going direct means reimplementing response parsing. |
| `aiohttp` | Same reason as httpx. Use Ollama's AsyncClient. |
| `numpy` / `pandas` | Massive dependencies for trivial math on small datasets. Use `statistics` stdlib. |
| `matplotlib` / `plotly` | PROJECT.md wants terminal bar charts, not browser graphs. rich handles terminal visualization. |
| `textual` | Full TUI framework. Out of scope per PROJECT.md ("dialog TUI" is explicitly excluded). |
| `click` | argparse is sufficient for a simple CLI. click adds a dependency without proportional value. |
| `tqdm` | rich subsumes tqdm's functionality and adds tables, styled output. One dependency, not two. |
| `pytest` | While a good test framework, the project uses simple scripts. If tests are added, pytest is fine but not part of the core runtime stack. Keep it as a dev dependency only. |

## Alternatives Considered (Summary)

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Terminal UI | rich | tqdm + tabulate | Two deps instead of one; less capable |
| Concurrency | asyncio + AsyncClient | threading + sync client | AsyncClient exists; threading adds complexity |
| Retry | tenacity | hand-rolled | Error-prone, missing jitter/backoff/async support |
| Data analysis | statistics (stdlib) | pandas | 50MB dependency for `mean()` on 20 rows |
| CLI | argparse (stdlib) | click | Existing code uses argparse; migration cost for no gain |
| HTTP | ollama SDK (wraps httpx) | raw httpx | Reimplements what SDK already does |

## Full Requirements

```
# requirements.txt (updated)
ollama>=0.6.1
pydantic>=2.12
rich>=14.0
tenacity>=9.0
```

```
# requirements-dev.txt (for contributors)
pytest>=8.0
ruff>=0.4.0
```

**Total new runtime dependencies: 2** (rich, tenacity). Both are well-maintained, widely-used, pure-Python packages with no C extension compilation issues on any platform.

## Version Compatibility Matrix

| Package | Min Python | Verified Version | PyPI Release Date |
|---------|------------|-----------------|-------------------|
| ollama | 3.8 | 0.6.1 | 2025-11-13 |
| pydantic | 3.9 | 2.12.5 | 2025-11-26 |
| rich | 3.8 | 14.3.3 | 2026-02-19 |
| tenacity | 3.10 | 9.1.4 | 2026-02-07 |

**Binding constraint: tenacity >=3.10** -- this sets the project's minimum Python version.

## Architecture Implications

1. **Concurrent benchmarking** uses `ollama.AsyncClient` with `asyncio.gather()` and a semaphore to limit concurrency. No new HTTP library needed.
2. **Parameter sweep** uses Pydantic models to define sweep configurations (ranges for num_ctx, num_gpu, temperature). Iterate with standard loops.
3. **Retry logic** wraps benchmark functions with `@tenacity.retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3))`.
4. **Progress tracking** uses `rich.progress.Progress` with custom columns showing model name, prompt count, and tokens/sec.
5. **Results display** uses `rich.table.Table` for ranked output and `rich.console.Console` for styled terminal output throughout.

## Sources

- Ollama Python SDK: https://pypi.org/project/ollama/ (v0.6.1, 2025-11-13) -- MEDIUM confidence (PyPI verified)
- Ollama AsyncClient docs: https://github.com/ollama/ollama-python -- HIGH confidence (official README)
- Rich: https://pypi.org/project/rich/ (v14.3.3, 2026-02-19) -- HIGH confidence (PyPI verified)
- Tenacity: https://pypi.org/project/tenacity/ (v9.1.4, 2026-02-07) -- HIGH confidence (PyPI verified)
- Pydantic: https://pypi.org/project/pydantic/ (v2.12.5, 2025-11-26) -- HIGH confidence (PyPI verified)
- Reference repo (alexziskind1/llama-throughput-lab): stdlib-heavy approach, confirms asyncio pattern is standard for this domain

---

*Researched: 2026-03-12*
