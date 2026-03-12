# Phase 2: Measurement Reliability - Research

**Researched:** 2026-03-12
**Domain:** Benchmark warmup, retry logic, prompt caching visibility, results organization
**Confidence:** HIGH

## Summary

Phase 2 adds four capabilities to make benchmark numbers trustworthy: (1) warmup runs to exclude model load overhead, (2) automatic retry with exponential backoff for transient failures, (3) visible prompt caching indicators across all output formats, and (4) results directory organization with .gitignore.

The codebase is well-prepared for these changes. tenacity is already a declared dependency (>=9.0). The Ollama SDK exposes `RequestError` (connection/network errors) and `ResponseError` (server errors with `status_code`), making retry targeting straightforward. The existing `prompt_cached` flag on `BenchmarkResult` and the cache-excluding logic in `runner.compute_averages()` provide the foundation for caching visibility. The main work is wiring these together and adding CLI flags.

**Primary recommendation:** Implement warmup as a lightweight `ollama.chat()` call with a short prompt before the prompt loop in `benchmark_model()`, wrap `run_single_benchmark()` inner logic with tenacity retry, and add `prompt_cached` columns to CSV/Markdown exporters.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 1 warmup run per model, before all prompts (not per-prompt), using short fixed prompt (e.g., "Hello")
- Show brief status line: "Warming up llama3.2:1b..." then "Ready"
- Add --skip-warmup CLI flag (default: warmup on)
- When --skip-warmup is used, show brief note: "Warmup skipped -- first run may include model load time"
- 3 retries by default with exponential backoff (using tenacity library)
- Add --max-retries N CLI flag (default: 3, set to 0 to disable)
- Retryable errors: connection errors, Ollama server errors, timeouts
- After retries exhausted: mark run as failed (success=False), continue to next prompt/run
- Show retry count during attempts: "Retry 1/3..."
- Per-run [cached] tag next to affected runs in terminal output
- First time caching is detected in a session, show one-liner explanation
- When ALL runs for a model are cached, show warning about unavailable prompt eval metrics
- All export formats include prompt caching indicators
- Flat results/ directory with timestamped filenames
- Add .gitignore inside results/ to prevent accidental commits
- Auto-create results/ on first run

### Claude's Discretion
- Exact backoff timing for retries (e.g., 1s/2s/4s or similar)
- Exact warmup prompt text
- Which tenacity exception types map to retryable errors
- .gitignore patterns inside results/

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BENCH-01 | Tool runs warmup requests before actual measurements to exclude model load overhead | Warmup via `ollama.chat()` with short prompt in `benchmark_model()` before prompt loop; --skip-warmup flag |
| BENCH-02 | Tool retries failed requests with exponential backoff (configurable max retries) | tenacity >=9.0 already in deps; wrap inner benchmark call with `@retry` decorator; --max-retries flag |
| BENCH-07 | Prompt caching detection excludes affected metrics from averages (instead of silently corrupting) | `runner.compute_averages()` already excludes cached; need visible [cached] tags and export indicators |
| UX-04 | All result files saved to results/ directory (not project root) | Exporters already default to `output_dir="results"`; add results/.gitignore |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tenacity | >=9.0 | Retry with exponential backoff | Already declared dependency; standard Python retry library |
| ollama | >=0.6 | Ollama API client | Already used; provides RequestError/ResponseError exceptions |
| rich | >=14.0 | Console output | Already used; provides status spinners for warmup display |

### Supporting
No new libraries needed. All phase work uses existing dependencies.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| tenacity | Manual retry loop | tenacity handles backoff, jitter, exception filtering cleanly; manual loop is error-prone |

**Installation:**
```bash
# No new packages -- all already in pyproject.toml
uv sync
```

## Architecture Patterns

### Changes to Existing Structure
```
llm_benchmark/
  config.py          # Add DEFAULT_MAX_RETRIES=3, DEFAULT_WARMUP_PROMPT
  cli.py             # Add --skip-warmup, --max-retries flags
  runner.py          # Add warmup_model(), wrap run_single_benchmark with retry
  exporters.py       # Add prompt_cached column to CSV/Markdown
results/
  .gitignore         # NEW: ignore benchmark output files
```

### Pattern 1: Warmup as Separate Function
**What:** `warmup_model(model_name)` sends a short chat request to load the model into GPU/RAM before measurement.
**When to use:** Called once per model in `benchmark_model()` before the prompt loop.
**Example:**
```python
# In runner.py
def warmup_model(model_name: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Send a short prompt to pre-load the model into memory."""
    console = get_console()
    console.print(f"  Warming up {model_name}...", end="")
    try:
        run_with_timeout(
            lambda: ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
            ),
            timeout,
        )
        console.print(" Ready")
        return True
    except Exception as exc:
        console.print(f" [yellow]Warmup failed: {exc}[/yellow]")
        return False
```

### Pattern 2: Tenacity Retry with Dynamic Configuration
**What:** Use tenacity's `retry` as a callable (not decorator) to support runtime `max_retries` parameter.
**When to use:** When retry count comes from CLI args and can't be baked into a decorator at import time.
**Example:**
```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep,
)
from ollama import RequestError, ResponseError

def _is_retryable_error(exc: BaseException) -> bool:
    """Determine if an exception is retryable."""
    if isinstance(exc, (ConnectionError, TimeoutError, RequestError)):
        return True
    if isinstance(exc, ResponseError) and exc.status_code >= 500:
        return True
    return False

def run_single_benchmark(
    model_name: str,
    prompt: str,
    verbose: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> BenchmarkResult:
    console = get_console()

    def _on_retry(retry_state):
        attempt = retry_state.attempt_number
        console.print(f"    [yellow]Retry {attempt}/{max_retries}...[/yellow]")

    # Build retrier dynamically based on max_retries
    retryer = retry(
        stop=stop_after_attempt(max_retries + 1),  # +1 because first attempt counts
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, RequestError)),
        before_sleep=_on_retry,
        reraise=True,
    ) if max_retries > 0 else None

    def _run_benchmark():
        # ... existing ollama.chat logic ...
        pass

    try:
        if retryer:
            raw_response = run_with_timeout(
                retryer(_run_benchmark), timeout * (max_retries + 1)
            )
        else:
            raw_response = run_with_timeout(_run_benchmark, timeout)
        # ... parse response ...
    except Exception as exc:
        return BenchmarkResult(
            model=model_name, prompt=prompt, success=False,
            error=str(exc),
        )
```

### Pattern 3: Session-level Caching State
**What:** Track whether the caching explanation has been shown in the current session to avoid repeating it.
**When to use:** In `benchmark_model()` or `_handle_run()` to show the one-liner only once.
**Example:**
```python
# In _handle_run or as a module-level variable in runner.py
_cache_explanation_shown = False

def _show_cache_indicator(console, result, cache_explanation_shown):
    """Show [cached] tag and optional one-time explanation."""
    if result.prompt_cached:
        console.print("    [dim][cached][/dim]", end="")
        if not cache_explanation_shown:
            console.print(
                "\n    [dim italic]Prompt caching: Ollama reused the prompt "
                "from memory, so prompt eval time is 0.[/dim italic]"
            )
            return True  # explanation now shown
    return cache_explanation_shown
```

### Anti-Patterns to Avoid
- **Retry wrapping the entire benchmark_model():** Retry should be per-request, not per-model. One failed prompt shouldn't retry the entire model benchmark.
- **Retry inside the timeout:** The timeout wraps the retried call. If you put retry inside run_with_timeout, a single timeout kills all retry attempts. Instead, either give the timeout wrapper a larger budget or put retry outside the timeout.
- **Decorator-style @retry on run_single_benchmark:** The max_retries comes from CLI args at runtime, so a static decorator won't work. Use tenacity's callable form or build the retrying wrapper dynamically.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Exponential backoff | Manual sleep loops with try/except | tenacity.wait_exponential | Handles jitter, min/max bounds, clean exception filtering |
| Retry state tracking | Counter variables + while loops | tenacity.retry with before_sleep callback | Cleaner, less error-prone, supports attempt counting |
| Console status indicator | print() with manual cursor management | rich Console.print() with markup | Already the project pattern; consistent styling |

## Common Pitfalls

### Pitfall 1: Timeout vs Retry Interaction
**What goes wrong:** If `run_with_timeout` wraps the retried function, a single timeout kills all retries. If retry wraps the timeout, each retry gets its own timeout budget.
**Why it happens:** Unclear layering of timeout and retry concerns.
**How to avoid:** Put retry logic INSIDE `run_single_benchmark` but OUTSIDE `run_with_timeout`. Each retry attempt gets its own timeout. Alternatively, retry should catch TimeoutError from `run_with_timeout` and retry the whole timeout-wrapped call.
**Warning signs:** Retries never fire because the outer timeout kills them first.

### Pitfall 2: Retrying Non-Retryable Errors
**What goes wrong:** Retrying a 404 "model not found" error wastes time and confuses users.
**Why it happens:** Catching all exceptions without filtering.
**How to avoid:** Only retry: `ConnectionError`, `TimeoutError`, `ollama.RequestError` (network), `ollama.ResponseError` with `status_code >= 500` (server errors). Do NOT retry `ResponseError` with 4xx codes (client errors like model not found).
**Warning signs:** "Retry 1/3..." followed by the same error three times.

### Pitfall 3: Warmup Counted in Results
**What goes wrong:** The warmup run's timing data leaks into the benchmark averages.
**Why it happens:** Warmup uses the same `run_single_benchmark` function and accidentally gets added to results.
**How to avoid:** Warmup should be a separate function that does NOT return a BenchmarkResult. Use a bare `ollama.chat()` call, discard the response. The warmup's only purpose is to load the model into GPU memory.

### Pitfall 4: Duplicate compute_averages
**What goes wrong:** `compute_averages` exists in both `models.py` (lines 86-125) and `runner.py` (lines 91-142). The runner version correctly excludes cached results from prompt eval; the models version does not.
**Why it happens:** Likely a leftover from Phase 1 development.
**How to avoid:** Remove the duplicate in `models.py` or make both identical. The runner version is the correct one (excludes cached). Tests import from `runner`, so that's the canonical version. Clean this up before adding more logic.

### Pitfall 5: CSV/Markdown Missing Prompt Cached Column
**What goes wrong:** JSON export already includes `prompt_cached` via `_result_to_dict()`, but CSV and Markdown exports don't show caching status.
**Why it happens:** CSV writer manually constructs rows without referencing all fields.
**How to avoid:** Add "Cached" column to CSV headers and row writer. Add [cached] indicator in Markdown detailed results.

## Code Examples

### Tenacity Retry with Custom Exception Filter
```python
# Source: tenacity library API + Ollama SDK exception types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)
from ollama import RequestError, ResponseError

def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient/retryable errors."""
    if isinstance(exc, (ConnectionError, TimeoutError, RequestError)):
        return True
    if isinstance(exc, ResponseError) and exc.status_code >= 500:
        return True
    return False

# Dynamic retry configuration (max_retries from CLI)
retryer = retry(
    stop=stop_after_attempt(max_retries + 1),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
```

### Ollama SDK Exception Types
```python
# Source: ollama SDK (verified via import inspection)
import ollama

# RequestError: raised for network/connection issues
# - No status_code attribute by default
# - Subclass of Exception

# ResponseError: raised for HTTP error responses from Ollama server
# - Has .status_code (int) and .error (str) attributes
# - status_code >= 500: server error (retryable)
# - status_code 4xx: client error (NOT retryable, e.g. model not found)
```

### Results .gitignore
```gitignore
# Benchmark result files -- do not commit
*.json
*.csv
*.md
!.gitignore
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No warmup | Warmup run before measurement | This phase | Load time excluded from throughput metrics |
| Crash on transient error | Retry with backoff | This phase | Benchmark completes even with flaky connections |
| Silent prompt caching | Visible [cached] indicators | This phase | Users understand why prompt eval is 0 |

## Open Questions

1. **Retry + Timeout layering**
   - What we know: Both mechanisms exist independently. Retry should wrap per-attempt, timeout per-attempt.
   - What's unclear: Should the overall timeout for a retried request be `timeout * (max_retries + 1)` or should each attempt have its own independent timeout?
   - Recommendation: Each retry attempt gets its own `timeout` seconds via `run_with_timeout`. The retry logic catches `TimeoutError` and retries. This is simpler and more predictable.

2. **Warmup failure handling**
   - What we know: Warmup failure means the model can't load.
   - What's unclear: Should the tool skip the model entirely or proceed (the first benchmark run will also fail)?
   - Recommendation: Log the failure, proceed with benchmarking. If the model truly can't load, benchmark runs will fail individually and get marked as failed. This is consistent with "never crash, always continue."

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | none (uses pyproject.toml defaults) |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | warmup_model() calls ollama.chat with short prompt | unit | `uv run pytest tests/test_runner.py::TestWarmupModel -x` | No -- Wave 0 |
| BENCH-01 | --skip-warmup flag parsed and passed through | unit | `uv run pytest tests/test_cli.py::TestSkipWarmup -x` | No -- Wave 0 |
| BENCH-02 | run_single_benchmark retries on ConnectionError | unit | `uv run pytest tests/test_runner.py::TestRetryLogic -x` | No -- Wave 0 |
| BENCH-02 | run_single_benchmark does NOT retry on 404 ResponseError | unit | `uv run pytest tests/test_runner.py::TestRetryLogic -x` | No -- Wave 0 |
| BENCH-02 | --max-retries 0 disables retry | unit | `uv run pytest tests/test_runner.py::TestRetryLogic -x` | No -- Wave 0 |
| BENCH-07 | [cached] tag shown in terminal for cached runs | unit | `uv run pytest tests/test_runner.py::TestCacheVisibility -x` | No -- Wave 0 |
| BENCH-07 | CSV export includes Cached column | unit | `uv run pytest tests/test_exporters.py::TestCsvCacheColumn -x` | No -- Wave 0 |
| BENCH-07 | Markdown export includes [cached] indicator | unit | `uv run pytest tests/test_exporters.py::TestMarkdownCacheIndicator -x` | No -- Wave 0 |
| UX-04 | Export functions default to results/ directory | unit | `uv run pytest tests/test_exporters.py -x` | No -- Wave 0 |
| UX-04 | results/.gitignore created | unit | `uv run pytest tests/test_exporters.py::TestResultsGitignore -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_exporters.py` -- covers UX-04, BENCH-07 export formats
- [ ] `tests/test_runner.py` -- add TestWarmupModel, TestRetryLogic, TestCacheVisibility classes
- [ ] `tests/test_cli.py` -- add tests for --skip-warmup, --max-retries flags

## Sources

### Primary (HIGH confidence)
- Ollama Python SDK -- verified `RequestError`, `ResponseError` exception types and `status_code` attribute via live import
- tenacity library -- verified `retry`, `stop_after_attempt`, `wait_exponential`, `retry_if_exception_type`, `before_sleep` APIs via live import
- Existing codebase -- read all source files (runner.py, models.py, exporters.py, cli.py, config.py, tests/)

### Secondary (MEDIUM confidence)
- None needed -- all findings verified against installed libraries

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- tenacity already installed and API verified
- Architecture: HIGH -- clear integration points in existing code
- Pitfalls: HIGH -- identified from direct code reading (e.g., duplicate compute_averages)

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable domain, no fast-moving dependencies)
