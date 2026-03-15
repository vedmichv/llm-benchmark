"""Benchmark execution, timeout wrapper, model offloading, correct averaging.

This module is the computational heart of llm_benchmark. It provides:
- run_single_benchmark(): Execute a single prompt against a model
- run_with_timeout(): Cross-platform timeout using threading (STAB-06)
- unload_model(): Offload models from GPU memory via keep_alive=0 (STAB-05)
- compute_averages(): Correct total_tokens/total_time averaging (STAB-04)
- benchmark_model(): Orchestrate benchmarking across prompts and runs
"""

from __future__ import annotations

import threading
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from llm_benchmark.backends import Backend, BackendError, BackendResponse
from llm_benchmark.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    get_console,
)
from llm_benchmark.models import (
    BenchmarkResult,
    ModelSummary,
)

# Known issues: (backend_name, error_substring) -> human-readable hint
KNOWN_ISSUES: dict[tuple[str, str], str] = {
    ("ollama", "timeout"): "Try a smaller model or increase --timeout",
    ("ollama", "connection refused"): "Is Ollama running? Start with: ollama serve",
    ("llama-cpp", "connection refused"): (
        "Is llama-server running? Start with: llama-server -m <model>"
    ),
    ("llama-cpp", "timeout"): "llama-cpp may be overloaded; try reducing --num-ctx",
    ("lm-studio", "connection refused"): (
        "Is LM Studio running? Start the server from the LM Studio app"
    ),
    ("lm-studio", "timeout"): "LM Studio may need a smaller model loaded",
}


def get_known_issue_hint(backend_name: str, error_msg: str) -> str | None:
    """Return a hint for a known error pattern, or None."""
    error_lower = error_msg.lower()
    for (bname, pattern), hint in KNOWN_ISSUES.items():
        if bname == backend_name and pattern in error_lower:
            return hint
    return None


def run_with_timeout(
    func: Any,
    timeout_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a function with a timeout. Uses threading for cross-platform support.

    Uses threading.Thread with join(timeout) for cross-platform
    compatibility (STAB-06). No Unix-only signal mechanisms.

    Args:
        func: Callable to execute.
        timeout_seconds: Maximum time in seconds.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        The return value of func.

    Raises:
        TimeoutError: If func does not complete within timeout_seconds.
    """
    result: list[Any] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as exc:
            error[0] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(
            f"Operation timed out after {timeout_seconds}s"
        )
    if error[0] is not None:
        raise error[0]
    return result[0]


def warmup_model(backend: Backend, model_name: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Send a short prompt to pre-load the model into memory.

    This ensures the first real benchmark run does not include model
    loading overhead in its timing measurements.

    Args:
        backend: Backend instance to use.
        model_name: Model name (e.g. "llama3.2:1b").
        timeout: Timeout in seconds for the warmup call.

    Returns:
        True if warmup succeeded, False otherwise.
    """
    console = get_console()
    console.print(f"  Warming up {model_name}...", end="")
    try:
        success = run_with_timeout(
            backend.warmup,
            timeout,
            model_name,
            timeout,
        )
        if success:
            console.print(" Ready")
        else:
            console.print(" [yellow]Warmup failed[/yellow]")
        return success
    except Exception as exc:
        console.print(f" [yellow]Warmup failed: {exc}[/yellow]")
        return False


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception is transient and should be retried.

    Retryable: BackendError with retryable=True.

    Non-retryable: TimeoutError (retrying a 200s timeout wastes minutes),
    BackendError with retryable=False (e.g. 404).
    """
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, BackendError):
        return exc.retryable
    return False


def _get_model_size_gb(backend: Backend, model_name: str) -> float | None:
    """Get on-disk model size in GB from the backend."""
    try:
        return backend.get_model_size(model_name)
    except Exception:
        return None


def detect_num_ctx(backend: Backend, model_name: str) -> int:
    """Auto-detect a reasonable num_ctx for a model.

    Scales context window based on model size to avoid OOM on small models.
    Thinking models get larger context than non-thinking, but still capped
    by model size to prevent KV cache from dominating memory.

    Returns:
        Recommended num_ctx value (2048-16384).
    """
    console = get_console()
    try:
        max_ctx = backend.detect_context_window(model_name)

        if max_ctx == 4096:
            # Default fallback from backend -- may be real or unknown
            pass

        # Check if thinking model (qwen3.5, deepseek-r1, etc.)
        is_thinking = any(
            t in model_name.lower()
            for t in ["qwen3.5", "deepseek-r1", "deepseek-r2"]
        )

        # Scale num_ctx by model size to avoid OOM on small models
        size_gb = _get_model_size_gb(backend, model_name)

        if size_gb is not None:
            if size_gb < 3:
                # Tiny models (<3 GB): 2048 for normal, 4096 for thinking
                target = 4096 if is_thinking else 2048
            elif size_gb < 8:
                # Small models (3-8 GB): 4096 for normal, 8192 for thinking
                target = 8192 if is_thinking else 4096
            elif size_gb < 20:
                # Medium models (8-20 GB): 8192 for normal, 16384 for thinking
                target = 16384 if is_thinking else 8192
            else:
                # Large models (20+ GB): 8192 for normal, 16384 for thinking
                target = 16384 if is_thinking else 8192
        else:
            target = 8192 if is_thinking else 4096

        target = min(target, max_ctx)

        label = " (thinking)" if is_thinking else ""
        size_label = f", {size_gb:.0f} GB" if size_gb else ""
        console.print(f"  [dim]Context: {target:,} tokens{label}{size_label}[/dim]")
        return target

    except Exception:
        return 4096


def unload_model(backend: Backend, model_name: str) -> bool:
    """Unload a model from GPU memory via Backend API.

    Uses the backend's unload mechanism to release the model
    from memory. No sudo required (STAB-05).

    Args:
        backend: Backend instance to use.
        model_name: Name of the model to unload (e.g. "llama3.2:1b").

    Returns:
        True if successful, False if an error occurred.
    """
    try:
        return backend.unload_model(model_name)
    except Exception:
        return False


def compute_averages(results: list[BenchmarkResult]) -> dict:
    """Correct averaging: total_tokens / total_time, not mean of rates.

    This implements the STAB-04 fix: sum all tokens, sum all durations,
    divide once. The arithmetic mean of rates is mathematically incorrect.

    Prompt-cached results (where prompt_cached=True) are excluded from
    prompt_eval calculations since they have zero prompt eval tokens/time.

    Args:
        results: List of benchmark results (may include failures).

    Returns:
        Dict with prompt_eval_ts, response_ts, total_ts keys,
        or empty dict if no successful results.
    """
    successful = [r for r in results if r.success and r.response is not None]
    if not successful:
        return {}

    # Response tokens/time: include all successful runs
    total_response_tokens = sum(r.response.eval_count for r in successful)
    total_response_time = sum(
        r.response.eval_duration for r in successful
    )

    # Prompt eval: exclude cached results (STAB-04 extension)
    non_cached = [r for r in successful if not r.prompt_cached]
    total_prompt_tokens = sum(
        r.response.prompt_eval_count for r in non_cached
    )
    total_prompt_time = sum(
        r.response.prompt_eval_duration for r in non_cached
    )

    # Total: use non-cached for prompt + all for response
    total_time = total_prompt_time + total_response_time
    total_tokens = total_prompt_tokens + total_response_tokens

    return {
        "prompt_eval_ts": (
            total_prompt_tokens / total_prompt_time
            if total_prompt_time > 0
            else 0
        ),
        "response_ts": (
            total_response_tokens / total_response_time
            if total_response_time > 0
            else 0
        ),
        "total_ts": total_tokens / total_time if total_time > 0 else 0,
    }


def run_single_benchmark(
    backend: Backend,
    model_name: str,
    prompt: str,
    verbose: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    num_ctx: int | None = None,
) -> BenchmarkResult:
    """Execute a single benchmark: one prompt against one model.

    Supports automatic retry of transient errors with exponential
    backoff via tenacity.

    Args:
        backend: Backend instance to use.
        model_name: Model name (e.g. "llama3.2:1b").
        prompt: The prompt text to send.
        verbose: If True, stream and display response chunks.
        timeout: Timeout in seconds for the benchmark run.
        max_retries: Maximum retry attempts (0 to disable retries).
        num_ctx: Context window size (None = backend default).

    Returns:
        BenchmarkResult with success=True on success, or
        success=False with error message on failure.
    """
    console = get_console()
    options = {"num_ctx": num_ctx} if num_ctx else {}
    messages = [{"role": "user", "content": prompt}]

    def _run_benchmark() -> BackendResponse:
        """Inner function executed within timeout wrapper."""
        if verbose:
            # Streaming mode: print response chunks
            result = backend.chat(
                model_name,
                messages,
                stream=True,
                options=options,
            )
            char_count = 0
            for chunk in result.chunks:
                if char_count < 200:
                    console.print(chunk, end="", highlight=False)
                    char_count += len(chunk)
                    if char_count >= 200:
                        console.print("...", end="", highlight=False)
            console.print()  # newline after streaming
            return result.response
        else:
            # Non-streaming mode
            response = backend.chat(
                model_name,
                messages,
                options=options,
            )
            return response

    def _run_with_timeout() -> BackendResponse:
        """Wrap _run_benchmark with timeout -- each retry gets full budget."""
        return run_with_timeout(_run_benchmark, timeout)

    try:
        if max_retries > 0:
            # Build dynamic tenacity retryer (max_retries is runtime)
            def _before_sleep(retry_state):
                attempt = retry_state.attempt_number
                console.print(
                    f"    [yellow]Retry {attempt}/{max_retries}...[/yellow]"
                )

            retryer = retry(
                stop=stop_after_attempt(max_retries + 1),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception(_is_retryable),
                before_sleep=_before_sleep,
                reraise=True,
            )
            response = retryer(_run_with_timeout)()
        else:
            response = _run_with_timeout()

        return BenchmarkResult(
            model=model_name,
            prompt=prompt,
            success=True,
            response=response,
            prompt_cached=response.prompt_cached,
        )

    except TimeoutError:
        # Unload and reload to reset state after timeout
        unload_model(backend, model_name)
        return BenchmarkResult(
            model=model_name,
            prompt=prompt,
            success=False,
            error=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        return BenchmarkResult(
            model=model_name,
            prompt=prompt,
            success=False,
            error=str(exc),
        )


def benchmark_model(
    backend: Backend,
    model_name: str,
    prompts: list[str],
    verbose: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    runs_per_prompt: int = 1,
    skip_warmup: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    num_ctx: int | None = None,
) -> ModelSummary:
    """Orchestrate benchmarking a single model across all prompts.

    Args:
        backend: Backend instance to use.
        model_name: Model name.
        prompts: List of prompt strings to benchmark.
        verbose: If True, stream responses.
        timeout: Per-run timeout in seconds.
        runs_per_prompt: Number of runs per prompt for statistical reliability.
        skip_warmup: If True, skip the warmup run before benchmarking.
        max_retries: Maximum retry attempts per run (0 to disable).

    Returns:
        ModelSummary with aggregated results.
    """
    console = get_console()
    all_results: list[BenchmarkResult] = []
    _cache_explanation_shown = False

    # Auto-detect optimal context window if not explicitly set
    if num_ctx is None:
        num_ctx = detect_num_ctx(backend, model_name)

    # Warmup: pre-load model to exclude load time from measurements
    if not skip_warmup:
        warmup_model(backend, model_name, timeout)
    else:
        console.print(
            "  [dim]Warmup skipped -- first run may include model load time[/dim]"
        )

    for prompt_idx, prompt in enumerate(prompts):
        console.print(
            f"  Prompt {prompt_idx + 1}/{len(prompts)}: "
            f"[dim]{prompt[:60]}...[/dim]"
        )

        for run_num in range(runs_per_prompt):
            if runs_per_prompt > 1:
                console.print(f"    Run {run_num + 1}/{runs_per_prompt}")

            result = run_single_benchmark(
                backend=backend,
                model_name=model_name,
                prompt=prompt,
                verbose=verbose,
                timeout=timeout,
                max_retries=max_retries,
                num_ctx=num_ctx,
            )
            all_results.append(result)

            if result.success and result.response:
                response_ts = (
                    result.response.eval_count
                    / result.response.eval_duration
                    if result.response.eval_duration > 0
                    else 0
                )
                cached_tag = " [dim]\\[cached][/dim]" if result.prompt_cached else ""
                console.print(
                    f"    [green]{result.response.eval_count} tokens "
                    f"@ {response_ts:.1f} t/s[/green]{cached_tag}"
                )
                if result.prompt_cached and not _cache_explanation_shown:
                    console.print(
                        "    [dim italic]Prompt caching: Ollama reused the prompt "
                        "from memory, so prompt eval time is 0.[/dim italic]"
                    )
                    _cache_explanation_shown = True
            elif not result.success:
                console.print(f"    [red]Failed: {result.error}[/red]")
                # Show known-issue hint if available
                hint = get_known_issue_hint(backend.name, result.error or "")
                if hint:
                    console.print(f"    [yellow]Hint: {hint}[/yellow]")
                if "Timeout" in (result.error or ""):
                    console.print(
                        "    [dim]Skipping remaining runs — reloading model...[/dim]"
                    )
                    unload_model(backend, model_name)
                    warmup_model(backend, model_name, timeout)
                    break  # skip remaining runs, move to next prompt

    # Warn if all successful results are cached
    successful = [r for r in all_results if r.success]
    if successful and all(r.prompt_cached for r in successful):
        console.print(
            "  [yellow]All runs cached -- prompt eval metrics "
            "unavailable for this model[/yellow]"
        )

    # Failure summary
    failed = [r for r in all_results if not r.success]
    if failed:
        console.print(
            f"  [yellow]Failures: {len(failed)}/{len(all_results)} runs failed[/yellow]"
        )

    avgs = compute_averages(all_results)

    return ModelSummary(
        model=model_name,
        results=all_results,
        avg_prompt_eval_ts=avgs.get("prompt_eval_ts", 0),
        avg_response_ts=avgs.get("response_ts", 0),
        avg_total_ts=avgs.get("total_ts", 0),
    )
