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

import ollama
from ollama import RequestError as _RequestError, ResponseError as _ResponseError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from llm_benchmark.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_WARMUP_PROMPT,
    get_console,
)
from llm_benchmark.models import (
    BenchmarkResult,
    ModelSummary,
    OllamaResponse,
    _ns_to_sec,
)


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


def warmup_model(model_name: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Send a short prompt to pre-load the model into memory.

    This ensures the first real benchmark run does not include model
    loading overhead in its timing measurements.

    Args:
        model_name: Ollama model name (e.g. "llama3.2:1b").
        timeout: Timeout in seconds for the warmup call.

    Returns:
        True if warmup succeeded, False otherwise.
    """
    console = get_console()
    console.print(f"  Warming up {model_name}...", end="")
    try:
        run_with_timeout(
            ollama.chat,
            timeout,
            model=model_name,
            messages=[{"role": "user", "content": DEFAULT_WARMUP_PROMPT}],
        )
        console.print(" Ready")
        return True
    except Exception as exc:
        console.print(f" [yellow]Warmup failed: {exc}[/yellow]")
        return False


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception is transient and should be retried.

    Retryable: ConnectionError, TimeoutError, ollama.RequestError,
    and ollama.ResponseError with status_code >= 500.

    Non-retryable: ollama.ResponseError with status_code < 500 (e.g. 404).
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, _RequestError):
        return True
    if isinstance(exc, _ResponseError):
        return getattr(exc, "status_code", 0) >= 500
    return False


def unload_model(model_name: str) -> bool:
    """Unload a model from GPU memory via Ollama API.

    Uses keep_alive=0 to tell Ollama to immediately release the model
    from memory. No sudo required (STAB-05).

    Args:
        model_name: Name of the model to unload (e.g. "llama3.2:1b").

    Returns:
        True if successful, False if an error occurred.
    """
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        return True
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
        _ns_to_sec(r.response.eval_duration) for r in successful
    )

    # Prompt eval: exclude cached results (STAB-04 extension)
    non_cached = [r for r in successful if not r.prompt_cached]
    total_prompt_tokens = sum(
        r.response.prompt_eval_count for r in non_cached
    )
    total_prompt_time = sum(
        _ns_to_sec(r.response.prompt_eval_duration) for r in non_cached
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
    model_name: str,
    prompt: str,
    verbose: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> BenchmarkResult:
    """Execute a single benchmark: one prompt against one model.

    Supports automatic retry of transient errors (ConnectionError,
    TimeoutError, 5xx) with exponential backoff via tenacity.

    Args:
        model_name: Ollama model name (e.g. "llama3.2:1b").
        prompt: The prompt text to send.
        verbose: If True, stream and display response chunks.
        timeout: Timeout in seconds for the benchmark run.
        max_retries: Maximum retry attempts (0 to disable retries).

    Returns:
        BenchmarkResult with success=True on success, or
        success=False with error message on failure.
    """
    console = get_console()

    def _run_benchmark() -> dict:
        """Inner function executed within timeout wrapper."""
        if verbose:
            # Streaming mode: print response chunks
            stream = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            last_chunk = None
            char_count = 0
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if char_count < 200:
                    console.print(content, end="", highlight=False)
                    char_count += len(content)
                    if char_count >= 200:
                        console.print("...", end="", highlight=False)
                last_chunk = chunk
            console.print()  # newline after streaming
            return last_chunk
        else:
            # Non-streaming mode
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response

    def _run_with_timeout() -> dict:
        """Wrap _run_benchmark with timeout -- each retry gets full budget."""
        return run_with_timeout(_run_benchmark, timeout)

    def _parse_response(raw_response: dict) -> BenchmarkResult:
        """Convert raw Ollama response to BenchmarkResult."""
        if raw_response is None:
            return BenchmarkResult(
                model=model_name,
                prompt=prompt,
                success=False,
                error="No response received",
            )

        # Convert SDK response to dict for Pydantic validation
        if hasattr(raw_response, "model_dump"):
            raw_response = raw_response.model_dump()
        elif hasattr(raw_response, "dict"):
            raw_response = raw_response.dict()

        response = OllamaResponse.model_validate(raw_response)

        return BenchmarkResult(
            model=model_name,
            prompt=prompt,
            success=True,
            response=response,
            prompt_cached=response.prompt_cached,
        )

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
            raw_response = retryer(_run_with_timeout)()
        else:
            raw_response = _run_with_timeout()

        return _parse_response(raw_response)

    except TimeoutError:
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
    model_name: str,
    prompts: list[str],
    verbose: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    runs_per_prompt: int = 1,
    skip_warmup: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ModelSummary:
    """Orchestrate benchmarking a single model across all prompts.

    Args:
        model_name: Ollama model name.
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

    # Warmup: pre-load model to exclude load time from measurements
    if not skip_warmup:
        warmup_model(model_name, timeout)
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
                model_name=model_name,
                prompt=prompt,
                verbose=verbose,
                timeout=timeout,
                max_retries=max_retries,
            )
            all_results.append(result)

            if result.success and result.response:
                response_ts = (
                    result.response.eval_count
                    / _ns_to_sec(result.response.eval_duration)
                    if result.response.eval_duration > 0
                    else 0
                )
                console.print(
                    f"    [green]{result.response.eval_count} tokens "
                    f"@ {response_ts:.1f} t/s[/green]"
                )
            elif not result.success:
                console.print(f"    [red]Failed: {result.error}[/red]")

    avgs = compute_averages(all_results)

    return ModelSummary(
        model=model_name,
        results=all_results,
        avg_prompt_eval_ts=avgs.get("prompt_eval_ts", 0),
        avg_response_ts=avgs.get("response_ts", 0),
        avg_total_ts=avgs.get("total_ts", 0),
    )
