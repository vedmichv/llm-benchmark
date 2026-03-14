"""Concurrent benchmarking orchestration using ThreadPoolExecutor.

Fires N parallel requests to the same model using a Backend instance
and measures wall-time aggregate throughput. Each request has its own
try/except so failures are isolated.

Exports:
    run_concurrent_batch: Sync wrapper for concurrent batch execution.
    benchmark_model_concurrent: Full concurrent benchmark orchestration.
    auto_detect_concurrency: Resource-based concurrency default.
"""

from __future__ import annotations

import concurrent.futures
import statistics
import time

from llm_benchmark.backends import Backend, BackendResponse
from llm_benchmark.config import (
    DEFAULT_CONCURRENT,
    DEFAULT_TIMEOUT,
    get_console,
)
from llm_benchmark.models import (
    BenchmarkResult,
    ConcurrentBatchResult,
)
from llm_benchmark.runner import warmup_model


def auto_detect_concurrency(
    ram_gb: float, gpu_vram_gb: float | None
) -> int:
    """Return a sensible default concurrency level based on available resources.

    Args:
        ram_gb: System RAM in gigabytes.
        gpu_vram_gb: GPU VRAM in gigabytes, or None if no discrete GPU.

    Returns:
        8 if gpu_vram_gb >= 16, 4 if ram_gb >= 32, else 2.
    """
    if gpu_vram_gb is not None and gpu_vram_gb >= 16:
        return 8
    if ram_gb >= 32:
        return 4
    return 2


def _single_request(
    backend: Backend,
    model: str,
    prompt: str,
    request_id: int,
    num_workers: int,
    console,
) -> BenchmarkResult:
    """Execute a single chat request via Backend and return a BenchmarkResult.

    On success, prints a live per-request summary line.
    On failure, returns BenchmarkResult(success=False) without stopping others.
    """
    try:
        response = backend.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Compute per-request throughput
        rate = response.eval_count / response.eval_duration if response.eval_duration > 0 else 0

        console.print(
            f"    Request {request_id + 1}/{num_workers}: "
            f"[green]{response.eval_count} tokens @ {rate:.1f} t/s[/green]"
        )

        return BenchmarkResult(
            model=model,
            prompt=prompt,
            success=True,
            response=response,
            prompt_cached=response.prompt_cached,
        )

    except Exception as exc:
        console.print(
            f"    Request {request_id + 1}/{num_workers}: "
            f"[red]FAILED - {exc}[/red]"
        )
        return BenchmarkResult(
            model=model,
            prompt=prompt,
            success=False,
            error=str(exc),
        )


def _run_batch(
    backend: Backend,
    model: str,
    prompt: str,
    n: int,
    timeout: int,
) -> ConcurrentBatchResult:
    """Fire N parallel requests via ThreadPoolExecutor, measure wall time.

    Each request runs in its own try/except via _single_request, so
    failures are isolated.
    """
    console = get_console()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        wall_start = time.perf_counter()
        futures = [
            pool.submit(_single_request, backend, model, prompt, i, n, console)
            for i in range(n)
        ]
        results = [f.result(timeout=timeout) for f in concurrent.futures.as_completed(futures)]
        wall_time_s = time.perf_counter() - wall_start

    successful = [r for r in results if r.success and r.response is not None]

    # Aggregate throughput: sum(all eval_count) / wall_time
    total_tokens = sum(r.response.eval_count for r in successful)
    aggregate_throughput_ts = (
        total_tokens / wall_time_s if wall_time_s > 0 else 0.0
    )

    # Average per-request throughput: mean of individual rates
    per_request_rates = []
    for r in successful:
        if r.response.eval_duration > 0:
            per_request_rates.append(r.response.eval_count / r.response.eval_duration)

    avg_request_throughput_ts = (
        statistics.mean(per_request_rates) if per_request_rates else 0.0
    )

    return ConcurrentBatchResult(
        model=model,
        prompt=prompt,
        num_workers=n,
        wall_time_s=wall_time_s,
        results=list(results),
        aggregate_throughput_ts=aggregate_throughput_ts,
        avg_request_throughput_ts=avg_request_throughput_ts,
    )


def run_concurrent_batch(
    backend: Backend,
    model: str,
    prompt: str,
    n: int = DEFAULT_CONCURRENT,
    timeout: int = DEFAULT_TIMEOUT,
) -> ConcurrentBatchResult:
    """Fire N concurrent requests to the backend via ThreadPoolExecutor.

    Args:
        backend: Backend instance.
        model: Model name.
        prompt: Prompt text.
        n: Number of concurrent requests.
        timeout: Per-request timeout in seconds.

    Returns:
        ConcurrentBatchResult with wall-time and per-request metrics.
    """
    return _run_batch(backend, model, prompt, n, timeout)


def benchmark_model_concurrent(
    backend: Backend,
    model_name: str,
    prompts: list[str],
    num_workers: int = DEFAULT_CONCURRENT,
    runs_per_prompt: int = 1,
    timeout: int = DEFAULT_TIMEOUT,
    skip_warmup: bool = False,
    verbose: bool = False,
) -> list[ConcurrentBatchResult]:
    """Orchestrate concurrent benchmarking for one model across prompts.

    Warms up the model once, then for each prompt (x runs_per_prompt),
    fires a concurrent batch of num_workers parallel requests.

    Args:
        backend: Backend instance.
        model_name: Model name.
        prompts: List of prompt strings.
        num_workers: Number of concurrent requests per batch.
        runs_per_prompt: How many times to repeat each prompt.
        timeout: Per-request timeout in seconds.
        skip_warmup: If True, skip warmup run.
        verbose: If True, show extra detail.

    Returns:
        List of ConcurrentBatchResult, one per prompt x run combination.
    """
    console = get_console()
    all_batches: list[ConcurrentBatchResult] = []

    # Warmup once
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

            console.print(
                f"  Launching {num_workers} concurrent requests..."
            )

            batch = run_concurrent_batch(
                backend=backend,
                model=model_name,
                prompt=prompt,
                n=num_workers,
                timeout=timeout,
            )
            all_batches.append(batch)

            # Summary line
            succeeded = sum(1 for r in batch.results if r.success)
            console.print(
                f"  [bold]Batch complete:[/bold] "
                f"{succeeded}/{num_workers} succeeded, "
                f"wall time {batch.wall_time_s:.2f}s, "
                f"aggregate {batch.aggregate_throughput_ts:.1f} t/s, "
                f"avg per-request {batch.avg_request_throughput_ts:.1f} t/s"
            )

    return all_batches
