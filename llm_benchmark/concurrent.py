"""Async concurrent benchmarking orchestration.

Fires N parallel requests to the same Ollama model using asyncio.gather
and measures wall-time aggregate throughput. Each request has its own
try/except so failures are isolated.

Exports:
    run_concurrent_batch: Sync wrapper for concurrent batch execution.
    benchmark_model_concurrent: Full concurrent benchmark orchestration.
    auto_detect_concurrency: Resource-based concurrency default.
"""

from __future__ import annotations

import asyncio
import statistics
import time

import ollama

from llm_benchmark.backends import BackendResponse
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


def _ns_to_sec(ns: int) -> float:
    """Convert nanoseconds to seconds (for raw ollama async responses)."""
    return ns / 1_000_000_000


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


async def _single_request(
    client: ollama.AsyncClient,
    model: str,
    prompt: str,
    request_id: int,
    num_workers: int,
    console,
) -> BenchmarkResult:
    """Execute a single async chat request and return a BenchmarkResult.

    On success, prints a live per-request summary line.
    On failure, returns BenchmarkResult(success=False) without stopping others.
    """
    try:
        raw_response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Convert SDK response to dict
        if hasattr(raw_response, "model_dump"):
            raw_response = raw_response.model_dump()
        elif hasattr(raw_response, "dict"):
            raw_response = raw_response.dict()

        # Build BackendResponse from raw Ollama data (nanoseconds -> seconds)
        msg = raw_response.get("message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        prompt_eval_count = raw_response.get("prompt_eval_count", 0)
        prompt_cached = prompt_eval_count == -1
        if prompt_cached:
            prompt_eval_count = 0

        response = BackendResponse(
            model=raw_response.get("model", model),
            content=content,
            done=raw_response.get("done", True),
            prompt_eval_count=prompt_eval_count,
            eval_count=raw_response.get("eval_count", 0),
            total_duration=_ns_to_sec(raw_response.get("total_duration", 0)),
            load_duration=_ns_to_sec(raw_response.get("load_duration", 0)),
            prompt_eval_duration=_ns_to_sec(raw_response.get("prompt_eval_duration", 0)),
            eval_duration=_ns_to_sec(raw_response.get("eval_duration", 0)),
            prompt_cached=prompt_cached,
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


async def _run_batch(
    model: str,
    prompt: str,
    n: int,
    timeout: int,
) -> ConcurrentBatchResult:
    """Fire N parallel requests via asyncio.gather, measure wall time.

    Each request runs in its own try/except via _single_request, so
    gather does NOT need return_exceptions.
    """
    console = get_console()

    async with ollama.AsyncClient() as client:
        tasks = [
            _single_request(client, model, prompt, i, n, console)
            for i in range(n)
        ]

        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
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
    model: str,
    prompt: str,
    n: int = DEFAULT_CONCURRENT,
    timeout: int = DEFAULT_TIMEOUT,
) -> ConcurrentBatchResult:
    """Sync wrapper: fire N concurrent async requests to Ollama.

    Args:
        model: Ollama model name.
        prompt: Prompt text.
        n: Number of concurrent requests.
        timeout: Per-request timeout in seconds.

    Returns:
        ConcurrentBatchResult with wall-time and per-request metrics.
    """
    return asyncio.run(_run_batch(model, prompt, n, timeout))


def benchmark_model_concurrent(
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

    Note: warmup_model now requires a backend parameter. For concurrent mode,
    we create a temporary backend for warmup only.

    Args:
        model_name: Ollama model name.
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
        from llm_benchmark.backends import create_backend
        backend = create_backend()
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
