"""Parameter sweep module: auto-tune num_ctx and num_gpu for each model.

Tests combinations of context window size (num_ctx) and GPU layer count
(num_gpu) per model, identifies the best configuration by highest
response tokens/second, and displays results in a Rich table.
"""

from __future__ import annotations

import ollama
from ollama import Options
from rich.table import Table

from llm_benchmark.config import (
    DEFAULT_TIMEOUT,
    SWEEP_NUM_CTX,
    SWEEP_PROMPT,
    get_console,
)
from llm_benchmark.models import (
    SweepConfigResult,
    SweepModelResult,
)
from llm_benchmark.runner import unload_model, warmup_model
from llm_benchmark.system import _get_gpu_info


def _ns_to_sec(ns: int) -> float:
    """Convert nanoseconds to seconds (for raw ollama responses)."""
    return ns / 1_000_000_000


def get_model_layers(model_name: str) -> int | None:
    """Detect the number of layers in a model from Ollama modelinfo.

    Calls ``ollama.show()`` and searches modelinfo keys for any key
    containing ``block_count`` (e.g. ``llama.block_count``).

    Args:
        model_name: Ollama model name (e.g. "llama3:8b").

    Returns:
        Number of layers as int, or None if undetectable.
    """
    try:
        info = ollama.show(model_name)
        modelinfo = getattr(info, "modelinfo", None) or {}
        for key, value in modelinfo.items():
            if "block_count" in key:
                return int(value)
    except Exception:
        pass
    return None


def build_sweep_configs(
    block_count: int | None, has_gpu: bool
) -> list[tuple[int, int]]:
    """Build the cross-product of num_ctx and num_gpu values to sweep.

    Args:
        block_count: Number of model layers (from get_model_layers), or
            None if unknown.
        has_gpu: Whether a GPU is available.

    Returns:
        List of (num_ctx, num_gpu) tuples to test.
    """
    num_gpu_values = [0]
    if block_count is not None and has_gpu:
        num_gpu_values.append(block_count // 2)
        num_gpu_values.append(block_count)

    configs: list[tuple[int, int]] = []
    for ctx in SWEEP_NUM_CTX:
        for gpu in num_gpu_values:
            configs.append((ctx, gpu))
    return configs


def _run_single_config(
    model_name: str, num_ctx: int, num_gpu: int, timeout: int
) -> SweepConfigResult:
    """Run a single sweep configuration and return the result.

    Args:
        model_name: Ollama model name.
        num_ctx: Context window size.
        num_gpu: Number of GPU layers.
        timeout: Timeout in seconds (currently informational; no threading
            wrapper since sweep configs use a short prompt).

    Returns:
        SweepConfigResult with metrics or success=False on error.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": SWEEP_PROMPT}],
            options=Options(num_ctx=num_ctx, num_gpu=num_gpu),
        )

        # Extract timing metrics
        eval_count = getattr(response, "eval_count", 0) or 0
        eval_duration = getattr(response, "eval_duration", 0) or 0
        total_duration = getattr(response, "total_duration", 0) or 0
        prompt_eval_count = getattr(response, "prompt_eval_count", 0) or 0
        prompt_eval_duration = getattr(response, "prompt_eval_duration", 0) or 0

        eval_sec = _ns_to_sec(eval_duration)
        total_sec = _ns_to_sec(total_duration)

        response_ts = eval_count / eval_sec if eval_sec > 0 else 0.0
        total_tokens = eval_count + prompt_eval_count
        total_time = eval_sec + _ns_to_sec(prompt_eval_duration)
        total_ts = total_tokens / total_time if total_time > 0 else 0.0

        return SweepConfigResult(
            model=model_name,
            num_ctx=num_ctx,
            num_gpu=num_gpu,
            response_ts=response_ts,
            total_ts=total_ts,
            eval_count=eval_count,
            total_duration_s=total_sec,
            success=True,
        )

    except Exception as exc:
        return SweepConfigResult(
            model=model_name,
            num_ctx=num_ctx,
            num_gpu=num_gpu,
            response_ts=0.0,
            total_ts=0.0,
            eval_count=0,
            total_duration_s=0.0,
            success=False,
            error=str(exc),
        )


def _display_sweep_table(sweep_result: SweepModelResult) -> None:
    """Display a Rich table of sweep results ranked by response t/s.

    Args:
        sweep_result: Completed sweep results for one model.
    """
    console = get_console()
    table = Table(title=f"Sweep Results: {sweep_result.model}")

    table.add_column("Rank", justify="right", style="dim")
    table.add_column("num_ctx", justify="right")
    table.add_column("num_gpu", justify="right")
    table.add_column("Response (t/s)", justify="right")
    table.add_column("Total (t/s)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Time (s)", justify="right")

    # Sort by response_ts descending; failed configs go last
    sorted_configs = sorted(
        sweep_result.configs,
        key=lambda c: (c.success, c.response_ts),
        reverse=True,
    )

    for rank, cfg in enumerate(sorted_configs, 1):
        is_best = (
            sweep_result.best_config is not None
            and cfg.num_ctx == sweep_result.best_config.num_ctx
            and cfg.num_gpu == sweep_result.best_config.num_gpu
            and cfg.success
        )
        style = "bold green" if is_best else ("red" if not cfg.success else None)

        if cfg.success:
            table.add_row(
                str(rank),
                str(cfg.num_ctx),
                str(cfg.num_gpu),
                f"{cfg.response_ts:.1f}",
                f"{cfg.total_ts:.1f}",
                str(cfg.eval_count),
                f"{cfg.total_duration_s:.2f}",
                style=style,
            )
        else:
            table.add_row(
                str(rank),
                str(cfg.num_ctx),
                str(cfg.num_gpu),
                "FAILED",
                "FAILED",
                "-",
                "-",
                style=style,
            )

    console.print(table)


def run_sweep_for_model(
    model_name: str,
    timeout: int = DEFAULT_TIMEOUT,
    skip_warmup: bool = False,
) -> SweepModelResult:
    """Run a full parameter sweep for one model.

    Tests all combinations of num_ctx and num_gpu, collects results,
    identifies the best configuration, and displays a summary table.

    Note: sweep still uses ollama SDK directly. warmup/unload now need
    a backend instance, so we create one temporarily.

    Args:
        model_name: Ollama model name (e.g. "llama3:8b").
        timeout: Timeout in seconds per config.
        skip_warmup: If True, skip the warmup run.

    Returns:
        SweepModelResult with all config results and best config.
    """
    console = get_console()

    # Create backend for warmup/unload
    from llm_benchmark.backends import create_backend
    backend = create_backend()

    # Detect GPU
    gpu_name, _ = _get_gpu_info()
    has_gpu = gpu_name != "No dedicated GPU"

    # Detect model layers
    block_count = get_model_layers(model_name)
    if block_count is not None:
        console.print(f"  Detected {block_count} layers for {model_name}")
    else:
        console.print(f"  [dim]Could not detect layer count for {model_name}[/dim]")

    # Build config matrix
    configs = build_sweep_configs(block_count, has_gpu)
    total = len(configs)
    console.print(f"  Sweeping {total} configurations...")

    # Warmup once
    if not skip_warmup:
        warmup_model(backend, model_name, timeout)

    # Run each config
    results: list[SweepConfigResult] = []
    for i, (num_ctx, num_gpu) in enumerate(configs):
        console.print(
            f"  Testing config {i + 1}/{total}: "
            f"num_ctx={num_ctx}, num_gpu={num_gpu}..."
        )
        result = _run_single_config(model_name, num_ctx, num_gpu, timeout)
        results.append(result)

        if result.success:
            console.print(
                f"    [green]{result.eval_count} tokens "
                f"@ {result.response_ts:.1f} t/s[/green]"
            )
        else:
            console.print(f"    [red]Failed: {result.error}[/red]")

        # Unload between configs to force clean reload with new options
        unload_model(backend, model_name)

    # Pick best
    successful = [c for c in results if c.success]
    best_config = (
        max(successful, key=lambda c: c.response_ts) if successful else None
    )

    sweep_result = SweepModelResult(
        model=model_name,
        configs=results,
        best_config=best_config,
    )

    # Summary
    if best_config:
        console.print(
            f"  [bold]Best config: num_ctx={best_config.num_ctx}, "
            f"num_gpu={best_config.num_gpu} "
            f"@ {best_config.response_ts:.1f} t/s[/bold]"
        )
    else:
        console.print("  [red]All configurations failed[/red]")

    _display_sweep_table(sweep_result)

    return sweep_result
