"""Result exporters: JSON, CSV, and Markdown writers.

All functions create the output directory if it doesn't exist and use
timestamped filenames (benchmark_YYYYMMDD_HHMMSS.{ext} for standard/concurrent,
sweep_YYYYMMDD_HHMMSS.{ext} for sweep results).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from llm_benchmark.models import ModelSummary, SystemInfo, _ns_to_sec


_RESULTS_GITIGNORE = """\
# Benchmark result files -- do not commit
*.json
*.csv
*.md
!.gitignore
"""


def _ensure_dir(output_dir: str | Path) -> Path:
    """Ensure the output directory exists and return it as a Path.

    If the directory name is 'results', auto-creates a .gitignore to
    prevent accidental commits of benchmark output files.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Auto-create .gitignore for results/ directories
    if path.name == "results":
        gitignore = path / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_RESULTS_GITIGNORE)

    return path


def _timestamp() -> str:
    """Return a timestamp string for filenames: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _result_to_dict(result) -> dict:
    """Convert a BenchmarkResult to a serializable dict."""
    d = {
        "model": result.model,
        "prompt": result.prompt,
        "success": result.success,
        "error": result.error,
        "prompt_cached": result.prompt_cached,
    }
    if result.response:
        r = result.response
        d["prompt_eval_count"] = r.prompt_eval_count
        d["eval_count"] = r.eval_count
        d["prompt_eval_duration_s"] = round(_ns_to_sec(r.prompt_eval_duration), 4)
        d["eval_duration_s"] = round(_ns_to_sec(r.eval_duration), 4)
        d["total_duration_s"] = round(_ns_to_sec(r.total_duration), 4)
        d["load_duration_s"] = round(_ns_to_sec(r.load_duration), 4)
        # Compute per-run rates
        d["prompt_eval_ts"] = round(
            r.prompt_eval_count / _ns_to_sec(r.prompt_eval_duration), 2
        ) if r.prompt_eval_duration > 0 else 0
        d["response_ts"] = round(
            r.eval_count / _ns_to_sec(r.eval_duration), 2
        ) if r.eval_duration > 0 else 0
    return d


# ---------------------------------------------------------------------------
# Standard exports
# ---------------------------------------------------------------------------

def export_json(
    results: list[ModelSummary],
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write benchmark results as JSON.

    Args:
        results: List of ModelSummary objects.
        system_info: Optional system information to include.
        output_dir: Directory for output file (created if needed).

    Returns:
        Path to the written JSON file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"benchmark_{_timestamp()}.json"

    data = {
        "generated": datetime.now().isoformat(),
        "mode": "standard",
        "system_info": system_info.model_dump() if system_info else None,
        "models": [],
    }

    for summary in results:
        model_data = {
            "model": summary.model,
            "averages": {
                "prompt_eval_ts": round(summary.avg_prompt_eval_ts, 2),
                "response_ts": round(summary.avg_response_ts, 2),
                "total_ts": round(summary.avg_total_ts, 2),
            },
            "runs": [_result_to_dict(r) for r in summary.results],
        }
        data["models"].append(model_data)

    filepath.write_text(json.dumps(data, indent=2, default=str))
    return filepath


def export_csv(
    results: list[ModelSummary],
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write benchmark results as CSV with one row per model per prompt.

    Args:
        results: List of ModelSummary objects.
        system_info: Optional system information (written as header rows).
        output_dir: Directory for output file (created if needed).

    Returns:
        Path to the written CSV file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"benchmark_{_timestamp()}.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # System info header rows
        if system_info:
            writer.writerow(["System Info"])
            writer.writerow(["CPU", system_info.cpu])
            writer.writerow(["RAM", f"{system_info.ram_gb:.1f} GB"])
            writer.writerow(["GPU", system_info.gpu])
            writer.writerow(["OS", system_info.os_name])
            writer.writerow(["Ollama", system_info.ollama_version])
            writer.writerow([])

        # Column headers
        writer.writerow([
            "Model",
            "Prompt",
            "Success",
            "Cached",
            "Prompt Tokens",
            "Response Tokens",
            "Prompt Eval (t/s)",
            "Response (t/s)",
            "Total Time (s)",
            "Error",
        ])

        for summary in results:
            for run in summary.results:
                if run.success and run.response:
                    r = run.response
                    prompt_ts = (
                        r.prompt_eval_count / _ns_to_sec(r.prompt_eval_duration)
                        if r.prompt_eval_duration > 0
                        else 0
                    )
                    response_ts = (
                        r.eval_count / _ns_to_sec(r.eval_duration)
                        if r.eval_duration > 0
                        else 0
                    )
                    writer.writerow([
                        run.model,
                        run.prompt[:60] + ("..." if len(run.prompt) > 60 else ""),
                        "Yes",
                        "Yes" if run.prompt_cached else "No",
                        r.prompt_eval_count,
                        r.eval_count,
                        f"{prompt_ts:.2f}",
                        f"{response_ts:.2f}",
                        f"{_ns_to_sec(r.total_duration):.2f}",
                        "",
                    ])
                else:
                    error_msg = (run.error or "")[:100]
                    writer.writerow([
                        run.model,
                        run.prompt[:60] + ("..." if len(run.prompt) > 60 else ""),
                        "No",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        error_msg,
                    ])

    return filepath


def export_markdown(
    results: list[ModelSummary],
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write benchmark results as a Markdown report.

    Args:
        results: List of ModelSummary objects.
        system_info: Optional system information to include.
        output_dir: Directory for output file (created if needed).

    Returns:
        Path to the written Markdown file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"benchmark_{_timestamp()}.md"

    lines: list[str] = []
    lines.append("# LLM Benchmark Results\n")
    lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # System information
    if system_info:
        lines.append("## System Information\n")
        lines.append(f"- **CPU:** {system_info.cpu}")
        lines.append(f"- **RAM:** {system_info.ram_gb:.1f} GB")
        lines.append(f"- **GPU:** {system_info.gpu}")
        if system_info.gpu_vram_gb is not None:
            lines.append(f"- **GPU VRAM:** {system_info.gpu_vram_gb:.1f} GB")
        lines.append(f"- **OS:** {system_info.os_name}")
        lines.append(f"- **Python:** {system_info.python_version}")
        lines.append(f"- **Ollama:** {system_info.ollama_version}")
        lines.append("")

    lines.append("---\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append(
        "| Model | Prompt Eval (t/s) | Response (t/s) | Total (t/s) |"
    )
    lines.append(
        "|-------|-------------------|----------------|-------------|"
    )
    for s in results:
        lines.append(
            f"| {s.model} | {s.avg_prompt_eval_ts:.2f} "
            f"| {s.avg_response_ts:.2f} | {s.avg_total_ts:.2f} |"
        )
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results\n")
    for s in results:
        lines.append(f"### {s.model}\n")
        for idx, run in enumerate(s.results):
            status = "OK" if run.success else "FAIL"
            cached_indicator = " [cached]" if run.prompt_cached else ""
            lines.append(
                f"{idx + 1}. **{status}**{cached_indicator} `{run.prompt[:50]}...`"
            )
            if run.success and run.response:
                r = run.response
                resp_ts = (
                    r.eval_count / _ns_to_sec(r.eval_duration)
                    if r.eval_duration > 0
                    else 0
                )
                lines.append(
                    f"   - {r.eval_count} tokens @ {resp_ts:.1f} t/s, "
                    f"{_ns_to_sec(r.total_duration):.2f}s total"
                )
            else:
                lines.append(f"   - Error: {run.error}")
        lines.append("")

    filepath.write_text("\n".join(lines))
    return filepath


# ---------------------------------------------------------------------------
# Concurrent exports
# ---------------------------------------------------------------------------

def export_concurrent_json(
    batch_results: list[list],
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write concurrent benchmark results as JSON.

    Args:
        batch_results: List of lists of ConcurrentBatchResult (one inner list
            per model).
        system_info: Optional system information to include.
        output_dir: Directory for output file (created if needed).

    Returns:
        Path to the written JSON file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"benchmark_{_timestamp()}.json"

    # Determine worker count from first batch
    num_workers = 0
    for model_batches in batch_results:
        if model_batches:
            num_workers = model_batches[0].num_workers
            break

    data = {
        "generated": datetime.now().isoformat(),
        "mode": "concurrent",
        "concurrent_workers": num_workers,
        "system_info": system_info.model_dump() if system_info else None,
        "models": [],
    }

    for model_batches in batch_results:
        if not model_batches:
            continue
        model_name = model_batches[0].model
        batches_data = []
        for batch in model_batches:
            batches_data.append({
                "prompt": batch.prompt,
                "num_workers": batch.num_workers,
                "wall_time_s": round(batch.wall_time_s, 4),
                "aggregate_throughput_ts": round(batch.aggregate_throughput_ts, 2),
                "avg_request_throughput_ts": round(batch.avg_request_throughput_ts, 2),
                "results": [_result_to_dict(r) for r in batch.results],
            })
        data["models"].append({
            "model": model_name,
            "batches": batches_data,
        })

    filepath.write_text(json.dumps(data, indent=2, default=str))
    return filepath


def export_concurrent_csv(
    batch_results: list[list],
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write concurrent benchmark results as CSV.

    Args:
        batch_results: List of lists of ConcurrentBatchResult.
        system_info: Optional system information.
        output_dir: Directory for output file.

    Returns:
        Path to the written CSV file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"benchmark_{_timestamp()}.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # System info header rows
        if system_info:
            writer.writerow(["System Info"])
            writer.writerow(["CPU", system_info.cpu])
            writer.writerow(["RAM", f"{system_info.ram_gb:.1f} GB"])
            writer.writerow(["GPU", system_info.gpu])
            writer.writerow(["OS", system_info.os_name])
            writer.writerow(["Ollama", system_info.ollama_version])
            writer.writerow(["Mode", "concurrent"])
            writer.writerow([])

        # Column headers
        writer.writerow([
            "Model",
            "Prompt",
            "Workers",
            "Wall Time (s)",
            "Aggregate (t/s)",
            "Avg Request (t/s)",
        ])

        for model_batches in batch_results:
            for batch in model_batches:
                writer.writerow([
                    batch.model,
                    batch.prompt[:60] + ("..." if len(batch.prompt) > 60 else ""),
                    batch.num_workers,
                    f"{batch.wall_time_s:.2f}",
                    f"{batch.aggregate_throughput_ts:.2f}",
                    f"{batch.avg_request_throughput_ts:.2f}",
                ])

    return filepath


def export_concurrent_markdown(
    batch_results: list[list],
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write concurrent benchmark results as a Markdown report.

    Args:
        batch_results: List of lists of ConcurrentBatchResult.
        system_info: Optional system information.
        output_dir: Directory for output file.

    Returns:
        Path to the written Markdown file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"benchmark_{_timestamp()}.md"

    # Determine worker count
    num_workers = 0
    for model_batches in batch_results:
        if model_batches:
            num_workers = model_batches[0].num_workers
            break

    lines: list[str] = []
    lines.append("# LLM Concurrent Benchmark Results\n")
    lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    lines.append(f"**Mode:** Concurrent ({num_workers} workers)\n")

    # System information
    if system_info:
        lines.append("## System Information\n")
        lines.append(f"- **CPU:** {system_info.cpu}")
        lines.append(f"- **RAM:** {system_info.ram_gb:.1f} GB")
        lines.append(f"- **GPU:** {system_info.gpu}")
        if system_info.gpu_vram_gb is not None:
            lines.append(f"- **GPU VRAM:** {system_info.gpu_vram_gb:.1f} GB")
        lines.append(f"- **OS:** {system_info.os_name}")
        lines.append(f"- **Python:** {system_info.python_version}")
        lines.append(f"- **Ollama:** {system_info.ollama_version}")
        lines.append("")

    lines.append("---\n")

    # Summary table
    lines.append("## Results\n")
    lines.append(
        "| Model | Prompt | Wall Time (s) | Aggregate (t/s) | Avg Request (t/s) |"
    )
    lines.append(
        "|-------|--------|---------------|-----------------|-------------------|"
    )

    for model_batches in batch_results:
        for batch in model_batches:
            prompt_short = batch.prompt[:50] + ("..." if len(batch.prompt) > 50 else "")
            lines.append(
                f"| {batch.model} | {prompt_short} "
                f"| {batch.wall_time_s:.2f} "
                f"| {batch.aggregate_throughput_ts:.2f} "
                f"| {batch.avg_request_throughput_ts:.2f} |"
            )
    lines.append("")

    lines.append(
        "> **Note:** Ollama may queue requests. Aggregate throughput shows "
        "total work per wall-clock second."
    )
    lines.append("")

    filepath.write_text("\n".join(lines))
    return filepath


# ---------------------------------------------------------------------------
# Sweep exports
# ---------------------------------------------------------------------------

def export_sweep_json(
    sweep_results: list,
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write parameter sweep results as JSON.

    Args:
        sweep_results: List of SweepModelResult objects.
        system_info: Optional system information.
        output_dir: Directory for output file.

    Returns:
        Path to the written JSON file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"sweep_{_timestamp()}.json"

    data = {
        "generated": datetime.now().isoformat(),
        "mode": "sweep",
        "system_info": system_info.model_dump() if system_info else None,
        "sweeps": [],
    }

    for sr in sweep_results:
        best = None
        if sr.best_config is not None:
            best = {
                "num_ctx": sr.best_config.num_ctx,
                "num_gpu": sr.best_config.num_gpu,
                "response_ts": round(sr.best_config.response_ts, 2),
                "total_ts": round(sr.best_config.total_ts, 2),
                "eval_count": sr.best_config.eval_count,
                "total_duration_s": round(sr.best_config.total_duration_s, 4),
            }

        configs_data = []
        for cfg in sr.configs:
            configs_data.append({
                "num_ctx": cfg.num_ctx,
                "num_gpu": cfg.num_gpu,
                "response_ts": round(cfg.response_ts, 2),
                "total_ts": round(cfg.total_ts, 2),
                "eval_count": cfg.eval_count,
                "total_duration_s": round(cfg.total_duration_s, 4),
                "success": cfg.success,
                "error": cfg.error,
            })

        data["sweeps"].append({
            "model": sr.model,
            "best_config": best,
            "configs": configs_data,
        })

    filepath.write_text(json.dumps(data, indent=2, default=str))
    return filepath


def export_sweep_csv(
    sweep_results: list,
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write parameter sweep results as CSV.

    Args:
        sweep_results: List of SweepModelResult objects.
        system_info: Optional system information.
        output_dir: Directory for output file.

    Returns:
        Path to the written CSV file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"sweep_{_timestamp()}.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # System info header rows
        if system_info:
            writer.writerow(["System Info"])
            writer.writerow(["CPU", system_info.cpu])
            writer.writerow(["RAM", f"{system_info.ram_gb:.1f} GB"])
            writer.writerow(["GPU", system_info.gpu])
            writer.writerow(["OS", system_info.os_name])
            writer.writerow(["Ollama", system_info.ollama_version])
            writer.writerow(["Mode", "sweep"])
            writer.writerow([])

        # Column headers
        writer.writerow([
            "Model",
            "num_ctx",
            "num_gpu",
            "Response (t/s)",
            "Total (t/s)",
            "Tokens",
            "Time (s)",
            "Best?",
        ])

        for sr in sweep_results:
            for cfg in sr.configs:
                is_best = (
                    sr.best_config is not None
                    and cfg.num_ctx == sr.best_config.num_ctx
                    and cfg.num_gpu == sr.best_config.num_gpu
                    and cfg.success
                )
                if cfg.success:
                    writer.writerow([
                        sr.model,
                        cfg.num_ctx,
                        cfg.num_gpu,
                        f"{cfg.response_ts:.2f}",
                        f"{cfg.total_ts:.2f}",
                        cfg.eval_count,
                        f"{cfg.total_duration_s:.2f}",
                        "Yes" if is_best else "No",
                    ])
                else:
                    writer.writerow([
                        sr.model,
                        cfg.num_ctx,
                        cfg.num_gpu,
                        "FAILED",
                        "FAILED",
                        "-",
                        "-",
                        "No",
                    ])

    return filepath


def export_sweep_markdown(
    sweep_results: list,
    system_info: SystemInfo | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Write parameter sweep results as a Markdown report.

    Args:
        sweep_results: List of SweepModelResult objects.
        system_info: Optional system information.
        output_dir: Directory for output file.

    Returns:
        Path to the written Markdown file.
    """
    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"sweep_{_timestamp()}.md"

    lines: list[str] = []
    lines.append("# LLM Parameter Sweep Results\n")
    lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    lines.append("**Mode:** Parameter Sweep\n")

    # System information
    if system_info:
        lines.append("## System Information\n")
        lines.append(f"- **CPU:** {system_info.cpu}")
        lines.append(f"- **RAM:** {system_info.ram_gb:.1f} GB")
        lines.append(f"- **GPU:** {system_info.gpu}")
        if system_info.gpu_vram_gb is not None:
            lines.append(f"- **GPU VRAM:** {system_info.gpu_vram_gb:.1f} GB")
        lines.append(f"- **OS:** {system_info.os_name}")
        lines.append(f"- **Python:** {system_info.python_version}")
        lines.append(f"- **Ollama:** {system_info.ollama_version}")
        lines.append("")

    lines.append("---\n")

    # Per-model sweep tables
    for sr in sweep_results:
        lines.append(f"## {sr.model}\n")

        # Sort configs by response_ts descending (successful first)
        sorted_configs = sorted(
            sr.configs,
            key=lambda c: (c.success, c.response_ts),
            reverse=True,
        )

        lines.append(
            "| Rank | num_ctx | num_gpu | Response (t/s) | Total (t/s) "
            "| Tokens | Time (s) | Note |"
        )
        lines.append(
            "|------|---------|---------|----------------|-------------|"
            "--------|----------|------|"
        )

        for rank, cfg in enumerate(sorted_configs, 1):
            is_best = (
                sr.best_config is not None
                and cfg.num_ctx == sr.best_config.num_ctx
                and cfg.num_gpu == sr.best_config.num_gpu
                and cfg.success
            )
            note = "**Recommended**" if is_best else ""

            if cfg.success:
                if is_best:
                    lines.append(
                        f"| **{rank}** | **{cfg.num_ctx}** | **{cfg.num_gpu}** "
                        f"| **{cfg.response_ts:.1f}** | **{cfg.total_ts:.1f}** "
                        f"| **{cfg.eval_count}** | **{cfg.total_duration_s:.2f}** "
                        f"| {note} |"
                    )
                else:
                    lines.append(
                        f"| {rank} | {cfg.num_ctx} | {cfg.num_gpu} "
                        f"| {cfg.response_ts:.1f} | {cfg.total_ts:.1f} "
                        f"| {cfg.eval_count} | {cfg.total_duration_s:.2f} "
                        f"| {note} |"
                    )
            else:
                lines.append(
                    f"| {rank} | {cfg.num_ctx} | {cfg.num_gpu} "
                    f"| FAILED | FAILED | - | - | {cfg.error or ''} |"
                )

        lines.append("")

    filepath.write_text("\n".join(lines))
    return filepath
