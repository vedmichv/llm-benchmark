"""Result exporters: JSON, CSV, and Markdown writers.

All functions create the output directory if it doesn't exist and use
timestamped filenames (benchmark_YYYYMMDD_HHMMSS.{ext}).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from llm_benchmark.models import ModelSummary, SystemInfo, _ns_to_sec


def _ensure_dir(output_dir: str | Path) -> Path:
    """Ensure the output directory exists and return it as a Path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
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
            lines.append(
                f"{idx + 1}. **{status}** `{run.prompt[:50]}...`"
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
