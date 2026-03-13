"""Analyze subcommand: sort, filter, and inspect benchmark results.

Provides sorting by throughput metrics, top-N filtering, and optional
per-run detail breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.table import Table

from llm_benchmark.config import get_console

VALID_SORT_KEYS = ["response_ts", "total_ts", "prompt_eval_ts", "load_time"]


def _compute_load_time_avg(model_data: dict) -> float:
    """Compute average load_duration_s from successful runs.

    Returns 0.0 if no run data is available.
    """
    runs = model_data.get("runs", [])
    successful = [r for r in runs if r.get("success", False)]
    if not successful:
        return 0.0
    total = sum(r.get("load_duration_s", 0.0) for r in successful)
    return total / len(successful)


def _get_sort_value(model_data: dict, sort_by: str) -> float:
    """Extract the numeric value to sort by.

    For throughput metrics (response_ts, total_ts, prompt_eval_ts), reads
    from model_data["averages"]. For load_time, computes from run data.
    Returns 0.0 if the key is missing.
    """
    if sort_by == "load_time":
        return _compute_load_time_avg(model_data)
    return model_data.get("averages", {}).get(sort_by, 0.0)


def analyze_results(
    filepath: str,
    sort_by: str = "response_ts",
    top_n: int | None = None,
    ascending: bool = False,
    detail: bool = False,
) -> None:
    """Load, sort, filter, and display benchmark results.

    Args:
        filepath: Path to a JSON results file.
        sort_by: Metric key to sort by. One of VALID_SORT_KEYS.
        top_n: If set, show only the top N models.
        ascending: If True, sort ascending (slowest first). Default descending.
        detail: If True, show per-run breakdown rows under each model.
    """
    console = get_console()

    # Validate sort key
    if sort_by not in VALID_SORT_KEYS:
        console.print(
            f"[red]Error: Invalid sort key '{sort_by}'. "
            f"Valid keys: {', '.join(VALID_SORT_KEYS)}[/red]"
        )
        return

    # Load JSON
    path = Path(filepath)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {path}[/red]")
        return
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error: Invalid JSON in {path}: {exc}[/red]")
        return

    models = data.get("models", [])
    if not models:
        console.print("[yellow]No model data found in results file.[/yellow]")
        return

    # Sort
    models_sorted = sorted(
        models,
        key=lambda m: _get_sort_value(m, sort_by),
        reverse=not ascending,
    )

    # Top-N filter
    if top_n is not None and top_n > 0:
        models_sorted = models_sorted[:top_n]

    # Build table
    filename_stem = Path(filepath).stem
    table = Table(title=f"Results: {filename_stem}")

    table.add_column("#", style="bold", justify="right")
    # Bold the sorted column header
    resp_header = (
        "[bold]Response (t/s)[/bold]"
        if sort_by == "response_ts"
        else "Response (t/s)"
    )
    total_header = (
        "[bold]Total (t/s)[/bold]"
        if sort_by == "total_ts"
        else "Total (t/s)"
    )
    prompt_header = (
        "[bold]Prompt Eval (t/s)[/bold]"
        if sort_by == "prompt_eval_ts"
        else "Prompt Eval (t/s)"
    )

    table.add_column("Model", style="cyan")
    table.add_column(resp_header, justify="right")
    table.add_column(total_header, justify="right")
    table.add_column(prompt_header, justify="right")

    if sort_by == "load_time":
        table.add_column("[bold]Load Time (s)[/bold]", justify="right")

    for rank, model in enumerate(models_sorted, start=1):
        avgs = model.get("averages", {})
        row: list[str] = [
            str(rank),
            model.get("model", "unknown"),
            f"{avgs.get('response_ts', 0):.2f}",
            f"{avgs.get('total_ts', 0):.2f}",
            f"{avgs.get('prompt_eval_ts', 0):.2f}",
        ]
        if sort_by == "load_time":
            row.append(f"{_compute_load_time_avg(model):.3f}")
        table.add_row(*row)

        # Detail mode: per-run breakdown
        if detail:
            for run in model.get("runs", []):
                if not run.get("success", False):
                    continue
                detail_row: list[str] = [
                    "",  # no rank
                    f"  [dim]{run.get('prompt', '')[:40]}[/dim]",
                    f"[dim]{run.get('response_ts', 0):.2f}[/dim]",
                    f"[dim]{run.get('total_duration_s', 0):.2f}[/dim]",
                    f"[dim]{run.get('prompt_eval_ts', 0):.2f}[/dim]",
                ]
                if sort_by == "load_time":
                    detail_row.append(f"[dim]{run.get('load_duration_s', 0):.3f}[/dim]")
                table.add_row(*detail_row)

    console.print(table)
