"""Results comparison logic for the ``compare`` subcommand.

Migrated from compare_results.py with raw print() calls replaced by
rich Console output (tables, styled text).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rich.table import Table

from llm_benchmark.config import get_console


def load_json_results(file_path: str | Path) -> dict[str, Any]:
    """Load benchmark results from a JSON file.

    Args:
        file_path: Path to the JSON results file.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        SystemExit: If file not found or invalid JSON.
    """
    console = get_console()
    path = Path(file_path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {path}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error: Invalid JSON in {path}: {exc}[/red]")
        sys.exit(1)


def compare_results(
    files: list[str | Path],
    labels: list[str] | None = None,
) -> None:
    """Compare multiple benchmark result files and display a comparison table.

    Args:
        files: Paths to JSON result files (2 or more).
        labels: Optional labels for each file. If None, auto-generated
            from filenames or "Run 1", "Run 2", etc.
    """
    console = get_console()

    if len(files) < 2:
        console.print("[red]Error: Need at least 2 files to compare[/red]")
        return

    # Load all results
    results_list = [load_json_results(f) for f in files]

    # Generate labels
    if labels and len(labels) == len(files):
        run_labels = list(labels)
    else:
        run_labels = []
        for idx, fp in enumerate(files):
            filename = Path(fp).stem
            if "_" in filename:
                parts = filename.split("_")
                if len(parts) >= 2:
                    run_labels.append(f"Run {parts[-2]}_{parts[-1]}")
                    continue
            run_labels.append(f"Run {idx + 1}")

    # Print system info comparison
    console.rule("[bold]Benchmark Comparison[/bold]")
    console.print()

    for results, label in zip(results_list, run_labels):
        sys_info = results.get("system_info")
        if sys_info:
            console.print(f"[bold]{label}:[/bold]")
            console.print(
                f"  GPU: {sys_info.get('gpu', 'N/A')}  |  "
                f"CPU: {sys_info.get('cpu', 'N/A')}  |  "
                f"RAM: {sys_info.get('ram_gb', 0):.0f} GB  |  "
                f"Ollama: {sys_info.get('ollama_version', 'N/A')}"
            )
        else:
            console.print(f"[bold]{label}:[/bold] System info not available")
        console.print()

    # Collect all unique models
    all_models: set[str] = set()
    for results in results_list:
        for model in results.get("models", []):
            all_models.add(model["model"])

    # Build comparison table for each model
    for model_name in sorted(all_models):
        table = Table(title=model_name, show_header=True)
        table.add_column("Metric", style="bold")
        for label in run_labels:
            table.add_column(label, justify="right")
        if len(run_labels) >= 2:
            table.add_column("Difference", justify="right")

        # Collect model stats
        model_stats = []
        for results in results_list:
            model_data = next(
                (m for m in results.get("models", []) if m["model"] == model_name),
                None,
            )
            model_stats.append(
                model_data["averages"] if model_data else None
            )

        # Rows for key metrics
        for metric_key, metric_label in [
            ("response_ts", "Response (t/s)"),
            ("total_ts", "Total (t/s)"),
            ("prompt_eval_ts", "Prompt Eval (t/s)"),
        ]:
            values = [
                s.get(metric_key) if s else None for s in model_stats
            ]
            row = [metric_label]
            for val in values:
                row.append(f"{val:.2f}" if val is not None else "N/A")

            if (
                len(values) >= 2
                and all(v is not None for v in values)
                and values[0] != 0
            ):
                diff = values[-1] - values[0]
                pct = diff / values[0] * 100
                color = "green" if diff > 0 else "red" if diff < 0 else "white"
                row.append(
                    f"[{color}]{diff:+.2f} ({pct:+.1f}%)[/{color}]"
                )
            elif len(run_labels) >= 2:
                row.append("-")

            table.add_row(*row)

        console.print(table)
        console.print()

    # Summary for 2-file comparison
    if len(results_list) == 2:
        faster = slower = unchanged = 0
        for model_name in all_models:
            stats = []
            for results in results_list:
                md = next(
                    (m for m in results.get("models", []) if m["model"] == model_name),
                    None,
                )
                stats.append(
                    md["averages"]["response_ts"] if md else None
                )
            if all(v is not None for v in stats):
                if stats[1] > stats[0]:
                    faster += 1
                elif stats[1] < stats[0]:
                    slower += 1
                else:
                    unchanged += 1

        console.rule("[bold]Summary[/bold]")
        console.print(
            f"Comparing {run_labels[0]} vs {run_labels[1]}:"
        )
        console.print(f"  Faster: [green]{faster}[/green] models")
        console.print(f"  Slower: [red]{slower}[/red] models")
        console.print(f"  Unchanged: {unchanged} models")
