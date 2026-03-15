"""Cross-backend comparison: orchestration, display, and export.

Compares the same models across multiple LLM backends (Ollama, llama.cpp,
LM Studio) and produces bar charts, matrix tables, and export files.

Exports:
    run_comparison: Orchestrate benchmark runs across all backends.
    render_comparison_bar_chart: Bar chart for single-model comparison.
    render_comparison_matrix: Rich matrix table for multi-model comparison.
    export_comparison_json: Write comparison results as JSON.
    export_comparison_markdown: Write comparison results as Markdown.
    match_gguf_to_ollama_name: Match GGUF files to Ollama model names.
    ComparisonResult: Pydantic model for comparison results.
    BackendModelResult: Pydantic model for per-backend per-model result.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from llm_benchmark.backends import create_backend
from llm_benchmark.config import get_console
from llm_benchmark.display import BAR_EMPTY, BAR_FULL, BAR_WIDTH
from llm_benchmark.preflight import run_preflight_checks
from llm_benchmark.runner import benchmark_model, unload_model


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class BackendModelResult(BaseModel):
    """Result for one model on one backend."""

    backend: str
    model: str
    avg_response_ts: float
    avg_prompt_eval_ts: float
    avg_total_ts: float


class ComparisonResult(BaseModel):
    """Unified comparison across all backends."""

    backends: list[str]
    models: list[str]
    results: list[BackendModelResult]
    winner_per_model: dict[str, str]  # model -> backend name
    overall_winner: str
    overall_wins: dict[str, int]  # backend -> win count


# ---------------------------------------------------------------------------
# GGUF Matching
# ---------------------------------------------------------------------------


def match_gguf_to_ollama_name(
    ollama_name: str,
    gguf_files: list[tuple[Path, str]],
) -> Path | None:
    """Find the best GGUF match for an Ollama model name.

    Strategy: normalize both names (lowercase, strip dots/hyphens),
    look for substring match. E.g. "llama3.2:1b" matches
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf".

    Args:
        ollama_name: Ollama model name (e.g. "llama3.2:1b").
        gguf_files: List of (path, display_name) tuples from scan_gguf_files().

    Returns:
        Path to the matching GGUF file, or None if no match.
    """
    # Extract base name and size tag: "llama3.2:1b" -> "llama32" + "1b"
    base = ollama_name.split(":")[0].lower().replace(".", "").replace("-", "")
    tag = ollama_name.split(":")[-1].lower() if ":" in ollama_name else ""

    for path, display_name in gguf_files:
        normalized = (
            path.stem.lower().replace("-", "").replace("_", "").replace(".", "")
        )
        if base in normalized:
            # Check size tag if present
            if tag and tag in normalized:
                return path
            elif not tag:
                return path

    # Fallback: partial match on display_name
    for path, display_name in gguf_files:
        if base in display_name.lower().replace("-", "").replace(" ", ""):
            return path

    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_comparison(
    backends: list,
    prompts: list[str],
    runs_per_prompt: int,
    timeout: int,
    skip_warmup: bool,
    max_retries: int,
    verbose: bool,
    skip_models: list[str],
    skip_checks: bool = False,
) -> ComparisonResult:
    """Run benchmarks on all backends sequentially and collect results.

    Args:
        backends: List of BackendStatus objects for detected backends.
        prompts: Prompts to benchmark.
        runs_per_prompt: Number of runs per prompt.
        timeout: Per-run timeout in seconds.
        skip_warmup: Skip model warmup.
        max_retries: Max retries per run.
        verbose: Stream responses.
        skip_models: Models to skip.
        skip_checks: Skip preflight RAM checks.

    Returns:
        ComparisonResult with aggregated data across all backends.
    """
    console = get_console()

    if len(backends) == 1:
        console.print(
            "[yellow]Only 1 backend detected -- running single-backend benchmark.[/yellow]"
        )

    all_backend_results: dict[str, list] = {}

    for status in backends:
        console.print()
        console.print(f"[bold]Benchmarking with {status.name}...[/bold]")

        backend = create_backend(status.name, port=status.port)

        # Get available models via preflight
        models = run_preflight_checks(
            backend=backend,
            skip_models=skip_models,
            skip_checks=skip_checks,
        )

        # For llama-cpp: try to match GGUF files to Ollama model names
        if status.name == "llama-cpp":
            models = _prepare_llamacpp_models(backend, models, all_backend_results)

        # For non-Ollama backends: match model names to Ollama canonical names
        if status.name != "ollama" and "ollama" in all_backend_results:
            ollama_names = [s.model for s in all_backend_results["ollama"]]
            models = _match_to_ollama_names(models, ollama_names)

        summaries = []
        for model_info in models:
            model_name = model_info["model"]
            canonical_name = model_info.get("_canonical_name", model_name)

            # For llama-cpp, set model path if available
            if status.name == "llama-cpp" and "_gguf_path" in model_info:
                backend._model_path = model_info["_gguf_path"]

            summary = benchmark_model(
                backend=backend,
                model_name=model_name,
                prompts=prompts,
                verbose=verbose,
                timeout=timeout,
                runs_per_prompt=runs_per_prompt,
                skip_warmup=skip_warmup,
                max_retries=max_retries,
            )
            # Use canonical name so models align across backends in the matrix
            if canonical_name != model_name:
                summary.model = canonical_name
            summaries.append(summary)
            unload_model(backend, model_name)

        all_backend_results[status.name] = summaries

    return _build_comparison_result(all_backend_results)


def _match_to_ollama_names(
    models: list[dict], ollama_names: list[str]
) -> list[dict]:
    """Match non-Ollama model names to Ollama canonical names.

    E.g. "llama-3.2-1b-instruct" (LM Studio) -> "llama3.2:1b" (Ollama).
    """
    for model_info in models:
        if "_canonical_name" in model_info:
            continue
        name = model_info["model"].lower().replace("-", "").replace("_", "").replace(".", "")
        for ollama_name in ollama_names:
            parts = ollama_name.split(":")
            base = parts[0].lower().replace(".", "").replace("-", "")
            tag = parts[-1].lower() if len(parts) > 1 else ""
            if base in name and (not tag or tag in name):
                model_info["_canonical_name"] = ollama_name
                break
    return models


def _prepare_llamacpp_models(
    backend,
    models: list[dict],
    existing_results: dict[str, list],
) -> list[dict]:
    """Match GGUF files to Ollama model names for llama-cpp backend.

    If other backends have already run, try to match their model names
    to available GGUF files.

    Args:
        backend: The llama-cpp backend instance.
        models: Models returned by preflight for llama-cpp.
        existing_results: Results from previously-run backends.

    Returns:
        Updated models list with _gguf_path entries where matches found.
    """
    console = get_console()

    # Collect Ollama model names from prior results for reverse matching
    ollama_names: list[str] = []
    for bname, summaries in existing_results.items():
        if bname == "ollama":
            ollama_names = [s.model for s in summaries]

    # If llama-cpp already has a loaded model, use it directly
    # but try to match its name to an Ollama canonical name
    if models and ollama_names:
        matched_models = []
        for model_info in models:
            gguf_name = model_info["model"].lower().replace("-", "").replace("_", "").replace(".", "")
            for ollama_name in ollama_names:
                # Extract base and tag: "llama3.2:1b" -> "llama32", "1b"
                parts = ollama_name.split(":")
                base = parts[0].lower().replace(".", "").replace("-", "")
                tag = parts[-1].lower() if len(parts) > 1 else ""
                if base in gguf_name and (not tag or tag in gguf_name):
                    model_info["_canonical_name"] = ollama_name
                    break
            matched_models.append(model_info)
        return matched_models

    try:
        from llm_benchmark.menu import scan_gguf_files

        hf_cache = Path.home() / ".cache" / "huggingface"
        gguf_files = scan_gguf_files(hf_cache)

        if not gguf_files:
            return models

        # Try to match existing models from other backends
        matched_models = []
        for model_info in models:
            gguf_path = match_gguf_to_ollama_name(
                model_info["model"], gguf_files
            )
            if gguf_path:
                model_info["_gguf_path"] = str(gguf_path)
                matched_models.append(model_info)
            else:
                console.print(
                    f"  [dim]No GGUF match for {model_info['model']} -- skipping[/dim]"
                )

        return matched_models if matched_models else models

    except Exception:
        return models


def _build_comparison_result(
    all_backend_results: dict[str, list],
) -> ComparisonResult:
    """Build a ComparisonResult from backend results.

    Args:
        all_backend_results: Dict mapping backend name to list of ModelSummary.

    Returns:
        ComparisonResult with winners computed.
    """
    backend_names = list(all_backend_results.keys())

    # Collect all unique models
    all_models: set[str] = set()
    for summaries in all_backend_results.values():
        for s in summaries:
            all_models.add(s.model)

    models_list = sorted(all_models)

    # Build flat results list and compute winners
    results: list[BackendModelResult] = []
    winner_per_model: dict[str, str] = {}
    wins: dict[str, int] = {name: 0 for name in backend_names}

    for model in models_list:
        best_rate = -1.0
        best_backend = ""
        backend_count = 0

        for bname in backend_names:
            summary = next(
                (s for s in all_backend_results[bname] if s.model == model),
                None,
            )
            if summary:
                backend_count += 1
                results.append(
                    BackendModelResult(
                        backend=bname,
                        model=model,
                        avg_response_ts=summary.avg_response_ts,
                        avg_prompt_eval_ts=summary.avg_prompt_eval_ts,
                        avg_total_ts=summary.avg_total_ts,
                    )
                )
                if summary.avg_response_ts > best_rate:
                    best_rate = summary.avg_response_ts
                    best_backend = bname

        if best_backend:
            winner_per_model[model] = best_backend
            # Only count wins for models present on 2+ backends
            if backend_count >= 2:
                wins[best_backend] = wins.get(best_backend, 0) + 1

    # Overall winner: backend with most wins
    overall_winner = max(wins, key=lambda k: wins[k]) if wins else ""

    return ComparisonResult(
        backends=backend_names,
        models=models_list,
        results=results,
        winner_per_model=winner_per_model,
        overall_winner=overall_winner,
        overall_wins=wins,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def render_comparison_bar_chart(
    backend_rates: list[tuple[str, float]],
    model_name: str,
) -> None:
    """Bar chart with backends as entries for a single model.

    Args:
        backend_rates: List of (backend_name, rate) tuples.
        model_name: Model name for the chart title.
    """
    if not backend_rates:
        return

    console = get_console()
    max_rate = max(r for _, r in backend_rates)
    max_name_len = max(len(n) for n, _ in backend_rates)

    console.print()
    console.print(f"[bold]{model_name} - Backend Comparison[/bold]")
    console.print()

    for name, rate in backend_rates:
        bar_len = round(BAR_WIDTH * rate / max_rate) if max_rate > 0 else 0
        bar = BAR_FULL * bar_len + BAR_EMPTY * (BAR_WIDTH - bar_len)
        star = " [bold yellow]*[/bold yellow]" if rate == max_rate else ""
        console.print(
            f"  {name:<{max_name_len}}  {bar}  {rate:>6.1f} t/s{star}"
        )


def render_comparison_matrix(
    results: dict[str, list],
) -> None:
    """Rich Table showing models as rows, backends as columns, with winner per row.

    Args:
        results: Dict mapping backend name to list of ModelSummary.
    """
    from rich.table import Table

    console = get_console()
    table = Table(show_header=True, title="Cross-Backend Comparison")
    table.add_column("Model", style="bold")

    backend_names = list(results.keys())
    for name in backend_names:
        table.add_column(name.title(), justify="right")
    table.add_column("Winner", justify="center", style="bold green")

    # Collect all unique models
    all_models: set[str] = set()
    for summaries in results.values():
        for s in summaries:
            all_models.add(s.model)

    wins: dict[str, int] = {name: 0 for name in backend_names}

    for model in sorted(all_models):
        row: list[str] = [model]
        rates: dict[str, float] = {}

        for bname in backend_names:
            summary = next(
                (s for s in results[bname] if s.model == model), None
            )
            if summary:
                rate = summary.avg_response_ts
                rates[bname] = rate
                row.append(f"{rate:.1f} t/s")
            else:
                row.append("--")

        if len(rates) >= 2:
            winner = max(rates, key=lambda k: rates[k])
            wins[winner] += 1
            row.append(winner.title())
        elif len(rates) == 1:
            row.append("--")
        else:
            row.append("--")

        table.add_row(*row)

    console.print(table)

    # Overall recommendation (only models on 2+ backends)
    total = sum(wins.values())
    if total > 0:
        overall_winner = max(wins, key=lambda k: wins[k])
        console.print(
            f"\n  Fastest backend: [bold]{overall_winner.title()}[/bold] "
            f"({wins[overall_winner]}/{total} comparable models)"
        )
    else:
        console.print(
            "\n  [dim]No models found on multiple backends for comparison[/dim]"
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_comparison_json(
    comparison: ComparisonResult,
    system_info,
    output_dir: str | Path = "results",
) -> Path:
    """Write comparison results as JSON.

    Args:
        comparison: ComparisonResult with aggregated data.
        system_info: SystemInfo instance.
        output_dir: Directory for output file.

    Returns:
        Path to the written JSON file.
    """
    from llm_benchmark.exporters import _ensure_dir, _timestamp

    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"comparison_{_timestamp()}.json"

    data = {
        "generated": datetime.now().isoformat(),
        "mode": "comparison",
        "system_info": system_info.model_dump() if system_info else None,
        "backends": comparison.backends,
        "models": comparison.models,
        "results": [r.model_dump() for r in comparison.results],
        "winner_per_model": comparison.winner_per_model,
        "overall_winner": comparison.overall_winner,
        "overall_wins": comparison.overall_wins,
    }

    filepath.write_text(json.dumps(data, indent=2, default=str))
    return filepath


def export_comparison_markdown(
    comparison: ComparisonResult,
    system_info,
    output_dir: str | Path = "results",
) -> Path:
    """Write comparison results as a Markdown report.

    Args:
        comparison: ComparisonResult with aggregated data.
        system_info: SystemInfo instance.
        output_dir: Directory for output file.

    Returns:
        Path to the written Markdown file.
    """
    from llm_benchmark.exporters import _ensure_dir, _timestamp

    out_dir = _ensure_dir(output_dir)
    filepath = out_dir / f"comparison_{_timestamp()}.md"

    lines: list[str] = []
    lines.append("# Cross-Backend Comparison Report\n")
    header_parts = [
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Backends:** {', '.join(comparison.backends)}",
        f"**Models:** {len(comparison.models)}",
    ]
    lines.append(" | ".join(header_parts) + "\n")

    # System info
    if system_info:
        lines.append(
            f"**System:** {system_info.cpu}, {system_info.ram_gb:.1f} GB RAM, "
            f"{system_info.gpu}, {system_info.os_name}\n"
        )

    lines.append("---\n")

    # Matrix table in Markdown
    lines.append("## Results Matrix\n")

    # Header row
    header = "| Model |"
    separator = "|-------|"
    for bname in comparison.backends:
        header += f" {bname.title()} |"
        separator += "--------|"
    header += " Winner |"
    separator += "--------|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for model in comparison.models:
        row = f"| {model} |"
        for bname in comparison.backends:
            result = next(
                (r for r in comparison.results if r.backend == bname and r.model == model),
                None,
            )
            if result:
                row += f" {result.avg_response_ts:.1f} t/s |"
            else:
                row += " -- |"

        winner = comparison.winner_per_model.get(model, "--")
        row += f" **{winner}** |"
        lines.append(row)

    lines.append("")

    # Overall recommendation
    total_models = len(comparison.models)
    winner_wins = comparison.overall_wins.get(comparison.overall_winner, 0)
    lines.append(
        f"Fastest backend: **{comparison.overall_winner}** "
        f"({winner_wins}/{total_models} models)\n"
    )

    # Per-model text bar charts
    lines.append("---\n")
    lines.append("## Per-Model Bar Charts\n")

    for model in comparison.models:
        model_results = [
            r for r in comparison.results if r.model == model
        ]
        if not model_results:
            continue

        lines.append(f"### {model}\n")
        lines.append("```")

        max_rate = max(r.avg_response_ts for r in model_results)
        max_name_len = max(len(r.backend) for r in model_results)

        for r in sorted(model_results, key=lambda x: x.avg_response_ts, reverse=True):
            bar_len = round(BAR_WIDTH * r.avg_response_ts / max_rate) if max_rate > 0 else 0
            bar = BAR_FULL * bar_len + BAR_EMPTY * (BAR_WIDTH - bar_len)
            star = " *" if r.avg_response_ts == max_rate else ""
            lines.append(
                f"  {r.backend:<{max_name_len}}  {bar}  {r.avg_response_ts:>6.1f} t/s{star}"
            )

        lines.append("```\n")

    filepath.write_text("\n".join(lines))
    return filepath
