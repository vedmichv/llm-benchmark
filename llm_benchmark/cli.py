"""Argparse CLI with run/compare/info/analyze subcommands.

Entry point for ``python -m llm_benchmark``. Dispatches to the appropriate
handler based on the subcommand.
"""

from __future__ import annotations

import argparse
import sys

from llm_benchmark.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RUNS_PER_PROMPT,
    DEFAULT_TIMEOUT,
    get_console,
    set_debug,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="llm_benchmark",
        description="Benchmark your local LLM models",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full stack traces on error",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--backend",
        choices=["ollama", "llama-cpp", "lm-studio", "all"],
        default="ollama",
        help="Backend to use (default: ollama, 'all' for cross-backend comparison)",
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Custom port for backend server",
    )
    run_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to GGUF model file (required for llama-cpp)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Stream model responses during benchmark",
    )
    run_parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip RAM/resource pre-flight checks",
    )
    run_parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="Model names to exclude from benchmarking",
    )
    run_parser.add_argument(
        "--prompt-set",
        choices=["small", "medium", "large"],
        default="medium",
        help="Prompt set to use (default: medium)",
    )
    run_parser.add_argument(
        "--prompts",
        nargs="*",
        help="Custom prompts (overrides --prompt-set)",
    )
    run_parser.add_argument(
        "--runs-per-prompt",
        type=int,
        default=DEFAULT_RUNS_PER_PROMPT,
        help=f"Number of runs per prompt (default: {DEFAULT_RUNS_PER_PROMPT})",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-run timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    run_parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Context window size (default: auto-detect per model)",
    )
    run_parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip model warmup before benchmarking",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            f"Max retries per failed run "
            f"(default: {DEFAULT_MAX_RETRIES}, 0 to disable)"
        ),
    )

    # Mutually exclusive group for --concurrent and --sweep
    mode_group = run_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--concurrent",
        type=int,
        nargs="?",
        const=-1,
        default=None,
        help="Run concurrent benchmarks (auto-detect workers if no value given)",
    )
    mode_group.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run parameter sweep (num_ctx/num_gpu) for each model",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare benchmark results"
    )
    compare_parser.add_argument(
        "files",
        nargs="+",
        help="JSON result files to compare",
    )
    compare_parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels for each result file",
    )

    # info subcommand
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument(
        "--backend",
        choices=["ollama", "llama-cpp", "lm-studio"],
        default="ollama",
        help="Backend to use (default: ollama)",
    )

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze and rank benchmark results"
    )
    analyze_parser.add_argument(
        "file",
        help="JSON result file to analyze",
    )
    analyze_parser.add_argument(
        "--sort",
        choices=["response_ts", "total_ts", "prompt_eval_ts", "load_time"],
        default="response_ts",
        help="Sort metric (default: response_ts)",
    )
    analyze_parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show top N models",
    )
    analyze_parser.add_argument(
        "--asc",
        action="store_true",
        help="Sort ascending (slowest first)",
    )
    analyze_parser.add_argument(
        "--detail",
        action="store_true",
        help="Show per-run breakdown",
    )

    return parser


def _print_failure_summary(
    failures: list[tuple[str, str]], backend_name: str
) -> None:
    """Print a summary table of models that failed during benchmarking."""
    from rich.table import Table

    from llm_benchmark.runner import get_known_issue_hint

    console = get_console()
    console.print()
    console.rule("[bold red]Failure Summary[/bold red]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="bold")
    table.add_column("Error")
    table.add_column("Hint", style="dim")

    for model_name, error_msg in failures:
        hint = get_known_issue_hint(backend_name, error_msg) or ""
        # Truncate long error messages for display
        display_error = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
        table.add_row(model_name, display_error, hint)

    console.print(table)
    console.print(
        f"  [yellow]{len(failures)} model(s) skipped due to errors[/yellow]"
    )
    console.print()


def _handle_run(args: argparse.Namespace) -> int:
    """Handle the 'run' subcommand."""
    from llm_benchmark.backends import create_backend
    from llm_benchmark.exporters import export_csv, export_json, export_markdown
    from llm_benchmark.preflight import run_preflight_checks
    from llm_benchmark.prompts import get_prompts
    from llm_benchmark.runner import benchmark_model, unload_model
    from llm_benchmark.system import format_system_summary, get_system_info

    console = get_console()

    # --- Comparison mode (--backend all) ---
    if args.backend == "all":
        from llm_benchmark.backends.detection import detect_backends
        from llm_benchmark.comparison import (
            export_comparison_json,
            export_comparison_markdown,
            render_comparison_bar_chart,
            render_comparison_matrix,
            run_comparison,
        )

        statuses = detect_backends()
        running = [s for s in statuses if s.running]
        if not running:
            console.print(
                "[red]No running backends detected.[/red] "
                "Start at least one backend (e.g. ollama serve) and retry."
            )
            return 1

        prompts = args.prompts or get_prompts(args.prompt_set)
        comparison = run_comparison(
            backends=running,
            prompts=prompts,
            runs_per_prompt=args.runs_per_prompt,
            timeout=args.timeout,
            skip_warmup=args.skip_warmup,
            max_retries=args.max_retries,
            verbose=args.verbose,
            skip_models=args.skip_models,
            skip_checks=args.skip_checks,
        )

        # Display comparison results in terminal
        if len(comparison.models) == 1:
            backend_rates = [
                (r.backend, r.avg_response_ts) for r in comparison.results
            ]
            render_comparison_bar_chart(backend_rates, comparison.models[0])
        else:
            results_by_backend: dict[str, list] = {}
            for r in comparison.results:
                results_by_backend.setdefault(r.backend, []).append(r)
            render_comparison_matrix(results_by_backend)

        # Export results
        first_backend = create_backend(running[0].name)
        system_info = get_system_info(backend=first_backend)
        json_path = export_comparison_json(comparison, system_info)
        md_path = export_comparison_markdown(comparison, system_info)
        console.print("[bold]Comparison results saved:[/bold]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  MD:   {md_path}")
        return 0

    backend = create_backend(args.backend, port=args.port)

    # Store model_path on backend for llama-cpp preflight/auto-start
    if hasattr(args, "model_path") and args.model_path is not None:
        backend._model_path = args.model_path  # type: ignore[attr-defined]

    # Pre-flight checks
    models = run_preflight_checks(
        backend=backend,
        skip_models=args.skip_models,
        skip_checks=args.skip_checks,
    )

    # System summary
    console.print(format_system_summary(backend=backend))
    console.print()

    system_info = get_system_info(backend=backend)

    # --- Sweep mode ---
    if args.sweep:
        from llm_benchmark.exporters import (
            export_sweep_csv,
            export_sweep_json,
            export_sweep_markdown,
        )
        from llm_benchmark.sweep import run_sweep_for_model

        console.print(
            f"[bold]Sweep mode:[/bold] Testing parameter combinations "
            f"for {len(models)} model(s)"
        )
        console.print()

        sweep_results = []
        for idx, model in enumerate(models):
            model_name = model['model']
            console.rule(
                f"[bold]{model_name}[/bold] ({idx + 1}/{len(models)})"
            )
            result = run_sweep_for_model(
                backend=backend,
                model_name=model_name,
                timeout=args.timeout,
                skip_warmup=args.skip_warmup,
            )
            sweep_results.append(result)
            console.print()

        # Bar chart for sweep best configs
        if sweep_results:
            from llm_benchmark.display import render_bar_chart as _render_sweep_chart

            sweep_rankings = sorted(
                [
                    (sr.model, sr.best_config.response_ts)
                    for sr in sweep_results
                    if sr.best_config is not None
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            if sweep_rankings:
                _render_sweep_chart(sweep_rankings)
                console.print()

        if sweep_results:
            json_path = export_sweep_json(sweep_results, system_info)
            csv_path = export_sweep_csv(sweep_results, system_info)
            md_path = export_sweep_markdown(sweep_results, system_info)
            console.print("[bold]Sweep results saved:[/bold]")
            console.print(f"  JSON: {json_path}")
            console.print(f"  CSV:  {csv_path}")
            console.print(f"  MD:   {md_path}")

        return 0

    # --- Concurrent mode ---
    if args.concurrent is not None:
        from llm_benchmark.concurrent import (
            auto_detect_concurrency,
            benchmark_model_concurrent,
        )
        from llm_benchmark.exporters import (
            export_concurrent_csv,
            export_concurrent_json,
            export_concurrent_markdown,
        )

        num_workers = args.concurrent
        if num_workers == -1:
            num_workers = auto_detect_concurrency(
                ram_gb=system_info.ram_gb,
                gpu_vram_gb=system_info.gpu_vram_gb,
            )
        console.print(f"[bold]Concurrent mode:[/bold] {num_workers} workers")
        console.print()

        # Determine prompts
        prompts = args.prompts or get_prompts(args.prompt_set)

        console.print(
            f"Benchmarking {len(models)} model(s) with "
            f"{len(prompts)} prompt(s), {args.runs_per_prompt} run(s) each, "
            f"{num_workers} concurrent workers"
        )
        console.print()

        all_batch_results = []
        for idx, model in enumerate(models):
            model_name = model['model']
            console.rule(
                f"[bold]{model_name}[/bold] ({idx + 1}/{len(models)})"
            )
            batches = benchmark_model_concurrent(
                backend=backend,
                model_name=model_name,
                prompts=prompts,
                num_workers=num_workers,
                runs_per_prompt=args.runs_per_prompt,
                timeout=args.timeout,
                skip_warmup=args.skip_warmup,
                verbose=args.verbose,
            )
            all_batch_results.append(batches)

            # Unload after each model
            unload_model(backend, model_name)
            console.print()

        # Bar chart for concurrent mode
        if all_batch_results:
            from llm_benchmark.display import render_bar_chart as _render_conc_chart

            conc_rankings = []
            for batches in all_batch_results:
                if batches:
                    model_name = batches[0].model
                    avg_agg = (
                        sum(b.aggregate_throughput_ts for b in batches) / len(batches)
                    )
                    conc_rankings.append((model_name, avg_agg))
            conc_rankings.sort(key=lambda x: x[1], reverse=True)
            if conc_rankings:
                _render_conc_chart(conc_rankings, metric_label="t/s (aggregate)")
                console.print()

        if all_batch_results:
            json_path = export_concurrent_json(all_batch_results, system_info)
            csv_path = export_concurrent_csv(all_batch_results, system_info)
            md_path = export_concurrent_markdown(
                all_batch_results, system_info
            )
            console.print("[bold]Concurrent results saved:[/bold]")
            console.print(f"  JSON: {json_path}")
            console.print(f"  CSV:  {csv_path}")
            console.print(f"  MD:   {md_path}")

        return 0

    # --- Standard mode ---
    # Determine prompts
    prompts = args.prompts or get_prompts(args.prompt_set)

    console.print(
        f"Benchmarking {len(models)} model(s) with "
        f"{len(prompts)} prompt(s), {args.runs_per_prompt} run(s) each"
    )
    console.print()

    # Run benchmarks
    all_summaries = []
    failures: list[tuple[str, str]] = []

    try:
        for idx, model in enumerate(models):
            model_name = model['model']
            console.rule(f"[bold]{model_name}[/bold] ({idx + 1}/{len(models)})")

            try:
                summary = benchmark_model(
                    backend=backend,
                    model_name=model_name,
                    prompts=prompts,
                    verbose=args.verbose,
                    timeout=args.timeout,
                    runs_per_prompt=args.runs_per_prompt,
                    skip_warmup=args.skip_warmup,
                    max_retries=args.max_retries,
                    num_ctx=getattr(args, "num_ctx", None),
                )
                all_summaries.append(summary)

                console.print(
                    f"  [green]Average: {summary.avg_response_ts:.1f} t/s[/green]"
                )
            except Exception as exc:
                error_msg = str(exc)
                console.print(
                    f"  [red]Error benchmarking {model_name}: {error_msg}[/red]"
                )
                failures.append((model_name, error_msg))
                # Auto-skip: continue with remaining models

            # Unload model after benchmarking
            unload_model(backend, model_name)
            console.print()
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Benchmark interrupted. Saving partial results...[/yellow]"
        )

    # Failure summary
    if failures:
        _print_failure_summary(failures, backend.name)

    # Bar chart display
    if all_summaries:
        from llm_benchmark.display import render_bar_chart

        rankings = sorted(
            [(s.model, s.avg_response_ts) for s in all_summaries],
            key=lambda x: x[1],
            reverse=True,
        )
        render_bar_chart(rankings)
        console.print()

    # Export results
    if all_summaries:
        json_path = export_json(all_summaries, system_info)
        csv_path = export_csv(all_summaries, system_info)
        md_path = export_markdown(all_summaries, system_info)
        console.print("[bold]Results saved:[/bold]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  CSV:  {csv_path}")
        console.print(f"  MD:   {md_path}")

    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    """Handle the 'compare' subcommand."""
    from llm_benchmark.compare import compare_results

    compare_results(files=args.files, labels=args.labels)
    return 0


def _handle_info(args: argparse.Namespace) -> int:
    """Handle the 'info' subcommand."""
    from llm_benchmark.backends import create_backend
    from llm_benchmark.system import get_backend_inventory, get_system_info

    console = get_console()
    backend_name = getattr(args, "backend", "ollama")
    backend = create_backend(backend_name)
    info = get_system_info(backend=backend)

    console.rule("[bold]System Information[/bold]")
    console.print(f"  [bold]CPU:[/bold]     {info.cpu}")
    console.print(f"  [bold]RAM:[/bold]     {info.ram_gb:.1f} GB")
    console.print(f"  [bold]GPU:[/bold]     {info.gpu}")
    if info.gpu_vram_gb is not None:
        console.print(f"  [bold]VRAM:[/bold]    {info.gpu_vram_gb:.1f} GB")
    console.print(f"  [bold]OS:[/bold]      {info.os_name}")
    console.print(f"  [bold]Python:[/bold]  {info.python_version}")
    console.print(f"  [bold]{info.backend_name.title()}:[/bold]  {info.backend_version}")
    console.print()
    console.print(get_backend_inventory())

    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    """Handle the 'analyze' subcommand."""
    from llm_benchmark.analyze import analyze_results

    analyze_results(
        filepath=args.file,
        sort_by=args.sort,
        top_n=args.top,
        ascending=args.asc,
        detail=args.detail,
    )
    return 0


_HANDLERS = {
    "run": _handle_run,
    "compare": _handle_compare,
    "info": _handle_info,
    "analyze": _handle_analyze,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parse arguments and dispatch to handler.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args_list = argv if argv is not None else sys.argv[1:]

    # --recommend: show all recommended models (force mode)
    if args_list == ["--recommend"]:
        try:
            from llm_benchmark.backends import create_backend
            from llm_benchmark.preflight import run_preflight_checks
            from llm_benchmark.recommend import offer_model_downloads
            from llm_benchmark.system import format_system_summary

            console = get_console()
            backend = create_backend()
            models = run_preflight_checks(backend=backend)
            console.print()
            console.print("[bold]LLM Benchmark — Model Recommender[/bold]")
            console.print()
            console.print(format_system_summary(backend=backend))
            console.print()
            offer_model_downloads(backend, models, force=True)
            return 0
        except KeyboardInterrupt:
            get_console().print("\n[yellow]Interrupted.[/yellow]")
            return 0

    # No-args: launch interactive menu
    if not args_list:
        try:
            from llm_benchmark.backends import create_backend
            from llm_benchmark.menu import (
                run_interactive_menu,
                select_backend_interactive,
            )
            from llm_benchmark.preflight import run_preflight_checks

            backend_name, port, model_path = select_backend_interactive()
            if backend_name == "all":
                # Compare shortcut from backend selector -- skip to comparison
                from llm_benchmark.menu import _mode_compare
                args = _mode_compare("ollama")
                set_debug(False)
                return _handle_run(args)
            backend = create_backend(backend_name, port=port)
            if model_path is not None:
                backend._model_path = model_path  # type: ignore[attr-defined]
            models = run_preflight_checks(backend=backend)
            args = run_interactive_menu(backend, models)
            set_debug(False)
            return _handle_run(args)
        except KeyboardInterrupt:
            get_console().print("\n[yellow]Interrupted.[/yellow]")
            return 0

    parser = _build_parser()
    args = parser.parse_args(args_list)

    set_debug(args.debug)

    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)
