"""Argparse CLI with run/compare/info subcommands.

Entry point for ``python -m llm_benchmark``. Dispatches to the appropriate
handler based on the subcommand.
"""

from __future__ import annotations

import argparse
import sys

from llm_benchmark.config import (
    DEFAULT_RUNS_PER_PROMPT,
    DEFAULT_TIMEOUT,
    get_console,
    set_debug,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="llm_benchmark",
        description="Benchmark your Ollama models",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full stack traces on error",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
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
    subparsers.add_parser("info", help="Show system information")

    return parser


def _handle_run(args: argparse.Namespace) -> int:
    """Handle the 'run' subcommand."""
    from llm_benchmark.exporters import export_csv, export_json, export_markdown
    from llm_benchmark.preflight import run_preflight_checks
    from llm_benchmark.prompts import get_prompts
    from llm_benchmark.runner import benchmark_model, unload_model
    from llm_benchmark.system import format_system_summary, get_system_info

    console = get_console()

    # Pre-flight checks
    models = run_preflight_checks(
        skip_models=args.skip_models,
        skip_checks=args.skip_checks,
    )

    # System summary
    console.print(format_system_summary())
    console.print()

    # Determine prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = get_prompts(args.prompt_set)

    console.print(
        f"Benchmarking {len(models)} model(s) with "
        f"{len(prompts)} prompt(s), {args.runs_per_prompt} run(s) each"
    )
    console.print()

    # Run benchmarks
    system_info = get_system_info()
    all_summaries = []

    for idx, model in enumerate(models):
        model_name = model.model
        console.rule(f"[bold]{model_name}[/bold] ({idx + 1}/{len(models)})")

        try:
            summary = benchmark_model(
                model_name=model_name,
                prompts=prompts,
                verbose=args.verbose,
                timeout=args.timeout,
                runs_per_prompt=args.runs_per_prompt,
            )
            all_summaries.append(summary)

            console.print(
                f"  [green]Average: {summary.avg_response_ts:.1f} t/s[/green]"
            )
        except Exception as exc:
            console.print(f"  [red]Error benchmarking {model_name}: {exc}[/red]")
            if idx < len(models) - 1:
                try:
                    answer = input("Continue with remaining models? [Y/n] ")
                    if answer.strip().lower() == "n":
                        break
                except (EOFError, KeyboardInterrupt):
                    break

        # Unload model after benchmarking
        unload_model(model_name)
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


def _handle_info(_args: argparse.Namespace) -> int:
    """Handle the 'info' subcommand."""
    from llm_benchmark.system import get_system_info

    console = get_console()
    info = get_system_info()

    console.rule("[bold]System Information[/bold]")
    console.print(f"  [bold]CPU:[/bold]     {info.cpu}")
    console.print(f"  [bold]RAM:[/bold]     {info.ram_gb:.1f} GB")
    console.print(f"  [bold]GPU:[/bold]     {info.gpu}")
    if info.gpu_vram_gb is not None:
        console.print(f"  [bold]VRAM:[/bold]    {info.gpu_vram_gb:.1f} GB")
    console.print(f"  [bold]OS:[/bold]      {info.os_name}")
    console.print(f"  [bold]Python:[/bold]  {info.python_version}")
    console.print(f"  [bold]Ollama:[/bold]  {info.ollama_version}")

    return 0


_HANDLERS = {
    "run": _handle_run,
    "compare": _handle_compare,
    "info": _handle_info,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parse arguments and dispatch to handler.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    set_debug(args.debug)

    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)
