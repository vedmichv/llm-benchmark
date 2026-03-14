"""Interactive menu for students who have never used CLI tools.

When ``python -m llm_benchmark`` is invoked with no arguments, the user
sees a numbered menu instead of an argparse error.

Exports:
    run_interactive_menu: Show menu, return populated argparse.Namespace.
"""

from __future__ import annotations

import argparse
import sys

from llm_benchmark.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RUNS_PER_PROMPT,
    DEFAULT_TIMEOUT,
    get_console,
)


def _prompt_choice(prompt_text: str, valid: set[str]) -> str:
    """Loop on ``input()`` until the user enters a value in *valid*.

    Handles ``EOFError`` and ``KeyboardInterrupt`` with a clean exit.
    """
    console = get_console()
    while True:
        try:
            choice = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            sys.exit(0)
        if choice in valid:
            return choice
        console.print("  Please enter " + ", ".join(sorted(valid)))


def _prompt_int(prompt_text: str, default: int) -> int:
    """Ask for an integer with a default value."""
    console = get_console()
    while True:
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            sys.exit(0)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            console.print(f"  Please enter a number (default: {default})")


def _build_namespace(
    *,
    prompt_set: str = "medium",
    prompts: list[str] | None = None,
    runs_per_prompt: int = DEFAULT_RUNS_PER_PROMPT,
    skip_warmup: bool = False,
    skip_models: list[str] | None = None,
) -> argparse.Namespace:
    """Return a fully populated ``argparse.Namespace`` for ``_handle_run``."""
    return argparse.Namespace(
        command="run",
        verbose=False,
        skip_checks=True,  # preflight already ran before menu
        skip_models=skip_models or [],
        prompt_set=prompt_set,
        prompts=prompts,
        runs_per_prompt=runs_per_prompt,
        timeout=DEFAULT_TIMEOUT,
        skip_warmup=skip_warmup,
        max_retries=DEFAULT_MAX_RETRIES,
        concurrent=None,
        sweep=False,
        debug=False,
        num_ctx=None,
    )


def _mode_quick(models: list) -> argparse.Namespace:
    """Quick test: smallest model, 1 short prompt, 1 run, skip warmup."""
    console = get_console()
    sorted_models = sorted(models, key=lambda m: m['size'])
    smallest = sorted_models[0]
    smallest_name = smallest['model']
    skip = [m['model'] for m in models if m['model'] != smallest_name]
    console.print(f"  Quick test with [bold]{smallest_name}[/bold]")
    return _build_namespace(
        prompt_set="small",
        prompts=["Write a one-sentence summary of what a CPU does."],
        runs_per_prompt=1,
        skip_warmup=True,
        skip_models=skip,
    )


def _mode_standard() -> argparse.Namespace:
    """Standard benchmark: medium prompts, 2 runs, warmup enabled."""
    return _build_namespace(
        prompt_set="medium",
        runs_per_prompt=2,
        skip_warmup=False,
    )


def _mode_full() -> argparse.Namespace:
    """Full benchmark: large prompts, 3 runs, warmup enabled."""
    return _build_namespace(
        prompt_set="large",
        runs_per_prompt=3,
        skip_warmup=False,
    )


def _mode_custom(models: list) -> argparse.Namespace:
    """Custom mode: user picks prompt set, runs, and models to skip."""
    console = get_console()

    # Prompt set selection
    console.print()
    console.print("  Prompt sets:")
    console.print("    1. Small  (3 prompts)")
    console.print("    2. Medium (5 prompts)")
    console.print("    3. Large  (11 prompts)")
    ps_choice = _prompt_choice("  Select prompt set [1-3]: ", {"1", "2", "3"})
    ps_map = {"1": "small", "2": "medium", "3": "large"}
    prompt_set = ps_map[ps_choice]

    # Runs per prompt
    runs = _prompt_int(
        f"  Runs per prompt (default {DEFAULT_RUNS_PER_PROMPT}): ",
        DEFAULT_RUNS_PER_PROMPT,
    )

    # Model selection
    console.print()
    console.print("  Available models:")
    for i, m in enumerate(models, 1):
        size_gb = m['size'] / (1024 ** 3)
        console.print(f"    {i}. {m['model']}  ({size_gb:.1f} GB)")

    skip_models: list[str] = []
    try:
        raw = input("  Skip models (e.g. 3,4) or Enter for all: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        sys.exit(0)

    if raw:
        for token in raw.replace(" ", "").split(","):
            try:
                idx = int(token) - 1
                if 0 <= idx < len(models):
                    skip_models.append(models[idx]['model'])
            except ValueError:
                pass

    return _build_namespace(
        prompt_set=prompt_set,
        runs_per_prompt=runs,
        skip_warmup=False,
        skip_models=skip_models,
    )


def run_interactive_menu(backend, models: list) -> argparse.Namespace:
    """Display the interactive menu and return a populated Namespace.

    Parameters
    ----------
    backend:
        Backend instance for system summary and model downloads.
    models:
        List of model dicts returned by ``run_preflight_checks``.

    Returns
    -------
    argparse.Namespace
        Arguments ready for ``_handle_run``.
    """
    from llm_benchmark.system import format_system_summary

    console = get_console()

    # Header
    console.print()
    console.print("[bold]LLM Benchmark[/bold]")
    console.print()
    console.print(format_system_summary(backend=backend))
    console.print()

    # Offer model downloads if RAM allows recommendations
    from llm_benchmark.recommend import offer_model_downloads

    models = offer_model_downloads(backend, models)
    console.print()

    # Menu
    console.print("  1. Quick test (~30 seconds)")
    console.print("  2. Standard benchmark")
    console.print("  3. Full benchmark")
    console.print("  4. Custom")
    console.print()

    choice = _prompt_choice("Select mode [1-4]: ", {"1", "2", "3", "4"})
    console.print()

    if choice == "1":
        return _mode_quick(models)
    if choice == "2":
        return _mode_standard()
    if choice == "3":
        return _mode_full()
    return _mode_custom(models)
