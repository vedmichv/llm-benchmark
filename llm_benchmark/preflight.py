"""Pre-flight checks: Ollama connectivity, model availability, RAM warnings.

Runs automatically before every benchmark. Connectivity and model checks
are blocking (sys.exit(1) on failure). RAM warnings are advisory only --
per user decision: "don't gatekeep."
"""

from __future__ import annotations

import platform
import sys

import ollama

from llm_benchmark.config import RAM_SAFETY_MULTIPLIER, get_console


def _get_system_ram_gb() -> float:
    """Detect total system RAM in GB (cross-platform, no psutil).

    Delegates to platform-specific mechanisms. Returns 0.0 if
    detection fails (in which case RAM checks are silently skipped).
    """
    current_os = platform.system()

    if current_os == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip()) / (1024**3)
        except (Exception,):
            pass
    elif current_os == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_kb = int(line.split()[1])
                        return ram_kb / (1024 * 1024)
        except (FileNotFoundError, PermissionError):
            pass
    elif current_os == "Windows":
        try:
            import subprocess

            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [
                    ln.strip()
                    for ln in result.stdout.strip().splitlines()
                    if ln.strip() and not ln.strip().startswith("Total")
                ]
                if lines:
                    return int(lines[0]) / (1024**3)
        except (Exception,):
            pass

    return 0.0


def check_ollama_connectivity() -> bool:
    """Check if the Ollama server is reachable.

    Returns True if ollama.list() succeeds. On failure, prints
    platform-specific instructions for starting Ollama.

    Returns:
        True if Ollama is reachable, False otherwise.
    """
    console = get_console()
    try:
        ollama.list()
        return True
    except Exception:
        os_name = platform.system()
        console.print("[red bold]Cannot connect to Ollama[/red bold]")
        console.print()
        if os_name == "Darwin":
            console.print("  Start Ollama from your Applications folder, or run:")
            console.print("  [cyan]ollama serve[/cyan]")
        elif os_name == "Windows":
            console.print("  Start the Ollama application from the Start menu")
        else:
            console.print("  Start Ollama with:")
            console.print("  [cyan]ollama serve[/cyan]")
        console.print()
        console.print("[dim]Download Ollama: https://ollama.com/download[/dim]")
        return False


def check_available_models(skip_models: list[str] | None = None) -> list:
    """Check for available Ollama models and filter by skip list.

    Args:
        skip_models: Model names to exclude from results.

    Returns:
        List of available model objects (from ollama.list()), filtered
        by skip_models. Empty list if no models are installed.
    """
    console = get_console()
    response = ollama.list()
    models = response.models

    # Filter skip list
    if skip_models:
        models = [m for m in models if m.model not in skip_models]

    if not models:
        console.print("[yellow]No models found![/yellow]")
        console.print("  Pull a small model to get started:")
        console.print("  [cyan]ollama pull llama3.2:1b[/cyan]")
        return []

    return models


def check_ram_for_models(models: list) -> None:
    """Warn if any model may exceed 80% of available system RAM.

    This is advisory only -- execution continues regardless. Uses
    RAM_SAFETY_MULTIPLIER from config to estimate in-memory size
    from the on-disk GGUF size.

    Args:
        models: List of model objects from ollama.list().
    """
    console = get_console()
    system_ram_gb = _get_system_ram_gb()

    if system_ram_gb <= 0:
        return  # Can't check without RAM info

    ram_threshold = system_ram_gb * 0.8

    for model in models:
        disk_size_gb = model.size / (1024**3)
        estimated_ram_gb = disk_size_gb * RAM_SAFETY_MULTIPLIER

        if estimated_ram_gb > ram_threshold:
            console.print(
                f"[yellow]Warning:[/yellow] {model.model} may require "
                f"~{estimated_ram_gb:.1f} GB RAM "
                f"({system_ram_gb:.0f} GB available)"
            )


def run_preflight_checks(
    skip_models: list[str] | None = None,
    skip_checks: bool = False,
) -> list:
    """Run all pre-flight checks in order.

    Chain: connectivity (blocking) -> models (blocking) -> RAM (warning only).

    Args:
        skip_models: Model names to exclude from benchmarking.
        skip_checks: If True, skip the RAM warning check.

    Returns:
        List of available model objects to benchmark.

    Raises:
        SystemExit: If Ollama is unreachable or no models are found.
    """
    # 1. Connectivity check (blocking)
    if not check_ollama_connectivity():
        sys.exit(1)

    # 2. Model availability (blocking)
    models = check_available_models(skip_models=skip_models)
    if not models:
        sys.exit(1)

    # 3. RAM warnings (advisory, skippable)
    if not skip_checks:
        check_ram_for_models(models)

    return models
