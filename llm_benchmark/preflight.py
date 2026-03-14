"""Pre-flight checks: backend connectivity, model availability, RAM warnings.

Runs automatically before every benchmark. Connectivity and model checks
are blocking (sys.exit(1) on failure). RAM warnings are advisory only --
per user decision: "don't gatekeep."
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

from llm_benchmark.backends import Backend
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
        except Exception:
            pass
    elif current_os == "Linux":
        try:
            with open("/proc/meminfo") as f:
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
        except Exception:
            pass

    return 0.0


def check_ollama_installed() -> bool:
    """Check if the Ollama binary is installed on the system.

    Uses shutil.which to locate the ollama binary. If not found, offers
    platform-specific installation with an interactive prompt.

    Returns:
        True if Ollama is installed (or was successfully installed), False otherwise.
    """
    if shutil.which("ollama"):
        return True

    console = get_console()
    os_name = platform.system()

    console.print("[yellow]Ollama is not installed.[/yellow]")
    console.print()

    if os_name == "Windows":
        install_cmd = "irm https://ollama.com/install.ps1 | iex"
    else:
        install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"

    console.print(f"  Install command: [cyan]{install_cmd}[/cyan]")
    console.print()

    try:
        answer = input("Install Ollama now? (y/N) ")
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer.strip().lower() == "y":
        subprocess.run(install_cmd, shell=True)
        if shutil.which("ollama"):
            console.print("[green]Ollama installed successfully![/green]")
            return True
        else:
            console.print("[red]Installation may have failed. Ollama binary not found.[/red]")
            return False

    console.print("[dim]Install Ollama manually: https://ollama.com/download[/dim]")
    return False


def check_backend_connectivity(backend: Backend) -> bool:
    """Check if the backend server is reachable.

    Returns True if backend.check_connectivity() succeeds. On failure, prints
    platform-specific instructions for starting the backend.

    Args:
        backend: Backend instance to check connectivity for.

    Returns:
        True if backend is reachable, False otherwise.
    """
    console = get_console()
    try:
        if backend.check_connectivity():
            return True
    except Exception:
        pass

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


def check_available_models(backend: Backend, skip_models: list[str] | None = None) -> list[dict]:
    """Check for available models and filter by skip list.

    Args:
        backend: Backend instance to query for models.
        skip_models: Model names to exclude from results.

    Returns:
        List of available model dicts (from backend.list_models()), filtered
        by skip_models. Empty list if no models are installed.
    """
    console = get_console()
    models = backend.list_models()

    # Filter skip list
    if skip_models:
        models = [m for m in models if m['model'] not in skip_models]

    if not models:
        console.print("[yellow]No models found![/yellow]")
        console.print("  Pull a small model to get started:")
        console.print("  [cyan]ollama pull llama3.2:1b[/cyan]")
        return []

    return models


def check_ram_for_models(models: list[dict]) -> None:
    """Warn if any model may exceed 80% of available system RAM.

    This is advisory only -- execution continues regardless. Uses
    RAM_SAFETY_MULTIPLIER from config to estimate in-memory size
    from the on-disk GGUF size.

    Args:
        models: List of model dicts from backend.list_models().
    """
    console = get_console()
    system_ram_gb = _get_system_ram_gb()

    if system_ram_gb <= 0:
        return  # Can't check without RAM info

    ram_threshold = system_ram_gb * 0.8

    for model in models:
        disk_size_gb = model['size'] / (1024**3)
        estimated_ram_gb = disk_size_gb * RAM_SAFETY_MULTIPLIER

        if estimated_ram_gb > ram_threshold:
            console.print(
                f"[yellow]Warning:[/yellow] {model['model']} may require "
                f"~{estimated_ram_gb:.1f} GB RAM "
                f"({system_ram_gb:.0f} GB available)"
            )


def run_preflight_checks(
    backend: Backend | None = None,
    skip_models: list[str] | None = None,
    skip_checks: bool = False,
) -> list[dict]:
    """Run all pre-flight checks in order.

    Chain: connectivity (blocking) -> models (blocking) -> RAM (warning only).

    Args:
        backend: Backend instance. If None, creates default ollama backend.
        skip_models: Model names to exclude from benchmarking.
        skip_checks: If True, skip the RAM warning check.

    Returns:
        List of available model dicts to benchmark.

    Raises:
        SystemExit: If backend is unreachable or no models are found.
    """
    # Create default backend if not provided
    if backend is None:
        from llm_benchmark.backends import create_backend
        backend = create_backend()

    # 0. Installation check (blocking)
    if not check_ollama_installed():
        sys.exit(1)

    # 1. Connectivity check (blocking)
    if not check_backend_connectivity(backend):
        sys.exit(1)

    # 2. Model availability (blocking)
    models = check_available_models(backend, skip_models=skip_models)
    if not models:
        sys.exit(1)

    # 3. RAM warnings (advisory, skippable)
    if not skip_checks:
        check_ram_for_models(models)

    return models
