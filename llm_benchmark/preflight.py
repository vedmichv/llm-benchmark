"""Pre-flight checks: backend installation, connectivity, model availability, RAM warnings.

Runs automatically before every benchmark. Installation, connectivity, and
model checks are blocking (sys.exit(1) on failure). RAM warnings are advisory
only -- per user decision: "don't gatekeep."
"""

from __future__ import annotations

import platform
import subprocess
import sys

from llm_benchmark.backends import Backend
from llm_benchmark.backends.detection import (
    auto_start_backend,
    detect_backends,
    get_install_instructions,
)
from llm_benchmark.config import RAM_SAFETY_MULTIPLIER, get_console

# Backend-specific start command hints shown when connectivity fails
_START_HINTS: dict[str, str] = {
    "ollama": "ollama serve",
    "llama-cpp": "llama-server -m <model> --port 8080",
    "lm-studio": "lms server start",
}

# Backend-specific model hints shown when no models found
_MODEL_HINTS: dict[str, str] = {
    "ollama": "ollama pull llama3.2:1b",
    "llama-cpp": "Download a GGUF model from huggingface.co",
    "lm-studio": "Load a model in LM Studio UI or via: lms load <model>",
}


def _get_system_ram_gb() -> float:
    """Detect total system RAM in GB (cross-platform, no psutil).

    Delegates to platform-specific mechanisms. Returns 0.0 if
    detection fails (in which case RAM checks are silently skipped).
    """
    current_os = platform.system()

    if current_os == "Darwin":
        try:
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


def check_backend_installed(backend_name: str) -> bool:
    """Check if the given backend is installed and optionally start it.

    Uses the detection module to check binary availability and port status.
    If installed but not running, prompts user to auto-start.
    If not installed, shows platform-specific install instructions.

    Args:
        backend_name: Backend identifier ('ollama', 'llama-cpp', 'lm-studio').

    Returns:
        True if backend is installed (and running or successfully started).
    """
    console = get_console()

    statuses = detect_backends()
    status = None
    for s in statuses:
        if s.name == backend_name:
            status = s
            break

    if status is None:
        console.print(f"[red]Unknown backend: {backend_name}[/red]")
        return False

    # Installed and running
    if status.installed and status.running:
        return True

    # Installed but not running -- offer to start
    if status.installed and not status.running:
        console.print(
            f"[yellow]{backend_name} is installed but not running.[/yellow]"
        )

        start_hint = _START_HINTS.get(backend_name, f"Start {backend_name}")
        console.print(f"  Start command: [cyan]{start_hint}[/cyan]")
        console.print()

        try:
            answer = input(f"Start {backend_name}? (y/N) ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer.strip().lower() == "y":
            try:
                auto_start_backend(backend_name)
                console.print(f"[green]{backend_name} started successfully![/green]")
                return True
            except Exception as exc:
                console.print(f"[red]Failed to start {backend_name}: {exc}[/red]")
                return False

        return False

    # Not installed -- show install instructions
    instructions = get_install_instructions(backend_name)
    console.print(f"[yellow]{backend_name} is not installed.[/yellow]")
    console.print()
    console.print(f"  Install: [cyan]{instructions}[/cyan]")
    return False


# Keep backward-compatible alias for existing tests and callers
def check_ollama_installed() -> bool:
    """Check if Ollama is installed. Backward-compatible wrapper.

    Returns:
        True if Ollama is installed and available.
    """
    return check_backend_installed("ollama")


def check_backend_connectivity(backend: Backend) -> bool:
    """Check if the backend server is reachable.

    Returns True if backend.check_connectivity() succeeds. On failure, prints
    backend-specific instructions for starting the server.

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

    backend_name = backend.name
    start_hint = _START_HINTS.get(backend_name, f"Start {backend_name}")

    console.print(f"[red bold]Cannot connect to {backend_name}[/red bold]")
    console.print()

    os_name = platform.system()
    if backend_name == "ollama":
        if os_name == "Darwin":
            console.print("  Start Ollama from your Applications folder, or run:")
            console.print(f"  [cyan]{start_hint}[/cyan]")
        elif os_name == "Windows":
            console.print("  Start the Ollama application from the Start menu")
        else:
            console.print(f"  Start with: [cyan]{start_hint}[/cyan]")
    else:
        console.print(f"  Start with: [cyan]{start_hint}[/cyan]")

    console.print()
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
        backend_name = backend.name
        hint = _MODEL_HINTS.get(backend_name, "Check your backend for available models")

        console.print("[yellow]No models found![/yellow]")
        console.print("  Get started:")
        console.print(f"  [cyan]{hint}[/cyan]")
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

    Chain: install (blocking) -> connectivity (blocking) -> models (blocking)
    -> RAM (warning only).

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

    # 0. Installation check (blocking) -- uses detection module for any backend
    if not check_backend_installed(backend.name):
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
