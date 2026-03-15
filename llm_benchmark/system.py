"""Cross-platform hardware and system information collection.

Detects CPU, RAM, GPU, OS, Python version, and backend version without
requiring psutil or any C-extension dependencies. Uses platform module
and subprocess calls to platform-specific tools (sysctl on macOS,
/proc on Linux, wmic on Windows).
"""

from __future__ import annotations

import platform
import subprocess
from typing import TYPE_CHECKING

from llm_benchmark.models import SystemInfo

if TYPE_CHECKING:
    from llm_benchmark.backends import Backend


def get_ollama_version() -> str:
    """Get the installed Ollama version by running ``ollama --version``.

    Returns:
        Version string (e.g. "0.6.1") or "Unknown" if detection fails.
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            # Normalize various output formats
            version = version.replace("ollama version is ", "")
            version = version.replace("ollama version ", "")
            return version
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "Unknown"


def _get_cpu_model() -> str:
    """Detect CPU model string, cross-platform."""
    current_os = platform.system()

    if current_os == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    elif current_os == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except (FileNotFoundError, PermissionError):
            pass
    elif current_os == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [
                    ln.strip()
                    for ln in result.stdout.strip().splitlines()
                    if ln.strip() and ln.strip() != "Name"
                ]
                if lines:
                    return lines[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return platform.processor() or "Unknown"


def _get_ram_gb() -> float:
    """Detect total system RAM in GB, cross-platform."""
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
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
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
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
            pass

    return 0.0


def _get_gpu_info() -> tuple[str, float | None]:
    """Detect GPU model and VRAM, cross-platform.

    Returns:
        Tuple of (gpu_model, gpu_vram_gb). GPU model defaults to
        "No dedicated GPU" if undetectable. VRAM may be None.
    """
    current_os = platform.system()

    # Try nvidia-smi first (Linux, Windows with NVIDIA)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu_model = parts[0].strip()
            vram_gb = None
            if len(parts) >= 2:
                vram_str = parts[1].strip().split()[0]
                vram_gb = float(vram_str) / 1024  # MiB -> GB
            return gpu_model, vram_gb
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # macOS: Apple Silicon has unified memory (GPU shares system RAM)
    if current_os == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                # Apple Silicon: GPU is integrated
                chip = result.stdout.strip()
                return f"{chip} (integrated GPU)", None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return "No dedicated GPU", None


def get_system_info(backend: Backend | None = None) -> SystemInfo:
    """Collect comprehensive system information.

    Detects CPU, RAM, GPU, OS, Python version, and backend info
    across macOS, Linux, and Windows without psutil.

    Args:
        backend: Optional Backend instance. If provided, uses backend.name
            and backend.version. If None, falls back to "ollama" and
            get_ollama_version() for backward compatibility.

    Returns:
        SystemInfo with all detected fields populated.
    """
    cpu = _get_cpu_model()
    ram_gb = round(_get_ram_gb(), 1)
    gpu, gpu_vram = _get_gpu_info()

    if backend is not None:
        backend_name = backend.name
        backend_version = backend.version
    else:
        backend_name = "ollama"
        backend_version = get_ollama_version()

    return SystemInfo(
        cpu=cpu,
        ram_gb=ram_gb,
        gpu=gpu,
        gpu_vram_gb=round(gpu_vram, 1) if gpu_vram is not None else None,
        os_name=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        backend_name=backend_name,
        backend_version=backend_version,
    )


def format_system_summary(backend: Backend | None = None) -> str:
    """Return a compact one-line summary for display before benchmarks.

    Uses rich markup for styled output. Example:
        "Apple M2 Pro | 16.0 GB RAM | Apple M2 Pro (integrated GPU) | Ollama 0.6.1"

    Includes a backends section showing all detected backends with status.

    Args:
        backend: Optional Backend instance for version info.

    Returns:
        Formatted string with rich markup.
    """
    info = get_system_info(backend=backend)
    parts = [
        f"[bold]{info.cpu}[/bold]",
        f"{info.ram_gb:.0f} GB RAM",
    ]

    if info.gpu_vram_gb is not None:
        parts.append(f"{info.gpu} ({info.gpu_vram_gb:.0f} GB VRAM)")
    else:
        parts.append(info.gpu)

    parts.append(f"{info.backend_name.title()} {info.backend_version}")

    summary_line = " | ".join(parts)

    # Append backend inventory
    try:
        from llm_benchmark.backends.detection import detect_backends

        statuses = detect_backends()
        backend_lines = []
        for bs in statuses:
            if bs.running:
                backend_lines.append(
                    f"  [green]{bs.name} (running, port {bs.port})[/green]"
                )
            elif bs.installed:
                backend_lines.append(
                    f"  [yellow]{bs.name} (installed, not running)[/yellow]"
                )
            else:
                backend_lines.append(
                    f"  [dim]{bs.name} (not installed)[/dim]"
                )

        if backend_lines:
            active_label = (
                f"\n[bold]Active backend:[/bold] "
                f"{info.backend_name} {info.backend_version}"
            )
            backends_section = "\n[bold]Backends:[/bold]\n" + "\n".join(backend_lines)
            summary_line = summary_line + active_label + backends_section
    except Exception:
        pass  # Detection failure should not break system summary

    return summary_line


def get_backend_inventory() -> str:
    """Return a rich-formatted string showing all backends with detailed status.

    Shows name, installed status, running status, port, and platform-specific
    install instructions for missing backends.

    Returns:
        Rich-formatted multi-line string with backend inventory.
    """
    try:
        from llm_benchmark.backends.detection import (
            detect_backends,
            get_install_instructions,
        )
    except ImportError:
        return "[dim]Backend detection not available[/dim]"

    statuses = detect_backends()
    lines: list[str] = []
    lines.append("[bold]Backend Inventory[/bold]")
    lines.append("")

    for bs in statuses:
        if bs.running:
            lines.append(
                f"  [green]{bs.name}[/green]  "
                f"installed={bs.installed}  running={bs.running}  "
                f"port={bs.port}"
            )
            if bs.binary_path:
                lines.append(f"    binary: {bs.binary_path}")
        elif bs.installed:
            lines.append(
                f"  [yellow]{bs.name}[/yellow]  "
                f"installed={bs.installed}  running={bs.running}  "
                f"port={bs.port}"
            )
            if bs.binary_path:
                lines.append(f"    binary: {bs.binary_path}")
        else:
            lines.append(
                f"  [dim]{bs.name}[/dim]  "
                f"installed=False  running=False"
            )
            instructions = get_install_instructions(bs.name)
            lines.append(f"    install: {instructions}")

    return "\n".join(lines)
