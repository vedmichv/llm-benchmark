"""Cross-platform hardware and system information collection.

Detects CPU, RAM, GPU, OS, Python version, and Ollama version without
requiring psutil or any C-extension dependencies. Uses platform module
and subprocess calls to platform-specific tools (sysctl on macOS,
/proc on Linux, wmic on Windows).
"""

from __future__ import annotations

import platform
import subprocess

from llm_benchmark.models import SystemInfo


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


def get_system_info() -> SystemInfo:
    """Collect comprehensive system information.

    Detects CPU, RAM, GPU, OS, Python version, and Ollama version
    across macOS, Linux, and Windows without psutil.

    Returns:
        SystemInfo with all detected fields populated.
    """
    cpu = _get_cpu_model()
    ram_gb = round(_get_ram_gb(), 1)
    gpu, gpu_vram = _get_gpu_info()
    ollama_ver = get_ollama_version()

    return SystemInfo(
        cpu=cpu,
        ram_gb=ram_gb,
        gpu=gpu,
        gpu_vram_gb=round(gpu_vram, 1) if gpu_vram is not None else None,
        os_name=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        ollama_version=ollama_ver,
    )


def format_system_summary() -> str:
    """Return a compact one-line summary for display before benchmarks.

    Uses rich markup for styled output. Example:
        "Apple M2 Pro | 16.0 GB RAM | Apple M2 Pro (integrated GPU) | Ollama 0.6.1"

    Returns:
        Formatted string with rich markup.
    """
    info = get_system_info()
    parts = [
        f"[bold]{info.cpu}[/bold]",
        f"{info.ram_gb:.0f} GB RAM",
    ]

    if info.gpu_vram_gb is not None:
        parts.append(f"{info.gpu} ({info.gpu_vram_gb:.0f} GB VRAM)")
    else:
        parts.append(info.gpu)

    parts.append(f"Ollama {info.ollama_version}")

    return " | ".join(parts)
