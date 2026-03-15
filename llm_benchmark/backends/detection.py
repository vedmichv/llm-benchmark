"""Backend detection, auto-start, and platform-specific install instructions.

Detects whether each supported backend (Ollama, llama.cpp, LM Studio) is
installed (binary found via shutil.which) and running (port responds).
Provides auto-start with health polling and per-OS install instructions.
"""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from llm_benchmark.backends import BackendError

# Backend definitions: (name, binary, default_port)
_BACKENDS: list[tuple[str, str, int]] = [
    ("ollama", "ollama", 11434),
    ("llama-cpp", "llama-server", 8080),
    ("lm-studio", "lms", 1234),
]

# Health check endpoints per backend
_HEALTH_ENDPOINTS: dict[str, str] = {
    "ollama": "/api/tags",
    "llama-cpp": "/health",
    "lm-studio": "/v1/models",
}

# Start commands per backend
_START_COMMANDS: dict[str, list[str]] = {
    "ollama": ["ollama", "serve"],
    "lm-studio": ["lms", "server", "start"],
}

# Install instructions: backend -> os -> instruction
_INSTALL_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "ollama": {
        "Darwin": "curl -fsSL https://ollama.com/install.sh | sh",
        "Linux": "curl -fsSL https://ollama.com/install.sh | sh",
        "Windows": "irm https://ollama.com/install.ps1 | iex",
    },
    "llama-cpp": {
        "Darwin": "brew install llama.cpp",
        "Linux": "sudo apt install llama.cpp OR sudo snap install llama-cpp",
        "Windows": "winget install ggml-org.llama-server",
    },
    "lm-studio": {
        "Darwin": "Download from https://lmstudio.ai/ then run: lms bootstrap",
        "Linux": "Download from https://lmstudio.ai/ then run: lms bootstrap (No MLX support on Linux)",
        "Windows": "Download from https://lmstudio.ai/ then run: lms bootstrap",
    },
}


@dataclass
class BackendStatus:
    """Status of a single backend: installed (binary found) and running (port open)."""

    name: str
    installed: bool
    running: bool
    binary_path: str | None
    port: int


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections.

    Args:
        host: Hostname or IP address.
        port: Port number to probe.
        timeout: Connection timeout in seconds.

    Returns:
        True if the port accepted a connection, False otherwise.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    finally:
        sock.close()


def detect_backends(
    port_overrides: dict[str, int] | None = None,
    host: str = "127.0.0.1",
) -> list[BackendStatus]:
    """Detect installed and running status for all supported backends.

    Args:
        port_overrides: Optional mapping of backend name to custom port.
        host: Host to probe for running backends.

    Returns:
        List of BackendStatus for ollama, llama-cpp, and lm-studio.
    """
    overrides = port_overrides or {}
    statuses: list[BackendStatus] = []

    for name, binary, default_port in _BACKENDS:
        port = overrides.get(name, default_port)
        binary_path = shutil.which(binary)
        installed = binary_path is not None
        running = _port_is_open(host, port) if installed else False

        statuses.append(
            BackendStatus(
                name=name,
                installed=installed,
                running=running,
                binary_path=binary_path,
                port=port,
            )
        )

    return statuses


def auto_start_backend(
    backend_name: str,
    *,
    port: int | None = None,
    model_path: str | None = None,
    host: str = "127.0.0.1",
    timeout: int = 30,
    log_dir: str = "results",
) -> subprocess.Popen:
    """Start a backend server and wait for it to become healthy.

    Args:
        backend_name: One of "ollama", "llama-cpp", "lm-studio".
        port: Custom port (uses backend default if None).
        model_path: Path to GGUF model file (required for llama-cpp).
        host: Host to bind the server to.
        timeout: Maximum seconds to wait for health check.
        log_dir: Directory for server log files.

    Returns:
        The subprocess.Popen object for the running server.

    Raises:
        ValueError: If llama-cpp is requested without model_path.
        BackendError: If server fails to start within timeout.
    """
    # Build command
    if backend_name == "llama-cpp":
        if model_path is None:
            raise ValueError(
                "model_path is required for llama-cpp backend"
            )
        effective_port = port or 8080
        cmd = [
            "llama-server",
            "-m", model_path,
            "--port", str(effective_port),
            "--host", host,
        ]
    elif backend_name in _START_COMMANDS:
        cmd = _START_COMMANDS[backend_name]
        effective_port = port or dict(
            (name, default_port) for name, _, default_port in _BACKENDS
        ).get(backend_name, 8080)
    else:
        raise ValueError(f"Unknown backend: {backend_name!r}")

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_path / f"{backend_name}.log"

    # Start server process
    log_file = open(log_file_path, "a")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    # Determine health endpoint
    endpoint = _HEALTH_ENDPOINTS.get(backend_name, "/health")
    health_url = f"http://{host}:{effective_port}{endpoint}"

    # Poll for health
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        # Check if process died
        if proc.poll() is not None:
            log_file.close()
            raise BackendError(
                f"{backend_name} process exited unexpectedly (code {proc.returncode})"
            )

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(health_url)
                if resp.status_code == 200:
                    return proc
        except Exception:
            pass

        time.sleep(1)

    # Timeout reached -- clean up
    proc.terminate()
    log_file.close()
    raise BackendError(f"{backend_name} failed to start within {timeout}s")


def get_install_instructions(
    backend_name: str,
    os_name: str | None = None,
) -> str:
    """Get platform-specific installation instructions for a backend.

    Args:
        backend_name: One of "ollama", "llama-cpp", "lm-studio".
        os_name: Platform name ("Darwin", "Linux", "Windows").
                 Auto-detected if None.

    Returns:
        Installation command or instructions string.
    """
    if os_name is None:
        os_name = platform.system()

    instructions = _INSTALL_INSTRUCTIONS.get(backend_name, {})
    return instructions.get(os_name, f"Visit the {backend_name} website for installation instructions")
