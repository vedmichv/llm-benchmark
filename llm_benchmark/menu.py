"""Interactive menu for students who have never used CLI tools.

When ``python -m llm_benchmark`` is invoked with no arguments, the user
sees a numbered menu instead of an argparse error.

Exports:
    run_interactive_menu: Show menu, return populated argparse.Namespace.
    select_backend_interactive: Show backend selection, return backend params.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

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


def _prompt_yn(prompt_text: str, default: bool = False) -> bool:
    """Ask a yes/no question. Returns bool."""
    console = get_console()
    try:
        raw = input(prompt_text).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        sys.exit(0)
    if not raw:
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# GGUF file scanning and metadata extraction
# ---------------------------------------------------------------------------

# GGUF magic number and header constants
_GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian


def extract_gguf_model_name(path: Path) -> str | None:
    """Extract the model name from GGUF file metadata.

    Reads the GGUF header to find the ``general.name`` key.
    Returns None if the file is not a valid GGUF or the key is missing.
    """
    try:
        with open(path, "rb") as f:
            # Read magic (4 bytes) + version (4 bytes) + tensor_count (8 bytes)
            # + metadata_kv_count (8 bytes)
            header = f.read(24)
            if len(header) < 24:
                return None
            magic, version, _tensor_count, kv_count = struct.unpack(
                "<IIqq", header
            )
            if magic != _GGUF_MAGIC or version < 2:
                return None

            # Parse key-value pairs looking for general.name
            for _ in range(kv_count):
                # Read key: uint64 length + string bytes
                key_len_bytes = f.read(8)
                if len(key_len_bytes) < 8:
                    return None
                (key_len,) = struct.unpack("<Q", key_len_bytes)
                key = f.read(key_len).decode("utf-8", errors="replace")

                # Read value type (uint32)
                vtype_bytes = f.read(4)
                if len(vtype_bytes) < 4:
                    return None
                (vtype,) = struct.unpack("<I", vtype_bytes)

                if vtype == 8:  # STRING type
                    val_len_bytes = f.read(8)
                    if len(val_len_bytes) < 8:
                        return None
                    (val_len,) = struct.unpack("<Q", val_len_bytes)
                    val = f.read(val_len).decode("utf-8", errors="replace")
                    if key == "general.name":
                        return val
                    continue

                # Skip non-string types by their known sizes
                # Types: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32,
                #        5=INT32, 6=FLOAT32, 7=BOOL, 8=STRING,
                #        9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
                skip_sizes = {
                    0: 1, 1: 1, 2: 2, 3: 2, 4: 4,
                    5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
                }
                if vtype in skip_sizes:
                    f.read(skip_sizes[vtype])
                elif vtype == 9:
                    # ARRAY: element_type (uint32) + count (uint64) then elements
                    # Too complex to skip generically, bail out
                    return None
                else:
                    return None
    except (OSError, struct.error):
        return None

    return None


def _clean_gguf_filename(path: Path) -> str:
    """Derive a display name from a GGUF filename."""
    name = path.stem  # strip .gguf
    # Strip common quantization suffixes for display
    return name.replace("-", " ").replace("_", " ")


def scan_gguf_files(directory: Path) -> list[tuple[Path, str]]:
    """Scan a directory recursively for GGUF files.

    Returns:
        List of (path, display_name) tuples sorted by filename.
    """
    if not directory.exists():
        return []

    results: list[tuple[Path, str]] = []
    for gguf_path in sorted(directory.rglob("*.gguf")):
        if not gguf_path.is_file():
            continue
        display = extract_gguf_model_name(gguf_path) or _clean_gguf_filename(
            gguf_path
        )
        results.append((gguf_path, display))
    return results


def _select_gguf_model() -> str:
    """Show GGUF file scanner and let user select or enter a path.

    Returns:
        Absolute path to the selected GGUF file.
    """
    console = get_console()
    console.print()
    console.print("  [bold]Select GGUF model file:[/bold]")

    # Scan common locations
    hf_cache = Path.home() / ".cache" / "huggingface"
    found = scan_gguf_files(hf_cache)

    if found:
        console.print()
        for i, (path, display) in enumerate(found, 1):
            size_gb = path.stat().st_size / (1024**3)
            console.print(f"    {i}. {display}  ({size_gb:.1f} GB)")
        console.print(f"    {len(found) + 1}. Enter path manually")
        console.print()

        valid = {str(i) for i in range(1, len(found) + 2)}
        choice = _prompt_choice(
            f"  Select model [1-{len(found) + 1}]: ", valid
        )
        idx = int(choice) - 1
        if idx < len(found):
            return str(found[idx][0])
    else:
        console.print("    No GGUF files found in ~/.cache/huggingface/")
        console.print()

    # Manual path entry
    console.print()
    while True:
        try:
            raw = input("  Enter path to GGUF file: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            sys.exit(0)
        path = Path(raw).expanduser()
        if path.exists() and path.suffix == ".gguf":
            return str(path)
        console.print("  [red]File not found or not a .gguf file[/red]")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def select_backend_interactive() -> tuple[str, int | None, str | None]:
    """Show backend selection menu and return chosen backend parameters.

    Returns:
        Tuple of (backend_name, port_override_or_none, model_path_or_none).
    """
    from llm_benchmark.backends.detection import (
        auto_start_backend,
        detect_backends,
    )

    console = get_console()

    statuses = detect_backends()
    available = [s for s in statuses if s.installed or s.running]

    console.print()
    console.print("  [bold]Backends:[/bold]")
    for i, s in enumerate(statuses, 1):
        if s.running:
            status_str = "[green]running[/green]"
        elif s.installed:
            status_str = "[yellow]installed, not running[/yellow]"
        else:
            status_str = "[dim]not installed[/dim]"
        console.print(f"    {i}. {s.name:<12} [{status_str}]")
    console.print()

    # Single available backend -- auto-select
    if len(available) == 1:
        selected = available[0]
        console.print(
            f"  Only {selected.name.title()} detected -- using {selected.name.title()}."
            f" Install llama.cpp or LM Studio for more options."
        )
        console.print()
    elif len(available) == 0:
        # No backends available at all -- fall back to ollama
        console.print(
            "  [red]No backends detected.[/red] Defaulting to Ollama."
        )
        return ("ollama", None, None)
    else:
        # Multiple backends -- let user choose
        valid_indices = {str(i) for i, s in enumerate(statuses, 1) if s.installed or s.running}
        choice = _prompt_choice(
            f"  Select backend [{','.join(sorted(valid_indices))}]: ",
            valid_indices,
        )
        selected = statuses[int(choice) - 1]

    # If installed but not running, offer to start
    model_path: str | None = None
    port: int | None = None

    if selected.name == "llama-cpp" and not selected.running:
        # Need GGUF model path first
        model_path = _select_gguf_model()
        if _prompt_yn(f"  Start {selected.name}? (y/N) "):
            auto_start_backend(
                selected.name, model_path=model_path, port=selected.port
            )
    elif not selected.running and selected.installed:
        if _prompt_yn(f"  Start {selected.name}? (y/N) "):
            auto_start_backend(selected.name, port=selected.port)

    # For llama-cpp that is already running, still might need model_path context
    if selected.name == "llama-cpp" and model_path is None:
        # Already running -- no model_path needed (server has its own model)
        pass

    return (selected.name, port, model_path)


def _build_namespace(
    *,
    prompt_set: str = "medium",
    prompts: list[str] | None = None,
    runs_per_prompt: int = DEFAULT_RUNS_PER_PROMPT,
    skip_warmup: bool = False,
    skip_models: list[str] | None = None,
    backend: str = "ollama",
    model_path: str | None = None,
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
        backend=backend,
        port=None,
        model_path=model_path,
    )


def _mode_quick(models: list, *, backend: str = "ollama", model_path: str | None = None) -> argparse.Namespace:
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
        backend=backend,
        model_path=model_path,
    )


def _mode_standard(*, backend: str = "ollama", model_path: str | None = None) -> argparse.Namespace:
    """Standard benchmark: medium prompts, 2 runs, warmup enabled."""
    return _build_namespace(
        prompt_set="medium",
        runs_per_prompt=2,
        skip_warmup=False,
        backend=backend,
        model_path=model_path,
    )


def _mode_full(*, backend: str = "ollama", model_path: str | None = None) -> argparse.Namespace:
    """Full benchmark: large prompts, 3 runs, warmup enabled."""
    return _build_namespace(
        prompt_set="large",
        runs_per_prompt=3,
        skip_warmup=False,
        backend=backend,
        model_path=model_path,
    )


def _mode_custom(
    models: list,
    *,
    backend: str = "ollama",
    model_path: str | None = None,
) -> argparse.Namespace:
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
        backend=backend,
        model_path=model_path,
    )


def _mode_compare(backend: str = "ollama") -> argparse.Namespace:
    """Compare backends mode: detect backends and return comparison Namespace."""
    from llm_benchmark.backends.detection import (
        detect_backends,
        get_install_instructions,
    )

    console = get_console()

    statuses = detect_backends()
    running = [s for s in statuses if s.running]

    if len(running) <= 1:
        console.print(
            "[yellow]Only 1 backend detected. "
            "Install another backend to compare:[/yellow]"
        )
        not_running = [s for s in statuses if not s.running]
        for s in not_running:
            hint = get_install_instructions(s.name)
            console.print(f"  {s.name}: {hint}")
        console.print()
    else:
        names = ", ".join(s.name for s in running)
        console.print(f"[bold]Comparing {len(running)} backends: {names}[/bold]")
        console.print()

    return _build_namespace(
        backend="all",
        prompt_set="medium",
        runs_per_prompt=2,
        skip_warmup=False,
        skip_models=None,
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

    backend_name = getattr(backend, "name", "ollama")
    model_path = getattr(backend, "_model_path", None)

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
    console.print("  5. Compare backends")
    console.print()

    choice = _prompt_choice("Select mode [1-5]: ", {"1", "2", "3", "4", "5"})
    console.print()

    if choice == "1":
        return _mode_quick(models, backend=backend_name, model_path=model_path)
    if choice == "2":
        return _mode_standard(backend=backend_name, model_path=model_path)
    if choice == "3":
        return _mode_full(backend=backend_name, model_path=model_path)
    if choice == "5":
        return _mode_compare(backend_name)
    return _mode_custom(models, backend=backend_name, model_path=model_path)
