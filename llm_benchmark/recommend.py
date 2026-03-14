"""Model recommendation engine: suggest models based on available RAM.

Provides tiered model recommendations filtered by system RAM and already-
installed models. Used by the interactive menu to help students download
appropriate models before benchmarking.

Exports:
    get_recommended_models: Return models matching RAM tier thresholds.
    filter_already_installed: Remove models the user already has.
    offer_model_downloads: Interactive UI for downloading recommended models.
"""

from __future__ import annotations

import subprocess

from llm_benchmark.config import get_console

# RAM thresholds per tier (in GB).  A tier is available when system RAM >= threshold.
RAM_TIER_THRESHOLDS: dict[str, int] = {
    "small": 0,
    "medium": 16,
    "large": 36,
    "xl": 64,
    "xxl": 100,
}

# Curated model list ordered by tier, then roughly by size.
# Updated 2026-03-14: replaced unstable qwen3.5 small/medium with stable alternatives.
# qwen3.5 has known Ollama bugs (#14579, #14662) — only 35b kept (works on 64GB+).
TIERED_MODELS: list[dict[str, str]] = [
    # --- small (any RAM) ---
    {"name": "llama3.2:1b", "size_label": "1B", "tier": "small", "description": "Fast general-purpose chat"},
    {"name": "qwen3:1.7b", "size_label": "1.7B", "tier": "small", "description": "Compact thinking model"},
    {"name": "phi4-mini", "size_label": "3.8B", "tier": "small", "description": "Compact reasoning model"},
    # --- medium (16 GB+) ---
    {"name": "gemma3:4b", "size_label": "4B", "tier": "medium", "description": "Google's efficient model"},
    {"name": "llama3.2:3b", "size_label": "3B", "tier": "medium", "description": "Balanced speed and quality"},
    {"name": "qwen3:8b", "size_label": "8B", "tier": "medium", "description": "Strong all-rounder (stable)"},
    {"name": "mistral:7b", "size_label": "7B", "tier": "medium", "description": "Classic 7B from Mistral AI"},
    # --- large (36 GB+) ---
    {"name": "gemma3:27b", "size_label": "27B", "tier": "large", "description": "Google's flagship (stable, fast)"},
    {"name": "gemma3:12b", "size_label": "12B", "tier": "large", "description": "Google's mid-size model"},
    {"name": "deepseek-r1:14b", "size_label": "14B", "tier": "large", "description": "Reasoning-focused model"},
    {"name": "qwen3:14b", "size_label": "14B", "tier": "large", "description": "Qwen 3 thinking (stable)"},
    # --- xl (64 GB+) ---
    {"name": "nemotron-3-super", "size_label": "120B MoE/12B active", "tier": "xl", "description": "NVIDIA hybrid Mamba-Transformer (new!)"},
    {"name": "qwen3.5:35b", "size_label": "35B MoE", "tier": "xl", "description": "Qwen 3.5 MoE (arena top-20, may timeout)"},
    {"name": "qwen3:32b", "size_label": "32B", "tier": "xl", "description": "Qwen 3 thinking (stable)"},
    {"name": "llama3.3:70b", "size_label": "70B", "tier": "xl", "description": "Meta flagship 70B model"},
    # --- xxl (100 GB+) ---
    {"name": "qwen3.5:122b", "size_label": "122B MoE", "tier": "xxl", "description": "Qwen 3.5 MoE (arena #8, 81 GB)"},
    {"name": "deepseek-r1:70b", "size_label": "70B", "tier": "xxl", "description": "DeepSeek reasoning 70B"},
]

# Tier display order (for grouped UI output).
_TIER_ORDER = ["small", "medium", "large", "xl", "xxl"]
_TIER_LABELS = {
    "small": "Small models (any hardware)",
    "medium": "Medium models (16 GB+ RAM)",
    "large": "Large models (36 GB+ RAM)",
    "xl": "Extra-large models (64 GB+ RAM)",
    "xxl": "Massive models (100 GB+ RAM)",
}


def get_recommended_models(ram_gb: float) -> list[dict[str, str]]:
    """Return models whose tier threshold is at or below *ram_gb*.

    Parameters
    ----------
    ram_gb:
        Total system RAM in gigabytes.

    Returns
    -------
    list[dict]
        Filtered subset of ``TIERED_MODELS``.
    """
    eligible_tiers = {
        tier for tier, threshold in RAM_TIER_THRESHOLDS.items() if ram_gb >= threshold
    }
    return [m for m in TIERED_MODELS if m["tier"] in eligible_tiers]


def filter_already_installed(
    recommended: list[dict[str, str]],
    installed_names: list[str],
) -> list[dict[str, str]]:
    """Remove models whose ``name`` appears in *installed_names*.

    Parameters
    ----------
    recommended:
        Model dicts from :func:`get_recommended_models`.
    installed_names:
        Names of models already pulled in Ollama.

    Returns
    -------
    list[dict]
        Models not yet installed.
    """
    installed_set = set(installed_names)
    return [m for m in recommended if m["name"] not in installed_set]


def offer_model_downloads(backend, installed_models: list, *, force: bool = False) -> list:
    """Interactive prompt to download recommended models.

    Called from the interactive menu after system info display.  Detects
    RAM, builds a filtered recommendation list, and lets the user pick
    models to pull via ``ollama pull``.

    Parameters
    ----------
    backend:
        Backend instance for re-fetching installed models after pull.
    installed_models:
        Model dicts from ``backend.list_models()`` (preflight).

    Returns
    -------
    list
        Updated model list (may include newly downloaded models).
    """
    from llm_benchmark.system import _get_gpu_info, _get_ram_gb

    console = get_console()

    # Detect total available memory (RAM + VRAM for discrete GPU, just RAM for unified)
    ram_gb = _get_ram_gb()
    _, vram_gb = _get_gpu_info()
    # On Apple Silicon vram_gb is None (unified memory = RAM already includes GPU)
    # On discrete GPU systems, sum RAM + VRAM for total model capacity
    total_memory_gb = ram_gb + (vram_gb or 0)
    if total_memory_gb <= 0:
        return installed_models

    recommended = get_recommended_models(total_memory_gb)
    installed_names = [m['model'] if isinstance(m, dict) else m.model for m in installed_models]

    if force:
        # Show all recommended models, mark installed ones
        available = recommended
    else:
        available = filter_already_installed(recommended, installed_names)

    if not available:
        return installed_models

    # Ask whether to show recommendations (skip prompt in force mode)
    if not force:
        try:
            answer = input("  Download recommended models? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return installed_models

        if answer != "y":
            return installed_models

    # Display numbered list grouped by tier
    console.print()
    number = 1
    number_map: dict[int, dict[str, str]] = {}
    current_tier = None
    installed_set = set(installed_names)

    for model in available:
        if model["tier"] != current_tier:
            current_tier = model["tier"]
            label = _TIER_LABELS.get(current_tier, current_tier)
            console.print(f"  [bold]{label}[/bold]")

        is_installed = model["name"] in installed_set
        if is_installed:
            console.print(
                f"    {number}. {model['name']}  ([green]{model['size_label']}[/green]) - {model['description']} [green]\\[installed][/green]",
                highlight=False,
            )
        else:
            console.print(
                f"    {number}. {model['name']}  ([green]{model['size_label']}[/green]) - {model['description']}",
                highlight=False,
            )
        number_map[number] = model
        number += 1

    console.print()

    # Ask which models to download
    try:
        raw = input("  Enter model numbers to download (e.g. 1,3,5) or 'all': ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return installed_models

    if not raw:
        return installed_models

    # Parse selection
    selected: list[dict[str, str]] = []
    if raw.lower() == "all":
        selected = list(number_map.values())
    else:
        for token in raw.replace(" ", "").split(","):
            try:
                idx = int(token)
                if idx in number_map:
                    selected.append(number_map[idx])
            except ValueError:
                pass

    if not selected:
        return installed_models

    # Pull each selected model (skip already installed)
    console.print()
    pulled_any = False
    for model in selected:
        if model["name"] in installed_set:
            console.print(f"  [dim]Skipping {model['name']} (already installed)[/dim]")
            continue
        pulled_any = True
        console.print(f"  Pulling [bold]{model['name']}[/bold] ...")
        subprocess.run(["ollama", "pull", model["name"]])
        console.print(f"  [green]Done:[/green] {model['name']}")

    console.print()

    if not pulled_any:
        return installed_models

    # Re-fetch installed models via backend
    try:
        return backend.list_models()
    except Exception:
        return installed_models
