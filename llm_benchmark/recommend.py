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
}

# Curated model list ordered by tier, then roughly by size.
TIERED_MODELS: list[dict[str, str]] = [
    # --- small (any RAM) ---
    {"name": "llama3.2:1b", "size_label": "1B", "tier": "small", "description": "Fast general-purpose chat"},
    {"name": "qwen2.5:0.5b", "size_label": "0.5B", "tier": "small", "description": "Smallest available model"},
    {"name": "phi4-mini", "size_label": "3.8B", "tier": "small", "description": "Compact reasoning model"},
    # --- medium (16 GB+) ---
    {"name": "llama3.2:3b", "size_label": "3B", "tier": "medium", "description": "Balanced speed and quality"},
    {"name": "phi4", "size_label": "14B", "tier": "medium", "description": "Strong reasoning at medium size"},
    {"name": "qwen2.5:7b", "size_label": "7B", "tier": "medium", "description": "Versatile 7B model"},
    {"name": "gemma3:4b", "size_label": "4B", "tier": "medium", "description": "Google's efficient model"},
    # --- large (36 GB+) ---
    {"name": "llama3.1:8b", "size_label": "8B", "tier": "large", "description": "High-quality 8B model"},
    {"name": "qwen2.5:14b", "size_label": "14B", "tier": "large", "description": "Strong multilingual model"},
    {"name": "gemma3:12b", "size_label": "12B", "tier": "large", "description": "Google's larger model"},
    {"name": "phi4:14b", "size_label": "14B", "tier": "large", "description": "Full-size reasoning model"},
    # --- xl (64 GB+) ---
    {"name": "llama3.3:70b", "size_label": "70B", "tier": "xl", "description": "Flagship large model"},
    {"name": "qwen2.5:32b", "size_label": "32B", "tier": "xl", "description": "High-capacity multilingual"},
]

# Tier display order (for grouped UI output).
_TIER_ORDER = ["small", "medium", "large", "xl"]
_TIER_LABELS = {
    "small": "Small models (any hardware)",
    "medium": "Medium models (16 GB+ RAM)",
    "large": "Large models (36 GB+ RAM)",
    "xl": "Extra-large models (64 GB+ RAM)",
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


def offer_model_downloads(installed_models: list) -> list:
    """Interactive prompt to download recommended models.

    Called from the interactive menu after system info display.  Detects
    RAM, builds a filtered recommendation list, and lets the user pick
    models to pull via ``ollama pull``.

    Parameters
    ----------
    installed_models:
        Model objects returned by ``ollama.list()`` (preflight).

    Returns
    -------
    list
        Updated model list (may include newly downloaded models).
    """
    from llm_benchmark.system import _get_ram_gb

    console = get_console()

    # Detect RAM and build recommendations
    ram_gb = _get_ram_gb()
    if ram_gb <= 0:
        return installed_models

    recommended = get_recommended_models(ram_gb)
    installed_names = [m.model for m in installed_models]
    available = filter_already_installed(recommended, installed_names)

    if not available:
        return installed_models

    # Ask whether to show recommendations
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

    for model in available:
        if model["tier"] != current_tier:
            current_tier = model["tier"]
            label = _TIER_LABELS.get(current_tier, current_tier)
            console.print(f"  [bold]{label}[/bold]")

        console.print(
            f"    {number}. {model['name']}  ({model['size_label']}) - {model['description']}"
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

    # Pull each selected model
    console.print()
    for model in selected:
        console.print(f"  Pulling [bold]{model['name']}[/bold] ...")
        subprocess.run(["ollama", "pull", model["name"]])
        console.print(f"  [green]Done:[/green] {model['name']}")

    console.print()

    # Re-fetch installed models
    import ollama as ollama_client

    try:
        response = ollama_client.list()
        return response.models
    except Exception:
        return installed_models
