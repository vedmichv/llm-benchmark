---
phase: quick
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - llm_benchmark/recommend.py
  - llm_benchmark/menu.py
  - tests/test_recommend.py
autonomous: true
requirements: [QUICK-1]

must_haves:
  truths:
    - "After system info display, user is asked if they want to download recommended models"
    - "Pressing N or Enter skips recommendation and goes straight to mode selection"
    - "Pressing Y shows a tiered model list filtered by available RAM"
    - "Already-installed models are skipped (not shown or marked as installed)"
    - "User can pick models by number and they are pulled via ollama pull"
  artifacts:
    - path: "llm_benchmark/recommend.py"
      provides: "Model recommendation logic and download UI"
      exports: ["get_recommended_models", "offer_model_downloads"]
    - path: "llm_benchmark/menu.py"
      provides: "Interactive menu with recommend hook"
    - path: "tests/test_recommend.py"
      provides: "Unit tests for recommendation tiers and filtering"
  key_links:
    - from: "llm_benchmark/menu.py"
      to: "llm_benchmark/recommend.py"
      via: "offer_model_downloads() call after system summary"
      pattern: "offer_model_downloads"
    - from: "llm_benchmark/recommend.py"
      to: "llm_benchmark/system.py"
      via: "_get_ram_gb or get_system_info for RAM detection"
      pattern: "_get_ram_gb|get_system_info"
---

<objective>
Add a model recommender step to the interactive menu. After displaying system info but before showing the mode selection (1-4), prompt the user: "Download recommended models? (y/N)". If yes, detect RAM, show a tiered model list with checkboxes, let user pick models by number, then pull each selected model via `ollama pull`. Skip models already installed.

Purpose: Students who just installed Ollama have zero models. This guides them to appropriate models for their hardware before benchmarking.
Output: New `recommend.py` module, updated `menu.py`, tests.
</objective>

<execution_context>
@/Users/viktor/.claude/get-shit-done/workflows/execute-plan.md
@/Users/viktor/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@llm_benchmark/menu.py
@llm_benchmark/system.py
@llm_benchmark/preflight.py
@llm_benchmark/config.py

<interfaces>
From llm_benchmark/system.py:
```python
def _get_ram_gb() -> float:  # private but can be reused pattern
def get_system_info() -> SystemInfo:
```

From llm_benchmark/config.py:
```python
def get_console() -> Console:  # Rich Console singleton — use for ALL output
```

From llm_benchmark/menu.py:
```python
def _prompt_choice(prompt_text: str, valid: set[str]) -> str:
def run_interactive_menu(models: list) -> argparse.Namespace:
```

From llm_benchmark/preflight.py:
```python
def _get_system_ram_gb() -> float:  # duplicated RAM detection (avoid circular imports)
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create recommend.py with tiered model recommendations</name>
  <files>llm_benchmark/recommend.py, tests/test_recommend.py</files>
  <behavior>
    - get_recommended_models(ram_gb=8.0) returns only small tier (<2B models)
    - get_recommended_models(ram_gb=16.0) returns small + medium tiers (up to 8B)
    - get_recommended_models(ram_gb=36.0) returns small + medium + large tiers (up to 30B)
    - get_recommended_models(ram_gb=64.0) returns all tiers including xl (70B)
    - Each model entry is a dict/dataclass with: name (str like "llama3.2:1b"), size_label (str like "1B"), tier (str), description (str)
    - filter_already_installed(recommended, installed_names) removes models whose name appears in installed_names
    - filter_already_installed with no overlap returns full list
    - filter_already_installed with all overlap returns empty list
  </behavior>
  <action>
    Create `llm_benchmark/recommend.py` with:

    1. A TIERED_MODELS list of dicts, each with keys: name, size_label, tier, description. Tiers:
       - "small" (any RAM): llama3.2:1b, qwen2.5:0.5b, phi4-mini
       - "medium" (16GB+): llama3.2:3b, phi4, qwen2.5:7b, gemma3:4b
       - "large" (36GB+): llama3.1:8b, qwen2.5:14b, gemma3:12b, phi4:14b
       - "xl" (64GB+): llama3.3:70b, qwen2.5:32b

    2. RAM_TIER_THRESHOLDS dict: {"small": 0, "medium": 16, "large": 36, "xl": 64}

    3. `get_recommended_models(ram_gb: float) -> list[dict]`: Filter TIERED_MODELS to only include tiers where ram_gb >= threshold.

    4. `filter_already_installed(recommended: list[dict], installed_names: list[str]) -> list[dict]`: Remove models whose "name" is in installed_names.

    5. `offer_model_downloads(installed_models: list) -> list`: The interactive UI function:
       - Detect RAM via the same pattern as preflight._get_system_ram_gb() (duplicate the helper or import from system — prefer importing _get_ram_gb from system.py since menu.py already imports from system).
       - Call get_recommended_models and filter_already_installed.
       - If no recommendations remain, return installed_models silently.
       - Ask "Download recommended models? (y/N): " using plain input(). Default is N (Enter skips).
       - If yes: display numbered list grouped by tier with tier headers. Format each line as: "  {n}. {name}  ({size_label}) - {description}"
       - Ask "Enter model numbers to download (e.g. 1,3,5) or 'all': " using plain input().
       - For each selected model, run `subprocess.run(["ollama", "pull", model_name])` and print progress via console.
       - After downloads, re-fetch installed models via `ollama.list()` and return the updated list.
       - Handle EOFError/KeyboardInterrupt gracefully (return installed_models unchanged).

    Use `from llm_benchmark.config import get_console` for all output. No print() calls.
    Use plain input() for prompts — no new dependencies.

    Write tests in tests/test_recommend.py covering get_recommended_models tier filtering and filter_already_installed logic. Do NOT test offer_model_downloads (it requires interactive input and subprocess).
  </action>
  <verify>
    <automated>cd /Users/viktor/Documents/GitHub/vedmich/llm-benchmark && uv run pytest tests/test_recommend.py -x -q</automated>
  </verify>
  <done>recommend.py exports get_recommended_models and offer_model_downloads. All tier filtering tests pass. filter_already_installed correctly removes installed models.</done>
</task>

<task type="auto">
  <name>Task 2: Wire recommend into interactive menu flow</name>
  <files>llm_benchmark/menu.py</files>
  <action>
    Modify `run_interactive_menu` in menu.py to call the recommender after system summary display but before the mode menu.

    In `run_interactive_menu`, after the line `console.print(format_system_summary())` and `console.print()`, add:

    ```python
    # Offer model downloads if RAM allows recommendations
    from llm_benchmark.recommend import offer_model_downloads
    models = offer_model_downloads(models)
    ```

    This is a lazy import (consistent with project pattern — see cli.py). The function returns the (potentially updated) model list, so the rest of the menu works with any newly downloaded models.

    Also update the no-args handler in cli.py: currently `run_preflight_checks()` is called before `run_interactive_menu(models)`. If the user downloads new models via the recommender, the models list is already refreshed inside `offer_model_downloads`. The `_handle_run` function calls `run_preflight_checks` again anyway, but with `skip_checks=True` in the namespace built by `_build_namespace`, so this is fine — no changes needed in cli.py.

    Verify existing tests still pass after the change.
  </action>
  <verify>
    <automated>cd /Users/viktor/Documents/GitHub/vedmich/llm-benchmark && uv run pytest tests/ -x -q</automated>
  </verify>
  <done>Interactive menu flow: header -> system info -> model recommender prompt -> mode selection (1-4). Existing tests pass. New models downloaded by recommender are included in the benchmark model list.</done>
</task>

</tasks>

<verification>
1. `uv run pytest tests/ -x -q` — all tests pass including new recommend tests
2. `uv run python -c "from llm_benchmark.recommend import get_recommended_models, offer_model_downloads; print('imports ok')"` — module imports cleanly
3. Manual: `python -m llm_benchmark` shows recommend prompt after system info, before mode menu
</verification>

<success_criteria>
- get_recommended_models returns correct tiers based on RAM thresholds
- filter_already_installed removes models already present in Ollama
- Interactive menu shows "Download recommended models? (y/N)" after system summary
- Pressing Enter/N skips to mode selection unchanged
- Pressing Y shows tiered list, user selects, ollama pull runs for each
- All existing and new tests pass
</success_criteria>

<output>
After completion, create `.planning/quick/1-add-model-recommender-to-interactive-men/1-SUMMARY.md`
</output>
