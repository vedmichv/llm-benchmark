# Phase 4: Student Experience - Research

**Researched:** 2026-03-13
**Domain:** Interactive CLI, terminal visualization, Markdown report enhancement, test coverage, GitHub Actions CI
**Confidence:** HIGH

## Summary

Phase 4 adds an interactive menu for CLI-novice students, a terminal bar chart after benchmarks, enhanced Markdown reports with rankings/recommendations, expanded test coverage (>60%), and GitHub Actions CI. The user has locked all major decisions: plain `input()` for the menu (zero new dependencies), Unicode block bars for the chart, in-place enhancement of `export_markdown()`, and ruff as the lint tool.

The existing codebase is well-structured for these additions. Current test coverage is already at 62% (1187 statements, 452 missed), so the >60% target is met but the priority modules (runner at 85%, models at 100%, exporters at 36%) need the exporters coverage raised. The main integration points are `cli.py:main()` for no-args menu detection, `cli.py:_handle_run()` for the bar chart, and `exporters.py:export_markdown()` for report enhancement.

**Primary recommendation:** Build the interactive menu as a new `menu.py` module that returns an `argparse.Namespace` compatible with `_handle_run`, keeping the menu logic cleanly separated from CLI parsing. Use the Rich Console singleton for all output including the bar chart. Enhance `export_markdown()` in-place to add rankings sections for all three modes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Plain `input()` prompts with numbered options -- zero new dependencies
- No-args behavior: menu replaces the current "subcommand required" error (remove `required=True` from subparsers)
- Show brief system info one-liner before mode selection
- Four modes: Quick test (~30s), Standard, Full, Custom
- Custom model selection: numbered list of pulled models, "Skip models (e.g. 3,4) or Enter for all:"
- Invalid input: re-prompt with "Please enter 1-4" hint, loop until valid
- Concurrent/sweep modes: CLI-only flags, not in menu
- Ctrl+C during benchmark: catch KeyboardInterrupt, export partial results
- Unicode block bars after all models complete, before "Results saved"
- Metric: response t/s only, sorted fastest-first
- Simple one-liner recommendation: "Best for your setup: {model} ({rate} t/s) -- fastest response generation"
- Enhanced Markdown replaces existing MD exporter (upgrade export_markdown, not separate file)
- One-line hardware summary in report header
- Rankings section with Unicode bar chart (text art, pasteable)
- All three modes (standard, concurrent, sweep) get enhanced format
- Concurrent ranking metric: aggregate throughput (t/s)
- Sweep report: per-model best config callout
- Priority modules for >60% coverage: runner, models, exporters
- GitHub Actions: ruff lint + pytest + python -m py_compile on push to main and PRs
- Python version: 3.12 only
- Ruff added as dev dependency in pyproject.toml

### Claude's Discretion
- Bar chart width and exact Unicode character choices
- Quick test: how to determine "smallest" model (by parameter count or name heuristic)
- Menu box styling (Rich Panel vs plain text)
- Exact ruff rule configuration
- Test fixture design and mock patterns

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UX-01 | Running tool with no arguments shows interactive menu (quick test / standard / full / custom) | Menu module with `input()` loops; modify `cli.py:main()` to detect no-args and delegate to menu; remove `required=True` from subparsers |
| UX-02 | Quick test mode runs ~30 seconds: smallest model, 1 prompt, confirms "everything works" | Sort models by `model.size` (on-disk bytes from `ollama.list()`), pick smallest, use 1 short prompt, 1 run, skip warmup for speed |
| UX-03 | End of benchmark shows ranked model comparison with visual bar chart in terminal | `render_bar_chart()` function using Rich Console, Unicode blocks, sorted by `avg_response_ts` descending |
| UX-05 | Results include system info, model rankings, and recommendations for optimal config | Recommendation one-liner from sorted ModelSummary list; system info already in exports; add rankings section to Markdown |
| UX-06 | Shareable report format (Markdown with system info + rankings + individual runs) | Enhance `export_markdown()` with rankings table + text bar chart + recommendation; same for concurrent and sweep variants |
| QUAL-03 | Unit tests with mocked Ollama for core functions (>60% coverage) | Current coverage: 62% overall. Exporters at 36% need attention. Add tests for new menu, bar chart, enhanced exports |
| QUAL-04 | GitHub Actions CI running lint (ruff) + compile check + unit tests | New `.github/workflows/ci.yml`; ruff as dev dependency; `python -m py_compile` check |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rich | >=14.0 | Console output, optional Panel for menu | Already a dependency; Console singleton pattern established |
| ruff | >=0.9 | Linting and formatting | Fast, replaces flake8+isort+black; community standard for Python CI |
| pytest | >=8.0 | Test framework | Already in test dependencies |
| pytest-cov | >=7.0 | Coverage reporting | Needed to verify >60% threshold in CI |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | Mocking Ollama calls in tests | All tests that touch runner/preflight/concurrent |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| plain input() | questionary/InquirerPy | User locked: zero new deps. input() is sufficient |
| Unicode text bars | Rich Bar/BarColumn | Rich bar components are for Progress, not static charts. Manual Unicode is simpler and pasteable |
| ruff | flake8+black | User locked: ruff. Single tool, faster, consistent |

**Installation:**
```bash
# Add ruff and pytest-cov to dev dependencies in pyproject.toml
uv add --dev ruff pytest-cov
```

## Architecture Patterns

### Recommended Project Structure
```
llm_benchmark/
  menu.py              # NEW: Interactive menu logic (input loops, mode config)
  display.py           # NEW: Bar chart rendering, recommendation formatting
  cli.py               # MODIFIED: no-args detection, bar chart call after exports
  exporters.py         # MODIFIED: enhanced Markdown with rankings sections
  config.py            # MODIFIED: add quick-test constants if needed
.github/
  workflows/
    ci.yml             # NEW: GitHub Actions CI workflow
```

### Pattern 1: Menu Module Returning Namespace
**What:** The menu module collects user choices and returns an `argparse.Namespace` object compatible with `_handle_run()`, so the existing handler needs zero changes for mode dispatch.
**When to use:** When adding an alternative input path (interactive vs CLI args) to the same execution logic.
**Example:**
```python
# menu.py
import argparse
from llm_benchmark.config import get_console

def run_interactive_menu(models: list) -> argparse.Namespace:
    """Present interactive menu, return Namespace matching _handle_run expectations."""
    console = get_console()

    # Show system info one-liner
    from llm_benchmark.system import format_system_summary
    console.print(format_system_summary())
    console.print()

    console.print("[bold]LLM Benchmark[/bold]")
    console.print()
    console.print("  1. Quick test (~30 seconds)")
    console.print("  2. Standard benchmark")
    console.print("  3. Full benchmark")
    console.print("  4. Custom")
    console.print()

    while True:
        choice = input("Select mode [1-4]: ").strip()
        if choice in ("1", "2", "3", "4"):
            break
        console.print("  Please enter 1-4")

    # Build Namespace based on choice
    # ... mode-specific logic ...
    return argparse.Namespace(
        verbose=False,
        skip_checks=False,
        skip_models=skip_models,
        prompt_set=prompt_set,
        prompts=None,
        runs_per_prompt=runs_per_prompt,
        timeout=300,
        skip_warmup=skip_warmup,
        max_retries=3,
        concurrent=None,
        sweep=False,
    )
```

### Pattern 2: No-Args Detection in main()
**What:** Check `sys.argv` before argparse parsing to intercept the no-arguments case.
**When to use:** When adding interactive mode without breaking existing CLI behavior.
**Example:**
```python
# In cli.py:main()
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    # No-args: launch interactive menu
    args_list = argv if argv is not None else sys.argv[1:]
    if not args_list:
        from llm_benchmark.menu import run_interactive_menu
        from llm_benchmark.preflight import run_preflight_checks

        models = run_preflight_checks()
        args = run_interactive_menu(models)
        set_debug(False)
        return _handle_run(args)

    args = parser.parse_args(argv)
    # ... rest of existing logic ...
```

### Pattern 3: Bar Chart as Pure Function
**What:** A function that takes a list of (model_name, rate) tuples and prints a bar chart using the console singleton.
**When to use:** After benchmark completion in `_handle_run`, before export messages.
**Example:**
```python
# display.py
from llm_benchmark.config import get_console

BAR_FULL = "\u2588"   # Full block
BAR_EMPTY = "\u2591"  # Light shade
BAR_WIDTH = 30

def render_bar_chart(
    rankings: list[tuple[str, float]],
    metric_label: str = "t/s",
) -> None:
    """Print a ranked bar chart to console.

    Args:
        rankings: List of (model_name, rate) sorted fastest-first.
        metric_label: Unit label for the metric.
    """
    console = get_console()
    if not rankings:
        return

    max_rate = rankings[0][1]
    max_name_len = max(len(name) for name, _ in rankings)

    console.print()
    console.print("[bold]Model Rankings[/bold]")
    console.print()

    for name, rate in rankings:
        bar_len = int((rate / max_rate) * BAR_WIDTH) if max_rate > 0 else 0
        empty_len = BAR_WIDTH - bar_len
        bar = BAR_FULL * bar_len + BAR_EMPTY * empty_len
        console.print(
            f"  {name:<{max_name_len}}  {bar}  {rate:.1f} {metric_label}"
        )

    # Recommendation
    best_name, best_rate = rankings[0]
    console.print()
    console.print(
        f"  Best for your setup: [bold]{best_name}[/bold] "
        f"({best_rate:.1f} {metric_label}) -- fastest response generation"
    )


def render_text_bar_chart(
    rankings: list[tuple[str, float]],
    metric_label: str = "t/s",
) -> str:
    """Return a text-only bar chart suitable for Markdown reports."""
    # Same logic but returns string without Rich markup
    ...
```

### Pattern 4: Enhanced Markdown Export (In-Place Upgrade)
**What:** Modify `export_markdown()` to add a Rankings section with text bar chart and recommendation between System Information and Detailed Results.
**When to use:** Upgrading the existing exporter without creating a separate function.
**Example structure in Markdown output:**
```markdown
# LLM Benchmark Results

**Generated:** 2026-03-13 14:30:00 | **Models:** 3 | **Mode:** Standard

**System:** Apple M2 Pro, 16 GB RAM, Apple M2 Pro (integrated GPU), macOS 15.3

---

## Rankings

llama3.2:3b   ████████████████████████████░░  45.2 t/s
llama3.2:1b   ████████████████████░░░░░░░░░░  32.1 t/s
phi:2.7b      ████████████░░░░░░░░░░░░░░░░░░  18.7 t/s

Best for your setup: llama3.2:3b (45.2 t/s) -- fastest response generation

## Summary
...
```

### Anti-Patterns to Avoid
- **Mixing input() and Rich prompts:** Rich.Prompt requires different error handling. User locked `input()` -- stick with it exclusively for interactive input.
- **Modifying argparse `required=True` globally:** Only change subparsers `required` -- keep individual subparser arguments unchanged.
- **Bar chart with Rich markup in Markdown:** The Markdown bar chart must use plain text Unicode characters only, no ANSI escape codes. Use a separate `render_text_bar_chart()` function.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Linting | Custom style checks | ruff | Comprehensive, fast, community standard |
| CI workflow | Shell scripts for testing | GitHub Actions workflow YAML | Standard, well-documented, free for public repos |
| Coverage measurement | Manual counting | pytest-cov | Accurate line-level coverage with threshold enforcement |
| Terminal colors | ANSI escape codes | Rich Console (already used) | Consistent, cross-platform, existing singleton |

**Key insight:** This phase adds no new runtime dependencies. All new functionality uses stdlib `input()`, existing Rich Console, and Unicode characters. Only dev dependencies (ruff, pytest-cov) are added.

## Common Pitfalls

### Pitfall 1: input() Blocks Event Loop
**What goes wrong:** `input()` is blocking and cannot be interrupted cleanly on all platforms.
**Why it happens:** Windows handles Ctrl+C differently during `input()`.
**How to avoid:** Wrap `input()` calls in try/except for EOFError and KeyboardInterrupt. For the menu, KeyboardInterrupt should exit cleanly. During benchmarks, catch it to export partial results.
**Warning signs:** Tests hanging on CI when `input()` is called without mocking.

### Pitfall 2: sys.argv Mutation in Tests
**What goes wrong:** Tests that manipulate `sys.argv` to test no-args behavior leak state to other tests.
**Why it happens:** `sys.argv` is global state.
**How to avoid:** Use the `argv` parameter of `main()` instead. Pass `argv=[]` for no-args testing. Always mock `input()` in menu tests.
**Warning signs:** Tests pass individually but fail when run together.

### Pitfall 3: Unicode Bar Characters Not Rendering
**What goes wrong:** Some terminal emulators (especially Windows cmd.exe) may not render Unicode block characters.
**Why it happens:** Legacy code pages don't support full Unicode.
**How to avoid:** The Rich Console handles encoding. For Markdown reports, use plain ASCII fallback if needed, but Unicode blocks are safe for GitHub/Discord/Slack rendering.
**Warning signs:** Garbled output on Windows terminals.

### Pitfall 4: Coverage Regression from New Untested Code
**What goes wrong:** Adding menu.py and display.py without tests drops overall coverage below 60%.
**Why it happens:** New code with 0% coverage dilutes the average.
**How to avoid:** Write tests for every new module. Current coverage is 62% -- adding ~150 lines of untested code would drop it below 60%. Every new function needs at least one test.
**Warning signs:** CI coverage check failing after adding new modules.

### Pitfall 5: Argparse SystemExit on Missing Subcommand
**What goes wrong:** With `required=False` on subparsers, `parser.parse_args([])` sets `args.command` to `None` instead of raising SystemExit.
**Why it happens:** Argparse behavior change between `required=True` and `required=False`.
**How to avoid:** After removing `required=True`, check `args.command is None` as the trigger for interactive menu mode. Don't rely on exception handling for this.
**Warning signs:** Menu not triggering, or "NoneType has no attribute" errors.

### Pitfall 6: Quick Test Model Selection Edge Cases
**What goes wrong:** If user has no models pulled, or only very large models, quick test fails or takes too long.
**Why it happens:** `run_preflight_checks()` already handles "no models" with sys.exit(1), but model size sorting needs a fallback.
**How to avoid:** Use `model.size` (bytes on disk) from `ollama.list()` response. This is always available. Sort ascending, pick first. If only 1 model exists, use it regardless of size.
**Warning signs:** AttributeError on model object, or test taking >30s.

## Code Examples

### Determining Smallest Model from Ollama List
```python
# The ollama.list() response has model objects with a .size attribute (bytes on disk)
# This is available for all models and correlates well with parameter count
def pick_smallest_model(models: list) -> str:
    """Pick the smallest available model by on-disk size.

    Args:
        models: List from run_preflight_checks() (ollama model objects).

    Returns:
        Model name string for the smallest model.
    """
    sorted_models = sorted(models, key=lambda m: m.size)
    return sorted_models[0].model
```
Confidence: HIGH -- `model.size` is a standard attribute in the ollama Python SDK response objects.

### Quick Test Mode Configuration
```python
QUICK_TEST_PROMPT = "Write a one-sentence summary of what a CPU does."
QUICK_TEST_RUNS = 1

# Quick test: smallest model, 1 short prompt, 1 run, skip warmup for speed
# Expected ~30s: model load (~10s) + inference (~15-20s for small model)
```

### GitHub Actions CI Workflow
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Lint with ruff
        run: uv run ruff check llm_benchmark/ tests/

      - name: Compile check
        run: |
          python -m py_compile llm_benchmark/__init__.py
          python -m py_compile llm_benchmark/cli.py
          python -m py_compile llm_benchmark/runner.py
          python -m py_compile llm_benchmark/models.py
          python -m py_compile llm_benchmark/exporters.py
          python -m py_compile llm_benchmark/config.py

      - name: Run tests
        run: uv run pytest tests/ -x -q
```
Confidence: HIGH -- standard GitHub Actions patterns; astral-sh/setup-uv is the official uv action.

### Ruff Configuration in pyproject.toml
```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "F",     # pyflakes
    "I",     # isort
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "SIM",   # flake8-simplify
]
```
Confidence: HIGH -- conservative rule set that catches real issues without being noisy.

### Mocking Pattern for Menu Tests
```python
# Test interactive menu without actual input()
from unittest.mock import patch, MagicMock

def test_menu_quick_test_mode():
    """Quick test mode returns correct Namespace."""
    from llm_benchmark.menu import run_interactive_menu

    mock_model = MagicMock()
    mock_model.model = "llama3.2:1b"
    mock_model.size = 1_000_000_000  # 1GB

    with patch("builtins.input", return_value="1"):
        args = run_interactive_menu([mock_model])

    assert args.prompt_set == "small"  # or custom quick-test prompt
    assert args.runs_per_prompt == 1
```

### KeyboardInterrupt Handling for Partial Results
```python
# In _handle_run, wrap the benchmark loop
try:
    for idx, model in enumerate(models):
        # ... benchmark logic ...
        all_summaries.append(summary)
except KeyboardInterrupt:
    console.print("\n[yellow]Benchmark interrupted. Saving partial results...[/yellow]")

# Export whatever we have (runs after both normal and interrupted flows)
if all_summaries:
    # ... export logic ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| flake8+black+isort | ruff (single tool) | 2023-2024 | Much faster, single config, replaces 3 tools |
| setup-python + pip | setup-uv + uv sync | 2024-2025 | Faster CI installs, lockfile-based reproducibility |
| actions/checkout@v3 | actions/checkout@v4 | 2023 | Node 20 runtime, faster |
| pytest only | pytest + pytest-cov | Always | Coverage enforcement in CI |

## Open Questions

1. **Quick test prompt length vs 30-second target**
   - What we know: Small models (1b params) generate ~30-50 t/s. A short prompt producing ~100 tokens takes ~2-3s for inference, but model loading can take 10-15s.
   - What's unclear: Whether skipping warmup makes quick test faster (no double-load) or slower (load counted in timing).
   - Recommendation: Skip warmup for quick test (single run, load time is acceptable). Use a short prompt from the "small" set. If model is already loaded from a previous run, this will be very fast.

2. **Bar chart width in narrow terminals**
   - What we know: Default terminal width is 80 columns. Model names can be 20+ chars.
   - What's unclear: Whether to dynamically size bars based on terminal width.
   - Recommendation: Use fixed BAR_WIDTH=30 characters. With 20-char model name, rate display, and padding, this fits in ~70 columns. Acceptable for 80+ column terminals which covers vast majority of student setups.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | pyproject.toml (implicit pytest discovery) |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -x -q --cov=llm_benchmark --cov-fail-under=60` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UX-01 | No-args launches menu; menu returns valid Namespace | unit | `uv run pytest tests/test_menu.py -x` | No -- Wave 0 |
| UX-02 | Quick test picks smallest model, 1 prompt, 1 run | unit | `uv run pytest tests/test_menu.py::TestQuickTest -x` | No -- Wave 0 |
| UX-03 | Bar chart renders sorted models with Unicode blocks | unit | `uv run pytest tests/test_display.py -x` | No -- Wave 0 |
| UX-05 | Recommendation one-liner generated from rankings | unit | `uv run pytest tests/test_display.py::TestRecommendation -x` | No -- Wave 0 |
| UX-06 | Enhanced Markdown has rankings + text bar chart | unit | `uv run pytest tests/test_exporters.py::TestEnhancedMarkdown -x` | No -- Wave 0 |
| QUAL-03 | >60% coverage across project | integration | `uv run pytest tests/ --cov=llm_benchmark --cov-fail-under=60` | Partial -- existing tests cover 62% |
| QUAL-04 | CI runs lint+tests+compile on push | smoke | `gh workflow run ci.yml` (manual verify) | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -x -q --cov=llm_benchmark --cov-fail-under=60`
- **Phase gate:** Full suite green + coverage >= 60% before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_menu.py` -- covers UX-01, UX-02 (menu logic, quick test selection)
- [ ] `tests/test_display.py` -- covers UX-03, UX-05 (bar chart rendering, recommendation)
- [ ] Enhanced tests in `tests/test_exporters.py` -- covers UX-06 (rankings in Markdown)
- [ ] `.github/workflows/ci.yml` -- covers QUAL-04
- [ ] `pyproject.toml` updates -- ruff + pytest-cov in dev deps, ruff config

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `cli.py`, `exporters.py`, `models.py`, `runner.py`, `preflight.py`, `system.py`, `config.py` -- read in full
- Existing test suite: 118 tests across 12 files, all passing
- Coverage report: 62% overall (1187 statements, 452 missed) -- measured with pytest-cov
- `pyproject.toml`: Current dependencies and build config

### Secondary (MEDIUM confidence)
- GitHub Actions patterns: `actions/checkout@v4`, `actions/setup-python@v5`, `astral-sh/setup-uv@v5` -- standard community patterns
- ruff configuration: Conservative rule set based on established community defaults

### Tertiary (LOW confidence)
- Quick test 30-second estimate: Based on typical small model performance (~30-50 t/s for 1b models), but actual timing depends heavily on hardware and whether model is already loaded

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new runtime deps, all patterns use existing Rich + stdlib
- Architecture: HIGH -- clean integration points identified in existing code
- Pitfalls: HIGH -- based on direct code analysis and established Python testing patterns
- Quick test timing: MEDIUM -- depends on hardware/model availability

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable domain, no fast-moving dependencies)
