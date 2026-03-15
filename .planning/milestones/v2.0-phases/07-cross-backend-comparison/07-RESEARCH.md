# Phase 7: Cross-Backend Comparison - Research

**Researched:** 2026-03-14
**Domain:** Multi-backend orchestration, comparison reporting, CLI/menu integration
**Confidence:** HIGH

## Summary

Phase 7 builds on the fully functional multi-backend infrastructure from Phases 5-6. The core challenge is orchestrating sequential benchmark runs across multiple backends with the same prompts, then aggregating results into comparison views (bar chart for single-model, matrix table for multi-model). All building blocks exist: `detect_backends()` for discovery, `create_backend()` factory, `benchmark_model()` for execution, `render_bar_chart()`/Rich Tables for display, and `_filename()`/exporters for output.

The implementation is primarily new orchestration code (a comparison runner that loops over backends), new display functions (comparison bar chart and matrix table), new exporters (comparison JSON + Markdown), menu option 5, CLI `--backend all` support, and README documentation. No new dependencies are needed. The existing patterns are well-established and should be followed exactly.

**Primary recommendation:** Build a `comparison.py` module containing the orchestration logic, comparison display, and comparison export functions. Wire it into `cli.py` (for `--backend all`) and `menu.py` (for option 5). Keep the module self-contained to minimize changes to existing code.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Single-model view: side-by-side horizontal bar chart with backends as bars (reuse/extend render_bar_chart pattern)
- Multi-model view: Rich matrix table (models as rows, backends as columns, response t/s values, winner starred per row)
- Winner metric: response t/s (eval_count/eval_duration)
- Missing backend/model combos show "--" in matrix (no error, no blocking)
- Overall recommendation: "Fastest backend: X (N/M models)" at bottom of matrix
- Export: Markdown + JSON comparison report alongside individual per-backend result files
- Sequential per backend: run ALL models on Backend A, then ALL on Backend B, etc.
- Auto-detect backends via detect_backends() -- only run on what's actually running
- If only 1 backend detected: warn and fall back to standard single-backend benchmark
- llama-cpp GGUF selection: auto-scan ~/.cache/huggingface with scan_gguf_files(), match by model name similarity to Ollama names, skip if no match
- Each backend's results saved individually (normal export) PLUS unified comparison report
- All existing flags (--prompt-set, --runs-per-prompt, --concurrent, etc.) apply identically to each backend run
- Menu option 5 always visible regardless of detected backends
- Menu option 5 auto-run comparison with medium prompts, 2 runs -- minimal friction
- If only 1 backend when menu option 5 selected: show "Install another backend to compare" with install hints
- Modes 1-4 keep existing backend selector when >1 detected (Phase 6 behavior unchanged)
- Mode 5 auto-uses all detected backends (no backend selection step)
- README: New "Multi-Backend Setup" section after existing Quick Start
- README: Per-OS guides with one-liner install + verify command for each backend
- README: New "Cross-Backend Comparison" section with CLI usage and crafted-but-realistic example output
- README example shows llama.cpp ~1.5x faster than Ollama on Apple Silicon

### Claude's Discretion
- GGUF-to-Ollama model name matching algorithm details
- Comparison JSON schema structure
- Exact terminal formatting and spacing of comparison output
- How to handle --concurrent with --backend all (sequential backends each running concurrent internally, or error)
- llama-cpp single-model-per-server handling during sequential comparison runs

### Deferred Ideas (OUT OF SCOPE)
None

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| COMP-01 | `--backend all` runs same prompts on all detected backends sequentially | Orchestration pattern in comparison.py; detect_backends() + create_backend() + benchmark_model() loop |
| COMP-02 | Single-model comparison: one model tested on all backends, side-by-side bar chart | Extend render_bar_chart() with backend names as entries instead of model names |
| COMP-03 | Full matrix mode: N models x M backends, comparison table with winner per model | Rich Table with dynamic columns per backend, star annotation on per-row max |
| COMP-04 | "Fastest backend" recommendation per model and overall in comparison report | Count wins across matrix rows, format summary line |
| COMP-05 | Comparison mode as menu option 5 ("Compare backends") | Add option to run_interactive_menu(), wire to comparison runner |
| DOC-01 | README updated with multi-backend quick start and per-OS setup guides | detection.py _INSTALL_INSTRUCTIONS already has per-OS data; format into README |
| DOC-02 | Backend comparison example in README showing real cross-backend output | Craft realistic output matching actual display format |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rich | (existing) | Terminal tables, styled output, bar charts | Already used throughout project for all display |
| pydantic | (existing) | Data models for comparison results | Already used for all models in models.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | (existing) | Backend connectivity checks during detection | Already imported in detection.py |

### Alternatives Considered
None -- no new dependencies needed. Everything builds on existing infrastructure.

## Architecture Patterns

### Recommended Project Structure
```
llm_benchmark/
  comparison.py          # NEW: orchestration, display, export for cross-backend comparison
  cli.py                 # MODIFY: add "all" to --backend choices, add comparison handler
  menu.py                # MODIFY: add option 5, add _mode_compare()
  display.py             # MODIFY: add render_comparison_bar_chart()
  exporters.py           # MODIFY: add export_comparison_json(), export_comparison_markdown()
  README.md              # MODIFY: add multi-backend setup and comparison sections
```

### Pattern 1: Comparison Orchestration
**What:** A `run_comparison()` function that loops over detected backends, runs benchmarks on each, and collects results into a unified structure.
**When to use:** When `--backend all` is passed via CLI or menu option 5 is selected.
**Example:**
```python
# Source: Derived from existing cli.py _handle_run pattern
def run_comparison(
    backends: list[BackendStatus],
    prompts: list[str],
    runs_per_prompt: int,
    timeout: int,
    skip_warmup: bool,
    max_retries: int,
    verbose: bool,
    skip_models: list[str],
) -> ComparisonResult:
    """Run benchmarks on all backends sequentially and collect results."""
    all_backend_results: dict[str, list[ModelSummary]] = {}

    for status in backends:
        backend = create_backend(status.name, port=status.port)
        # For llama-cpp: need GGUF model path handling
        models = run_preflight_checks(backend=backend, skip_models=skip_models)

        summaries = []
        for model_info in models:
            summary = benchmark_model(
                backend=backend,
                model_name=model_info['model'],
                prompts=prompts,
                # ... other params
            )
            summaries.append(summary)
            unload_model(backend, model_info['model'])

        all_backend_results[status.name] = summaries
        # Export individual backend results as normal

    return ComparisonResult(backend_results=all_backend_results)
```

### Pattern 2: Matrix Table Display
**What:** Rich Table with dynamic columns -- one per detected backend -- showing response t/s per model, with star on the winner.
**When to use:** Multi-model comparison (N models x M backends).
**Example:**
```python
# Source: Derived from compare.py Rich Table pattern
from rich.table import Table

def render_comparison_matrix(
    results: dict[str, list[ModelSummary]],
) -> None:
    console = get_console()
    table = Table(show_header=True, title="Cross-Backend Comparison")
    table.add_column("Model", style="bold")

    backend_names = list(results.keys())
    for name in backend_names:
        table.add_column(name.title(), justify="right")
    table.add_column("Winner", justify="center", style="bold green")

    # Collect all unique models
    all_models = set()
    for summaries in results.values():
        for s in summaries:
            all_models.add(s.model)

    wins: dict[str, int] = {name: 0 for name in backend_names}

    for model in sorted(all_models):
        row = [model]
        rates: dict[str, float] = {}
        for bname in backend_names:
            summary = next(
                (s for s in results[bname] if s.model == model), None
            )
            if summary:
                rate = summary.avg_response_ts
                rates[bname] = rate
                row.append(f"{rate:.1f} t/s")
            else:
                row.append("--")

        if rates:
            winner = max(rates, key=rates.get)
            wins[winner] += 1
            row.append(winner.title())
        else:
            row.append("--")

        table.add_row(*row)

    console.print(table)

    # Overall recommendation
    total = sum(wins.values())
    if total > 0:
        overall_winner = max(wins, key=wins.get)
        console.print(
            f"\n  Fastest backend: [bold]{overall_winner.title()}[/bold] "
            f"({wins[overall_winner]}/{total} models)"
        )
```

### Pattern 3: GGUF-to-Ollama Model Name Matching (Claude's Discretion)
**What:** Fuzzy matching between GGUF file names and Ollama model names to auto-select GGUF files for llama-cpp comparison.
**When to use:** When `--backend all` includes llama-cpp and no explicit `--model-path` is given.
**Example:**
```python
def match_gguf_to_ollama_name(
    ollama_name: str,
    gguf_files: list[tuple[Path, str]],
) -> Path | None:
    """Find the best GGUF match for an Ollama model name.

    Strategy: normalize both names (lowercase, strip tags/quantization),
    look for substring match. E.g. "llama3.2:1b" matches
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf".
    """
    # Extract base name: "llama3.2:1b" -> "llama3.2" + "1b"
    base = ollama_name.split(":")[0].lower().replace(".", "").replace("-", "")
    tag = ollama_name.split(":")[-1].lower() if ":" in ollama_name else ""

    for path, display_name in gguf_files:
        normalized = path.stem.lower().replace("-", "").replace("_", "").replace(".", "")
        if base in normalized:
            # Check size tag if present
            if tag and tag in normalized:
                return path
            elif not tag:
                return path

    # Fallback: partial match on display_name
    for path, display_name in gguf_files:
        if base in display_name.lower().replace("-", "").replace(" ", ""):
            return path

    return None
```

### Pattern 4: Handling --concurrent with --backend all (Claude's Discretion)
**What:** When both `--concurrent` and `--backend all` are used, run each backend sequentially but allow concurrent mode within each backend's run.
**Recommendation:** Allow it -- each backend runs its concurrent benchmarks independently. This is the most intuitive behavior for users.

### Pattern 5: llama-cpp Single-Model-Per-Server (Claude's Discretion)
**What:** llama-cpp's llama-server loads one model at a time. During comparison, when iterating models, the server must be restarted with each new GGUF file.
**Recommendation:** For each model in the comparison loop, if the backend is llama-cpp: stop the server, restart with the new GGUF path, wait for health check, then run benchmarks. Use the existing `auto_start_backend()` pattern. If no GGUF match is found for a model, skip it (show "--" in matrix).

### Anti-Patterns to Avoid
- **Running backends in parallel:** Backends share GPU/CPU resources. Running simultaneously would corrupt timing measurements. Always sequential.
- **Blocking on missing backends:** If only 1 backend is detected, warn and fall back gracefully. Never error out.
- **Modifying existing mode behavior:** Modes 1-4 must remain unchanged. The backend selector from Phase 6 stays as-is.
- **Creating a new CLI subcommand:** Comparison is a mode of `run`, not a separate subcommand. Use `--backend all` on the existing `run` subcommand.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend detection | Custom port scanning | `detect_backends()` from detection.py | Already handles all 3 backends, port overrides, binary detection |
| Backend creation | Manual imports | `create_backend()` factory | Already handles all backends with proper kwargs |
| Benchmark execution | Custom timing loop | `benchmark_model()` from runner.py | Has warmup, retry, timeout, caching detection |
| Bar chart rendering | Custom Unicode art | `render_bar_chart()` from display.py | Consistent look with rest of tool |
| File export | Custom file I/O | Extend `_filename()` and `_ensure_dir()` from exporters.py | Handles results/ dir, .gitignore, timestamps |
| GGUF scanning | Custom file walker | `scan_gguf_files()` from menu.py | Already handles metadata extraction |
| Install instructions | Hardcoded strings | `get_install_instructions()` from detection.py | Per-OS instructions already maintained |

**Key insight:** Phase 7 is an orchestration layer that composes existing building blocks. Almost no low-level logic needs to be written.

## Common Pitfalls

### Pitfall 1: llama-cpp Model Switching
**What goes wrong:** llama-server only serves one model at a time. Attempting to benchmark multiple models without restarting the server gives wrong results or errors.
**Why it happens:** Ollama and LM Studio can switch models transparently; llama-cpp cannot.
**How to avoid:** For llama-cpp, restart the server process with a new `--model` flag for each GGUF file. Use `auto_start_backend()` with the new model path. Kill the old process first.
**Warning signs:** Same timing numbers for different "models" on llama-cpp.

### Pitfall 2: Model Name Mismatch Across Backends
**What goes wrong:** Ollama uses "llama3.2:1b" while llama-cpp uses GGUF filenames and LM Studio uses its own naming. The matrix table needs to align models across backends.
**Why it happens:** Each backend has its own model naming convention.
**How to avoid:** Use Ollama model names as the canonical reference. For llama-cpp, match GGUF files to Ollama names via fuzzy matching. For LM Studio, the model name from its API response should be close enough. Show "--" for any unmatched combinations.
**Warning signs:** Matrix table showing the same model as different rows.

### Pitfall 3: Preflight Running Per-Backend
**What goes wrong:** `run_preflight_checks()` may block if a backend is not running, causing the comparison to hang.
**Why it happens:** Preflight includes connectivity checks that block on failure.
**How to avoid:** For comparison mode, skip connectivity-blocking checks since we already know which backends are running from `detect_backends()`. Only run model listing per backend (non-blocking).
**Warning signs:** Comparison mode hanging on a backend that was detected but went down.

### Pitfall 4: Menu Signature Change Breaking Existing Tests
**What goes wrong:** Adding option 5 to the menu changes the valid choices set, breaking test mocks.
**Why it happens:** Tests mock `_prompt_choice` with specific valid sets.
**How to avoid:** Update test mocks to include "5" in valid choices. Keep backward-compatible return type from `run_interactive_menu()`.
**Warning signs:** test_menu.py failures after adding option 5.

### Pitfall 5: Comparison Export Filename Collision
**What goes wrong:** Individual backend exports and comparison export could have similar names, confusing users.
**Why it happens:** All use the `_filename()` helper with similar prefixes.
**How to avoid:** Use a distinct prefix like "comparison" for the unified report: `comparison_YYYYMMDD_HHMMSS.json`.
**Warning signs:** Users seeing extra files they don't understand in results/.

## Code Examples

### Comparison Data Model
```python
# New Pydantic models for comparison results
class BackendModelResult(BaseModel):
    """Result for one model on one backend."""
    backend: str
    model: str
    avg_response_ts: float
    avg_prompt_eval_ts: float
    avg_total_ts: float

class ComparisonResult(BaseModel):
    """Unified comparison across all backends."""
    backends: list[str]
    models: list[str]
    results: list[BackendModelResult]
    # Computed
    winner_per_model: dict[str, str]  # model -> backend name
    overall_winner: str
    overall_wins: dict[str, int]  # backend -> win count
```

### CLI Integration for --backend all
```python
# In cli.py _build_parser(), modify --backend choices:
run_parser.add_argument(
    "--backend",
    choices=["ollama", "llama-cpp", "lm-studio", "all"],
    default="ollama",
    help="Backend to use (default: ollama, 'all' for cross-backend comparison)",
)

# In cli.py _handle_run(), add comparison branch:
if args.backend == "all":
    from llm_benchmark.comparison import run_comparison
    return run_comparison(args)
```

### Menu Option 5 Integration
```python
# In menu.py run_interactive_menu(), extend menu:
console.print("  1. Quick test (~30 seconds)")
console.print("  2. Standard benchmark")
console.print("  3. Full benchmark")
console.print("  4. Custom")
console.print("  5. Compare backends")
console.print()

choice = _prompt_choice("Select mode [1-5]: ", {"1", "2", "3", "4", "5"})

if choice == "5":
    return _mode_compare(backend, models)
```

### Single-Model Comparison Bar Chart
```python
# Extend display.py pattern for backend comparison
def render_comparison_bar_chart(
    backend_rates: list[tuple[str, float]],
    model_name: str,
) -> None:
    """Bar chart with backends as entries for a single model."""
    console = get_console()
    max_rate = max(r for _, r in backend_rates) if backend_rates else 0
    max_name_len = max(len(n) for n, _ in backend_rates) if backend_rates else 0

    console.print()
    console.print(f"[bold]{model_name} - Backend Comparison[/bold]")
    console.print()

    for name, rate in backend_rates:
        bar_len = round(BAR_WIDTH * rate / max_rate) if max_rate > 0 else 0
        bar = BAR_FULL * bar_len + BAR_EMPTY * (BAR_WIDTH - bar_len)
        star = " [bold yellow]*[/bold yellow]" if rate == max_rate else ""
        console.print(f"  {name:<{max_name_len}}  {bar}  {rate:>6.1f} t/s{star}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single Ollama only | Multi-backend via Backend Protocol | Phase 5-6 | All backends share common interface |
| CLI-only backend selection | Interactive menu backend selector | Phase 6 (06-05) | Users can pick backend from menu |
| No cross-backend comparison | This phase adds it | Phase 7 | Students can see which runtime is fastest |

**Existing infrastructure ready for use:**
- `detect_backends()` returns `list[BackendStatus]` with installed/running status
- `create_backend(name, port=)` creates any backend instance
- `benchmark_model()` works with any Backend instance
- `render_bar_chart()` takes `list[tuple[str, float]]` -- generic enough to use for backends
- `_filename()` already handles backend names in filenames
- `scan_gguf_files()` and `extract_gguf_model_name()` ready for GGUF matching
- `get_install_instructions()` has per-OS data for all 3 backends

## Open Questions

1. **--concurrent with --backend all**
   - What we know: Each backend can handle concurrent mode independently
   - Recommendation: Allow it. Run each backend sequentially, but within each backend, use concurrent mode. This is the intuitive composition of both flags.

2. **llama-cpp server restart between models**
   - What we know: `auto_start_backend()` exists for starting, but there is no `stop_backend()` function
   - Recommendation: Track the subprocess.Popen object returned by `auto_start_backend()`, call `proc.terminate()` + `proc.wait()` before starting with the next model. May need a small helper function.

3. **LM Studio model switching**
   - What we know: LM Studio can load multiple models via its API
   - What's unclear: Whether `list_models()` returns all available or only loaded models
   - Recommendation: Use `list_models()` to get available models, benchmark those that match the Ollama model list. Skip non-matching ones.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COMP-01 | --backend all runs on all detected backends sequentially | unit | `uv run pytest tests/test_comparison.py::test_run_comparison_sequential -x` | No -- Wave 0 |
| COMP-02 | Single-model bar chart with backends | unit | `uv run pytest tests/test_comparison.py::test_single_model_bar_chart -x` | No -- Wave 0 |
| COMP-03 | Matrix table N models x M backends | unit | `uv run pytest tests/test_comparison.py::test_matrix_table -x` | No -- Wave 0 |
| COMP-04 | Fastest backend recommendation | unit | `uv run pytest tests/test_comparison.py::test_winner_recommendation -x` | No -- Wave 0 |
| COMP-05 | Menu option 5 compare backends | unit | `uv run pytest tests/test_menu.py::test_mode_compare -x` | No -- Wave 0 |
| DOC-01 | README multi-backend quick start | manual-only | Visual inspection of README.md | N/A |
| DOC-02 | Backend comparison example in README | manual-only | Visual inspection of README.md | N/A |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_comparison.py` -- covers COMP-01 through COMP-04
- [ ] Update `tests/test_menu.py` -- covers COMP-05 (option 5)
- [ ] Update `tests/test_cli.py` -- covers `--backend all` parsing

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `cli.py`, `runner.py`, `menu.py`, `display.py`, `exporters.py`, `compare.py`, `backends/__init__.py`, `backends/detection.py`, `models.py`, `conftest.py` -- all directly read and analyzed
- CONTEXT.md -- user decisions fully documented

### Secondary (MEDIUM confidence)
- llama-server single-model constraint -- documented in STATE.md blockers and confirmed by codebase (model_path param in auto_start_backend)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, all existing code verified
- Architecture: HIGH - all integration points identified and code patterns verified
- Pitfalls: HIGH - derived from actual codebase constraints (llama-cpp single-model, naming mismatches)

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable -- no external dependency changes)
