# Phase 4: Student Experience - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Interactive CLI menu for students who have never used CLI tools, visual ranked results after benchmarks, enhanced shareable Markdown reports with rankings and recommendations, unit tests with mocked Ollama (>60% coverage), and GitHub Actions CI (lint + tests + compile check). Concurrent/sweep modes stay CLI-only — menu covers the core student journey.

</domain>

<decisions>
## Implementation Decisions

### Interactive menu
- Plain `input()` prompts with numbered options — zero new dependencies
- No-args behavior: menu replaces the current "subcommand required" error (remove `required=True` from subparsers)
- Show brief system info one-liner before mode selection (CPU, RAM, GPU) so students see hardware context
- Four modes:
  - **Quick test (~30s):** auto-picks smallest pulled model, 1 short prompt, 1 run. Confirms "everything works"
  - **Standard:** medium prompts, 2 runs (matches current defaults)
  - **Full:** large prompts, 3 runs
  - **Custom:** student picks prompt set (small/medium/large), runs per prompt, and model selection
- Custom model selection: numbered list of pulled models, "Skip models (e.g. 3,4) or Enter for all:"
- Invalid input: re-prompt with "Please enter 1-4" hint, loop until valid
- Concurrent/sweep modes: CLI-only flags, not in menu
- Ctrl+C during benchmark: catch KeyboardInterrupt, export partial results collected so far

### Terminal bar chart
- Unicode block bars (████░░░) after all models complete, before "Results saved" message
- Metric: response t/s only (single metric, sorted fastest-first)
- Simple one-liner recommendation below chart: "Best for your setup: {model} ({rate} t/s) — fastest response generation"

### Report format & sharing
- Enhanced Markdown replaces existing MD exporter (upgrade export_markdown, not a separate file)
- One-line hardware summary in report header: "System: {CPU}, {RAM} GB RAM, {GPU}, {OS}"
- Rankings section with Unicode bar chart (text art, pasteable into GitHub/Discord/Slack)
- Simple one-liner recommendation: "Best for your setup: {model} ({rate} t/s)"
- All three modes (standard, concurrent, sweep) get the enhanced format
- Concurrent ranking metric: aggregate throughput (t/s)
- Sweep report: per-model best config callout ("Best config for {model}: num_ctx={X}, num_gpu={Y} ({rate} t/s)")

### Test & CI
- Priority modules for >60% coverage: runner, models, exporters (core pipeline students depend on)
- GitHub Actions: ruff lint + pytest + python -m py_compile — runs on push to main and PRs
- Python version: 3.12 only (matches pyproject.toml requires-python)
- Ruff added as dev dependency in pyproject.toml (students and CI use same version)

### Claude's Discretion
- Bar chart width and exact Unicode character choices
- Quick test: how to determine "smallest" model (by parameter count or name heuristic)
- Menu box styling (Rich Panel vs plain text)
- Exact ruff rule configuration
- Test fixture design and mock patterns

</decisions>

<specifics>
## Specific Ideas

- Menu preview shown during discussion: numbered list with mode descriptions and time estimates
- Bar chart style: `████████████████████  45.2 t/s` with `░` for empty portion
- Report header: compact one-line format with date, model count, and mode label
- Model skip interface: "Skip models (e.g. 3,4) or Enter for all:" — skip-based rather than include-based

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `config.get_console()`: Rich Console singleton — use for all menu output and bar chart rendering
- `preflight.run_preflight_checks()`: Returns list of available models — reuse for menu model listing
- `exporters.export_markdown()`: Existing MD exporter to enhance in-place with rankings section
- `exporters._ensure_dir()` / `_timestamp()`: Shared export helpers
- `models.ModelSummary`: Has `avg_response_ts` — use for ranking and bar chart
- `models.SystemInfo`: Has all hardware fields for report header

### Established Patterns
- Lazy imports in CLI handlers for fast startup (`from llm_benchmark.exporters import ...` inside handler)
- Argparse subcommand dispatch via `_HANDLERS` dict in cli.py
- Console singleton — never use `print()` directly
- Results saved to `results/` with auto `.gitignore`

### Integration Points
- `cli.py:main()` — add no-args menu detection before `parser.parse_args()`
- `cli.py:_handle_run()` — add bar chart + recommendation after exports
- `exporters.py:export_markdown()` — enhance with rankings section
- `pyproject.toml` — add ruff to dev dependencies
- `.github/workflows/` — new CI workflow file

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-student-experience*
*Context gathered: 2026-03-13*
