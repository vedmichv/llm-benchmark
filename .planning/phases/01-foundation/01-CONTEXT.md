# Phase 1: Foundation - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Consolidate the existing codebase (benchmark.py + extended_benchmark.py) into a proper Python package (`llm_benchmark/`) with cross-platform stability and pre-flight checks. Students can clone the repo and run `python -m llm_benchmark run` on Windows, macOS, or Linux without crashes, with clear errors when something is wrong.

Requirements: STAB-01, STAB-02, STAB-03, STAB-04, STAB-05, STAB-06, QUAL-01, QUAL-02, QUAL-05

</domain>

<decisions>
## Implementation Decisions

### Consolidation Strategy
- Migrate everything from extended_benchmark.py into the new package structure (system info, exports, prompt sets, model offloading, progress tracking)
- benchmark.py becomes redundant and is removed after migration
- compare_results.py becomes a subcommand (`python -m llm_benchmark compare`)
- Remove setup_passwordless_sudo.sh — obsolete since offloading uses keep_alive=0 API
- Keep run.py as a thin convenience wrapper that calls `python -m llm_benchmark`; remove run.sh/run.bat/run.ps1 if run.py covers all platforms

### Package Layout
- Modular split inside `llm_benchmark/`:
  - `__init__.py` — package init
  - `__main__.py` — entry point for `python -m llm_benchmark`
  - `cli.py` — argparse with subcommands (run, compare, info)
  - `runner.py` — benchmark execution logic
  - `models.py` — Pydantic response models
  - `system.py` — hardware/system info collection
  - `exporters.py` — JSON, CSV, Markdown output
  - `preflight.py` — Ollama connectivity + hardware checks
  - `prompts.py` — prompt sets (small/medium/large)
  - `config.py` — defaults, constants
- CLI uses explicit subcommands: `run`, `compare`, `info`
- `pyproject.toml` for modern Python packaging (replaces requirements.txt)
- Python 3.12+ minimum, `uv` as recommended package manager/runner
- Dependencies: ollama, pydantic, rich, tenacity

### Pre-flight Checks
- Ollama connectivity check blocks execution with platform-specific fix instructions if Ollama is not running
- RAM/GPU check warns but continues — don't gatekeep, let Ollama handle swapping
- Pre-flight runs automatically on every `run` command; fast checks always, slow checks skippable with `--skip-checks`
- No models found: suggest pulling a specific small model (`ollama pull llama3.2:1b`) and exit
- Compact one-line system summary before benchmark starts; full details via `info` subcommand

### Error Experience
- Friendly + actionable error messages with emoji indicators
- No stack traces unless `--debug` flag is passed
- When a model fails mid-benchmark: ask user whether to continue with remaining models (`Continue with remaining models? [Y/n]`)
- `--debug` flag for full stack traces and verbose logging (separate from `--verbose` which controls response streaming)
- Use `rich` library for colored/formatted terminal output
- English only — no localization

### Claude's Discretion
- Exact rich formatting choices (panels, tables, colors)
- Internal module boundaries (what helper functions go where)
- Pydantic model field naming and validation approach
- pyproject.toml build backend choice (hatchling, setuptools, etc.)
- Exact RAM estimation heuristic for model size warnings

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `extended_benchmark.py`: System info collection (CPU, RAM, GPU detection across platforms), export formats (JSON/CSV/MD), prompt sets (small/medium/large), model offloading via `keep_alive=0`, threading-based timeouts, progress tracking — all migrate into the new package
- `benchmark.py`: Pydantic models (Message, OllamaResponse) with field validation — migrate to `models.py`
- `compare_results.py`: Results comparison logic — migrate to `compare` subcommand
- `run.py`: Cross-platform launcher pattern — keep as thin wrapper

### Established Patterns
- Pydantic 2.x for response validation with `field_validator` and `model_validate`
- `ollama.chat()` with streaming support
- Nanosecond-to-second conversion for timing metrics
- Prompt caching detection (prompt_eval_count == -1)

### Integration Points
- Ollama SDK (`ollama.list()`, `ollama.chat()`) — primary external dependency
- CLI entry via `__main__.py` → argparse in `cli.py` → subcommand handlers
- Export files written to `results/` directory (not project root — UX-04 from Phase 2, but directory structure set up now)

</code_context>

<specifics>
## Specific Ideas

- Student runs: `uv run python -m llm_benchmark run` — uv handles venv and deps automatically
- Error messages should feel helpful, not technical — "your model may be too large" not "OOM error"
- Compact system summary inspired by neofetch style — one-line, informative, not overwhelming
- The `--debug` and `--verbose` flags serve different purposes: verbose = stream model responses, debug = show internal diagnostics

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-03-12*
