# Requirements: LLM Benchmark v2

**Defined:** 2026-03-12
**Core Value:** Student clones repo, runs one command, gets clear answer about their hardware's LLM capabilities.

## v1 Requirements

### Stability (STAB)

- [x] **STAB-01**: Benchmark runs without crashes on Windows, macOS, and Linux
- [x] **STAB-02**: Tool checks Ollama connectivity before starting and shows actionable error if unreachable
- [x] **STAB-03**: Tool checks available RAM and GPU before benchmark and warns if resources are insufficient for selected models
- [x] **STAB-04**: Throughput averaging uses total_tokens/total_time (not arithmetic mean of rates)
- [x] **STAB-05**: Model offloading works without sudo via Ollama API (keep_alive=0)
- [x] **STAB-06**: Timeouts work cross-platform via threading (no signal.SIGALRM)

### Benchmarking (BENCH)

- [x] **BENCH-01**: Tool runs warmup requests before actual measurements to exclude model load overhead
- [x] **BENCH-02**: Tool retries failed requests with exponential backoff (configurable max retries)
- [x] **BENCH-03**: User can run concurrent benchmark with --concurrent N (parallel requests to same model)
- [x] **BENCH-04**: Concurrent mode reports aggregate throughput (total_tokens/wall_time) and per-request average
- [x] **BENCH-05**: User can run parameter sweep with --sweep (auto-explore num_ctx, num_gpu combinations)
- [x] **BENCH-06**: Sweep reports best configuration found with throughput numbers
- [x] **BENCH-07**: Prompt caching detection excludes affected metrics from averages (instead of silently corrupting)

### Analysis (ANLZ)

- [x] **ANLZ-01**: User can sort benchmark results by any metric (response_ts, total_ts, load_time)
- [x] **ANLZ-02**: User can filter top-N results from CSV/JSON output
- [x] **ANLZ-03**: User can compare results from multiple runs side-by-side

### UX (UX)

- [x] **UX-01**: Running tool with no arguments shows interactive menu (quick test / standard / full / custom)
- [x] **UX-02**: Quick test mode runs ~30 seconds: smallest model, 1 prompt, confirms "everything works"
- [x] **UX-03**: End of benchmark shows ranked model comparison with visual bar chart in terminal
- [x] **UX-04**: All result files saved to results/ directory (not project root)
- [x] **UX-05**: Results include system info, model rankings, and recommendations for optimal config
- [x] **UX-06**: Shareable report format (Markdown with system info + rankings + individual runs)

### Code Quality (QUAL)

- [x] **QUAL-01**: Single consolidated benchmark module (no benchmark.py vs extended_benchmark.py confusion)
- [x] **QUAL-02**: Python package structure (llm_benchmark/ with submodules)
- [x] **QUAL-03**: Unit tests with mocked Ollama for core functions (>60% coverage)
- [x] **QUAL-04**: GitHub Actions CI running lint (ruff) + compile check + unit tests
- [x] **QUAL-05**: Python >=3.10 requirement (Pydantic 2.x + tenacity compatibility)

## v2.0 Requirements (Multi-Backend Benchmark)

### Backend Abstraction (BACK)

- [ ] **BACK-01**: Backend Protocol defines chat(), list_models(), unload_model() with normalized BackendResponse
- [ ] **BACK-02**: BackendResponse normalizes all timing data to seconds regardless of backend (Ollama ns, llama.cpp ms, LM Studio pre-computed)
- [ ] **BACK-03**: OllamaBackend wraps existing ollama.chat() code with zero behavior change for users
- [ ] **BACK-04**: Runner accepts Backend instance instead of calling ollama directly
- [ ] **BACK-05**: All existing tests pass after refactor with no Ollama-specific type leaks

### New Backends (BEND)

- [ ] **BEND-01**: LlamaCppBackend connects to llama-server via httpx, reads native /completion timings
- [ ] **BEND-02**: LMStudioBackend connects to LM Studio via httpx, reads native /api/v1/ stats
- [ ] **BEND-03**: Auto-detect installed backends by checking binary presence (shutil.which)
- [ ] **BEND-04**: Auto-start backends if installed but not running (ollama serve, llama-server, lms server start)
- [ ] **BEND-05**: Backend-specific preflight checks (non-fatal: skip unavailable backends gracefully)

### CLI Integration (CLI)

- [ ] **CLI-01**: `--backend` flag accepts ollama, llama-cpp, lm-studio, all (default: ollama)
- [ ] **CLI-02**: Interactive menu shows detected backends and lets user choose after mode selection
- [ ] **CLI-03**: Backend name included in export filenames, JSON metadata, and Markdown reports
- [ ] **CLI-04**: System summary shows backend name and version
- [ ] **CLI-05**: Backend choice only prompted when >1 backend detected (no noise for Ollama-only users)

### Cross-Backend Comparison (COMP)

- [ ] **COMP-01**: `--backend all` runs same prompts on all detected backends sequentially
- [ ] **COMP-02**: Single-model comparison: one model tested on all backends, side-by-side bar chart
- [ ] **COMP-03**: Full matrix mode: N models × M backends, comparison table with winner per model
- [ ] **COMP-04**: "Fastest backend" recommendation per model and overall in comparison report
- [ ] **COMP-05**: Comparison mode as menu option 5 ("Compare backends")

### Cross-Platform (PLAT)

- [ ] **PLAT-01**: All backends work on macOS (Apple Silicon Metal), Windows (CUDA/CPU), Linux (CUDA/CPU)
- [ ] **PLAT-02**: llama.cpp install detection and auto-start per OS (Homebrew/winget/apt)
- [ ] **PLAT-03**: LM Studio install detection and auto-start per OS (no MLX on Linux noted)

### Documentation (DOC)

- [ ] **DOC-01**: README updated with multi-backend quick start and per-OS setup guides for all 3 backends
- [ ] **DOC-02**: Backend comparison example in README showing real cross-backend output

## v3 Requirements (Future)

### Advanced Benchmarking

- **ADVB-01**: Multi-turn conversation benchmark (not just single prompts)
- **ADVB-02**: Token quality assessment (not just speed)
- **ADVB-03**: Memory profiling during inference
- **ADVB-04**: Model size vs speed tradeoff visualization

### Advanced UX

- **ADVX-01**: HTML report with interactive charts
- **ADVX-02**: Web UI dashboard for results
- **ADVX-03**: Student results comparison/leaderboard

## Out of Scope

| Feature | Reason |
|---------|--------|
| nginx round-robin load balancing | Enterprise feature, too complex for course |
| dialog TUI dependency | Extra dependency; simple CLI menu is sufficient |
| 30+ environment variables | Too confusing; CLI flags + sensible defaults |
| Real-time streaming dashboard | Web UI is v3; terminal output sufficient |
| Docker packaging | Students need to understand local setup, not hide it |
| Cloud API backends (OpenAI, Anthropic) | Focus on local inference only |
| Custom llama.cpp compilation | Students use Homebrew/winget/apt prebuilt |
| llama-cpp-python binding | Tool talks to llama-server via HTTP, not C++ bindings |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STAB-01 | Phase 1 | Complete |
| STAB-02 | Phase 1 | Complete |
| STAB-03 | Phase 1 | Complete |
| STAB-04 | Phase 1 | Complete |
| STAB-05 | Phase 1 | Complete |
| STAB-06 | Phase 1 | Complete |
| QUAL-01 | Phase 1 | Complete |
| QUAL-02 | Phase 1 | Complete |
| QUAL-05 | Phase 1 | Complete |
| BENCH-01 | Phase 2 | Complete |
| BENCH-02 | Phase 2 | Complete |
| BENCH-07 | Phase 2 | Complete |
| UX-04 | Phase 2 | Complete |
| BENCH-03 | Phase 3 | Complete |
| BENCH-04 | Phase 3 | Complete |
| BENCH-05 | Phase 3 | Complete |
| BENCH-06 | Phase 3 | Complete |
| ANLZ-01 | Phase 3 | Complete |
| ANLZ-02 | Phase 3 | Complete |
| ANLZ-03 | Phase 3 | Complete |
| UX-01 | Phase 4 | Complete |
| UX-02 | Phase 4 | Complete |
| UX-03 | Phase 4 | Complete |
| UX-05 | Phase 4 | Complete |
| UX-06 | Phase 4 | Complete |
| QUAL-03 | Phase 4 | Complete |
| QUAL-04 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after initial definition*
