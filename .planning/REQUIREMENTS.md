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
- [ ] **BENCH-03**: User can run concurrent benchmark with --concurrent N (parallel requests to same model)
- [ ] **BENCH-04**: Concurrent mode reports aggregate throughput (total_tokens/wall_time) and per-request average
- [ ] **BENCH-05**: User can run parameter sweep with --sweep (auto-explore num_ctx, num_gpu combinations)
- [ ] **BENCH-06**: Sweep reports best configuration found with throughput numbers
- [x] **BENCH-07**: Prompt caching detection excludes affected metrics from averages (instead of silently corrupting)

### Analysis (ANLZ)

- [x] **ANLZ-01**: User can sort benchmark results by any metric (response_ts, total_ts, load_time)
- [x] **ANLZ-02**: User can filter top-N results from CSV/JSON output
- [x] **ANLZ-03**: User can compare results from multiple runs side-by-side

### UX (UX)

- [ ] **UX-01**: Running tool with no arguments shows interactive menu (quick test / standard / full / custom)
- [ ] **UX-02**: Quick test mode runs ~30 seconds: smallest model, 1 prompt, confirms "everything works"
- [ ] **UX-03**: End of benchmark shows ranked model comparison with visual bar chart in terminal
- [x] **UX-04**: All result files saved to results/ directory (not project root)
- [ ] **UX-05**: Results include system info, model rankings, and recommendations for optimal config
- [ ] **UX-06**: Shareable report format (Markdown with system info + rankings + individual runs)

### Code Quality (QUAL)

- [x] **QUAL-01**: Single consolidated benchmark module (no benchmark.py vs extended_benchmark.py confusion)
- [x] **QUAL-02**: Python package structure (llm_benchmark/ with submodules)
- [ ] **QUAL-03**: Unit tests with mocked Ollama for core functions (>60% coverage)
- [ ] **QUAL-04**: GitHub Actions CI running lint (ruff) + compile check + unit tests
- [x] **QUAL-05**: Python >=3.10 requirement (Pydantic 2.x + tenacity compatibility)

## v2 Requirements

### Advanced Benchmarking

- **ADVB-01**: Multi-turn conversation benchmark (not just single prompts)
- **ADVB-02**: Token quality assessment (not just speed)
- **ADVB-03**: Memory profiling during inference
- **ADVB-04**: Model size vs speed tradeoff visualization

### Advanced UX

- **ADVX-01**: HTML report with interactive charts
- **ADVX-02**: Web UI dashboard for results
- **ADVX-03**: Automated model recommendations based on hardware profile
- **ADVX-04**: Student results comparison/leaderboard

## Out of Scope

| Feature | Reason |
|---------|--------|
| llama.cpp server direct support | Ollama abstracts this; students shouldn't need to build from source |
| nginx round-robin load balancing | Enterprise feature, too complex for course |
| dialog TUI dependency | Extra dependency; simple CLI menu is sufficient |
| 30+ environment variables | Too confusing; CLI flags + sensible defaults |
| Real-time streaming dashboard | Web UI is v2; terminal output is sufficient for v1 |
| Docker packaging | Students need to understand local setup, not hide it |

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
| BENCH-03 | Phase 3 | Pending |
| BENCH-04 | Phase 3 | Pending |
| BENCH-05 | Phase 3 | Pending |
| BENCH-06 | Phase 3 | Pending |
| ANLZ-01 | Phase 3 | Complete |
| ANLZ-02 | Phase 3 | Complete |
| ANLZ-03 | Phase 3 | Complete |
| UX-01 | Phase 4 | Pending |
| UX-02 | Phase 4 | Pending |
| UX-03 | Phase 4 | Pending |
| UX-05 | Phase 4 | Pending |
| UX-06 | Phase 4 | Pending |
| QUAL-03 | Phase 4 | Pending |
| QUAL-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-03-12 after initial definition*
