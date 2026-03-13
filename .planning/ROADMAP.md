# Roadmap: LLM Benchmark v2

## Overview

Transform the existing LLM benchmark tool from a working-but-fragile single-script prototype into a reliable, consolidated Python package that students can clone and run on any platform. The journey moves from stabilizing the foundation (consolidation, cross-platform, pre-flight checks), through measurement reliability (warmup, retry, correct metrics), into advanced benchmarking capabilities (concurrent mode, parameter sweep, analysis), and finishes with student-facing polish (interactive menu, visual results, tests, CI).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Consolidate codebase into package structure with cross-platform stability and pre-flight checks (completed 2026-03-12)
- [ ] **Phase 2: Measurement Reliability** - Ensure benchmark numbers are accurate with warmup, retry, correct averaging, and organized output
- [ ] **Phase 3: Advanced Benchmarking** - Add concurrent mode, parameter sweep, and results analysis capabilities
- [ ] **Phase 4: Student Experience** - Interactive CLI, visual results, shareable reports, tests, and CI

## Phase Details

### Phase 1: Foundation
**Goal**: Students can install and run the tool on any platform without crashes, with clear errors when something is wrong
**Depends on**: Nothing (first phase)
**Requirements**: STAB-01, STAB-02, STAB-03, STAB-04, STAB-05, STAB-06, QUAL-01, QUAL-02, QUAL-05
**Success Criteria** (what must be TRUE):
  1. Student can clone the repo and run `python -m llm_benchmark` on Windows, macOS, or Linux without import errors or platform crashes
  2. If Ollama is not running, the tool prints a clear message telling the student how to start it (not a stack trace)
  3. If the student's machine has insufficient RAM for a selected model, they see a warning before the benchmark starts
  4. There is one benchmark module (no confusion between benchmark.py and extended_benchmark.py)
  5. The project uses Python 3.10+ with Pydantic 2.x and has a proper package layout (llm_benchmark/)
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md -- Package scaffold, pyproject.toml, data models, config, prompts
- [ ] 01-02-PLAN.md -- Core benchmark runner, system info, exporters, compare
- [ ] 01-03-PLAN.md -- CLI entry points, pre-flight checks, cleanup old files

### Phase 2: Measurement Reliability
**Goal**: Benchmark numbers are trustworthy -- warmup excludes cold-start overhead, transient failures retry automatically, and metrics are mathematically correct
**Depends on**: Phase 1
**Requirements**: BENCH-01, BENCH-02, BENCH-07, UX-04
**Success Criteria** (what must be TRUE):
  1. Running a benchmark shows a warmup pass before measurement begins, and the reported throughput excludes model load time
  2. If Ollama returns a transient error mid-benchmark, the tool retries automatically (up to configurable limit) instead of crashing
  3. When prompt caching is detected (prompt_eval_count = -1), the affected metrics are excluded from averages with a visible note
  4. All result files (JSON, CSV, Markdown) are saved into a results/ directory, not the project root
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md -- Warmup runs, tenacity retry logic, --skip-warmup and --max-retries CLI flags
- [ ] 02-02-PLAN.md -- Cache visibility in terminal and exports, results/.gitignore, duplicate cleanup

### Phase 3: Advanced Benchmarking
**Goal**: Students can stress-test models with concurrent requests and automatically find the best configuration for their hardware
**Depends on**: Phase 2
**Requirements**: BENCH-03, BENCH-04, BENCH-05, BENCH-06, ANLZ-01, ANLZ-02, ANLZ-03
**Success Criteria** (what must be TRUE):
  1. Student can run `--concurrent N` and see aggregate throughput (total tokens/wall time) for N parallel requests to the same model
  2. Student can run `--sweep` and the tool automatically tests combinations of num_ctx and num_gpu, reporting the best configuration found
  3. Student can sort benchmark results by any metric and filter to top-N models from saved results
  4. Student can compare results from two different runs side-by-side to see how hardware or config changes affected performance
**Plans**: 4 plans

Plans:
- [ ] 03-01-PLAN.md -- Concurrent benchmarking module (data models, async orchestration, tests)
- [ ] 03-02-PLAN.md -- Parameter sweep module (num_ctx/num_gpu sweep, best config detection)
- [ ] 03-03-PLAN.md -- Analyze subcommand and compare enhancements (sort, filter, arrows, winner)
- [ ] 03-04-PLAN.md -- CLI wiring and mode-aware exporters (integrate all Phase 3 modules)

### Phase 4: Student Experience
**Goal**: Students who have never used CLI tools can run benchmarks through an interactive menu, see visual ranked results, and share reports with classmates
**Depends on**: Phase 3
**Requirements**: UX-01, UX-02, UX-03, UX-05, UX-06, QUAL-03, QUAL-04
**Success Criteria** (what must be TRUE):
  1. Running the tool with no arguments presents an interactive menu (quick test / standard / full / custom) that requires no CLI knowledge
  2. Quick test mode completes in roughly 30 seconds and confirms "everything works" with the smallest available model
  3. After a benchmark completes, the student sees a ranked bar chart in the terminal showing model comparison
  4. Results include system info, model rankings, and a recommendation for which model/config is best for this hardware
  5. Unit tests exist with mocked Ollama covering core functions (>60% coverage) and GitHub Actions runs lint + tests on push
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 3/3 | Complete   | 2026-03-12 |
| 2. Measurement Reliability | 0/2 | Not started | - |
| 3. Advanced Benchmarking | 0/4 | Not started | - |
| 4. Student Experience | 0/0 | Not started | - |
