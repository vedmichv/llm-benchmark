# Roadmap: LLM Benchmark v2

## Milestones

- v1.0 Benchmark Tool - Phases 1-4 (shipped 2026-03-13)
- v2.0 Multi-Backend Benchmark - Phases 5-7 (in progress)

## Phases

<details>
<summary>v1.0 Benchmark Tool (Phases 1-4) - SHIPPED 2026-03-13</summary>

- [x] **Phase 1: Foundation** - Consolidate codebase into package structure with cross-platform stability and pre-flight checks
- [x] **Phase 2: Measurement Reliability** - Warmup, retry, correct averaging, organized output
- [x] **Phase 3: Advanced Benchmarking** - Concurrent mode, parameter sweep, results analysis
- [x] **Phase 4: Student Experience** - Interactive menu, visual results, shareable reports, tests, CI

</details>

### v2.0 Multi-Backend Benchmark

- [ ] **Phase 5: Backend Abstraction** - Extract Backend Protocol and OllamaBackend adapter with zero user-visible change
- [ ] **Phase 6: New Backends** - Add llama.cpp and LM Studio backends with CLI integration and cross-platform support
- [ ] **Phase 7: Cross-Backend Comparison** - Full comparison mode with matrix benchmarking, reports, and documentation

## Phase Details

### Phase 5: Backend Abstraction
**Goal**: The entire codebase talks to a Backend protocol instead of Ollama directly, with zero behavior change for existing users
**Depends on**: Phase 4
**Requirements**: BACK-01, BACK-02, BACK-03, BACK-04, BACK-05
**Success Criteria** (what must be TRUE):
  1. Running `python -m llm_benchmark run` produces identical output to v1.0 -- no user-visible change whatsoever
  2. All 152+ existing tests pass without modification to test assertions (only import paths may change)
  3. `runner.py` contains zero direct `ollama.chat()` calls -- all inference goes through a Backend instance
  4. A `BackendResponse` model exists that normalizes all timing data to seconds, and OllamaBackend converts nanoseconds internally
**Plans**: TBD

### Phase 6: New Backends
**Goal**: Students can benchmark models on llama.cpp and LM Studio in addition to Ollama, with auto-detection and per-OS support
**Depends on**: Phase 5
**Requirements**: BEND-01, BEND-02, BEND-03, BEND-04, BEND-05, CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, PLAT-01, PLAT-02, PLAT-03
**Success Criteria** (what must be TRUE):
  1. Student can run `python -m llm_benchmark run --backend llama-cpp` and get benchmark results with native timing metrics from llama-server
  2. Student can run `python -m llm_benchmark run --backend lm-studio` and get benchmark results with native timing metrics from LM Studio
  3. Running with no `--backend` flag defaults to Ollama with no change in behavior (backward compatible)
  4. When multiple backends are installed, the interactive menu shows detected backends and lets the user choose; when only Ollama is available, no backend prompt appears
  5. If a backend is installed but not running, the tool attempts to auto-start it (ollama serve, llama-server, lms server start)
**Plans**: TBD

### Phase 7: Cross-Backend Comparison
**Goal**: Students can compare the same model across all backends side-by-side and see which runtime is fastest on their hardware
**Depends on**: Phase 6
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-04, COMP-05, DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. Student can run `--backend all` and the tool benchmarks the same prompts on every detected backend sequentially, producing a unified report
  2. Comparison report shows a side-by-side bar chart (single model) or matrix table (N models x M backends) with a "fastest backend" winner per model
  3. Full matrix mode (N models x M backends) produces a comparison table with winner highlighted per model and an overall recommendation
  4. "Compare backends" appears as menu option 5 in the interactive menu
  5. README includes multi-backend quick start, per-OS setup guides for all 3 backends, and a real cross-backend comparison output example

## Progress

**Execution Order:**
Phases execute in numeric order: 5 -> 6 -> 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 3/3 | Complete | 2026-03-12 |
| 2. Measurement Reliability | v1.0 | 2/2 | Complete | 2026-03-12 |
| 3. Advanced Benchmarking | v1.0 | 4/4 | Complete | 2026-03-13 |
| 4. Student Experience | v1.0 | 3/3 | Complete | 2026-03-13 |
| 5. Backend Abstraction | v2.0 | 0/? | Not started | - |
| 6. New Backends | v2.0 | 0/? | Not started | - |
| 7. Cross-Backend Comparison | v2.0 | 0/? | Not started | - |
