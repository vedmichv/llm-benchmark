# Research Summary: LLM Benchmark v2

**Domain:** LLM benchmarking tool (Ollama-based, student-facing)
**Researched:** 2026-03-12
**Overall confidence:** HIGH

## Executive Summary

The v2 improvements to this LLM benchmarking tool -- concurrent benchmarking, parameter sweep, retry logic, and enhanced analysis -- are well-served by a minimal stack addition of just two new runtime dependencies: `rich` for terminal UI and `tenacity` for retry logic. The critical insight is that the Ollama Python SDK already ships with an `AsyncClient` class built on httpx, so concurrent benchmarking requires zero new HTTP libraries. The project can use `asyncio.gather()` with a semaphore and the existing SDK.

The Python minimum version should be bumped to 3.10. The stated 3.6+ is already fiction (Pydantic 2.x needs 3.9+), and tenacity 9.x requires 3.10+. Students on a current DevOps course will have 3.10+ available on all platforms.

The reference repo (alexziskind1/llama-throughput-lab) uses a stdlib-heavy approach with raw HTTP against llama.cpp server. This project's advantage is the Ollama SDK abstraction, which simplifies concurrent code significantly via AsyncClient. The reference repo's architectural patterns (sweep configs, CSV analysis) are worth borrowing, but its dependency choices are not -- it targets a different runtime (llama.cpp directly) and uses `dialog` for TUI, which PROJECT.md explicitly excludes.

Rich is the clear choice for terminal output. It replaces manual print formatting with progress bars, styled tables for ranked results, and colored console output -- all from one dependency. This covers the "visual ranked results" and "progress tracking" requirements without pulling in a full TUI framework.

## Key Findings

**Stack:** Add `rich` and `tenacity` to existing `ollama` + `pydantic`. Use stdlib `asyncio` + Ollama's `AsyncClient` for concurrency. Total: 4 runtime dependencies.
**Architecture:** AsyncClient + asyncio.gather for concurrent benchmarks; Pydantic models for sweep configs; tenacity decorators for retry.
**Critical pitfall:** Do not bypass the Ollama SDK with raw httpx/aiohttp -- the SDK handles response parsing, streaming, and model management. Going around it duplicates work.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Code consolidation and foundation** - Merge two benchmark files, bump Python to 3.10, add rich for output
   - Addresses: Code consolidation, results directory structure, visual output
   - Avoids: Building new features on a fragmented codebase

2. **Retry and robustness** - Add tenacity retry, warmup runs, error handling
   - Addresses: Retry with backoff, warmup runs, cross-platform stability
   - Avoids: Building concurrent features without error handling foundation

3. **Concurrent benchmarking** - AsyncClient-based parallel requests
   - Addresses: Concurrent benchmark mode, progress bars during long runs
   - Avoids: Concurrency bugs by building on stable retry/error handling

4. **Parameter sweep and analysis** - Sweep configs, enhanced results, reports
   - Addresses: Parameter sweep, enhanced analysis, HTML/Markdown reports, quick verification mode
   - Avoids: Feature overload in earlier phases

**Phase ordering rationale:**
- Consolidation first because building concurrent features across two separate benchmark files is a maintenance trap
- Retry before concurrency because concurrent code without error handling produces confusing failures
- Sweep after concurrency because sweeps benefit from concurrent execution

**Research flags for phases:**
- Phase 3 (concurrency): May need deeper research on Ollama server behavior under concurrent load (does it queue or reject?)
- Phase 4 (parameter sweep): Standard patterns, unlikely to need research

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified on PyPI with release dates |
| Features | HIGH | Requirements clearly defined in PROJECT.md |
| Architecture | HIGH | AsyncClient pattern documented in Ollama SDK README |
| Pitfalls | MEDIUM | Ollama server concurrent behavior not fully verified |

## Gaps to Address

- Ollama server behavior under concurrent load (queuing, max connections, error responses) needs testing during Phase 3
- Exact rich progress bar integration pattern with async generators needs prototyping
- Whether `keep_alive=0` model offloading works correctly during concurrent requests

---

*Researched: 2026-03-12*
