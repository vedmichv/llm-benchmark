# Domain Pitfalls

**Domain:** LLM benchmarking tool (Ollama-based, student-facing, cross-platform)
**Researched:** 2026-03-12

## Critical Pitfalls

Mistakes that cause rewrites, incorrect data, or student frustration severe enough to abandon the tool.

### Pitfall 1: Measuring Cold Load as Benchmark Performance

**What goes wrong:** The first inference request to a model includes model loading time (VRAM allocation, weight loading from disk). This can be 5-30 seconds depending on model size and hardware. If this is included in "tokens per second" metrics, the first run produces dramatically lower numbers than subsequent runs, corrupting averages.

**Why it happens:** Ollama's `total_duration` field includes `load_duration` on the first request. Developers sum durations or take first-run data without separating model load from inference.

**Consequences:** Students see wildly inconsistent numbers between first and second runs. They conclude the tool is broken, or worse, make hardware conclusions based on bad data. The existing codebase already includes `load_duration` in `total_duration` reporting (benchmark.py line 113) without clearly separating it.

**Prevention:**
- Implement mandatory warmup run(s) before collecting metrics. At minimum, one throwaway request with `num_predict: 1` to force model load.
- Exclude `load_duration` from throughput calculations (already partially done via separate `prompt_eval_duration` and `eval_duration`, but the `total_time` field still uses `total_duration` which includes load).
- Clearly label first-run vs steady-state metrics in output.

**Warning signs:** Large variance between run 1 and run 2 for the same model/prompt. `load_duration` values > 1 second.

**Detection:** Compare first-run `total_ts` against subsequent runs. If ratio > 2x, warmup is not working.

**Phase:** Should be addressed in the stability/accuracy phase (early). This is foundational to data quality.

---

### Pitfall 2: Prompt Caching Silently Corrupts Metrics

**What goes wrong:** Ollama caches prompt evaluation results. When the same prompt is sent to the same model twice, Ollama returns `prompt_eval_count: -1` and `prompt_eval_duration: 0` (or near-zero) on the second request. The current codebase silently converts -1 to 0, making prompt eval throughput appear as 0 t/s.

**Why it happens:** The current validator (benchmark.py lines 31-39) prints a warning but sets `prompt_eval_count = 0`. Downstream calculations then produce `0 / duration = 0 t/s` for prompt eval, and total throughput is also wrong because it only counts response tokens.

**Consequences:** With `runs_per_prompt > 1` (extended_benchmark.py default is 2), the second run always has corrupted prompt eval metrics. Averages are dragged down. Students see misleading numbers suggesting prompt processing is extremely slow.

**Prevention:**
- For multi-run benchmarks: either (a) vary prompts slightly between runs (add invisible whitespace or unique prefix), or (b) explicitly mark cached runs and exclude prompt eval metrics from their averages, or (c) unload and reload the model between runs to clear the KV cache.
- Never silently convert -1 to 0. Store it as a distinct "cached" state.
- Calculate averages only from non-cached prompt eval values.

**Warning signs:** `prompt_eval_count` returning -1 or 0 on any run after the first.

**Detection:** Flag in result output when prompt caching is detected. Add a `cached: bool` field to BenchmarkResult.

**Phase:** Must be fixed in the accuracy/measurement phase. This is the most impactful data-quality bug in the existing codebase.

---

### Pitfall 3: Concurrent Requests Competing for GPU Memory

**What goes wrong:** When adding concurrent benchmark mode (parallel requests), multiple simultaneous requests to Ollama compete for GPU VRAM. Ollama serializes requests internally for a single model, but if concurrent tests hit different models, Ollama may try to load multiple models simultaneously, causing OOM errors, model offloading to CPU, or Ollama crashes.

**Why it happens:** Developers implement concurrency at the Python level (threading/asyncio) without understanding that Ollama has its own model scheduling. Ollama's `OLLAMA_NUM_PARALLEL` setting controls concurrent request handling per model, but cross-model concurrency is constrained by VRAM.

**Consequences:** Benchmarks crash midway. Students with limited VRAM (8GB) see OOM errors. Worse: Ollama silently falls back to CPU inference when VRAM is exhausted, producing dramatically different (slower) numbers that look like the model is just slow.

**Prevention:**
- Concurrent testing should ONLY test multiple simultaneous requests to the SAME model (aggregate throughput). Never load multiple models concurrently.
- Before concurrent tests, check available VRAM and warn if < model size + overhead.
- Set `OLLAMA_NUM_PARALLEL` explicitly and document it.
- Monitor `ollama ps` during concurrent runs to verify model stays GPU-resident.
- Add a `--concurrent N` flag with sensible defaults (2-4) and a max cap.

**Warning signs:** Sudden drops in t/s during concurrent runs. Ollama returning errors about memory. `ollama ps` showing models on CPU when they should be on GPU.

**Detection:** Pre-flight check: query model size from `ollama show <model>`, compare against available VRAM.

**Phase:** Concurrent testing phase. Must be carefully designed from the start, not bolted onto sequential logic.

---

### Pitfall 4: Windows/WSL Path and Process Confusion

**What goes wrong:** Students on Windows may have Ollama installed natively (Windows service) or inside WSL. The Python tool may run natively on Windows or inside WSL. These four combinations (native Python + native Ollama, WSL Python + WSL Ollama, WSL Python + native Ollama, native Python + WSL Ollama) each behave differently. Lock files, temp paths, subprocess calls, and network endpoints all vary.

**Why it happens:** The codebase uses Unix-isms: `/tmp/` for lock files (line 59 of extended_benchmark.py), `os.kill(pid, 0)` for process checking (line 71), `journalctl` for logs (line 358), `sysctl` for system info. These work differently or not at all on native Windows.

**Consequences:**
- Lock file path `/tmp/ollama_benchmark.lock` does not exist on native Windows (should use `tempfile.gettempdir()` -- actually already does via the constant, but the error messages print Unix-style `rm` commands).
- `os.kill(pid, 0)` on Windows raises `PermissionError` for running processes, not `OSError` -- behavior differs from Unix.
- Students get confusing error messages telling them to run `rm /tmp/...` when they are on Windows.
- WSL Python connecting to native Windows Ollama requires `OLLAMA_HOST=http://host.docker.internal:11434` or similar -- not documented.

**Prevention:**
- Use `tempfile.gettempdir()` consistently (already done for the constant, but error messages should use it too).
- Abstract PID checking into a cross-platform function.
- Detect WSL (check `/proc/version` for "microsoft" or "WSL") and adjust Ollama connection accordingly.
- Replace `rm` commands in error messages with platform-appropriate instructions.
- Test all three platforms in CI.

**Warning signs:** Students reporting "lock file" errors on Windows. "Ollama not found" errors in WSL when Ollama is installed on Windows host.

**Detection:** CI matrix with Windows, macOS, and Linux. Manual testing on WSL.

**Phase:** Cross-platform stability phase. Should be one of the first things addressed.

---

### Pitfall 5: Averaging Throughput Rates Incorrectly

**What goes wrong:** The current code averages tokens-per-second values across runs: `avg_response_ts = sum(r.response_ts) / len(runs)`. This is mathematically wrong when runs have different token counts or durations. The correct approach is `total_tokens / total_time`, not `mean(tokens/time)`.

**Why it happens:** Averaging rates (ratios) is a common statistical error. If run 1 produces 100 tokens at 50 t/s (2 seconds) and run 2 produces 10 tokens at 10 t/s (1 second), the naive average is 30 t/s. The correct weighted average is 110 tokens / 3 seconds = 36.7 t/s.

**Consequences:** Results are biased toward runs with fewer tokens. Short responses disproportionately influence the average. Students comparing models may get misleading rankings.

**Prevention:**
- Calculate aggregate throughput as `sum(all_tokens) / sum(all_durations)` for each metric.
- If averaging rates, use harmonic mean or weighted average by token count.
- Display both per-run and aggregate metrics so students can see variance.

**Warning signs:** Average t/s that does not match manual calculation from total tokens and total time.

**Detection:** Add a sanity check: compare `avg_ts` against `total_tokens / total_time`. Flag if they differ by more than 5%.

**Phase:** Accuracy/measurement phase. Should be fixed alongside the prompt caching issue.

---

## Moderate Pitfalls

### Pitfall 6: Daemon Threads Leaking on Timeout

**What goes wrong:** The `run_with_timeout()` function (extended_benchmark.py lines 124-143) starts daemon threads. When a timeout occurs, the thread is abandoned (still running in the background). The Ollama request continues consuming resources. Over a long benchmark session with multiple timeouts, zombie threads accumulate.

**Prevention:**
- Use `concurrent.futures.ThreadPoolExecutor` with proper cancellation.
- After timeout, call `ollama.abort()` or unload the model to force-stop the in-flight request.
- Track active threads and warn if count exceeds a threshold.
- Consider subprocess-based isolation for individual benchmark runs (each run in a child process that can be killed).

**Phase:** Concurrent testing phase. Thread management must be solid before adding concurrency.

---

### Pitfall 7: No Control Over Generation Length

**What goes wrong:** Without setting `num_predict`, different models generate wildly different response lengths for the same prompt. One model might produce 50 tokens, another 2000. This makes throughput comparisons misleading because longer responses are more "warmed up" (higher sustained t/s) while shorter ones include more startup overhead per token.

**Prevention:**
- Add `--max-tokens` parameter that sets `num_predict` in Ollama options.
- For fair comparison benchmarks, set a fixed token count (e.g., 256 or 512).
- For "natural" benchmarks, let models generate freely but report token counts prominently alongside t/s.
- Always display response token count next to throughput so students understand the relationship.

**Warning signs:** Large variance in `eval_count` across models for the same prompt. Models with short responses appearing slower.

**Phase:** Parameter sweep phase. Natural integration point when adding configurable Ollama options.

---

### Pitfall 8: ANSI Color Codes Breaking on Windows CMD

**What goes wrong:** The current `run.py` uses Unicode symbols (checkmark, warning sign, info symbol) and ANSI escape codes. On Windows CMD (not PowerShell, not Windows Terminal), these render as garbage characters. The `Colors.disable()` method (line 27) checks for `ANSICON` but not for Windows Terminal or modern PowerShell which DO support ANSI.

**Prevention:**
- Use `os.environ.get('WT_SESSION')` to detect Windows Terminal.
- Use `colorama` library or check `sys.stdout.isatty()` combined with terminal capability detection.
- Replace Unicode symbols with ASCII fallbacks on unsupported terminals: checkmark -> `[OK]`, warning -> `[WARN]`, info -> `[INFO]` (partially already done in extended_benchmark.py).
- Test output in: Windows CMD, PowerShell, Windows Terminal, macOS Terminal, Linux terminal.

**Phase:** Cross-platform stability phase.

---

### Pitfall 9: Ollama API Version Drift

**What goes wrong:** The Ollama Python SDK is unpinned in requirements.txt. Ollama server updates independently of the SDK. Response field names, streaming behavior, and error types can change between versions. The code assumes specific response structure (e.g., `last_element.get("message", {}).get("content", "")` on line 532 of extended_benchmark).

**Prevention:**
- Pin `ollama>=0.4.0,<1.0.0` in requirements.txt (or whatever the current stable range is).
- Use the Pydantic model validation as a safety net (already done, which is good).
- Add a startup version check: query `ollama --version` and warn if outside tested range.
- When Ollama SDK returns `ChatResponse` objects (newer versions), handle both dict and object access patterns (partially done with `model_dump()` fallback).

**Warning signs:** `ValidationError` from Pydantic on response parsing. `AttributeError` on response objects.

**Phase:** Stability phase. Pin versions early, add version checking early.

---

### Pitfall 10: Students Not Having Models Pulled

**What goes wrong:** Students clone the repo, run the tool, and get "no models found" or immediate failures because they have not pulled any models with `ollama pull`. The tool lists all installed models, finds zero, and either errors out or produces empty results.

**Prevention:**
- Pre-flight check: if `ollama list` returns zero models, print clear instructions with specific model recommendations for their hardware tier.
- Suggest small models for low-RAM systems: `ollama pull qwen2.5:1.5b` (needs ~2GB).
- Add `--auto-pull` flag that pulls a default small model if none exists.
- Include hardware-based recommendations: "You have 8GB RAM, recommended models: ..." in pre-flight output.

**Warning signs:** Zero models returned from `ollama list`. Students posting "nothing happens" issues.

**Phase:** Student UX phase. Critical for first-run experience.

---

### Pitfall 11: Parameter Sweep Combinatorial Explosion

**What goes wrong:** When implementing parameter sweeps (auto-exploring `num_ctx`, `num_gpu`, `temperature`), the number of combinations grows multiplicatively. 3 models x 4 context sizes x 3 GPU layer counts x 3 temperatures = 108 benchmark runs. At 2-5 minutes each, that is 3-9 hours.

**Prevention:**
- Default sweep should be minimal: 2-3 values per parameter, only sweep one parameter at a time.
- Add `--sweep-budget` parameter (max total runs or max time).
- Show estimated runtime before starting sweep and require confirmation.
- Use Latin hypercube sampling or similar for multi-parameter exploration instead of full grid search.
- Make `temperature` NOT part of throughput sweeps (it does not affect inference speed, only output quality). Only sweep parameters that affect performance: `num_ctx`, `num_gpu`, `num_batch`.

**Warning signs:** Sweep taking more than 30 minutes on a student machine. Students killing the process midway.

**Phase:** Parameter sweep phase. Design the interface to prevent misuse.

---

### Pitfall 12: Thermal Throttling Corrupting Long Benchmarks

**What goes wrong:** During long benchmark sessions (especially parameter sweeps), GPU and CPU temperatures rise. Modern hardware throttles clock speeds at thermal limits. Later benchmark runs produce lower throughput than earlier ones, not because of the model or parameters, but because of hardware thermal state.

**Prevention:**
- Add optional temperature monitoring (nvidia-smi for GPU temp, platform-specific for CPU).
- Between model benchmarks, add a configurable cooldown delay (default: 5-10 seconds).
- Randomize benchmark order so thermal effects distribute evenly across models rather than penalizing later-tested models.
- Flag results where GPU temp exceeded throttle threshold during the run.
- Document in output: "Model X was tested at position N in the queue."

**Warning signs:** Gradual decline in t/s across sequential models. Last-tested model consistently scoring lowest.

**Detection:** Compare first-model vs last-model scores across multiple benchmark sessions with reversed order.

**Phase:** Accuracy/measurement phase. At minimum, add cooldown between models.

---

## Minor Pitfalls

### Pitfall 13: Lock File Path in Error Messages is Not Cross-Platform

**What goes wrong:** Error messages (extended_benchmark.py line 75) print `rm {LOCK_FILE}` which shows a Unix-style `rm` command. Windows students do not have `rm`.

**Prevention:** Use `platform.system()` to show `del` on Windows, `rm` on Unix. Or just say "Delete the file at: {path}".

**Phase:** Cross-platform stability phase.

---

### Pitfall 14: CSV Export Lacks Proper Escaping

**What goes wrong:** If prompts contain commas, quotes, or newlines, CSV output may be malformed. The current code uses Python's `csv` module (which handles this correctly), but any manual string concatenation for CSV would break.

**Prevention:** Always use the `csv` module's `writer` for CSV output. Never manually format CSV strings. Test with prompts containing commas and quotes.

**Phase:** Export/reporting phase. Verify with edge-case prompts.

---

### Pitfall 15: Ollama Connection Errors Not Distinguished from Model Errors

**What goes wrong:** When Ollama is not running, the Python SDK throws a `ConnectionError`. When a model fails to load (too large for RAM), Ollama returns a different error. Both are currently caught by broad `except Exception` blocks and reported as "Benchmark error" without distinguishing the cause.

**Prevention:**
- Catch `ConnectionError` / `httpx.ConnectError` specifically and print "Ollama is not running. Start it with `ollama serve`."
- Catch `ollama.ResponseError` for model-specific issues and print the actual error message from Ollama.
- Add pre-flight connectivity check before starting any benchmarks.

**Phase:** Stability phase. Better error messages are low-effort, high-impact.

---

### Pitfall 16: num_ctx Not Factored Into Benchmark Comparisons

**What goes wrong:** Different models have different default context window sizes (`num_ctx`). A model running with `num_ctx=2048` uses less VRAM and may be faster than the same model at `num_ctx=8192`. Students comparing models without controlling for context size are comparing apples to oranges.

**Prevention:**
- Query each model's default `num_ctx` via `ollama show <model>` and include it in results.
- When comparing models, normalize by showing the effective context size.
- In parameter sweep mode, make `num_ctx` the first parameter to explore.

**Phase:** Parameter sweep phase.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Cross-platform stability | Windows path/process differences (#4, #8, #13) | Abstract platform-specific code into utility functions. CI matrix with all 3 OS. Test WSL explicitly. |
| Measurement accuracy | Cold load (#1), prompt caching (#2), averaging math (#5) | Warmup runs, cache detection, weighted averages. These must be fixed before any other feature work -- bad data undermines everything. |
| Concurrent testing | GPU memory contention (#3), thread leaks (#6) | Same-model-only concurrency, pre-flight VRAM check, proper thread lifecycle. |
| Parameter sweeps | Combinatorial explosion (#11), thermal throttling (#12), context size (#16) | Budget caps, estimated runtime display, cooldown delays, randomized order. |
| Student UX | No models installed (#10), confusing errors (#15) | Pre-flight checks, hardware-based model recommendations, specific error messages. |
| Export/reporting | CSV edge cases (#14), misleading averages (#5) | Use csv module, display both per-run and aggregate metrics. |
| Ollama SDK stability | API version drift (#9) | Pin SDK version, add version check at startup. |

## Sources

- Direct codebase analysis of benchmark.py, extended_benchmark.py, run.py
- Known issues documented in .planning/codebase/CONCERNS.md
- Ollama API behavior: prompt caching issue documented at https://github.com/ollama/ollama/issues/2068 (referenced in codebase)
- Statistical averaging of rates: harmonic mean vs arithmetic mean is a well-established mathematical principle
- Confidence: HIGH for codebase-specific pitfalls (direct code evidence), MEDIUM for Ollama behavioral pitfalls (based on API documentation and codebase comments), MEDIUM for cross-platform issues (based on codebase patterns and platform knowledge)

---

*Pitfalls audit: 2026-03-12*
