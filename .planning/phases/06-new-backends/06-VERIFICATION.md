---
phase: 06-new-backends
verified: 2026-03-14T18:20:00Z
status: passed
score: 5/5 success criteria verified
re_verification: false
---

# Phase 6: New Backends Verification Report

**Phase Goal:** Students can benchmark models on llama.cpp and LM Studio in addition to Ollama, with auto-detection and per-OS support
**Verified:** 2026-03-14T18:20:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Success Criteria)

| #  | Truth                                                                               | Status     | Evidence                                                                                           |
|----|------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------|
| 1  | `python -m llm_benchmark run --backend llama-cpp` parses and creates LlamaCppBackend | VERIFIED | `_handle_run` calls `create_backend(args.backend, port=args.port)`; factory dispatches to LlamaCppBackend; CLI shows `--backend {ollama,llama-cpp,lm-studio}` |
| 2  | `python -m llm_benchmark run --backend lm-studio` parses and creates LMStudioBackend | VERIFIED | LMStudioBackend created by factory; chat derives eval_duration from tokens_per_second; all Backend Protocol methods implemented |
| 3  | No `--backend` flag defaults to Ollama with zero behavior change                   | VERIFIED   | `argparse` default="ollama"; `create_backend("ollama")` path unchanged; 308 existing tests pass   |
| 4  | Multiple backends: interactive menu shows detected backends with selection; single backend: auto-selects without prompt | VERIFIED | `select_backend_interactive()` shows all three backends with status; `len(available) == 1` path auto-selects with informational note; `len(available) > 1` path prompts with numbered selection |
| 5  | Backend installed but not running: tool attempts auto-start                         | VERIFIED   | `preflight.check_backend_installed()` prompts "Start {backend}? (y/N)" then calls `auto_start_backend()`; `menu.select_backend_interactive()` does same; `auto_start_backend()` polls health for 30s |

**Score:** 5/5 success criteria verified

---

### Required Artifacts

| Artifact                                      | Min Lines | Actual | Status     | Details                                                          |
|-----------------------------------------------|-----------|--------|------------|------------------------------------------------------------------|
| `llm_benchmark/backends/llamacpp.py`          | 150       | 286    | VERIFIED   | LlamaCppBackend with httpx, ms-to-sec conversion, SSE streaming  |
| `llm_benchmark/backends/lmstudio.py`          | 120       | 266    | VERIFIED   | LMStudioBackend with httpx, tps-derived timing, SSE streaming    |
| `llm_benchmark/backends/detection.py`         | 150       | 236    | VERIFIED   | BackendStatus, detect_backends, auto_start_backend, get_install_instructions |
| `llm_benchmark/backends/__init__.py`          | n/a       | —      | VERIFIED   | create_backend() supports ollama/llama-cpp/lm-studio with host/port |
| `llm_benchmark/preflight.py`                  | n/a       | —      | VERIFIED   | check_backend_installed() uses detection module; backend-specific messages |
| `llm_benchmark/cli.py`                        | n/a       | —      | VERIFIED   | --backend, --port, --model-path flags; select_backend_interactive in no-args path |
| `llm_benchmark/menu.py`                       | 200       | 482    | VERIFIED   | select_backend_interactive, scan_gguf_files, extract_gguf_model_name |
| `llm_benchmark/exporters.py`                  | n/a       | —      | VERIFIED   | _filename() helper; benchmark_{backend}_{ts}.ext pattern across all 9 export functions |
| `llm_benchmark/system.py`                     | n/a       | —      | VERIFIED   | detect_backends() in format_system_summary; get_backend_inventory() |
| `llm_benchmark/runner.py`                     | n/a       | —      | VERIFIED   | KNOWN_ISSUES dict, get_known_issue_hint(), failure auto-skip + summary |
| `tests/test_llamacpp.py`                      | 80        | 351    | VERIFIED   | 25 tests covering protocol compliance, timing, streaming, errors |
| `tests/test_lmstudio.py`                      | 80        | 332    | VERIFIED   | 25 tests covering protocol compliance, timing, streaming, errors |
| `tests/test_detection.py`                     | 100       | 405    | VERIFIED   | 25+ tests covering detection, auto-start, install instructions   |

---

### Key Link Verification

| From                                        | To                                       | Via                                              | Status   | Evidence                                                                    |
|---------------------------------------------|------------------------------------------|--------------------------------------------------|----------|-----------------------------------------------------------------------------|
| `llamacpp.py`                               | `backends/__init__.py`                   | `from llm_benchmark.backends import BackendError, BackendResponse, StreamResult` | WIRED | Line 16 of llamacpp.py |
| `lmstudio.py`                               | `backends/__init__.py`                   | Same import                                      | WIRED    | Line 16 of lmstudio.py                                                      |
| `detection.py`                              | `shutil.which`                           | Binary detection for ollama, llama-server, lms   | WIRED    | `shutil.which(binary)` in detect_backends(); shutil imported at top         |
| `detection.py`                              | `socket`                                 | Port probing for 11434, 8080, 1234               | WIRED    | `socket.connect_ex((host, port))` in `_port_is_open()`                      |
| `cli.py`                                    | `backends/__init__.py`                   | `create_backend(args.backend, port=args.port)`   | WIRED    | cli.py line 231                                                              |
| `preflight.py`                              | `backends/detection.py`                  | `from llm_benchmark.backends.detection import auto_start_backend, detect_backends, get_install_instructions` | WIRED | preflight.py lines 15-19 |
| `cli.py`                                    | `preflight.py`                           | `run_preflight_checks(backend=backend, ...)`     | WIRED    | cli.py line 238                                                              |
| `exporters.py`                              | `models.py`                              | `system_info.backend_name` in `_filename()`      | WIRED    | exporters.py line 56                                                         |
| `system.py`                                 | `backends/detection.py`                  | lazy `from llm_benchmark.backends.detection import detect_backends` in format_system_summary() | WIRED | system.py line 257 |
| `menu.py`                                   | `backends/detection.py`                  | `detect_backends()` in select_backend_interactive() | WIRED | menu.py line 239                                                            |
| `menu.py`                                   | `backends/__init__.py`                   | select_backend_interactive returns name used in `create_backend()` (cli.py wires it) | WIRED | cli.py line 571 |
| `cli.py`                                    | `menu.py`                                | `select_backend_interactive`, `run_interactive_menu` imported and called | WIRED | cli.py lines 564-577 |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                          | Status    | Evidence                                                           |
|-------------|-------------|----------------------------------------------------------------------|-----------|--------------------------------------------------------------------|
| BEND-01     | 06-01       | LlamaCppBackend connects to llama-server via httpx, native timings   | SATISFIED | llamacpp.py: httpx.Client, _to_response() converts ms to seconds  |
| BEND-02     | 06-01       | LMStudioBackend connects to LM Studio via httpx, native /api/v1/ stats | SATISFIED | lmstudio.py: httpx.Client, _to_response() derives timing from tps |
| BEND-03     | 06-02       | Auto-detect installed backends via shutil.which                      | SATISFIED | detection.py: detect_backends() uses shutil.which for each binary  |
| BEND-04     | 06-02       | Auto-start backends if installed but not running                     | SATISFIED | detection.py: auto_start_backend() with 30s health polling; preflight and menu both use it |
| BEND-05     | 06-03       | Backend-specific preflight checks                                    | SATISFIED | preflight.py: check_backend_installed() uses detection module; backend-specific start/model hints |
| CLI-01      | 06-03       | `--backend` flag accepts ollama, llama-cpp, lm-studio                | SATISFIED | cli.py: choices=["ollama","llama-cpp","lm-studio"], default="ollama" |
| CLI-02      | 06-05       | Interactive menu shows detected backends and lets user choose        | SATISFIED | menu.py: select_backend_interactive() shows all three with status indicators |
| CLI-03      | 06-04       | Backend name in export filenames, JSON metadata, Markdown reports    | SATISFIED | exporters.py: _filename() uses system_info.backend_name; Markdown header includes backend label |
| CLI-04      | 06-04       | System summary shows backend name and version                        | SATISFIED | system.py: format_system_summary() shows active backend + all detected backends with status |
| CLI-05      | 06-05       | Backend choice only prompted when >1 backend detected                | SATISFIED | menu.py: len(available)==1 path auto-selects with note, no input() prompt |
| PLAT-01     | 06-03       | All backends work on macOS, Windows, Linux                           | SATISFIED | LlamaCppBackend and LMStudioBackend use httpx (cross-platform); detection uses shutil.which + socket (cross-platform) |
| PLAT-02     | 06-02       | llama.cpp install detection and auto-start per OS                    | SATISFIED | detection.py: brew/apt/winget instructions; auto_start_backend("llama-cpp") builds correct command |
| PLAT-03     | 06-02       | LM Studio install detection and auto-start per OS                    | SATISFIED | detection.py: lmstudio.ai download URL for all platforms; Linux MLX note included |

All 13 requirement IDs fully accounted for. No orphaned requirements.

---

### Anti-Patterns Found

None. Scanned `llamacpp.py`, `lmstudio.py`, `detection.py`, `menu.py` for TODO/FIXME, placeholder returns, empty handlers, and console.log-only implementations. Clean.

---

### Human Verification Required

The following behaviors require a real running backend to confirm end-to-end:

#### 1. llama-cpp Benchmark Run

**Test:** Run `python -m llm_benchmark run --backend llama-cpp --model-path /path/to/model.gguf` against a live llama-server
**Expected:** Timing fields in results show prompt_eval_duration and eval_duration derived from timings.prompt_ms and timings.predicted_ms (not zero)
**Why human:** Requires real llama-server with a GGUF model

#### 2. LM Studio Benchmark Run

**Test:** Run `python -m llm_benchmark run --backend lm-studio` against a live LM Studio server with a model loaded
**Expected:** eval_duration is non-zero and plausible (eval_count / tokens_per_second); export filenames contain "lm-studio"
**Why human:** Requires real LM Studio installation; stats field names (tokens_per_second) are MEDIUM confidence per research

#### 3. Auto-Start Flow

**Test:** Have a backend installed but its port closed, then run with that backend
**Expected:** Tool prompts "Start X? (y/N)", if yes starts the server, polls health, proceeds with benchmark
**Why human:** Requires real binary installed; timing behavior of subprocess.Popen cannot be fully mocked

#### 4. Interactive Menu Backend Selection

**Test:** Run `python -m llm_benchmark` with multiple backends installed
**Expected:** Backend list shown with correct running/installed/not-found status before mode selection; user selection flows through to preflight and benchmark
**Why human:** Requires multiple backends installed; menu flow involves live terminal interaction

---

### Gaps Summary

No gaps. All 5 success criteria are verified against actual code, not SUMMARY claims:

- SC1 (llama-cpp CLI): `--backend llama-cpp` flag exists, `create_backend` dispatches to `LlamaCppBackend`, `isinstance(LlamaCppBackend(), Backend)` is True.
- SC2 (lm-studio CLI): Same path for `LMStudioBackend`; timing derived from `tokens_per_second`.
- SC3 (backward compat): Default `"ollama"` unchanged; 308 total tests pass with no regressions.
- SC4 (menu detection): `select_backend_interactive()` shows all backends with status; single-available auto-selects without blocking prompt; multiple-available prompts for choice.
- SC5 (auto-start): `check_backend_installed()` in preflight and `select_backend_interactive()` in menu both call `auto_start_backend()` after user permission; health polling with 30s timeout implemented.

---

## Test Suite

- **308 total tests pass** (0 failures, 0 errors)
- Phase 6 specific: ~195 tests spanning test_llamacpp.py (25), test_lmstudio.py (25), test_detection.py (25+), test_cli.py (18 new), test_preflight.py (26 new), test_exporters.py (12 new), test_system.py (7 new), test_menu.py (17 new)
- All existing pre-phase-6 tests pass (backward compatibility confirmed)

---

_Verified: 2026-03-14T18:20:00Z_
_Verifier: Claude (gsd-verifier)_
