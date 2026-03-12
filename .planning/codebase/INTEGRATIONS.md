# External Integrations

**Analysis Date:** 2026-03-12

## APIs & External Services

**Ollama LLM Inference Server:**
- Ollama HTTP API - Primary integration for LLM inference
  - SDK/Client: `ollama` Python package (PyPI)
  - Connection: Default localhost:11434 (configurable via Ollama environment)
  - Usage:
    - `ollama.chat()` - Execute inference on LLM models with optional streaming
    - `ollama.list()` - Enumerate available models on the server
  - Response Format: Structured responses validated via Pydantic models
  - Streaming Support: Optional streaming mode for real-time token output
  - Files: `benchmark.py:42-83`, `extended_benchmark.py:150-168`

## Data Storage

**Databases:**
- Not used - Stateless benchmarking application

**File Storage:**
- Local filesystem only
- Output formats:
  - JSON - Benchmark results saved as `results_*.json` in local directory
  - CSV - Results optionally exported as `.csv` files
  - Text - Console output with formatted statistics
- Lock file: Temporary file-based concurrency control (`{tempdir}/ollama_benchmark.lock`)
- Files: `extended_benchmark.py:300+` (JSON/CSV export logic)

**Caching:**
- Not used - No external caching layer
- Note: Ollama may use prompt caching (handled with validation warning at `benchmark.py:31-38`)

## Authentication & Identity

**Auth Provider:**
- None - Ollama assumes local/trusted network deployment
- No API keys, tokens, or credentials required

**Access Control:**
- Process-level concurrency control via lock files
- Prevents concurrent benchmark runs via PID checking (`extended_benchmark.py:61-94`)

## Monitoring & Observability

**Error Tracking:**
- None - Built-in error handling and reporting only

**Logs:**
- Console output only
- Verbose mode streaming: `benchmark.py:48-60`
- System info collection logged: `extended_benchmark.py:220+`
- Timing and performance metrics printed to console
- Files: `benchmark.py:90-116`, `extended_benchmark.py:` (inference_stats function)

**System Diagnostics:**
- CPU information: Via `/proc/cpuinfo` (Linux), `sysctl` (macOS), or `platform.processor()`
- Memory information: Via `/proc/meminfo` (Linux) or `sysctl` (macOS)
- GPU detection: Via `nvidia-smi` command on Linux
- GPU driver info: Via nvidia-smi or `system_profiler` on macOS
- Files: `extended_benchmark.py:220-320`

## CI/CD & Deployment

**Hosting:**
- Local machine or network-accessible Ollama server
- WSL (Windows Subsystem for Linux) supported with special handling
- Cross-platform launcher scripts for automated execution

**CI Pipeline:**
- None detected - Manual execution via CLI

**Launch Methods:**
- Direct: `python benchmark.py` or `python extended_benchmark.py`
- Universal launcher: `python run.py` (handles cross-platform setup)
- Platform-specific: `./run.sh` (Linux/macOS), `run.ps1` (PowerShell), `run.bat` (CMD)

## Environment Configuration

**Required env vars:**
- None strictly required for basic operation

**Optional env vars (Ollama-controlled):**
- `OLLAMA_HOST` - Ollama server address (defaults to localhost:11434)
- `OLLAMA_MODELS` - Model directory location
- `OLLAMA_GPU` - GPU configuration

**Secrets location:**
- No secrets management - Application is stateless and credential-free
- Note: Ollama authentication configured at server level (not in benchmark app)

## Webhooks & Callbacks

**Incoming:**
- None - Application does not expose any HTTP endpoints

**Outgoing:**
- None - Application only initiates requests to Ollama server

## External Command Execution

**System Calls:**
- `ollama --version` - Version detection (`extended_benchmark.py:232`)
- `ollama list` - Model enumeration (`benchmark.py:146`)
- `sysctl` commands - macOS system info (`extended_benchmark.py:254-288`)
- `nvidia-smi` - GPU detection on Linux (`extended_benchmark.py:298+`)
- `system_profiler` - GPU info on macOS

**Cross-Platform Support:**
- Platform detection: `platform.system()` for OS identification
- Conditional logic for Linux, macOS, Windows
- Timeout handling: Threading-based timeouts (compatible with all platforms)
- Files: `extended_benchmark.py:243-320`

---

*Integration audit: 2026-03-12*
