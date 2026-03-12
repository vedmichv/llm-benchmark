# Technology Stack

**Analysis Date:** 2026-03-12

## Languages

**Primary:**
- Python 3.6+ - All core application logic, CLI tools, and utilities

**Secondary:**
- Bash - Cross-platform shell scripts for launcher (`.sh` files)
- PowerShell - Windows launcher script (`.ps1`)
- Batch - Windows launcher script (`.bat`)

## Runtime

**Environment:**
- Python interpreter (3.6 or higher required)
- Cross-platform support: Linux, macOS, Windows (WSL compatible)

**Package Manager:**
- pip - Python package management
- Lockfile: `requirements.txt` present

## Frameworks

**Core:**
- Ollama Python SDK - Communication with Ollama LLM inference server
- Pydantic 2.x - Data validation and model serialization

**CLI:**
- argparse - Standard library, command-line argument parsing

**Testing:**
- No formal testing framework detected (manual test scripts only)

**Build/Dev:**
- subprocess - Running external commands (Ollama CLI, system info collection)
- threading - Cross-platform timeout implementation for operations

## Key Dependencies

**Critical:**
- `ollama` - Python client SDK for Ollama inference server API
  - Used for: `ollama.chat()` with streaming support, `ollama.list()` for model enumeration
  - Lines: `benchmark.py:4`, `extended_benchmark.py:23`
- `pydantic` - Data validation using Python type hints
  - Used for: `BaseModel`, `Field`, `field_validator` for type-safe response parsing
  - Lines: `benchmark.py:5-8`, `extended_benchmark.py:24`

**Standard Library (used heavily):**
- `argparse` - CLI argument parsing
- `subprocess` - External command execution (Ollama CLI, system diagnostics)
- `platform` - OS detection and system information collection
- `json` - Benchmark result serialization
- `csv` - Results export
- `threading` - Cross-platform timeout handling
- `datetime` - Timestamp tracking
- `pathlib` - Cross-platform file path handling
- `os` - File and process management
- `tempfile` - Lock file management
- `signal` / `atexit` - Clean shutdown and process cleanup

## Configuration

**Environment:**
- No environment variables required for basic operation
- Ollama server must be running and accessible via default connection (localhost:11434)
- Lock file management: Uses `tempfile.gettempdir()` for concurrent benchmark prevention

**Build:**
- No build configuration files detected
- Direct Python script execution (no compilation step)

## Platform Requirements

**Development:**
- Python 3.6+ interpreter
- Ollama installed and running (`ollama serve`)
- No additional system dependencies beyond Python and Ollama

**Production:**
- Ollama server deployed and accessible
- Python 3.6+ runtime
- Cross-platform support with platform-specific launcher scripts:
  - `run.sh` - Linux/macOS launcher
  - `run.ps1` - Windows PowerShell launcher
  - `run.bat` - Windows batch launcher
  - `run.py` - Universal Python launcher

**System Information Collected:**
- CPU model and core count
- Total RAM
- GPU availability and VRAM (via nvidia-smi on Linux, system_profiler on macOS)
- CUDA version (if GPU available)
- OS and version information
- Hostname

---

*Stack analysis: 2026-03-12*
