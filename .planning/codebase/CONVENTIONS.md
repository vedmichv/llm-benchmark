# Coding Conventions

**Analysis Date:** 2026-03-12

## Naming Patterns

**Files:**
- `snake_case` for all Python files: `benchmark.py`, `extended_benchmark.py`, `compare_results.py`, `test_ollama.py`, `run.py`
- Main entry points use descriptive names: `benchmark.py`, `extended_benchmark.py`
- Test files use `test_` prefix: `test_ollama.py`

**Functions:**
- `snake_case` for all function names
- Descriptive action-based names: `run_benchmark()`, `get_benchmark_models()`, `inference_stats()`, `average_stats()`, `check_python()`, `check_ollama()`, `install_dependencies()`
- Utility functions prefixed by operation: `nanosec_to_sec()`, `run_with_timeout()`
- Getter functions prefixed with `get_`: `get_benchmark_models()`, `get_venv_python()`
- Checker functions prefixed with `check_`: `check_python()`, `check_ollama()`, `check_dependencies()`

**Variables:**
- `snake_case` for all variables and parameters
- Clear, explicit names: `model_name`, `prompt`, `verbose`, `skip_models`, `prompts`, `responses`, `benchmarks`
- Temporary loop variables use single letters: `r`, `m`, `v`, `c` for short loops
- Constants in `UPPER_SNAKE_CASE`: `PROMPT_SETS`, `LOCK_FILE`

**Types/Classes:**
- `PascalCase` for all class names: `Message`, `OllamaResponse`, `BenchmarkResult`, `ModelBenchmarkSummary`, `Colors`
- Descriptive, noun-based names indicating what the class represents

## Code Style

**Formatting:**
- No formatter explicitly configured (flake8, black, isort not present)
- Line length appears to be ~80-100 characters based on code patterns
- Consistent 4-space indentation
- No trailing whitespace observed
- Functions separated by 2 blank lines at module level

**Linting:**
- No linting configuration files present (no `.flake8`, `.pylintrc`, `.isort.cfg`)
- Code follows PEP 8 style conventions implicitly:
  - 2 blank lines between top-level functions
  - 1 blank line between methods
  - Consistent spacing around operators

## Import Organization

**Order:**
1. Standard library imports: `argparse`, `sys`, `os`, `json`, `csv`, `time`, `subprocess`, `tempfile`, `threading`, `signal`, `platform`, `atexit`
2. Third-party imports: `ollama`, `pydantic`
3. Local imports: None present in this codebase

**Path Aliases:**
- Not used in this codebase
- All imports are absolute (no relative imports)

**Example from `benchmark.py` (lines 1-11):**
```python
import argparse
from typing import List

import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from datetime import datetime
```

Note: Mixed order observed (datetime imported after pydantic), but generally standard library → third-party pattern

## Error Handling

**Patterns:**
- Try-except blocks used for external system calls and file operations
- Specific exception types caught: `FileNotFoundError`, `json.JSONDecodeError`, `subprocess.TimeoutExpired`, `Exception` for fallback
- Explicit error messages printed to console with context
- Functions return `False` or `None` on error; `True` on success
- Error status communicated through return codes in `sys.exit()` calls
- Pydantic field validators used for data validation (`validate_prompt_eval_count`)

**Example from `run.py` (lines 192-208):**
```python
try:
    # Upgrade pip quietly
    subprocess.run(
        [python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'],
        capture_output=True,
        timeout=60
    )

    # Install requirements
    result = subprocess.run(
        [python_path, '-m', 'pip', 'install', '-r', str(requirements_file)],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode == 0:
        print_success("Dependencies installed")
        return True
```

## Logging

**Framework:** `print()` statements only (no logging module)

**Patterns:**
- Direct console output via `print()` for all messages
- Simple print statements for basic output: `print(f"Evaluating models: {model_names}\n")`
- Helper functions for formatted output in `run.py`: `print_header()`, `print_success()`, `print_error()`, `print_warning()`, `print_info()`
- Formatted strings with f-strings and escape sequences for tables/separators
- ANSI color codes used for terminal output in `run.py` (lines 14-36)

**Example from `run.py` (lines 44-50):**
```python
def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")
```

## Comments

**When to Comment:**
- Module-level docstrings used on all scripts: `"""Extended LLM Benchmark Script..."""`
- Inline comments used for non-obvious logic and clarifications
- Comments explain the "why" not the "what"

**JSDoc/TSDoc:**
- Not applicable (Python project)
- Docstrings used for functions and classes in Pydantic models
- Standard docstring format: triple quotes with description

**Example from `extended_benchmark.py` (line 2-5):**
```python
"""
Extended LLM Benchmark Script
Runs benchmarks on each model separately with proper model offloading,
progress tracking, and detailed result collection.
"""
```

## Function Design

**Size:**
- Small to medium functions (15-40 lines typical)
- Larger functions (50-80 lines) used for orchestration: `run_benchmark()` in `extended_benchmark.py`, `main()` functions
- No excessively large functions identified

**Parameters:**
- Functions accept specific parameters rather than dictionaries
- Optional parameters use default values: `skip_models: List[str] = None`
- Type hints present throughout: `model_name: str`, `prompt: str`, `verbose: bool`
- Maximum 3-5 parameters per function typical

**Return Values:**
- Functions return typed values with clear semantics
- Boolean returns for success/failure: `check_python()`, `check_ollama()`
- Data objects returned: `OllamaResponse`, `BenchmarkResult`
- `None` returned on fatal errors
- Multiple returns on success (e.g., tuple of (success, has_venv))

**Example from `run.py` (lines 125-163):**
```python
def setup_venv():
    """Setup virtual environment and install dependencies"""
    venv_dir = Path('.venv')

    # Check if venv exists and is valid
    python_in_venv = venv_dir / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')
    if venv_dir.exists() and python_in_venv.exists():
        print_info("Virtual environment already exists")
        return True, True  # (success, has_venv)
```

## Module Design

**Exports:**
- No explicit `__all__` definitions
- Functions and classes implicitly public (no leading underscore convention)
- Signal handlers prefixed with underscore: `_signal` to avoid collision (line 107)

**Barrel Files:**
- Not used in this codebase
- Single-file scripts without module organization

## Data Structures

**Pydantic Models:**
- Used for API response validation and type safety
- `Message` model: `role: str`, `content: str` (lines 14-16 in `benchmark.py`)
- `OllamaResponse` model: Comprehensive timing and token data (lines 19-40 in `benchmark.py`)
- `BenchmarkResult` model: Single benchmark run data (lines 171-186 in `extended_benchmark.py`)
- `ModelBenchmarkSummary` model: Aggregated results (lines 188-200 in `extended_benchmark.py`)

**Field Validators:**
- Used for data transformation and validation: `@field_validator("prompt_eval_count")`
- Handles edge cases like prompt caching (value = -1 → converted to 0)

---

*Convention analysis: 2026-03-12*
