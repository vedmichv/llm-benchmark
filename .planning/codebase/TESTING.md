# Testing Patterns

**Analysis Date:** 2026-03-12

## Test Framework

**Runner:**
- Manual testing via CLI execution
- No test framework configured (pytest, unittest, etc.)
- Single test file: `test_ollama.py` runs standalone

**Assertion Library:**
- Not applicable - manual return code checks instead

**Run Commands:**
```bash
python test_ollama.py              # Run Ollama CLI verification tests
python benchmark.py                # Run main benchmark
python run.py                       # Run with setup/dependency check
```

## Test File Organization

**Location:**
- Co-located with source files at project root
- Test file: `/Users/viktor/Documents/GitHub/vedmich/llm-benchmark/test_ollama.py`

**Naming:**
- `test_` prefix for test files: `test_ollama.py`
- Function names describe what is tested: `test_ollama_cli()`

**Structure:**
```
llm-benchmark/
├── benchmark.py              # Main benchmark script
├── extended_benchmark.py     # Extended benchmark with advanced features
├── run.py                    # Cross-platform launcher/setup
├── test_ollama.py            # Manual integration test
└── compare_results.py        # Results comparison utility
```

## Test Structure

**Suite Organization:**
```python
def test_ollama_cli():
    """Test Ollama using CLI"""
    print("=" * 60)
    print("Testing Ollama via CLI")
    print("=" * 60)
    print()

    # Test 1: Check ollama version
    print("Test 1: Checking Ollama version...")
    try:
        result = subprocess.run(['ollama', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print(f"✗ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
```

**Patterns:**
- Linear test execution: sequential checks with early exit on failure
- Each test grouped by comment: `# Test 1:`, `# Test 2:`, `# Test 3:`
- Each check follows same pattern: try-except with specific success/failure output
- Manual print statements for test output (no framework formatting)
- Success indicated by `✓` symbol, failure by `✗` symbol
- Return `True` on complete success, `False` on any failure
- Final sys.exit based on return value

## Mocking

**Framework:** None

**Patterns:**
- No mocking library used
- External subprocess calls not mocked: `ollama list`, `ollama run`
- Direct system integration testing only

**What to Mock:**
- Not applicable - testing philosophy prefers real system verification

**What NOT to Mock:**
- Ollama CLI interactions: Direct execution via `subprocess.run()`
- Filesystem operations: Direct Path operations with actual file access

## Fixtures and Factories

**Test Data:**
- No test fixtures or factories defined
- Prompt test sets defined as constants in `extended_benchmark.py` (lines 26-56):

```python
PROMPT_SETS = {
    "small": [
        # Quick test set (3 prompts)
        "Write a Python function to calculate the factorial of a number",
        "Explain the difference between HTTP and HTTPS",
        "Write a binary search algorithm in Python"
    ],
    "medium": [
        # Standard test set (5 prompts) - original defaults
        ...
    ],
    "large": [
        # Comprehensive test set (11 prompts)
        ...
    ]
}
```

**Location:**
- Prompt sets defined at module level in `extended_benchmark.py`
- No separate fixtures directory

## Coverage

**Requirements:** None enforced

**View Coverage:**
- No coverage tools configured
- Manual verification through CLI output examination

## Test Types

**Unit Tests:**
- Not formally implemented
- Manual validation through `test_ollama.py` covers system-level functionality
- Pydantic model validation handled by framework automatically

**Integration Tests:**
- `test_ollama.py` provides CLI integration testing:
  - Test 1: `ollama --version` verification
  - Test 2: `ollama list` model listing
  - Test 3: `ollama run qwen3-coder:30b 'Count from 1 to 5'` model execution

**E2E Tests:**
- Benchmark execution itself is E2E test: `benchmark.py` → `run_benchmark()` → Ollama API
- Manual verification via CLI execution

## Common Patterns

**Async Testing:**
- Not applicable (synchronous subprocess calls)
- Threading used in `extended_benchmark.py` for timeout implementation (lines 124-143):

```python
def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """Run a function with a timeout. Works on all platforms (no SIGALRM)."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise BenchmarkTimeoutError(f"Operation timed out after {timeout_sec}s")
    if error[0]:
        raise error[0]
    return result[0]
```

**Error Testing:**
Pattern observed in `test_ollama.py` (lines 18-30):
```python
try:
    result = subprocess.run(['ollama', '--version'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"✓ {result.stdout.strip()}")
    else:
        print(f"✗ Failed: {result.stderr}")
        return False
except Exception as e:
    print(f"✗ Error: {e}")
    return False
```

## Manual Testing Checklist

**Validation Steps (from `test_ollama.py`):**

1. **Version Check:**
   - Verify Ollama CLI is installed and working
   - Check return code from `ollama --version`
   - Print version string on success

2. **Model Listing:**
   - Query available models via `ollama list`
   - Count and display model names
   - Verify output format

3. **Model Execution:**
   - Run actual inference with `ollama run <model> <prompt>`
   - Verify return code = 0
   - Clean ANSI codes from output
   - Display last 200 characters of response

**Success Criteria:**
- All 3 tests pass with return code 0
- No exceptions raised
- Output printed with success indicators (`✓`)

## Testing Philosophy

**Direct System Integration:**
- Tests verify real Ollama installation and execution
- No mocking or simulation of Ollama behavior
- Relies on working Ollama server at test time

**Manual Verification:**
- Human inspection of test output
- Symbol-based pass/fail indicators
- Explicit error messages for debugging

**Limitations:**
- No continuous integration setup detected
- No automated test runner
- Manual invocation required
- No test parallelization

---

*Testing analysis: 2026-03-12*
