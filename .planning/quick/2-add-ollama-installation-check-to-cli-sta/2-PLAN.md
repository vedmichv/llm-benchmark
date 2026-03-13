---
phase: quick
plan: 2
type: execute
wave: 1
depends_on: []
files_modified:
  - llm_benchmark/preflight.py
  - tests/test_preflight.py
autonomous: true
requirements: [quick-2]
must_haves:
  truths:
    - "When ollama binary is not found, user sees platform-specific install command"
    - "User can accept auto-install and ollama gets installed via subprocess"
    - "User can decline install and program exits with clear message"
    - "When ollama binary exists, check is transparent (no prompt)"
  artifacts:
    - path: "llm_benchmark/preflight.py"
      provides: "check_ollama_installed() function with interactive install flow"
      contains: "shutil.which"
    - path: "tests/test_preflight.py"
      provides: "Tests for installation check scenarios"
      contains: "TestOllamaInstallation"
  key_links:
    - from: "llm_benchmark/preflight.py"
      to: "run_preflight_checks"
      via: "check_ollama_installed called before check_ollama_connectivity"
      pattern: "check_ollama_installed.*check_ollama_connectivity"
---

<objective>
Add an Ollama installation check to CLI startup that runs before existing preflight checks.

Purpose: Students who don't have Ollama installed get a helpful prompt offering to install it automatically, rather than a cryptic connection error.
Output: Updated preflight.py with `check_ollama_installed()` and corresponding tests.
</objective>

<execution_context>
@/Users/viktor/.claude/get-shit-done/workflows/execute-plan.md
@/Users/viktor/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@llm_benchmark/preflight.py
@llm_benchmark/system.py
@llm_benchmark/cli.py
@tests/test_preflight.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add check_ollama_installed() to preflight.py with tests</name>
  <files>llm_benchmark/preflight.py, tests/test_preflight.py</files>
  <behavior>
    - Test: When shutil.which("ollama") returns a path, check_ollama_installed() returns True without any output
    - Test: When shutil.which("ollama") returns None and user inputs "n", function returns False and console shows platform-specific install command (darwin: "curl -fsSL https://ollama.com/install.sh | sh", windows: "irm https://ollama.com/install.ps1 | iex", linux: "curl -fsSL https://ollama.com/install.sh | sh")
    - Test: When shutil.which("ollama") returns None, user inputs "y", and subprocess succeeds, and shutil.which then returns a path, function returns True
    - Test: When shutil.which("ollama") returns None, user inputs "y", but subprocess fails, function returns False with error message
    - Test: run_preflight_checks calls check_ollama_installed before check_ollama_connectivity, and exits if it returns False
  </behavior>
  <action>
    1. Add `check_ollama_installed() -> bool` to preflight.py:
       - Use `shutil.which("ollama")` to check if binary exists. If found, return True immediately.
       - If not found, detect OS via `platform.system()`.
       - Print "[yellow]Ollama is not installed.[/yellow]" then show the install command:
         - Darwin/Linux: `curl -fsSL https://ollama.com/install.sh | sh`
         - Windows: `irm https://ollama.com/install.ps1 | iex`
       - Prompt with `input("Install Ollama now? (y/N) ")`. Use try/except for EOFError/KeyboardInterrupt (treat as decline).
       - If user enters "y" or "Y": run the install command via `subprocess.run(cmd, shell=True)`. After install, verify with `shutil.which("ollama")` again. If found, print "[green]Ollama installed successfully![/green]" and return True. If not found, print "[red]Installation may have failed...[/red]" and return False.
       - If user declines: print "[dim]Install Ollama manually: https://ollama.com/download[/dim]" and return False.

    2. Update `run_preflight_checks()` to call `check_ollama_installed()` as step 0, before connectivity check:
       ```python
       # 0. Installation check (blocking)
       if not check_ollama_installed():
           sys.exit(1)
       ```

    3. Add `TestOllamaInstallation` class to test_preflight.py with the tests described in behavior. Mock `shutil.which`, `platform.system`, `subprocess.run`, and `builtins.input`. Use `patch` on the preflight module's imports.

    4. Update the existing `TestRunPreflightChecks` tests to also mock `check_ollama_installed` returning True so they don't break.
  </action>
  <verify>
    <automated>cd /Users/viktor/Documents/GitHub/vedmich/llm-benchmark && uv run pytest tests/test_preflight.py -x -v</automated>
  </verify>
  <done>
    - check_ollama_installed() exists and is called first in run_preflight_checks
    - All 5 new test cases pass (binary found, user declines, user accepts success, user accepts failure, preflight integration)
    - All existing preflight tests still pass
    - No ruff/lint errors
  </done>
</task>

</tasks>

<verification>
uv run pytest tests/test_preflight.py -x -v
uv run ruff check llm_benchmark/preflight.py tests/test_preflight.py
</verification>

<success_criteria>
- `shutil.which("ollama")` check runs before any Ollama API calls
- Platform-specific install commands shown for darwin/linux/windows
- Interactive y/N prompt with safe default (N)
- Subprocess install attempt on user approval
- Post-install verification via shutil.which
- All tests pass, no regressions
</success_criteria>

<output>
After completion, create `.planning/quick/2-add-ollama-installation-check-to-cli-sta/2-SUMMARY.md`
</output>
</task>
